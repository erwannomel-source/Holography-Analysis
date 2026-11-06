"""Notebook state, preview controls and analysis worker lifecycle."""
from __future__ import annotations

import html
import json
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event, Thread
from collections import OrderedDict
from .config import AnalysisConfig, parse_numbers
from .preparation import normalize_particle_choice
from .analysis import analyse_images
from .profiles import resolve_path, read_profile, validate_profile
from .outputs import input_manifest
from .ui_controls import NotebookControlsMixin
from .interaction import DetectionInteractionMixin
from .preview import NotebookPreviewMixin
from .notebook_images import append_png, clear_images
from .figure_browser import FigureBrowserMixin
from .particle_crop_controls import ParticleCropControlsMixin
from .preparation_cache import snapshot_preparation


def create_notebook_interface(base_dir=None,
                              defaults=None) -> 'NotebookInterface':
    """Create the Jupyter interface for display(interface.widget)."""
    return NotebookInterface(base_dir=base_dir, defaults=defaults)


class NotebookInterface(NotebookControlsMixin, DetectionInteractionMixin,
                        NotebookPreviewMixin, FigureBrowserMixin,
                        ParticleCropControlsMixin):
    """Notebook controls and output; analysis is implemented in the functions above."""

    def __init__(self, base_dir=None, defaults=None):
        import ipywidgets as W
        from IPython.display import display
        self.W = W
        self.display = display
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.paths = []
        self.choices = {}
        self.center_overrides = {}
        self.clicked_particles = {}
        self.crop_overrides = {}
        self._loaded_key = None
        self._detection_key = None
        self._bg_cache = None
        self._preview = None
        # Bound memory: retain at most three images and 32 prepared crops.
        self._detection_cache = OrderedDict()
        self._crop_cache = OrderedDict()
        self._preview_fingerprint = None
        self._seen_fingerprints = {}
        self._background_fingerprint = None
        self._pending_guess = False
        self._pending_crop = False
        self._syncing_particle_crop = False
        self._manual_baseline = None
        self._canvas_enabled = None
        self._canvas = None
        self._changing = False
        self._busy = False
        self._analysis_running = False
        self._settings_error = ''
        self._crop_summaries = OrderedDict()
        self._figure_groups = OrderedDict()
        self._figure_labels = {}
        self._figure_store = None
        self._figure_serial = 0
        self._syncing_figure_choice = False
        self._analysis_paths = ()
        self._cancel_event = Event()
        self._worker = None
        # The kernel event loop owns widget updates. Fitting runs in a worker.
        from IPython import get_ipython
        kernel = getattr(get_ipython(), 'kernel', None)
        self._io_loop = getattr(kernel, 'io_loop', None)
        self.last_result = None
        cfg = AnalysisConfig.from_dict(defaults if defaults is not None else {
            'click_to_detect': True,
            'detect_search_circle_only': True
        })
        self._analysis_figure_limit = cfg.max_displayed_figures
        self._build_widgets(cfg)
        settings = self._build_settings(cfg)
        self._build_layout(settings)
        self._connect_callbacks()
        self._validate_settings()
        self._availability()

    def close(self):
        """Close this interface before rerunning its notebook startup cell."""
        from .notebook import close_interface
        close_interface(self)

    def _connect_callbacks(self):
        """Bind user actions after all widgets have been created."""
        self.load_button.on_click(
            lambda _: self._perform(self._load_dataset, self.preview_output))
        self.preview_button.on_click(
            lambda _: self._perform(self._preview_image, self.preview_output))
        self.initial_button.on_click(lambda _: self._perform(
            self._show_edited_initial, self.preview_output))
        self.current_button.on_click(lambda _: self._start_analysis(False))
        self.series_button.on_click(lambda _: self._start_analysis(True))
        self.select_all_button.on_click(lambda _: self._perform(
            self._select_all_particles, self.message_output))
        self.apply_guess_button.on_click(lambda _: self._perform(
            self._apply_manual_center, self.preview_output))
        self.reset_guess_button.on_click(lambda _: self._perform(
            self._reset_manual_center, self.preview_output))
        self.discard_guess_button.on_click(
            lambda _: self._discard_manual_edits())
        for button in self._stop_buttons:
            button.on_click(self._stop_analysis)
        self.save_profile_button.on_click(
            lambda _: self._perform(self._save_profile, self.message_output))
        self.load_profile_button.on_click(
            lambda _: self._perform(self._load_profile, self.preview_output))
        self.image_choice.observe(self._image_changed, names='value')
        self.previous_image_button.on_click(lambda _: self._navigate_image(-1))
        self.next_image_button.on_click(lambda _: self._navigate_image(1))
        self.particle_choice.observe(self._particle_changed, names='value')
        self.auto_selection.observe(self._automatic_selection_changed,
                                    names='value')
        self.guess_particle.observe(self._guess_particle_changed,
                                    names='value')
        self.crop_particle.observe(self._crop_particle_changed, names='value')
        self.figure_particle.observe(self._figure_particle_changed, names='value')
        for control in (self.particle_crop_mode, self.particle_crop_height,
                        self.particle_crop_width):
            control.observe(self._particle_crop_edited, names='value')
        self.apply_particle_crop.on_click(lambda _: self._perform(
            self._apply_particle_crop, self.preview_output))
        self.discard_particle_crop.on_click(lambda _: self._discard_particle_crop())
        self.x_guess_px.observe(self._manual_guess_changed, names='value')
        self.y_guess_px.observe(self._manual_guess_changed, names='value')
        self.detect_point_button.on_click(self._detect_coordinate_particle)
        self.map_image_only.observe(lambda _: self._update_map_display(),
                                    names='value')
        self.deselect_mode.observe(self._deselect_mode_changed, names='value')
        for control in self.controls.values():
            control.observe(self._settings_changed, names='value')

    def _settings_changed(self, _):
        """Validate edited values immediately without loading images or fitting."""
        if self._changing:
            return
        if not self.controls[
                'click_to_detect'].value and self.deselect_mode.value:
            self.deselect_mode.value = False
        self._validate_settings()
        if not self._pending_crop:
            self._load_particle_crop_fields()
        self._availability()

    def _validate_settings(self):
        """Show configuration errors separately from analysis progress messages."""
        try:
            cfg = self.get_config()
        except (ValueError, TypeError, OverflowError) as exc:
            self._settings_error = str(exc)
            self.settings_validation.value = (
                '<p style="margin:6px 0;color:#a32020;">'
                f'<b>Check settings:</b> {html.escape(str(exc))}</p>')
            self.acquisition_summary.value = '<p>Check the settings highlighted below.</p>'
        else:
            self._settings_error = ''
            self.settings_validation.value = ''
            self._update_acquisition_summary(cfg)

    def _availability(self):
        loaded = bool(self.paths) and not self._busy
        pending = self._pending_guess or self._pending_crop
        for control in [
                self.image_choice, self.preview_button,
                self.current_button, self.series_button
        ]:
            control.disabled = not loaded
        self.particle_choice.disabled = not loaded or self._preview is None
        self.auto_selection.disabled = not loaded
        empty_selection = self._preview is not None and not self.particle_choice.value
        self.current_button.disabled = not loaded or empty_selection
        self.select_all_button.disabled = (not loaded or self._preview is None
                                           or not self._preview[-1])
        center_unavailable = (not loaded or self._preview is None
                              or self.guess_particle.value is None or self._pending_crop)
        for control in [
                self.guess_particle, self.x_guess_px, self.y_guess_px,
                self.apply_guess_button, self.reset_guess_button
        ]:
            control.disabled = center_unavailable
        self.initial_button.disabled = center_unavailable or pending
        self.crop_particle.disabled = (not loaded or self._preview is None
                                       or not self.crop_particle.options
                                       or pending
                                       or bool(self._settings_error))
        self.figure_particle.disabled = self._busy or not self.figure_particle.options
        self.discard_guess_button.disabled = not loaded or not self._pending_guess
        self.map_image_only.disabled = not loaded or self._canvas_enabled is False
        self._update_map_display()
        if pending:
            for control in [
                    self.current_button, self.series_button,
                    self.save_profile_button, self.load_profile_button,
                    self.load_button, self.image_choice, self.guess_particle,
                    self.particle_choice, self.auto_selection,
                    self.select_all_button, self.preview_button
            ]:
                control.disabled = True
        elif not self._busy:
            self.save_profile_button.disabled = False
            self.load_profile_button.disabled = False
            self.load_button.disabled = False
        if self._settings_error:
            for control in [
                    self.load_button, self.preview_button,
                    self.initial_button,
                    self.current_button, self.series_button,
                    self.apply_guess_button, self.reset_guess_button,
                    self.select_all_button, self.save_profile_button
            ]:
                control.disabled = True
        click_mode = self.controls['click_to_detect'].value
        self.deselect_mode.disabled = (not loaded or self._preview is None
                                       or not click_mode or pending
                                       or bool(self._settings_error))
        self.auto_selection.disabled = self.auto_selection.disabled or click_mode
        index = self.image_choice.value
        navigation_locked = (self.image_choice.disabled or self._analysis_running
                             or index is None)
        self.previous_image_button.disabled = navigation_locked or index <= 0
        self.next_image_button.disabled = (
            navigation_locked or index >= len(self.paths) - 1)
        for button in self._stop_buttons:
            button.disabled = not self._analysis_running or self._cancel_event.is_set(
            )
        self._sync_simple_controls()
        self._particle_crop_availability()

    def _set_busy(self, busy):
        self._busy = busy
        if busy:
            self._disabled = []

            def visit(widget):
                if widget in self._stop_buttons:
                    return
                if 'disabled' in widget.traits():
                    self._disabled.append((widget, widget.disabled))
                    widget.disabled = True
                for child in getattr(widget, 'children', ()):
                    visit(child)

            visit(self.widget)
        else:
            for widget, disabled in getattr(self, '_disabled', []):
                widget.disabled = disabled
        self._availability()
        self._sync_fixed_crop_controls()

    def _perform(self, action, output):
        if self._busy:
            return
        self._set_busy(True)
        if output is self.preview_output:
            clear_images(self.preview_images)
        # Replace widget contents directly: clear_output messages can leave
        # previous figures visible when the frontend handles widget callbacks.
        output.outputs = ()
        try:
            # Catch inside Output: in a live IPython kernel its context manager
            # otherwise consumes exceptions before our status handler sees them.
            with output:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore',
                            category=FutureWarning,
                            module=r'xarray\.core\.missing',
                        )
                        action()
                except KeyboardInterrupt:
                    self.status.value = 'Operation stopped.'
                except Exception as exc:
                    self.status.value = f'<span style="color:#a32020">{html.escape(str(exc))}</span>'
        finally:
            self._set_busy(False)

    def get_config(self) -> AnalysisConfig:
        values = {
            name: control.value
            for name, control in self.controls.items()
        }
        # Automatic notebook detection always uses the search circle.
        values['detect_search_circle_only'] = True
        # The result browser displays particle figures only, not detection maps.
        values['show_detection_plots'] = False
        for name in [
                'illum_polarization', 'n_bounds', 'r_bounds', 'r_guesses',
                'z_values', 'z_bounds'
        ]:
            values[name] = parse_numbers(values[name])
        cfg = AnalysisConfig.from_dict(values)
        if (self._preview is not None
                and self._preview[0] == self.image_choice.value):
            cfg = cfg.centered_for_image(self._preview[2].shape)
        return cfg

    def _dataset_key(self):
        return (str(resolve_path(self.image_folder.path.value, self.base_dir)),
                str(resolve_path(self.background.path.value,
                                 self.base_dir)), self.pattern.value.strip())

    def _check_dataset(self):
        if not self.paths:
            raise ValueError('Load images before continuing.')
        if self._loaded_key != self._dataset_key():
            raise ValueError(
                'Click Load images after changing directories, the background or the file pattern.'
            )

    def _load_dataset(self, show_preview=True, image_paths=None):
        self._check_pending_guess()
        cfg = self.get_config()
        folder, bg_name, pattern = self._dataset_key()
        folder = Path(folder)
        bg_path = Path(bg_name)
        if not folder.is_dir():
            raise ValueError('Choose an accessible image directory.')
        if not bg_path.is_file():
            raise ValueError('Choose an accessible background image.')
        if not pattern:
            raise ValueError('Enter a file pattern, for example *.tif.')
        if image_paths is None:
            paths = sorted(p.resolve() for p in folder.glob(pattern)
                           if p.is_file() and p.resolve() != bg_path)
        else:
            # Analysis settings retain the exact input list and its original order.
            paths = [Path(p).resolve() for p in image_paths]
            if any(not p.is_file() or p == bg_path for p in paths):
                raise ValueError(
                    'An image saved in the analysis settings is missing or is the background.'
                )
        if not paths:
            raise ValueError(f'No images matching {pattern} in {folder}.')
        old_name = self.paths[
            self.image_choice.
            value].name if self.paths and self.image_choice.value is not None else None
        if self._loaded_key != (str(folder), str(bg_path), pattern):
            self.choices.clear()
            self.center_overrides.clear()
            self.clicked_particles.clear()
            self.crop_overrides.clear()
            self._seen_fingerprints.clear()
            self._background_fingerprint = None
            self._detection_cache.clear()
            self._crop_cache.clear()
        self.paths = paths
        self._loaded_key = (str(folder), str(bg_path), pattern)
        self._preview = None
        self._preview_fingerprint = None
        self._sync_crop_particle()
        self._crop_summaries.clear()
        self.crop_summary.value = ''
        self._changing = True
        try:
            self.image_choice.options = [(f'{i} — {path.name}', i)
                                         for i, path in enumerate(paths)]
            self.image_choice.value = next(
                (i for i, p in enumerate(paths) if p.name == old_name), 0)
        finally:
            self._changing = False
        self.status.value = f'{len(paths)} hologram(s) loaded.'
        if show_preview:
            self._preview_image()

    def _image_changed(self, change):
        if self._changing or self._busy or change['new'] is None:
            return
        self._perform(self._preview_image, self.preview_output)

    def _navigate_image(self, step):
        """Use the normal image-change workflow without losing unapplied edits."""
        if (self._changing or self._busy or self._analysis_running
                or self._pending_guess or self._pending_crop
                or self.image_choice.disabled or self.image_choice.value is None):
            return
        index = self.image_choice.value + step
        if 0 <= index < len(self.paths):
            self.image_choice.value = index

    def _select_all_particles(self):
        """Select every candidate and preview one crop without starting a fit."""
        self._check_pending_guess()
        self._require_current_preview()
        index = self.image_choice.value
        if (self._preview is None or self._preview[0] != index
                or self._detection_key != self.get_config().detection_key()):
            raise ValueError(
                'Click Refresh / check particles before selecting all particles.')
        candidates = self._preview[-1]
        if not candidates:
            raise ValueError('No particles detected in this image.')
        path = self.paths[index]
        self._changing = True
        try:
            self.auto_selection.value = False
            numbers = tuple(range(1, len(candidates) + 1))
            self.particle_choice.value = numbers
            self.choices[path.name] = list(numbers)
        finally:
            self._changing = False
        self.batch_mode.value = 'all'
        self._synchronize_editor()
        if self._preview[2] is not None:
            self._render_detection()
            self._show_selected_crop()
        self.status.value = (
            f'All {len(candidates)} detected particles selected in {html.escape(path.name)}. '
            'Click Analyse this image to analyse them all.')

    def _particle_changed(self, change):
        if self._changing or self._busy or not self.paths:
            return
        path = self.paths[self.image_choice.value]
        self.choices[path.name] = list(change['new'])
        self._changing = True
        try:
            self.auto_selection.value = False
        finally:
            self._changing = False
        self.batch_mode.value = 'saved'
        self._synchronize_editor()
        if self._preview is not None and self._preview[2] is not None:
            self._render_detection()
            if self.crop_particle.value is not None:
                self._perform(self._show_selected_crop, self.preview_output)
        self._selection_status()

    def _synchronize_editor(self):
        """Keep the manual-center editor consistent for every selection path."""
        numbers = tuple(self.particle_choice.value)
        previous_editor = self.guess_particle.value
        self._changing = True
        try:
            self.guess_particle.options = [(f'Candidate {number}', number)
                                           for number in numbers]
            self.guess_particle.value = (previous_editor
                                         if previous_editor in numbers else
                                         (numbers[0] if numbers else None))
        finally:
            self._changing = False
        self._load_manual_center_fields()
        self._sync_crop_particle()

    def _automatic_selection_changed(self, change):
        if self._changing or self._busy or not self.paths:
            return
        path = self.paths[self.image_choice.value]
        if change['new']:
            self.choices[path.name] = None
            self.batch_mode.value = 'auto'
            self._perform(self._preview_image, self.preview_output)
        else:
            self.choices[path.name] = list(self.particle_choice.value)
            self.batch_mode.value = 'saved'
            self._selection_status()

    def _selection_status(self):
        """Describe manual changes without reloading images or simulating holograms."""
        numbers = self.particle_choice.value
        if numbers:
            self.status.value = (
                f'{len(numbers)} particle(s) selected: {", ".join(map(str, numbers))}. '
                'Click Refresh / check particles to preview, or Analyse this image to fit this selection.'
            )
        else:
            self.status.value = 'No particles selected. Select at least one particle to analyse this image.'
        self._availability()

    def _on_progress(self, value, maximum, message):
        self.image_progress.max = maximum
        self.image_progress.value = value
        if not self._cancel_event.is_set():
            self.status.value = html.escape(message)
        if 'completed' not in message:
            self.fit_progress.value = 0

    def _on_trial(self, value, maximum, trial):
        """Update the progress bar without duplicating the trial table in the log."""
        self.fit_progress.description = (
            'Reused fit' if trial.get('phase') == 'cache' else
            'Profile starts' if trial.get('phase') == 'learned_search' else
            'Explore starts' if trial.get('search_step') == 'short_fits' else
            'Initial search' if trial.get('status') == 'searching' else 'Local fits')
        self.fit_progress.max = maximum
        self.fit_progress.value = value

    def _post_ui(self, callback, *args):
        """Send worker output to Jupyter's event loop, in the order it was emitted."""
        if self._io_loop is None:
            # Also support scripts and tests without a running Jupyter kernel.
            callback(*args)
        else:
            self._io_loop.add_callback(callback, *args)

    def _log_analysis(self, message):
        self._post_ui(self._append_analysis_log, str(message))

    def _append_analysis_log(self, message):
        """Route fitted values to their field without duplicating them in the log."""
        if message.startswith('Fitted parameters —'):
            separator = '\n\n' if self.fit_parameters_copy.value else ''
            self.fit_parameters_copy.value += separator + message
        else:
            self.fit_output.append_stdout(message + '\n')

    def _show_preview_figure(self, png_data):
        """Show previews independently of stdout routing in Notebook and Lab."""
        append_png(self.preview_images, png_data)

    def _show_analysis_figure(self, png_data, particle=None):
        """Queue image retention with the display limit captured for this run."""
        self._post_ui(self._append_analysis_figure, png_data,
                      self._analysis_figure_limit, particle)

    def _show_particle_figure(self, png_data, image_index, particle_number):
        self._show_analysis_figure(png_data, (int(image_index), int(particle_number)))

    def _start_analysis(self, series: bool) -> None:
        """Validate and snapshot controls, then return so Stop clicks can arrive."""
        if self._busy:
            return
        try:
            self._check_pending_guess()
            self._check_dataset()
            cfg = self.get_config()
            if self._detection_key != cfg.detection_key():
                raise ValueError(
                    'Click Refresh / check particles after changing calibration or detection settings.'
                )
            if not series and self._preview is not None and not self.particle_choice.value:
                raise ValueError(
                    'Select at least one particle before analysing this image.'
                )
            if self._preview is not None:
                self._require_current_preview()
            if series:
                self._check_saved_image_fingerprints()
                if (self.crop_overrides and self.batch_mode.value != 'saved'
                        and (cfg.click_to_detect or self.batch_mode.value == 'all')):
                    raise ValueError(
                        'Choose Saved choices as the series mode to keep individual crops: '
                        'a new full-image detection uses different candidate numbers.')
            parameters = {
                'image_paths':
                tuple(self.paths),
                'bg_path':
                resolve_path(self.background.path.value, self.base_dir),
                'cfg':
                cfg,
                'output_root':
                resolve_path(self.output_folder.path.value, self.base_dir),
                'run_name':
                self.analysis_folder_name.value,
                'image_indices':
                None if series else [self.image_choice.value],
                'mode':
                self.batch_mode.value if series else 'saved',
                'choices': {
                    i: normalize_particle_choice(self.choices[path.name])
                    for i, path in enumerate(self.paths)
                    if path.name in self.choices
                },
                'clicked_particles': {
                    i: [dict(c) for c in self.clicked_particles[path.name]]
                    for i, path in enumerate(self.paths)
                    if path.name in self.clicked_particles
                },
                'crop_overrides': {
                    i: {number: dict(settings) for number, settings in
                        self.crop_overrides[path.name].items()}
                    for i, path in enumerate(self.paths)
                    if path.name in self.crop_overrides
                },
                'center_overrides': {
                    i: self.center_overrides[path.name]
                    for i, path in enumerate(self.paths)
                    if path.name in self.center_overrides
                },
            }
            parameters['preparation_cache'] = snapshot_preparation(
                self._detection_cache, self._crop_cache,
                parameters['image_paths'], parameters['bg_path'], cfg,
                parameters['mode'], parameters['image_indices'])
        except Exception as exc:
            self.status.value = f'<span style="color:#a32020">{html.escape(str(exc))}</span>'
            self.fit_output.append_stdout(f'Error: {exc}\n')
            return

        self._cancel_event = Event()
        self._analysis_figure_limit = cfg.max_displayed_figures
        self._analysis_paths = parameters['image_paths']
        self._analysis_running = True
        self._set_busy(True)
        self.fit_output.outputs = ()
        self.fit_parameters_copy.value = ''
        self._reset_figure_browser()
        self.result_links.value = ''
        self.result_table.value = ''
        self.trial_table.value = ''
        self.last_result = None
        self.image_progress.value = 0
        self.fit_progress.value = 0
        self.status.value = 'Analysis running. Click Stop to cancel.'
        self._worker = Thread(
            target=self._run_analysis,
            args=(parameters, ),
            name='hologram-analysis',
            daemon=True,
        )
        try:
            self._worker.start()
        except Exception as exc:
            self._finish_analysis(None, exc)

    def _stop_analysis(self, _):
        """Signal cancellation; leave controls locked until saving is complete."""
        if not self._analysis_running or self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self.status.value = (
            'Stopping… waiting for the current calculation to return and saving completed results.'
        )
        self._availability()

    def _run_analysis(self, parameters):
        """Worker entry point: compute only; queue all widget and output updates."""
        result = None
        error = None
        try:
            result = analyse_images(
                **parameters,
                cancel_event=self._cancel_event,
                log=self._log_analysis,
                log_trial_details=False,
                on_figure=self._show_analysis_figure,
                on_particle_figure=self._show_particle_figure,
                on_progress=lambda *args: self._post_ui(
                    self._on_progress, *args),
                on_trial=lambda *args: self._post_ui(self._on_trial, *args),
            )
        except (KeyboardInterrupt, Exception) as exc:
            error = exc
        finally:
            self._post_ui(self._finish_analysis, result, error)

    def _finish_analysis(self, result, error=None):
        """Show saved results and always restore controls, including after errors."""
        try:
            if result is not None:
                self.last_result = result
                self._index_saved_figures(result)
                self._render_results(result)
                state = 'Analysis stopped' if result.interrupted else 'Analysis completed'
                self.status.value = (
                    f'{state}: {len(result.summary_rows)} selected fit(s), '
                    f'{len(result.error_rows)} error(s).')
            elif isinstance(error, KeyboardInterrupt):
                self.status.value = 'Analysis stopped.'
            else:
                raise error or RuntimeError('Analysis returned no result.')
        except Exception as exc:
            self.status.value = f'<span style="color:#a32020">{html.escape(str(exc))}</span>'
            self.fit_output.append_stdout(f'Error: {exc}\n')
        finally:
            self._analysis_running = False
            self._set_busy(False)

    @staticmethod
    def _table(rows, columns, limit=100):
        """Render compact, content-sized tables independently of notebook CSS."""
        if not rows:
            return '<p>No results in this table.</p>'

        def cell(value, key):
            if value is None:
                return ''
            if key == 'solution_ambiguous':
                return 'Yes' if value in (True, 'True', 'true') else 'No'
            if key == 'fit_acceptable':
                return 'Yes' if value in (True, 'True', 'true') else 'No'
            if key == 'relative_improvement' and isinstance(value, (int, float)):
                return html.escape(f'{100 * value:.1f}%')
            if isinstance(value, float):
                return html.escape(f'{value:.6g}')
            return html.escape(str(value))

        cell_style = (
            'padding:6px 10px;white-space:nowrap;vertical-align:middle;'
            'font-size:13px;line-height:1.4;font-variant-numeric:tabular-nums;'
            'border-bottom:1px solid rgba(127,127,127,0.2);')

        def alignment(key):
            return 'center' if key == 'selected' else 'right'

        head = ''.join(
            f'<th style="{cell_style}text-align:{alignment(key)};'
            f'background:rgba(127,127,127,0.14);">{html.escape(label)}</th>'
            for key, label in columns)
        body = ''.join(
            f'<tr style="background:{"rgba(127,127,127,0.06)" if index % 2 else "transparent"}">'
            + ''.join(f'<td style="{cell_style}text-align:{alignment(key)};">'
                      f'{cell(row.get(key), key)}</td>'
                      for key, _ in columns) + '</tr>'
            for index, row in enumerate(rows[:limit]))
        caption = f'{len(rows)} row(s).'
        if len(rows) > limit:
            caption += f' Showing the first {limit}; the CSV contains all rows.'
        return (
            '<div style="display:block;min-width:0;max-width:100%;">'
            f'<p style="margin:4px 0 8px;font-size:13px;line-height:1.4;">{caption}</p>'
            '<div style="overflow:auto;max-width:100%;max-height:420px;">'
            '<table style="display:table;width:max-content;max-width:none;'
            'table-layout:auto;border-collapse:collapse;margin:0;">'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>'
        )

    def _render_results(self, result):
        import base64
        links = []
        for name, label in [('summary.csv', 'Best fits (CSV)'),
                            ('trials.csv', 'Initial parameters and fit trials (CSV)'),
                            ('settings.json', 'Analysis settings (JSON)')]:
            path = result.output_dir / name
            if not path.is_file():
                continue
            if path.stat().st_size <= 5_000_000:
                mime = 'application/json' if path.suffix == '.json' else 'text/csv'
                data = base64.b64encode(path.read_bytes()).decode('ascii')
                links.append(
                    f'<a download="{name}" href="data:{mime};base64,{data}">{label}</a>'
                )
            else:
                links.append(f'{label} : file saved in the output directory')
        self.result_links.value = (
            f'<p>Results saved in <code>{html.escape(str(result.output_dir))}</code></p>'
            + '<p>' + ' &nbsp; | &nbsp; '.join(links) + '</p>')
        self.result_table.value = self._table(
            result.summary_rows, [('image_index', 'Image'),
                                  ('particle_number', 'Candidate'),
                                  ('nguess', 'N guess'),
                                  ('rguess_um', 'R guess (µm)'),
                                  ('zguess_um', 'Zguess (µm)'),
                                  ('initial_source', 'Initial source'),
                                  ('guess_mode', 'Guess search'),
                                  ('x_um', 'x (µm)'), ('y_um', 'y (µm)'),
                                  ('z_um', 'z (µm)'), ('r_um', 'Radius (µm)'),
                                  ('n', 'Refractive index'), ('chisq', 'Chi²'),
                                  ('relative_improvement', 'Improvement')])
        self.result_table.value += (
            '<p style="font-size:13px;"><b>Guess search</b> describes how '
            'starting values were found: <code>full</code> runs the broad '
            'search; <code>learned</code> reuses successful starts from this '
            'experiment when they suit the current particle; '
            '<code>fallback_full</code> means those learned starts did not '
            'suit it, so the broad search ran. This affects initialization, '
            'not the fitted parameter bounds. A blank value means no '
            'experiment-profile search was used.</p>')
        self.trial_table.value = self._table(result.trial_rows,
                                             [('image_index', 'Image'),
                                              ('particle_number', 'Candidate'),
                                              ('nguess', 'N guess'),
                                              ('rguess_um', 'R guess (µm)'),
                                              ('zguess_um', 'Zguess (µm)'),
                                              ('initial_source', 'Initial source'),
                                              ('initial_score', 'Sampled MSE'),
                                              ('status', 'Status'),
                                              ('z_um', 'fitted z (µm)'),
                                              ('chisq', 'Chi²'),
                                              ('relative_improvement', 'Improvement'),
                                              ('error', 'Error'),
                                              ('selected', 'Selected'),
                                              ('iterations', 'Iterations'),
                                              ('evaluations', 'Evaluations')])

    def _save_profile(self):
        self._check_pending_guess()
        cfg = self.get_config()
        if self.paths:
            self._check_dataset()
            if self._detection_key != cfg.detection_key():
                raise ValueError(
                    'Click Refresh / check particles before saving changed detection settings.'
                )
            if self._preview is not None:
                self._require_current_preview()
            self._check_saved_image_fingerprints()
        path = resolve_path(self.profile_path.value, self.base_dir)
        if path.suffix.lower() != '.json':
            raise ValueError('The profile file must have a .json extension.')
        payload = {
            'version':
            5,
            'config':
            asdict(cfg),
            'image_folder':
            str(resolve_path(self.image_folder.path.value, self.base_dir)),
            'background_path':
            str(resolve_path(self.background.path.value, self.base_dir)),
            'output_folder':
            str(resolve_path(self.output_folder.path.value, self.base_dir)),
            'analysis_folder_name':
            self.analysis_folder_name.value,
            'pattern':
            self.pattern.value,
            'batch_mode':
            self.batch_mode.value,
            'choices':
            self.choices,
            'center_overrides':
            self.center_overrides,
            'clicked_particles':
            self.clicked_particles,
            'crop_overrides':
            self.crop_overrides,
            'current_image':
            self.paths[self.image_choice.value].name if self.paths else None
        }
        if self.paths:
            payload['image_paths'] = [
                str(image.resolve()) for image in self.paths
            ]
            payload['input_manifest'] = input_manifest(
                self.paths, payload['background_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) +
                        '\n',
                        encoding='utf-8')
        self.status.value = f'Profile saved: {html.escape(str(path))}'

    def _load_profile(self):
        self._check_pending_guess()
        path = resolve_path(self.profile_path.value, self.base_dir)
        payload = validate_profile(read_profile(path), self.base_dir)
        cfg = AnalysisConfig.from_dict(payload['config'])
        reset_automatic_choices = not cfg.click_to_detect and not cfg.detect_search_circle_only
        cfg = replace(cfg, detect_search_circle_only=True)
        if payload.get('batch_mode', 'saved') not in ('saved', 'auto', 'all'):
            raise ValueError('Invalid series mode in the profile.')
        choices = payload.get('choices', {})
        if not isinstance(choices, dict):
            raise ValueError('Invalid particle choices in the profile.')
        choices = {
            name: normalize_particle_choice(value)
            for name, value in choices.items()
        }
        center_overrides = payload.get('center_overrides', {})
        if not isinstance(center_overrides, dict):
            raise ValueError('Invalid manual centers in the profile.')
        previous_changing = self._changing
        self._changing = True
        try:
            for name, value in asdict(cfg).items():
                if name == 'detect_search_circle_only':
                    continue
                self.controls[name].value = ', '.join(
                    f'{v:g}' for v in value) if isinstance(value, tuple) else value
        finally:
            self._changing = previous_changing
        self._sync_fixed_crop_controls()
        self._sync_simple_controls()
        self._validate_settings()
        self.image_folder.path.value = payload.get('image_folder', '')
        self.background.path.value = payload.get('background_path', '')
        self.output_folder.path.value = payload.get('output_folder',
                                                    'results_auto_multi_z')
        self.analysis_folder_name.value = payload.get(
            'analysis_folder_name', '')
        self.pattern.value = payload.get('pattern', '*.tif')
        self.batch_mode.value = (
            'saved' if cfg.click_to_detect else
            payload.get('batch_mode', 'auto'))
        self._loaded_key = None
        self._detection_key = cfg.detection_key()
        self.choices.clear()
        self.center_overrides.clear()
        self.clicked_particles.clear()
        self.crop_overrides.clear()
        if 'image_paths' in payload:
            self._load_dataset(show_preview=False,
                               image_paths=payload['image_paths'])
        else:
            self._load_dataset(show_preview=False)
        self.choices.update(choices)
        self.center_overrides.update(center_overrides)
        self.crop_overrides.update(payload.get('crop_overrides', {}))
        if reset_automatic_choices:
            # Restricting the detector changes candidate numbering in old profiles.
            self.choices.clear()
            self.center_overrides.clear()
            self.crop_overrides.clear()
            self.status.value = (
                'Automatic detection now uses the search circle; select particles again.'
            )
        self.clicked_particles.update(payload.get('clicked_particles', {}))
        current = payload.get('current_image')
        self._changing = True
        try:
            self.image_choice.value = next(
                (i for i, p in enumerate(self.paths) if p.name == current), 0)
        finally:
            self._changing = False
        self._preview_image()
        self.status.value = f'Profile loaded: {html.escape(str(path))}'
