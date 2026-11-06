"""Construction and layout of English notebook controls."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import html
from .widgets import PathPicker
from . import __version__
from .simple_controls import SimpleControlsMixin


class NotebookControlsMixin(SimpleControlsMixin):
    """Build widgets without mixing layout with preview and worker state."""

    def _build_widgets(self, cfg):
        """Create file inputs, action buttons and output widgets."""
        W = self.W
        self.image_folder = PathPicker('Image directory', self.base_dir)
        self.background = PathPicker('Background image',
                                     self.base_dir,
                                     kind='file',
                                     suffixes=('.tif', '.tiff', '.png'))
        self.output_folder = PathPicker('Output directory',
                                        self.base_dir,
                                        value='results_auto_multi_z')
        self.analysis_folder_name = W.Text(
            value='',
            description='Analysis folder name',
            placeholder='Empty = automatic dated name',
            tooltip=(
                'Folder created inside Output directory. If the name already '
                'exists, _2, _3, etc. is added automatically.'),
            style={'description_width': '160px'},
            layout=W.Layout(width='95%'))
        self.pattern = W.Text(value='*.tif',
                              description='File pattern',
                              style={'description_width': '160px'})
        self.load_button = W.Button(description='Load images',
                                    button_style='primary',
                                    icon='folder-open')
        self.image_choice = W.Dropdown(options=[],
                                       description='Image',
                                       layout=W.Layout(width='auto', flex='1 1 300px',
                                                       min_width='200px'))
        self.previous_image_button = W.Button(
            description='Previous', icon='chevron-left',
            tooltip='Show the previous image', disabled=True,
            layout=W.Layout(width='110px', margin='0 0 0 6px'))
        self.next_image_button = W.Button(
            description='Next', icon='chevron-right',
            tooltip='Show the next image', disabled=True,
            layout=W.Layout(width='100px', margin='0 0 0 6px'))
        self.image_navigation = W.HBox(
            [self.image_choice, self.previous_image_button, self.next_image_button],
            layout=W.Layout(width='95%', flex_flow='row wrap'))
        self.auto_selection = W.Checkbox(value=not cfg.click_to_detect,
                                         description='Auto in target area')
        self.particle_choice = W.SelectMultiple(
            options=[],
            value=(),
            rows=7,
            description='Particles',
            layout=W.Layout(width='70%'),
        )
        self.select_all_button = W.Button(
            description='Select all particles',
            icon='check-square-o',
            tooltip='Select every detected particle in the displayed image',
            layout=W.Layout(width='190px'),
        )
        self.guess_particle = W.Dropdown(options=[],
                                         description='Edit candidate',
                                         style={'description_width': '160px'},
                                         layout=W.Layout(width='95%'))
        self.x_guess_px = W.IntText(
            value=0,
            description='Manual x guess (vertical, px)',
            style={'description_width': '240px'},
            layout=W.Layout(width='95%'))
        self.y_guess_px = W.IntText(
            value=0,
            description='Manual y guess (horizontal, px)',
            style={'description_width': '240px'},
            layout=W.Layout(width='95%'))
        self.apply_guess_button = W.Button(description='Apply x/y guess',
                                           icon='check',
                                           button_style='info')
        self.reset_guess_button = W.Button(description='Use detected center',
                                           icon='undo')
        self.discard_guess_button = W.Button(description='Discard edits',
                                             icon='times')
        self.guess_status = W.HTML()
        self.settings_validation = W.HTML()
        self.acquisition_summary = W.HTML()
        self.crop_summary = W.HTML()
        self.crop_particle = W.Dropdown(
            options=[], description='Preview particle', disabled=True,
            style={'description_width': '160px'},
            layout=W.Layout(width='95%'),
            tooltip='Choose one selected particle to inspect its crop without changing the analysis selection.')
        self._build_particle_crop_widgets(cfg)
        self.figure_particle = W.Dropdown(
            options=[], description='Result figures', disabled=True,
            style={'description_width': '160px'},
            layout=W.Layout(width='95%'),
            tooltip='Choose an image and particle to view its completed analysis figures.')
        self.figure_notice = W.HTML(value='No analysis figures yet.')
        self.preview_button = W.Button(
            description='Refresh / check particles',
            icon='refresh',
            tooltip='Refresh the map and inspect the crop chosen in Preview particle.',
            layout=W.Layout(width='auto'))
        self.initial_button = W.Button(
            description='Show initial hologram',
            icon='eye',
            tooltip='Simulate the chosen particle using the initial parameters, before fitting.',
            layout=W.Layout(width='auto'))
        self.map_hint = W.HTML()
        self.deselect_mode = W.ToggleButton(
            value=False,
            description='Deselect particle',
            icon='eraser',
            tooltip=
            'Removal-only mode: click near a selected marker to deselect it. Click this button again to resume detection.',
            layout=W.Layout(width='190px'),
        )
        self.click_x_px = W.IntText(
            value=int(cfg.target_x_px),
            description='Point x (vertical, px)',
            style={'description_width': '160px'})
        self.click_y_px = W.IntText(
            value=int(cfg.target_y_px),
            description='Point y (horizontal, px)',
            style={'description_width': '160px'})
        self.detect_point_button = W.Button(
            description='Detect at x/y', icon='crosshairs', disabled=True,
            tooltip='Center and select a particle at these image coordinates when map clicks are unavailable.')
        self.click_coordinate_box = W.Accordion(
            children=[W.VBox([self.click_x_px, self.click_y_px,
                              self.detect_point_button])],
            selected_index=None, layout=W.Layout(display='none'))
        self.click_coordinate_box.set_title(
            0, 'Enter point coordinates (map fallback)')
        self.map_image_only = W.Checkbox(
            value=not cfg.click_to_detect,
            description='Use image map',
            tooltip=
            'Show the numbered image map and use the Particles list to select candidates',
        )
        self._map_snapshot = W.Image(format='png',
                                     layout=W.Layout(width='100%',
                                                     height='auto',
                                                     max_width='100%',
                                                     display='none'))
        self.detection_canvas_box = W.VBox(
            [self._map_snapshot], layout=W.Layout(width='100%'))
        self.current_button = W.Button(description='Analyse this image',
                                       button_style='success',
                                       icon='play')
        self.series_button = W.Button(description='Analyse series',
                                      button_style='success',
                                      icon='play')
        self.current_stop_button = W.Button(
            description='Stop',
            button_style='danger',
            icon='stop',
            disabled=True,
            tooltip='Stop the current analysis and save completed results',
            layout=W.Layout(width='90px'),
        )
        self.series_stop_button = W.Button(
            description='Stop',
            button_style='danger',
            icon='stop',
            disabled=True,
            tooltip='Stop the current analysis and save completed results',
            layout=W.Layout(width='90px'),
        )
        self._stop_buttons = (self.current_stop_button,
                              self.series_stop_button)
        self.batch_mode = W.Dropdown(options=[
            ('Saved choices (click mode skips unchosen images)', 'saved'),
            ('Auto selection in target area', 'auto'),
            ('All detected candidates', 'all')
        ],
                                     value='saved' if cfg.click_to_detect else 'auto',
                                     description='Series mode',
                                     style={'description_width': '110px'},
                                     layout=W.Layout(width='95%'))
        self.profile_picker = PathPicker(
            'JSON profile',
            self.base_dir,
            kind='file',
            value='analysis_profile.json',
            suffixes=('.json', ),
        )
        self.profile_picker.browse.description = 'Choose profile…'
        self.profile_picker.browse.tooltip = 'Choose a saved JSON profile to load'
        # Keep one shared path for manual input, saving and loading profiles.
        self.profile_path = self.profile_picker.path
        self.save_profile_button = W.Button(description='Save profile',
                                            icon='save')
        self.load_profile_button = W.Button(description='Load profile',
                                            icon='upload')
        self.status = W.HTML(
            value=
            'Choose the image directory and background, then click Load images.'
        )
        self.image_progress = W.IntProgress(min=0,
                                            max=1,
                                            value=0,
                                            description='Images')
        self.fit_progress = W.IntProgress(min=0,
                                          max=cfg.initial_search_trials
                                          if cfg.auto_initial_guess else
                                          len(cfg.r_guesses) * len(cfg.z_values),
                                          value=0,
                                          description='Fit trials')
        self.preview_output = W.Output()
        self.fit_output = W.Output()
        self.fit_parameters_copy = W.Textarea(
            value='',
            description='Fitted values',
            placeholder='Fitted x, y, z, radius and refractive index will appear here.',
            tooltip='Click in this box, then use Cmd+A / Cmd+C or Ctrl+A / Ctrl+C.',
            rows=10,
            continuous_update=False,
            style={'description_width': '140px'},
            layout=W.Layout(width='100%'))
        self.message_output = W.Output()
        # Images have their own binary widget state. Captured stdout must never
        # replace their contents when Notebook synchronizes its Output models.
        self.preview_images = W.VBox(layout=W.Layout(width='100%'))
        self.fit_images = W.VBox(layout=W.Layout(width='100%'))
        self.result_links = W.HTML()
        self.result_table = W.HTML()
        self.trial_table = W.HTML()
        module_path = Path(
            __file__).resolve().parent.parent / 'hologram_functions.py'
        self.module_info = W.HTML(
            f'<small>Interface version {html.escape(__version__)} · '
            f'<code>{html.escape(str(module_path))}</code></small>')

    def _build_settings(self, cfg):
        """Build a short workflow and keep detailed controls in folded panels."""
        W = self.W
        self.controls = {}
        metadata = {
            'spacing': ('Pixel spacing (µm/px)',
                'Physical pixel size in micrometres. Use the calibration for your acquisition.'),
            'medium_index': ('Medium refractive index',
                'Refractive index of the liquid surrounding the particle.'),
            'illum_wavelen': ('Wavelength (µm)',
                'Illumination wavelength in micrometres; 660 nm is 0.660 µm.'),
            'NA': ('Objective NA',
                'Objective numerical aperture. It must not exceed the objective refractive index.'),
            'noil': ('Objective refractive index',
                'Refractive index used for the objective/lens model.'),
            'target_x_px': ('Target x (vertical, px)',
                'Vertical coordinate of the automatic selection area, in full-image pixels.'),
            'target_y_px': ('Target y (horizontal, px)',
                'Horizontal coordinate of the automatic selection area, in full-image pixels.'),
            'search_radius_px': ('Search radius (px)',
                'Radius of the automatic selection area; manual selection can include candidates outside it.'),
            'click_to_detect': ('Detect particles on click',
                'Click a particle to refine its centre and adaptive crop locally. No whole-image candidate search is performed.'),
            'detect_size_filter': ('Filter by hologram size',
                'Detect at the requested fringe-pattern sizes and reject smaller/larger patterns. Off preserves the original detector.'),
            'detect_min_diameter_px': ('Min. hologram diameter (px)',
                'Smallest visible hologram fringe-pattern diameter to accept, in image pixels. This is not the fitted physical sphere radius.'),
            'detect_max_diameter_px': ('Max. hologram diameter (px)',
                'Largest visible hologram fringe-pattern diameter to accept, in image pixels. Set a broad range for weak/overlapping fringes.'),
            'detect_sigma': ('Smoothing (sigma)',
                'Gaussian smoothing scale for detection, in pixels. The measured intensities used by the fit are not smoothed.'),
            'detect_min_distance': ('Minimum separation (px)',
                'Minimum distance between detected centers, in pixels; larger values can merge nearby candidates.'),
            'detect_border': ('Excluded border (px)',
                'Ignore candidate centers within this many pixels of an image edge.'),
            'max_particles': ('Maximum candidates',
                'Keep at most this many detections, prioritizing stronger signals; choose which ones to fit afterward.'),
            'refine_center': ('Refine center',
                'Refine centers during click localization or before automatic cropping. Applied manual centers are preserved.'),
            'crop_height_px': ('Fixed-mode height (px)',
                'Vertical size used only when automatic cropping is off; choose an even size of at least 4 pixels.'),
            'crop_width_px': ('Fixed-mode width (px)',
                'Horizontal size used only when automatic cropping is off; choose an even size of at least 4 pixels.'),
            'adaptive_crop': ('Automatic extent + rectangle',
                'Measure each hologram’s fringe extent, then choose a fitting crop using inexpensive image processing.'),
            'adaptive_crop_rectangle': ('Rectangular reduction',
                'Reduce the automatic extent to a rectangular band when fringe sampling permits it.'),
            'adaptive_crop_min_px': ('Crop safety minimum side (px)',
                'Lower bound on either crop side. The x/y optimizer range is restricted to the selected crop.'),
            'adaptive_crop_max_px': ('Adaptive maximum side (px)',
                'Upper bound on either automatic crop side. Very extended fringes may reach this limit.'),
            'adaptive_crop_margin_px': ('Fringe safety margin (px)',
                'Extra margin around the estimated fringe extent before selecting the rectangle.'),
            'auto_initial_guess': ('Find initial parameters automatically',
                'Search radius, particle index and depth within their bounds for each particle, then refine several distinct starts on every crop pixel. Manual guesses remain search hints and preview inputs.'),
            'initial_search_trials': ('Automatic search: local fits',
                'Maximum number of distinct coarse-search candidates to refine on the full crop. More local fits take longer; the best finite, converged full-crop score wins.'),
            'initial_search_max_pixels': ('Automatic search: sampled pixels',
                'Maximum original crop pixels used per coarse simulation. Sampling preserves the calibrated physical pixel coordinates and intensities; final fits use every crop pixel.'),
            'search_effort': ('Search effort',
                'Fast: up to 3 local fits with 1600 sampled pixels. Balanced: up to 5 with 3600. Thorough: run the original broad search with 10 local fits, 3600 sampled pixels and no early finish; refresh enabled experiment profiles. Custom preserves detailed settings.'),
            'adaptive_fit_budget': ('Adaptive fit budget',
                'In automatic initialization, finish early only when independent starts reach the same strong solution. The local-fit count remains a maximum.'),
            'reuse_fit_cache': ('Reuse unchanged calculations',
                'Reuse completed fits only when crop data, coordinates, calibration and fitting settings are unchanged. Figure and saving options still apply to this run.'),
            'reuse_experiment_guesses': ('Reuse learned guesses',
                'Learn useful starting parameters from successful automatic fits in this experiment. Screen learned starts on each new particle and run the full search when they do not match well.'),
            'persist_experiment_guesses': ('Keep experiment profile',
                'Save learned starts in the output directory so this experiment can reuse them after the notebook is closed. Used only with automatic initialization and Reuse learned guesses.'),
            'experiment_name': ('Experiment name',
                'Optional experiment label. Leave empty to use the image directory, acquisition settings and background. Use a different name to separate experiments acquired in the same directory.'),
            'n_guess': ('Initial particle refractive index',
                'Starting particle refractive index in manual mode; a search hint in automatic mode. Also used by Show initial hologram. It remains free within its bounds.'),
            'r_guesses': ('Initial radius guesses (µm)',
                'Radius guesses in micrometres. Manual mode tries each with every depth guess; automatic mode uses them as search hints. The first is used by Show initial hologram.'),
            'illum_polarization': ('Polarization (x, y)',
                'Two polarization components separated by a comma; they cannot both be zero.'),
            'n_bounds': ('Refractive-index bounds (min, max)',
                'Lower and upper limits for the fitted particle refractive index.'),
            'r_bounds': ('Radius bounds (µm)',
                'Lower and upper limits for the fitted particle radius, in micrometres.'),
            'z_values': ('Initial z guesses (µm)',
                'Depth guesses in micrometres. Manual mode tries each with every radius guess; automatic mode uses them as search hints. Negative values require negative z bounds.'),
            'z_bounds': ('Z bounds (µm)',
                'Lower and upper limits for the fitted depth, in micrometres. Include negative values to explore both sides of focus.'),
            'xy_fit_margin_um': ('Maximum x/y displacement (µm)',
                'Maximum fitted displacement in each direction around the detected or manually applied center.'),
            'show_detection_plots': ('Detection maps during series analysis',
                'Display the detection map during analysis; saving PNGs is controlled separately.'),
            'show_crop_plots': ('Centers and crops during series analysis',
                'Display crop boundaries and normalized crops during analysis; saving PNGs is controlled separately.'),
            'show_fit_plots': ('Best-fit comparisons',
                'Display experimental/best-fit comparisons; saving PNGs is controlled separately.'),
            'save_figures': ('Save analysis figures',
                'Save analysis figures in the output folder even when their notebook display is disabled.'),
            'max_displayed_figures': ('Maximum displayed figures',
                'Maximum images shown per particle in Result figures. Other particles remain selectable; text logs and saved PNGs are preserved.'),
        }
        for name, value in asdict(cfg).items():
            if name == 'detect_search_circle_only':
                continue
            label, help_text = metadata[name]
            options = {
                'description': label,
                'style': {
                    'description_width': '240px'
                },
                'layout': W.Layout(width='95%')
            }
            options['tooltip'] = help_text
            if name == 'search_effort':
                control = W.Dropdown(
                    options=[('Fast', 'fast'), ('Balanced', 'balanced'),
                             ('Thorough', 'thorough'), ('Custom', 'custom')],
                    value=value, **options)
            elif name in ('target_x_px', 'target_y_px'):
                control = W.FloatText(value=value, disabled=True, **options)
                control.tooltip = 'Automatically centered for each image.'
            elif isinstance(value, bool):
                control = W.Checkbox(value=value, **options)
            elif isinstance(value, int):
                control = W.IntText(value=value, **options)
            elif isinstance(value, tuple):
                control = W.Text(value=', '.join(f'{v:g}' for v in value),
                                 **options)
            elif isinstance(value, str):
                control = W.Text(value=value, **options)
            else:
                control = W.FloatText(value=value, **options)
            self.controls[name] = control
        self._syncing_crop_mode = False
        self.crop_mode = W.ToggleButtons(
            options=[('Fixed crop', 'fixed'),
                     ('Automatic crop', 'automatic'),
                     ('Automatic crop + rectangle', 'rectangle')],
            value=('rectangle' if cfg.adaptive_crop_rectangle else 'automatic')
            if cfg.adaptive_crop else 'fixed',
            description='Default crop',
            tooltips=[
                'Use the height and width entered below.',
                'Adapt the crop size to each hologram, retaining its full measured fringe extent.',
                'Adapt the extent, then reduce pixels with a rectangular band when suitable.'
            ],
            layout=W.Layout(width='100%', flex_flow='row wrap'),
        )
        self.crop_mode.observe(self._crop_mode_changed, names='value')
        self.controls['adaptive_crop'].observe(
            lambda _: self._sync_fixed_crop_controls(), names='value')
        self.controls['adaptive_crop_rectangle'].observe(
            lambda _: self._sync_fixed_crop_controls(), names='value')
        self.controls['detect_size_filter'].observe(
            lambda _: self._sync_detection_size_controls(), names='value')
        self.controls['click_to_detect'].observe(
            lambda _: self._sync_detection_size_controls(), names='value')
        self._sync_fixed_crop_controls()

        self._syncing_simple_controls = False
        self.selection_method = W.ToggleButtons(
            options=[('Click', 'click'), ('Automatic', 'automatic')],
            value='click' if cfg.click_to_detect else 'automatic',
            description='Selection method',
            style={'description_width': '140px'},
            tooltips=['Click particles to add or remove them.',
                      'Detect candidates within the centred target area.'],
        )
        self.initialization = W.ToggleButtons(
            options=[('Automatic', 'automatic'), ('Manual', 'manual')],
            value='automatic' if cfg.auto_initial_guess else 'manual',
            description='Initialization',
            style={'description_width': '140px'},
        )
        self.selection_help = W.HTML()
        self.search_hints = W.VBox([
            W.HTML('<p>Automatic mode treats these values as optional hints. '
                   'Manual mode fits every radius / depth combination.</p>'),
            self.controls['n_guess'], self.controls['r_guesses'],
            self.controls['z_values'], self.initial_button,
        ])
        self.optional_search_hints = W.Accordion(
            children=[self.search_hints], selected_index=None)
        self.optional_search_hints.set_title(0, 'Optional search hints')
        self.initial_parameters = W.VBox()
        self.search_effort_row = W.VBox([self.controls['search_effort']])
        self.experiment_guess_settings = W.VBox([
            self.controls['reuse_experiment_guesses'],
            self.controls['persist_experiment_guesses'],
            self.controls['experiment_name'],
            W.HTML('<p style="margin:4px 0;font-size:13px;">A new experiment '
                   'begins with the full search. Later particles first try learned '
                   'starts; a weak or unconfirmed match triggers the full search again. '
                   'Refractive index and depth remain free during fitting. '
                   'Leave the experiment name empty for automatic grouping, '
                   'or name a new experiment to keep it separate in the same '
                   'image directory. Keep the same output directory to reuse '
                   'a saved experiment profile. Thorough always runs the full '
                   'search to refresh the profile.</p>'),
        ])
        self.advanced_toggle = W.ToggleButton(
            value=False, description='Advanced settings', icon='sliders',
            tooltip='Show detection, physical bounds, detailed search budgets, figures and profiles.',
            layout=W.Layout(width='200px'))
        self.acquisition_settings = W.Accordion(children=[W.VBox([
            self.controls[key] for key in
            ('spacing', 'medium_index', 'illum_wavelen', 'NA', 'noil',
             'illum_polarization')
        ])], selected_index=None)
        self.acquisition_settings.set_title(0, 'Acquisition settings')
        self.manual_center_settings = W.Accordion(children=[W.VBox([
            W.HTML('<p>Choose a selected candidate, edit its x/y coordinates, '
                   'then apply the change.</p>'),
            self.guess_particle, self.x_guess_px, self.y_guess_px,
            self.guess_status,
            W.HBox([self.apply_guess_button, self.reset_guess_button,
                    self.discard_guess_button],
                   layout=W.Layout(flex_flow='row wrap')),
        ])], selected_index=None)
        self.manual_center_settings.set_title(
            0, 'Correct selected particle center')
        detection = W.VBox([
            W.HTML('<p>The target area is centred automatically for each image. '
                   'The size filter refers to the visible fringe diameter in pixels.</p>'),
            *[self.controls[key] for key in
              ('search_radius_px', 'detect_size_filter',
               'detect_min_diameter_px', 'detect_max_diameter_px',
               'refine_center', 'adaptive_crop_min_px',
               'adaptive_crop_max_px', 'adaptive_crop_margin_px',
               'detect_sigma', 'detect_min_distance', 'detect_border',
               'max_particles')],
        ])
        fitting = W.VBox([
            W.HTML('<p>Physical bounds limit the search. Include negative and '
                   'positive Z bounds only when both sides of focus are relevant '
                   'to the acquisition. Changing a detailed search budget selects '
                   '<b>Custom</b> effort.</p>'),
            *[self.controls[key] for key in
              ('z_bounds', 'n_bounds', 'r_bounds', 'xy_fit_margin_um',
               'initial_search_trials', 'initial_search_max_pixels',
               'adaptive_fit_budget', 'reuse_fit_cache')],
        ])
        figures = W.VBox([
            W.HTML('<p>Display and saving are independent. These options keep '
                   'your existing display preferences.</p>'),
            *[self.controls[key] for key in
              ('show_crop_plots', 'show_fit_plots',
               'save_figures', 'max_displayed_figures')],
        ])
        files_profiles = W.VBox([
            self.output_folder.widget, self.analysis_folder_name, self.pattern,
            self.profile_picker.widget,
            W.HBox([self.save_profile_button, self.load_profile_button]),
        ])
        self.advanced_settings = W.Accordion(
            children=[detection, fitting, figures, files_profiles],
            selected_index=None, layout=W.Layout(display='none'))
        for index, title in enumerate(('Detection and cropping',
                                       'Fitting settings', 'Figures and output',
                                       'Files, series and profiles')):
            self.advanced_settings.set_title(index, title)
        self.selection_method.observe(self._selection_method_changed, names='value')
        self.initialization.observe(self._initialization_changed, names='value')
        self.controls['search_effort'].observe(self._search_effort_changed, names='value')
        for name in ('initial_search_trials', 'initial_search_max_pixels',
                     'adaptive_fit_budget'):
            self.controls[name].observe(self._search_budget_changed, names='value')
        self.advanced_toggle.observe(lambda _: self._sync_simple_controls(), names='value')
        self._sync_simple_controls()
        return self.advanced_settings

    def _crop_mode_changed(self, change):
        """Translate the exclusive mode buttons into saved configuration flags."""
        if self._syncing_crop_mode:
            return
        self._syncing_crop_mode = True
        previous_changing = self._changing
        self._changing = True
        try:
            self.controls['adaptive_crop'].value = change['new'] != 'fixed'
            self.controls['adaptive_crop_rectangle'].value = change[
                'new'] == 'rectangle'
        finally:
            self._changing = previous_changing
            self._syncing_crop_mode = False
        self._sync_fixed_crop_controls()
        self._settings_changed(change)

    def _sync_fixed_crop_controls(self):
        """Inactive fixed dimensions never drive automatic crop selection."""
        if not self._syncing_crop_mode:
            self._syncing_crop_mode = True
            try:
                self.crop_mode.value = (
                    'rectangle' if self.controls['adaptive_crop_rectangle'].
                    value else 'automatic'
                ) if self.controls['adaptive_crop'].value else 'fixed'
            finally:
                self._syncing_crop_mode = False
        self.crop_mode.disabled = bool(
            getattr(self, '_busy', False)
            or getattr(self, '_analysis_running', False))
        disabled = bool(
            getattr(self, '_busy', False)
            or getattr(self, '_analysis_running', False)
            or self.controls['adaptive_crop'].value)
        for name in ('crop_height_px', 'crop_width_px'):
            self.controls[name].disabled = disabled
        self._sync_detection_size_controls()

    def _sync_detection_size_controls(self):
        """Inactive size limits remain visible without affecting detection."""
        size_disabled = bool(
            getattr(self, '_busy', False)
            or getattr(self, '_analysis_running', False)
            or not self.controls['detect_size_filter'].value)
        self.controls['detect_size_filter'].disabled = bool(
            getattr(self, '_busy', False)
            or getattr(self, '_analysis_running', False)
            or self.controls['click_to_detect'].value)
        for name in ('detect_min_diameter_px', 'detect_max_diameter_px'):
            self.controls[name].disabled = size_disabled

    def _build_layout(self, settings):
        """Arrange controls with one Stop button next to each analysis button."""
        W = self.W
        self.analysis_details = W.Accordion(
            children=[self.fit_output], selected_index=None)
        self.analysis_details.set_title(0, 'Analysis details')
        results = W.Accordion(
            children=[self.result_table, self.trial_table,
                      W.VBox([self.figure_particle, self.figure_notice,
                              self.fit_parameters_copy,
                              self.fit_images, self.analysis_details])])
        for i, title in enumerate(
            ['Best fits', 'All fitting trials',
             'Figures and analysis log']):
            results.set_title(i, title)
        results.selected_index = 2
        self.widget = W.VBox([
            W.HTML('<h3>1. Load holograms</h3>'),
            self.image_folder.widget,
            self.background.widget,
            self.load_button,
            self.acquisition_summary,
            self.acquisition_settings,
            W.HTML('<h3>2. Select particles and check the crop</h3>'),
            self.image_navigation,
            self.selection_method,
            self.manual_center_settings,
            self.selection_help,
            W.HBox([self.particle_choice, self.select_all_button],
                   layout=W.Layout(flex_flow='row wrap')),
            self.auto_selection,
            W.HBox([self.preview_button, self.deselect_mode, self.map_image_only],
                   layout=W.Layout(flex_flow='row wrap')),
            self.click_coordinate_box,
            self.map_hint,
            self.detection_canvas_box,
            self.crop_mode,
            self.controls['crop_height_px'],
            self.controls['crop_width_px'],
            self.crop_particle,
            self.particle_crop_settings,
            self.crop_summary,
            self.preview_output,
            self.preview_images,
            W.HTML('<h3>3. Analyse</h3>'),
            self.initialization,
            self.search_effort_row,
            self.experiment_guess_settings,
            self.initial_parameters,
            self.advanced_toggle,
            settings,
            self.settings_validation,
            W.HBox([
                W.HBox([self.current_button, self.current_stop_button]),
                W.HBox([self.series_button, self.series_stop_button]),
            ], layout=W.Layout(flex_flow='row wrap')),
            self.status,
            W.HBox([self.image_progress, self.fit_progress]),
            self.result_links,
            results,
            self.message_output,
            self.module_info,
        ])
