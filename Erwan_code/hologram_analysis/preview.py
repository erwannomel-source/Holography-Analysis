"""Cached image previews, particle crops and staged manual center corrections."""
from __future__ import annotations

import html
from pathlib import Path
import holopy as hp
from .profiles import resolve_path
from .preparation import load_and_detect, normalize_particle_choice, select_candidates, prepare_particle, apply_center_override, safe_crop_shape
from .plotting import show_crop_control, show_initial_guess
from .notebook_images import clear_images
from .particle_crops import particle_crop_config
from .preparation_cache import file_fingerprint, prepared_particle_key


class NotebookPreviewMixin:
    """Preview operations share caches and never start an optimizer."""

    def _sync_crop_particle(self):
        """Keep the displayed crop separate from the particles chosen for fitting."""
        numbers = tuple(self.particle_choice.value) if self._preview is not None else ()
        previous = self.crop_particle.value
        changing = self._changing
        self._changing = True
        try:
            self.crop_particle.options = [(f'Candidate {n}', n) for n in numbers]
            self.crop_particle.value = previous if previous in numbers else (
                numbers[0] if numbers else None)
        finally:
            self._changing = changing
        if previous != self.crop_particle.value or not numbers:
            clear_images(self.preview_images)
            self._crop_summaries.clear()
            self.crop_summary.value = ''
        self.crop_particle.disabled = (not numbers or self._busy
                                       or self._pending_guess or self._pending_crop
                                       or bool(self._settings_error))
        self._load_particle_crop_fields()

    def _crop_particle_changed(self, change):
        if self._changing or self._busy or change['new'] is None:
            return
        self._perform(self._show_selected_crop, self.preview_output)

    def _show_selected_crop(self):
        self._check_pending_guess()
        cfg = self._require_current_preview()
        number = self.crop_particle.value
        if number is not None and number in self.particle_choice.value:
            self._show_candidate_crop(int(number), cfg)

    def _background_image(self, cfg):
        path = resolve_path(self.background.path.value, self.base_dir)
        key = (self._file_fingerprint(path), cfg.spacing)
        if self._bg_cache is None or self._bg_cache[0] != key:
            self._bg_cache = (key, hp.load_image(str(path),
                                                 spacing=cfg.spacing))
        return self._bg_cache[1]

    def _synchronize_detection(self, cfg):
        key = cfg.detection_key()
        if self._detection_key is not None and self._detection_key != key:
            self.choices.clear()
            self.center_overrides.clear()
            self.clicked_particles.clear()
            self.crop_overrides.clear()
            self._preview = None
            self._preview_fingerprint = None
            self._detection_cache.clear()
            self._crop_cache.clear()
            self._seen_fingerprints.clear()
            self.status.value = (
                'Detection changed; saved selections and manual centers were reset.'
            )
        self._detection_key = key

    @staticmethod
    def _file_fingerprint(path):
        """Use file size and modification time to invalidate replaced images."""
        return file_fingerprint(path)

    def _detection_fingerprint(self, cfg):
        path = self.paths[self.image_choice.value]
        bg = resolve_path(self.background.path.value, self.base_dir)
        return self._file_fingerprint(path), self._file_fingerprint(
            bg), cfg.detection_key()

    def _cached_detection(self, cfg):
        fingerprint = self._detection_fingerprint(cfg)
        if fingerprint not in self._detection_cache:
            self._detection_cache[fingerprint] = load_and_detect(
                self.paths[self.image_choice.value],
                self._background_image(cfg),
                cfg,
            )
            if len(self._detection_cache) > 3:
                removed, _ = self._detection_cache.popitem(last=False)
                # Crops can be views into an image; evict them with that image.
                for crop_key in list(self._crop_cache):
                    if crop_key[0] == removed:
                        del self._crop_cache[crop_key]
        self._detection_cache.move_to_end(fingerprint)
        return self._detection_cache[fingerprint], fingerprint

    def _require_current_preview(self):
        """Reject edits against an image or detector that changed since preview."""
        self._check_dataset()
        cfg = self.get_config()
        if (self._preview is None
                or self._preview[0] != self.image_choice.value
                or self._detection_key != cfg.detection_key()):
            raise ValueError(
                'Click Refresh / check particles before editing: the image or detection settings changed.'
            )
        if (self._preview_fingerprint is not None and self._preview_fingerprint
                != self._detection_fingerprint(cfg)):
            raise ValueError(
                'The image or background file changed. Click Refresh / check particles before editing.'
            )
        return cfg

    def _check_pending_guess(self):
        if self._pending_guess:
            raise ValueError(
                'Apply x/y guess or Discard edits before continuing.')
        if self._pending_crop:
            raise ValueError(
                'Apply particle crop or Discard crop edits before continuing.')

    def _check_saved_image_fingerprints(self):
        """Avoid applying saved candidate numbers to files replaced since selection."""
        for path in self.paths:
            if (path.name not in self.choices
                    and path.name not in self.center_overrides
                    and path.name not in self.crop_overrides
                    and path.name not in self.clicked_particles):
                continue
            previous = self._seen_fingerprints.get(str(path))
            if previous is not None and previous[0] != self._file_fingerprint(
                    path):
                raise ValueError(
                    f'{path.name} changed since its particle selection. '
                    'Refresh that image before analysing or saving.')

    def _load_manual_center_fields(self):
        """Show the detected or saved center for the candidate selected in the editor."""
        if self._preview is None or self.guess_particle.value is None or not self.paths:
            self._manual_baseline = None
            self._pending_guess = False
            self.guess_status.value = 'Select a particle to edit its center.'
            return
        number = int(self.guess_particle.value)
        candidates = self._preview[-1]
        if not 1 <= number <= len(candidates):
            return
        path = self.paths[self.image_choice.value]
        override = self.center_overrides.get(path.name, {}).get(str(number))
        center = override or candidates[number - 1]
        self._changing = True
        try:
            self.x_guess_px.value = int(center['x'])
            self.y_guess_px.value = int(center['y'])
        finally:
            self._changing = False
        self._manual_baseline = (int(center['x']), int(center['y']))
        self._pending_guess = False
        self.guess_status.value = ('Applied manual center.' if override else
                                   'Using the detected center.')

    def _manual_guess_changed(self, change):
        if self._changing or self._busy or self.guess_particle.value is None:
            return
        values = (int(self.x_guess_px.value), int(self.y_guess_px.value))
        self._pending_guess = self._manual_baseline is not None and values != self._manual_baseline
        self.guess_status.value = (
            '<span style="color:#a36100">Unapplied x/y edits: click Apply x/y guess '
            'or Discard edits before analysing or saving.</span>' if
            self._pending_guess else 'Center fields match the saved center.')
        self._availability()

    def _discard_manual_edits(self):
        if self._busy:
            return
        self._load_manual_center_fields()
        self._availability()
        if self._canvas_enabled and self._preview is not None:
            self._draw_detection(self.get_config())

    def _guess_particle_changed(self, change):
        if self._changing or self._busy:
            return
        self._load_manual_center_fields()
        self._availability()

    def _apply_manual_center(self):
        """Store a per-image, per-candidate x/y center and refresh the crop preview."""
        cfg = self._require_current_preview()
        if self._preview is None or self.guess_particle.value is None:
            raise ValueError(
                'Show an image and choose a candidate before editing its x/y guess.'
            )
        number = int(self.guess_particle.value)
        cfg = particle_crop_config(cfg, self.crop_overrides.get(
            self.paths[self.image_choice.value].name, {}).get(str(number)))
        override = {
            'x': int(self.x_guess_px.value),
            'y': int(self.y_guess_px.value)
        }
        apply_center_override(self._preview[-1][number - 1], override,
                              self._preview[2].shape)
        # Validate crop geometry before committing a center that cannot be used.
        if cfg.adaptive_crop:
            from .cropping import bounded_window
            bounded_window(
                self._preview[2].shape, (override['x'], override['y']),
                (cfg.adaptive_crop_min_px, cfg.adaptive_crop_min_px))
        else:
            safe_crop_shape(self._preview[2].shape, override['x'],
                            override['y'], cfg.crop_height_px,
                            cfg.crop_width_px)
        path = self.paths[self.image_choice.value]
        self.center_overrides.setdefault(path.name, {})[str(number)] = override
        self._load_manual_center_fields()
        self._refresh_corrected_candidate(number, cfg)
        self.status.value = (
            f'Candidate {number} in {html.escape(path.name)} now uses '
            f'x={override["x"]}, y={override["y"]} px.')

    def _reset_manual_center(self):
        """Remove the current candidate's override and return to its detected center."""
        cfg = self._require_current_preview()
        if self._preview is None or self.guess_particle.value is None:
            raise ValueError('Show an image and choose a candidate first.')
        number = int(self.guess_particle.value)
        path = self.paths[self.image_choice.value]
        image_overrides = self.center_overrides.get(path.name, {})
        image_overrides.pop(str(number), None)
        if not image_overrides:
            self.center_overrides.pop(path.name, None)
        self._load_manual_center_fields()
        self._refresh_corrected_candidate(number, cfg)
        self.status.value = (
            f'Candidate {number} in {html.escape(path.name)} uses its detected center again.'
        )

    def _preview_image(self, show_initial=False):
        """Reuse loaded images and crops; initial simulation is an explicit action."""
        self._check_pending_guess()
        self._check_dataset()
        cfg = self.get_config()
        self._synchronize_detection(cfg)
        index = self.image_choice.value
        path = self.paths[index]
        loaded, fingerprint = self._cached_detection(cfg)
        data, image, residual, activity, candidates = loaded
        cfg = cfg.centered_for_image(image.shape)
        self._changing = True
        try:
            self.controls['target_x_px'].value = cfg.target_x_px
            self.controls['target_y_px'].value = cfg.target_y_px
        finally:
            self._changing = False
        background_changed = (self._background_fingerprint is not None and
                              self._background_fingerprint != fingerprint[1])
        if background_changed:
            self.choices.clear()
            self.center_overrides.clear()
            self.clicked_particles.clear()
            self.crop_overrides.clear()
            self._crop_cache.clear()
            self._seen_fingerprints.clear()
            self.status.value = (
                'Background changed; saved selections and manual centers were reset.'
            )
        previous_fingerprint = self._seen_fingerprints.get(str(path))
        if previous_fingerprint is not None and previous_fingerprint != fingerprint:
            self.choices.pop(path.name, None)
            self.center_overrides.pop(path.name, None)
            self.clicked_particles.pop(path.name, None)
            self.crop_overrides.pop(path.name, None)
            self.status.value = (
                'Image or background changed; this image\'s saved selections were reset.'
            )
        self._background_fingerprint = fingerprint[1]
        self._seen_fingerprints[str(path)] = fingerprint
        if cfg.click_to_detect:
            from .selection import clicked_activity, validate_clicked_particles
            candidates = validate_clicked_particles(
                self.clicked_particles.get(path.name, []), image.shape)
            self.clicked_particles[path.name] = candidates
            activity = clicked_activity(residual, candidates, cfg)
        self._preview = (index, data, image, residual, activity, candidates)
        self._preview_fingerprint = fingerprint
        self._crop_summaries.clear()
        self.crop_summary.value = ''
        options = self._candidate_options()
        choice = normalize_particle_choice(self.choices.get(path.name))
        automatic = choice is None and not cfg.click_to_detect
        if cfg.click_to_detect and choice is None:
            choice = list(range(1, len(candidates) + 1))
        if automatic:
            try:
                selected = select_candidates(candidates, cfg)
            except ValueError:
                selected = []
            numbers = tuple(number for number, _ in selected)
        elif choice == -1:
            selected = list(enumerate(candidates, 1))
            numbers = tuple(number for number, _ in selected)
        else:
            saved_numbers = [choice] if isinstance(choice, int) else choice
            numbers = tuple(number for number in saved_numbers
                            if 1 <= number <= len(candidates))
            if len(numbers) != len(saved_numbers):
                self.status.value = (
                    'Saved selection adjusted because some candidates are no longer available.'
                )
                self.choices[path.name] = list(numbers)
            selected = select_candidates(candidates,
                                         cfg,
                                         particle_number=numbers)
        self._changing = True
        try:
            self.auto_selection.value = automatic
            self.particle_choice.options = options
            self.particle_choice.value = numbers
        finally:
            self._changing = False
        self._synchronize_editor()
        self._render_detection(cfg)
        if not candidates and cfg.click_to_detect:
            self.status.value = 'Click a particle on the image to detect it locally; no particles have been chosen yet.'
            self._availability()
            return
        if not candidates:
            self.status.value = 'No candidates detected: adjust the detection settings.'
            return
        if not selected:
            self.status.value = (
                'No candidate selected automatically. Select particles manually in the list.'
                if automatic else
                'No particles selected. Select particles manually in the list.'
            )
            return
        self._show_candidate_crop(int(self.crop_particle.value), cfg)
        if show_initial:
            self._show_edited_initial()
        self.status.value = f'Image {index} checked: {len(candidates)} candidate(s), {len(selected)} selected.'

    def _candidate_options(self):
        path = self.paths[self.image_choice.value]
        overrides = self.center_overrides.get(path.name, {})
        options = []
        for number, detected in enumerate(self._preview[-1], 1):
            center = overrides.get(str(number), detected)
            suffix = ' (manual)' if str(number) in overrides else ''
            options.append((
                f'Candidate {number} — x={center["x"]}, y={center["y"]} px{suffix}',
                number,
            ))
        return options

    def _prepared_candidate(self, number, cfg):
        index, data, image, residual, _, candidates = self._preview
        path = self.paths[index]
        cfg = particle_crop_config(cfg, self.crop_overrides.get(path.name, {}).get(str(number)))
        override = self.center_overrides.get(path.name, {}).get(str(number))
        candidate = apply_center_override(candidates[number - 1], override,
                                          image.shape)
        neighbors = [
            apply_center_override(
                c,
                self.center_overrides.get(path.name, {}).get(str(i)),
                image.shape) for i, c in enumerate(candidates, 1)
            if i != number
        ]
        key = prepared_particle_key(
            self._preview_fingerprint or (str(path), id(data)),
            number, candidate, cfg, neighbors)
        if key not in self._crop_cache:
            self._crop_cache[key] = prepare_particle(data,
                                                     image,
                                                     residual,
                                                     candidate,
                                                     cfg,
                                                     log=lambda _: None,
                                                     neighbors=neighbors)
            if len(self._crop_cache) > 32:
                self._crop_cache.popitem(last=False)
        self._crop_cache.move_to_end(key)
        return self._crop_cache[key]

    def _show_candidate_crop(self, number, cfg):
        h, x, y, crop_shape, xguess, yguess = self._prepared_candidate(
            number, cfg)
        path = self.paths[self.image_choice.value]
        info = getattr(h, 'attrs', {}).get('crop_selection', {})
        changing = self._changing
        self._changing = True
        try:
            self.crop_particle.value = number
        finally:
            self._changing = changing
        self._load_particle_crop_fields()
        clear_images(self.preview_images)
        self._crop_summaries.clear()
        self._update_crop_summary(number, crop_shape, info)
        show_crop_control(
            self._preview[2],
            h,
            x,
            y,
            crop_shape,
            f'{path.name} — candidate {number}',
            on_figure=self._show_preview_figure,
        )
        return info

    def _update_crop_summary(self, number, crop_shape, info):
        """Summarize prepared crops without recomputing detection or simulations."""
        height, width = map(int, crop_shape)
        confidence = info.get('extent_confidence', 'fixed crop')
        self._crop_summaries[number] = (height, width, confidence)
        self._render_crop_summary()

    def _render_crop_summary(self):
        """Refresh summary rows after a crop update or removal of a selection."""
        if not self._crop_summaries:
            self.crop_summary.value = ''
            return
        style = 'padding:4px 10px;text-align:right;white-space:nowrap;'
        header = ''.join(f'<th style="{style}">{label}</th>'
                         for label in ('Candidate', 'Height × width (px)',
                                       'Pixels', 'Extent confidence'))
        rows = ''
        for candidate, (height, width,
                        confidence) in self._crop_summaries.items():
            cells = (str(candidate), f'{height} × {width}',
                     f'{height * width:,}', html.escape(str(confidence)))
            rows += '<tr>' + ''.join(f'<td style="{style}">{cell}</td>'
                                     for cell in cells) + '</tr>'
        path = self.paths[self.image_choice.value]
        self.crop_summary.value = (
            '<div style="max-width:100%;overflow:auto;font-size:13px;">'
            f'<p style="margin:6px 0;"><b>Crop summary:</b> {html.escape(path.name)}</p>'
            '<table style="width:max-content;table-layout:auto;border-collapse:collapse;">'
            f'<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _refresh_corrected_candidate(self, number, cfg):
        """Update only the edited crop; preserve other cached candidates."""
        self._changing = True
        try:
            numbers = self.particle_choice.value
            self.particle_choice.options = self._candidate_options()
            self.particle_choice.value = numbers
        finally:
            self._changing = False
        self._render_detection(cfg)
        self._show_candidate_crop(number, cfg)

    def _show_edited_initial(self):
        """Simulate one initial hologram only when explicitly requested."""
        self._check_pending_guess()
        cfg = self._require_current_preview()
        number = self.guess_particle.value
        if number is None:
            raise ValueError(
                'Choose a particle in Edit candidate before simulating an initial hologram.'
            )
        h, _, _, _, xguess, yguess = self._prepared_candidate(int(number), cfg)
        show_initial_guess(h,
                           xguess,
                           yguess,
                           cfg,
                           on_figure=self._show_preview_figure)

