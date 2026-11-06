"""Interactive detection maps with a static fallback for notebook frontends."""
from __future__ import annotations

import html
import numpy as np


class DetectionInteractionMixin:
    """Render selection and manual corrections without starting a hologram fit."""

    def _render_detection(self, cfg=None):
        if self._preview is None:
            return
        cfg = cfg or self.get_config()
        if cfg.click_to_detect:
            from .selection import clicked_activity
            index, data, image, residual, _, candidates = self._preview
            selected = set(self.particle_choice.value)
            overrides = self.center_overrides.get(self.paths[index].name, {})
            activity = clicked_activity(residual, [
                overrides.get(str(i), c)
                for i, c in enumerate(candidates, 1) if i in selected
            ], cfg)
            self._preview = (index, data, image, residual, activity,
                             candidates)
        if self._canvas_enabled is None:
            try:
                from matplotlib.figure import Figure
                from ipympl.backend_nbagg import FigureManager
                from .canvas import ManagedCanvas
                figure = Figure(constrained_layout=True)
                self._map_axes = figure.subplots(1, 2)
                self._populate_detection(cfg, figure)
                canvas = ManagedCanvas(figure)
                # The manager dispatches frontend mouse/resize messages. Create
                # it explicitly without registering a global pyplot figure.
                self._canvas_manager = FigureManager(canvas, id(self))
                canvas.retain_manager(self._canvas_manager)
                canvas.toolbar_position = 'bottom'
                canvas.layout.width = '100%'
                canvas.layout.height = '900px' if cfg.click_to_detect else '620px'
                canvas.layout.display = 'none'
                canvas.on_frontend_ready = self._update_map_display
                self._canvas = canvas
                self._install_deselect_tool(canvas)
                canvas.mpl_connect('button_press_event', self._on_map_click)
                self.detection_canvas_box.children = (self._map_snapshot,
                                                      canvas)
                self._canvas_enabled = True
                # The constructor already rendered this populated figure.
                self._map_snapshot.value = canvas.prepare_display()
                self._update_map_display()
                return
            except ImportError:
                self._canvas_enabled = False
                self.detection_canvas_box.children = (self._map_snapshot, )
        if not self._canvas_enabled:
            from .plotting import show_detection_maps
            _, _, image, _, activity, candidates = self._preview
            path = self.paths[self.image_choice.value]
            show_detection_maps(
                image,
                activity,
                candidates,
                path.name,
                cfg,
                selected_numbers=self.particle_choice.value,
                selected_only=True,
                center_overrides=self.center_overrides.get(path.name, {}),
                on_figure=lambda png: setattr(self._map_snapshot, 'value', png
                                              ),
                log=lambda _: None,
            )
            self._update_map_display()
            return
        self._draw_detection(cfg)

    def _update_map_display(self):
        """Keep one visible map, with a PNG fallback until frontend drawing works."""
        interactive = (self._canvas_enabled and self._canvas.frontend_ready
                       and not self.map_image_only.value)
        self._sync_click_fallback(interactive)
        if self._canvas is not None:
            was_visible = self._canvas.layout.display != 'none'
            self._canvas.layout.display = '' if interactive else 'none'
            if interactive and not was_visible:
                # Publish layout before pixels; schedule on the kernel loop so
                # the frontend can reveal its view before receiving the frame.
                self._post_ui(self._canvas.refresh_visible)
        self._map_snapshot.layout.display = (
            'none' if interactive or not self._map_snapshot.value else '')
        if interactive and self.controls[
                'click_to_detect'].value and self.deselect_mode.value:
            self.map_hint.value = (
                '<b>Deselect mode:</b> click near a selected marker to remove it. '
                'No particles are detected or added in this mode. '
                'Click the eraser or <b>Deselect particle</b> again to resume detection.'
            )
        elif interactive and self.controls['click_to_detect'].value:
            self.map_hint.value = (
                'Click near a particle to center and select it. '
                'Repeated clicks keep one selection. '
                'Use the eraser or <b>Deselect particle</b> to remove it. '
                'Only selected particles are shown. Turn off toolbar pan/zoom before clicking.'
            )
        elif interactive:
            self.map_hint.value = (
                'Click near a numbered marker to select or deselect it. '
                'Use <b>Edit candidate</b> and the x/y fields to correct a center, '
                'then click <b>Apply x/y guess</b>. '
                'Turn off the toolbar pan/zoom mode before selecting. '
                'If the canvas is blank, enable <b>Use image map</b>.')
        elif self._map_snapshot.value:
            if self.deselect_mode.value and self.controls[
                    'click_to_detect'].value:
                self.map_hint.value = (
                    '<b>Deselect mode:</b> turn off <b>Use image map</b> for interactive removal, '
                    'or deselect entries in the <b>Particles</b> list. '
                    'Disable <b>Deselect particle</b> to resume detection.')
                return
            reason = (
                'Image map mode: turn off <b>Use image map</b> for interactive clicks. '
                if self.map_image_only.value else
                'Interactive clicks are waiting for the notebook connection. '
                if self._canvas_enabled else
                'Interactive clicks are unavailable in this notebook. ')
            self.map_hint.value = (
                reason +
                ('Use <b>Enter point coordinates</b> to center and select a particle. '
                 if self.controls['click_to_detect'].value else '') +
                'Numbered particle map: use the <b>Particles</b> list to select '
                'one or several candidates. Green markers show your selection; '
                'magenta markers show corrected centers. Use the manual x/y '
                'fields to adjust a center.')

    def _sync_click_fallback(self, interactive):
        """Offer point entry only while the click map is a static image."""
        fallback = (self.controls['click_to_detect'].value
                    and self._preview is not None and not interactive)
        self.click_coordinate_box.layout.display = '' if fallback else 'none'
        disabled = (not fallback or not self.paths or self._busy
                    or self._analysis_running or self._pending_guess
                    or self._pending_crop
                    or bool(self._settings_error) or self.deselect_mode.value)
        for control in (self.click_x_px, self.click_y_px,
                        self.detect_point_button):
            control.disabled = disabled

    def _detect_coordinate_particle(self, _):
        """Use the same localization path as a click, including its edit guards."""
        if self.detect_point_button.disabled:
            return
        self._perform(
            lambda: self._detect_clicked_particle(self.click_x_px.value,
                                                   self.click_y_px.value),
            self.preview_output)

    def _install_deselect_tool(self, canvas):
        """Add a removal-mode toggle to the existing ipympl navigation toolbar."""
        toolbar = canvas.toolbar
        toolbar.deselect_particle = self._toggle_deselect_mode
        toolbar.toolitems = list(toolbar.toolitems) + [(
            'Deselect particle',
            'Toggle removal-only mode; click near a selected marker to deselect it.',
            'eraser', 'deselect_particle')]

    def _toggle_deselect_mode(self, *args):
        """Handle the native toolbar button with the same state as the UI toggle."""
        if not self.deselect_mode.disabled:
            self.deselect_mode.value = not self.deselect_mode.value

    def _deselect_mode_changed(self, change):
        """Keep removal mode explicit and release pan/zoom before removing markers."""
        self.deselect_mode.button_style = 'danger' if change['new'] else ''
        toolbar = getattr(getattr(self, '_canvas', None), 'toolbar', None)
        if change['new'] and toolbar is not None:
            mode = str(getattr(toolbar, 'mode', '')).lower()
            if 'pan' in mode:
                toolbar.pan()
            elif 'zoom' in mode:
                toolbar.zoom()
        self._availability()
        if (self._canvas_enabled and self._preview is not None
                and not self._busy and not self._settings_error):
            self._draw_detection(self.get_config())

    def _draw_detection(self, cfg):
        self._populate_detection(cfg, self._canvas.figure)
        self._canvas.draw()
        self._map_snapshot.value = self._canvas.prepare_display()
        self._update_map_display()

    def _populate_detection(self, cfg, figure):
        """Draw markers before publishing the widget's initial image to Jupyter."""
        from matplotlib.patches import Circle
        index, _, image, _, activity, candidates = self._preview
        cfg = cfg.centered_for_image(image.shape)
        path = self.paths[index]
        left, right = self._map_axes
        # Give the clickable hologram most of the map instead of splitting the
        # available notebook width equally with its diagnostic activity image.
        # Only reset dimensions on a mode change; preserve frontend resizes
        # between clicks and Matplotlib's data-to-screen coordinate mapping.
        if getattr(self, '_map_click_layout', None) != cfg.click_to_detect:
            figure.set_size_inches((14, 9) if cfg.click_to_detect else (12, 5.8))
            left.get_gridspec().set_width_ratios(
                (3, 1) if cfg.click_to_detect else (1, 1))
            self._map_click_layout = cfg.click_to_detect
            if self._canvas is not None:
                self._canvas.layout.height = (
                    '900px' if cfg.click_to_detect else '620px')
        for axis in self._map_axes:
            axis.clear()
            axis.set_xlabel('y HoloPy (pixels, horizontal)')
            axis.set_ylabel('x HoloPy (pixels, vertical)')
        left.imshow(image, cmap='gray', origin='upper')
        right.imshow(activity, cmap='magma', origin='upper')
        if not cfg.click_to_detect:
            left.add_patch(
                Circle(
                    (cfg.target_y_px, cfg.target_x_px),
                    cfg.search_radius_px,
                    fill=False,
                    edgecolor='cyan',
                    linewidth=1.5,
                ))
            left.plot(cfg.target_y_px, cfg.target_x_px, 'c+')
        overrides = self.center_overrides.get(path.name, {})
        selected = set(self.particle_choice.value)
        for number, detected in enumerate(candidates, 1):
            if number not in selected:
                continue
            center = overrides.get(str(number), detected)
            color = 'lime' if number in selected else 'yellow'
            left.plot(center['y'],
                      center['x'],
                      'o',
                      color=color,
                      fillstyle='none',
                      markersize=14)
            left.text(center['y'] + 8,
                      center['x'] - 8,
                      str(number),
                      color=color,
                      weight='bold')
            if str(number) in overrides:
                left.plot(detected['y'],
                          detected['x'],
                          'o',
                          color='silver',
                          fillstyle='none',
                          markersize=10,
                          alpha=0.6)
                left.plot([detected['y'], center['y']],
                          [detected['x'], center['x']],
                          '--',
                          color='magenta',
                          linewidth=1)
                left.plot(center['y'],
                          center['x'],
                          '+',
                          color='magenta',
                          markersize=12)
        if self._pending_guess:
            left.plot(self.y_guess_px.value,
                      self.x_guess_px.value,
                      'c+',
                      markersize=16)
        left.set_xlim(-.5, image.shape[1] - .5)
        left.set_ylim(image.shape[0] - .5, -.5)
        left.set_title('Deselect mode: click near a selected marker'
                       if cfg.click_to_detect and self.deselect_mode.value else
                       'Click near a particle to center and select it'
                       if cfg.click_to_detect else
                       'Green = selected; magenta = corrected center')
        right.set_title('Local fringe activity' if cfg.
                        click_to_detect else 'Fringe activity')
        figure.suptitle(path.name)

    def _on_map_click(self, event):
        """Detect locally or toggle a candidate without starting a fit."""
        if self._busy or self._preview is None or event.button != 1:
            return
        if event.inaxes is not self._map_axes[
                0] or event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(getattr(self, '_canvas', None), 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', ''):
            return
        try:
            self._require_current_preview()
            if self.get_config().click_to_detect:
                if self.deselect_mode.value:
                    self._perform(
                        lambda: self._deselect_nearest_particle(
                            event.ydata, event.xdata), self.preview_output)
                    return
                self._perform(
                    lambda: self._detect_clicked_particle(
                        int(round(event.ydata)), int(round(event.xdata))),
                    self.preview_output)
                return
            self._check_pending_guess()
            candidates = self._preview[-1]
            if not candidates:
                return
            path = self.paths[self.image_choice.value]
            overrides = self.center_overrides.get(path.name, {})
            centers = [
                overrides.get(str(number), candidate)
                for number, candidate in enumerate(candidates, 1)
            ]
            distances = [
                np.hypot(c['x'] - event.ydata, c['y'] - event.xdata)
                for c in centers
            ]
            nearest = int(np.argmin(distances))
            if distances[nearest] > 25:
                return
            numbers = set(self.particle_choice.value)
            number = nearest + 1
            if number in numbers:
                numbers.remove(number)
            else:
                numbers.add(number)
            self.particle_choice.value = tuple(sorted(numbers))
        except ValueError as exc:
            self.status.value = f'<span style="color:#a32020">{html.escape(str(exc))}</span>'

    def _deselect_nearest_particle(self, row, col):
        """Remove only an existing selection; never localize, add or fit a particle."""
        self._check_pending_guess()
        cfg = self._require_current_preview()
        index, _, image, _, _, candidates = self._preview
        selected = tuple(self.particle_choice.value)
        if not selected:
            self.status.value = 'No selected particles to deselect.'
            return
        if not np.isfinite([row, col]).all() or not (
                0 <= row < image.shape[0] and 0 <= col < image.shape[1]):
            return
        path = self.paths[index]
        overrides = self.center_overrides.get(path.name, {})
        distances = []
        for number in selected:
            center = overrides.get(str(number), candidates[number - 1])
            distances.append(np.hypot(center['x'] - row, center['y'] - col))
        nearest = int(np.argmin(distances))
        if distances[nearest] > 40:
            self.status.value = 'No selected marker near this click; no particles were added.'
            return
        removed = selected[nearest]
        remaining = tuple(number for number in selected if number != removed)
        self._changing = True
        try:
            self.auto_selection.value = False
            self.particle_choice.value = remaining
            self.choices[path.name] = list(remaining)
        finally:
            self._changing = False
        self._synchronize_editor()
        self._crop_summaries.pop(removed, None)
        self._render_crop_summary()
        self._render_detection(cfg)
        self._show_selected_crop()
        self.status.value = f'Particle {removed} deselected. Deselect mode is still active.'
        self._availability()

    def _detect_clicked_particle(self, row, col):
        """Center one particle and reuse its identifier on repeated clicks."""
        if self.deselect_mode.value:
            return self._deselect_nearest_particle(row, col)
        from .selection import localize_particle, clicked_activity
        self._check_pending_guess()
        self._require_current_preview()
        cfg = self.get_config()
        index, data, image, residual, activity, candidates = self._preview
        path = self.paths[index]
        if not cfg.click_to_detect:
            raise ValueError(
                'Enable Detect particles on click for local selection.')
        selected = set(self.particle_choice.value)
        overrides = self.center_overrides.get(path.name, {})
        hit_radius = max(4, min(12, cfg.detect_min_distance / 2))

        def nearest_match(point, selected_only=False):
            matches = []
            for i, candidate in enumerate(candidates):
                if selected_only and i + 1 not in selected:
                    continue
                # A manual correction and its original detected position both
                # refer to the same stable particle identifier.
                centers = (candidate, overrides.get(str(i + 1), candidate))
                distance = min(np.hypot(c['x'] - point[0], c['y'] - point[1])
                               for c in centers)
                if distance <= hit_radius:
                    matches.append((distance, i))
            return min(matches)[1] if matches else None

        hit = nearest_match((row, col), selected_only=True)
        info = {}
        already_selected = False
        if hit is not None and str(hit + 1) in overrides:
            # An explicit manual center takes precedence over another click.
            number = hit + 1
            already_selected = number in selected
            selected.add(number)
        else:
            neighbors = [
                overrides.get(str(i), c) for i, c in enumerate(candidates, 1)
            ]
            candidate, info, _ = localize_particle(residual, row, col, cfg,
                                                   neighbors,
                                                   measure_crop=False)
            existing = hit if hit is not None else nearest_match(
                (candidate['x'], candidate['y']))
            if info.get('center_confident') is False:
                if existing is None:
                    self.status.value = (
                        'No reliable particle center found near this click. '
                        'Click closer to its center; no new particle was selected.')
                    return
                # Keep an existing selection rather than replacing its center
                # with an uncertain click or adding an unverified duplicate.
                candidate = dict(candidates[existing])
            candidates = [dict(c) for c in candidates]
            if existing is None:
                candidates.append(candidate)
                number = len(candidates)
            else:
                number = existing + 1
                already_selected = number in selected
                if str(number) not in overrides:
                    candidates[existing] = candidate
            selected.add(number)
        self.clicked_particles[path.name] = [dict(c) for c in candidates]
        activity = clicked_activity(residual, candidates, cfg)
        self._preview = (index, data, image, residual, activity, candidates)
        self._crop_cache.clear()
        self._changing = True
        try:
            self.auto_selection.value = False
            self.particle_choice.options = self._candidate_options()
            self.particle_choice.value = tuple(sorted(selected))
            self.choices[path.name] = list(sorted(selected))
        finally:
            self._changing = False
        self._synchronize_editor()
        self._render_detection(cfg)
        self.guess_particle.value = number
        crop_info = self._show_candidate_crop(number, cfg)
        warning = ' '.join(dict.fromkeys(filter(None, (
            info.get('warning', ''), crop_info.get('warning', '')))))
        message = (f'Particle {number} is already selected; no duplicate was added.'
                   if already_selected else
                   f'Particle {number} centered and selected from your click.')
        self.status.value = f'{message} {html.escape(warning)}'
        self._availability()
