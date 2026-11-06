"""Keep the short notebook workflow synchronized with saved configuration."""
from __future__ import annotations

SEARCH_EFFORT_PRESETS = {
    'fast': (3, 1600, True),
    'balanced': (5, 3600, True),
    'thorough': (10, 3600, False),
}


class SimpleControlsMixin:
    """Presentation controls never alter the acquired data or figure options."""

    def _selection_method_changed(self, change):
        if self._changing or self._syncing_simple_controls:
            return
        click = change['new'] == 'click'
        self.controls['click_to_detect'].value = click
        # Series selection follows the visible workflow; no separate mode
        # control is needed in the interface.
        self.batch_mode.value = 'saved' if click else 'auto'

    def _initialization_changed(self, change):
        if self._changing or self._syncing_simple_controls:
            return
        self.controls['auto_initial_guess'].value = change['new'] == 'automatic'

    def _search_effort_changed(self, change):
        if self._changing or self._syncing_simple_controls:
            return
        preset = SEARCH_EFFORT_PRESETS.get(change['new'])
        if preset is not None:
            self._changing = True
            try:
                trials, pixels, adaptive = preset
                self.controls['initial_search_trials'].value = trials
                self.controls['initial_search_max_pixels'].value = pixels
                self.controls['adaptive_fit_budget'].value = adaptive
            finally:
                self._changing = False
        self._settings_changed(change)

    def _search_budget_changed(self, change):
        """Detailed overrides become Custom without replacing their values."""
        if self._changing or self._syncing_simple_controls:
            return
        if self.controls['search_effort'].value != 'custom':
            self._changing = True
            try:
                self.controls['search_effort'].value = 'custom'
            finally:
                self._changing = False
        self._settings_changed(change)

    def _sync_simple_controls(self):
        """Reflect profile/manual edits without applying a search preset again."""
        if not hasattr(self, 'selection_method'):
            return
        self._syncing_simple_controls = True
        try:
            click = self.controls['click_to_detect'].value
            automatic = self.controls['auto_initial_guess'].value
            self.selection_method.value = 'click' if click else 'automatic'
            self.map_image_only.layout.display = 'none' if click else ''
            self.deselect_mode.layout.display = '' if click else 'none'
            if click and self.map_image_only.value:
                self.map_image_only.value = False
            self.initialization.value = 'automatic' if automatic else 'manual'
            self.initial_parameters.children = (
                (self.optional_search_hints,) if automatic else
                (self.search_hints,))
            self.optional_search_hints.selected_index = (
                self.optional_search_hints.selected_index if automatic else None)
            self.search_effort_row.layout.display = '' if automatic else 'none'
            self.experiment_guess_settings.layout.display = (
                '' if automatic else 'none')
            self.selection_help.value = (
                'Click near a particle on the map to center and select it. '
                'Repeated clicks keep one selection; use Deselect particle '
                'or the eraser to remove it. Use Refresh / check particles to check the crop.'
                if click else
                'Refresh detection, then choose particles on the map or in '
                'the Particles list. Auto in target area selects the centred area.')
            locked = bool(self._busy or self._analysis_running
                          or self._pending_guess or self._pending_crop)
            self.selection_method.disabled = locked
            self.initialization.disabled = locked
            self.controls['search_effort'].disabled = locked or not automatic
            for name in ('initial_search_trials', 'initial_search_max_pixels',
                         'adaptive_fit_budget'):
                self.controls[name].disabled = locked or not automatic
            self.controls['reuse_fit_cache'].disabled = bool(
                self._busy or self._analysis_running)
            self.controls['reuse_experiment_guesses'].disabled = (
                locked or not automatic)
            learned_inactive = (locked or not automatic
                                or not self.controls[
                                    'reuse_experiment_guesses'].value)
            for name in ('persist_experiment_guesses', 'experiment_name'):
                self.controls[name].disabled = learned_inactive
            self.advanced_settings.layout.display = (
                '' if self.advanced_toggle.value else 'none')
            self.auto_selection.layout.display = '' if not click else 'none'
            if self._preview is not None and self._preview[2] is not None:
                previous_changing = self._changing
                self._changing = True
                try:
                    for key, size in zip(('target_x_px', 'target_y_px'),
                                         self._preview[2].shape):
                        self.controls[key].value = (size - 1) / 2
                finally:
                    self._changing = previous_changing
        finally:
            self._syncing_simple_controls = False

    def _update_acquisition_summary(self, cfg):
        """Leave calibration and depth limits visible when details are folded."""
        lo, hi = cfg.z_bounds
        self.acquisition_summary.value = (
            '<p style="margin:6px 0;font-size:13px;">'
            f'<b>Calibration:</b> {cfg.spacing:.9g} µm/px · '
            f'<b>Depth range:</b> {lo:g} to {hi:g} µm · '
            f'<b>Wavelength:</b> {cfg.illum_wavelen:g} µm</p>')

