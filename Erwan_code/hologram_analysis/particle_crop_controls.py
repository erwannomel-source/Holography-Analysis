"""Per-particle crop editing beside the crop preview selector."""
from __future__ import annotations

import html

from .particle_crops import normalize_particle_crops


class ParticleCropControlsMixin:
    def _build_particle_crop_widgets(self, cfg):
        W = self.W
        self.particle_crop_mode = W.Dropdown(
            options=[('Use default crop', 'default'), ('Fixed crop', 'fixed'),
                     ('Automatic crop', 'automatic'),
                     ('Automatic crop + rectangle', 'rectangle')],
            value='default', description='Crop for particle',
            style={'description_width': '160px'}, layout=W.Layout(width='95%'))
        self.particle_crop_height = W.IntText(
            value=cfg.crop_height_px, description='Particle height (px)',
            style={'description_width': '160px'}, layout=W.Layout(width='95%'))
        self.particle_crop_width = W.IntText(
            value=cfg.crop_width_px, description='Particle width (px)',
            style={'description_width': '160px'}, layout=W.Layout(width='95%'))
        self.apply_particle_crop = W.Button(description='Apply particle crop',
                                           icon='check', button_style='info')
        self.discard_particle_crop = W.Button(description='Discard crop edits',
                                             icon='undo')
        self.particle_crop_status = W.HTML(value='Select a particle to choose its crop.')
        particle_crop_editor = W.VBox([
            W.HTML('<p>Choose a particle in Preview particle, set its crop, then apply. '
                   'Other particles keep their own choice or the default. '
                   'Use Saved choices for a series with individual crops.</p>'),
            self.particle_crop_mode, self.particle_crop_height,
            self.particle_crop_width,
            W.HBox([self.apply_particle_crop, self.discard_particle_crop],
                   layout=W.Layout(flex_flow='row wrap')),
            self.particle_crop_status])
        self.particle_crop_settings = W.Accordion(
            children=[particle_crop_editor], selected_index=None)
        self.particle_crop_settings.set_title(0, 'Individual particle crop')

    def _particle_crop_values(self):
        mode = self.particle_crop_mode.value
        if mode == 'default':
            return None
        values = {'mode': mode}
        if mode == 'fixed':
            values.update(height_px=self.particle_crop_height.value,
                          width_px=self.particle_crop_width.value)
        return values

    def _saved_particle_crop(self):
        if not self.paths or self.crop_particle.value is None:
            return None
        path = self.paths[self.image_choice.value]
        return self.crop_overrides.get(path.name, {}).get(str(self.crop_particle.value))

    def _load_particle_crop_fields(self):
        settings = self._saved_particle_crop()
        self._syncing_particle_crop = True
        try:
            self.particle_crop_mode.value = settings['mode'] if settings else 'default'
            self.particle_crop_height.value = (
                settings.get('height_px', self.controls['crop_height_px'].value)
                if settings else self.controls['crop_height_px'].value)
            self.particle_crop_width.value = (
                settings.get('width_px', self.controls['crop_width_px'].value)
                if settings else self.controls['crop_width_px'].value)
        finally:
            self._syncing_particle_crop = False
        self._pending_crop = False
        if self.crop_particle.value is None or self._preview is None:
            message = 'Select a particle to choose its crop.'
        elif settings:
            message = f'Candidate {self.crop_particle.value} uses its own {settings["mode"]} crop.'
        else:
            message = f'Candidate {self.crop_particle.value} uses the default crop settings.'
        self.particle_crop_status.value = message
        self._particle_crop_availability()

    def _particle_crop_edited(self, change):
        if self._syncing_particle_crop or self._changing or self._busy:
            return
        self._pending_crop = self._particle_crop_values() != self._saved_particle_crop()
        self.particle_crop_status.value = (
            'Unapplied crop edits: click Apply particle crop or Discard crop edits.'
            if self._pending_crop else 'Crop fields match the saved choice.')
        self._availability()

    def _particle_crop_availability(self):
        unavailable = (not self.paths or self._preview is None
                       or self.crop_particle.value is None or self._busy
                       or self._analysis_running or self._pending_guess
                       or bool(self._settings_error))
        for control in (self.particle_crop_mode, self.particle_crop_height,
                        self.particle_crop_width):
            control.disabled = unavailable
        fixed = self.particle_crop_mode.value == 'fixed'
        for control in (self.particle_crop_height, self.particle_crop_width):
            control.layout.display = '' if fixed else 'none'
        self.apply_particle_crop.disabled = unavailable or not self._pending_crop
        self.discard_particle_crop.disabled = unavailable or not self._pending_crop

    def _apply_particle_crop(self):
        if self._pending_guess:
            raise ValueError('Apply x/y guess or Discard edits before changing the crop.')
        cfg = self._require_current_preview()
        number = self.crop_particle.value
        if number is None or number not in self.particle_choice.value:
            raise ValueError('Select a particle before changing its crop.')
        values = self._particle_crop_values()
        if values is not None:
            values = normalize_particle_crops({str(number): values})[str(number)]
        path = self.paths[self.image_choice.value]
        overrides = self.crop_overrides.setdefault(path.name, {})
        previous = overrides.get(str(number))
        # Prepare the proposed crop before committing it; invalid geometry must
        # leave the previously applied per-particle settings intact.
        if values is None:
            overrides.pop(str(number), None)
        else:
            overrides[str(number)] = values
        try:
            self._prepared_candidate(int(number), cfg)
        except Exception:
            if previous is None:
                overrides.pop(str(number), None)
            else:
                overrides[str(number)] = previous
            if not overrides:
                self.crop_overrides.pop(path.name, None)
            raise
        if not overrides:
            self.crop_overrides.pop(path.name, None)
        self._load_particle_crop_fields()
        self._show_candidate_crop(int(number), cfg)
        self.status.value = f'Crop applied to candidate {number} in {html.escape(path.name)}.'

    def _discard_particle_crop(self):
        if self._busy:
            return
        self._load_particle_crop_fields()
        if self._preview is not None and self.crop_particle.value is not None:
            self._perform(self._show_selected_crop, self.preview_output)
        else:
            self._availability()
