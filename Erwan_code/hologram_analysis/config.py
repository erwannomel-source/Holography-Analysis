"""Configuration, units, validation and cooperative cancellation."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from holopy.scattering.theory import MieLens

MIN_CROP_PX = 4


class AnalysisCancelled(KeyboardInterrupt):
    """Stop a run without treating cancellation as a failed fit.

    Inheriting from KeyboardInterrupt prevents generic optimizer error handlers
    from swallowing the stop request. Trials completed before cancellation are
    attached so the image runner can still save them.
    """

    def __init__(self, trials=None):
        super().__init__('Analysis stopped by the user.')
        self.trials = [] if trials is None else trials


def check_cancelled(cancel_event):
    """Raise at safe boundaries when either Stop button has been clicked."""
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled()


@dataclass(frozen=True)
class AnalysisConfig:
    """Physical and numerical settings, independent of the notebook interface."""
    spacing: float = 0.060185185
    medium_index: float = 1.4
    illum_wavelen: float = 0.660
    illum_polarization: tuple = (1.0, 0.0)
    NA: float = 1.2
    noil: float = 1.331
    target_x_px: int = 520
    target_y_px: int = 520
    search_radius_px: float = 400.0
    click_to_detect: bool = False
    detect_search_circle_only: bool = False
    detect_sigma: float = 3.0
    detect_size_filter: bool = False
    detect_min_diameter_px: int = 40
    detect_max_diameter_px: int = 240
    detect_min_distance: int = 20
    detect_border: int = 35
    max_particles: int = 15
    refine_center: bool = True
    crop_height_px: int = 300
    crop_width_px: int = 300
    adaptive_crop: bool = False
    adaptive_crop_rectangle: bool = True
    adaptive_crop_min_px: int = 32
    adaptive_crop_max_px: int = 256
    adaptive_crop_margin_px: int = 6
    xy_fit_margin_um: float = 3.0
    n_guess: float = 1.33
    r_guesses: tuple = (2.0,)
    n_bounds: tuple = (1.2, 1.8)
    r_bounds: tuple = (0.3, 20.0)
    z_values: tuple = (5.0, 15.0, 25.0, 40.0)
    z_bounds: tuple = (0.0, 60.0)
    auto_initial_guess: bool = False
    initial_search_trials: int = 5
    initial_search_max_pixels: int = 3600
    search_effort: str = 'custom'
    adaptive_fit_budget: bool = True
    reuse_fit_cache: bool = True
    reuse_experiment_guesses: bool = True
    persist_experiment_guesses: bool = True
    experiment_name: str = ''
    show_detection_plots: bool = True
    show_crop_plots: bool = True
    show_fit_plots: bool = True
    save_figures: bool = True
    max_displayed_figures: int = 6

    def validate(self) -> AnalysisConfig:
        for label, value in [('Calibration', self.spacing),
                             ('Medium refractive index', self.medium_index),
                             ('Wavelength', self.illum_wavelen),
                             ('NA', self.NA),
                             ('Objective refractive index', self.noil),
                             ('Search radius', self.search_radius_px),
                             ('Smoothing', self.detect_sigma),
                             ('X/Y fit margin', self.xy_fit_margin_um)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f'{label} : enter a finite, strictly positive number.')
        if self.NA > self.noil:
            raise ValueError(
                'NA must not exceed the objective refractive index.')
        if self.search_effort not in ('fast', 'balanced', 'thorough', 'custom'):
            raise ValueError('Search effort must be Fast, Balanced, Thorough or Custom.')
        for label, value, minimum, maximum in [
                ('Initial search finalists', self.initial_search_trials, 1, 20),
                ('Initial search pixels', self.initial_search_max_pixels, 16, 100000)]:
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or not minimum <= value <= maximum):
                raise ValueError(
                    f'{label}: enter an integer between {minimum} and {maximum}.')
        for label, value in [('Minimum separation', self.detect_min_distance),
                             ('Excluded border', self.detect_border),
                             ('Maximum candidates', self.max_particles),
                             ('Maximum displayed figures',
                              self.max_displayed_figures)]:
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f'{label} : enter a positive integer.')
        for label, value in [('Adaptive minimum', self.adaptive_crop_min_px),
                             ('Adaptive maximum', self.adaptive_crop_max_px),
                             ('Fringe margin', self.adaptive_crop_margin_px)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)) or value < 1:
                raise ValueError(f'{label}: enter a positive integer.')
        if (self.adaptive_crop_min_px < MIN_CROP_PX
                or self.adaptive_crop_min_px % 2
                or self.adaptive_crop_max_px % 2
                or self.adaptive_crop_max_px < self.adaptive_crop_min_px):
            raise ValueError(
                f'Adaptive crop limits must be even, at least {MIN_CROP_PX} pixels, '
                'and maximum must not be smaller than minimum.')
        for name in ('adaptive_crop', 'adaptive_crop_rectangle',
                     'click_to_detect', 'detect_search_circle_only',
                     'detect_size_filter', 'refine_center',
                     'show_detection_plots', 'show_crop_plots',
                     'show_fit_plots', 'save_figures', 'auto_initial_guess',
                     'adaptive_fit_budget', 'reuse_fit_cache',
                     'reuse_experiment_guesses', 'persist_experiment_guesses'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name}: enter a boolean value.')
        if (not isinstance(self.experiment_name, str)
                or len(self.experiment_name.strip()) > 160
                or any(not char.isprintable() for char in self.experiment_name)):
            raise ValueError(
                'Experiment name: enter printable text of at most 160 characters.')
        if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 16
                for value in (self.detect_min_diameter_px,
                              self.detect_max_diameter_px)):
            raise ValueError(
                'Hologram diameter limits must be integers of at least 16 pixels.'
            )
        if self.detect_max_diameter_px < self.detect_min_diameter_px:
            raise ValueError(
                'Maximum hologram diameter must not be smaller than minimum.')
        if not self.adaptive_crop and any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < MIN_CROP_PX or value % 2
                for value in (self.crop_height_px, self.crop_width_px)):
            raise ValueError(
                f'Crop height and width must be even and at least {MIN_CROP_PX} pixels.'
            )
        if not np.isfinite([self.target_x_px, self.target_y_px]).all():
            raise ValueError('Target coordinates must be finite.')
        if (len(self.illum_polarization) != 2
                or not np.isfinite(self.illum_polarization).all()
                or np.linalg.norm(self.illum_polarization) == 0):
            raise ValueError(
                'Polarization must contain two finite numbers, with at least one nonzero value.'
            )
        if not self.z_values:
            raise ValueError('Enter at least one finite initial z guess.')
        if not self.r_guesses:
            raise ValueError(
                'Enter at least one positive initial radius guess.')
        for label, bounds, guesses in [
            ('Particle refractive index', self.n_bounds, [self.n_guess]),
            ('Radius', self.r_bounds, self.r_guesses),
            ('Z depth', self.z_bounds, self.z_values)
        ]:
            if (len(bounds) != 2 or not np.isfinite(bounds).all()
                    or bounds[0] >= bounds[1]
                    or any(not np.isfinite(g) or not bounds[0] < g < bounds[1]
                           for g in guesses)):
                raise ValueError(
                    f'{label} : initial values must lie strictly within the bounds.'
                )
        if self.r_bounds[0] <= 0 or self.n_bounds[0] <= 0:
            raise ValueError(
                'Radius and refractive-index bounds must be positive.')
        # The name is a human-readable label, never a path or file name.
        return replace(self, experiment_name=self.experiment_name.strip())

    @property
    def r_guess(self):
        """First radius start, retained for existing initial-hologram previews."""
        return self.r_guesses[0]

    def theory(self):
        return MieLens(np.arcsin(self.NA / self.noil))

    def crop_key(self):
        """All settings that change a prepared crop, including adaptive geometry."""
        fixed_dimensions = ((None, None) if self.adaptive_crop else
                            (self.crop_height_px, self.crop_width_px))
        return (*fixed_dimensions, self.refine_center, self.xy_fit_margin_um,
                self.adaptive_crop,
                self.adaptive_crop_rectangle if self.adaptive_crop else None,
                self.adaptive_crop_min_px, self.adaptive_crop_max_px,
                self.adaptive_crop_margin_px)

    def centered_for_image(self, shape) -> AnalysisConfig:
        """Use the geometric image center in HoloPy row/column pixels."""
        if len(shape) != 2 or any(size < 1 for size in shape):
            raise ValueError('Expected a nonempty two-dimensional image.')
        return replace(self, target_x_px=(shape[0] - 1) / 2,
                       target_y_px=(shape[1] - 1) / 2)

    def detection_key(self) -> tuple:
        """Candidate numbers become invalid when detection settings change."""
        circle_scope = self.detect_search_circle_only and not self.click_to_detect
        size_filter = self.detect_size_filter and not self.click_to_detect
        return (self.spacing, self.medium_index, self.illum_wavelen,
                tuple(self.illum_polarization), self.detect_sigma,
                self.detect_min_distance, self.detect_border,
                self.max_particles, self.click_to_detect, circle_scope,
                self.search_radius_px if circle_scope else None, size_filter,
                self.detect_min_diameter_px if size_filter else None,
                self.detect_max_diameter_px if size_filter else None)

    @classmethod
    def from_dict(cls, values: dict) -> AnalysisConfig:
        values = dict(values)
        if 'search_effort' not in values:
            pair = (values.get('initial_search_trials', 5),
                    values.get('initial_search_max_pixels', 3600))
            values['search_effort'] = {
                (3, 1600): 'fast', (5, 3600): 'balanced',
                (10, 3600): 'thorough',
            }.get(pair, 'custom')
        values.setdefault('adaptive_fit_budget', values['search_effort'] != 'thorough')
        # Ignore retired cross-crop fit checks in existing profiles/settings.
        # They must never re-enable reference fitting or reject old profiles.
        for key in ('adaptive_crop_validate', 'adaptive_crop_rtol',
                    'adaptive_crop_z_atol_um', 'adaptive_crop_n_atol'):
            values.pop(key, None)
        # Old profiles contain a single radius. Apply the new two-start default
        # while retaining all their other settings, including explicit crops.
        values.pop('r_guess', None)
        if 'crop_size_px' in values:
            legacy_size = values.pop('crop_size_px')
            values.setdefault('crop_height_px', legacy_size)
            values.setdefault('crop_width_px', legacy_size)
        fields = cls.__dataclass_fields__
        if set(values) - set(fields):
            raise ValueError('The profile contains unknown settings.')
        for key in [
                'illum_polarization', 'n_bounds', 'r_bounds', 'r_guesses',
                'z_values', 'z_bounds'
        ]:
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values).validate()


def parse_numbers(text: str) -> tuple[float, ...]:
    """Parse numbers separated by commas or semicolons."""
    parts = str(text).replace(';', ',').split(',')
    if not parts or any(not p.strip() for p in parts):
        raise ValueError(
            'Enter comma-separated numbers, for example 5, 15, 25, 40.')
    try:
        values = tuple(float(p.strip()) for p in parts)
    except ValueError as exc:
        raise ValueError('Use a decimal point, for example 1.45.') from exc
    if not np.isfinite(values).all():
        raise ValueError('Numbers must be finite.')
    return values
