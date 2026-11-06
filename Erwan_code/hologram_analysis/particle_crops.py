"""Validated crop choices attached to one image and stable particle identifier."""
from __future__ import annotations

from dataclasses import replace
from numbers import Integral

from .config import MIN_CROP_PX


def normalize_particle_crops(values):
    """Normalize a candidate-number -> crop-settings map without changing it."""
    if not isinstance(values, dict):
        raise ValueError('Particle crops must be a dictionary of candidate numbers.')
    normalized = {}
    for number, settings in values.items():
        if (isinstance(number, bool) or not isinstance(number, (str, Integral))
                or not str(number).isdecimal() or int(number) < 1):
            raise ValueError('Particle crop numbers must be positive integers.')
        key = str(int(number))
        if key in normalized:
            raise ValueError('Duplicate particle crop number.')
        if not isinstance(settings, dict):
            raise ValueError('Each particle crop must contain a crop mode.')
        mode = settings.get('mode')
        if mode not in ('fixed', 'automatic', 'rectangle'):
            raise ValueError('Particle crop mode must be Fixed or Automatic, with optional rectangle reduction.')
        fields = {'mode', 'height_px', 'width_px'} if mode == 'fixed' else {'mode'}
        if set(settings) != fields:
            raise ValueError('Fixed particle crops require height_px and width_px; automatic crops require only a mode.')
        record = {'mode': mode}
        if mode == 'fixed':
            for name in ('height_px', 'width_px'):
                value = settings[name]
                if (isinstance(value, bool) or not isinstance(value, Integral)
                        or value < MIN_CROP_PX or value % 2):
                    raise ValueError(f'Particle crop dimensions must be even integers of at least {MIN_CROP_PX} pixels.')
                record[name] = int(value)
        normalized[key] = record
    return normalized


def particle_crop_config(cfg, settings=None):
    """Derive one particle's crop settings, preserving the shared configuration."""
    if settings is None:
        return cfg
    settings = normalize_particle_crops({'1': settings})['1']
    mode = settings['mode']
    changes = dict(adaptive_crop=mode != 'fixed',
                   adaptive_crop_rectangle=mode == 'rectangle')
    if mode == 'fixed':
        changes.update(crop_height_px=settings['height_px'],
                       crop_width_px=settings['width_px'])
    return replace(cfg, **changes).validate()
