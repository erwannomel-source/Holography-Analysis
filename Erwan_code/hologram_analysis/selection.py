"""Local click localization and optional circle-scoped automatic detection."""
from __future__ import annotations

from numbers import Real
import math
import numpy as np
from scipy.ndimage import gaussian_filter

from .cropping import bounded_window
from .detection import _fringe_support


def validate_clicked_particles(records, shape=None):
    """Normalize saved centers before using their stable, one-based identifiers."""
    if not isinstance(records, list):
        raise ValueError('Clicked particles must be a list of center records.')
    normalized = []
    for record in records:
        if not isinstance(record, dict) or not {'x', 'y'} <= record.keys():
            raise ValueError(
                'Each clicked particle must contain x/y pixel coordinates.')
        center = {}
        for axis in ('x', 'y'):
            value = record[axis]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0
                    or value != int(value)):
                raise ValueError(
                    'Clicked centers must be finite nonnegative integer pixels.'
                )
            center[axis] = int(value)
        if shape is not None and not (0 < center['x'] < shape[0] - 1
                                      and 0 < center['y'] < shape[1] - 1):
            raise ValueError(
                'A clicked particle center is outside the usable image.')
        center['strength'] = 0.0
        for name in ('strength', 'hologram_diameter_px'):
            if name in record:
                value = record[name]
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or value < 0):
                    raise ValueError(
                        'Invalid clicked-particle detection metadata.')
                center[name] = float(value)
        center['local_detection'] = True
        normalized.append(center)
    return normalized


def localize_particle(residual, row, col, cfg, neighbors=(), *, measure_crop=True):
    """Use a user-supplied seed, refine its center and measure its local crop.

    No full-image candidate search or HoloPy fit is performed. Automatic size
    limits do not reject an explicitly clicked particle. Uncertain fringes keep
    the clicked center and conservative crop, with a visible warning.
    Set ``measure_crop=False`` when the caller prepares the selected crop next;
    center confidence, candidate metadata and activity remain available without
    measuring a crop that would be discarded.
    """
    from .click_centering import refine_clicked_center
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or not np.isfinite(residual).all():
        raise ValueError('Click localization requires a finite two-dimensional image.')
    row, col = int(row), int(col)
    if not (0 < row < residual.shape[0] - 1
            and 0 < col < residual.shape[1] - 1):
        raise ValueError(
            'Click inside the image, away from its outermost pixels.')
    localization = dict(seed_px=(row, col), center_px=(row, col),
                        confident=None, reason='Automatic centering is disabled.')
    if cfg.refine_center:
        (row, col), localization = refine_clicked_center(
            residual, row, col, maximum=cfg.adaptive_crop_max_px)
    bounds = bounded_window(residual.shape, (row, col),
                            (cfg.adaptive_crop_max_px, ) * 2)
    x0, x1, y0, y1 = bounds
    local = residual[x0:x1, y0:y1]
    highpass = local - gaussian_filter(local, 1.)
    noise = max(
        float(1.4826 * np.median(np.abs(highpass - np.median(highpass))) /
              .87), 1e-10)
    diameter, _ = _fringe_support(residual, (row, col),
                                      cfg.adaptive_crop_max_px, noise)
    info = {}
    if measure_crop:
        from .preparation import particle_crop_info
        # A reselected saved point is not its own neighboring particle.
        snap = max(4, min(12, cfg.detect_min_distance / 2))
        neighbors = [
            n for n in neighbors if np.hypot(n['x'] - row, n['y'] - col) > snap
        ]
        info = particle_crop_info(residual, (row, col),
                                  cfg,
                                  neighbors,
                                  local_detection=True)
    info['center_confident'] = localization['confident']
    info['center_localization'] = localization
    if localization['confident'] is False:
        info['warning'] = ' '.join(filter(None, (
            info.get('warning'), 'Center localization uncertain: '
            + localization['reason'] + ' Click closer to the particle center.')))
    activity = np.zeros_like(residual)
    activity[x0:x1, y0:y1] = gaussian_filter(local**2, cfg.detect_sigma)
    candidate = dict(x=int(row),
                     y=int(col),
                     strength=float(np.mean(local**2)),
                     local_detection=True)
    if diameter is not None:
        candidate['hologram_diameter_px'] = float(diameter)
    return candidate, info, activity


def clicked_activity(residual, records, cfg):
    """Render energy only around explicitly chosen centers, without redetection."""
    activity = np.zeros_like(residual)
    for candidate in records:
        x0, x1, y0, y1 = bounded_window(residual.shape,
                                        (candidate['x'], candidate['y']),
                                        (cfg.adaptive_crop_max_px, ) * 2)
        local = gaussian_filter(residual[x0:x1, y0:y1]**2, cfg.detect_sigma)
        activity[x0:x1, y0:y1] = np.maximum(activity[x0:x1, y0:y1], local)
    return activity


def detect_in_circle(residual, cfg):
    """Detect only centers inside the target circle, retaining filter context.

    Crop the processing window when a padded circle is smaller than the image.
    Large kernels/circles may require the full image; the center mask still
    avoids screening or ranking candidates outside the circle.
    """
    from .preparation import _detect_residual
    height, width = residual.shape
    cfg = cfg.centered_for_image(residual.shape)
    largest = (max(cfg.detect_sigma, cfg.detect_max_diameter_px / 3) *
               1.8 if cfg.detect_size_filter else cfg.detect_sigma)
    padding = int(np.ceil(4 * largest)) + cfg.detect_min_distance + 12
    radius = cfg.search_radius_px
    x0 = max(0, int(np.floor(cfg.target_x_px - radius - padding)))
    x1 = min(height, int(np.ceil(cfg.target_x_px + radius + padding + 1)))
    y0 = max(0, int(np.floor(cfg.target_y_px - radius - padding)))
    y1 = min(width, int(np.ceil(cfg.target_y_px + radius + padding + 1)))
    full_activity = np.zeros_like(residual)
    if x0 >= x1 or y0 >= y1:
        return full_activity, []
    rows, cols = np.indices((x1 - x0, y1 - y0))
    rows += x0
    cols += y0
    mask = ((rows - cfg.target_x_px)**2 + (cols - cfg.target_y_px)**2
            <= radius**2)
    mask &= (rows >= cfg.detect_border) & (rows < height - cfg.detect_border)
    mask &= (cols >= cfg.detect_border) & (cols < width - cfg.detect_border)
    if not mask.any():
        return full_activity, []
    activity, candidates = _detect_residual(
        residual[x0:x1, y0:y1],
        cfg.detect_sigma,
        cfg.detect_min_distance,
        0,
        cfg.max_particles,
        cfg.detect_min_diameter_px if cfg.detect_size_filter else 0,
        cfg.detect_max_diameter_px if cfg.detect_size_filter else 0,
        candidate_mask=mask)
    full_activity[x0:x1, y0:y1] = activity
    for c in candidates:
        c.update(x=c['x'] + x0, y=c['y'] + y0)
    return full_activity, candidates
