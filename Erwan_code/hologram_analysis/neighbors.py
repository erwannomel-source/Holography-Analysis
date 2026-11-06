"""Private local fringe checks that protect crops without adding map markers.

These image-pattern proposals are crop context, never fitted particle records.
Only clearly separated, coherent rings become ownership exclusions. Ambiguous
features retain a warning rather than an invented center that could cut data.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (gaussian_filter, gaussian_filter1d, map_coordinates,
                           maximum_filter)

from .cropping import bounded_window
from .detection import _fringe_support


def _polar_profile(values, center, maximum):
    """Measure signed radial contrast and angular coverage in local pixels."""
    radii = np.arange(int(maximum) + 1, dtype=float)
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rows = center[0] + radii[:, None] * np.cos(angles)
    cols = center[1] + radii[:, None] * np.sin(angles)
    valid = ((rows >= 0) & (rows <= values.shape[0] - 1) & (cols >= 0)
             & (cols <= values.shape[1] - 1))
    samples = map_coordinates(values, [rows, cols], order=1, mode='nearest')
    count = valid.sum(axis=1)
    means = (samples * valid).sum(axis=1) / np.maximum(count, 1)
    power = (samples**2 * valid).sum(axis=1) / np.maximum(count, 1)
    return means, power, count


def _distinct_rings(values, center, diameter, noise, minimum_transitions=2):
    """Require signed interference lobes, not just a spot or energy maximum."""
    means, power, counts = _polar_profile(values, center, diameter / 2)
    means = gaussian_filter1d(means, .65)
    cutoff = np.maximum(4 * noise / np.sqrt(np.maximum(counts, 1)),
                        .08 * np.max(np.abs(means)))
    coherent = means**2 / np.maximum(power, 1e-20) >= .3
    valid = (counts >= 24) & coherent & (np.abs(means) > cutoff)
    # Isolated extrema must span several pixels; camera noise cannot establish
    # an independent fringe sequence merely by alternating single-pixel signs.
    positive = negative = False
    runs = []
    current, length = 0, 0
    for sign in np.where(valid, np.sign(means), 0):
        if sign == current:
            length += 1
        else:
            if current and length >= 2:
                runs.append(int(current))
            current, length = sign, 1
    if current and length >= 2:
        runs.append(int(current))
    for sign in runs:
        positive |= sign > 0
        negative |= sign < 0
    transitions = sum(a != b for a, b in zip(runs, runs[1:]))
    return positive and negative and transitions >= minimum_transitions


def _known_context(known, center, separation):
    """Keep caller-owned centers, dropping the target and exact duplicates."""
    output = []
    for record in known:
        point = (int(record['x']), int(record['y']))
        if np.hypot(point[0] - center[0], point[1] - center[1]) <= separation:
            continue
        if any(
                np.hypot(point[0] - p['x'], point[1] - p['y']) <= separation
                for p in output):
            continue
        output.append(dict(record))
    return output


def find_local_neighbors(residual, center, cfg, known_neighbors=()):
    """Return private crop neighbors and uncertainty from a bounded local view.

    The returned dictionary contains ``neighbors`` (global x/y coordinates),
    ``search_bounds_px``, ``ambiguous`` and ``warning``. Known centers retain
    their metadata. New centers require coherent alternating interference rings
    and do not acquire public particle identifiers. Uncertain proposals are not
    returned as exclusions: callers retain the warning and require stronger
    two-sided fringe agreement before selecting a reduced fitting rectangle.

    This heuristic screens image patterns, not physical spheres. No HoloPy fit,
    whole-image filtering, target-center refinement or intensity change occurs.
    """
    image = np.asarray(residual)
    if image.ndim != 2:
        raise ValueError(
            'Local neighbor checks require a two-dimensional image.')
    center = tuple(int(v) for v in center)
    separation = max(4., min(12., cfg.detect_min_distance / 2))
    known = _known_context(known_neighbors, center, separation)
    budget = int(cfg.adaptive_crop_max_px)
    # Context padding observes rings centered near the fitting-window edge.
    # It is capped independently of the full-frame image dimensions.
    padding = min(48, max(24, int(cfg.adaptive_crop_margin_px) * 2))
    bounds = bounded_window(image.shape, center, (budget + 2 * padding, ) * 2)
    x0, x1, y0, y1 = bounds
    local = np.asarray(image[x0:x1, y0:y1], dtype=float)
    result = dict(neighbors=known,
                  search_bounds_px=bounds,
                  ambiguous=False,
                  warning='')
    if not np.isfinite(local).all():
        result.update(
            ambiguous=True,
            warning='Local neighbor context contains nonfinite pixels; '
            'particle isolation cannot be established.')
        return result
    highpass = local - gaussian_filter(local, 1.)
    noise = max(
        float(1.4826 * np.median(np.abs(highpass - np.median(highpass))) /
              .87), 1e-10)
    local_center = (center[0] - x0, center[1] - y0)
    rows, cols = np.indices(local.shape)
    distance = np.hypot(rows - local_center[0], cols - local_center[1])
    profile, _, _ = _polar_profile(local, local_center,
                                   np.ceil(distance.max()))
    profile = gaussian_filter1d(profile, .65)
    # Suppress the chosen hologram's concentric rings before proposing centers;
    # otherwise its own bright fringes produce many false neighboring particles.
    remainder = local - np.interp(distance, np.arange(len(profile)), profile)
    energy = remainder**2
    scales = (max(3., budget / 64), max(6.,
                                        budget / 24), max(12., budget / 10))
    activity = np.zeros(local.shape)
    for sigma in scales:
        response = (gaussian_filter(energy, sigma) -
                    gaussian_filter(energy, sigma * 1.8))
        activity = np.maximum(activity, response)
    middle = float(np.median(activity))
    spread = float(1.4826 * np.median(np.abs(activity - middle)))
    threshold = max(middle + 8 * spread, noise**2 * .7)
    size = max(9, int(cfg.detect_min_distance / 2) * 2 + 1)
    maxima = (activity == maximum_filter(activity, size=size))
    maxima &= activity > threshold
    maxima &= distance > max(8., separation)
    # A neighbor outside this circle cannot enter the bounded fitting window.
    maxima &= distance <= np.sqrt(2) * budget / 2 + padding
    points = sorted(np.argwhere(maxima),
                    key=lambda rc: activity[tuple(rc)],
                    reverse=True)[:18]
    accepted = []
    ambiguous = False
    # Imported lazily to avoid the preparation/cropping import cycle.
    from .preparation import refine_ring_center
    for row, col in points:
        point = (int(row), int(col))
        refined = refine_ring_center(local, *point, radius=6)
        diameter, symmetry = _fringe_support(local, point, budget, noise)
        if refined != point:
            other_diameter, other_symmetry = _fringe_support(
                local, refined, budget, noise)
            if other_symmetry > symmetry + 1e-3:
                point, diameter, symmetry = refined, other_diameter, other_symmetry
        global_point = (point[0] + x0, point[1] + y0)
        if np.hypot(global_point[0] - center[0],
                    global_point[1] - center[1]) <= separation:
            continue
        if any(
                np.hypot(global_point[0] - p['x'], global_point[1] -
                         p['y']) <= separation for p in known + accepted):
            continue
        available_radius = min(point[0], point[1],
                               local.shape[0] - 1 - point[0],
                               local.shape[1] - 1 - point[1])
        incomplete = (diameter is None or available_radius < diameter / 2 + 6)
        if diameter is None or diameter < 16 or symmetry < .4:
            # A boundary can displace an energy peak where the existing center
            # refiner cannot operate. Weaker symmetry is sufficient only for
            # uncertainty, and still requires measurable signed fringe lobes.
            # A compact bright spot is neither a neighbor nor an overlap warning.
            possible = (diameter is None or diameter
                        >= 16) and (symmetry >= .25 or
                                    (incomplete and symmetry >= .15))
            if possible:
                ambiguous |= _distinct_rings(local,
                                             point,
                                             diameter or budget,
                                             noise,
                                             minimum_transitions=1)
            continue
        if not _distinct_rings(local, point, diameter, noise):
            if incomplete:
                ambiguous |= _distinct_rings(local,
                                             point,
                                             diameter,
                                             noise,
                                             minimum_transitions=1)
            continue
        accepted.append(
            dict(x=int(global_point[0]),
                 y=int(global_point[1]),
                 strength=float(activity[row, col]),
                 hologram_diameter_px=float(diameter),
                 private_neighbor=True))
    result['neighbors'] = known + accepted
    result['ambiguous'] = bool(ambiguous)
    if ambiguous:
        result['warning'] = (
            'Local patterns may overlap or lack a reliable '
            'neighbor center; particle isolation is uncertain.')
    return result
