"""Optional size-guided hologram detection using image processing only."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates


def _fringe_support(residual, center, maximum, noise):
    """Estimate visible, angularly coherent support around a proposed center.

    The value is an image-pattern diameter, not the physical sphere diameter.
    A noise floor and relative contrast floor prevent a compact bright speck's
    weak background-subtraction halo from masquerading as a large hologram.
    Partial rings need at least twelve observed angular samples. A truncated
    support cannot establish that the pattern is below the requested maximum.
    """
    radii = np.arange(1., np.ceil(maximum * .65) + 2)
    angles = np.linspace(0., 2 * np.pi, 48, endpoint=False)
    rows = center[0] + radii[:, None] * np.cos(angles)
    cols = center[1] + radii[:, None] * np.sin(angles)
    valid = ((rows >= 0) & (rows <= residual.shape[0] - 1) & (cols >= 0) &
             (cols <= residual.shape[1] - 1))
    counts = valid.sum(axis=1)
    values = map_coordinates(residual, [rows, cols], order=1, mode='nearest')
    means = np.sum(values * valid, axis=1) / np.maximum(counts, 1)
    power = np.sum(values**2 * valid, axis=1) / np.maximum(counts, 1)
    observed = counts >= 12
    if not observed.any():
        return None, 0.0
    # Compare centers using all observed annuli, subtracting the estimated
    # camera-noise contribution. This avoids moving a hollow pattern toward
    # a random symmetry maximum in its almost empty inner region.
    coherent_energy = np.maximum(means**2 - noise**2 / np.maximum(counts, 1),
                                 0)
    signal_energy = np.maximum(power - noise**2, 0)
    numerator = float(np.sum(coherent_energy[observed] * counts[observed]))
    denominator = float(np.sum(signal_energy[observed] * counts[observed]))
    symmetry = numerator / max(denominator, 1e-20)
    peak = float(np.max(np.abs(means[observed])))
    floor = np.maximum(3 * noise / np.sqrt(np.maximum(counts, 1)), .05 * peak)
    coherence = means**2 / np.maximum(power, 1e-20)
    visible = observed & (np.abs(means) > floor) & (coherence >= .25)
    indices = np.flatnonzero(visible)
    if len(indices) < 3:
        return None, symmetry
    last = int(indices[-1])
    if last >= len(radii) - 2:
        return None, symmetry
    return float(2 * (radii[last] + .5)), symmetry


def detect_sized_candidates(residual,
                            sigma,
                            min_distance,
                            border,
                            limit,
                            minimum,
                            maximum,
                            center_refiner=None,
                            candidate_mask=None):
    """Find energy centers at up to three scales and filter their visible size.

    Size screening happens before the candidate-count limit. Thus rejected small
    features do not consume the slots intended for larger holograms. The method
    refines an energy peak only when full-pattern radial symmetry improves.
    It assumes approximately circular interference patterns; it is a configurable
    screening heuristic, not particle identification or a calibrated size fit.
    """
    if (isinstance(minimum,
                   (bool, np.bool_)) or isinstance(maximum, (bool, np.bool_))
            or not isinstance(minimum, (int, np.integer))
            or not isinstance(maximum, (int, np.integer)) or minimum < 16
            or maximum < minimum):
        raise ValueError(
            'Hologram diameter limits must be integers: '
            'minimum at least 16 pixels, maximum at least minimum.')
    if maximum > 2 * max(residual.shape):
        raise ValueError(
            'Maximum hologram diameter exceeds twice the image size; '
            'choose a size visible in this image.')
    energy = residual**2
    scales = np.geomspace(minimum, maximum, 3 if maximum > minimum else 1)
    activity = np.zeros_like(residual, dtype=float)
    for diameter in scales:
        width = max(float(sigma), float(diameter) / 3.)
        response = (gaussian_filter(energy, width) -
                    gaussian_filter(energy, width * 1.8))
        activity = np.maximum(activity, response)
    middle = np.median(activity)
    mad = np.median(np.abs(activity - middle))
    threshold = middle + 8 * 1.4826 * mad
    separation = max(min_distance, minimum // 4)
    maxima = activity == maximum_filter(activity, size=2 * separation + 1)
    maxima &= activity > threshold
    if candidate_mask is not None:
        maxima &= candidate_mask
    if border > 0:
        maxima[:border, :] = False
        maxima[-border:, :] = False
        maxima[:, :border] = False
        maxima[:, -border:] = False
    coords = sorted(np.argwhere(maxima),
                    key=lambda rc: activity[tuple(rc)],
                    reverse=True)
    highpass = residual - gaussian_filter(residual, 1.)
    noise = max(
        float(1.4826 * np.median(np.abs(highpass - np.median(highpass))) /
              .87), 1e-10)
    candidates = []
    for row, col in coords:
        center = (int(row), int(col))
        diameter, symmetry = _fringe_support(residual, center, maximum, noise)
        if center_refiner is not None:
            refined = tuple(center_refiner(residual, *center))
            if refined != center:
                refined_diameter, refined_symmetry = _fringe_support(
                    residual, refined, maximum, noise)
                if refined_symmetry > symmetry + 1e-3:
                    center, diameter = refined, refined_diameter
        if not (border <= center[0] < residual.shape[0] - border
                and border <= center[1] < residual.shape[1] - border):
            continue
        if candidate_mask is not None and not candidate_mask[center[0],
                                                             center[1]]:
            continue
        if any(
                np.hypot(center[0] - c['x'], center[1] - c['y']) < min_distance
                for c in candidates):
            continue
        if diameter is None or not minimum <= diameter <= maximum:
            continue
        candidates.append({
            'x': int(center[0]),
            'y': int(center[1]),
            'strength': float(activity[row, col]),
            'hologram_diameter_px': diameter
        })
        if len(candidates) >= limit:
            break
    return activity, candidates
