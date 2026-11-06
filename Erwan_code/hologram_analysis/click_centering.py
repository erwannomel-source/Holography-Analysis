"""Bounded image-based localization of the concentric fringes near a click.

This is a geometric aid to selection, not an optical fit. All interpolation and
smoothing are private measurements; the data used for fitting remain untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

from .cropping import bounded_window


def _ring_scores(image, centers, noise, radii):
    """Score radial variation versus angular disagreement, excluding image edges.

    Subtracting each candidate's radial mean prevents a nearly constant bright
    patch from scoring as a circular hologram. Angular outliers (a speck or a
    neighbor crossing a small arc) are limited using a robust within-ring scale.
    Several radial extents let a compact hologram survive unrelated outer rings.
    """
    angles = np.linspace(0., 2 * np.pi, 64, endpoint=False)
    rows = centers[:, 0, None, None] + radii[None, :, None] * np.cos(angles)
    cols = centers[:, 1, None, None] + radii[None, :, None] * np.sin(angles)
    valid = ((rows >= 0) & (rows <= image.shape[0] - 1) & (cols >= 0)
             & (cols <= image.shape[1] - 1))
    samples = map_coordinates(image, [rows, cols], order=1, mode='nearest')
    counts = valid.sum(axis=2)
    samples[~valid] = np.nan
    # Avoid empty-slice warnings on tiny frames; these annuli stay masked out.
    samples[counts == 0] = 0.
    # A local candidate can be near the frame edge. Annuli with less than a
    # quarter-circle of real pixels cannot establish its symmetry reliably.
    observed = counts >= 24
    with np.errstate(invalid='ignore'):
        medians = np.nanmedian(samples, axis=2)
        deviation = np.nanmedian(np.abs(samples - medians[..., None]), axis=2)
    limit = np.maximum(4 * 1.4826 * deviation, 3 * noise)
    clipped = np.clip(samples, medians[..., None] - limit[..., None],
                      medians[..., None] + limit[..., None])
    clipped = np.where(valid, clipped, 0.)
    means = clipped.sum(axis=2) / np.maximum(counts, 1)
    angular_power = (clipped**2).sum(axis=2) / np.maximum(counts, 1)
    angular_variance = np.maximum(angular_power - means**2, 0.)
    best = np.zeros(len(centers))
    extent = np.zeros(len(centers), dtype=int)
    # The large support is important for hollow/large holograms: a short-radius
    # search can otherwise settle on a bright fringe rather than its center.
    supports = sorted(set(min(float(radii[-1]), v) for v in (32., 56., 80.)))
    for maximum in supports:
        mask = observed & (radii[None, :] <= maximum)
        weights = counts * mask
        total = np.maximum(weights.sum(axis=1), 1)
        offset = (means * weights).sum(axis=1) / total
        radial_variance = (((means - offset[:, None])**2 * weights).sum(
            axis=1) / total)
        disagreement = (angular_variance * weights).sum(axis=1) / total
        coherence = radial_variance / np.maximum(
            radial_variance + disagreement + noise**2, 1e-20)
        usable = (mask.sum(axis=1) >= 8) & (
            radial_variance > 4 * noise**2 / 24)
        # A coherent spot is not evidence of interference rings. Check the
        # alternating extrema only for promising candidates, after scoring.
        for i in np.flatnonzero(usable & (coherence > best) & (coherence >= .28)):
            profile = gaussian_filter1d(means[i, mask[i]], .65)
            span = float(np.ptp(profile))
            prominence = max(4 * noise / np.sqrt(24), .12 * span)
            peaks = find_peaks(profile, prominence=prominence, distance=2)[0]
            troughs = find_peaks(-profile, prominence=prominence, distance=2)[0]
            extrema = np.concatenate((peaks, troughs))
            # The particle's core is an extremum at the inner endpoint and is
            # therefore absent from find_peaks. A clear core plus a ring of the
            # opposite sign also establishes an alternating radial pattern.
            # Do not reject compact holograms whose outer rings have faded.
            core = profile[0]
            opposite_ring = np.any(
                (profile[extrema] * core < 0)
                & (np.abs(profile[extrema])
                   > max(4 * noise / np.sqrt(24), .12 * abs(core))))
            core_and_ring = (abs(core) > prominence and opposite_ring)
            if len(extrema) >= 2 or core_and_ring:
                best[i] = coherence[i]
                extent[i] = int(maximum)
    return best, extent


def refine_clicked_center(residual, row, col, maximum=256):
    """Return a nearby ring center and explicit confidence metadata.

    Search at most 40 pixels from the user's click, with coarse candidates then
    integer-pixel polishing of separated maxima. A candidate must exhibit both
    coherent angular symmetry and alternating radial fringes. Among comparable
    centers prefer the one nearest the click, avoiding jumps to a stronger
    neighboring particle. Uncertain or boundary-only evidence retains the seed.
    """
    image = np.asarray(residual, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Click localization requires a finite two-dimensional image.')
    seed = np.asarray((int(row), int(col)))
    radius = min(40, max(24, int(maximum) // 4))
    context = int(2 * (radius + min(80, max(32, int(maximum) // 2))))
    x0, x1, y0, y1 = bounded_window(image.shape, seed, (context, context))
    local = image[x0:x1, y0:y1]
    highpass = local - gaussian_filter(local, 1.)
    noise = max(float(1.4826 * np.median(np.abs(
        highpass - np.median(highpass))) / .87), 1e-10)
    # Suppress pixel-scale noise for geometric measurements only.
    smooth = gaussian_filter(local, .8)
    maximum_radius = min(80, max(32, int(maximum) // 2))
    radii = np.arange(3., maximum_radius + 1, 2.)
    metadata = dict(seed_px=tuple(int(v) for v in seed),
                    center_px=tuple(int(v) for v in seed),
                    search_radius_px=radius, score=0., noise=noise,
                    confident=False, reason='No clear concentric fringes near the click.')

    def allowed(points):
        return ((points[:, 0] > 0) & (points[:, 0] < image.shape[0] - 1)
                & (points[:, 1] > 0) & (points[:, 1] < image.shape[1] - 1)
                & (np.linalg.norm(points - seed, axis=1) <= radius))

    shifts = np.arange(-radius, radius + 1, 4)
    dr, dc = np.meshgrid(shifts, shifts, indexing='ij')
    coarse = seed + np.column_stack((dr.ravel(), dc.ravel()))
    coarse = coarse[allowed(coarse)]
    scores, _ = _ring_scores(smooth, coarse - (x0, y0), noise, radii)
    # Retain separated basins so an overlapping stronger pattern cannot prevent
    # the intended, slightly weaker particle from being polished.
    starts = []
    for i in np.argsort(-scores, kind='stable'):
        if scores[i] <= 0:
            break
        if all(np.linalg.norm(coarse[i] - p) > 8 for p in starts):
            starts.append(coarse[i])
        if len(starts) >= 8:
            break
    if not starts:
        return tuple(seed), metadata
    offsets = np.arange(-4, 5)
    dr, dc = np.meshgrid(offsets, offsets, indexing='ij')
    offsets = np.column_stack((dr.ravel(), dc.ravel()))
    fine = np.unique(np.concatenate([p + offsets for p in starts]), axis=0)
    fine = fine[allowed(fine)]
    scores, extents = _ring_scores(smooth, fine - (x0, y0), noise, radii)
    peaks = []
    for i in np.argsort(-scores, kind='stable'):
        if scores[i] < max(.28, .60 * float(scores.max())):
            break
        nearby = np.max(np.abs(fine - fine[i]), axis=1) <= 3
        if np.any(scores[nearby] > scores[i] + 1e-12):
            continue
        if all(np.linalg.norm(fine[i] - fine[j]) > 8 for j in peaks):
            peaks.append(int(i))
    if not peaks:
        return tuple(seed), metadata
    chosen = min(peaks, key=lambda i: (np.linalg.norm(fine[i] - seed), -scores[i]))
    center = fine[chosen]
    metadata.update(score=float(scores[chosen]), radial_extent_px=int(extents[chosen]))
    if np.linalg.norm(center - seed) >= radius - 2:
        metadata['reason'] = ('The best concentric pattern reaches the local search '
                              'boundary; click closer to its center.')
        return tuple(seed), metadata
    metadata.update(center_px=tuple(int(v) for v in center), confident=True,
                    reason='Centered on nearby concentric fringes.')
    return metadata['center_px'], metadata
