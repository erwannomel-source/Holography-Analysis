"""Compact fringe-based windows, using image processing before any fit.

All bounds are (row_start, row_stop, column_start, column_stop), with exclusive
stops. Filtering estimates crop geometry only; fitted intensities are untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (gaussian_filter, gaussian_filter1d, label)

from .config import MIN_CROP_PX


def bounded_window(shape, center, sizes):
    """Shift an even rectangle inside the image, keeping the actual center."""
    bounds = []
    for length, position, requested in zip(shape, center, sizes):
        if not 0 < position < length - 1:
            raise ValueError(
                'The particle center must be inside the image, '
                'away from its outermost pixel, for lateral fitting.')
        size = min(int(requested), length - length % 2)
        if size < MIN_CROP_PX:
            raise ValueError(
                'The image is too small for a valid fitting crop.')
        size -= size % 2
        start = int(np.clip(position - size // 2, 0, length - size))
        bounds.extend((start, start + size))
    return tuple(bounds)


def window_from_support(shape, center, low, high, minimum, maximum):
    """Enclose measured support without forcing equal margins on both sides."""
    bounds = []
    for length, position, first, last in zip(shape, center, low, high):
        size = max(minimum, int(last - first))
        size += size % 2
        size = min(size, maximum, length - length % 2)
        # Preserve both the signal bounds and the original guess whenever possible.
        # If the minimum dimension leaves spare room, use it to balance the
        # margins about the particle instead of about an asymmetric noise tail.
        if last - first <= size:
            start = int(
                np.clip(
                    int(position) - size // 2, np.ceil(last - size),
                    np.floor(first)))
        else:
            start = int(np.floor((first + last - size) / 2))
        start = int(np.clip(start, 0, length - size))
        start = min(start, int(position) - 1)
        start = max(start, int(position) - size + 2, 0)
        bounds.extend((start, start + size))
    return tuple(bounds)


def _owned_reference(bounds, center, neighbors):
    """Return whether every reference corner belongs to the target detection."""
    for neighbor in neighbors:
        x, y = neighbor['x'], neighbor['y']
        if bounds[0] <= x < bounds[1] and bounds[2] <= y < bounds[3]:
            return False
        for row in (bounds[0], bounds[1] - 1):
            for col in (bounds[2], bounds[3] - 1):
                if ((row - x)**2 + (col - y)**2
                        < (row - center[0])**2 + (col - center[1])**2):
                    return False
    return True


def _radial_profiles(values, distance, angles, owned, step, limit, sectors):
    """Pool pixels into annuli and angular sectors without interpolation."""
    bins = int(np.ceil(limit / step)) + 1
    radial = np.minimum(np.floor(distance / step).astype(int), bins - 1)
    sector = np.minimum((angles * sectors / (2 * np.pi)).astype(int),
                        sectors - 1)
    valid = owned & (distance <= limit)
    labels = (sector * bins + radial)[valid]
    counts = np.bincount(labels,
                         minlength=sectors * bins).reshape(sectors, bins)
    total = np.bincount(labels,
                        weights=values[valid],
                        minlength=sectors * bins).reshape(sectors, bins)
    signed = total / np.maximum(counts, 1)
    return signed, counts, radial, valid


def _hysteresis_extent(values, counts, sigma, step, seed_radius):
    """Follow connected alternating lobes, retaining weaker outer fringes.

    Four nominal noise standard errors seeds a train; two continue its tail.
    These thresholds are image-processing heuristics, not calibrated confidence
    intervals. Isolated one-bin noise excursions cannot create additional lobes.
    """
    amplitude = np.abs(values)
    snr = amplitude / np.maximum(sigma, 1e-12)
    near = (np.arange(len(values)) * step <= seed_radius) & (counts >= 8)
    peak = float(amplitude[near].max()) if near.any() else 0.0
    # Pooling many pixels can make an almost invisible tail nominally
    # significant. Follow useful lobes, not arbitrarily small averaged texture.
    weak = (snr >= 2.0) & (amplitude >= .02 * peak) & (counts >= 8)
    weak[:int(np.ceil(4 / step))] = False
    indices = np.flatnonzero(weak)
    if not len(indices):
        return 0.0, (), 0.0
    groups, group = [], [int(indices[0])]
    for index in indices[1:]:
        if (np.sign(values[index]) != np.sign(values[group[-1]])
                or index - group[-1] > 2):
            groups.append(group)
            group = []
        group.append(int(index))
    groups.append(group)
    lobes = []
    for group in groups:
        peak = int(group[np.argmax(snr[group])])
        if len(group) >= 2 or snr[peak] >= 5:
            lobes.append((group[0], group[-1], np.sign(values[peak]), peak,
                          float(snr[peak])))
    first = next((i for i, lobe in enumerate(lobes)
                  if lobe[4] >= 4 and lobe[0] * step <= seed_radius), None)
    if first is None:
        return 0.0, (), 0.0
    connected = [lobes[first]]
    separations = []
    for lobe in lobes[first + 1:]:
        previous = connected[-1]
        # Permit one invisible half-cycle, but never restart at a detached patch.
        gap = (max(3.0 / step, 1.6 *
                   np.median(separations)) if separations else 8.0 / step)
        if lobe[0] - previous[1] > gap:
            break
        if lobe[2] == previous[2]:
            # A broad same-sign tail does not supply another interference fringe.
            if lobe[1] - previous[0] > max(
                    8.0 / step,
                    2 * np.median(separations) if separations else 8.0 / step):
                break
            connected[-1] = (previous[0], lobe[1], previous[2], previous[3],
                             max(previous[4], lobe[4]))
        else:
            if separations and lobe[1] - lobe[0] > max(
                    4.0 / step, 2.0 * np.median(separations)):
                break
            separations.append(lobe[3] - previous[3])
            connected.append(lobe)
    if len(connected) < 3 or sum(lobe[4] >= 4 for lobe in connected) < 2:
        return 0.0, (), 0.0
    # The caller adds the explicit fringe margin. Do not also extrapolate an
    # unobserved half-lobe, which doubles the padding on compact holograms.
    extent = (connected[-1][1] + 1) * step
    return extent, tuple(lobe[3] for lobe in connected), float(
        np.median([lobe[4] for lobe in connected]))


def _activity_extent(local, distance, owned, camera_noise, cfg, lateral):
    """Conservative data-derived extent when resolved fringe trains are absent.

    Smoothed fringe activity joins bright/dark lobes for extent estimation only.
    The near-center connected component is enclosed, including a larger safety
    margin. If activity itself is unresolved, its size cannot be inferred: retain
    the automatically available image/neighbor context, with an explicit warning.
    """
    activity = gaussian_filter(local**2, max(2.0, cfg.detect_sigma))
    samples = activity[owned]
    if not len(samples):
        return None, 'No isolated pixels are available for extent estimation'
    quiet = samples[samples <= np.quantile(samples, .5)]
    floor = max(float(np.median(quiet)), camera_noise**2 * .8)
    scatter = float(1.4826 * np.median(np.abs(quiet - np.median(quiet))))
    near = owned & (distance <= max(12, lateral, 3 * cfg.detect_sigma))
    if not near.any():
        return None, 'The detected center has no isolated local activity'
    peak = float(activity[near].max())
    if peak <= floor + max(8 * scatter, camera_noise**2 * .5, 1e-12):
        return None, 'No above-noise interference extent could be established'
    threshold = floor + max(3 * scatter, (peak - floor) * .015, 1e-12)
    components, _ = label(owned & (activity > threshold))
    seeds = near & (activity >= floor + .25 * (peak - floor))
    identifiers = np.unique(components[seeds])
    identifiers = identifiers[identifiers > 0]
    if not len(identifiers):
        return None, 'Local fringe activity is not connected reliably'
    support = np.isin(components, identifiers)
    radii = distance[support]
    weights = np.maximum(activity[support] - floor, 0)
    order = np.argsort(radii)
    cumulative = np.cumsum(weights[order])
    if not len(cumulative) or cumulative[-1] <= 0:
        return None, 'No reliable fringe activity survives background removal'
    # Keep almost all measured local activity; unresolved patterns are never
    # aggressively converted into a narrow band.
    selected = np.searchsorted(cumulative, .995 * cumulative[-1])
    extent = float(radii[order[min(selected, len(order) - 1)]])
    return extent, 'Conservative connected activity extent; fringe phase is uncertain'


def _strip_inside_extent(extent, center, sizes):
    """Place an even band inside the extent reference, preserving its center."""
    bounds = []
    for low, high, position, requested in zip(extent[::2], extent[1::2],
                                              center, sizes):
        size = min(int(requested), high - low)
        size -= size % 2
        first = int(np.clip(int(position) - size // 2, low, high - size))
        bounds.extend((first, first + size))
    return tuple(bounds)


def _optimize_fringe_rectangle(local, rows, cols, distance, angles, owned,
                               extent, radius, center, camera_noise, cfg,
                               neighbors=(), isolation_uncertain=False):
    """Search even horizontal/vertical bands before any optical model call.

    A rectangle intentionally samples arcs rather than retaining whole rings.
    Every resolved lobe must retain >=25% of its available pixels and >=12
    samples, with opposing angular sectors, consistent phase and >=80% contrast.
    Each opposing half must also retain the reference phase and contrast;
    uncertain neighbor context requires stronger agreement, not a global veto.
    These inexpensive coverage checks select geometry before the fit. They
    are not a calibrated accuracy test or a comparison of physical fits.
    """
    step = 1.0
    signed, counts, radial, valid = _radial_profiles(local, distance, angles,
                                                     owned, step, radius, 1)
    profile, totals = gaussian_filter1d(signed[0], .45), counts[0]
    sigma = camera_noise * 1.5 / np.sqrt(np.maximum(totals, 1))
    snr = np.abs(profile) / np.maximum(sigma, 1e-12)
    significant = (snr >= 4) & (totals >= 12)
    significant[:4] = False
    # Evaluate the peaks of each resolved positive/negative lobe. Near a zero
    # crossing a relative contrast test has no physical meaning.
    peaks, group = [], []
    for index in np.flatnonzero(significant):
        if group and (np.sign(profile[index]) != np.sign(profile[group[-1]])
                      or index - group[-1] > 2):
            if len(group) >= 2 or max(snr[group]) >= 6:
                peaks.append(int(group[np.argmax(np.abs(profile[group]))]))
            group = []
        group.append(int(index))
    if group:
        if len(group) >= 2 or max(snr[group]) >= 6:
            peaks.append(int(group[np.argmax(np.abs(profile[group]))]))
    if len(peaks) < 3:
        return extent, dict(
            sampling_status='Full extent retained: unresolved radial contrast')
    height, width = extent[1] - extent[0], extent[3] - extent[2]
    # The requested x/y displacement is a maximum optimizer range, not a
    # required image size. crop_fit_bounds intersects it with the actual grid.
    minimum = max(cfg.adaptive_crop_min_px,
                  2 * int(np.ceil(3 * cfg.detect_sigma)))
    minimum += minimum % 2
    candidates = {extent}
    for short in range(minimum, max(height, width), 2):
        for sizes in ((min(short, height), width),
                      (height, min(short, width))):
            band = _strip_inside_extent(extent, center, sizes)
            candidates.add(band)
            if not neighbors:
                continue
            # A small shift can clear a diagonal neighbor without losing either
            # opposing arc. Preserve a central box and stay inside the extent.
            for axis in (0, 1):
                start, stop = band[2 * axis:2 * axis + 2]
                # Offset allowance follows the protected central box, not the
                # outer fringe padding. Reducing padding must not prevent a
                # clean, slightly asymmetric band from being considered.
                spare = min((stop - start - minimum) // 2, minimum // 2)
                for offset in range(-spare, spare + 1, 2):
                    first, last = start + offset, stop + offset
                    if (first < extent[2 * axis]
                            or last > extent[2 * axis + 1]
                            or first > center[axis] - minimum // 2
                            or last < center[axis] + minimum // 2):
                        continue
                    shifted = list(band)
                    shifted[2 * axis:2 * axis + 2] = (first, last)
                    candidates.add(tuple(shifted))
    # Average a small same-phase radial band around each peak. A single noisy
    # one-pixel annulus should not veto all otherwise representative rectangles.
    lobe_bins = np.full(len(totals), -1, dtype=int)
    for index, peak in enumerate(peaks):
        first, last = max(4, peak - 1), min(len(totals), peak + 2)
        bins = np.arange(first, last)
        bins = bins[np.sign(profile[bins]) == np.sign(profile[peak])]
        lobe_bins[bins] = index
    lobe_labels = lobe_bins[radial]
    measured = valid & (lobe_labels >= 0)
    labels = lobe_labels[measured]
    reference_counts = np.bincount(labels, minlength=len(peaks))
    reference_values = np.bincount(labels, weights=local[measured],
                                   minlength=len(peaks))
    reference_means = reference_values / np.maximum(reference_counts, 1)
    angular_sector = np.minimum((angles * 16 / (2 * np.pi)).astype(int), 15)
    ordered = sorted(candidates,
                     key=lambda b: ((b[1] - b[0]) * (b[3] - b[2]),
                                    abs(b[0] + b[1] - 2 * center[0])
                                    + abs(b[2] + b[3] - 2 * center[1]), b))
    for bounds in ordered:
        if bounds == extent:
            continue
        # A square's contaminated corner does not imply its narrower bands are
        # contaminated. Check the pixels that would actually enter the fit.
        if not _owned_reference(bounds, center, neighbors):
            continue
        inside = measured & (rows >= bounds[0]) & (rows < bounds[1]) & (
            cols >= bounds[2]) & (cols < bounds[3])
        selected_counts = np.bincount(lobe_labels[inside], minlength=len(peaks))
        values = np.bincount(lobe_labels[inside],
                             weights=local[inside],
                             minlength=len(peaks))
        means = values / np.maximum(selected_counts, 1)
        fractions = selected_counts / np.maximum(reference_counts, 1)
        contrast = np.abs(means) / np.maximum(np.abs(reference_means), 1e-12)
        if (np.any(selected_counts < 12) or np.any(fractions < .25)
                or np.any(contrast < (.9 if isolation_uncertain else .8))
                or np.any(np.sign(means) != np.sign(reference_means))):
            continue
        shortened_height = bounds[1] - bounds[0] < height
        long_axis = (cols, center[1]) if shortened_height else (rows, center[0])
        both_sides = True
        for half in (long_axis[0] < long_axis[1],
                     long_axis[0] >= long_axis[1]):
            pixels = inside & half
            counts_half = np.bincount(lobe_labels[pixels], minlength=len(peaks))
            values_half = np.bincount(lobe_labels[pixels],
                                      weights=local[pixels],
                                      minlength=len(peaks))
            means_half = values_half / np.maximum(counts_half, 1)
            # Weak outer lobes can be resolved by the full annulus while each
            # half remains noisy. Compare agreement at its measured precision;
            # a nominal contrast ratio alone would let one faint bin veto every
            # band. Strong missing or reversed lobes still fail these checks.
            tolerance = 2 * camera_noise * 1.5 / np.sqrt(
                np.maximum(counts_half, 1))
            aligned = means_half * np.sign(reference_means)
            if (np.any(counts_half < 6)
                    or np.any(aligned < -tolerance)
                    or np.any(aligned + tolerance < (
                        .8 if isolation_uncertain else .6)
                        * np.abs(reference_means))):
                both_sides = False
                break
        if not both_sides:
            continue
        angular_ok = True
        for index in range(len(peaks)):
            # The protected central box retains inner annuli in full. Requiring
            # three pixels in each tiny 16-sector arc there would reject even
            # the complete reference solely because the ring has few pixels.
            if selected_counts[index] >= .999 * reference_counts[index]:
                continue
            sector_counts = np.bincount(angular_sector[inside
                                                  & (lobe_labels == index)],
                                        minlength=16)
            active = sector_counts >= 3
            # Both opposing sides must contribute: a one-sided fragment cannot
            # establish reliable 2D location solely from a protected central box.
            if not any(active[k] and active[(k + 8) % 16] for k in range(8)):
                angular_ok = False
                break
        if angular_ok:
            return bounds, dict(
                sampling_status='Rectangle selected: resolved lobes sampled on opposing arcs',
                sampling_min_fraction=float(fractions.min()),
                sampling_min_contrast_ratio=float(contrast.min()),
                sampling_resolved_lobes=len(peaks))
    return extent, dict(
        sampling_status='Full extent retained: no isolated band met fringe sampling checks')


def select_adaptive_crop(residual, center, cfg, neighbors=(),
                         isolation_uncertain=False):
    """Estimate each hologram's extent, then select a fitting rectangle.

    Stage 1 is independent of manual fixed crop height/width. Strong directional
    fringes and angularly pooled weak fringes set an automatic reference extent.
    Ambiguous phase uses conservative local activity or available image/neighbor
    context, never a configured fixed-size fallback. Stage 2 tests even-sized
    bands, requiring radial/angular sampling without any reference-window fit.
    Filtering affects geometry only; HoloPy receives the original intensities.
    """
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2:
        raise ValueError(
            'Adaptive crop extent requires a two-dimensional image.')
    shape = residual.shape
    search = bounded_window(shape, center, (cfg.adaptive_crop_max_px, ) * 2)
    lateral = int(np.ceil(cfg.xy_fit_margin_um / cfg.spacing)) + 1
    info = dict(method='fallback',
                reason='',
                warning='',
                bounds_px=search,
                search_bounds_px=search,
                center_px=tuple(int(v) for v in center),
                signal_fraction=None,
                extent_confidence='unresolved',
                extent_reason='',
                extent_radius_px=None,
                optimization_pixel_reduction=0.0)
    x0, x1, y0, y1 = search
    local = residual[x0:x1, y0:y1]
    rows, cols = np.indices(local.shape)
    rows, cols = rows + x0, cols + y0
    distance = np.hypot(rows - center[0], cols - center[1])
    angles = np.mod(np.arctan2(cols - center[1], rows - center[0]), 2 * np.pi)
    owned = np.ones(local.shape, dtype=bool)
    for neighbor in neighbors:
        other_distance = (rows - neighbor['x'])**2 + (cols - neighbor['y'])**2
        owned &= distance**2 <= other_distance
        owned &= other_distance >= (cfg.detect_min_distance / 2)**2
    limit = cfg.adaptive_crop_max_px / 2 - 2
    warnings = []
    truncated = False
    finite = np.isfinite(local)
    if not finite.all():
        # A nonfinite geometry estimate cannot justify reducing usable data.
        extent, reason = search, 'Nonfinite extent data: available context retained'
        warnings.append('Fringe extent could not be measured reliably.')
        radius = None
        confidence = 'unresolved'
        smooth = None
        camera_noise = 0.0
    else:
        smooth = gaussian_filter(local, .8)
        highpass = local - gaussian_filter(local, 1.0)
        noise_values = highpass[owned] if owned.any() else highpass.ravel()
        camera_noise = max(
            float(1.4826 *
                  np.median(np.abs(noise_values - np.median(noise_values)))) /
            .87, 1e-8)
        info['noise_power'] = camera_noise**2
        candidates = []
        # Wide angular pooling recovers coherent low-contrast fringes that the
        # old strict 16-sector test systematically replaced with fixed windows.
        for sectors in (16, 8, 4, 1):
            profiles, counts, _, _ = _radial_profiles(smooth, distance, angles,
                                                      owned, 1.0, limit,
                                                      sectors)
            profiles = gaussian_filter1d(profiles, .45, axis=1)
            ends, strengths = [], []
            for direction, profile in enumerate(profiles):
                count = counts[direction]
                outer = (np.arange(len(profile)) >= .8 * limit) & (count >= 8)
                baseline = float(np.median(
                    profile[outer])) if outer.any() else 0.0
                sigma = camera_noise * 1.5 / np.sqrt(np.maximum(count, 1))
                end, peaks, strength = _hysteresis_extent(
                    profile - baseline, count, sigma, 1.0,
                    max(24.0, min(limit * .5, 2.0 * lateral)))
                if end > 0:
                    ends.append(end)
                    strengths.append(strength)
            needed = {16: 6, 8: 3, 4: 2, 1: 1}[sectors]
            if len(ends) >= needed:
                radius = float(np.quantile(ends, .9))
                candidates.append((sectors, radius, len(ends),
                                   float(np.median(strengths)), tuple(ends)))
                # Strong fine-sector estimates are selected directly below;
                # pooled profiles cannot affect their extent or diagnostics.
                # Weak estimates still need every pooling level for their tail.
                if sectors == 16 and candidates[-1][3] >= 6:
                    break
        if candidates:
            sectors, radius, coherent, strength, directional_ends = candidates[
                0]
            confidence = 'high' if sectors == 16 and strength >= 6 else 'weak'
            reason = ('Directional interference fringe extent'
                      if confidence == 'high' else
                      'Angularly pooled weak interference fringe extent')
            # Fine directional sectors already resolve the useful extent. For
            # weaker estimates only, pooling may recover tails within one
            # fringe period, without bridging a gap to a detached neighbor.
            if confidence != 'high':
                credible = [value[1] for value in candidates
                            if value[1] <= radius + 8]
                radius = max(credible)
            info.update(coherent_sectors=coherent,
                        extent_profile_sectors=sectors,
                        directional_extent_px=directional_ends)
            padding = cfg.adaptive_crop_margin_px
        else:
            radius, reason = _activity_extent(local, distance, owned,
                                              camera_noise, cfg, lateral)
            confidence = 'unresolved'
            padding = max(cfg.adaptive_crop_margin_px * 1.5,
                          3 * cfg.detect_sigma)
            warnings.append(
                'Fringe phase is uncertain; conservative extent retained.')
        if radius is None:
            # No fringe-based size is observable. Automatically retain the
            # largest useful local context, restricted by nearest detections.
            nearest = min((np.hypot(center[0] - c['x'], center[1] - c['y'])
                           for c in neighbors),
                          default=np.inf)
            available = (cfg.adaptive_crop_max_px /
                         2 if not neighbors else min(limit, nearest * .45))
            context_radius = max(lateral + cfg.adaptive_crop_margin_px,
                                 available)
            size = min(
                cfg.adaptive_crop_max_px,
                max(cfg.adaptive_crop_min_px,
                    2 * int(np.ceil(context_radius))))
            extent = bounded_window(shape, center, (size, size))
        else:
            half_size = int(np.ceil(radius + padding))
            # Clip the measured support at an image edge instead of shifting a
            # full symmetric square inward and adding empty opposite margins.
            extent = window_from_support(
                shape, center,
                tuple(max(0, c - half_size) for c in center),
                tuple(min(length, c + half_size)
                      for c, length in zip(center, shape)),
                cfg.adaptive_crop_min_px, cfg.adaptive_crop_max_px)
            if radius + padding >= cfg.adaptive_crop_max_px / 2:
                warnings.append(
                    'Fringes reach the search budget; coverage is truncated.')
                truncated = True
                # Observed resolved lobes can still establish a representative
                # band; the warning records that their outer extent is limited.
    reference_safe = _owned_reference(extent, center, neighbors)
    bounds = extent
    sampling = dict(sampling_status='Conservative full extent retained')
    if not cfg.adaptive_crop_rectangle:
        sampling = dict(
            sampling_status=
            'Automatic extent retained: rectangular reduction disabled')
    elif confidence != 'unresolved':
        bounds, sampling = _optimize_fringe_rectangle(smooth, rows, cols,
                                                      distance, angles,
                                                      owned, extent,
                                                      min(radius, limit),
                                                      center, camera_noise,
                                                      cfg, neighbors,
                                                      isolation_uncertain)
    if not reference_safe and bounds == extent:
        warnings.append('Measured extent includes pixels closer to a neighbor.'
                        + (' No isolated representative rectangle met the '
                           'fringe sampling checks.'
                           if cfg.adaptive_crop_rectangle else ''))
    if any(value == 0
           for value in (bounds[0], bounds[2])) or (bounds[1] == shape[0]
                                                    or bounds[3] == shape[1]):
        warnings.append('Image boundary limits available fringe coverage.')
    pixels = lambda b: (b[1] - b[0]) * (b[3] - b[2])
    info.update(
        method='adaptive' if confidence != 'unresolved' else 'fallback',
        reason=reason,
        warning=' '.join(warnings),
        bounds_px=tuple(bounds),
        extent_bounds_px=tuple(extent),
        extent_confidence=confidence,
        extent_reason=reason,
        extent_radius_px=radius,
        extent_truncated=truncated,
        optimization_pixel_reduction=1.0 - pixels(bounds) / pixels(extent),
        **sampling)
    if radius is not None and finite.all():
        inside = (rows >= bounds[0]) & (rows < bounds[1]) & (
            cols >= bounds[2]) & (cols < bounds[3]) & (distance
                                                       <= radius) & owned
        measured = owned & (distance <= radius)
        weights = np.maximum(smooth**2 - camera_noise**2 / (4 * np.pi), 0)
        total = float(weights[measured].sum())
        info['signal_fraction'] = float(weights[inside].sum() /
                                        total) if total else None
    return info


def extract_window(data, bounds, info):
    """Slice the original normalized data without moving physical coordinates."""
    x0, x1, y0, y1 = bounds
    h = data.isel(x=slice(x0, x1), y=slice(y0, y1)).copy(deep=False)
    h.attrs = dict(h.attrs, crop_selection=dict(info, bounds_px=bounds))
    return h
