"""Image preparation, candidate selection and crop geometry."""
from __future__ import annotations

import numpy as np
import holopy as hp
from typing import Callable, NamedTuple, TypedDict
from xarray import DataArray
from holopy.core.process import bg_correct, normalize
from scipy.ndimage import gaussian_filter, maximum_filter, map_coordinates
from .config import AnalysisConfig, MIN_CROP_PX


class Candidate(TypedDict, total=False):
    """Detected center in image pixels; x is the row, y is the column."""

    x: int
    y: int
    strength: float
    hologram_diameter_px: float
    local_detection: bool
    manual_center: bool


class ManualCenter(TypedDict):
    """A user correction in image pixels, before conversion to micrometers."""

    x: int
    y: int


class PreparedParticle(NamedTuple):
    """Crop with its image center in pixels and fit guesses in micrometers."""

    hologram: DataArray
    x_px: int
    y_px: int
    crop_shape_px: tuple[int, int]
    xguess_um: float
    yguess_um: float


def detect_candidates(hologram,
                      sigma=3.0,
                      min_distance=20,
                      border=35,
                      limit=15,
                      min_diameter_px=0,
                      max_diameter_px=0):
    """Find energy centers, optionally screening visible hologram diameters.

    Zero diameter limits retain the original detector exactly. Positive limits
    use size-guided image processing; normalization and the residual passed to
    cropping are unchanged. Sizes are in image pixels, not fitted micrometres.
    """
    data = hologram.squeeze()
    if tuple(data.dims) != ('x', 'y'):
        raise ValueError(
            f"Unexpected dimensions : {data.dims}; expected ('x', 'y')")
    image = np.asarray(data, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(
            'The image must be a 2D array without NaN or infinite values.')

    # Subtract the local background to remove slow variations; squaring reveals
    # bright and dark fringes. Smoothing combines the energy of the rings.
    residual = image - gaussian_filter(image, sigma=45)
    activity, candidates = _detect_residual(residual, sigma, min_distance,
                                            border, limit, min_diameter_px,
                                            max_diameter_px)
    return image, residual, activity, candidates


def _detect_residual(residual,
                     sigma,
                     min_distance,
                     border,
                     limit,
                     min_diameter_px=0,
                     max_diameter_px=0,
                     candidate_mask=None):
    """Detect on prepared residuals, preserving their full-image normalization."""
    if min_diameter_px or max_diameter_px:
        from .detection import detect_sized_candidates
        activity, candidates = detect_sized_candidates(
            residual,
            sigma,
            min_distance,
            border,
            limit,
            min_diameter_px,
            max_diameter_px,
            center_refiner=refine_ring_center,
            candidate_mask=candidate_mask)
        return activity, candidates
    activity = gaussian_filter(residual**2, sigma=sigma)
    mid = np.median(activity)
    mad = np.median(np.abs(activity - mid))
    threshold = mid + 8 * 1.4826 * mad
    maxima = (activity == maximum_filter(activity, size=2 * min_distance + 1))
    maxima &= activity > threshold
    if candidate_mask is not None:
        maxima &= candidate_mask
    if border > 0:
        maxima[:border, :] = False
        maxima[-border:, :] = False
        maxima[:, :border] = False
        maxima[:, -border:] = False
    coords = np.argwhere(maxima)
    coords = sorted(coords, key=lambda rc: activity[tuple(rc)], reverse=True)
    candidates = []
    for row, col in coords:
        if all(
                np.hypot(row - c['x'], col - c['y']) >= min_distance
                for c in candidates):
            candidates.append({
                'x': int(row),
                'y': int(col),
                'strength': float(activity[row, col])
            })
        if len(candidates) >= limit:
            break
    return activity, candidates


def refine_ring_center(residual: np.ndarray,
                       row: int,
                       col: int,
                       radius: int = 10) -> tuple[int, int]:
    """Refine a detected center using the symmetry of nearby intensity rings."""
    # Prefer centers with similar intensities around each ring.
    # Limit the search area to avoid switching to a different particle.
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    radii = np.arange(7, 37, 3)
    dr = radii[:, None] * np.cos(angles)[None, :]
    dc = radii[:, None] * np.sin(angles)[None, :]
    rows, columns = np.meshgrid(
        np.arange(row - radius, row + radius + 1),
        np.arange(col - radius, col + radius + 1),
        indexing='ij',
    )
    valid = ((rows >= 37) & (columns >= 37)
             & (rows < residual.shape[0] - 37) &
             (columns < residual.shape[1] - 37))
    rows, columns = rows[valid], columns[valid]
    if not len(rows):
        return int(row), int(col)
    # Batch the same interpolation queries, preserving the original row/column
    # traversal and first-maximum tie breaking while avoiding 441 Python calls.
    values = map_coordinates(
        residual,
        [rows[:, None, None] + dr, columns[:, None, None] + dc],
        order=1,
        mode='nearest',
    )
    ring_mean = values.mean(axis=2)
    scores = np.sum(ring_mean**2,
                    axis=1) / (np.mean(values**2, axis=2).sum(axis=1) + 1e-12)
    best = int(np.argmax(scores))
    return int(rows[best]), int(columns[best])


def safe_crop_size(shape, row, col, preferred):
    """Return an even crop size that stays inside the image and is at least MIN_CROP_PX."""
    if preferred < MIN_CROP_PX or preferred % 2:
        raise ValueError(
            f'CROP_SIZE_PX must be even and at least {MIN_CROP_PX}.')
    max_half = min(row, col, shape[0] - row, shape[1] - col)
    size = min(preferred, 2 * max_half)
    size -= size % 2
    if size < MIN_CROP_PX:
        raise ValueError(
            f'The center is too close to the edge for a {MIN_CROP_PX}-pixel crop.'
        )
    return size


def safe_crop_shape(shape, row, col, preferred_height, preferred_width):
    """Return even crop dimensions that stay inside the image and are at least MIN_CROP_PX."""

    def safe_dimension(length, center, preferred, label):
        if preferred < MIN_CROP_PX or preferred % 2:
            raise ValueError(
                f'{label} must be even and at least {MIN_CROP_PX} pixels.')
        max_half = min(center, length - center)
        size = min(preferred, 2 * max_half)
        size -= size % 2
        if size < MIN_CROP_PX:
            raise ValueError(
                f'The center is too close to the edge for a {MIN_CROP_PX}-pixel crop.'
            )
        return size

    height = safe_dimension(shape[0], row, preferred_height, 'Crop height')
    width = safe_dimension(shape[1], col, preferred_width, 'Crop width')
    return height, width


def load_prepared_image(raw_holo_path, bg, cfg):
    """Load and normalize an image without detecting candidate particles."""
    raw_holo = hp.load_image(
        str(raw_holo_path),
        spacing=cfg.spacing,
        medium_index=cfg.medium_index,
        illum_wavelen=cfg.illum_wavelen,
        illum_polarization=cfg.illum_polarization,
    )
    data_holo = normalize(bg_correct(raw_holo, bg))
    data = data_holo.squeeze()
    if tuple(data.dims) != ('x', 'y'):
        raise ValueError("Expected a two-dimensional HoloPy x/y image.")
    image = np.asarray(data, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('The image must be finite and two-dimensional.')
    residual = image - gaussian_filter(image, sigma=45)
    return data_holo, image, residual


def load_and_detect(raw_holo_path, bg, cfg):
    """Prepare once; optional click mode delays detection until a point is chosen."""
    data_holo, image, residual_image = load_prepared_image(
        raw_holo_path, bg, cfg)
    if cfg.click_to_detect:
        return data_holo, image, residual_image, np.zeros_like(image), []
    if cfg.detect_search_circle_only:
        from .selection import detect_in_circle
        activity, candidates = detect_in_circle(residual_image, cfg)
    else:
        activity, candidates = _detect_residual(
            residual_image, cfg.detect_sigma, cfg.detect_min_distance,
            cfg.detect_border, cfg.max_particles,
            cfg.detect_min_diameter_px if cfg.detect_size_filter else 0,
            cfg.detect_max_diameter_px if cfg.detect_size_filter else 0)
    return data_holo, image, residual_image, activity, candidates


def normalize_particle_choice(value):
    """Accept legacy auto/all/single choices and explicit lists, including no selection."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if any(
                isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, np.integer)) or number < 1
                for number in value):
            raise ValueError(
                'Selected particle numbers must be positive integers.')
        numbers = [int(number) for number in value]
        if len(set(numbers)) != len(numbers):
            raise ValueError('A particle number cannot be selected twice.')
        return numbers
    if isinstance(value,
                  (bool, np.bool_)) or not isinstance(value,
                                                      (int, np.integer)):
        raise ValueError('Invalid particle choice in the profile.')
    if value != -1 and value < 1:
        raise ValueError('Invalid particle choice in the profile.')
    return int(value)


def select_candidates(candidates,
                      cfg,
                      particle_number=None,
                      fit_all=False,
                      log=print):
    """Select explicit candidate numbers, all candidates, or the best near the target."""
    if isinstance(particle_number, (list, tuple)):
        selected = []
        for number in normalize_particle_choice(particle_number):
            selected.extend(
                select_candidates(candidates,
                                  cfg,
                                  particle_number=number,
                                  log=log))
        return selected
    number = particle_number
    if number is not None:
        if (isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, np.integer))
                or not 1 <= number <= len(candidates)):
            raise ValueError(
                f'The candidate number must be between 1 and {len(candidates)} '
                'for this image.')
        candidate = candidates[int(number) - 1]
        distance = np.hypot(candidate['x'] - cfg.target_x_px,
                            candidate['y'] - cfg.target_y_px)
        if distance > cfg.search_radius_px and not candidate.get(
                'local_detection', False):
            log(f'Warning: candidate {number} is {distance:.1f} px from the target, '
                'outside the circle; the explicit candidate choice is kept.')
        return [(int(number), candidate)]
    if fit_all:
        return list(enumerate(candidates, 1))
    nearby = [
        candidate for candidate in candidates
        if np.hypot(candidate['x'] - cfg.target_x_px, candidate['y'] -
                    cfg.target_y_px) <= cfg.search_radius_px
    ]
    if not nearby:
        raise ValueError(
            'No candidate inside the cyan circle: adjust the target and search radius '
            'or select a candidate number.')
    candidate = max(nearby,
                    key=lambda item: item['strength'] /
                    (1 + (np.hypot(item['x'] - cfg.target_x_px, item['y'] - cfg
                                   .target_y_px) / 35)**2))
    return [(candidates.index(candidate) + 1, candidate)]


def particle_crop_info(residual_image,
                       center,
                       cfg,
                       neighbors=(),
                       local_detection=False):
    """Estimate geometry, with private local-neighbor safeguards for clicked seeds."""
    from .cropping import select_adaptive_crop
    context = None
    if local_detection:
        from .neighbors import find_local_neighbors
        context = find_local_neighbors(residual_image, center, cfg, neighbors)
        neighbors = context['neighbors']
    info = select_adaptive_crop(
        residual_image, center, cfg, neighbors,
        isolation_uncertain=bool(context and context['ambiguous']))
    if context:
        info['local_neighbors_px'] = tuple(
            (int(c['x']), int(c['y'])) for c in neighbors)
        info['local_neighbor_ambiguous'] = bool(context['ambiguous'])
        if context['warning']:
            info['warning'] = ' '.join(
                filter(None, (info.get('warning'), context['warning'])))
    return info


def prepare_particle(
        data_holo: DataArray,
        image: np.ndarray,
        residual_image: np.ndarray,
        candidate: Candidate,
        cfg: AnalysisConfig,
        log: Callable = print,
        neighbors=(),
) -> PreparedParticle:
    """Prepare a rectangular crop and initial x/y positions in micrometers."""
    x_center, y_center = candidate['x'], candidate['y']
    # A manually corrected center is already the user's intended initial position.
    if (cfg.refine_center and not candidate.get('manual_center', False)
            and not candidate.get('local_detection', False)):
        x_center, y_center = refine_ring_center(residual_image, x_center,
                                                y_center)
    if cfg.adaptive_crop:
        from .cropping import extract_window
        info = particle_crop_info(residual_image, (x_center, y_center),
                                  cfg,
                                  neighbors,
                                  local_detection=candidate.get(
                                      'local_detection', False))
        h = extract_window(data_holo, info['bounds_px'], info)
        crop_shape = (h.sizes['x'], h.sizes['y'])
        log(f'Crop {crop_shape[0]} × {crop_shape[1]} px ({info["method"]}): {info["reason"]}'
            )
        if info['warning']:
            log(f'Crop warning: {info["warning"]}')
        return PreparedParticle(h, x_center, y_center, crop_shape,
                                float(data_holo.coords['x'].values[x_center]),
                                float(data_holo.coords['y'].values[y_center]))
    crop_height, crop_width = safe_crop_shape(image.shape, x_center, y_center,
                                              cfg.crop_height_px,
                                              cfg.crop_width_px)
    if (crop_height, crop_width) != (cfg.crop_height_px, cfg.crop_width_px):
        log(f'Warning: crop reduced to {crop_height} × {crop_width} pixels near the edge.'
            )
    x_start = x_center - crop_height // 2
    y_start = y_center - crop_width // 2
    h = data_holo.isel(x=slice(x_start, x_start + crop_height),
                       y=slice(y_start, y_start + crop_width))
    h = h.copy(deep=False)
    h.attrs = dict(h.attrs,
                   crop_selection={
                       'method':
                       'fixed',
                       'reason':
                       'Adaptive cropping disabled',
                       'warning':
                       '',
                       'bounds_px': (x_start, x_start + crop_height, y_start,
                                     y_start + crop_width)
                   })
    xguess = float(data_holo.coords['x'].values[x_center])
    yguess = float(data_holo.coords['y'].values[y_center])
    return PreparedParticle(h, x_center, y_center, (crop_height, crop_width),
                            xguess, yguess)


def apply_center_override(
    candidate: Candidate,
    override: ManualCenter | None,
    image_shape: tuple[int, int],
) -> Candidate:
    """Return a candidate using a validated manual x/y center in pixel coordinates."""
    if override is None:
        return candidate
    if not isinstance(override, dict) or set(override) != {'x', 'y'}:
        raise ValueError(
            'Each manual center must contain x and y pixel coordinates.')
    x, y = override['x'], override['y']
    if (isinstance(x, (bool, np.bool_)) or isinstance(y, (bool, np.bool_))
            or not isinstance(x, (int, np.integer))
            or not isinstance(y, (int, np.integer))):
        raise ValueError(
            'Manual x and y guesses must be whole pixel coordinates.')
    if not 0 <= x < image_shape[0] or not 0 <= y < image_shape[1]:
        raise ValueError(
            f'Manual center x={x}, y={y} is outside the image bounds '
            f'(x: 0–{image_shape[0] - 1}, y: 0–{image_shape[1] - 1}).')
    updated = dict(candidate)
    updated.update(x=int(x), y=int(y), manual_center=True)
    return updated
