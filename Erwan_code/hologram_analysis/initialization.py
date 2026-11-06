"""Bounded, reproducible forward search followed by diverse local-fit starts.

Every score uses one fixed subset of original detector pixels and one intensity
scale. Coordinates, optical theory and calibration are never rescaled. The
coarse scores select starts only; final fits are judged on the entire crop.
"""
from __future__ import annotations

from itertools import product
from time import perf_counter
from dataclasses import replace
import warnings
import numpy as np
import holopy as hp
from holopy.scattering import Sphere, calc_holo
from holopy.inference import ExactModel, NmpfitStrategy, prior

from .config import check_cancelled
from .optical_cache import with_mie_matrix_cache


def _axis_values(bounds, generated, hints=()):
    """Keep finite guesses strictly inside hard bounds, with bounded hints."""
    lower, upper = bounds
    hints = tuple(hints)
    if len(hints) > 4:
        hints = tuple(hints[i] for i in
                      np.linspace(0, len(hints) - 1, 4).astype(int))
    return sorted({float(value) for value in (*generated, *hints)
                   if np.isfinite(value) and lower < value < upper})


def search_axes(cfg, z_values):
    """Explore radius scales, both permitted focus sides and index contrasts.

    Small index differences from the medium need finer spacing than a uniform
    sweep over the full n interval. Broad index samples also cover particles
    with stronger contrast. User guesses add hints rather than limit the scan.
    """
    rlow, rhigh = cfg.r_bounds
    nlow, nhigh = cfg.n_bounds
    zlow, zhigh = cfg.z_bounds
    rinset = (rhigh - rlow) * 1e-4
    ninset = (nhigh - nlow) * 1e-4
    zinset = (zhigh - zlow) * 1e-4
    radii = _axis_values(cfg.r_bounds,
                        np.geomspace(rlow + rinset, rhigh - rinset, 8),
                        cfg.r_guesses)
    contrasts = tuple(cfg.medium_index + sign * delta
                      for delta in (.002, .01, .05) for sign in (-1, 1))
    indices = _axis_values(cfg.n_bounds,
                          (*np.linspace(nlow + ninset, nhigh - ninset, 5),
                           *contrasts), (cfg.n_guess,))
    depths = _axis_values(cfg.z_bounds,
                         np.linspace(zlow + zinset, zhigh - zinset, 12),
                         z_values)
    return radii, indices, depths


def sampled_detector(h, max_pixels):
    """Use a fixed stratified grid; retain the original physical coordinates."""
    height, width = h.sizes['x'], h.sizes['y']
    if height * width <= max_pixels:
        return h
    nx = max(2, min(height, int(np.sqrt(max_pixels * height / width))))
    ny = max(2, min(width, max_pixels // nx))
    while nx * ny > max_pixels:
        if nx >= ny:
            nx -= 1
        else:
            ny -= 1
    rng = np.random.default_rng(0)

    def positions(length, count):
        edges = np.linspace(0, length, count + 1).astype(int)
        return np.array([rng.integers(a, b)
                         for a, b in zip(edges[:-1], edges[1:])])

    return h.isel(x=positions(height, nx), y=positions(width, ny))


def distinct_starts(scored, cfg):
    """Avoid spending every local fit on nearly identical coarse candidates."""
    selected = []
    for candidate in sorted(scored, key=lambda item: item['initial_score']):
        if all(_candidate_separation(candidate, kept, cfg) >= 1.
               for kept in selected):
            selected.append(candidate)
        if len(selected) >= cfg.initial_search_trials:
            break
    return selected


def _candidate_separation(a, b, cfg):
    """Dimensionless distance used consistently for basin diversity checks."""
    zscale = max(5., .1 * (cfg.z_bounds[1] - cfg.z_bounds[0]))
    radius = np.log(a['rguess_um'] / b['rguess_um']) / np.log(1.5)
    depth = (a['zguess_um'] - b['zguess_um']) / zscale
    index = (np.arcsinh((a['nguess'] - cfg.medium_index) / .002)
             - np.arcsinh((b['nguess'] - cfg.medium_index) / .002))
    return float(np.sqrt(radius**2 + depth**2 + index**2))


def _candidate_key(candidate):
    return tuple(float(candidate[key])
                 for key in ('nguess', 'rguess_um', 'zguess_um'))


def staged_grid(radii, indices, depths, cfg, z_hints):
    """Split the existing grid into a broad half-grid and its exact remainder.

    The checkerboard preserves the full range of every physical axis, unlike
    shrinking bounds or retaining only low forward scores. Explicit manual
    hint combinations are added to the first stage. Small grids and Thorough
    mode retain the complete scan used before staged search was introduced.
    """
    grid = list(product(radii, indices, depths))
    if (len(grid) < 100 or not getattr(cfg, 'adaptive_fit_budget', True)
            or getattr(cfg, 'search_effort', 'custom') == 'thorough'):
        return grid, []
    hint_radii = set(cfg.r_guesses)
    hint_depths = set(z_hints)
    first, remaining = [], []
    for ir, radius in enumerate(radii):
        for ni, index in enumerate(indices):
            for zi, depth in enumerate(depths):
                hinted = (index == cfg.n_guess and radius in hint_radii
                          and depth in hint_depths)
                target = first if (ir + ni + zi) % 2 == 0 or hinted else remaining
                target.append((radius, index, depth))
    return first, remaining


def _escalation_reason(scored, refined, cfg, baseline_mse, failures):
    """Conservative sampled-data heuristic, never a physical fit certificate.

    Early stopping requires strong prediction agreement *and* repeated local
    convergence from independent seeds, no competing basin or bound warning,
    and enough diverse final starts. Unknown/iteration-limited convergence is
    useful for seeding but never justifies skipping the remaining half-grid.
    """
    if failures:
        return 'some exploratory evaluations failed'
    if not scored or not refined or not np.isfinite(baseline_mse) or baseline_mse <= 0:
        return 'insufficient exploratory evidence'
    ranked = sorted(scored + refined, key=lambda item: item['initial_score'])
    best = ranked[0]
    if best['initial_score'] > .3 * baseline_mse:
        return 'sampled agreement is still limited'
    for key, bounds in (('nguess', cfg.n_bounds), ('rguess_um', cfg.r_bounds),
                        ('zguess_um', cfg.z_bounds)):
        fraction = (best[key] - bounds[0]) / (bounds[1] - bounds[0])
        if min(fraction, 1. - fraction) < .01:
            return 'the best sampled candidate is close to a parameter bound'
    close_score = best['initial_score'] + .02 * baseline_mse
    if any(item['initial_score'] <= close_score
           and _candidate_separation(item, best, cfg) >= 1.
           for item in ranked[1:]):
        return 'different parameter basins have similar sampled scores'
    independent_seeds = []
    for candidate in sorted(refined, key=lambda item: item['initial_score']):
        if (candidate.get('initial_converged') is not True
                or candidate.get('initial_optimizer_status') not in (1, 2, 3, 4)
                or candidate['initial_score'] > close_score
                or _candidate_separation(candidate, best, cfg) >= .35):
            continue
        if not all(f'seed_{key}' in candidate
                   for key in ('nguess', 'rguess_um', 'zguess_um')):
            continue
        seed = {key: candidate[f'seed_{key}']
                for key in ('nguess', 'rguess_um', 'zguess_um')}
        if all(_candidate_separation(seed, other, cfg) >= 1.
               for other in independent_seeds):
            independent_seeds.append(seed)
    if len(independent_seeds) < 3:
        return 'independent short fits have not confirmed a stable basin'
    if len(distinct_starts(ranked, cfg)) < cfg.initial_search_trials:
        return 'too few diverse candidates are available for the final fits'
    return ''


def exploratory_seed_pool(scored, radii, depths, cfg, z_hints=None):
    """Retain coarse winners and explore seeds that initially score poorly.

    A raw forward score can favor weak, smooth patterns over a promising basin
    needing a substantial radius/index/depth adjustment. Reserve a small bounded
    budget for manual hints and radius/contrast/focus coverage before ranking.
    """
    seeds = []
    seen = set()

    def add(n, r, z):
        key = (float(n), float(r), float(z))
        if (key in seen or not np.isfinite(key).all()
                or not cfg.n_bounds[0] < n < cfg.n_bounds[1]
                or not cfg.r_bounds[0] < r < cfg.r_bounds[1]
                or not cfg.z_bounds[0] < z < cfg.z_bounds[1]):
            return
        seen.add(key)
        seeds.append(dict(nguess=float(n), rguess_um=float(r),
                          zguess_um=float(z), initial_source='auto'))

    for item in distinct_starts(scored, replace(cfg, initial_search_trials=3)):
        add(item['nguess'], item['rguess_um'], item['zguess_um'])
    # Endpoint combinations preserve useful suggestions even when their first
    # forward simulations are worse than every shortlisted raw-score candidate.
    hints_r = sorted(set(cfg.r_guesses))
    hints_z = sorted(set(cfg.z_values if z_hints is None else z_hints))
    hints_z = [z for z in hints_z if np.isfinite(z)
               and cfg.z_bounds[0] < z < cfg.z_bounds[1]]
    if not hints_z:
        hints_z = [depths[0], depths[-1]]
    for radius in (hints_r[0], hints_r[-1]):
        for depth in (hints_z[0], hints_z[-1]):
            add(cfg.n_guess, radius, depth)
    representatives = sorted({radii[int(q * (len(radii) - 1))]
                              for q in (.35, .6, .8)}, reverse=True)
    focus = []
    low, high = cfg.z_bounds
    if high > 0:
        focus.append(max(0., low) + .4 * (high - max(0., low)))
    if low < 0:
        focus.append(min(0., high) - .4 * (min(0., high) - low))
    contrasts = tuple(n for n in (cfg.medium_index + .002, cfg.medium_index - .002)
                      if cfg.n_bounds[0] < n < cfg.n_bounds[1]) or (cfg.n_guess,)
    for radius in representatives:
        for depth in focus:
            for index in contrasts:
                add(index, radius, depth)
    return seeds[:16]


def _first_stage_seeds(pool, cfg, limit=12):
    """Reserve hints, representative radii and branch coverage in a short budget."""
    if len(pool) <= limit:
        return pool
    # The original pool begins with up to three coarse winners and four manual
    # endpoint combinations. Preserve them even if their raw scores are poor.
    selected = list(pool[:7])
    remaining = list(pool[7:])

    def branch(seed):
        return (int(np.sign(seed['nguess'] - cfg.medium_index)),
                int(np.sign(seed['zguess_um'])))

    while remaining and len(selected) < limit:
        branches = {branch(seed) for seed in selected}
        radii = {seed['rguess_um'] for seed in selected}
        # Stable max keeps the original pool order when coverage is equal.
        choice = max(range(len(remaining)), key=lambda i:
                     (2 * (branch(remaining[i]) not in branches)
                      + (remaining[i]['rguess_um'] not in radii)))
        selected.append(remaining.pop(choice))
    return selected


def polish_candidates(detector, seeds, xguess, yguess, cfg, theory,
                      cancel_event=None, on_progress=None):
    """Briefly optimize n/r/z on the same sampled detector before selection.

    Centers stay fixed at this stage. A finite bounded iterate may be a useful
    start even when 15 iterations do not converge; it is never a final result.
    Final fits still use every original crop pixel and convergence screening.
    """
    refined = []
    evaluations = 0
    failures = 0
    observed = np.asarray(detector, dtype=float)

    def simulate(*args, **kwargs):
        nonlocal evaluations
        check_cancelled(cancel_event)
        evaluations += 1
        prediction = calc_holo(*args, **kwargs)
        check_cancelled(cancel_event)
        return prediction

    def progress(step):
        if on_progress:
            on_progress(step, len(seeds), {
                'status': 'searching', 'phase': 'initial_search',
                'search_step': 'short_fits', 'search_evaluations': evaluations,
                'search_pixels': int(detector.sizes['x'] * detector.sizes['y']),
            })

    progress(0)
    for step, seed in enumerate(seeds, 1):
        check_cancelled(cancel_event)
        try:
            sphere = Sphere(
                n=prior.Uniform(*cfg.n_bounds, seed['nguess']),
                r=prior.Uniform(*cfg.r_bounds, seed['rguess_um']),
                center=[xguess, yguess,
                        prior.Uniform(*cfg.z_bounds, seed['zguess_um'])])
            model = ExactModel(
                scatterer=sphere, calc_func=simulate, theory=theory,
                medium_index=cfg.medium_index, illum_wavelen=cfg.illum_wavelen,
                illum_polarization=cfg.illum_polarization,
                noise_sd=float(detector.std() / detector.mean()))
            with warnings.catch_warnings():
                # Reaching this deliberately short budget is expected. The
                # candidate must still pass the complete final fit later.
                warnings.filterwarnings('ignore', message='Minimizer Convergence Failed.*',
                                        category=UserWarning)
                result = hp.fit(detector, model, strategy=NmpfitStrategy(maxiter=15))
            check_cancelled(cancel_event)
            params = result.parameters
            n, r, z = (float(params[k]) for k in ('n', 'r', 'center.2'))
            if (not np.isfinite([n, r, z]).all()
                    or not cfg.n_bounds[0] < n < cfg.n_bounds[1]
                    or not cfg.r_bounds[0] < r < cfg.r_bounds[1]
                    or not cfg.z_bounds[0] < z < cfg.z_bounds[1]):
                raise ValueError('Exploratory parameters must remain inside bounds.')
            prediction = result.hologram.transpose(*detector.dims)
            if prediction.shape != detector.shape or any(
                    not np.array_equal(prediction.coords[dim], detector.coords[dim])
                    for dim in detector.dims):
                raise ValueError('Exploratory fit must match the sampled detector grid.')
            values = np.asarray(prediction, dtype=float)
            score = float(np.mean((observed - values)**2))
            if not np.isfinite(values).all() or not np.isfinite(score):
                raise ValueError('Exploratory prediction must be finite.')
            details = getattr(result, 'mpfit_details', None)
            status = (details.get('status') if isinstance(details, dict)
                      else getattr(details, 'status', None))
            explicit = (details.get('converged') if isinstance(details, dict)
                        else getattr(details, 'converged', None))
            recognized = (isinstance(status, (int, np.integer))
                          and not isinstance(status, (bool, np.bool_)))
            status = int(status) if recognized else None
            converged = (status in (1, 2, 3, 4, 6, 7, 8)
                         and (explicit is None or bool(explicit)))
            refined.append(dict(nguess=n, rguess_um=r, zguess_um=z,
                                initial_source='auto', initial_score=score,
                                initial_refined=True,
                                initial_converged=converged,
                                initial_optimizer_status=status,
                                seed_nguess=seed['nguess'],
                                seed_rguess_um=seed['rguess_um'],
                                seed_zguess_um=seed['zguess_um']))
        except Exception:
            failures += 1
        check_cancelled(cancel_event)
        progress(step)
    return refined, evaluations, failures


def search_initial_parameters(h, xguess, yguess, z_values, cfg,
                              cancel_event=None, on_progress=None, log=print,
                              theory=None):
    """Return diverse starts using bounded forward and short local exploration."""
    started = perf_counter()
    check_cancelled(cancel_event)
    detector = sampled_detector(h, cfg.initial_search_max_pixels)
    observed = np.asarray(detector, dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError('Initial search requires finite crop intensities.')
    radii, indices, depths = search_axes(cfg, z_values)
    total = len(radii) * len(indices) * len(depths)
    pixels = int(detector.sizes['x'] * detector.sizes['y'])
    if not total:
        raise ValueError('No initial-search values lie inside the fitting bounds.')
    first_grid, remaining_grid = staged_grid(
        radii, indices, depths, cfg, z_values)
    staged = bool(remaining_grid)
    log(f'Initial search: up to {total} simulations on {pixels} original pixels; '
        f'up to {cfg.initial_search_trials} full-crop fits.')
    if staged:
        log(f'Initial search: broad first stage uses {len(first_grid)} grid '
            'candidates; the remaining candidates are tested unless short fits '
            'provide strong, stable agreement.')
    if cfg.z_bounds[0] >= 0:
        log('Search range: negative z is excluded by Z bounds '
            f'{cfg.z_bounds}. Use signed bounds to explore both sides of focus.')
    elif cfg.z_bounds[1] <= 0:
        log('Search range: positive z is excluded by Z bounds '
            f'{cfg.z_bounds}. Use signed bounds to explore both sides of focus.')
    theory = cfg.theory() if theory is None else theory
    scan_theory = with_mie_matrix_cache(theory)
    scored = []
    refined = []
    failures = 0
    polish_failures = 0
    extra_evaluations = 0
    attempted = set()
    polished_seeds = set()
    last_error = ''
    stage_number = 1

    def progress():
        if on_progress:
            on_progress(len(attempted), total, {
                'status': 'searching', 'phase': 'initial_search',
                'search_pixels': pixels, 'search_evaluations': len(attempted),
                'search_failures': failures,
                'search_stage': stage_number,
            })

    def scan(grid):
        nonlocal failures, last_error
        progress()
        for step, (radius, index, depth) in enumerate(grid, 1):
            check_cancelled(cancel_event)
            key = (float(index), float(radius), float(depth))
            if key in attempted:
                continue
            attempted.add(key)
            try:
                prediction = calc_holo(
                    detector, Sphere(n=index, r=radius,
                                     center=[xguess, yguess, depth]),
                    medium_index=cfg.medium_index,
                    illum_wavelen=cfg.illum_wavelen,
                    illum_polarization=cfg.illum_polarization, theory=scan_theory)
                prediction = prediction.transpose(*detector.dims)
                if prediction.shape != detector.shape or any(
                        not np.array_equal(prediction.coords[dim], detector.coords[dim])
                        for dim in detector.dims):
                    raise ValueError('Initial simulation must match the detector grid.')
                values = np.asarray(prediction, dtype=float)
                score = float(np.mean((observed - values)**2))
                if not np.isfinite(values).all() or not np.isfinite(score):
                    raise ValueError('Initial simulation has nonfinite intensities.')
                scored.append(dict(nguess=index, rguess_um=radius,
                                   zguess_um=depth, initial_score=score,
                                   initial_source='auto'))
            except Exception as exc:
                failures += 1
                last_error = str(exc)
            check_cancelled(cancel_event)
            if step % 20 == 0 or step == len(grid):
                progress()

    def polish(pool, limit):
        nonlocal extra_evaluations, polish_failures
        seeds = [seed for seed in pool
                 if _candidate_key(seed) not in polished_seeds][:limit]
        if not seeds:
            return
        polished_seeds.update(_candidate_key(seed) for seed in seeds)
        log(f'Initial search: {len(seeds)} short exploratory fits before '
            'selecting the full-crop starts.')
        found, evaluations, failed = polish_candidates(
            detector, seeds, xguess, yguess, cfg, theory,
            cancel_event=cancel_event, on_progress=on_progress)
        refined.extend(found)
        extra_evaluations += evaluations
        polish_failures += failed

    scan(first_grid)
    if scored:
        pool = exploratory_seed_pool(scored, radii, depths, cfg, z_hints=z_values)
        polish(_first_stage_seeds(pool, cfg) if staged else pool,
               12 if staged else 16)
    escalation_reason = ''
    escalated = False
    if staged:
        baseline_mse = float(np.mean((observed - observed.mean())**2))
        escalation_reason = _escalation_reason(
            scored, refined, cfg, baseline_mse, failures + polish_failures)
        if escalation_reason:
            escalated = True
            stage_number = 2
            log('Initial search: expanding to the complete grid because '
                f'{escalation_reason}. Already tested candidates are reused.')
            scan(remaining_grid)
            if scored:
                pool = exploratory_seed_pool(
                    scored, radii, depths, cfg, z_hints=z_values)
                polish(pool, max(0, 16 - len(polished_seeds)))
        else:
            log('Initial search: sampled fits agree strongly and repeatedly; '
                'proceeding to full-crop checks. This is an acceleration '
                'heuristic, not proof of a unique or physically correct fit.')
    if not scored:
        raise ValueError(f'All {len(attempted)} initial simulations failed: {last_error}')
    if polish_failures:
        log(f'Initial search: {polish_failures} short exploratory fits failed; '
            'valid forward candidates are retained.')
    starts = distinct_starts(scored + refined, cfg)
    metadata = dict(search_evaluations=len(attempted) + extra_evaluations,
                    search_pixels=pixels,
                    search_duration_s=perf_counter() - started,
                    search_failures=failures + polish_failures,
                    search_polished_starts=len(refined),
                    search_stages=stage_number,
                    search_escalated=escalated,
                    search_escalation_reason=escalation_reason,
                    search_grid_evaluations=len(attempted),
                    search_short_fits=len(polished_seeds))
    if failures:
        log(f'Initial search: {failures} failed simulations skipped.')
    log('Initial search complete. Refining ' + str(len(starts)) + ' starts: '
        + '; '.join(f'n={s["nguess"]:.6g}, r={s["rguess_um"]:.4g} µm, '
                    f'z={s["zguess_um"]:.4g} µm' for s in starts))
    return starts, metadata
