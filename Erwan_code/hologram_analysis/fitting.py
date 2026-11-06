"""Independent radius/depth starts with the existing fit objective and validation."""
from __future__ import annotations
from itertools import product
from dataclasses import replace
from hashlib import sha256
import json

from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np
import holopy as hp
from scipy.ndimage import gaussian_filter
from holopy.scattering import Sphere, calc_holo
from holopy.inference import prior, ExactModel
from .config import AnalysisConfig, AnalysisCancelled, check_cancelled
from .initialization import search_initial_parameters
from .fit_cache import fit_cache_key, get_cached_fit, put_cached_fit
from .diagnostics import same_final_solution as _same_final_solution

# Practical screening against a constant image, not a statistical accuracy test.
MIN_RELATIVE_IMPROVEMENT = 0.10


def _seed_branch(index, depth, medium):
    return (int(np.sign(index - medium)), int(np.sign(depth)))


def _stable_full_fit_agreement(trials, starts, cfg):
    """Conservative budget heuristic; never a proof of physical correctness."""
    if (not cfg.auto_initial_guess or not cfg.adaptive_fit_budget
            or cfg.search_effort == 'thorough'):
        return False
    attempted = [t for t in trials if t.get('status') != 'duplicate_start']
    if len(attempted) < 3:
        return False
    if any(t.get('converged') is not True
           or t.get('optimizer_status') not in (1, 2, 3, 4) for t in attempted):
        return False
    required = {_seed_branch(s['nguess'], s['zguess_um'], cfg.medium_index)
                for s in starts}
    covered = {_seed_branch(t['nguess'], t['zguess_um'], cfg.medium_index)
               for t in attempted}
    if not required <= covered:
        return False
    converged = [t for t in attempted if t.get('fit_acceptable') is True]
    if len(converged) < 3:
        return False
    best = min(converged, key=lambda t: t['chisq'])
    if ((best.get('relative_improvement') or 0.) < .7
            or best.get('boundary_parameters')):
        return False
    near = [t for t in converged if t['chisq'] <= best['chisq'] * 1.05 + 1e-15]
    if any(not _same_final_solution(best, t) for t in near):
        return False
    consistent = [t for t in near if (t.get('relative_improvement') or 0.) >= .7
                  and not t.get('boundary_parameters')]
    return len(consistent) >= 3


def _duplicate_refined_start(initial, trials, x_start, y_start, cfg):
    """Skip only an already refined start very close to a completed solution."""
    if (not cfg.auto_initial_guess or not cfg.adaptive_fit_budget
            or cfg.search_effort == 'thorough' or not initial.get('initial_refined')):
        return False
    for trial in trials:
        if trial.get('converged') is not True or not trial.get('fit_acceptable'):
            continue
        if (_seed_branch(initial['nguess'], initial['zguess_um'], cfg.medium_index)
                != _seed_branch(trial['n'], trial['z_um'], cfg.medium_index)):
            continue
        if (abs(initial['rguess_um'] - trial['r_um']) <= .01 * trial['r_um']
                and abs(initial['nguess'] - trial['n']) <= .0002
                and abs(initial['zguess_um'] - trial['z_um']) <= .1
                and abs(x_start - trial['x_um']) <= .2
                and abs(y_start - trial['y_um']) <= .2):
            return True
    return False


def crop_fit_bounds(h: Any, axis: str, guess_um: float,
                    margin_um: float) -> tuple[float, float]:
    """Restrict a lateral center to the crop's physical grid and local search margin.

    HoloPy retains the full image coordinate origin when cropping. Bounds must
    therefore use the crop coordinates, rather than its pixel count or a fixed
    range starting at zero. An initial center on/outside an edge is rejected so
    the uniform prior always starts strictly inside its bounds.
    """
    if axis not in h.coords:
        raise ValueError(f'The crop is missing its {axis} coordinate.')
    coordinates = np.asarray(h.coords[axis], dtype=float)
    if coordinates.size < 2 or not np.isfinite(coordinates).all():
        raise ValueError(
            f'The crop must have at least two finite {axis} coordinates.')
    if not np.isfinite([guess_um, margin_um]).all() or margin_um <= 0:
        raise ValueError(
            'Lateral guesses and the positive x/y search margin must be finite.'
        )
    lower = max(float(coordinates.min()), float(guess_um - margin_um))
    upper = min(float(coordinates.max()), float(guess_um + margin_um))
    if not lower < guess_um < upper:
        raise ValueError(
            f'The {axis} guess must lie strictly inside the crop coordinates.')
    return lower, upper


def _detail_value(details: Any, key: str) -> Any:
    return details.get(key) if isinstance(details, dict) else getattr(
        details, key, None)


def optimizer_diagnostics(result: Any) -> dict[str, Any]:
    """Read the installed HoloPy MPFIT result without assuming a fit converged.

    MPFIT statuses 1–4 and 6–8 are successful stopping criteria. Status 5 means
    the iteration limit was reached. Missing/unrecognized status is unknown,
    even when a HoloPy version happens to expose a permissive `converged` flag.
    """
    details = getattr(result, 'mpfit_details', None)
    status = _detail_value(details, 'status')
    if isinstance(
            status,
        (int, np.integer)) and not isinstance(status, (bool, np.bool_)):
        status = int(status)
    else:
        status = None
    explicit_convergence = _detail_value(details, 'converged')
    if status in (1, 2, 3, 4, 6, 7, 8):
        converged = explicit_convergence is None or bool(explicit_convergence)
    elif status is not None and (status <= 0 or status == 5):
        converged = False
    else:
        converged = None

    def counter(key: str) -> Optional[int]:
        value = _detail_value(details, key)
        return int(value) if isinstance(value,
                                        (int,
                                         np.integer)) and value >= 0 else None

    warning = ''
    if converged is None:
        warning = 'Optimizer convergence is unknown; this trial is excluded from selection.'
    elif not converged:
        reason = 'iteration limit reached' if status == 5 else 'optimizer did not converge'
        warning = f'{reason.capitalize()}; this trial is excluded from selection.'
    elif status in (6, 7, 8):
        warning = 'Optimizer stopped at numerical precision; inspect the residuals.'
    return {
        'converged': converged,
        'optimizer_status': status,
        'iterations': counter('niter'),
        'evaluations': counter('nfev'),
        'quality_warning': warning,
    }


def fit_quality_diagnostics(
    parameters: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    chisq: float,
    baseline_mse: float,
) -> dict[str, Any]:
    """Compare to the best constant image and flag parameters near a hard bound.

    These checks support visual inspection; they cannot establish that the
    single-sphere optical model is appropriate for an experimental image.
    """
    boundary = []
    for key, (lower, upper) in bounds.items():
        value = float(parameters[key])
        tolerance = max((upper - lower) * 1e-9, 1e-12)
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError(
                f'The fitted {key} parameter lies outside its bounds.')
        if min(abs(value - lower),
               abs(upper - value)) <= (upper - lower) * 1e-3:
            boundary.append(key)
    improvement = (baseline_mse -
                   chisq) / baseline_mse if baseline_mse > 0 else None
    warnings = []
    if improvement is None or improvement <= 0:
        warnings.append('The fit does not improve on a uniform image.')
    elif improvement < 0.1:
        warnings.append(
            'The fit improves on a uniform image by less than 10%.')
    elif improvement < 0.5:
        warnings.append(
            'Weak match: the fit removes less than 50% of the intensity '
            'variance. Inspect the fringes and neighboring particles; '
            'passing the minimum screen does not establish a good fit.')
    if boundary:
        labels = {
            'n': 'refractive index',
            'r': 'radius',
            'center.0': 'x',
            'center.1': 'y',
            'center.2': 'z'
        }
        warnings.append('Parameters near fitting bounds: ' +
                        ', '.join(labels.get(key, key)
                                  for key in boundary) + '.')
    return {
        'baseline_mse': baseline_mse,
        'relative_improvement': improvement,
        'boundary_parameters': ', '.join(boundary),
        'quality_warning': ' '.join(warnings),
    }


def residual_metrics(data: Any, prediction: Any) -> dict[str, Any]:
    """Describe residuals on exactly the same finite detector grid.

    Normalized MSE uses the experimental intensity variance, rather than an
    assumed camera-noise variance. Correlations and smoothed residual energy
    describe spatial structure; they are not calibrated significance tests or
    reduced chi-squared values. Gaussian smoothing is used only for diagnostics
    and never changes the intensities or objective supplied to HoloPy.
    """
    try:
        prediction = prediction.transpose(*data.dims)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            'Residuals require matching detector dimensions.') from exc
    if prediction.shape != data.shape or any(
            not np.array_equal(data.coords[dim], prediction.coords[dim])
            for dim in data.dims):
        raise ValueError('Residuals require exactly the same detector grid.')
    for dim in data.dims:
        coordinates = np.asarray(data.coords[dim], dtype=float)
        if not np.isfinite(coordinates).all():
            raise ValueError('Residual coordinates must be finite.')
    values = np.asarray(data, dtype=float)
    model = np.asarray(prediction, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(model).all():
        raise ValueError('Residual intensities must be finite.')
    try:
        residual = np.asarray(
            (data - prediction).squeeze().transpose('x', 'y'), dtype=float)
    except ValueError as exc:
        raise ValueError(
            'Residuals require one two-dimensional detector plane.') from exc
    if residual.size == 0:
        raise ValueError('Residuals require a nonempty detector plane.')

    def correlation(axis):
        first = np.take(residual, range(residual.shape[axis] - 1),
                        axis=axis).ravel()
        second = np.take(residual, range(1, residual.shape[axis]),
                         axis=axis).ravel()
        if first.size < 2:
            return None
        first = first - first.mean()
        second = second - second.mean()
        # Floating-point cancellation in a constant residual must not produce
        # a seemingly meaningful correlation.
        numerical_floor = 64 * np.finfo(float).eps * max(
            1., float(np.max(np.abs(residual))))
        if first.std() <= numerical_floor or second.std() <= numerical_floor:
            return None
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        return float(np.clip(np.dot(first, second) / denominator, -1., 1.))

    mse = float(np.mean(residual**2))
    variance = float(np.var(values))
    coherent = gaussian_filter(residual, 1., mode='reflect')
    return {
        'pixel_count': int(residual.size),
        'residual_rmse': float(np.sqrt(mse)),
        'residual_mean': float(residual.mean()),
        'residual_std': float(residual.std()),
        'normalized_mse': mse / variance if variance > 0 else None,
        'residual_corr_x': correlation(0),
        'residual_corr_y': correlation(1),
        'residual_coherent_mse': float(np.mean(coherent**2)),
    }


def _experiment_cache_key(h, xguess, yguess, z_values, cfg, seeds, scope=None):
    revision = sha256(json.dumps(dict(scope=scope, seeds=seeds), sort_keys=True,
                                default=str).encode('utf-8')).hexdigest()
    return fit_cache_key(h, xguess, yguess, z_values, cfg) + ':experiment:' + revision


def _restore_cached_fit(payload, elapsed, on_trial, log, *, experiment=False):
    """Replay cached trials while counting only work performed on this call."""
    result, selected, trials = payload
    fields = ('duration_s', 'search_duration_s', 'evaluations', 'iterations',
              'search_evaluations', 'search_grid_evaluations', 'search_short_fits',
              'search_stages', 'search_polished_starts', 'completed_fit_trials',
              'duplicate_starts_skipped')
    if experiment:
        fields += ('learned_search_evaluations', 'learned_search_duration_s',
                   'learned_search_failures')
    for trial in trials:
        for field in fields:
            trial['cached_' + field] = trial.get(field, 0)
            trial[field] = 0
        trial['cache_hit'] = True
    if trials:
        trials[0]['duration_s'] = elapsed
    log('Reused unchanged fit: image, settings and experiment profile match.'
        if experiment else
        'Reused unchanged fit: crop, center, calibration and search settings match.')
    if on_trial:
        on_trial(1, 1, dict(selected or {}, phase='cache', cache_hit=True))
    return result, selected, trials


def _combined_experiment_trials(groups, selected, mode, key, count, reason=''):
    """Count searches once per stage and local attempts once per trial."""
    combined = [trial for group in groups for trial in group]
    sums = {}
    for name in ('search_duration_s', 'search_evaluations',
                 'search_grid_evaluations', 'search_short_fits',
                 'search_polished_starts', 'search_stages',
                 'planned_fit_trials', 'completed_fit_trials',
                 'duplicate_starts_skipped'):
        sums[name] = sum(max((float(t.get(name) or 0) for t in group), default=0)
                         for group in groups)
        if name != 'search_duration_s':
            sums[name] = int(sums[name])
    learned_duration = (max((float(t.get('search_duration_s') or 0)
                             for t in groups[0]), default=0)
                        if mode in ('learned', 'fallback_full') else 0.)
    learned_metadata = {
        name: max((int(t.get(name) or 0) for t in groups[0]), default=0)
        if mode in ('learned', 'fallback_full') else 0
        for name in ('learned_search_evaluations', 'learned_search_failures',
                     'learned_search_pixels')}
    for trial in combined:
        trial.update(sums, **learned_metadata, selected=trial is selected,
                     guess_mode=mode, guess_profile_key=key,
                     guess_profile_entries=count,
                     guess_reuse_attempted=mode in ('learned', 'fallback_full'),
                     guess_reuse_accepted=mode == 'learned',
                     guess_fallback_reason=reason,
                     learned_search_duration_s=learned_duration)
    return combined


def fit_particle_multi_z(
    h: Any, xguess: float, yguess: float, z_values: tuple[float, ...],
    cfg: AnalysisConfig, on_trial: Optional[Callable] = None,
    cancel_event: Any = None, log: Callable = print, strategy: Any = None,
    log_trial_details: bool = True, initial_parameters: Optional[dict] = None,
    guess_library: Any = None, experiment_key: Optional[str] = None,
    guess_source: Optional[str] = None,
) -> tuple[Any, Optional[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse learned starts when appropriate, with the original broad fallback.

Only broad-search results teach the experiment profile. Fast results remain
independently fitted measurements, rather than new self-confirming anchors.
Manual initialization retains its original behavior.
    """
    options = dict(on_trial=on_trial, cancel_event=cancel_event, log=log,
                   strategy=strategy, log_trial_details=log_trial_details,
                   initial_parameters=initial_parameters)
    enabled = (guess_library is not None and experiment_key
               and cfg.auto_initial_guess and cfg.reuse_experiment_guesses
               and initial_parameters is None)
    if not enabled:
        return _fit_particle_multi_z(h, xguess, yguess, z_values, cfg, **options)
    cfg.validate()
    check_cancelled(cancel_event)
    seeds = guess_library.seeds(experiment_key, cfg)
    cache_started = perf_counter()
    if cfg.reuse_fit_cache and strategy is None and cfg.search_effort != 'thorough':
        try:
            key = _experiment_cache_key(h, xguess, yguess, z_values, cfg, seeds,
                                        experiment_key)
            cached = get_cached_fit(key)
        except (ValueError, TypeError, OverflowError):
            cached = None
        check_cancelled(cancel_event)
        if cached is not None:
            return _restore_cached_fit(
                cached, perf_counter()-cache_started, on_trial, log,
                experiment=True)
    settings = replace(cfg, reuse_fit_cache=False)
    fast = (None, None, [])
    fast_metadata = {}
    attempted = bool(seeds and cfg.search_effort != 'thorough')
    reason = ''
    if attempted:
        from .learned_initialization import (screen_learned_starts,
                                             reusable_fit_is_reliable)
        starts, fast_metadata = screen_learned_starts(
            h, xguess, yguess, seeds, cfg, cancel_event, on_trial, log)
        fast = _fit_particle_multi_z(
            h, xguess, yguess, z_values, settings, **options,
            _provided_starts=starts, _search_metadata=fast_metadata)
        for trial in fast[2]:
            trial['guess_stage'] = 'learned'
        accepted, reason = reusable_fit_is_reliable(fast[1], fast[2], seeds, cfg)
    else:
        accepted = False
    if accepted:
        result, selected = fast[:2]
        trials = _combined_experiment_trials(
            [fast[2]], selected, 'learned', experiment_key, len(seeds))
        log('Experiment profile: learned starts confirmed on this crop; '
            'the broad initial search was skipped.')
    else:
        if attempted:
            log(f'Experiment profile: {reason}; running the original broad search.')
        elif cfg.search_effort == 'thorough':
            log('Thorough search: running the original broad search and updating the experiment profile.')
        else:
            log('New experiment profile: running the original broad search first.')

        def full_progress(i, total, trial):
            if on_trial:
                on_trial(len(fast[2])+i, len(fast[2])+total, trial)

        try:
            full = _fit_particle_multi_z(
                h, xguess, yguess, z_values, settings,
                **dict(options, on_trial=full_progress))
        except AnalysisCancelled as exc:
            for t in exc.trials:
                t['guess_stage'] = 'full'
            records = _combined_experiment_trials(
                [fast[2], exc.trials], None,
                'fallback_full' if attempted else 'full',
                experiment_key, len(seeds), reason)
            raise AnalysisCancelled(records) from None
        for trial in full[2]:
            trial['guess_stage'] = 'full'
        available = [(r, t) for r, t, _ in (fast, full)
                     if r is not None and t is not None]
        result, selected = (min(available, key=lambda pair: pair[1]['chisq'])
                            if available else (None, None))
        # Keep quality-filtered broad anchors; do not replace them by fast fits.
        groups = [fast[2], full[2]] if attempted else [full[2]]
        if cancel_event is not None and cancel_event.is_set():
            records = _combined_experiment_trials(
                groups, None, 'fallback_full' if attempted else 'full',
                experiment_key, len(seeds), reason)
            raise AnalysisCancelled(records)
        guess_library.observe(experiment_key, full[2], cfg, source=guess_source)
        trials = _combined_experiment_trials(
            groups, selected, 'fallback_full' if attempted else 'full',
            experiment_key, len(seeds), reason)
        # A fully failed screening still consumed work, although it had no
        # local trials to carry its timing and evaluation metadata.
        if attempted and not fast[2]:
            for trial in trials:
                for name in ('search_duration_s', 'search_evaluations',
                             'search_grid_evaluations', 'search_stages'):
                    trial[name] += fast_metadata.get(name, 0)
                trial['learned_search_duration_s'] = fast_metadata.get('search_duration_s', 0.)
                trial['learned_search_evaluations'] = fast_metadata.get('search_evaluations', 0)
                trial['learned_search_pixels'] = fast_metadata.get('learned_search_pixels', 0)
                trial['learned_search_failures'] = fast_metadata.get('learned_search_failures', 0)
    if cancel_event is not None and cancel_event.is_set():
        for trial in trials:
            trial['selected'] = False
        raise AnalysisCancelled(trials)
    if cfg.reuse_fit_cache and strategy is None and cfg.search_effort != 'thorough':
        try:
            key = _experiment_cache_key(h, xguess, yguess, z_values, cfg,
                                        guess_library.seeds(experiment_key, cfg),
                                        experiment_key)
            put_cached_fit(key, (result, selected, trials))
        except (ValueError, TypeError, OverflowError):
            pass
    return result, selected, trials


def _fit_particle_multi_z(
    h: Any,
    xguess: float,
    yguess: float,
    z_values: tuple[float, ...],
    cfg: AnalysisConfig,
    on_trial: Optional[Callable] = None,
    cancel_event: Any = None,
    log: Callable = print,
    strategy: Any = None,
    log_trial_details: bool = True,
    initial_parameters: Optional[dict] = None,
    _provided_starts: Optional[list] = None,
    _search_metadata: Optional[dict] = None,
) -> tuple[Any, Optional[dict[str, Any]], list[dict[str, Any]]]:
    """Select coarse starts or manual radius/z pairs, then fit every crop pixel.

    Only converged fits improving on a constant image by at least 10% are
    selected. This screening rule does not establish physical model accuracy.
    An optional HoloPy strategy preserves caller-selected optimizer settings.
    """
    cfg.validate()
    check_cancelled(cancel_event)
    theory = cfg.theory()
    x_bounds = crop_fit_bounds(h, 'x', xguess, cfg.xy_fit_margin_um)
    y_bounds = crop_fit_bounds(h, 'y', yguess, cfg.xy_fit_margin_um)
    x_start = xguess if initial_parameters is None else float(
        initial_parameters['center.0'])
    y_start = yguess if initial_parameters is None else float(
        initial_parameters['center.1'])
    if not x_bounds[0] < x_start < x_bounds[1] or not y_bounds[
            0] < y_start < y_bounds[1]:
        raise ValueError(
            'Warm-start coordinates must lie inside the original lateral bounds.'
        )
    parameter_bounds = {
        'n': cfg.n_bounds,
        'r': cfg.r_bounds,
        'center.0': x_bounds,
        'center.1': y_bounds,
        'center.2': cfg.z_bounds,
    }

    image_values = np.asarray(h, dtype=float)
    if not np.isfinite(image_values).all():
        raise ValueError('The crop contains NaN or infinite values.')
    baseline_mse = float(np.mean((image_values - image_values.mean())**2))

    noise_sd = float(h.std() / h.mean())
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError(
            'The crop must have a finite, strictly positive noise_sd.')

    cache_key = None
    cache_started = perf_counter()
    if (cfg.reuse_fit_cache and strategy is None and initial_parameters is None
            and not (cfg.auto_initial_guess and cfg.search_effort == 'thorough')
            and _provided_starts is None):
        try:
            cache_key = fit_cache_key(h, xguess, yguess, z_values, cfg)
        except (ValueError, TypeError, OverflowError):
            # Optional reuse must not reject otherwise valid image metadata.
            cache_key = None
        cached = None if cache_key is None else get_cached_fit(cache_key)
        check_cancelled(cancel_event)
        if cached is not None:
            return _restore_cached_fit(
                cached, perf_counter()-cache_started, on_trial, log)

    search_metadata = dict(search_evaluations=0, search_pixels=0,
                           search_duration_s=0.)
    if _provided_starts is not None:
        starts = _provided_starts
        search_metadata.update(_search_metadata or {})
    elif cfg.auto_initial_guess:
        starts, search_metadata = search_initial_parameters(
            h, x_start, y_start, z_values, cfg, cancel_event=cancel_event,
            on_progress=on_trial, log=log, theory=theory)
    else:
        starts = [dict(nguess=cfg.n_guess, rguess_um=float(radius),
                       zguess_um=float(value), initial_source='manual',
                       initial_score=None)
                  for radius, value in product(cfg.r_guesses, z_values)]

    trials = []
    best_result = None
    best_trial = None
    evaluation_count = 0

    def cancellable_hologram(*args, **kwargs):
        # HoloPy calls calc_func for each model evaluation during optimization.
        # A compiled evaluation must return before Python can act on Stop.
        nonlocal evaluation_count
        check_cancelled(cancel_event)
        evaluation_count += 1
        hologram = calc_holo(*args, **kwargs)
        check_cancelled(cancel_event)
        return hologram

    total_trials = len(starts)
    budget_stopped = False
    skipped = 0
    for trial_index, initial in enumerate(starts, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise AnalysisCancelled(trials)
        zguess = initial['zguess_um']
        rguess = initial['rguess_um']
        nguess = initial['nguess']
        started = perf_counter()
        evaluation_count = 0
        trial = {
            **initial,
            **search_metadata,
            'zguess_um': zguess,
            'rguess_um': rguess,
            'status': 'error',
            'chisq': None,
            'n': None,
            'r_um': None,
            'x_um': None,
            'y_um': None,
            'z_um': None,
            'selected': False,
            'error': '',
            'converged': None,
            'optimizer_status': None,
            'iterations': None,
            'evaluations': None,
            'duration_s': None,
            'baseline_mse': baseline_mse,
            'relative_improvement': None,
            'quality_warning': '',
            'boundary_parameters': '',
            'fit_acceptable': False,
            'cache_hit': False,
        }
        if _duplicate_refined_start(initial, trials, x_start, y_start, cfg):
            skipped += 1
            trial.update(status='duplicate_start', duration_s=0., evaluations=0,
                         quality_warning='Refined start already represented by a converged fit.')
            trials.append(trial)
            if on_trial:
                on_trial(trial_index, total_trials, trial)
            continue
        trials.append(trial)
        try:
            if not np.isfinite(zguess) or not cfg.z_bounds[0] < zguess < cfg.z_bounds[1]:
                raise ValueError(
                    'zguess must be finite and strictly within the fitting bounds.'
                )
            x = prior.Uniform(lower_bound=x_bounds[0],
                              upper_bound=x_bounds[1],
                              guess=x_start)
            y = prior.Uniform(lower_bound=y_bounds[0],
                              upper_bound=y_bounds[1],
                              guess=y_start)
            z = prior.Uniform(lower_bound=cfg.z_bounds[0],
                              upper_bound=cfg.z_bounds[1],
                              guess=zguess)
            parameterized_sphere = Sphere(
                n=prior.Uniform(*cfg.n_bounds, nguess),
                r=prior.Uniform(*cfg.r_bounds, rguess),
                center=[x, y, z],
            )
            model = ExactModel(
                scatterer=parameterized_sphere,
                calc_func=cancellable_hologram,
                theory=theory,
                medium_index=cfg.medium_index,
                illum_wavelen=cfg.illum_wavelen,
                illum_polarization=cfg.illum_polarization,
                noise_sd=noise_sd,
            )
            fit_result = hp.fit(h, model) if strategy is None else hp.fit(
                h, model, strategy=strategy)
            check_cancelled(cancel_event)
            trial.update(optimizer_diagnostics(fit_result))
            fitted_holo = fit_result.hologram.transpose(*h.dims)
            # Check coordinates to prevent partial comparisons through xarray alignment.
            if fitted_holo.shape != h.shape or any(
                    not np.array_equal(h.coords[dim], fitted_holo.coords[dim])
                    for dim in h.dims):
                raise ValueError(
                    'The fitted hologram and crop must use the same grid.')
            if not np.isfinite(np.asarray(fitted_holo)).all():
                raise ValueError(
                    'The fitted hologram contains NaN or infinite values.')
            chisq = float(hp.core.math.chisq(h, fitted_holo))
            params = fit_result.parameters
            values = [
                params[key]
                for key in ('n', 'r', 'center.0', 'center.1', 'center.2')
            ]
            if not np.isfinite(chisq) or not np.isfinite(values).all():
                raise ValueError(
                    'The fit score or parameters contain NaN or infinite values.'
                )
            quality = fit_quality_diagnostics(params, parameter_bounds, chisq,
                                              baseline_mse)
            quality['quality_warning'] = ' '.join(
                warning for warning in (trial['quality_warning'],
                                        quality['quality_warning']) if warning)
            trial.update(quality)
            acceptable = (trial['converged'] is True
                          and quality['relative_improvement'] is not None
                          and quality['relative_improvement'] >= MIN_RELATIVE_IMPROVEMENT)
            trial.update(
                status=(('weak_fit' if quality['relative_improvement'] < .5 else 'ok')
                        if acceptable else 'poor_fit'
                        if trial['converged'] else 'unknown_convergence'
                        if trial['converged'] is None else 'not_converged'),
                fit_acceptable=acceptable,
                chisq=chisq,
                n=float(params['n']),
                r_um=float(params['r']),
                x_um=float(params['center.0']),
                y_um=float(params['center.1']),
                z_um=float(params['center.2']))
            if trial['quality_warning'] and log_trial_details:
                log(f'    nguess={nguess:g}, rguess={rguess:g} µm, zguess={zguess:g} µm : {trial["quality_warning"]}'
                    )
            if acceptable and (best_trial is None
                                       or chisq < best_trial['chisq']):
                best_result, best_trial = fit_result, trial
        except AnalysisCancelled:
            trial.update(status='cancelled', error='Stopped by the user.')
            raise AnalysisCancelled(trials) from None
        except Exception as exc:
            # A failed radius/depth start does not prevent subsequent trials.
            trial['error'] = f'{type(exc).__name__}: {exc}'
            if log_trial_details:
                log(f'    nguess={nguess:g}, rguess={rguess:g} µm, zguess={zguess:g} µm : {trial["error"]}'
                    )
        finally:
            trial['duration_s'] = perf_counter() - started
            if trial['evaluations'] is None:
                trial['evaluations'] = evaluation_count
        if on_trial is not None:
            on_trial(trial_index, total_trials, trial)
        if trial_index < total_trials and _stable_full_fit_agreement(trials, starts, cfg):
            budget_stopped = True
            log(f'Adaptive budget: stopped after {trial_index}/{total_trials} starts; '
                'three strong converged solutions agree after testing candidate branches. '
                'Use Thorough to run every start.')
            if on_trial:
                on_trial(trial_index, trial_index, dict(trial, phase='budget_complete'))
            break

    if best_trial is not None:
        best_trial['selected'] = True
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled(trials)
    for trial in trials:
        trial.update(planned_fit_trials=total_trials,
                     completed_fit_trials=len(trials) - skipped,
                     duplicate_starts_skipped=skipped,
                     fit_budget_stopped=budget_stopped)
    if skipped:
        log(f'Adaptive budget: skipped {skipped} already represented refined start(s).')
    if cache_key is not None:
        put_cached_fit(cache_key, (best_result, best_trial, trials))
    return best_result, best_trial, trials
