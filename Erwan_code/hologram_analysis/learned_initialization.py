"""Cheap current-image screening of joint starts learned by broad searches.

This accelerates initialization; it neither fixes physical parameters nor
certifies a global optimum. The production fitter still optimizes every pixel.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from holopy.scattering import Sphere, calc_holo

from .config import check_cancelled
from .initialization import sampled_detector
from .diagnostics import same_final_solution
from .optical_cache import with_mie_matrix_cache


def branch(seed, medium):
    return (int(np.sign(seed['nguess'] - medium)),
            int(np.sign(seed['zguess_um'])))


def _inside(seed, cfg):
    return all(np.isfinite(seed[k]) and lo < seed[k] < hi
               for k, (lo, hi) in (('nguess', cfg.n_bounds),
                                    ('rguess_um', cfg.r_bounds),
                                    ('zguess_um', cfg.z_bounds)))


def learned_candidates(seeds, cfg):
    """Expand a small joint library without losing permitted contrast/focus sides.

Radius/depth perturbations cover nearby particles. A sparse depth sweep can
find a different axial location without rebuilding the full radius/index grid.
Mirrored values are hypotheses only; the optical model scores them normally.
"""
    anchors = []
    # Reserve branch representatives before filling the limited anchor budget.
    for seed in seeds:
        if _inside(seed, cfg) and branch(seed, cfg.medium_index) not in {
                branch(s, cfg.medium_index) for s in anchors}:
            anchors.append(dict(seed))
    for seed in seeds:
        if len(anchors) >= 6:
            break
        if _inside(seed, cfg) and not any(
                all(seed[k] == a[k] for k in ('nguess', 'rguess_um', 'zguess_um'))
                for a in anchors):
            anchors.append(dict(seed))
    for seed in list(anchors):
        reflected_n = 2 * cfg.medium_index - seed['nguess']
        for n, z in ((reflected_n, seed['zguess_um']),
                     (seed['nguess'], -seed['zguess_um']),
                     (reflected_n, -seed['zguess_um'])):
            hypothesis = dict(seed, nguess=n, zguess_um=z)
            if _inside(hypothesis, cfg) and branch(hypothesis, cfg.medium_index) not in {
                    branch(s, cfg.medium_index) for s in anchors}:
                anchors.append(hypothesis)
    generated = {}

    def add(n, r, z):
        candidate = dict(nguess=float(n), rguess_um=float(r), zguess_um=float(z),
                         initial_source='experiment_profile', initial_refined=False)
        if _inside(candidate, cfg):
            generated.setdefault((candidate['nguess'], candidate['rguess_um'],
                                  candidate['zguess_um']), candidate)

    zlow, zhigh = cfg.z_bounds
    inset = max(1e-6, (zhigh-zlow)*1e-4)
    swept = set()
    for seed in anchors:
        n, r, z = (seed[k] for k in ('nguess', 'rguess_um', 'zguess_um'))
        delta = max(1., .2 * abs(z))
        for radius in (.85*r, r, 1.18*r):
            for depth in (z-delta, z, z+delta):
                add(n, radius, depth)
        # New particles can have different radii and depths even in the same
        # sample. A small joint sweep covers that variation at learned index
        # contrasts without rebuilding the original full n/r/z grid.
        seed_branch = branch(seed, cfg.medium_index)
        radii = (.5*r, r, 2*r) if seed_branch not in swept else (r,)
        swept.add(seed_branch)
        for radius in radii:
            for depth in np.linspace(zlow+inset, zhigh-inset, 6):
                add(n, radius, depth)
    return list(generated.values())


def screen_learned_starts(h, xguess, yguess, seeds, cfg,
                         cancel_event=None, on_trial=None, log=print):
    """Return a few starts after forward simulations on fixed original pixels."""
    started = perf_counter()
    detector = sampled_detector(h, min(1600, cfg.initial_search_max_pixels))
    observed = np.asarray(detector, dtype=float)
    theory = with_mie_matrix_cache(cfg.theory())
    candidates = learned_candidates(seeds, cfg)
    scored = []
    failures = 0
    log(f'Experiment profile: testing {len(candidates)} learned starts on '
        f'{int(detector.size)} original pixels.')
    for index, candidate in enumerate(candidates, 1):
        check_cancelled(cancel_event)
        try:
            predicted = calc_holo(detector, Sphere(
                n=candidate['nguess'], r=candidate['rguess_um'],
                center=[xguess, yguess, candidate['zguess_um']]), theory=theory,
                medium_index=cfg.medium_index, illum_wavelen=cfg.illum_wavelen,
                illum_polarization=cfg.illum_polarization).transpose(*detector.dims)
            if predicted.shape != detector.shape or any(
                    not np.array_equal(predicted.coords[d], detector.coords[d])
                    for d in detector.dims):
                raise ValueError('Learned-start simulation must match the current detector.')
            values = np.asarray(predicted, dtype=float)
            score = float(np.mean((observed-values)**2))
            if not np.isfinite(values).all() or not np.isfinite(score):
                raise ValueError('Learned-start predictions must be finite.')
            scored.append(dict(candidate, initial_score=score))
        except Exception:
            failures += 1
        check_cancelled(cancel_event)
        if on_trial and (index == 1 or index % 10 == 0 or index == len(candidates)):
            on_trial(index, len(candidates), dict(
                status='searching', phase='learned_search',
                search_evaluations=index, search_pixels=int(detector.size)))
    ranked = sorted(scored, key=lambda s: s['initial_score'])
    selected = []
    for seed in ranked:
        if branch(seed, cfg.medium_index) not in {
                branch(s, cfg.medium_index) for s in selected}:
            selected.append(seed)
    limit = max(3, min(4, cfg.initial_search_trials))
    # Distinct perturbations provide local confirmation on all original pixels.
    for seed in ranked:
        if len(selected) >= limit:
            break
        if all(abs(np.log(seed['rguess_um']/s['rguess_um'])) > .06
               or abs(seed['zguess_um']-s['zguess_um']) > max(.5, .05*abs(s['zguess_um']))
               or abs(seed['nguess']-s['nguess']) > .0002 for s in selected):
            selected.append(seed)
    # Sparse libraries can still contain fewer useful distinct starts. The
    # reliability check will require confirmation or request the broad search.
    selected.sort(key=lambda s: s['initial_score'])
    metadata = dict(search_evaluations=len(candidates),
                    search_pixels=int(detector.size),
                    search_duration_s=perf_counter()-started,
                    search_grid_evaluations=len(candidates), search_short_fits=0,
                    search_polished_starts=0, search_stages=1,
                    search_escalated=False, search_escalation_reason='',
                    learned_search_evaluations=len(candidates),
                    learned_search_failures=failures,
                    learned_search_pixels=int(detector.size))
    return selected, metadata


def reusable_fit_is_reliable(selected, trials, seeds, cfg):
    """A conservative speed heuristic, not a statistical or physical guarantee."""
    if (selected is None or selected.get('converged') is not True
            or selected.get('fit_acceptable') is not True):
        return False, 'learned starts did not produce an acceptable converged fit'
    if (selected.get('chisq') is None or selected.get('relative_improvement') is None
            or not np.isfinite([selected['chisq'], selected['relative_improvement']]).all()
            or selected['chisq'] < 0):
        return False, 'the learned fit has invalid quality metrics'
    if selected.get('optimizer_status') not in (1, 2, 3, 4):
        return False, 'the optimizer did not confirm normal convergence'
    if selected.get('boundary_parameters'):
        return False, 'the learned fit approaches a physical fitting bound'
    qualities = [float(s['relative_improvement']) for s in seeds
                 if s.get('relative_improvement') is not None
                 and np.isfinite(s['relative_improvement'])]
    minimum = max(.7, min(.9, float(np.median(qualities))-.15)) if qualities else .7
    if (selected.get('relative_improvement') or 0.) < minimum:
        return False, 'the new image is not explained well enough by the learned starts'
    near = [t for t in trials if t.get('converged') is True
            and t.get('optimizer_status') in (1, 2, 3, 4)
            and not t.get('boundary_parameters') and t.get('fit_acceptable') is True
            and t.get('chisq') is not None
            and t['chisq'] <= selected['chisq']*1.05 + 1e-15]
    if len(near) < 2:
        return False, 'independent learned starts have not confirmed the current fit'
    if any(not same_final_solution(selected, trial) for trial in near):
        return False, 'learned starts found competing parameter solutions'
    return True, ''
