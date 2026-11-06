"""Fit a prepared crop and record its diagnostics without cross-crop refits."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from .config import AnalysisCancelled
from .fitting import residual_metrics
from .diagnostics import solution_ambiguity


def fit_prepared_crop(h,
                      full_data,
                      xguess,
                      yguess,
                      cfg,
                      fit_function,
                      on_trial=None,
                      **fit_options):
    """Search or use manual starts, then fit only the prepared crop.

    The full_data argument is retained for caller compatibility. Crop geometry
    is decided before fitting; no reference-window fit or parameter-drift
    threshold is used. Preserve same-grid validity checks and Stop behavior.
    """
    started = perf_counter()
    winner_metrics = {}

    def record_crop(trial):
        trial.update(crop_stage='primary',
                     crop_height_px=h.sizes['x'],
                     crop_width_px=h.sizes['y'])

    def progress(i, total, trial):
        record_crop(trial)
        if on_trial:
            on_trial(i, total, trial)

    try:
        best, selected, records = fit_function(h,
                                               xguess,
                                               yguess,
                                               cfg.z_values,
                                               cfg,
                                               on_trial=progress,
                                               **fit_options)
    except AnalysisCancelled as exc:
        for record in exc.trials:
            record_crop(record)
            record['selected'] = False
        raise AnalysisCancelled(list(exc.trials)) from None
    trials = list(records)
    for record in trials:
        record_crop(record)
    if best is not None and selected is not None:
        try:
            winner_metrics = residual_metrics(h, best.hologram)
            selected.update(winner_metrics)
        except ValueError as exc:
            # Compatible custom fitters must satisfy the same grid checks.
            selected.update(selected=False, status='error', error=str(exc))
            best, selected = None, None

    info = dict(h.attrs.get('crop_selection', {}))
    for trial in trials:
        trial['selected'] = trial is selected
    durations = [trial.get('duration_s') for trial in trials]
    complete_durations = bool(durations) and all(
        value is not None and np.isfinite(value) for value in durations)
    total = (sum(float(value) for value in durations) if complete_durations
             else perf_counter() - started)
    if complete_durations:
        total += max((float(trial.get('search_duration_s', 0.))
                      for trial in trials), default=0.)
    info.update(
        pixel_count=int(h.sizes['x'] * h.sizes['y']),
        total_fit_time_s=float(total),
        converged=None if selected is None else selected.get('converged'))
    extent = info.get('extent_bounds_px')
    if extent is not None:
        extent_pixels = (extent[1] - extent[0]) * (extent[3] - extent[2])
        info['optimization_pixel_reduction'] = (
            1 - info['pixel_count'] / extent_pixels if extent_pixels else None)
    if selected is not None:
        selected.update(winner_metrics, total_fit_time_s=float(total))
        info.update(winner_metrics)
    info.update(solution_ambiguity(trials, selected))
    h.attrs['crop_selection'] = info
    return h, best, selected, trials
