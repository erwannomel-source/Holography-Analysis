"""Inexpensive agreement and ambiguity checks using fits already run.

The limits below are practical screening rules, not confidence intervals or a
calibrated statistical test. Parameter agreement controls search shortcuts;
the ambiguity report is informational and never changes fit selection.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from sys import float_info

NEAR_BEST_SCORE_FRACTION = 0.05
RADIUS_SPREAD_FRACTION = 0.10
DEPTH_SPREAD_FRACTION = 0.10
DEPTH_SPREAD_MIN_UM = 1.0
INDEX_SPREAD_MIN = 0.01


def same_final_solution(first, second):
    """Require close finite physical parameters before shortening a search."""
    names = ('r_um', 'n', 'z_um', 'x_um', 'y_um')
    try:
        a = {name: float(first[name]) for name in names}
        b = {name: float(second[name]) for name in names}
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (not all(isfinite(value) for value in (*a.values(), *b.values()))
            or min(a['r_um'], b['r_um'], a['n'], b['n']) <= 0):
        return False
    return (abs(a['r_um'] - b['r_um']) <= .02 * max(a['r_um'], b['r_um'])
            and abs(a['n'] - b['n']) <= .001
            and abs(a['z_um'] - b['z_um']) <= max(.25, .02 * abs(a['z_um']))
            and abs(a['x_um'] - b['x_um']) <= .2
            and abs(a['y_um'] - b['y_um']) <= .2)


def _finite_number(value):
    """Accept finite numeric values, excluding booleans and missing fields."""
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _fit_values(record):
    if not isinstance(record, Mapping) or record.get('converged') is not True:
        return None
    values = {
        name: _finite_number(record.get(name))
        for name in ('chisq', 'r_um', 'z_um', 'n')
    }
    if any(value is None for value in values.values()):
        return None
    if values['chisq'] < 0 or values['r_um'] <= 0 or values['n'] <= 0:
        return None
    return values


def _primary_grid(record):
    """Require explicit primary-stage dimensions before comparing starts."""
    if record.get('crop_stage') != 'primary':
        return None
    dimensions = [
        _finite_number(record.get(name))
        for name in ('crop_height_px', 'crop_width_px')
    ]
    if any(value is None or value <= 0 or not value.is_integer()
           for value in dimensions):
        return None
    return tuple(int(value) for value in dimensions)


def _same_optional_bounds(left, right):
    """Respect an explicit origin if a custom fitter supplies crop bounds."""
    for name in ('crop_bounds_px', 'bounds_px'):
        a, b = left.get(name), right.get(name)
        if a is None and b is None:
            continue
        try:
            if a is None or b is None or tuple(a) != tuple(b):
                return False
        except TypeError:
            return False
    return True


def _same_start(left, right, left_values, right_values):
    """Avoid counting a selected record copy as an additional start."""
    return left_values == right_values and all(
        left.get(name) == right.get(name)
        for name in ('rguess_um', 'zguess_um'))


def solution_ambiguity(trials, selected):
    """Describe competing solutions without additional fits or mutation.

    Compare only finite, explicitly converged primary fits on the winner's
    grid. The winner is included even when a custom fitter omitted grid tags;
    in that case other fits are skipped because their shared grid is unknown.
    A near-best fit has a score no more than 5% above the winner. When the
    winning score is exactly zero, an additive machine epsilon handles rounding.

    Flag disagreement if the near-best range spans at least 10% of the winning
    radius, max(1 micrometre, 10% of its absolute depth), or 0.01 in refractive
    index. These are transparent, pragmatic information thresholds, not a
    statistical uncertainty estimate. The existing winner remains unchanged.
    """
    result = {
        'solution_ambiguous': False,
        'near_best_count': 0,
        'ambiguity_note': '',
        'near_best_ranges': {},
    }
    winner = _fit_values(selected)
    if winner is None:
        return result

    near_best = [winner]
    grid = _primary_grid(selected)
    score_limit = (winner['chisq'] * (1 + NEAR_BEST_SCORE_FRACTION)
                   if winner['chisq'] else float_info.epsilon)
    matched_winner = False
    for record in trials or ():
        values = _fit_values(record)
        if values is None:
            continue
        if record is selected:
            matched_winner = True
            continue
        if grid is None or _primary_grid(record) != grid:
            continue
        if not _same_optional_bounds(record, selected):
            continue
        if not matched_winner and _same_start(record, selected, values,
                                              winner):
            matched_winner = True
            continue
        if values['chisq'] <= score_limit:
            near_best.append(values)

    ranges = {
        name: [
            min(values[name] for values in near_best),
            max(values[name] for values in near_best)
        ]
        for name in ('r_um', 'z_um', 'n')
    }
    limits = {
        'r_um':
        RADIUS_SPREAD_FRACTION * winner['r_um'],
        'z_um':
        max(DEPTH_SPREAD_MIN_UM, DEPTH_SPREAD_FRACTION * abs(winner['z_um'])),
        'n':
        INDEX_SPREAD_MIN,
    }
    names = {'r_um': 'radius', 'z_um': 'depth', 'n': 'refractive index'}
    differences = [
        names[name] for name, bounds in ranges.items()
        if bounds[1] - bounds[0] >= limits[name]
    ]
    result.update(solution_ambiguous=bool(differences),
                  near_best_count=len(near_best),
                  near_best_ranges=ranges)
    if differences:
        result['ambiguity_note'] = (
            f"{len(near_best)} converged starts within 5% of the best Chi² "
            f"disagree on {', '.join(differences)}. "
            "Informational heuristic; not a confidence interval.")
    return result
