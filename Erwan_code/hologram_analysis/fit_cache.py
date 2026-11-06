"""Memory-bounded cache for fits on exactly unchanged detector data/settings.

Content hashes cover data, physical coordinates, metadata, center guesses and
numerical settings. Stored/retrieved payloads are independent copies. The cache
does not read files or persist across kernel restarts, and never stores a running
optimizer or a cancelled result.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import gc
from hashlib import sha256
import json
import sys
from threading import RLock
from types import (BuiltinFunctionType, CodeType, FrameType, FunctionType,
                   MethodType, ModuleType)

import numpy as np
from holopy.scattering import calc_holo
from . import __version__

CAPACITY = 32
MAX_BYTES = 128 * 1024 * 1024
_CACHE = OrderedDict()
_SIZES = {}
_LOCK = RLock()
_cache_bytes = 0
_hits = 0
_misses = 0


def _estimated_memory(payload):
    """Estimate retained Python/NumPy storage without serializing fit objects.

    Follow strong references once, including xarray/pandas backing arrays, but
    never walk functions, classes or modules into the whole interpreter. This
    is a retention budget rather than a process-RSS measurement; native memory
    owned by third-party objects may not be fully visible to Python.
    """
    ignored = (type, ModuleType, FunctionType, BuiltinFunctionType, MethodType,
               CodeType, FrameType)
    builtins = (dict, list, tuple, set, frozenset, str, bytes, bytearray,
                int, float, complex, bool, type(None))
    pending, seen, total = [payload], set(), 0
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen or isinstance(value, ignored):
            continue
        seen.add(identity)
        if isinstance(value, np.ndarray):
            # NumPy's own size method includes owned data. Views retain their
            # base allocation, which the GC does not otherwise report here.
            total += np.ndarray.__sizeof__(value)
            base = np.ndarray.base.__get__(value)
            if base is not None:
                pending.append(base)
            if np.ndarray.dtype.__get__(value).hasobject:
                pending.extend(np.ndarray.flat.__get__(value))
        else:
            # Bypass user-defined __sizeof__/properties and pickle hooks.
            total += (sys.getsizeof(value) if type(value) in builtins else
                      object.__sizeof__(value) + (16 if gc.is_tracked(value) else 0))
            pending.extend(gc.get_referents(value))
    return total


def _discard(key):
    """Drop one entry and its accounting while the caller holds the lock."""
    global _cache_bytes
    if key in _CACHE:
        del _CACHE[key]
        _cache_bytes -= _SIZES.pop(key)


def _trim():
    while _CACHE and (len(_CACHE) > CAPACITY or _cache_bytes > MAX_BYTES):
        _discard(next(iter(_CACHE)))


def configure_fit_cache(*, capacity=None, max_bytes=None):
    """Set process-local limits, immediately evicting least-recently-used fits.

    Either limit may be zero to disable storage. No analysis settings or
    scientific results are changed, and existing callers need no new options.
    """
    global CAPACITY, MAX_BYTES
    for name, value in (('capacity', capacity), ('max_bytes', max_bytes)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or value < 0):
            raise ValueError(f'Cache {name} must be a nonnegative integer.')
    with _LOCK:
        if capacity is not None:
            CAPACITY = int(capacity)
        if max_bytes is not None:
            MAX_BYTES = int(max_bytes)
        _trim()
        return cache_info()

# These settings act before a prepared detector/center reaches the fitter, or
# control presentation/persistence afterwards. Actual detector bytes, physical
# coordinates and center guesses below already capture preparation changes.
# Keep this an exclusion list: future fitting settings must invalidate by default.
# Learned reuse policy remains hashed; its wrapper also hashes the actual
# experiment scope and seed library, independently of the human-readable name.
_NON_FIT_SETTINGS = (
    'target_x_px', 'target_y_px', 'search_radius_px', 'click_to_detect',
    'detect_search_circle_only', 'detect_sigma', 'detect_size_filter',
    'detect_min_diameter_px', 'detect_max_diameter_px', 'detect_min_distance',
    'detect_border', 'max_particles', 'refine_center', 'crop_height_px',
    'crop_width_px', 'adaptive_crop', 'adaptive_crop_rectangle',
    'adaptive_crop_min_px', 'adaptive_crop_max_px', 'adaptive_crop_margin_px',
    'show_detection_plots', 'show_crop_plots', 'show_fit_plots', 'save_figures',
    'max_displayed_figures', 'reuse_fit_cache', 'persist_experiment_guesses',
    'experiment_name',
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=value.shape,
                    values=value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'values'):
        return _json_value(np.asarray(value.values))
    return repr(value)


def fit_cache_key(h, xguess, yguess, z_values, cfg):
    """Hash actual inputs, rather than trusting an image path or modification time."""
    settings = asdict(cfg)
    for name in _NON_FIT_SETTINGS:
        settings.pop(name, None)
    attributes = {key: value for key, value in h.attrs.items()
                  if key != 'crop_selection'}
    digest = sha256()
    digest.update(json.dumps(dict(code=__version__, format=1, settings=settings,
                                 center=[xguess, yguess], z_values=tuple(z_values),
                                 dimensions=list(h.dims), attributes=attributes),
                             default=_json_value, sort_keys=True,
                             allow_nan=False).encode('utf-8'))

    def add_array(name, values):
        array = np.asarray(values)
        digest.update(name.encode('utf-8'))
        digest.update(str((array.dtype.str, array.shape)).encode('ascii'))
        if array.dtype.hasobject:
            digest.update(json.dumps(array.tolist(), default=_json_value,
                                     sort_keys=True).encode('utf-8'))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())

    add_array('intensity', h.values)
    for name in sorted(h.coords):
        add_array(name, h.coords[name].values)
    return digest.hexdigest()


def _clone(payload):
    copied = deepcopy(payload)
    result = copied[0]
    if result is not None and hasattr(result, 'model'):
        # The optimizer's cancellation closure belongs to its original run.
        # Cached FitResult.hologram is already materialized; future forwarding
        # should use the ordinary calculation, not an old cancellation event.
        model = result.model
        if hasattr(model, 'calc_func'):
            model.calc_func = calc_holo
    return copied


def get_cached_fit(key):
    global _hits, _misses
    with _LOCK:
        if key not in _CACHE:
            _misses += 1
            return None
        value = _CACHE[key]
        try:
            copied = _clone(value)
        except Exception:
            _discard(key)
            _misses += 1
            return None
        _CACHE.move_to_end(key)
        _hits += 1
        return copied


def put_cached_fit(key, payload):
    """A copy failure only disables reuse; it must never fail the analysis."""
    global _cache_bytes
    with _LOCK:
        if CAPACITY == 0 or MAX_BYTES == 0:
            return False
    try:
        copied = _clone(payload)
        size = _estimated_memory(copied)
    except Exception:
        return False
    with _LOCK:
        # Recheck limits after cloning: another thread may have reconfigured
        # them. An oversized replacement must not destroy a valid old entry.
        if CAPACITY == 0 or size > MAX_BYTES:
            return False
        _discard(key)
        _CACHE[key] = copied
        _SIZES[key] = size
        _cache_bytes += size
        _trim()
    return True


def clear_fit_cache():
    global _hits, _misses, _cache_bytes
    with _LOCK:
        _CACHE.clear()
        _SIZES.clear()
        _cache_bytes = 0
        _hits = _misses = 0


def cache_info():
    with _LOCK:
        return dict(entries=len(_CACHE), capacity=CAPACITY,
                    hits=_hits, misses=_misses, bytes=_cache_bytes,
                    max_bytes=MAX_BYTES)
