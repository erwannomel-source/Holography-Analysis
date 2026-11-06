"""Reuse z-independent Mie matrices during one exploratory forward search.

The ordinary HoloPy calculator still evaluates every field and intensity. Only
its two scattering matrices are retained, with exact physical/quadrature keys.
The cache belongs to one temporary theory; fitted models keep ordinary MieLens.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from threading import RLock

import numpy as np
from holopy.scattering.theory import MieLens

try:
    from holopy.scattering.theory.mielensfunctions import MieLensCalculator
except ImportError:  # Older/different HoloPy builds use their ordinary theory.
    MieLensCalculator = None


MATRIX_CACHE_CAPACITY = 128
MATRIX_CACHE_MAX_BYTES = 4 * 1024 * 1024


def _array_key(value):
    array = np.ascontiguousarray(value)
    return array.dtype.str, array.shape, array.tobytes()


class _MatrixCache:
    """Small instance-local LRU, also bounded for nondefault quadrature sizes."""

    def __init__(self):
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = RLock()

    def __getstate__(self):
        # Reconstructed theories never share tables or nonserializable locks.
        return {}

    def __setstate__(self, state):
        self.__init__()


class _CachedMieLensCalculator(MieLensCalculator or object):
    def __init__(self, matrix_cache, **kwargs):
        self._matrix_cache = matrix_cache
        super().__init__(**kwargs)

    def _precompute_scattering_matrices(self):
        # The actual nodes capture quadrature count and lens angle. The size
        # parameter and index ratio already include wavelength and medium index.
        key = (MieLensCalculator._precompute_scattering_matrices,
               _array_key(self.index_ratio), _array_key(self.size_parameter),
               _array_key(self._theta_pts))
        cache = self._matrix_cache
        with cache.lock:
            if key in cache.entries:
                self._scat_perp_values, self._scat_prll_values = cache.entries[key]
                cache.entries.move_to_end(key)
                return
        super()._precompute_scattering_matrices()
        values = (np.array(self._scat_perp_values, copy=True),
                  np.array(self._scat_prll_values, copy=True))
        for value in values:
            value.setflags(write=False)
        size = sum(value.nbytes for value in values) + sum(
            len(part[2]) for part in key[1:])
        if size > MATRIX_CACHE_MAX_BYTES:
            return
        with cache.lock:
            # Concurrent duplicate calculations are harmless; keep one entry.
            if key in cache.entries:
                values = cache.entries.pop(key)
            else:
                cache.bytes += size
            cache.entries[key] = values
            while (len(cache.entries) > MATRIX_CACHE_CAPACITY
                   or cache.bytes > MATRIX_CACHE_MAX_BYTES):
                removed_key, removed_values = cache.entries.popitem(last=False)
                cache.bytes -= sum(value.nbytes for value in removed_values) + sum(
                    len(part[2]) for part in removed_key[1:])
        self._scat_perp_values, self._scat_prll_values = values


class _CachedMieLens(MieLens):
    def __init__(self, lens_angle=1.0, calculator_accuracy_kwargs=None):
        super().__init__(lens_angle=lens_angle,
                         calculator_accuracy_kwargs=deepcopy(
                             calculator_accuracy_kwargs or {}))
        self._matrix_cache = _MatrixCache()
        self._cache_supported = True

    def _create_calculator(self, particle_kz=None, index_ratio=None,
                           size_parameter=None):
        if self._cache_supported:
            try:
                return _CachedMieLensCalculator(
                    matrix_cache=self._matrix_cache,
                    particle_kz=particle_kz, index_ratio=index_ratio,
                    size_parameter=size_parameter, lens_angle=self.lens_angle,
                    **self.calculator_accuracy_kwargs)
            except (AttributeError, TypeError):
                # Private HoloPy details can vary between installations. Keep
                # the original numerical implementation as the fallback.
                self._cache_supported = False
        return super()._create_calculator(
            particle_kz=particle_kz, index_ratio=index_ratio,
            size_parameter=size_parameter)


def with_mie_matrix_cache(theory):
    """Return a temporary optimized plain MieLens, preserving custom theories."""
    if (type(theory) is not MieLens or MieLensCalculator is None
            or not hasattr(MieLensCalculator, '_precompute_scattering_matrices')
            or set(vars(theory)) - {'lens_angle', 'calculator_accuracy_kwargs'}):
        return theory
    return _CachedMieLens(
        lens_angle=theory.lens_angle,
        calculator_accuracy_kwargs=theory.calculator_accuracy_kwargs)
