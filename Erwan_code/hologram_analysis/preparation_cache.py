"""Detached, read-only preview snapshots handed to the analysis worker."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np


def file_fingerprint(path):
    """Match preview invalidation: resolved path, modification time and size."""
    path = Path(path)
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def _center_key(candidate):
    return (candidate['x'], candidate['y'],
            bool(candidate.get('manual_center')),
            bool(candidate.get('local_detection')))


def prepared_particle_key(fingerprint, number, candidate, cfg, neighbors):
    """Include the effective centre, crop settings and neighbor geometry."""
    return (fingerprint, int(number), _center_key(candidate), cfg.crop_key(),
            tuple(_center_key(c) for c in neighbors) if cfg.adaptive_crop else ())


def _freeze_array(values):
    copied = np.array(values, copy=True)
    copied.flags.writeable = False
    return copied


def _freeze_hologram(h):
    copied = h.copy(deep=True)
    copied.attrs = deepcopy(h.attrs)
    copied.data.flags.writeable = False
    return copied


def _working_hologram(h):
    # Geometry and intensities stay read-only; fit diagnostics may update attrs.
    copied = h.copy(deep=False)
    copied.attrs = deepcopy(h.attrs)
    return copied


class PreparationSnapshot:
    """Own detached buffers so worker diagnostics cannot modify the preview."""

    def __init__(self, detections, particles):
        self._images = {}
        self._particles = {}
        for key, loaded in detections.items():
            data, image, residual, activity, candidates = loaded
            self._images[key] = (
                _freeze_hologram(data), _freeze_array(image),
                _freeze_array(residual), _freeze_array(activity),
                deepcopy(candidates))
        for key, prepared in particles.items():
            if key[0] in self._images:
                self._particles[key] = prepared._replace(
                    hologram=_freeze_hologram(prepared.hologram))

    def image(self, path, bg_path, cfg):
        """Recheck files/settings at use time; a miss keeps the ordinary path."""
        try:
            key = (file_fingerprint(path), file_fingerprint(bg_path),
                   cfg.detection_key())
        except OSError:
            return None, None
        loaded = self._images.get(key)
        if loaded is None:
            return None, None
        data, image, residual, activity, candidates = loaded
        return (_working_hologram(data), image, residual, activity,
                deepcopy(candidates)), key

    def particle(self, fingerprint, number, candidate, cfg, neighbors):
        if fingerprint is None:
            return None
        key = prepared_particle_key(fingerprint, number, candidate, cfg, neighbors)
        prepared = self._particles.get(key)
        if prepared is None:
            return None
        return prepared._replace(hologram=_working_hologram(prepared.hologram))


def snapshot_preparation(detections, particles, paths, bg_path, cfg, mode,
                         image_indices=None):
    """Copy only previews compatible with this run; never load or detect here."""
    if not detections:
        return None
    detection_cfg = cfg if mode == 'saved' else replace(
        cfg, click_to_detect=False,
        detect_search_circle_only=cfg.detect_search_circle_only
        if mode == 'auto' else False)
    indices = range(len(paths)) if image_indices is None else image_indices
    background = file_fingerprint(bg_path)
    requested = {str(Path(paths[i]).resolve()) for i in indices}
    images = {}
    for key, loaded in detections.items():
        if (key[0][0] not in requested or key[1] != background
                or key[2] != detection_cfg.detection_key()):
            continue
        try:
            if file_fingerprint(key[0][0]) == key[0]:
                images[key] = loaded
        except OSError:
            # A missing image is handled per image by analyse_images. Optional
            # preview reuse must not abort an otherwise valid whole series.
            continue
    if not images:
        return None
    crops = {key: value for key, value in particles.items() if key[0] in images}
    return PreparationSnapshot(images, crops)
