"""Browse completed particle figures without keeping every PNG in widget memory."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from .notebook_images import append_png, clear_images, validated_png


class FigureBrowserMixin:
    """Retain per-particle PNGs until the next run or interface closure."""

    def _clear_figure_store(self):
        if self._figure_store is not None:
            self._figure_store.cleanup()
            self._figure_store = None
        self._figure_groups.clear()
        self._figure_labels.clear()

    def _reset_figure_browser(self):
        clear_images(self.fit_images)
        self._clear_figure_store()
        self._figure_serial = 0
        self._sync_figure_options()
        self.figure_notice.value = 'No analysis figures yet.'

    def _sync_figure_options(self, preferred=None):
        self._syncing_figure_choice = True
        try:
            previous = self.figure_particle.value
            self.figure_particle.options = [
                (self._figure_labels[key], key) for key in self._figure_groups]
            choice = preferred if preferred in self._figure_groups else previous
            self.figure_particle.value = (
                choice if choice in self._figure_groups else
                next(iter(self._figure_groups), None))
            self.figure_particle.disabled = self._busy or not self._figure_groups
        finally:
            self._syncing_figure_choice = False

    def _particle_figure_label(self, key, image_path=None):
        image_index, particle_number = key
        if image_path is None and 0 <= image_index < len(self._analysis_paths):
            image_path = self._analysis_paths[image_index]
        name = f' — {Path(image_path).name}' if image_path else ''
        return f'Image {image_index}{name} — Candidate {particle_number}'

    def _append_analysis_figure(self, png_data, limit, particle=None):
        """Called on the kernel thread after the worker emits a figure."""
        try:
            png = validated_png(png_data)
            # The result browser is deliberately particle-only. Detection maps
            # and other unassigned figures must not create a misleading
            # "Overview / other figures" choice.
            if particle is None:
                return
            if limit < 1:
                raise ValueError('The figure display limit must be positive.')
            key = particle
            new_group = key not in self._figure_groups
            if self._figure_store is None:
                self._figure_store = TemporaryDirectory(prefix='hologram-figures-')
            self._figure_serial += 1
            path = Path(self._figure_store.name) / f'{self._figure_serial:08d}.png'
            path.write_bytes(png)
            records = self._figure_groups.get(key, []) + [path]
            self._figure_groups[key] = records[-limit:]
            for discarded in records[:-limit]:
                if discarded.parent == Path(self._figure_store.name):
                    discarded.unlink()
            self._figure_labels[key] = self._particle_figure_label(key)
            self._sync_figure_options(preferred=key if new_group else None)
            if self.figure_particle.value == key:
                self._render_particle_figures()
        except Exception as exc:
            self.fit_output.append_stderr(f'Figure display failed: {exc}\n')

    def _figure_particle_changed(self, change):
        if (self._syncing_figure_choice or self.figure_particle.disabled
                or change['new'] is None):
            return
        try:
            self._render_particle_figures()
        except Exception as exc:
            self.figure_notice.value = 'The selected figures could not be loaded.'
            self.fit_output.append_stderr(f'Figure display failed: {exc}\n')

    def _render_particle_figures(self):
        key = self.figure_particle.value
        paths = self._figure_groups.get(key, [])
        # Read before replacing the current gallery so a missing file cannot
        # leave an incomplete mix of figures belonging to different particles.
        images = [validated_png(path.read_bytes()) for path in paths]
        clear_images(self.fit_images)
        for png in images:
            append_png(self.fit_images, png)
        self.figure_notice.value = (
            f'{len(images)} figure(s) for this selection.' if images else
            'No figures for this particle. Enable Best-fit comparisons or Save analysis figures before the next analysis.')

    def _index_saved_figures(self, result):
        """Include exported PNGs even when their display during fitting was off."""
        for row in result.summary_rows:
            if row.get('image_index') is None or row.get('particle_number') is None:
                continue
            key = (int(row['image_index']), int(row['particle_number']))
            files = []
            if row.get('result_file'):
                result_path = Path(row['result_file'])
                files = [result_path.with_name(result_path.stem + suffix)
                         for suffix in ('_crop_bounds.png', '_normalized_crop.png',
                                        '_fit_comparison.png')]
                files = [path for path in files if path.is_file()]
            if files:
                self._figure_groups[key] = files[-self._analysis_figure_limit:]
            else:
                self._figure_groups.setdefault(key, [])
            self._figure_labels[key] = self._particle_figure_label(
                key, row.get('image_path'))
        self._sync_figure_options()
        if self.figure_particle.value is not None:
            self._render_particle_figures()
