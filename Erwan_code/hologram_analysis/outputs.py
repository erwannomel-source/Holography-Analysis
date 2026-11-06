"""CSV checkpoints and compatibility helpers for saved analysis results."""
from __future__ import annotations

import csv
import io
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Union
from . import __version__

INITIAL_SEARCH_FIELDS = [
    'nguess', 'initial_source', 'initial_score', 'search_evaluations',
    'search_pixels', 'search_duration_s', 'search_polished_starts',
    'initial_refined', 'seed_nguess', 'seed_rguess_um', 'seed_zguess_um',
    'cache_hit', 'cached_duration_s', 'cached_search_duration_s',
    'cached_evaluations', 'cached_iterations', 'cached_search_evaluations',
    'cached_search_grid_evaluations', 'cached_search_short_fits',
    'cached_search_stages', 'cached_search_polished_starts',
    'cached_completed_fit_trials', 'cached_duplicate_starts_skipped',
    'planned_fit_trials', 'completed_fit_trials',
    'duplicate_starts_skipped', 'fit_budget_stopped', 'search_stages',
    'search_escalated', 'search_escalation_reason', 'search_grid_evaluations',
    'search_short_fits',
    'guess_mode', 'guess_stage', 'guess_profile_key', 'guess_profile_entries',
    'guess_reuse_attempted', 'guess_reuse_accepted', 'guess_fallback_reason',
    'learned_search_evaluations', 'learned_search_duration_s',
    'learned_search_pixels', 'learned_search_failures',
    'cached_learned_search_evaluations', 'cached_learned_search_duration_s',
    'cached_learned_search_failures',
]

SUMMARY_FIELDS = [
    'image_index', 'image_path', 'particle_number', 'x_px', 'y_px',
    'crop_height_px', 'crop_width_px', 'xguess_um', 'yguess_um', 'rguess_um',
    'zguess_um', 'x_um', 'y_um', 'z_um', 'r_um', 'n', 'chisq', 'result_file',
    'crop_bounds_px', 'crop_method', 'crop_warning', 'pixel_count',
    'converged', 'total_fit_time_s', 'preparation_time_s', 'residual_rmse',
    'residual_mean', 'residual_std', 'normalized_mse', 'residual_corr_x',
    'residual_corr_y', 'residual_coherent_mse', 'extent_bounds_px',
    'extent_radius_px', 'extent_confidence', 'extent_reason',
    'optimization_pixel_reduction', 'solution_ambiguous', 'near_best_count',
    'ambiguity_note', 'near_best_ranges', 'fit_acceptable', 'baseline_mse',
    'relative_improvement', 'quality_warning', 'boundary_parameters',
    *INITIAL_SEARCH_FIELDS,
]

TRIAL_FIELDS = [
    'image_index', 'image_path', 'particle_number', 'rguess_um', 'zguess_um',
    'status', 'chisq', 'n', 'r_um', 'x_um', 'y_um', 'z_um', 'selected',
    'error', 'crop_stage', 'crop_height_px', 'crop_width_px', 'pixel_count',
    'preparation_time_s', 'total_fit_time_s', 'residual_rmse', 'residual_mean',
    'residual_std', 'normalized_mse', 'residual_corr_x', 'residual_corr_y',
    'residual_coherent_mse', *INITIAL_SEARCH_FIELDS,
]

DIAGNOSTIC_FIELDS = [
    'converged', 'optimizer_status', 'iterations', 'evaluations', 'duration_s',
    'baseline_mse', 'relative_improvement', 'quality_warning',
    'boundary_parameters', 'fit_acceptable',
]
TRIAL_FIELDS += DIAGNOSTIC_FIELDS

QUALITY_FIELDS = [
    'image_index', 'image_path', 'particle_number', 'zguess_um', 'chisq',
    *DIAGNOSTIC_FIELDS
]

ERROR_FIELDS = [
    'image_index', 'image_path', 'particle_number', 'stage', 'error'
]


@dataclass
class RunResult:
    """Saved output directory, completed fits, trial records and run status."""
    output_dir: Path
    summary_rows: list[dict[str, Any]]
    trial_rows: list[dict[str, Any]]
    error_rows: list[dict[str, Any]]
    interrupted: bool = False
    quality_rows: list[dict[str, Any]] = field(default_factory=list)


def software_versions() -> dict[str, Any]:
    """Record the interpreter and installed packages without importing them."""
    packages = {}
    for name in ('holopy', 'numpy', 'scipy', 'xarray', 'matplotlib',
                 'ipywidgets', 'ipympl', 'ipython', 'lmfit'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    if packages['holopy'] in (None, 'unknown'):
        # HoloPy source installations may not report a distribution version.
        # Reuse the already-imported module without triggering another import.
        packages['holopy'] = getattr(sys.modules.get('holopy'), '__version__',
                                     packages['holopy'])
    return {
        'analysis_code_version': __version__,
        'python': sys.version,
        'platform': platform.platform(),
        'packages': packages,
    }


def input_manifest(paths: Iterable[Union[str, Path]],
                   background_path: Union[str, Path]) -> list[dict[str, Any]]:
    """Describe exact inputs cheaply; image contents are never read or hashed."""
    records = []
    for role, path in [('hologram', path)
                       for path in paths] + [('background', background_path)]:
        path = Path(path).expanduser().resolve()
        record = {'role': role, 'path': str(path)}
        try:
            stat = path.stat()
            record.update(size_bytes=stat.st_size,
                          modified_ns=stat.st_mtime_ns)
        except OSError as exc:
            record.update(size_bytes=None, modified_ns=None, error=str(exc))
        records.append(record)
    return records


def write_json(path: Union[str, Path], value: Any) -> None:
    """Replace JSON atomically, using the same-directory temporary file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value,
                  stream,
                  ensure_ascii=False,
                  indent=2,
                  default=_json_default)
        stream.write('\n')
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    """Keep NumPy scalars/arrays numeric in exported parameter dictionaries."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def serialize_intervals(result: Any) -> Any:
    """Export existing HoloPy parameter uncertainties without calculating them.

    FitResult stores a sequence of UncertainValue objects. Lightweight results
    and older objects may omit intervals; absent values remain JSON null.
    """
    intervals = getattr(result, 'intervals', None)
    if intervals is None:
        return None
    records = []
    for interval in intervals:
        record = {'name': getattr(interval, 'name', None)}
        for field_name in ('guess', 'plus', 'minus'):
            value = getattr(interval, field_name, None)
            try:
                value = float(value)
                record[field_name] = value if math.isfinite(value) else None
            except (TypeError, ValueError):
                record[field_name] = None
        records.append(record)
    return records


def write_csv(path: Union[str, Path], rows: Iterable[Mapping[str, Any]],
              fields: list[str]) -> None:
    """Replace a CSV atomically so an interrupted write cannot truncate it."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=fields,
                                extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_values(row))
    temporary.replace(path)


def _csv_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """Use the same cell encoding for checkpoints and final CSV exports."""
    return {
        key:
        json.dumps(value, ensure_ascii=False) if isinstance(
            value, (list, tuple, dict)) else value
        for key, value in row.items()
    }


def _csv_payload(rows: Iterable[Mapping[str, Any]], fields: list[str],
                 include_header: bool) -> bytes:
    """Serialize new rows before touching a checkpoint on disk."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    if include_header:
        writer.writeheader()
    writer.writerows(_csv_values(row) for row in rows)
    return stream.getvalue().encode('utf-8')


def _complete_csv_lines(stream):
    """Ignore a torn final physical line, including within a quoted CSV cell."""
    for line in stream:
        if not line.endswith('\n'):
            return
        yield line


def save_tables(result: RunResult) -> None:
    """Persist all current records, including headers when a table is empty."""
    write_csv(result.output_dir / 'summary.csv', result.summary_rows,
              SUMMARY_FIELDS)
    write_csv(result.output_dir / 'trials.csv', result.trial_rows,
              TRIAL_FIELDS)


class RunJournal:
    """Checkpoint completed rows directly in the two CSVs, without sidecar files.

    Only new rows are appended. A failed commit rolls both files back to their
    previous byte offsets, so retrying it cannot duplicate completed records.
    Final consolidation uses save_tables() and its atomic file replacements.
    CSV checkpoints require one writer; they do not provide a database-style
    transaction across files after an abrupt process or machine failure.
    """

    _tables = ('summary_rows', 'trial_rows', 'error_rows', 'quality_rows')
    _csv_tables = (('summary_rows', 'summary.csv', SUMMARY_FIELDS),
                   ('trial_rows', 'trials.csv', TRIAL_FIELDS))

    def __init__(self, result: RunResult):
        self.result = result
        self._counts = {name: 0 for name in self._tables}
        # Support resuming a recovered result without duplicating its rows.
        # Existing checkpoints are inspected only once, never per particle.
        for name, filename, fields in self._csv_tables:
            path = result.output_dir / filename
            if path.is_file() and path.stat().st_size:
                self._counts[name] = self._existing_count(
                    path, getattr(result, name), fields)

    @staticmethod
    def _existing_count(path, rows, fields):
        """Require a complete checkpoint matching the supplied in-memory rows."""
        with path.open('rb') as stream:
            stream.seek(-1, io.SEEK_END)
            if stream.read(1) != b'\n':
                raise ValueError(
                    'Incomplete CSV checkpoint; recover_run() and '
                    'save_tables() before resuming.')
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.reader(stream, strict=True)
            if next(reader, None) != fields:
                raise ValueError(f'Unexpected CSV columns in {path.name}.')
            count = 0
            try:
                for cells in reader:
                    if count >= len(rows):
                        raise ValueError('CSV checkpoint contains records '
                                         'absent from the current run.')
                    values = _csv_values(rows[count])
                    expected = [
                        '' if values.get(key) is None else str(values[key])
                        for key in fields
                    ]
                    if cells != expected:
                        raise ValueError('CSV checkpoint differs from the '
                                         'current run records.')
                    count += 1
            except csv.Error as exc:
                raise ValueError(
                    'Incomplete CSV checkpoint; recover_run() and '
                    'save_tables() before resuming.') from exc
        return count

    @staticmethod
    def _append(path, payload):
        """Append an already serialized batch and flush it before returning."""
        with path.open('ab') as stream:
            if stream.write(payload) != len(payload):
                raise OSError(f'Incomplete write to {path.name}.')
            stream.flush()

    def commit(self) -> None:
        counts = {
            name: len(getattr(self.result, name))
            for name in self._tables
        }
        if any(counts[name] < self._counts[name] for name in self._tables):
            raise ValueError(
                'Completed run records must not be removed during a run.')
        if counts == self._counts:
            return
        batches = []
        for name, filename, fields in self._csv_tables:
            path = self.result.output_dir / filename
            exists = path.exists()
            offset = path.stat().st_size if exists else 0
            rows = getattr(self.result, name)[self._counts[name]:counts[name]]
            if rows or not offset:
                batches.append((path, exists, offset,
                                _csv_payload(rows, fields, not offset)))
        attempted = []
        try:
            for path, exists, offset, payload in batches:
                attempted.append((path, exists, offset))
                self._append(path, payload)
        except BaseException:
            # Include the currently failing file: write/flush may have written
            # only part of its batch. Counters remain unchanged for a retry.
            for path, existed, offset in reversed(attempted):
                if existed:
                    with path.open('r+b') as stream:
                        stream.truncate(offset)
                elif path.exists():
                    path.unlink()
            raise
        self._counts = counts


def _read_csv_rows(path):
    """Restore numeric and boolean CSV values; leave paths and messages as text."""
    if not path.is_file():
        return []
    text_fields = {
        'image_path', 'result_file', 'status', 'error', 'stage',
        'quality_warning', 'boundary_parameters', 'ambiguity_note',
        'initial_source',
    }
    rows = []
    with path.open(newline='', encoding='utf-8', errors='replace') as stream:
        reader = csv.reader(_complete_csv_lines(stream), strict=True)
        try:
            fields = next(reader)
        except (StopIteration, csv.Error):
            return rows
        while True:
            try:
                values = next(reader)
            except (StopIteration, csv.Error):
                break
            if len(values) != len(fields):
                break
            row = dict(zip(fields, values))
            restored = {}
            for key, value in row.items():
                if not value:
                    continue
                if key in text_fields:
                    restored[key] = value
                elif value in ('True', 'False'):
                    restored[key] = value == 'True'
                else:
                    try:
                        restored[key] = json.loads(value)
                    except ValueError:
                        restored[key] = value
            rows.append(restored)
    return rows


def recover_run(output_dir: Union[str, Path]) -> RunResult:
    """Read completed CSV checkpoints or journals from older analysis versions.

    The caller can use save_tables() to regenerate the downloadable CSV files.
    Recovery is read-only and ignores incomplete .tmp files and a torn final
    CSV record, including a partially written quoted multiline field.
    """
    result = RunResult(Path(output_dir), [], [], [], interrupted=True)
    records = sorted((result.output_dir / 'records').glob('*.json'))
    if not records:
        result.summary_rows = _read_csv_rows(result.output_dir / 'summary.csv')
        result.trial_rows = _read_csv_rows(result.output_dir / 'trials.csv')
        return result
    for path in records:
        payload = json.loads(path.read_text(encoding='utf-8'))
        for name in RunJournal._tables:
            rows = payload.get(name, [])
            if not isinstance(rows, list) or any(not isinstance(row, dict)
                                                 for row in rows):
                raise ValueError(
                    f'Invalid run journal table in {path.name}: {name}')
            getattr(result, name).extend(rows)
    return result
