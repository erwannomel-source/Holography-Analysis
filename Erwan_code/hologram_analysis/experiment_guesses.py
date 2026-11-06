"""Small, persistent collections of useful starts for one experiment.

These are starting points, never fixed parameters or reference measurements.
Each new hologram must screen and freely refit them. Distinct depth/index-sign
branches are retained separately, rather than averaging ambiguous solutions.
Only ordinary JSON numbers/strings are saved; loading never executes code.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from threading import RLock


SCHEMA_VERSION = 1
POLICY_VERSION = 'mielens-experiment-starts-v1'
MAX_EXPERIMENTS = 32
MAX_SEEDS = 24
MAX_FILE_BYTES = 4 * 1024 * 1024
MIN_TRAINING_IMPROVEMENT = .5


def _number(value):
    """Accept scalar finite numbers, without turning booleans into indices."""
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _bounds(cfg):
    result = {}
    for name in ('n_bounds', 'r_bounds', 'z_bounds'):
        try:
            values = getattr(cfg, name)
            if len(values) != 2:
                return None
            lower, upper = map(_number, values)
        except (TypeError, AttributeError):
            return None
        if lower is None or upper is None or lower >= upper:
            return None
        if name != 'z_bounds' and lower <= 0:
            return None
        result[name] = (lower, upper)
    medium = _number(getattr(cfg, 'medium_index', None))
    return result if medium is not None and medium > 0 else None


def _interior(record, bounds):
    for name, bound_name in (('n', 'n_bounds'), ('r_um', 'r_bounds'),
                             ('z_um', 'z_bounds')):
        value = record[name]
        lower, upper = bounds[bound_name]
        # Match the fit's near-boundary warning, including old saved records.
        margin = max((upper - lower) * 1e-3, 1e-12)
        if not lower + margin < value < upper - margin:
            return False
    return True


def _fingerprint(path):
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
        identity = [stat.st_size, stat.st_mtime_ns]
    except OSError:
        identity = None
    digest = hashlib.sha256()
    try:
        with resolved.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        content = digest.hexdigest()
    except OSError:
        content = None
    return {'path': str(resolved), 'size_mtime_ns': identity,
            'content_sha256': content}


def experiment_key(image_path, bg_path, cfg):
    """Scope starts by sample folder/name, optics, bounds and background.

    Changing a crop, image number, center or search budget does not reset an
    experiment. A changed background or physical setting does. Explicit names
    may group sample folders; users should give different conditions distinct
    names. No image pixels or particle coordinates enter the saved library.
    """
    bounds = _bounds(cfg)
    if bounds is None:
        raise ValueError('Experiment guesses require finite physical bounds.')
    name = str(getattr(cfg, 'experiment_name', '') or '').strip()
    scope = ({'name': name} if name else
             {'folder': str(Path(image_path).expanduser().resolve().parent)})
    optics = {}
    for field in ('spacing', 'medium_index', 'illum_wavelen', 'NA', 'noil'):
        value = _number(getattr(cfg, field, None))
        if value is None or value <= 0:
            raise ValueError('Experiment guesses require finite optical settings.')
        optics[field] = value
    try:
        polarization = list(map(_number, cfg.illum_polarization))
    except (AttributeError, TypeError):
        polarization = []
    if (len(polarization) != 2 or None in polarization
            or not any(polarization)):
        raise ValueError('Experiment guesses require a finite polarization.')
    optics['illum_polarization'] = polarization
    payload = {'schema_version': SCHEMA_VERSION, 'policy': POLICY_VERSION,
               'theory': 'HoloPy.MieLens(asin(NA/noil))',
               'normalization': 'image/background; global-mean-one-v1',
               'scope': scope, 'optics': optics, 'bounds': bounds,
               'background': _fingerprint(bg_path)}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _record(value):
    """Validate a record from JSON or optimizer output before storing it."""
    if not isinstance(value, dict):
        return None
    result = {}
    for field in ('n', 'r_um', 'z_um', 'relative_improvement', 'training_score'):
        number = _number(value.get(field))
        if number is None:
            return None
        result[field] = number
    if (result['n'] <= 0 or result['r_um'] <= 0
            or not MIN_TRAINING_IMPROVEMENT <= result['relative_improvement'] <= 1
            or result['training_score'] < 0):
        return None
    sequence = value.get('sequence', 0)
    if (isinstance(sequence, bool) or not isinstance(sequence, int)
            or not 0 <= sequence <= 10 ** 18):
        return None
    source = value.get('source', '')
    if not isinstance(source, str):
        return None
    result.update(sequence=sequence, source=source[:1024])
    return result


def _branch(record, medium):
    def sign(value):
        return (value > 0) - (value < 0)
    return sign(record['n'] - medium), sign(record['z_um'])


def _quality(record):
    # Improvement is normalized by each image's own variance, unlike raw MSE.
    return record['relative_improvement'], record['sequence']


def _same_start(left, right, medium):
    if _branch(left, medium) != _branch(right, medium):
        return False
    return (abs(left['n'] - right['n']) <= max(
                1e-6, abs(left['n'] - medium) * .01)
            and abs(left['r_um'] - right['r_um']) <= max(.01, left['r_um'] * .01)
            and abs(left['z_um'] - right['z_um']) <= max(.05, abs(left['z_um']) * .01))


def _distance(left, right, medium):
    if _branch(left, medium) != _branch(right, medium):
        return 10.
    dn = abs(left['n'] - right['n']) / max(
        .00005, abs(left['n'] - medium) * .2,
        abs(right['n'] - medium) * .2)
    dr = abs(math.log(left['r_um'] / right['r_um'])) / .25
    dz = abs(left['z_um'] - right['z_um']) / max(
        1., abs(left['z_um']) * .25, abs(right['z_um']) * .25)
    return min(math.sqrt(dn * dn + dr * dr + dz * dz), 10.)


def _retain(records, medium):
    """Keep the best of every sign branch, then cover radius/depth diversity."""
    ranked = sorted(records, key=_quality, reverse=True)
    if len(ranked) <= MAX_SEEDS:
        return ranked
    kept = []
    branches = set()
    for record in ranked:
        branch = _branch(record, medium)
        if branch not in branches:
            kept.append(record)
            branches.add(branch)
    remaining = [record for record in ranked if record not in kept]
    while remaining and len(kept) < MAX_SEEDS:
        def priority(record):
            diversity = min(_distance(record, other, medium) for other in kept)
            return diversity * record['relative_improvement'], _quality(record)
        chosen = max(remaining, key=priority)
        kept.append(chosen)
        remaining.remove(chosen)
    return sorted(kept, key=_quality, reverse=True)


class ExperimentGuessLibrary:
    """Bounded seed memory with optional, explicit atomic JSON persistence.

    Construction only reads. I/O errors leave usable in-memory seeds and are
    exposed through ``warning``; they do not abort an image analysis.
    """

    def __init__(self, path=None):
        self.path = None if path is None else Path(path).expanduser().resolve()
        self.warning = ''
        self._experiments = OrderedDict()
        self._sequence = 0
        self._dirty = False
        self._lock = RLock()
        if self.path is not None:
            self._load()

    def _load(self):
        try:
            if not self.path.exists():
                return
            if self.path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('file is too large')
            with self.path.open(encoding='utf-8') as stream:
                payload = json.load(stream)
            if (not isinstance(payload, dict)
                    or payload.get('schema_version') != SCHEMA_VERSION
                    or payload.get('policy') != POLICY_VERSION
                    or not isinstance(payload.get('experiments'), list)):
                raise ValueError('incompatible library format')
            experiments = payload['experiments']
            if len(experiments) > MAX_EXPERIMENTS:
                raise ValueError('too many experiment scopes')
            loaded = OrderedDict()
            invalid_records = False
            for experiment in experiments:
                if (not isinstance(experiment, dict)
                        or not self._valid_key(experiment.get('key'))
                        or not isinstance(experiment.get('records'), list)
                        or len(experiment['records']) > MAX_SEEDS
                        or experiment['key'] in loaded):
                    raise ValueError('invalid experiment structure')
                records = []
                for value in experiment['records']:
                    record = _record(value)
                    if record is None:
                        invalid_records = True
                    else:
                        records.append(record)
                if records:
                    loaded[experiment['key']] = records
            self._experiments = loaded
            self._sequence = max((record['sequence']
                                  for records in loaded.values()
                                  for record in records), default=0)
            if invalid_records:
                self.warning = 'Invalid saved experiment starts were ignored.'
        except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
            self.warning = f'Experiment start library was ignored: {exc}.'

    @staticmethod
    def _valid_key(key):
        return isinstance(key, str) and 0 < len(key) <= 512

    def seeds(self, key, cfg):
        """Return independent starting-point dictionaries, never x/y or priors."""
        bounds = _bounds(cfg)
        if not self._valid_key(key) or bounds is None:
            return []
        with self._lock:
            records = self._experiments.get(key, ())
            if records:
                self._experiments.move_to_end(key)
            return [dict(nguess=record['n'], rguess_um=record['r_um'],
                         zguess_um=record['z_um'],
                         relative_improvement=record['relative_improvement'],
                         training_score=record['training_score'],
                         source=record['source'])
                    for record in records if _interior(record, bounds)]

    def observe(self, key, trials, cfg, source=None):
        """Learn well-screened, converged, interior fits; return change count.

        Statuses 6--8 (numerical precision stopping) and unknown convergence are
        deliberately excluded from training even if accepted for display.
        """
        bounds = _bounds(cfg)
        if not self._valid_key(key) or bounds is None:
            return 0
        changed = 0
        with self._lock:
            records = list(self._experiments.get(key, ()))
            for trial in trials:
                if (not isinstance(trial, dict) or trial.get('converged') is not True
                        or trial.get('fit_acceptable') is not True
                        or trial.get('boundary_parameters')):
                    continue
                status = trial.get('optimizer_status')
                if (isinstance(status, bool)
                        or _number(status) not in (1., 2., 3., 4.)):
                    continue
                record = _record({
                    'n': trial.get('n'), 'r_um': trial.get('r_um'),
                    'z_um': trial.get('z_um'),
                    'relative_improvement': trial.get('relative_improvement'),
                    'training_score': trial.get('chisq'),
                    'source': str(source or trial.get('source') or '')[:1024],
                    'sequence': self._sequence + 1})
                if record is None or not _interior(record, bounds):
                    continue
                self._sequence += 1
                duplicate = next((old for old in records
                                  if _same_start(old, record, cfg.medium_index)), None)
                if duplicate is not None:
                    if _quality(record) <= _quality(duplicate):
                        continue
                    records.remove(duplicate)
                records.append(record)
                changed += 1
            if changed:
                self._experiments[key] = _retain(records, cfg.medium_index)
                self._experiments.move_to_end(key)
                while len(self._experiments) > MAX_EXPERIMENTS:
                    self._experiments.popitem(last=False)
                self._dirty = True
        return changed

    def flush(self):
        """Atomically write changed seeds when a persistence path was supplied."""
        with self._lock:
            if self.path is None or not self._dirty:
                return False
            temporary_path = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                payload = {'schema_version': SCHEMA_VERSION,
                           'policy': POLICY_VERSION,
                           'experiments': [dict(key=key, records=records)
                                           for key, records in self._experiments.items()]}
                with tempfile.NamedTemporaryFile(
                        mode='w', encoding='utf-8', prefix=self.path.name + '.',
                        suffix='.tmp', dir=self.path.parent, delete=False) as stream:
                    temporary_path = Path(stream.name)
                    json.dump(payload, stream, allow_nan=False, indent=2)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.path)
                self._dirty = False
                self.warning = ''
                return True
            except (OSError, ValueError, TypeError) as exc:
                self.warning = f'Experiment starts remain in memory; saving failed: {exc}.'
                return False
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass


_SHARED_LIBRARIES = {}
_SHARED_LOCK = RLock()


def get_guess_library(path=None):
    """Reuse memory across separate image-analysis calls within this process."""
    identity = None if path is None else str(Path(path).expanduser().resolve())
    with _SHARED_LOCK:
        if identity not in _SHARED_LIBRARIES:
            _SHARED_LIBRARIES[identity] = ExperimentGuessLibrary(identity)
        return _SHARED_LIBRARIES[identity]
