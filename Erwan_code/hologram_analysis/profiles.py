"""Profile compatibility and file path resolution."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any, Union
from .config import AnalysisConfig
from .preparation import normalize_particle_choice
from .selection import validate_clicked_particles
from .particle_crops import normalize_particle_crops


def resolve_path(value: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError('Choose a file or directory path.')
    path = Path(str(value).strip()).expanduser()
    return (path if path.is_absolute() else Path(base_dir) / path).resolve()


def read_profile(path: Union[str, Path]) -> dict[str, Any]:
    """Read an interface profile or convert analysis settings to the same format."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('The JSON file must contain a settings object.')
    if (not isinstance(payload.get('version'), bool)
            and payload.get('version') in (1, 2, 3, 4, 5)):
        return payload
    if 'version' in payload or not {
            'config', 'image_paths', 'background_path'
    } <= payload.keys():
        raise ValueError(
            'Choose a JSON profile saved with Save profile or an analysis settings.json file.'
        )

    image_names = payload['image_paths']
    if not isinstance(image_names, list) or not image_names or any(
            not isinstance(name, str) or not name.strip()
            for name in image_names):
        raise ValueError(
            'The analysis settings contain an invalid image list.')
    images = [resolve_path(name, path.parent) for name in image_names]
    if len({image.parent for image in images}) != 1:
        raise ValueError(
            'The notebook requires all profile images to be in the same directory.'
        )
    indices = payload.get('image_indices', [0])
    if not isinstance(indices, list) or not indices or any(
            isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < len(images) for index in indices):
        raise ValueError(
            'The analysis settings contain invalid image indices.')

    saved_choices = payload.get('particle_choices', {})
    if not isinstance(saved_choices, dict):
        raise ValueError(
            'The analysis settings contain invalid particle choices.')
    choices = {}
    for key, value in saved_choices.items():
        if not isinstance(
                key,
                str) or not key.isdecimal() or not 0 <= int(key) < len(images):
            raise ValueError(
                'A particle choice refers to an invalid image index.')
        choices[images[int(key)].name] = value

    saved_centers = payload.get('center_overrides', {})
    if not isinstance(saved_centers, dict):
        raise ValueError(
            'The analysis settings contain invalid manual centers.')
    center_overrides = {}
    for key, value in saved_centers.items():
        if not isinstance(
                key,
                str) or not key.isdecimal() or not 0 <= int(key) < len(images):
            raise ValueError(
                'A manual center refers to an invalid image index.')
        center_overrides[images[int(key)].name] = value

    saved_clicks = payload.get('clicked_particles', {})
    if not isinstance(saved_clicks, dict):
        raise ValueError('Invalid clicked particles in analysis settings.')
    clicked_particles = {}
    for key, records in saved_clicks.items():
        if not isinstance(
                key,
                str) or not key.isdecimal() or not 0 <= int(key) < len(images):
            raise ValueError(
                'Clicked particles refer to an invalid image index.')
        clicked_particles[images[int(key)].name] = validate_clicked_particles(
            records)
    background = payload['background_path']
    saved_crops = payload.get('crop_overrides', {})
    if not isinstance(saved_crops, dict):
        raise ValueError('Invalid particle crops in analysis settings.')
    crop_overrides = {}
    for key, records in saved_crops.items():
        if (not isinstance(key, str) or not key.isdecimal()
                or not 0 <= int(key) < len(images)):
            raise ValueError('Particle crops refer to an invalid image index.')
        crop_overrides[images[int(key)].name] = normalize_particle_crops(records)
    if not isinstance(background, str) or not background.strip():
        raise ValueError(
            'The analysis settings contain an invalid background path.')
    suffixes = {image.suffix for image in images}
    return {
        'version':
        1,
        'config':
        payload['config'],
        'image_folder':
        str(images[0].parent),
        'image_paths': [str(image) for image in images],
        'background_path':
        str(resolve_path(background, path.parent)),
        'output_folder':
        str(path.parent.parent),
        'analysis_folder_name':
        payload.get('analysis_folder_name', ''),
        'pattern':
        '*' + images[0].suffix if len(suffixes) == 1 else '*',
        'batch_mode':
        payload.get('selection_mode', 'saved'),
        'choices':
        choices,
        'center_overrides':
        center_overrides,
        'clicked_particles':
        clicked_particles,
        'crop_overrides': crop_overrides,
        **({
            'input_manifest': payload['input_manifest']
        } if 'input_manifest' in payload else {}),
        'current_image':
        images[indices[0]].name,
    }


def _readable_file(path, label):
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f'{label} is missing or unreadable: {path}')


def _manual_centers(values, image_names):
    """Validate nested candidate centers before applying any notebook settings."""
    if not isinstance(values, dict):
        raise ValueError('Invalid manual centers in the profile.')
    normalized = {}
    for image_name, candidates in values.items():
        if image_name not in image_names or not isinstance(candidates, dict):
            raise ValueError(
                'A manual center refers to an unknown image or invalid candidate map.'
            )
        centers = {}
        for number, center in candidates.items():
            if (isinstance(number, bool) or not isinstance(number, (str, int))
                    or not str(number).isdecimal() or int(number) < 1):
                raise ValueError(
                    'Manual center candidate numbers must be positive integers.'
                )
            number = str(int(number))
            if number in centers:
                raise ValueError('Duplicate manual center candidate number.')
            if not isinstance(center, dict) or set(center) != {'x', 'y'}:
                raise ValueError(
                    'Each manual center must contain x and y pixel coordinates.'
                )
            coordinates = {}
            for axis in ('x', 'y'):
                value = center[axis]
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or value < 0
                        or value != int(value)):
                    raise ValueError(
                        'Manual x/y centers must be finite, nonnegative integer pixels.'
                    )
                coordinates[axis] = int(value)
            centers[number] = coordinates
        normalized[image_name] = centers
    return normalized


def validate_profile(payload: dict[str, Any],
                     base_dir: Union[str, Path]) -> dict[str, Any]:
    """Return a normalized, accessible profile before mutating interface state.

    This validates paths and nested selections without reading image contents or
    launching detection. Pixel bounds are checked later against the loaded image.
    Profiles with an exact image list keep its order; legacy profiles use a sorted
    glob excluding the background image.
    """
    if not isinstance(payload, dict):
        raise ValueError('The profile must contain a settings object.')
    config = payload.get('config')
    if not isinstance(config, dict):
        raise ValueError('The profile contains an invalid configuration.')
    cfg = AnalysisConfig.from_dict(config)
    mode = payload.get('batch_mode', 'saved')
    if mode not in ('saved', 'auto', 'all'):
        raise ValueError('Invalid series mode in the profile.')
    pattern = payload.get('pattern', '*.tif')
    if not isinstance(pattern, str) or not pattern.strip() or Path(
            pattern).is_absolute():
        raise ValueError('The profile contains an invalid image pattern.')
    background = resolve_path(payload.get('background_path', ''), base_dir)
    _readable_file(background, 'Background image')
    exact_paths = payload.get('image_paths')
    if exact_paths is not None:
        if (not isinstance(exact_paths, list) or not exact_paths
                or any(not isinstance(path, str) or not path.strip()
                       for path in exact_paths)):
            raise ValueError(
                'The profile contains an invalid exact image list.')
        images = [resolve_path(path, base_dir) for path in exact_paths]
        folder = resolve_path(
            payload.get('image_folder', str(images[0].parent)), base_dir)
    else:
        folder = resolve_path(payload.get('image_folder', ''), base_dir)
        if not folder.is_dir() or not os.access(folder, os.R_OK):
            raise ValueError(
                f'Image directory is missing or unreadable: {folder}')
        images = sorted(path.resolve() for path in folder.glob(pattern)
                        if path.is_file())
        images = [path for path in images if path != background]
    if not images:
        raise ValueError('No hologram images match this profile.')
    if len(set(images)) != len(images):
        raise ValueError('The profile contains duplicate image paths.')
    if any(image.parent != folder for image in images):
        raise ValueError(
            'All profile images must be in the saved image directory.')
    for image in images:
        if image == background:
            raise ValueError(
                'The background image cannot also be a hologram input.')
        _readable_file(image, 'Hologram image')
    if 'input_manifest' in payload:
        _validate_input_manifest(payload['input_manifest'], images, background,
                                 base_dir)
    output = resolve_path(payload.get('output_folder', 'results_auto_multi_z'),
                          base_dir)
    if output.exists() and not output.is_dir():
        raise ValueError(f'Output directory refers to a file: {output}')
    existing_parent = next(
        (path for path in (output, *output.parents) if path.exists()), None)
    if existing_parent is None or not os.access(existing_parent, os.W_OK):
        raise ValueError(f'Output directory is not writable: {output}')
    names = {image.name for image in images}
    choices = payload.get('choices', {})
    if not isinstance(choices, dict) or any(name not in names
                                            for name in choices):
        raise ValueError('A saved particle choice refers to an unknown image.')
    choices = {
        name: normalize_particle_choice(value)
        for name, value in choices.items()
    }
    saved_clicks = payload.get('clicked_particles', {})
    if not isinstance(saved_clicks, dict) or any(name not in names
                                                 for name in saved_clicks):
        raise ValueError('Clicked particles refer to an unknown image.')
    clicked_particles = {
        name: validate_clicked_particles(records)
        for name, records in saved_clicks.items()
    }
    if cfg.click_to_detect:
        for name, choice in choices.items():
            numbers = [choice
                       ] if isinstance(choice, int) and choice > 0 else choice
            if isinstance(numbers, list) and any(
                    n > len(clicked_particles.get(name, [])) for n in numbers):
                raise ValueError(
                    'A clicked particle choice is not present in the saved centers.'
                )
    centers = _manual_centers(payload.get('center_overrides', {}), names)
    saved_crops = payload.get('crop_overrides', {})
    if not isinstance(saved_crops, dict) or any(name not in names for name in saved_crops):
        raise ValueError('Particle crops refer to an unknown image.')
    crops = {name: normalize_particle_crops(records)
             for name, records in saved_crops.items()}
    if cfg.click_to_detect and any(
            int(number) > len(clicked_particles.get(name, []))
            for name, records in crops.items() for number in records):
        raise ValueError('A particle crop refers to a clicked particle that was not saved.')
    if cfg.click_to_detect and any(
            int(number) > len(clicked_particles.get(name, []))
            for name, edits in centers.items() for number in edits):
        raise ValueError(
            'A manual edit refers to a clicked particle that was not saved.')
    current = payload.get('current_image')
    if current is not None and current not in names:
        raise ValueError(
            'The current image is missing from the saved image list.')
    return {
        **payload,
        'config': asdict(cfg),
        'batch_mode': mode,
        'pattern': pattern,
        'image_folder': str(folder),
        'image_paths': [str(image) for image in images],
        'background_path': str(background),
        'output_folder': str(output),
        'choices': choices,
        'center_overrides': centers,
        'clicked_particles': clicked_particles,
        'crop_overrides': crops,
        'current_image': current or images[0].name,
    }


def _validate_input_manifest(records, images, background, base_dir):
    """Prevent saved candidate numbers being applied to replaced input images."""
    if not isinstance(records, list):
        raise ValueError('The saved input information must be a list.')
    expected = {('hologram', image)
                for image in images} | {('background', background)}
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                'The saved input information contains an invalid record.')
        key = (record.get('role'),
               resolve_path(record.get('path', ''), base_dir))
        if key not in expected or key in seen:
            raise ValueError(
                'The saved input information refers to duplicate or unknown files.'
            )
        seen.add(key)
        size, modified = record.get('size_bytes'), record.get('modified_ns')
        if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                or isinstance(modified, bool)
                or not isinstance(modified, int)):
            raise ValueError(
                'The saved input size or modification time is invalid.')
        stat = key[1].stat()
        if (size, modified) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError(
                f'Input changed since these selections were saved: {key[1].name}. '
                'Load the images and select the particles again before saving a new profile.'
            )
    if seen != expected:
        raise ValueError('The saved input information is incomplete.')
