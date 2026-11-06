"""Run orchestration, cancellation and completed-result persistence."""
from __future__ import annotations

import pickle
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
import numpy as np
import holopy as hp
from .config import AnalysisConfig, AnalysisCancelled, check_cancelled
from .preparation import load_and_detect, normalize_particle_choice, select_candidates, prepare_particle, apply_center_override
from .fitting import fit_particle_multi_z
from .crop_validation import fit_prepared_crop
from .particle_crops import normalize_particle_crops, particle_crop_config
from .preparation_cache import PreparationSnapshot
from .experiment_guesses import get_guess_library, experiment_key
from .plotting import show_detection_maps, show_crop_control, show_fit_comparison
from .outputs import (RunResult, RunJournal, save_tables, software_versions,
                      input_manifest, write_json, DIAGNOSTIC_FIELDS,
                      INITIAL_SEARCH_FIELDS)

RESIDUAL_FIELDS = (
    'residual_rmse',
    'residual_mean',
    'residual_std',
    'normalized_mse',
    'residual_corr_x',
    'residual_corr_y',
    'residual_coherent_mse',
)


def _record_particle_costs(trials, preparation_time, fallback_fit_time=0.0):
    """Include coarse search and every local attempt in the recorded fit time."""
    durations = [
        float(trial['duration_s']) for trial in trials
        if trial.get('duration_s') is not None
        and np.isfinite(float(trial['duration_s']))
    ]
    # Production trials all include timing. Compatible custom fit functions
    # may omit it; retain the measured elapsed time rather than report zero.
    # Each local trial describes the same coarse search; count it once.
    search_time = max((float(trial.get('search_duration_s') or 0.0)
                       for trial in trials), default=0.0)
    total_time = (sum(durations) + search_time
                  if len(durations) == len(trials) else fallback_fit_time)
    for trial in trials:
        trial.update(preparation_time_s=preparation_time,
                     total_fit_time_s=total_time)
        if trial.get('crop_height_px') and trial.get('crop_width_px'):
            trial['pixel_count'] = (trial['crop_height_px'] *
                                    trial['crop_width_px'])
    return total_time


def analyse_images(image_paths,
                   bg_path,
                   cfg,
                   output_root,
                   image_indices=None,
                   mode='saved',
                   choices=None,
                   center_overrides=None,
                   on_progress=None,
                   on_trial=None,
                   cancel_event=None,
                   log=print,
                   on_figure=None,
                   log_trial_details=True,
                   clicked_particles=None,
                   on_particle_figure=None,
                   crop_overrides=None,
                   run_name=None,
                   preparation_cache=None):
    """Analyse images and save completed results after each candidate.

    mode: 'auto' = target area, 'all' = all candidates, 'saved' = per-image choices then auto.
    choices: image index -> candidate number(s), None for auto or -1 for all.
    center_overrides: image index -> candidate number -> manual x/y center in pixels.
    An empty list skips that image when using saved choices.
    cancel_event: threading.Event shared with the notebook's Stop buttons.
    log/on_figure: optional callbacks for thread-safe notebook output.
    on_particle_figure: optional PNG callback with image index and particle number;
        when supplied, particle figures use it instead of the bytes-only on_figure.
    crop_overrides: image index -> candidate number -> individual crop settings.
    log_trial_details: include optimizer diagnostics and selected-depth text in the log.
    preparation_cache: optional detached preview snapshot, revalidated before reuse.
    """
    cfg.validate()
    if preparation_cache is not None and not isinstance(
            preparation_cache, PreparationSnapshot):
        raise ValueError('Preparation cache must be a validated preview snapshot.')
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError('No holograms loaded.')
    indices = list(range(
        len(paths))) if image_indices is None else list(image_indices)
    if (not indices or len(set(indices)) != len(indices) or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer)) or not 0 <= i < len(paths)
            for i in indices)):
        raise ValueError('Invalid image selection.')
    if mode not in ('auto', 'all', 'saved'):
        raise ValueError('Unknown selection mode.')
    choices = {} if choices is None else {
        key: normalize_particle_choice(value)
        for key, value in choices.items()
    }
    center_overrides = {} if center_overrides is None else center_overrides
    crop_overrides = {} if crop_overrides is None else crop_overrides
    if not isinstance(crop_overrides, dict):
        raise ValueError('Particle crops must be indexed by image number.')
    normalized_crops = {}
    for index, crops in crop_overrides.items():
        key = str(index)
        if (isinstance(index, (bool, np.bool_)) or not key.isdecimal()
                or not 0 <= int(key) < len(paths) or str(int(key)) in normalized_crops):
            raise ValueError('Particle crops refer to an invalid or duplicate image index.')
        normalized_crops[str(int(key))] = normalize_particle_crops(crops)
    crop_overrides = normalized_crops
    from .selection import validate_clicked_particles
    clicked_particles = {} if clicked_particles is None else clicked_particles
    if not isinstance(clicked_particles, dict):
        raise ValueError("Clicked particles must be indexed by image number.")
    clicked_particles = {
        str(k): validate_clicked_particles(v)
        for k, v in clicked_particles.items()
    }
    if any(not k.isdecimal() or not 0 <= int(k) < len(paths)
           for k in clicked_particles):
        raise ValueError("Clicked particles refer to an invalid image index.")
    if cfg.click_to_detect and mode == 'saved' and any(
            int(number) > len(clicked_particles.get(index, []))
            for index, records in crop_overrides.items() for number in records):
        raise ValueError('A particle crop refers to a clicked particle that was not saved.')
    if not isinstance(center_overrides, dict):
        raise ValueError('Manual center overrides must be a dictionary.')
    if mode != 'saved' and (cfg.click_to_detect or
                            (mode == 'all' and cfg.detect_search_circle_only)):
        # Automatic/full-image IDs are not the click/circle candidate IDs.
        # Applying edits from that other numbering would move the wrong particle.
        if center_overrides:
            log('Automatic mode uses newly detected candidate numbers; '
                'saved click/circle centre edits are not applied.')
        center_overrides = {}
        if crop_overrides:
            log('Automatic mode uses newly detected candidate numbers; '
                'saved click/circle crop choices are not applied.')
        crop_overrides = {}
    bg_path = Path(bg_path)
    if not bg_path.is_file():
        raise FileNotFoundError(f'Background image not found : {bg_path}')
    output_root = Path(output_root).expanduser().resolve()
    requested_name = str(run_name or '').strip()
    if requested_name:
        forbidden = '<>:"/\\|?*'
        if (requested_name in ('.', '..')
                or any(char in requested_name for char in forbidden)
                or any(not char.isprintable() for char in requested_name)):
            raise ValueError(
                'Analysis folder name must be one simple folder name without '
                'slashes or special file-name characters.')
        run_dir = output_root / requested_name
        suffix = 2
        while run_dir.exists():
            run_dir = output_root / f'{requested_name}_{suffix}'
            suffix += 1
    else:
        run_dir = output_root / f'analysis_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run_dir.mkdir(parents=True, exist_ok=False)
    guess_library = None
    guess_keys = {}
    if cfg.auto_initial_guess and cfg.reuse_experiment_guesses:
        guess_library = get_guess_library(
            output_root / 'experiment_guesses.json'
            if cfg.persist_experiment_guesses else None)
        if guess_library.warning:
            log(f'Experiment profile: {guess_library.warning}')
    result = RunResult(run_dir, [], [], [])
    settings = {
        'config': asdict(cfg),
        'image_paths': [str(p.resolve()) for p in paths],
        'background_path': str(bg_path.resolve()),
        'image_indices': [int(i) for i in indices],
        'selection_mode': mode,
        'analysis_folder_name': run_dir.name,
        'particle_choices': {
            str(k): v
            for k, v in choices.items()
        },
        'center_overrides': {
            str(image_index): {
                str(particle_number): override
                for particle_number, override in image_overrides.items()
            }
            for image_index, image_overrides in center_overrides.items()
        },
        'clicked_particles': clicked_particles,
        'crop_overrides': crop_overrides,
        'software': software_versions(),
        'input_manifest': input_manifest(paths, bg_path),
        'experiment_guess_profile': (str(guess_library.path)
                                     if guess_library is not None
                                     and guess_library.path is not None else None),
    }
    write_json(run_dir / 'settings.json', settings)
    save_tables(result)
    journal = RunJournal(result)
    log(f'Output folder: {run_dir}')
    # Delay loading the background until inside the cancellation-aware loop.
    bg = None
    try:
        for image_step, image_index in enumerate(indices, 1):
            info = {
                'image_index': int(image_index),
                'image_path': str(paths[image_index].resolve())
            }
            try:
                check_cancelled(cancel_event)
                if on_progress:
                    on_progress(
                        image_step - 1, len(indices),
                        f'Image {image_step}/{len(indices)} : {paths[image_index].name}'
                    )
                log(f'\nImage {image_index}: {paths[image_index].name}')
                explicit_clicks = clicked_particles.get(str(image_index), [])
                if cfg.click_to_detect and mode == 'saved' and not explicit_clicks:
                    log('No clicked particles saved for this image; skipping it.'
                        )
                    continue
                detection_cfg = cfg if mode == 'saved' else replace(
                    cfg,
                    click_to_detect=False,
                    detect_search_circle_only=cfg.detect_search_circle_only
                    if mode == 'auto' else False)
                loaded, preview_fingerprint = (None, None)
                if preparation_cache is not None:
                    loaded, preview_fingerprint = preparation_cache.image(
                        paths[image_index], bg_path, detection_cfg)
                if loaded is None:
                    if bg is None:
                        bg = hp.load_image(str(bg_path), spacing=cfg.spacing)
                    check_cancelled(cancel_event)
                    loaded = load_and_detect(paths[image_index], bg, detection_cfg)
                data_holo, image, residual_image, activity, candidates = loaded
                cfg = cfg.centered_for_image(image.shape)
                if cfg.click_to_detect and mode == 'saved':
                    from .selection import clicked_activity
                    candidates = validate_clicked_particles(
                        explicit_clicks, image.shape)
                    activity = clicked_activity(residual_image, candidates,
                                                cfg)
                check_cancelled(cancel_event)
                choice = choices.get(
                    int(image_index)) if mode == 'saved' else None
                if cfg.click_to_detect and mode == 'saved' and choice is None:
                    choice = list(range(1, len(candidates) + 1))
                selected = select_candidates(
                    candidates,
                    cfg,
                    particle_number=None if choice == -1 else choice,
                    fit_all=mode == 'all' or choice == -1,
                    log=log) if candidates else []
                image_overrides = center_overrides.get(
                    image_index, center_overrides.get(str(image_index), {}))
                if cfg.show_detection_plots:
                    show_detection_maps(
                        image,
                        activity,
                        candidates,
                        paths[image_index].name,
                        cfg,
                        on_figure=on_figure,
                        log=log,
                        selected_numbers=[number for number, _ in selected],
                        selected_only=True,
                        center_overrides=image_overrides if isinstance(
                            image_overrides, dict) else {})
                if not candidates:
                    log('No candidates detected.')
                    continue
                if not selected:
                    log('No particles selected; skipping this image.')
                for particle_number, candidate in selected:
                    check_cancelled(cancel_event)
                    log(f'Candidate {particle_number}')
                    preparation_time = 0.0
                    fitting_started = None
                    try:
                        if not isinstance(image_overrides, dict):
                            raise ValueError(
                                'Manual centers for an image must be a dictionary.'
                            )
                        override = image_overrides.get(
                            particle_number,
                            image_overrides.get(str(particle_number)))
                        candidate = apply_center_override(
                            candidate, override, image.shape)
                        particle_cfg = particle_crop_config(cfg, crop_overrides.get(
                            str(image_index), {}).get(str(particle_number)))
                        preparation_started = perf_counter()
                        neighbors = [
                            apply_center_override(
                                c, image_overrides.get(i, image_overrides.get(str(i))),
                                image.shape)
                            for i, c in enumerate(candidates, 1)
                            if i != particle_number]
                        prepared = None
                        if preparation_cache is not None:
                            prepared = preparation_cache.particle(
                                preview_fingerprint, particle_number, candidate,
                                particle_cfg, neighbors)
                        if prepared is None:
                            prepared = prepare_particle(
                                data_holo, image, residual_image, candidate,
                                particle_cfg, log=log, neighbors=neighbors)
                        h, x_center, y_center, crop_shape, xguess, yguess = prepared
                        preparation_time = perf_counter() - preparation_started
                        check_cancelled(cancel_event)
                        log(f'Center x={x_center}, y={y_center} px ; '
                            f'crop={crop_shape[0]} × {crop_shape[1]} px ; '
                            f'xguess={xguess:.3f}, yguess={yguess:.3f} µm')
                        fitting_started = perf_counter()
                        learned_options = {}
                        if guess_library is not None:
                            scope = str(paths[image_index].parent.resolve())
                            if scope not in guess_keys:
                                guess_keys[scope] = experiment_key(
                                    paths[image_index], bg_path, cfg)
                            learned_options = dict(guess_library=guess_library,
                                                   experiment_key=guess_keys[scope],
                                                   guess_source=f'{paths[image_index].resolve()}#particle{particle_number}')
                        h, best_result, best_trial, trials = fit_prepared_crop(
                            h,
                            data_holo,
                            xguess,
                            yguess,
                            particle_cfg,
                            fit_particle_multi_z,
                            on_trial=on_trial,
                            log_trial_details=log_trial_details,
                            cancel_event=cancel_event,
                            log=log,
                            **learned_options)
                        crop_shape = (h.sizes['x'], h.sizes['y'])
                        crop_info = h.attrs.get('crop_selection', {})
                        total_fit_time = _record_particle_costs(
                            trials, preparation_time,
                            perf_counter() - fitting_started)
                        result.trial_rows.extend(
                            dict(
                                info, particle_number=particle_number, **trial)
                            for trial in trials)
                        if best_result is None:
                            if any(trial.get('converged') is True
                                   for trial in trials):
                                reason = (
                                    'converged trials did not improve on a '
                                    'uniform image by at least 10%.')
                            else:
                                reason = 'no valid trial converged.'
                            raise RuntimeError(
                                f'No satisfactory fit: {reason} '
                                'All attempted fits are retained in trials.csv.')
                        fitted_parameters = {
                            'n': float(best_trial['n']),
                            'r': float(best_trial['r_um']),
                            'center.0': float(best_trial['x_um']),
                            'center.1': float(best_trial['y_um']),
                            'center.2': float(best_trial['z_um']),
                        }
                        log(
                            f'Fitted parameters — {paths[image_index].name}, '
                            f'candidate {particle_number}:\n'
                            '{\n'
                            f"  'n': {fitted_parameters['n']!r},\n"
                            f"  'r': {fitted_parameters['r']!r},\n"
                            f"  'center.0': {fitted_parameters['center.0']!r},\n"
                            f"  'center.1': {fitted_parameters['center.1']!r},\n"
                            f"  'center.2': {fitted_parameters['center.2']!r}\n"
                            '}')
                        result_path = run_dir / f'results_{image_index:04d}_{paths[image_index].stem}_p{particle_number:03d}.pkl'
                        with open(result_path, 'wb') as stream:
                            pickle.dump(best_result.parameters, stream)
                        result.summary_rows.append(
                            dict(info,
                                 particle_number=particle_number,
                                 x_px=x_center,
                                 y_px=y_center,
                                 crop_height_px=crop_shape[0],
                                 crop_width_px=crop_shape[1],
                                 crop_method=crop_info.get('method'),
                                 crop_warning=crop_info.get('warning'),
                                 crop_bounds_px=crop_info.get('bounds_px'),
                                 extent_bounds_px=crop_info.get(
                                     'extent_bounds_px'),
                                 extent_radius_px=crop_info.get(
                                     'extent_radius_px'),
                                 extent_confidence=crop_info.get(
                                     'extent_confidence'),
                                 extent_reason=crop_info.get('extent_reason'),
                                 optimization_pixel_reduction=crop_info.get(
                                     'optimization_pixel_reduction'),
                                 pixel_count=crop_shape[0] * crop_shape[1],
                                 converged=best_trial.get('converged'),
                                 total_fit_time_s=total_fit_time,
                                 preparation_time_s=preparation_time,
                                 solution_ambiguous=crop_info.get(
                                     'solution_ambiguous', False),
                                 near_best_count=crop_info.get(
                                     'near_best_count', 1),
                                 ambiguity_note=crop_info.get(
                                     'ambiguity_note', ''),
                                 near_best_ranges=crop_info.get(
                                     'near_best_ranges', {}),
                                 xguess_um=xguess,
                                 yguess_um=yguess,
                                 rguess_um=best_trial.get('rguess_um'),
                                 zguess_um=best_trial['zguess_um'],
                                 x_um=best_trial['x_um'],
                                 y_um=best_trial['y_um'],
                                 z_um=best_trial['z_um'],
                                 r_um=best_trial['r_um'],
                                 n=best_trial['n'],
                                 chisq=best_trial['chisq'],
                                 result_file=str(result_path),
                                 **{
                                     name: best_trial.get(name)
                                     for name in (
                                         *INITIAL_SEARCH_FIELDS,
                                         'fit_acceptable', 'baseline_mse',
                                         'relative_improvement',
                                         'quality_warning',
                                         'boundary_parameters')
                                 },
                                 **{
                                     name:
                                     best_trial.get(name, crop_info.get(name))
                                     for name in RESIDUAL_FIELDS
                                 }))
                        result.quality_rows.append(
                            dict(info,
                                 particle_number=particle_number,
                                 zguess_um=best_trial['zguess_um'],
                                 chisq=best_trial['chisq'],
                                 **{
                                     name: best_trial.get(name)
                                     for name in DIAGNOSTIC_FIELDS
                                 }))
                        if log_trial_details:
                            log(f'Selected: rguess={best_trial.get("rguess_um")} µm, zguess={best_trial["zguess_um"]:g} µm, z={best_trial["z_um"]:.3f} µm, '
                                f'chi²={best_trial["chisq"]:.6g}')
                        improvement = best_trial.get('relative_improvement')
                        if improvement is not None and np.isfinite(improvement):
                            log('Fit quality: '
                                f'{100 * improvement:.1f}% less squared error '
                                'than a uniform image. The 10% acceptance '
                                'threshold is a screening heuristic, not a '
                                'guarantee of physical accuracy.')
                        if best_trial.get('quality_warning'):
                            log(f'Fit warning: {best_trial["quality_warning"]}')
                        # Figure display and PNG export are independent. Skip
                        # rendering entirely when neither output is requested.
                        check_cancelled(cancel_event)
                        particle_figure = on_figure
                        if on_particle_figure is not None:
                            def particle_figure(png, i=int(image_index),
                                                number=int(particle_number)):
                                on_particle_figure(png, i, number)
                        if cfg.save_figures or cfg.show_crop_plots:
                            show_crop_control(
                                image,
                                h,
                                x_center,
                                y_center,
                                crop_shape,
                                f'{paths[image_index].name} — candidate {particle_number}',
                                on_figure=particle_figure,
                                save_bounds_path=(
                                    result_path.with_name(result_path.stem +
                                                          '_crop_bounds.png')
                                    if cfg.save_figures else None),
                                save_crop_path=(result_path.with_name(
                                    result_path.stem + '_normalized_crop.png')
                                                if cfg.save_figures else None),
                                display=cfg.show_crop_plots)
                        check_cancelled(cancel_event)
                        if cfg.save_figures or cfg.show_fit_plots:
                            agreement = best_trial.get('relative_improvement')
                            fit_label = ('LOW AGREEMENT — ' if agreement is not None
                                         and agreement < .5 else '')
                            show_fit_comparison(
                                h,
                                best_result.hologram,
                                fit_label +
                                f'{paths[image_index].name} — candidate {particle_number}\n'
                                f'x={best_trial["x_um"]:.4g} µm  |  '
                                f'y={best_trial["y_um"]:.4g} µm  |  '
                                f'z={best_trial["z_um"]:.4g} µm\n'
                                f'radius={best_trial["r_um"]:.4g} µm  |  '
                                f'n={best_trial["n"]:.6g}',
                                on_figure=particle_figure,
                                save_comparison_path=(result_path.with_name(
                                    result_path.stem +
                                    '_fit_comparison.png') if cfg.save_figures
                                                      else None),
                                display=cfg.show_fit_plots)
                    except AnalysisCancelled as exc:
                        _record_particle_costs(
                            exc.trials, preparation_time,
                            0.0 if fitting_started is None else
                            perf_counter() - fitting_started)
                        result.trial_rows.extend(
                            dict(
                                info, particle_number=particle_number, **trial)
                            for trial in exc.trials)
                        raise
                    except Exception as exc:
                        result.error_rows.append(
                            dict(info,
                                 particle_number=particle_number,
                                 stage='particle',
                                 error=f'{type(exc).__name__}: {exc}'))
                        log(f'Candidate skipped: {exc}')
                    finally:
                        if guess_library is not None:
                            guess_library.flush()
                            if guess_library.warning:
                                log(f'Experiment profile: {guess_library.warning}')
                        journal.commit()
                check_cancelled(cancel_event)
            except KeyboardInterrupt:
                result.interrupted = True
                log('Analysis stopped; completed fits have been saved.')
                break
            except Exception as exc:
                result.error_rows.append(
                    dict(info,
                         particle_number='',
                         stage='image',
                         error=f'{type(exc).__name__}: {exc}'))
                log(f'Image skipped: {exc}')
            finally:
                journal.commit()
                if on_progress and not result.interrupted:
                    on_progress(
                        image_step, len(indices),
                        f'Image {image_step}/{len(indices)} completed')
    finally:
        journal.commit()
        if isinstance(sys.exc_info()[1],
                      KeyboardInterrupt) or (cancel_event is not None
                                             and cancel_event.is_set()):
            result.interrupted = True
        # Final consolidation also runs after cooperative Stop.
        save_tables(result)
    log(f'\n{len(result.summary_rows)} selected fit(s), {len(result.error_rows)} error(s).'
        )
    return result
