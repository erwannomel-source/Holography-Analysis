"""Compatibility entry point for analyse_hologrammes.ipynb and existing scripts.

Implementation lives in hologram_analysis/. Import this module as before;
configuration, preparation, fitting, outputs and notebook controls remain available.
"""
import numpy as np
import matplotlib.pyplot as plt
import holopy as hp
from holopy.scattering import Sphere, calc_holo
from holopy.inference import prior, ExactModel
from holopy.scattering.theory import MieLens
from hologram_analysis import __version__
from hologram_analysis.cropping import select_adaptive_crop, bounded_window
from hologram_analysis.config import (
    MIN_CROP_PX,
    AnalysisCancelled,
    check_cancelled,
    AnalysisConfig,
    parse_numbers,
)
from hologram_analysis.preparation import (
    Candidate,
    ManualCenter,
    PreparedParticle,
    detect_candidates,
    refine_ring_center,
    safe_crop_size,
    safe_crop_shape,
    load_and_detect,
    normalize_particle_choice,
    select_candidates,
    prepare_particle,
    apply_center_override,
)
from hologram_analysis.plotting import (
    show_figure,
    show_detection_maps,
    show_crop_control,
    show_fit_comparison,
    show_initial_guess,
)
from hologram_analysis.fitting import (
    crop_fit_bounds,
    optimizer_diagnostics,
    fit_quality_diagnostics,
    fit_particle_multi_z,
)
from hologram_analysis.outputs import (
    SUMMARY_FIELDS,
    TRIAL_FIELDS,
    ERROR_FIELDS,
    QUALITY_FIELDS,
    RunResult,
    RunJournal,
    write_csv,
    save_tables,
    recover_run,
    software_versions,
)
from hologram_analysis.analysis import analyse_images
from hologram_analysis.profiles import resolve_path, read_profile, validate_profile
from hologram_analysis.widgets import PathPicker
from hologram_analysis.interface import create_notebook_interface, NotebookInterface
