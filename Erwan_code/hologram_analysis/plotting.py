"""Detection, crop, initial simulation and fit diagnostic figures."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import numpy as np
from holopy.scattering import Sphere, calc_holo
from .config import AnalysisConfig


def new_figure(nrows=1, ncols=1, **kwargs):
    """Create a figure outside pyplot.

    Under the ipympl backend, plt.subplots() publishes an interactive canvas
    widget that is orphaned by plt.close(), which raises errors in the notebook
    when the frontend later sends it a message. These figures are only rendered
    to PNG, so they never create a canvas widget.
    """
    figure = Figure(**kwargs)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(nrows, ncols)


def show_figure(figure, on_figure=None, save_path=None, display=True):
    """Render once, optionally save, then display or send the PNG to the UI.

    Saved diagnostic images are independent of notebook display preferences.
    """
    try:
        with BytesIO() as buffer:
            figure.savefig(buffer, format='png', bbox_inches='tight')
            png = buffer.getvalue()
        if save_path is not None:
            destination = Path(save_path)
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            temporary.write_bytes(png)
            temporary.replace(destination)
        if not display:
            return
        if on_figure is None:
            from IPython.display import Image, display as display_image
            display_image(Image(data=png))
        else:
            on_figure(png)
    finally:
        # Existing scripts may pass a pyplot/ipympl figure. Detach its frontend
        # callbacks before closing it; figures created here use Agg exclusively.
        if hasattr(figure.canvas, 'on_msg'):
            from .notebook import close_canvas
            close_canvas(figure.canvas)
        plt.close(figure)


def show_detection_maps(
    image: Any,
    activity: Any,
    candidates: list[dict],
    title: str,
    cfg: AnalysisConfig,
    on_figure: Optional[Callable] = None,
    log: Callable = print,
    selected_numbers=None,
    center_overrides=None,
    selected_only=False,
) -> None:
    """Show the candidate selection and any manual corrections in image pixels.

    A candidate's detected position remains visible when its fit/crop center
    has been changed manually. This makes the effect of corrections explicit.
    """
    cfg = cfg.centered_for_image(image.shape)
    fig, axes = new_figure(
        1, 2, figsize=(14, 9) if cfg.click_to_detect else (15, 7))
    axes[0].get_gridspec().set_width_ratios(
        (3, 1) if cfg.click_to_detect else (1, 1))
    axes[0].imshow(image, cmap='gray', origin='upper')
    if not cfg.click_to_detect:
        axes[0].add_patch(
            Circle((cfg.target_y_px, cfg.target_x_px),
                   cfg.search_radius_px,
                   fill=False,
                   edgecolor='cyan',
                   linewidth=1.5))
        axes[0].plot(cfg.target_y_px, cfg.target_x_px, 'c+', markersize=10)
    selected_only = selected_only or cfg.click_to_detect
    selected = None if selected_numbers is None else set(selected_numbers)
    overrides = center_overrides or {}
    for number, candidate in enumerate(candidates, 1):
        if selected_only and number not in (selected or set()):
            continue
        override = overrides.get(str(number), overrides.get(number))
        center = candidate if override is None else override
        color = 'lime' if selected is not None and number in selected else 'yellow'
        if override is not None:
            axes[0].plot(candidate['y'],
                         candidate['x'],
                         'o',
                         color='silver',
                         fillstyle='none',
                         markersize=9)
            axes[0].plot([candidate['y'], center['y']],
                         [candidate['x'], center['x']],
                         '--',
                         color='magenta',
                         linewidth=1)
            axes[0].plot(center['y'],
                         center['x'],
                         '+',
                         color='magenta',
                         markersize=14)
        axes[0].plot(center['y'],
                     center['x'],
                     'o',
                     color=color,
                     fillstyle='none',
                     markersize=14)
        axes[0].text(center['y'] + 9,
                     center['x'] - 9,
                     str(number),
                     color=color,
                     weight='bold')
    selection_label = '; green = selected' if selected is not None else ''
    correction_label = '; magenta = corrected center' if overrides else ''
    axes[0].set_xlim(-.5, image.shape[1] - .5)
    axes[0].set_ylim(image.shape[0] - .5, -.5)
    axes[0].set_title(
        'Click near a particle to center and select it' if cfg.
        click_to_detect else 'Candidates; cyan = search area' +
        selection_label + correction_label)
    axes[1].imshow(activity, cmap='magma', origin='upper')
    axes[1].set_title(
        'Local fringe activity' if cfg.click_to_detect else 'Fringe activity')
    for ax in axes:
        ax.set_xlabel('y HoloPy (pixels, horizontal)')
        ax.set_ylabel('x HoloPy (pixels, vertical)')
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .96))
    show_figure(fig, on_figure)
    for number, candidate in enumerate(candidates, 1):
        if selected_only and number not in (selected or set()):
            continue
        override = overrides.get(str(number), overrides.get(number))
        center = candidate if override is None else override
        distance = np.hypot(center['x'] - cfg.target_x_px,
                            center['y'] - cfg.target_y_px)
        correction = (f'; detected x={candidate["x"]}, y={candidate["y"]} px'
                      if override is not None else '')
        log(f'{number}: x={center["x"]}, y={center["y"]} px ; '
            f'distance to target={distance:.1f} px{correction}')


def show_crop_control(image,
                      h,
                      x_center,
                      y_center,
                      crop_shape,
                      title,
                      on_figure=None,
                      save_bounds_path=None,
                      save_crop_path=None,
                      display=True):
    """Show automatic fringe extent, actual fitting window and particle offset.

    The extent reference and the fitting window use original image coordinates.
    Two PNGs are retained for compatibility with existing notebook/output flows.
    Legacy fixed crops without extent metadata retain their original view.
    """
    crop_height, crop_width = crop_shape
    info = h.attrs.get('crop_selection', {})
    bounds = info.get('bounds_px')
    extent = info.get('extent_bounds_px')
    x_start, y_start = ((bounds[0], bounds[2]) if bounds else
                        (x_center - crop_height // 2,
                         y_center - crop_width // 2))
    fig, ax = new_figure(figsize=(8, 8))
    ax.imshow(image, cmap='gray', origin='upper')
    ax.add_patch(
        Rectangle(
            (y_start - 0.5, x_start - 0.5),
            crop_width,
            crop_height,
            fill=False,
            edgecolor='cyan',
            linewidth=2,
            label=(f'Stage 2: fitting crop {crop_height} × {crop_width} px'
                   if extent else None)))
    ax.plot(y_center,
            x_center,
            'c+',
            markersize=18,
            markeredgewidth=2,
            label='Particle center' if extent else None)
    if extent:
        ex0, ex1, ey0, ey1 = extent
        extent_height, extent_width = ex1 - ex0, ey1 - ey0
        ax.add_patch(
            Rectangle(
                (ey0 - .5, ex0 - .5),
                extent_width,
                extent_height,
                fill=False,
                edgecolor='darkorange',
                linestyle='--',
                linewidth=2,
                label=
                f'Stage 1: fringe extent {extent_height} × {extent_width} px'))
        proposal = info.get('proposed_bounds_px')
        final_bounds = bounds or (x_start, x_start + crop_height, y_start,
                                  y_start + crop_width)
        if proposal is not None and tuple(proposal) != tuple(final_bounds):
            px0, px1, py0, py1 = proposal
            ax.add_patch(
                Rectangle(
                    (py0 - .5, px0 - .5),
                    py1 - py0,
                    px1 - px0,
                    fill=False,
                    edgecolor='magenta',
                    linestyle=':',
                    linewidth=2,
                    label=
                    f'Initial proposal {px1 - px0} × {py1 - py0} px (not retained)'
                ))
        pixels = crop_height * crop_width
        reduction = 100 * (1 - pixels / (extent_height * extent_width))
        confidence = info.get('extent_confidence', 'unresolved')
        area_change = (f'{reduction:.1f}% fewer pixels'
                       if reduction >= 0 else f'{-reduction:.1f}% more pixels')
        title += (f'\nExtent confidence: {confidence}; '
                  f'{area_change} than extent reference')
        if info.get('rectangle_refined'):
            title += '\nCrop check: wider rectangle retained'
        elif info.get('validation') == 'expanded':
            title += '\nCrop check: initial rectangle rejected; extent retained'
        # A per-particle context makes differing automatic extents visible.
        # The axes retain original image coordinates and actual center offsets.
        low_x, high_x = min(ex0, x_start), max(ex1, x_start + crop_height)
        low_y, high_y = min(ey0, y_start), max(ey1, y_start + crop_width)
        margin = max(12, int(.12 * max(high_x - low_x, high_y - low_y)))
        ax.set_xlim(max(-.5, low_y - margin - .5),
                    min(image.shape[1] - .5, high_y + margin - .5))
        ax.set_ylim(min(image.shape[0] - .5, high_x + margin - .5),
                    max(-.5, low_x - margin - .5))
        ax.legend(loc='upper right', fontsize=8, framealpha=.9)
    ax.set(xlabel='y HoloPy (pixels, horizontal)',
           ylabel='x HoloPy (pixels, vertical)',
           title=title)
    if save_bounds_path is None and display:
        show_figure(fig, on_figure)
    else:
        show_figure(fig, on_figure, save_bounds_path, display)
    fig, ax = new_figure(figsize=(6, 6))
    im = ax.imshow(h.squeeze().transpose('x', 'y'), cmap='gray')
    ax.plot(y_center - y_start, x_center - x_start, 'c+', markersize=12)
    normalized_title = (
        f'Final normalized crop: {crop_height} × {crop_width} px'
        if extent else 'Normalized crop')
    ax.set(title=normalized_title, xlabel='y (pixels)', ylabel='x (pixels)')
    fig.colorbar(im, ax=ax)
    if save_crop_path is None and display:
        show_figure(fig, on_figure)
    else:
        show_figure(fig, on_figure, save_crop_path, display)


def show_fit_comparison(
    h: Any,
    fitted_holo: Any,
    title: str,
    on_figure: Optional[Callable] = None,
    save_comparison_path=None,
    save_residual_path=None,
    display=True,
    include_residual=False,
) -> None:
    """Compare on one intensity scale; optionally save a separate residual map."""
    fitted_holo = fitted_holo.transpose(*h.dims)
    if fitted_holo.shape != h.shape or any(
            not np.array_equal(h.coords[dim], fitted_holo.coords[dim])
            for dim in h.dims):
        raise ValueError(
            'The fitted hologram and crop must use the same grid.')
    fig, axes = new_figure(1, 2, figsize=(10, 5.2))
    vmin, vmax = float(h.min()), float(h.max())
    for ax, data, label in zip(axes, [h, fitted_holo],
                               ['Experimental', 'Best fit']):
        im = ax.imshow(data.squeeze().transpose('x', 'y'),
                       cmap='gray',
                       vmin=vmin,
                       vmax=vmax)
        ax.set(title=label, xlabel='y (pixels)', ylabel='x (pixels)')
        fig.colorbar(im, ax=ax)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .82))
    if save_comparison_path is None and display:
        show_figure(fig, on_figure)
    else:
        show_figure(fig, on_figure, save_comparison_path, display)
    if include_residual or save_residual_path is not None:
        residual = (h - fitted_holo).squeeze().transpose('x', 'y')
        # A symmetric scale prevents the colour map from hiding sign or offset.
        limit = max(float(np.max(np.abs(residual))), np.finfo(float).eps)
        fig, ax = new_figure(figsize=(6, 5))
        im = ax.imshow(residual, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(title='Residual (experimental − fit)',
               xlabel='y (pixels)',
               ylabel='x (pixels)')
        fig.colorbar(im, ax=ax, label='Normalized intensity difference')
        fig.suptitle(title)
        fig.tight_layout()
        show_figure(fig, on_figure, save_residual_path, display)


def show_initial_guess(h, xguess, yguess, cfg, on_figure=None):
    sphere = Sphere(n=cfg.n_guess,
                    r=cfg.r_guess,
                    center=[xguess, yguess, cfg.z_values[0]])
    initial = calc_holo(h,
                        sphere,
                        theory=cfg.theory(),
                        medium_index=cfg.medium_index,
                        illum_wavelen=cfg.illum_wavelen,
                        illum_polarization=cfg.illum_polarization)
    print(
        f'Initial hologram: rguess={cfg.r_guess:g} µm; zguess={cfg.z_values[0]:g} µm'
    )
    fig, ax = new_figure(figsize=(6, 6))
    im = ax.imshow(initial.squeeze().transpose('x', 'y'), cmap='gray')
    ax.set(title='Initial hologram', xlabel='y (pixels)', ylabel='x (pixels)')
    fig.colorbar(im, ax=ax)
    show_figure(fig, on_figure)
    return initial
