"""Load current notebook code and retire obsolete interactive widgets."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


def close_canvas(canvas):
    """Detach comm callbacks before closing a canvas with queued frontend messages."""
    for callback in tuple(canvas._msg_callbacks.callbacks):
        canvas.on_msg(callback, remove=True)
    toolbar = getattr(canvas, 'toolbar', None)
    canvas.close()
    if toolbar is not None:
        toolbar.close()


def close_interface(interface):
    """Release a previous interface before recreating it in the same kernel."""
    worker = getattr(interface, '_worker', None)
    if getattr(interface, '_analysis_running',
               False) or (worker is not None and worker.is_alive()):
        raise RuntimeError(
            'Stop the current analysis and wait for it to finish '
            'before rerunning the startup cells.')
    canvas = getattr(interface, '_canvas', None)
    if canvas is not None:
        close_canvas(canvas)
    if hasattr(interface, '_clear_figure_store'):
        interface._clear_figure_store()

    visited = set()

    def close_widget(widget):
        if widget is None or id(widget) in visited:
            return
        visited.add(id(widget))
        for child in getattr(widget, 'children', ()):
            close_widget(child)
        widget.close()

    close_widget(getattr(interface, 'widget', None))
    # Crop-mode flags remain internal widgets for profile/script compatibility.
    # Close these models too, although they are not individual visible controls.
    for control in getattr(interface, 'controls', {}).values():
        close_widget(control)
    for name in ('_detection_cache', '_crop_cache'):
        cache = getattr(interface, name, None)
        if cache is not None:
            cache.clear()
    interface._preview = None


def close_orphaned_canvases():
    """Remove old ipympl canvases whose managers have already been detached."""
    # ipywidgets 8 keeps live comm widgets here. The public Widget.widgets alias
    # is deprecated; iterate a snapshot because close() removes its own entry.
    from ipywidgets.widgets.widget import _instances
    for widget in tuple(_instances.values()):
        if (getattr(widget, '_model_name', None) == 'MPLCanvasModel'
                and getattr(widget, 'manager', None) is None
                and getattr(widget, '_retained_manager', None) is None):
            close_canvas(widget)


def load_notebook_module(module_folder, previous_interface=None):
    """Import fresh analysis modules without restarting the HoloPy kernel."""
    folder = Path(module_folder).resolve()
    if not (folder / 'hologram_functions.py').is_file() or not (
            folder / 'hologram_analysis').is_dir():
        raise FileNotFoundError(
            'Keep hologram_functions.py and the complete '
            'hologram_analysis directory beside the notebook.')
    if previous_interface is not None:
        close_interface(previous_interface)
    close_orphaned_canvases()
    sys.path.insert(0, str(folder))
    importlib.invalidate_caches()
    for name in tuple(sys.modules):
        if (name == 'hologram_functions' or name == 'hologram_analysis'
                or name.startswith('hologram_analysis.')):
            del sys.modules[name]
    return importlib.import_module('hologram_functions')
