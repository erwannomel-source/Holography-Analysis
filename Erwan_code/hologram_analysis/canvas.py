"""Coordinate interactive rendering and reuse its pixels for notebook snapshots."""
from __future__ import annotations

from threading import RLock
from base64 import b64encode
from io import BytesIO
from traitlets import default
import numpy as np
from PIL import Image

from ipympl.backend_nbagg import Canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg


class ManagedCanvas(Canvas):
    """Keep the interactive manager available during display and image export.

    Matplotlib's print_figure temporarily sets canvas.manager to None. ipympl
    can receive a redraw while that export is active, including a nested draw
    callback. Retain the manager and serialize drawing/exporting on this canvas.
    """

    def __init__(self, figure, *args, **kwargs):
        self._render_lock = RLock()
        self._retained_manager = None
        self._disposed = False
        self._snapshot_png = None
        self._rendered_renderer = None
        self._rendered_key = None
        self.frontend_ready = False
        self.on_frontend_ready = None
        # Widget models are created when the comm opens, before a VBox displays
        # its children. ipympl reads the initial image only once, so prepare it
        # before DOMWidget.__init__ publishes a zero-sized, empty model.
        with BytesIO() as buffer:
            initial_canvas = FigureCanvasAgg(figure)
            initial_canvas.print_png(buffer)
            self._snapshot_png = buffer.getvalue()
            self._initial_data_url = (
                'data:image/png;base64,' +
                b64encode(self._snapshot_png).decode('ascii'))
        self._initial_size = tuple(figure.bbox.size)
        super().__init__(figure, *args, **kwargs)
        # Reuse the constructor's complete frame for the first fallback image.
        self.renderer = initial_canvas.renderer
        self._lastKey = (*figure.bbox.size, figure.dpi)
        self._rendered_renderer = self.renderer
        self._rendered_key = self._lastKey
        # Agg export methods can bypass this class's draw(). Track the actual
        # figure draw as well, so exports and browser resizes invalidate pixels.
        self.mpl_connect('draw_event', self._record_render)

    def _record_render(self, event):
        self._snapshot_png = None
        self._rendered_renderer = event.renderer
        self._rendered_key = (*self.figure.bbox.size, self.figure.dpi)

    def _snapshot_from_renderer(self):
        """Encode the already rendered full frame once, never a diff image."""
        if self._snapshot_png is None:
            pixels = np.asarray(self.renderer.buffer_rgba())
            with BytesIO() as buffer:
                Image.fromarray(pixels).save(buffer, format='PNG')
                self._snapshot_png = buffer.getvalue()
        return self._snapshot_png

    @default('_data_url')
    def _initial_image(self):
        return self._initial_data_url

    @default('_size')
    def _initial_dimensions(self):
        return self._initial_size

    def retain_manager(self, manager):
        """Attach the manager before publishing the canvas to the notebook."""
        self._retained_manager = manager
        self.manager = manager

    def _ensure_manager(self):
        if self.manager is None:
            self.manager = self._retained_manager
        if self.manager is None:
            raise RuntimeError(
                'The interactive canvas must have a manager before drawing.')
        return self.manager

    def _handle_message(self, widget, content, buffers):
        with self._render_lock:
            if self._disposed:
                return
            self._ensure_manager()
            result = super()._handle_message(widget, content, buffers)
            if content['type'] == 'initialized':
                # The browser model has connected. Send a complete image now;
                # an earlier draw request can precede its message listener.
                self._force_full = True
                self.draw()
            if content['type'] in ('initialized',
                                   'draw') and not self.frontend_ready:
                self.frontend_ready = True
                if self.on_frontend_ready is not None:
                    self.on_frontend_ready()
            return result

    def prepare_display(self):
        """Provide valid initial image and dimensions for a canvas inside a VBox.

        Child widgets do not invoke _repr_mimebundle_, which normally sets this
        display state. A raw base64 string and the default (0, 0) size cannot be
        displayed by ipympl's browser model.
        """
        with self._render_lock:
            key = (*self.figure.bbox.size, self.figure.dpi)
            if (self.figure.stale or self._rendered_key != key
                    or self._rendered_renderer is not getattr(
                        self, 'renderer', None)):
                self.draw()
            png = self._snapshot_from_renderer()
            self._data_url = 'data:image/png;base64,' + b64encode(png).decode(
                'ascii')
            ratio = self.device_pixel_ratio
            width, height = self.figure.bbox.size
            self._size = (width / ratio, height / ratio)
            return png

    def get_diff_image(self):
        """Reuse WebAgg's full PNG for the static fallback when available."""
        with self._render_lock:
            data = super().get_diff_image()
            if data is not None and self._current_image_mode == 'full':
                self._snapshot_png = data
            return data

    def draw(self):
        with self._render_lock:
            if self._disposed:
                return
            manager = self._ensure_manager()
            self._png_is_old = True
            try:
                FigureCanvasAgg.draw(self)
            finally:
                # Use the retained reference rather than canvas.manager, which
                # Matplotlib may temporarily detach inside an export callback.
                manager.refresh_all()

    def draw_idle(self):
        with self._render_lock:
            if not self._disposed:
                self._ensure_manager()
                return super().draw_idle()

    def blit(self, bbox=None):
        with self._render_lock:
            if not self._disposed:
                self._ensure_manager()
                self._snapshot_png = None
                return super().blit(bbox)

    def refresh_visible(self):
        """Resend dimensions and a full frame when the interactive view is shown."""
        with self._render_lock:
            if self._disposed:
                return
            manager = self._ensure_manager()
            manager.resize(*self._size)
            self._force_full = True
            self.draw()

    def send_binary(self, data):
        """Send encoded PNG bytes even before ipympl's frontend has connected.

        ipympl 0.9.7 replaces its PNG payload with a raw RGBA array while
        updating the initial snapshot. Browsers cannot decode that array as
        an image. Keep the snapshot and the binary frame as separate values.
        """
        if self.syncing_data_url:
            self._data_url = (
                'data:image/png;base64,' +
                b64encode(self._snapshot_from_renderer()).decode('ascii'))
        self.send({'data': '{"type": "binary"}'}, buffers=[data])

    def print_figure(self, *args, **kwargs):
        with self._render_lock:
            return super().print_figure(*args, **kwargs)

    def close(self):
        """Ignore late redraw/resize messages after the widget is replaced."""
        with self._render_lock:
            if self._disposed:
                return
            self._disposed = True
            self.on_msg(self._handle_message, remove=True)
            super().close()
