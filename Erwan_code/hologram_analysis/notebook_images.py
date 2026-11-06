"""PNG galleries independent of notebook stdout capture and Output state."""
from __future__ import annotations

def clear_images(container):
    """Remove views before closing their models and releasing their buffers."""
    previous = container.children
    container.children = ()
    for widget in previous:
        widget.close()


def validated_png(png_data):
    """Normalize binary figure data before storing or publishing it."""
    if not isinstance(png_data, (bytes, bytearray, memoryview)):
        raise TypeError('Figure data must contain PNG bytes.')
    png = bytes(png_data)
    if not png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('The figure did not produce a PNG image.')
    return png


def append_png(container, png_data, limit=None):
    """Publish a PNG using the standard Image widget's binary value trait.

    Output widgets synchronize captured streams from the frontend as well as
    receiving Python updates. Keeping images in separate models prevents a
    delayed stdout state from replacing the figure list after a button click.
    """
    from ipywidgets import Image, Layout

    png = validated_png(png_data)
    if limit is not None and limit < 1:
        raise ValueError('The figure display limit must be positive.')
    image = Image(value=png, format='png',
                  layout=Layout(width='auto', height='auto', max_width='100%'))
    children = container.children + (image,)
    discarded = () if limit is None else children[:-limit]
    container.children = children if limit is None else children[-limit:]
    for widget in discarded:
        widget.close()

