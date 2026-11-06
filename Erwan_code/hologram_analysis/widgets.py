"""Notebook file navigation controls."""
from __future__ import annotations

import html
from pathlib import Path
from .profiles import resolve_path


class PathPicker:
    """Browse files inside Jupyter without an external dialog."""

    def __init__(self,
                 label,
                 base_dir,
                 kind='directory',
                 value='',
                 suffixes=()):
        import ipywidgets as W
        self.base_dir = Path(base_dir).resolve()
        self.kind = kind
        self.suffixes = tuple(s.lower() for s in suffixes)
        self.path = W.Text(
            value=str(value),
            description=label,
            tooltip=
            'Enter a path or use Browse. Relative paths are resolved from the notebook directory.',
            layout=W.Layout(width='78%'),
            style={'description_width': '160px'})
        self.browse = W.Button(
            description='Browse…',
            tooltip=
            'Browse directories and choose a path without leaving the notebook.',
            layout=W.Layout(width='140px'))
        self.current = W.Text(
            value=str(self.base_dir),
            description='Directory',
            tooltip=
            'Current browsing directory. Enter a directory and press Return to list it.',
            layout=W.Layout(width='100%'))
        self.entries = W.Select(
            options=[],
            rows=7,
            tooltip=
            'Highlight an entry, then use Open directory or Select to confirm your choice.',
            layout=W.Layout(width='100%'))
        self.parent = W.Button(description='Parent directory')
        self.open = W.Button(description='Open directory')
        self.choose = W.Button(
            description='Select highlighted folder' if kind ==
            'directory' else 'Select file')
        self.choose_current = W.Button(description='Use current directory')
        self.close = W.Button(description='Close')
        self.message = W.HTML()
        self.panel = W.VBox([
            self.current, self.entries,
            W.HBox([
                self.parent, self.open, self.choose, self.choose_current,
                self.close
            ]), self.message
        ])
        self.choose_current.layout.display = '' if kind == 'directory' else 'none'
        self.panel.layout.display = 'none'
        self.widget = W.VBox([W.HBox([self.path, self.browse]), self.panel])
        self.browse.on_click(self._browse)
        self.current.on_submit(lambda _: self._list())
        self.parent.on_click(self._parent)
        self.open.on_click(self._open)
        self.choose.on_click(self._choose)
        self.choose_current.on_click(self._choose_current)
        self.close.on_click(
            lambda _: setattr(self.panel.layout, 'display', 'none'))

    def _error(self, exc):
        self.message.value = f'<span style="color:#a32020">{html.escape(str(exc))}</span>'

    def _browse(self, _):
        try:
            folder = resolve_path(
                self.path.value,
                self.base_dir) if self.path.value.strip() else self.base_dir
            if folder.is_file():
                folder = folder.parent
            if not folder.is_dir():
                folder = self.base_dir
            self.current.value = str(folder)
            self.panel.layout.display = ''
            self._list()
        except Exception as exc:
            self._error(exc)

    def _list(self):
        try:
            folder = resolve_path(self.current.value, self.base_dir)
            if not folder.is_dir():
                raise ValueError('This path is not an accessible directory.')
            self.current.value = str(folder)
            paths = sorted(folder.iterdir(),
                           key=lambda p: (not p.is_dir(), p.name.lower()))
            options = []
            for path in paths:
                if path.is_dir():
                    options.append(('Directory — ' + path.name, str(path)))
                elif self.kind == 'file' and (not self.suffixes
                                              or path.suffix.lower()
                                              in self.suffixes):
                    options.append(('File — ' + path.name, str(path)))
            self.entries.options = options
            self.entries.value = None
            self.message.value = (
                'Select a file, then click Select file. Use Open directory to enter a folder.'
                if self.kind == 'file' else
                'Highlight a folder, then click Select highlighted folder. '
                'Use current directory chooses the folder shown above.')
        except Exception as exc:
            self._error(exc)

    def _parent(self, _):
        try:
            self.current.value = str(
                resolve_path(self.current.value, self.base_dir).parent)
            self._list()
        except Exception as exc:
            self._error(exc)

    def _open(self, _):
        try:
            if self.entries.value is None:
                raise ValueError('Select a directory from the list.')
            path = Path(self.entries.value)
            if not path.is_dir():
                raise ValueError('Select a directory to open.')
            self.current.value = str(path)
            self._list()
        except Exception as exc:
            self._error(exc)

    def _choose(self, _):
        try:
            if self.kind == 'directory':
                if self.entries.value is None:
                    raise ValueError(
                        'Highlight a folder, or click Use current directory.')
                selected = Path(self.entries.value)
                if not selected.is_dir():
                    raise ValueError('Choose a directory.')
            else:
                if self.entries.value is None:
                    raise ValueError('Select a file from the list.')
                selected = Path(self.entries.value)
                if not selected.is_file():
                    raise ValueError('Select a file, then click Select file.')
            self.path.value = str(selected.resolve())
            self.panel.layout.display = 'none'
        except Exception as exc:
            self._error(exc)

    def _choose_current(self, _):
        """Choose the displayed directory explicitly, independently of list selection."""
        try:
            selected = resolve_path(self.current.value, self.base_dir)
            if not selected.is_dir():
                raise ValueError('This path is not an accessible directory.')
            self.path.value = str(selected)
            self.panel.layout.display = 'none'
        except Exception as exc:
            self._error(exc)
