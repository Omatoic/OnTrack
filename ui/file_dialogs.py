"""Make every QFileDialog in the app (OnTrack + the embedded roto editor)
open in the folder last used for that kind of file.

``install()`` wraps the QFileDialog static helpers once at startup, so call
sites keep using QFileDialog directly. The "kind" is derived from the name
filter (videos, images, .exe, .json, ...) so e.g. video pickers share one
folder while executable pickers remember another.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog

_KINDS = {
    "video": {"mp4", "m4v", "mov", "avi", "mkv", "webm"},
    "image": {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif", "webp", "exr"},
}


def _prefs() -> QSettings:
    return QSettings("OnTrack", "OnTrack")


def _kind(name_filter: str) -> str:
    exts = [e.lower() for e in re.findall(r"\*\.(\w+)", name_filter or "")]
    for kind, members in _KINDS.items():
        if any(e in members for e in exts):
            return kind
    return exts[0] if exts else "any"


def _start_dir(kind: str, given: str) -> str:
    """Absolute paths passed by the caller win; otherwise prefix the
    remembered folder (keeping a suggested filename like 'matte.mkv')."""
    if given and Path(given).is_absolute():
        return given
    prefs = _prefs()
    for key in (kind, "any"):
        last = prefs.value(f"last_dir/{key}", "", type=str)
        if last and Path(last).is_dir():
            return str(Path(last) / given) if given else last
    return given


def _remember(kind: str, path: str, is_dir: bool = False) -> None:
    if not path:
        return
    folder = str(Path(path) if is_dir else Path(path).parent)
    prefs = _prefs()
    prefs.setValue(f"last_dir/{kind}", folder)
    prefs.setValue("last_dir/any", folder)


def install() -> None:
    if getattr(QFileDialog, "_ontrack_wrapped", False):
        return
    orig_open = QFileDialog.getOpenFileName
    orig_open_many = QFileDialog.getOpenFileNames
    orig_save = QFileDialog.getSaveFileName
    orig_dir = QFileDialog.getExistingDirectory

    def _file_dialog(orig, many: bool = False):
        def wrapper(parent=None, caption="", dir="", filter="", *args, **kwargs):
            kind = _kind(filter)
            result = orig(parent, caption, _start_dir(kind, dir), filter, *args, **kwargs)
            chosen = result[0]
            _remember(kind, chosen[0] if many and chosen else ("" if many else chosen))
            return result
        return staticmethod(wrapper)

    def get_dir(parent=None, caption="", dir="", *args, **kwargs):
        path = orig_dir(parent, caption, _start_dir("folder", dir), *args, **kwargs)
        _remember("folder", path, is_dir=True)
        return path

    QFileDialog.getOpenFileName = _file_dialog(orig_open)
    QFileDialog.getOpenFileNames = _file_dialog(orig_open_many, many=True)
    QFileDialog.getSaveFileName = _file_dialog(orig_save)
    QFileDialog.getExistingDirectory = staticmethod(get_dir)
    QFileDialog._ontrack_wrapped = True
