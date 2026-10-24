"""Roto tab: embeds the vendored Sammie-Roto editor (roto/sammie_main.py)
directly inside OnTrack and turns its per-object masks into matte videos
the Track tab can consume.

The editor must run in-process, so OnTrack needs a Python environment with
Sammie's dependencies (torch, av, PySide6, diffusers). run.bat launches
OnTrack with the Sammie-Roto venv for that reason.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import roto

# The vendored Sammie UI was designed for a light native theme; OnTrack's
# global dark stylesheet doesn't cover enough widget types to keep it legible.
# This dark theme matches OnTrack's palette and overrides the app stylesheet
# for the embedded subtree.
SAMMIE_DARK_QSS = """
QWidget {
    background-color: #121212;
    color: #f0f0f0;
    font-family: "Segoe UI", "Microsoft Sans Serif", sans-serif;
}
QMainWindow, QDialog, QMenuBar, QMenu { background-color: #121212; }
QGroupBox {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #f0f0f0;
    font-weight: bold;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit,
QListWidget, QTreeWidget, QTableWidget, QAbstractSpinBox {
    background-color: #252525;
    color: #f0f0f0;
    border: 1px solid #333333;
    border-radius: 5px;
    padding: 4px;
}
QComboBox QAbstractItemView {
    background-color: #252525;
    color: #f0f0f0;
    selection-background-color: #3a7bd5;
}
QPushButton {
    background-color: #2c2c2c;
    color: #f0f0f0;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton:hover { background-color: #3a3a3a; }
QPushButton:pressed { background-color: #252525; }
QPushButton:checked { background-color: #3a7bd5; border-color: #3a7bd5; }
QPushButton:disabled { color: #666666; background-color: #1e1e1e; }
QMenu::item:selected { background-color: #3a7bd5; }
QMenu::separator { height: 1px; background: #333333; }
QTabWidget::pane { border: 1px solid #333333; background-color: #121212; }
QTabBar::tab {
    background-color: #1e1e1e;
    color: #a0a0a0;
    padding: 6px 14px;
    border: 1px solid #333333;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected { background-color: #252525; color: #f0f0f0; }
QProgressBar {
    background-color: #252525;
    color: #f0f0f0;
    border: 1px solid #333333;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk { background-color: #3a7bd5; border-radius: 4px; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 15px; height: 15px;
    background-color: #252525;
    border: 1px solid #555555;
    border-radius: 3px;
}
QCheckBox::indicator:checked { background-color: #3a7bd5; border-color: #3a7bd5; }
QRadioButton::indicator { border-radius: 8px; }
QRadioButton::indicator:checked { background-color: #3a7bd5; border-color: #3a7bd5; }
QSlider::groove:horizontal { height: 6px; background-color: #333333; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 14px; margin: -5px 0; border-radius: 7px; background-color: #3a7bd5;
}
QSlider::sub-page:horizontal { background-color: #3a7bd5; border-radius: 3px; }
QHeaderView::section {
    background-color: #1e1e1e;
    color: #a0a0a0;
    border: none;
    border-bottom: 1px solid #333333;
    padding: 4px 6px;
}
QTreeWidget::item:selected, QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #3a7bd5;
}
QScrollBar:vertical {
    background: #1e1e1e; width: 10px; margin: 0;
}
QScrollBar::handle:vertical { background: #444444; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1e1e1e; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #444444; border-radius: 5px; min-width: 20px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QToolTip {
    background-color: #252525;
    color: #f0f0f0;
    border: 1px solid #333333;
    padding: 4px;
}
QSplitter::handle { background-color: #333333; }
QToolBar { background-color: #1e1e1e; border: none; spacing: 4px; }
QStatusBar { background-color: #1e1e1e; color: #a0a0a0; }
QLabel { background-color: transparent; }
"""


class MatteWorker(QThread):
    """Merge roto mask PNGs and encode a matte video off the UI thread."""

    finished_ok = Signal(str, str)   # matte path, role
    failed = Signal(str, str)        # error, role
    progress = Signal(str)

    def __init__(self, object_ids, role, ffmpeg: Path, fps: float, output: Path | None = None,
                 binary: bool = True, parent=None):
        super().__init__(parent)
        self._object_ids = object_ids
        self._role = role
        self._ffmpeg = ffmpeg
        self._fps = fps
        self._output = output
        self._binary = binary

    def run(self) -> None:
        try:
            session = roto.session_info() or {}
            total = int(session.get("total_frames") or 0)
            if total <= 0:
                self.failed.emit("Roto session reports no frames.", self._role)
                return
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged = roto.roto_temp() / "ontrack_matte"
            nonempty = roto.build_merged_masks(self._object_ids, total, merged, binary=self._binary)
            if nonempty == 0:
                self.failed.emit("Selected roto objects produced no masks.", self._role)
                return
            out = self._output or roto.roto_root() / "exports" / f"matte_{stamp}.mkv"
            path = roto.encode_matte_video(merged, out, self._ffmpeg, self._fps, log=print)
            self.finished_ok.emit(str(path), self._role)
        except Exception as exc:
            self.failed.emit(str(exc), self._role)


class _ImportWorker(QThread):
    """Import sammie_main (torch + deps) off the UI thread."""

    done = Signal(object)   # module
    failed = Signal(str)

    def run(self) -> None:
        try:
            root = roto.roto_root()
            os.chdir(root)
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import sammie_main
            self.done.emit(sammie_main)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RotoWidget(QWidget):
    """Hosts the Sammie-Roto editor and exposes its masks to OnTrack."""

    objects_changed = Signal()
    matte_ready = Signal(str, str)          # path, role ("ignore" | "object")
    status_message = Signal(str)

    def __init__(self, ffmpeg_getter, fps_getter, video_getter, parent=None):
        super().__init__(parent)
        self._ffmpeg_getter = ffmpeg_getter
        self._fps_getter = fps_getter
        self._video_getter = video_getter
        self._sammie = None                  # embedded sammie_main.MainWindow
        self._embed_failed: str | None = None
        self._loading = False
        self._importer: _ImportWorker | None = None
        self._pending_video: Path | None = None
        self._worker: MatteWorker | None = None
        self._objects: list[roto.MaskObject] = []

        self._build_ui()

        self._watch = QTimer(self)
        self._watch.setInterval(2000)
        self._watch.timeout.connect(self.refresh_objects)
        self._watch.start()

    # ---- UI ----

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self._splitter, 1)

        self._placeholder = QLabel(
            "Roto editor loads when you open this tab.\n"
            "(First load imports torch + SAM2 — this can take a moment.)"
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888888;")
        self._splitter.addWidget(self._placeholder)

        panel = QWidget()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(340)
        side = QVBoxLayout(panel)
        side.setSpacing(8)

        open_row = QHBoxLayout()
        open_btn = QPushButton("Open Current Video")
        open_btn.setToolTip("Load the Track tab's video into the roto editor")
        open_btn.clicked.connect(lambda: self.open_editor(self._video_getter()))
        open_row.addWidget(open_btn)
        open_other = QPushButton("Open Other…")
        open_other.setToolTip("Load any video or image into the roto editor")
        open_other.clicked.connect(self._pick_video)
        open_row.addWidget(open_other)
        side.addLayout(open_row)

        self._session_label = QLabel("No roto session")
        self._session_label.setWordWrap(True)
        self._session_label.setStyleSheet("color: #a0a0a0; font-size: 9pt;")
        side.addWidget(self._session_label)

        masks_label = QLabel("Masks — pick these in the Track tab's mask dropdowns")
        masks_label.setWordWrap(True)
        masks_label.setStyleSheet("font-weight: bold;")
        side.addWidget(masks_label)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Object", "Frames", "Coverage"])
        self._tree.setRootIsDecorated(False)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        side.addWidget(self._tree, 1)

        export_btn = QPushButton("Export Matte Video…")
        export_btn.setToolTip(
            "Encode a white-on-black matte video for the selected objects "
            "(or all objects if nothing is selected)."
        )
        export_btn.clicked.connect(self._export_matte)
        side.addWidget(export_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        side.addWidget(self._status)
        side.addStretch()

        self._splitter.addWidget(panel)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setCollapsible(1, False)

    # ---- Embedded editor ----

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._ensure_editor)

    def _ensure_editor(self) -> bool:
        if self._sammie is not None:
            return True
        if self._embed_failed is not None or self._loading:
            return False
        self._loading = True
        self._placeholder.setText("Loading roto editor (torch + SAM2)...")
        self._set_status("Loading roto editor...")
        self._importer = _ImportWorker(self)
        self._importer.done.connect(self._on_module_imported)
        self._importer.failed.connect(self._on_import_failed)
        self._importer.start()
        return False

    def _on_module_imported(self, sammie_main) -> None:
        try:
            self._sammie = sammie_main.MainWindow()
            self._sammie.setStyleSheet(SAMMIE_DARK_QSS)
            # Keep Segmentation + Matting (for refining mattes); hide Object
            # Removal, which stays available in the standalone tool.
            tabw = self._sammie.sidebar.tab_widget
            tabw.removeTab(2)
            self._sammie.setParent(self)
            self._splitter.insertWidget(0, self._sammie)
            self._splitter.setStretchFactor(0, 1)
            self._placeholder.hide()
            self._set_status("Roto editor loaded.")
        except Exception:
            self._on_import_failed(traceback.format_exc())
            return
        self._loading = False
        if self._pending_video is not None:
            video, self._pending_video = self._pending_video, None
            self._open_in_sammie(video)

    def _on_import_failed(self, tb: str) -> None:
        self._embed_failed = tb
        self._loading = False
        self._placeholder.setText(
            "Roto editor failed to load.\n\n"
            "OnTrack must run on a Python with Sammie's dependencies\n"
            "(torch, av, PySide6). Just run main.py — it relaunches under\n"
            "the Sammie-Roto venv automatically.\n\n" + tb
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._set_status("Roto editor failed to load.")

    def _pick_video(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Open in Roto Editor", "",
            "Videos/Images (*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All files (*.*)",
        )
        if path:
            self.open_editor(Path(path))

    def _open_in_sammie(self, video: Path) -> None:
        try:
            self._sammie.load_file(str(video))
            self._set_status(f"Opened in roto editor: {video.name}")
        except Exception as exc:
            self._set_status(f"Failed to load video in roto editor: {exc}")

    def open_editor(self, video: Path | None = None) -> bool:
        """Load a video into the embedded roto editor (kept for the Track tab API)."""
        if not self._ensure_editor():
            if video is not None:
                self._pending_video = video
            return False
        if video is not None:
            self._open_in_sammie(video)
        return True

    def shutdown(self) -> None:
        """Persist the embedded editor's session on app close.

        Calls the editor's save routines directly rather than routing through
        its closeEvent, which can raise when the window is a child widget
        being torn down (and leave OnTrack's own closeEvent in a bad state).
        """
        if self._sammie is None:
            return
        w, self._sammie = self._sammie, None
        for fn in (
            lambda: w._save_current_ui_state(),
            lambda: w.settings_mgr.save_points(w.point_manager.get_all_points()),
            lambda: w.settings_mgr.save_session_settings(),
            lambda: w._save_window_and_splitter_settings(),
        ):
            try:
                fn()
            except Exception:
                pass
        try:
            if hasattr(w, "console_redirect"):
                w.console_redirect.close()
        except Exception:
            pass

    # ---- Mask listing ----

    def objects(self) -> list[roto.MaskObject]:
        return self._objects

    def refresh_objects(self) -> None:
        session = roto.session_info()
        if session:
            video = Path(session.get("video_file_path", "")).name
            total = session.get("total_frames", "?")
            self._session_label.setText(f"{video} — {total} frames @ {session.get('video_fps', '?')} fps")
        else:
            self._session_label.setText("No roto session")

        objects = roto.list_mask_objects()
        if [o.object_id for o in objects] == [o.object_id for o in self._objects] and \
           [o.frame_count for o in objects] == [o.frame_count for o in self._objects]:
            return
        self._objects = objects

        self._tree.clear()
        total = int((session or {}).get("total_frames") or 0)
        if objects:
            all_item = QTreeWidgetItem(
                ["All objects", str(sum(o.frame_count for o in objects)), ""]
            )
            all_item.setData(0, Qt.ItemDataRole.UserRole, None)
            self._tree.addTopLevelItem(all_item)
        for obj in objects:
            coverage = f"{obj.frame_count / total * 100:.0f}%" if total else "-"
            name = obj.name
            if obj.refined_frames:
                name += f"  (refined {len(obj.refined_frames)}f)"
            item = QTreeWidgetItem([name, str(obj.frame_count), coverage])
            item.setData(0, Qt.ItemDataRole.UserRole, obj.object_id)
            self._tree.addTopLevelItem(item)
        self.objects_changed.emit()

    # ---- Matte building ----

    def _selected_object_ids(self) -> list[int] | None:
        """Object ids selected in the tree; None = all objects."""
        items = self._tree.selectedItems()
        if not items:
            return None
        ids = [it.data(0, Qt.ItemDataRole.UserRole) for it in items]
        return None if any(i is None for i in ids) else [int(i) for i in ids]

    def _export_matte(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        object_ids = self._selected_object_ids()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Matte Video", "matte.mkv", "Matte video (*.mkv *.mov);;All files (*.*)"
        )
        if not path:
            return
        # Keep soft alpha edges in exported mattes; tracking still gets binary.
        self._start_matte_worker("export", object_ids, output=Path(path), binary=False)

    def _start_matte_worker(self, role: str, object_ids: list[int] | None, output: Path | None = None,
                            binary: bool = True) -> bool:
        if self._worker is not None and self._worker.isRunning():
            self._set_status("A matte build is already in progress.")
            return False
        ffmpeg = Path(self._ffmpeg_getter())
        if not ffmpeg.exists():
            QMessageBox.warning(self, "Missing ffmpeg", f"ffmpeg not found: {ffmpeg}")
            return False
        fps = self._fps_getter() or 24.0
        self._worker = MatteWorker(object_ids, role, ffmpeg, float(fps), output, binary, self)
        self._worker.finished_ok.connect(self._on_matte_done)
        self._worker.failed.connect(self._on_matte_failed)
        self._worker.progress.connect(self._set_status)
        self._set_status("Building matte video from roto masks...")
        self._worker.start()
        return True

    def request_matte(self, role: str, object_ids: list[int] | None) -> bool:
        """Build a matte video for the given objects and emit matte_ready."""
        return self._start_matte_worker(role, object_ids)

    def _on_matte_done(self, path: str, role: str) -> None:
        self._set_status(f"Matte ready: {path}")
        self.matte_ready.emit(path, role)

    def _on_matte_failed(self, message: str, role: str) -> None:
        self._set_status(f"Matte build failed: {message}")
        QMessageBox.warning(self, "Matte build failed", message)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)
        self.status_message.emit(text)
