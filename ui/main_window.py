from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.exporters.ae_jsx import generate_jsx as generate_ae_jsx
from core.exporters.blender_py import generate_py as generate_blender_py
from core.orientation import (
    apply_orientation,
    apply_scale,
    estimate_world_forward,
    estimate_world_up,
    resolve_floor_and_wall,
)
from core.parser import build_camera_track
from core.smoothing import smooth_object_track, smooth_track
from core.learned_features import learned_unavailable_reason
from core.paths import (
    clear_workspace,
    default_binary_paths,
    is_temp_path,
    purge_stale_temp,
    session_temp_root,
)
from core.pipeline import PipelineConfig, PipelineRunner
from core.pointcloud import parse_points3d_txt
from core.project import load_project, save_project
from core.video_info import probe_video
from ui.ar_preview_widget import ARPreviewWidget
from ui.dependencies_widget import DependenciesWidget, missing_downloads
from ui.file_dialogs import install as install_file_dialog_memory
from ui.roto_widget import RotoWidget
from ui.undistort_widget import UndistortWidget
from ui.style import apply as apply_stylesheet
from ui.video_preview_widget import VideoPreviewWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OnTrack")
        self.resize(1100, 750)
        self.setAcceptDrops(True)

        self._runner: PipelineRunner | None = None
        self._sparse_dir: Path | None = None
        self._fps = 24.0
        self._video_file: Path | None = None
        self._video_info: dict[str, Any] = {}
        self._raw_track: list[dict[str, Any]] | None = None
        self._raw_track_colmap: list[dict[str, Any]] | None = None
        self._oriented_track: list[dict[str, Any]] | None = None
        self._object_track: list[dict[str, Any]] | None = None
        self._point_cloud: list[dict[str, Any]] | None = None
        self._frames_dir: Path | None = None
        self._orientation: dict[str, Any] | None = None
        self._last_config: PipelineConfig | None = None
        self._smoothing_window: int = 0
        self._scale: float = 1.0
        self._z_rotation: float = 0.0
        self._project_dir: Path | None = None
        self._project_file: Path | None = None
        self._dirty = False
        self._workspace_dir: Path | None = None
        # Scratch workspaces/mattes this instance created in the OS temp dir;
        # deleted when superseded and on exit.
        self._temp_workspaces: list[Path] = []
        self._temp_mattes: set[Path] = set()
        self._pending_mask_role: str | None = None   # set by "Create new mask…"

        purge_stale_temp()
        self._build_ui()
        self._set_defaults()
        for edit in self.findChildren(QLineEdit):
            edit.textChanged.connect(lambda *_: setattr(self, "_dirty", True))
        for combo in self.findChildren(QComboBox):
            combo.currentIndexChanged.connect(lambda *_: setattr(self, "_dirty", True))
        for spin in self.findChildren(QSpinBox):
            spin.valueChanged.connect(lambda *_: setattr(self, "_dirty", True))
        for check in self.findChildren(QCheckBox):
            check.toggled.connect(lambda *_: setattr(self, "_dirty", True))

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(16, 16, 16, 16)

        # --- Menubar ---
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Open Project…", self._load_project).setShortcut("Ctrl+O")
        file_menu.addAction("Save", lambda: self._save_project(False)).setShortcut("Ctrl+S")
        file_menu.addAction("Save As…", lambda: self._save_project(True)).setShortcut("Ctrl+Shift+S")
        settings_menu = self.menuBar().addMenu("&Settings")
        settings_menu.addAction("Tracking Settings…", lambda: self._show_advanced(0))
        settings_menu.addAction("Dependencies…", lambda: self._show_advanced(1))

        # --- Header ---
        header = QLabel("OnTrack")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #f0f0f0;")
        root_layout.addWidget(header)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(16)

        # --- Tabbed controls (full width) ---
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabBar::tab {
                background-color: #1e1e1e;
                color: #a0a0a0;
                padding: 8px 16px;
                border: 1px solid #333333;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #252525;
                color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #333333;
                background-color: #121212;
            }
        """)

        # Track tab
        track_tab = QWidget()
        track_layout = QHBoxLayout(track_tab)
        track_layout.setSpacing(16)

        # Left: video preview + metadata
        track_left = QVBoxLayout()
        track_left.setSpacing(12)

        self._video_preview = VideoPreviewWidget()
        self._video_preview.setMinimumSize(480, 270)
        track_left.addWidget(self._video_preview, 1)

        btn_layout = QHBoxLayout()
        self._add_video_btn = QPushButton("Add Video")
        self._add_video_btn.setMinimumHeight(42)
        self._add_video_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_video_btn.setObjectName("addVideoButton")
        self._add_video_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 13px;
                background-color: #3a7bd5;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #4a8be5; }
            QPushButton:pressed { background-color: #2a6bc5; }
        """)
        self._add_video_btn.clicked.connect(self._pick_video)
        btn_layout.addWidget(self._add_video_btn, 1)

        self._load_project_btn = QPushButton("Load Project")
        self._load_project_btn.setMinimumHeight(42)
        self._load_project_btn.clicked.connect(self._load_project)
        btn_layout.addWidget(self._load_project_btn)

        self._save_project_btn = QPushButton("Save Project")
        self._save_project_btn.setMinimumHeight(42)
        self._save_project_btn.setEnabled(True)
        self._save_project_btn.clicked.connect(lambda: self._save_project(save_as=False))
        btn_layout.addWidget(self._save_project_btn)

        self._save_as_btn = QPushButton("Save Project As...")
        self._save_as_btn.setMinimumHeight(42)
        self._save_as_btn.setEnabled(True)
        self._save_as_btn.clicked.connect(lambda: self._save_project(save_as=True))
        btn_layout.addWidget(self._save_as_btn)

        track_left.addLayout(btn_layout)

        self._drop_hint = QLabel("Supported: .mp4 .mov .avi .mkv  —  drag & drop or click Add Video")
        self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setStyleSheet("color: #888888; font-size: 11px;")
        track_left.addWidget(self._drop_hint)

        meta_group = QGroupBox("Video Info")
        meta_layout = QFormLayout(meta_group)
        self._meta_filename = QLabel("-")
        self._meta_filename.setWordWrap(True)
        self._meta_resolution = QLabel("-")
        self._meta_duration = QLabel("-")
        self._meta_fps = QLabel("-")
        self._meta_frames = QLabel("-")
        meta_layout.addRow("File:", self._meta_filename)
        meta_layout.addRow("Resolution:", self._meta_resolution)
        meta_layout.addRow("Duration:", self._meta_duration)
        meta_layout.addRow("FPS:", self._meta_fps)
        meta_layout.addRow("Frames:", self._meta_frames)
        track_left.addWidget(meta_group)

        track_layout.addLayout(track_left, 1)

        # Right: output / binaries / settings / run / log
        track_right = QVBoxLayout()
        track_right.setSpacing(12)

        # Output
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        self._out_path = self._file_row(out_layout, "Output folder:", "Browse", self._pick_output, directory=True)
        self._out_path.setPlaceholderText(
            "Temporary — cleared on exit unless you Save Project"
        )
        self._out_path.setToolTip(
            f"Leave empty to work in {session_temp_root()} (deleted on exit; Save Project keeps the results).\n"
            "Choose a folder to keep every run's workspace there permanently."
        )
        self._matte_src = self._mask_source_combo("ignore")
        self._matte_path = self._file_row(
            out_layout, "Ignore matte (optional):", "Browse", self._pick_matte, extra=self._matte_src
        )
        self._matte_path.setPlaceholderText(
            "Black/white matte video matching the frame count — white regions are ignored"
        )
        self._objmask_src = self._mask_source_combo("object")
        self._object_mask_path = self._file_row(
            out_layout, "Object mask (object track):", "Browse", self._pick_object_mask, extra=self._objmask_src
        )
        self._object_mask_path.setPlaceholderText(
            "White/alpha mask video isolating the moving object — required for Object Track mode"
        )
        self._object_mask_path.setEnabled(False)
        self._objmask_src.setEnabled(False)

        grow_row = QHBoxLayout()
        grow_row.addWidget(QLabel("Matte grow/shrink (px):"))
        self._matte_shrink_spin = QSpinBox()
        self._matte_shrink_spin.setRange(-200, 200)
        self._matte_shrink_spin.setValue(0)
        self._matte_shrink_spin.setToolTip(
            "Positive values grow the ignored matte (~1px per step), negative values shrink it."
            " Subject-edge features often need +10 to +30 to fully exclude the silhouette."
        )
        grow_row.addWidget(self._matte_shrink_spin)
        grow_row.addStretch()
        out_layout.addLayout(grow_row)
        track_right.addWidget(out_group)

        # Settings (essentials only — advanced options live in the Settings menu)
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 240)
        self._fps_spin.setValue(24)
        settings_layout.addWidget(self._fps_spin)

        settings_layout.addWidget(QLabel("Frame skip:"))
        self._frame_skip_spin = QSpinBox()
        self._frame_skip_spin.setRange(0, 100)
        self._frame_skip_spin.setValue(0)
        settings_layout.addWidget(self._frame_skip_spin)

        settings_layout.addWidget(QLabel("Track type:"))
        self._track_type_combo = QComboBox()
        self._track_type_combo.addItem("Classic (SIFT)", "sift")
        self._track_type_combo.addItem("Learned (SuperPoint + LightGlue)", "learned")
        self._track_type_combo.addItem("Object Track", "object")
        self._track_type_combo.setToolTip(
            "Classic: COLMAP SIFT features, best for sharp, textured footage.\n"
            "Learned: neural features on the GPU (DirectML), more robust on blurry or low-texture shots,"
            " but slower and with slightly lower sub-pixel precision.\n"
            "Object Track: solves a rigid moving object using an object mask, after the camera is tracked."
        )
        self._track_type_combo.currentIndexChanged.connect(self._on_track_type_changed)
        settings_layout.addWidget(self._track_type_combo)

        advanced_btn = QPushButton("Advanced…")
        advanced_btn.clicked.connect(lambda: self._show_advanced(0))
        settings_layout.addWidget(advanced_btn)
        settings_layout.addStretch()

        track_right.addWidget(settings_group)

        # Advanced settings live in a dialog so the tab stays clean. The
        # widgets are created up front because _set_defaults() fills them.
        self._build_advanced_dialog()

        # Run
        run_layout = QHBoxLayout()
        self._run_btn = QPushButton("Run Tracking")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setObjectName("runButton")
        self._run_btn.clicked.connect(self._run_tracking)
        run_layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setObjectName("cancelButton")
        self._cancel_btn.clicked.connect(self._cancel_tracking)
        run_layout.addWidget(self._cancel_btn)

        run_layout.addStretch()
        track_right.addLayout(run_layout)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        track_right.addWidget(self._progress)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Process output will appear here...")
        track_right.addWidget(self._log)

        track_layout.addLayout(track_right, 1)

        self._tabs.addTab(track_tab, "Track")
        self._track_tab = track_tab

        # Roto tab
        self._roto = RotoWidget(
            ffmpeg_getter=lambda: self._ffmpeg_path.text(),
            fps_getter=lambda: self._video_info.get("fps") or float(self._fps_spin.value()),
            video_getter=lambda: self._video_file,
        )
        self._roto.objects_changed.connect(self._refresh_mask_combos)
        self._roto.objects_changed.connect(lambda: setattr(self, "_dirty", True))
        self._roto.session_changed.connect(lambda: setattr(self, "_dirty", True))
        self._roto.matte_ready.connect(self._on_matte_ready)
        self._roto.status_message.connect(self._log.append)
        self._tabs.insertTab(0, self._roto, "Roto")

        # Preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self._ar_preview = ARPreviewWidget()
        self._ar_preview.orientation_applied.connect(self._on_ar_orientation_applied)
        self._ar_preview.origin_set.connect(self._on_ar_origin_set)
        self._ar_preview.scale_applied.connect(self._on_ar_scale_applied)
        self._ar_preview.z_rotation_changed.connect(self._on_ar_z_rotation_changed)
        self._ar_preview.resolve_requested.connect(self._on_resolve_requested)
        self._ar_preview.smoothing_changed.connect(self._on_smoothing_changed)
        preview_layout.addWidget(self._ar_preview)
        self._tabs.addTab(preview_tab, "Preview")

        # Undistort tab
        undistort_tab = UndistortWidget(
            ffmpeg_path_getter=lambda: self._ffmpeg_path.text())
        self._undistort = undistort_tab
        self._tabs.insertTab(1, undistort_tab, "Undistort")

        # Export tab
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        export_layout.setContentsMargins(8, 8, 8, 8)
        export_layout.setSpacing(12)

        export_btn_layout = QHBoxLayout()
        self._export_ae_btn = QPushButton("Export After Effects (.jsx)")
        self._export_ae_btn.setEnabled(False)
        self._export_ae_btn.clicked.connect(self._export_ae)
        export_btn_layout.addWidget(self._export_ae_btn)

        self._export_blender_btn = QPushButton("Export Blender (.py)")
        self._export_blender_btn.setEnabled(False)
        self._export_blender_btn.clicked.connect(self._export_blender)
        export_btn_layout.addWidget(self._export_blender_btn)

        export_btn_layout.addStretch()
        export_layout.addLayout(export_btn_layout)
        export_layout.addStretch()
        self._tabs.addTab(export_tab, "Export")

        self._tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self._tabs, 1)
        root_layout.addLayout(main_layout, 1)

    def _build_advanced_dialog(self) -> None:
        """Settings dialog: 'Tracking' page for tuning knobs and a
        'Dependencies' page for binaries + downloadable models."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumSize(720, 560)
        layout = QVBoxLayout(dlg)

        tabs = QTabWidget(dlg)

        # ---- Tracking page ----
        tracking_page = QWidget()
        page = QVBoxLayout(tracking_page)

        tune_group = QGroupBox("Tracking Tuning")
        form = QFormLayout(tune_group)

        self._max_size_spin = QSpinBox()
        self._max_size_spin.setRange(0, 8192)
        self._max_size_spin.setValue(0)
        self._max_size_spin.setSpecialValueText("None")
        self._max_size_spin.setToolTip(
            "Default is full resolution, matching the reference batch workflow.\n"
            "4K+ footage: set ~1920 for a much faster, more stable CPU SIFT extraction."
        )
        form.addRow("Max frame size (px):", self._max_size_spin)

        self._matcher_combo = QComboBox()
        self._matcher_combo.addItems(["sequential", "exhaustive"])
        self._matcher_combo.setToolTip(
            "Sequential is recommended for video. Exhaustive compares every frame against every other frame."
        )
        form.addRow("Matcher:", self._matcher_combo)
        page.addWidget(tune_group)

        opts_group = QGroupBox("Options")
        opts = QVBoxLayout(opts_group)
        self._gpu_check = QCheckBox("Use GPU for SIFT")
        self._gpu_check.setToolTip("Use the GPU for feature extraction/matching. The reference workflow uses CPU.")
        opts.addWidget(self._gpu_check)
        page.addWidget(opts_group)
        page.addStretch()
        tabs.addTab(tracking_page, "Tracking")

        # ---- Dependencies page ----
        deps_page = QWidget()
        deps_layout = QVBoxLayout(deps_page)
        bin_group = QGroupBox("Binaries (auto-detected)")
        bin_layout = QVBoxLayout(bin_group)
        self._ffmpeg_path = self._file_row(bin_layout, "FFmpeg:", "Browse", self._pick_ffmpeg)
        self._colmap_path = self._file_row(bin_layout, "COLMAP:", "Browse", self._pick_colmap)
        self._glomap_path = self._file_row(bin_layout, "GLOMAP:", "Browse", self._pick_glomap)
        deps_layout.addWidget(bin_group)
        self._deps_widget = DependenciesWidget(binary_paths_getter=self._configured_binaries)
        self._deps_widget.binaries_installed.connect(self._on_binaries_installed)
        deps_layout.addWidget(self._deps_widget, 1)
        tabs.addTab(deps_page, "Dependencies")

        layout.addWidget(tabs, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._settings_tabs = tabs
        self._advanced_dlg = dlg

    def _show_advanced(self, tab: int = 0) -> None:
        self._settings_tabs.setCurrentIndex(tab)
        self._deps_widget.refresh()
        self._advanced_dlg.exec()

    def _check_dependencies_first_run(self) -> None:
        """Prompt to download missing essentials. Shown on every launch while
        something is missing, until the user ticks "Don't ask again"."""
        prefs = QSettings("OnTrack", "OnTrack")
        if prefs.value("deps/dont_ask", False, type=bool):
            return
        downloads = missing_downloads(self._configured_binaries())
        if not downloads:
            return

        total_mb = sum(mb for _label, mb in downloads)
        box = QMessageBox(self)
        box.setWindowTitle("Set up dependencies")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(
            f"OnTrack needs to download the following (~{total_mb} MB total):\n\n• "
            + "\n• ".join(f"{label}  (~{mb} MB)" for label, mb in downloads)
            + "\n\nIf you already have FFmpeg/COLMAP/GLOMAP elsewhere, choose Later and"
            " point to them in Settings → Dependencies."
        )
        download_btn = box.addButton("Download now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        dont_ask = QCheckBox("Don't ask again")
        box.setCheckBox(dont_ask)
        box.exec()
        prefs.setValue("deps/dont_ask", dont_ask.isChecked())
        if box.clickedButton() is download_btn:
            self._settings_tabs.setCurrentIndex(1)
            self._deps_widget.refresh()
            QTimer.singleShot(0, self._deps_widget.install_essentials)
            self._advanced_dlg.exec()

    def _configured_binaries(self) -> dict[str, str]:
        return {
            "ffmpeg": self._ffmpeg_path.text(),
            "colmap": self._colmap_path.text(),
            "glomap": self._glomap_path.text(),
        }

    def _on_binaries_installed(self, keys: list) -> None:
        bins = default_binary_paths()
        edits = {"ffmpeg": self._ffmpeg_path, "colmap": self._colmap_path, "glomap": self._glomap_path}
        for key in keys:
            edits[key].setText(str(bins[key]))
            self._log.append(f"Installed {key}: {bins[key]}")
        self._deps_widget.refresh()

    def _file_row(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        button_text: str,
        callback,
        directory: bool = False,
        extra: QWidget | None = None,
    ) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit()
        edit.setMinimumWidth(300)
        row.addWidget(edit, 1)
        if extra is not None:
            row.addWidget(extra)
        btn = QPushButton(button_text)
        btn.clicked.connect(lambda: callback(edit))
        row.addWidget(btn)
        parent_layout.addLayout(row)
        return edit

    # ---- Roto mask sources ----

    def _mask_source_combo(self, role: str) -> QComboBox:
        combo = QComboBox()
        combo.setMinimumWidth(170)
        combo.setToolTip(
            "Pick a mask made in the Roto tab, or 'Create new mask' to open the roto editor."
        )
        combo.currentIndexChanged.connect(lambda _i, c=combo, r=role: self._on_mask_source(r, c))
        self._populate_mask_combo(combo)
        return combo

    def _populate_mask_combo(self, combo: QComboBox) -> None:
        combo.blockSignals(True)
        current = combo.currentData()
        combo.clear()
        combo.addItem("File / none", None)
        for obj in self._roto_objects():
            label = f"Roto: {obj.name} ({obj.frame_count}f)"
            combo.addItem(label, ("roto", obj.object_id))
        if self._roto_objects():
            combo.addItem("Roto: All objects", ("roto", "all"))
        combo.addItem("+ Create new mask…", ("new",))
        idx = self._find_combo_data(combo, current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    @staticmethod
    def _find_combo_data(combo: QComboBox, value) -> int:
        """QComboBox.findData compares Python objects by identity, so look
        tuple item data up by equality instead."""
        return next((i for i in range(combo.count()) if combo.itemData(i) == value), -1)

    def _roto_objects(self):
        return self._roto.objects() if hasattr(self, "_roto") else []

    def _refresh_mask_combos(self) -> None:
        self._populate_mask_combo(self._matte_src)
        self._populate_mask_combo(self._objmask_src)
        if self._roto_objects():
            self._log.append("Roto masks detected — select them via the mask dropdowns in the Track tab.")

    def _on_mask_source(self, role: str, combo: QComboBox) -> None:
        data = combo.currentData()
        if not data:
            return
        if data[0] == "new":
            video = self._video_file
            if video is None:
                QMessageBox.information(self, "No video", "Load a video first, then create a mask.")
            else:
                self._pending_mask_role = role
                self._tabs.setCurrentWidget(self._roto)
                self._roto.open_editor(video)
                self._log.append(
                    "Video sent to the Roto tab — the mask you create there is loaded"
                    " automatically when you come back to the Track tab."
                )
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
            return
        if data[0] == "roto":
            object_ids = None if data[1] == "all" else [int(data[1])]
            if not self._roto.request_matte(role, object_ids):
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)

    def _on_tab_changed(self, _index: int) -> None:
        """Returning to Track after 'Create new mask': pick up the new mask."""
        role = self._pending_mask_role
        if role is None or self._tabs.currentWidget() is not self._track_tab:
            return
        self._roto.refresh_objects()
        objects = self._roto_objects()
        if not objects:
            self._log.append("No roto masks yet — create one in the Roto tab, or pick a file.")
            return
        self._pending_mask_role = None
        combo = self._matte_src if role == "ignore" else self._objmask_src
        self._populate_mask_combo(combo)
        target = ("roto", objects[0].object_id) if len(objects) == 1 else ("roto", "all")
        idx = self._find_combo_data(combo, target)
        if idx >= 0:
            combo.setCurrentIndex(idx)   # triggers _on_mask_source → matte build

    def _on_matte_ready(self, path: str, role: str) -> None:
        if role not in ("ignore", "object"):
            self._log.append(f"Roto matte exported: {path}")
            return
        target = self._matte_path if role == "ignore" else self._object_mask_path
        combo = self._matte_src if role == "ignore" else self._objmask_src
        target.setText(path)
        if is_temp_path(path):
            self._temp_mattes.add(Path(path))
        self._log.append(f"Roto matte ready for {'ignore matte' if role == 'ignore' else 'object mask'}: {path}")
        combo.blockSignals(True)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _set_defaults(self) -> None:
        bins = default_binary_paths()
        self._ffmpeg_path.setText(str(bins["ffmpeg"]))
        self._colmap_path.setText(str(bins["colmap"]))
        self._glomap_path.setText(str(bins["glomap"]))
        QTimer.singleShot(800, self._check_dependencies_first_run)

    # ---- Pickers ----

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "", "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*.*)"
        )
        if path:
            self._load_video(Path(path))

    def _pick_output(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            edit.setText(path)

    def _pick_matte(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ignore Matte",
            "",
            "Videos/Images (*.mp4 *.mov *.avi *.mkv *.png *.jpg);;All files (*.*)",
        )
        if path:
            edit.setText(path)

    def _pick_object_mask(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Object Mask",
            "",
            "Videos/Images (*.mp4 *.mov *.avi *.mkv *.png *.jpg);;All files (*.*)",
        )
        if path:
            edit.setText(path)

    def _pick_ffmpeg(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ffmpeg.exe", "", "Executable (*.exe)")
        if path:
            edit.setText(path)

    def _pick_colmap(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select colmap.exe", "", "Executable (*.exe)")
        if path:
            edit.setText(path)

    def _pick_glomap(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select glomap.exe", "", "Executable (*.exe)")
        if path:
            edit.setText(path)

    # ---- Video loading ----

    def _load_video(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(self, "Missing file", f"Video not found:\n{path}")
            return

        self._video_file = path
        self._dirty = True
        self._log.append(f"Loaded video: {path}")

        ffprobe = Path(self._ffmpeg_path.text()).parent / "ffprobe.exe"
        self._video_info = probe_video(path, ffprobe if ffprobe.exists() else None)
        self._update_metadata()

        # Auto-set FPS from video metadata if available.
        if self._video_info.get("fps"):
            fps = round(self._video_info["fps"])
            self._fps_spin.setValue(fps)

        self._video_preview.load_video(path)

    def _update_metadata(self) -> None:
        info = self._video_info
        if not info:
            self._meta_filename.setText("-")
            self._meta_resolution.setText("-")
            self._meta_duration.setText("-")
            self._meta_fps.setText("-")
            self._meta_frames.setText("-")
            return

        self._meta_filename.setText(info.get("filename", "-"))

        width = info.get("width")
        height = info.get("height")
        self._meta_resolution.setText(
            f"{width} x {height}" if width and height else "-"
        )

        duration = info.get("duration")
        if duration is not None:
            minutes = int(duration // 60)
            seconds = duration % 60
            self._meta_duration.setText(f"{minutes}:{seconds:05.2f}")
        else:
            self._meta_duration.setText("-")

        fps = info.get("fps")
        self._meta_fps.setText(f"{fps:.3f}" if fps else "-")

        frame_count = info.get("frame_count")
        self._meta_frames.setText(str(frame_count) if frame_count else "-")

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return

        path = Path(urls[0].toLocalFile())
        ext = path.suffix.lower()
        if ext in {".mp4", ".mov", ".avi", ".mkv"}:
            self._load_video(path)
            event.acceptProposedAction()
        else:
            QMessageBox.warning(self, "Invalid file", "Please drop a video file (.mp4, .mov, .avi, .mkv).")
            event.ignore()

    # ---- Tracking ----

    def _on_track_type_changed(self) -> None:
        is_sift = self._track_type_combo.currentData() == "sift"
        is_object = self._track_type_combo.currentData() == "object"
        self._gpu_check.setEnabled(is_sift)
        self._object_mask_path.setEnabled(is_object)
        self._objmask_src.setEnabled(is_object)
        if not is_object:
            self._object_mask_path.clear()

    def _run_tracking(self) -> None:
        if not self._video_file or not self._video_file.exists():
            QMessageBox.warning(self, "Missing input", "Please select a valid input video.")
            return

        for name, path in [("FFmpeg", self._ffmpeg_path.text()),
                           ("COLMAP", self._colmap_path.text()),
                           ("GLOMAP", self._glomap_path.text())]:
            if not Path(path).exists():
                QMessageBox.warning(self, "Missing binary", f"{name} binary not found: {path}")
                return

        if self._track_type_combo.currentData() == "learned":
            reason = learned_unavailable_reason()
            if reason:
                QMessageBox.warning(self, "Learned tracking unavailable", reason)
                return

        if self._track_type_combo.currentData() == "object":
            reason = learned_unavailable_reason()
            if reason:
                QMessageBox.warning(self, "Object tracking unavailable", reason)
                return
            mask_text = self._object_mask_path.text().strip()
            if not mask_text or not Path(mask_text).exists():
                QMessageBox.warning(self, "Missing object mask", "Object Track mode requires a valid object mask video.")
                return

        self._log.clear()
        self._progress.setValue(0)
        self._sparse_dir = None
        self._export_ae_btn.setEnabled(False)
        self._export_blender_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        matte_text = self._matte_path.text().strip()
        mask_video = Path(matte_text) if matte_text else None
        if mask_video is not None and not mask_video.exists():
            QMessageBox.warning(self, "Missing matte", f"Matte file not found: {mask_video}")
            return

        obj_mask_text = self._object_mask_path.text().strip()
        object_mask_video = Path(obj_mask_text) if obj_mask_text else None
        if self._track_type_combo.currentData() == "object" and (object_mask_video is None or not object_mask_video.exists()):
            QMessageBox.warning(self, "Missing object mask", "Object Track mode requires a valid object mask video.")
            return

        config = PipelineConfig(
            input_video=self._video_file,
            output_dir=Path(self._out_path.text().strip() or session_temp_root()),
            mask_video=mask_video,
            object_mask_video=object_mask_video,
            object_primitive="box",
            object_size=(1.0, 1.0, 1.0),
            mask_grow=self._matte_shrink_spin.value(),
            ffmpeg=Path(self._ffmpeg_path.text()),
            colmap=Path(self._colmap_path.text()),
            glomap=Path(self._glomap_path.text()),
            fps=float(self._fps_spin.value()),
            max_size=self._max_size_spin.value() or None,
            frame_skip=self._frame_skip_spin.value(),
            matcher_type=self._matcher_combo.currentText(),
            use_gpu=self._gpu_check.isChecked(),
            feature_mode=self._track_type_combo.currentData(),
        )

        self._fps = config.fps
        self._last_config = config
        self._runner = PipelineRunner(config)
        self._runner.log.connect(self._on_log)
        self._runner.progress.connect(self._on_progress)
        self._runner.finished.connect(self._on_finished)
        self._runner.export_ready.connect(self._on_export_ready)
        self._runner.start()

    def _cancel_tracking(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self._log.append("Cancelling...")
        self._cancel_btn.setEnabled(False)

    def _on_log(self, message: str) -> None:
        self._log.append(message)

    def _on_progress(self, percent: int, step: str) -> None:
        self._progress.setValue(percent)
        self.statusBar().showMessage(f"{step} ({percent}%)")

    def _on_finished(self, success: bool, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if success:
            self._progress.setValue(100)
            self._log.append(message)
            self._update_export_buttons()
        else:
            self._log.append(f"Failed: {message}")
            QMessageBox.critical(self, "Tracking failed", message)

    def _on_export_ready(self, sparse_dir: str) -> None:
        try:
            self._handle_export_ready(Path(sparse_dir))
        except Exception as exc:
            self._log.append(f"Post-processing failed: {exc}")

    def _handle_export_ready(self, sparse_dir: Path) -> None:
        self._sparse_dir = sparse_dir
        self._log.append(f"Sparse model ready: {sparse_dir}")

        # Load object track if it exists.
        object_track_path = self._sparse_dir.parent.parent / "06_object" / "object_track.json"
        if object_track_path.exists():
            try:
                import json
                with open(object_track_path) as f:
                    self._object_track = json.load(f)
                self._log.append(f"Loaded {len(self._object_track)} object track frames.")
            except Exception as exc:
                self._log.append(f"Could not load object track: {exc}")
                self._object_track = None

        cameras_txt = self._sparse_dir / "cameras.txt"
        images_txt = self._sparse_dir / "images.txt"
        if not cameras_txt.exists() or not images_txt.exists():
            return

        try:
            self._raw_track = build_camera_track(images_txt, cameras_txt, target="blender")
            self._raw_track_colmap = build_camera_track(images_txt, cameras_txt, target="raw")
            self._log.append(f"Loaded {len(self._raw_track)} keyframes for preview/orientation.")
        except Exception as exc:
            self._log.append(f"Could not build preview track: {exc}")
            return

        # Load extracted frames and point cloud into the AR preview.
        if self._runner is not None and self._runner.workspace is not None:
            self._frames_dir = self._runner.workspace.frames
            points3d_txt = self._runner.workspace.sparse_model_dir() / "points3D.txt"
            self._point_cloud = parse_points3d_txt(points3d_txt) if points3d_txt.exists() else []
            try:
                self._ar_preview.load_track(
                    self._frames_dir, self._raw_track_colmap, self._point_cloud,
                    z_rotation=self._z_rotation, object_track=self._object_track
                )
                self._log.append(f"AR preview ready ({len(self._point_cloud)} point-cloud points).")
            except Exception as exc:
                self._log.append(f"Could not load AR preview: {exc}")

        self._workspace_dir = self._runner.workspace.base if self._runner and self._runner.workspace else self._workspace_dir
        self._adopt_temp_workspace(self._workspace_dir)
        self._save_project_btn.setEnabled(True)
        self._save_as_btn.setEnabled(True)
        self._dirty = True

    # ---- Export ----

    def _do_export(self, target: str) -> None:
        self._log.append(f"[export debug] target={target}")
        track = self._oriented_track if self._oriented_track is not None else self._raw_track
        if not track:
            QMessageBox.warning(self, "No track", "Run tracking first.")
            return
        object_track = self._object_track
        if self._smoothing_window >= 3:
            track = smooth_track(track, self._smoothing_window)
            if object_track:
                object_track = smooth_object_track(object_track, self._smoothing_window)

        cloud = self._point_cloud or []
        if self._orientation is not None:
            R = np.asarray(self._orientation["rotation"])
            origin = np.asarray(self._orientation["origin"])
            if abs(self._z_rotation) > 1e-6:
                rad = math.radians(self._z_rotation)
                c, s = math.cos(rad), math.sin(rad)
                Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                R = Rz @ R
            cloud = [
                {**p, "position": (R @ (np.asarray(p["position"]) - origin)).tolist()}
                for p in cloud
            ]

        try:
            if target == "ae":
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save After Effects Script", "GLOMAP_Camera_Track.jsx", "JSX (*.jsx)"
                )
                if path:
                    generate_ae_jsx(track, self._fps, Path(path), point_cloud=cloud, object_track=object_track)
                    self._log.append(f"Exported AE script: {path}")
            else:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save Blender Script", "GLOMAP_Camera_Track.py", "Python (*.py)"
                )
                if path:
                    generate_blender_py(
                        track,
                        self._fps,
                        Path(path),
                        point_cloud=cloud,
                        world_space="blender" if self._oriented_track is not None else "glomap",
                        object_track=object_track,
                    )
                    self._log.append(f"Exported Blender script: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            self._log.append(f"Export error: {exc}")

    def _export_ae(self) -> None:
        self._do_export("ae")

    def _export_blender(self) -> None:
        self._do_export("blender")

    def _update_export_buttons(self) -> None:
        has_track = bool(self._raw_track or self._oriented_track)
        self._export_ae_btn.setEnabled(has_track)
        self._export_blender_btn.setEnabled(has_track)

    def _on_resolve_requested(self, bad_ids: list[int]) -> None:
        """Prune the given points3D ids and re-run GLOMAP on the same workspace."""
        if self._last_config is None or self._workspace_dir is None:
            QMessageBox.warning(self, "Cannot re-solve", "Run a full track first.")
            return
        if not bad_ids:
            QMessageBox.information(self, "Nothing to prune", "No points exceed the threshold.")
            return
        if self._runner is not None and self._runner.is_running():
            QMessageBox.warning(self, "Busy", "A tracking run is already in progress.")
            return

        try:
            from dataclasses import replace
            config = replace(
                self._last_config,
                resolve_only=True,
                prune_point_ids=list(map(int, bad_ids)),
                resume_workspace=Path(self._workspace_dir),
            )
            self._run_btn.setEnabled(False)
            self._cancel_btn.setEnabled(True)
            self._log.append(f"Re-solve requested: pruning {len(bad_ids)} points...")
            self._runner = PipelineRunner(config)
            self._runner.log.connect(self._on_log)
            self._runner.progress.connect(self._on_progress)
            self._runner.finished.connect(self._on_finished)
            self._runner.export_ready.connect(self._on_export_ready)
            self._runner.start()
        except Exception as exc:
            self._run_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)
            self._log.append(f"Re-solve failed to start: {exc}")
            QMessageBox.critical(self, "Re-solve failed", str(exc))

    def _on_smoothing_changed(self) -> None:
        """Re-load the preview and exports with Savitzky-Golay smoothed poses."""
        self._smoothing_window = int(self._ar_preview._smooth_window.value())
        if self._raw_track_colmap is None or self._frames_dir is None:
            return
        orientation = None
        if self._orientation is not None:
            orientation = (
                np.asarray(self._orientation["rotation"]),
                np.asarray(self._orientation["origin"]),
            )
        track = self._raw_track_colmap
        obj = self._object_track
        if self._smoothing_window >= 3:
            track = smooth_track(track, self._smoothing_window)
            if obj:
                obj = smooth_object_track(obj, self._smoothing_window)
        self._ar_preview.load_track(
            self._frames_dir, track, self._point_cloud, orientation,
            z_rotation=self._z_rotation, object_track=obj,
        )
        if self._smoothing_window >= 3:
            self._log.append(f"Temporal smoothing applied (window {self._smoothing_window} frames).")
        else:
            self._log.append("Temporal smoothing disabled.")
        self._dirty = True

    def _on_ar_orientation_applied(
        self,
        floor_positions: list[np.ndarray],
        wall_positions: list[np.ndarray],
        origin_position: np.ndarray | None,
    ) -> None:
        if self._raw_track is None:
            return
        rotation, origin = resolve_floor_and_wall(
            floor_positions,
            wall_positions,
            origin_position,
            up_hint=estimate_world_up(self._raw_track),
            forward_hint=estimate_world_forward(self._raw_track),
        )
        self._oriented_track = apply_orientation(self._raw_track, rotation, origin)
        self._orientation = {
            "rotation": rotation.tolist(),
            "origin": origin.tolist(),
            "z_rotation": self._z_rotation,
        }
        self._recompute_oriented_track()
        self._update_export_buttons()
        self._dirty = True
        self._log.append("AR preview orientation applied to main track.")
        if len(floor_positions) >= 3:
            floor_t = np.array([rotation @ (np.asarray(p) - origin) for p in floor_positions])
            self._log.append(
                "[orient debug] floor Z after orient (should be ~0): "
                f"min={floor_t[:, 2].min():.4f} max={floor_t[:, 2].max():.4f}"
            )
        self._log.append(f"[orient debug] origin (raw frame) = {np.round(origin, 4).tolist()}")
        self._log.append(f"[orient debug] rotation = {np.round(rotation, 4).tolist()}")
        first = self._oriented_track[0]
        self._log.append(
            f"[orient debug] first camera raw pos = {np.round(self._raw_track[0]['position'], 4).tolist()}, "
            f"oriented pos = {np.round(first['position'], 4).tolist()}"
        )
        fwd = rotation @ estimate_world_forward(self._raw_track)
        self._log.append(f"[orient debug] mean camera forward after orient = {np.round(fwd, 3).tolist()}")

    def _on_ar_origin_set(self, origin: np.ndarray) -> None:
        if self._raw_track is None:
            return
        if self._orientation is not None:
            rotation = np.asarray(self._orientation["rotation"])
        else:
            rotation, _ = resolve_floor_and_wall(
                [],
                [],
                np.asarray(origin),
                up_hint=estimate_world_up(self._raw_track),
                forward_hint=estimate_world_forward(self._raw_track),
            )
        self._orientation = {
            "rotation": rotation.tolist(),
            "origin": np.asarray(origin).tolist(),
            "z_rotation": self._z_rotation,
        }
        self._recompute_oriented_track()
        self._update_export_buttons()
        self._dirty = True
        self._log.append(f"AR origin set (raw frame) = {np.round(origin, 4).tolist()}")

    def _on_ar_z_rotation_changed(self, degrees: float) -> None:
        self._z_rotation = float(degrees)
        self._recompute_oriented_track()
        self._dirty = True

    def _recompute_oriented_track(self) -> None:
        """Rebuild the oriented track using orientation plus any Z rotation."""
        if self._raw_track is None:
            return
        if self._orientation is None:
            self._oriented_track = None
            return
        R = np.asarray(self._orientation["rotation"])
        origin = np.asarray(self._orientation["origin"])
        if abs(self._z_rotation) > 1e-6:
            rad = math.radians(self._z_rotation)
            c, s = math.cos(rad), math.sin(rad)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            R = Rz @ R
        self._oriented_track = apply_orientation(self._raw_track, R, origin)

    def _on_ar_scale_applied(self, scale: float) -> None:
        if self._raw_track is None or self._point_cloud is None:
            return
        if self._oriented_track is not None:
            self._oriented_track, _ = apply_scale(self._oriented_track, [], scale)
        raw_scaled, cloud_scaled = apply_scale(self._raw_track, self._point_cloud, scale)
        self._raw_track = raw_scaled
        if self._raw_track_colmap is not None:
            self._raw_track_colmap, _ = apply_scale(self._raw_track_colmap, [], scale)
        self._point_cloud = cloud_scaled
        if self._orientation is not None:
            self._orientation = {
                **self._orientation,
                "origin": (np.asarray(self._orientation["origin"]) * scale).tolist(),
            }
        self._scale *= scale
        self._dirty = True
        self._log.append(f"Applied scale factor: {scale:.4f} (total: {self._scale:.4f}).")

    # ---- Project save/load ----

    def _gather_settings(self) -> dict[str, Any]:
        return {
            "fps": self._fps_spin.value(),
            "max_size": self._max_size_spin.value(),
            "frame_skip": self._frame_skip_spin.value(),
            "matcher": self._matcher_combo.currentText(),
            "track_type": self._track_type_combo.currentData(),
            "mask_path": self._matte_path.text(),
            "object_mask_path": self._object_mask_path.text(),
            "mask_grow": self._matte_shrink_spin.value(),
            "smoothing_window": self._smoothing_window,
            "use_gpu": self._gpu_check.isChecked(),
            "binary_paths": {
                "ffmpeg": self._ffmpeg_path.text(),
                "colmap": self._colmap_path.text(),
                "glomap": self._glomap_path.text(),
            },
        }

    # ---- Temp workspace housekeeping ----

    def _adopt_temp_workspace(self, workspace: Path | None) -> None:
        """Record a scratch workspace and drop older unsaved ones it replaces."""
        if workspace is None or not is_temp_path(workspace):
            return
        workspace = Path(workspace)
        for old in self._temp_workspaces:
            if old != workspace:
                clear_workspace(old)
        self._temp_workspaces = [workspace]

    def _cleanup_temp(self) -> None:
        for path in [*self._temp_workspaces, *self._temp_mattes]:
            try:
                if path.is_dir():
                    clear_workspace(path)
                elif path.exists():
                    path.unlink()
            except OSError:
                pass
        self._temp_workspaces.clear()
        self._temp_mattes.clear()

    def _persist_temp_mattes(self, project_dir: Path) -> None:
        """Copy matte videos living in the temp folder into the project so the
        saved settings don't point at files that vanish on exit."""
        for edit in (self._matte_path, self._object_mask_path):
            src = Path(edit.text().strip()) if edit.text().strip() else None
            if src is None or not src.exists() or not is_temp_path(src):
                continue
            dest = project_dir / "mattes" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            edit.setText(str(dest))

    def _save_project(self, save_as: bool = False) -> bool:
        if self._runner is not None and self._runner.is_running():
            QMessageBox.warning(self, "Tracking in progress", "Wait for tracking to finish before saving the session.")
            return False
        source_workspace = self._workspace_dir
        if source_workspace is not None and not Path(source_workspace).exists():
            source_workspace = None

        if save_as or self._project_file is None:
            suggested = str(self._project_file) if self._project_file else f"{self._video_file.stem if self._video_file else 'Untitled'}.json"
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save Project As", suggested, "OnTrack Project (*.json)"
            )
            if not chosen:
                return False
            project_file = Path(chosen)
            if project_file.suffix.lower() != ".json":
                project_file = project_file.with_suffix(".json")
        else:
            project_file = self._project_file
        project_dir_path = project_file.parent

        try:
            roto_archive = self._roto.save_session(project_file)
            self._persist_temp_mattes(project_dir_path)
            save_project(
                project_dir_path,
                self._video_file,
                workspace_dir=source_workspace,
                settings=self._gather_settings(),
                oriented_track=self._oriented_track,
                orientation=self._orientation,
                scale=self._scale,
                z_rotation=self._z_rotation,
                project_filename=project_file.name,
                session={
                    "roto_archive": roto_archive,
                    "undistort": self._undistort.session_state(),
                    "active_tab": self._tabs.currentIndex(),
                    "object_track": self._object_track,
                },
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            self._log.append(f"Save project failed: {exc}")
            return False

        self._project_dir = project_dir_path
        self._project_file = project_file
        workspace_name = "workspace" if project_file.name == "project.json" else f"{project_file.stem}.workspace"
        self._workspace_dir = project_dir_path / workspace_name if source_workspace is not None else None
        self._dirty = False
        self._log.append(f"Project saved to: {project_dir_path}")
        return True

    def _load_project(self) -> None:
        project_file_str, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "OnTrack Project (*.json)"
        )
        if not project_file_str:
            return

        try:
            data = load_project(Path(project_file_str))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self._log.append(f"Load project failed: {exc}")
            return

        if self._runner is not None and self._runner.is_running():
            QMessageBox.warning(self, "Tracking in progress", "Wait for tracking to finish before opening a session.")
            return
        session = data.get("session", {})
        try:
            archive_name = session.get("roto_archive")
            archive_path = Path(project_file_str).parent / archive_name if archive_name else None
            if archive_path is not None and archive_path.resolve().parent != Path(project_file_str).resolve().parent:
                raise ValueError("Roto archive must be alongside the project file.")
            self._pending_mask_role = None
            self._roto.restore_session(archive_path)
            self._undistort.restore_session(session.get("undistort", {}))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        # Restore settings.
        settings = data.get("settings", {})
        self._fps_spin.setValue(settings.get("fps", 24))
        self._max_size_spin.setValue(settings.get("max_size", 0))
        self._frame_skip_spin.setValue(settings.get("frame_skip", 0))
        self._matcher_combo.setCurrentText(settings.get("matcher", "sequential"))
        self._track_type_combo.setCurrentIndex(max(0, self._track_type_combo.findData(settings.get("track_type", "sift"))))
        self._on_track_type_changed()
        self._object_mask_path.setText(settings.get("object_mask_path", ""))
        self._matte_path.setText(settings.get("mask_path", ""))
        self._matte_shrink_spin.setValue(settings.get("mask_grow", 0))
        self._ar_preview._smooth_window.setValue(settings.get("smoothing_window", 0))
        self._smoothing_window = settings.get("smoothing_window", 0)
        self._gpu_check.setChecked(settings.get("use_gpu", False))
        if "binary_paths" in settings:
            self._ffmpeg_path.setText(settings["binary_paths"].get("ffmpeg", ""))
            self._colmap_path.setText(settings["binary_paths"].get("colmap", ""))
            self._glomap_path.setText(settings["binary_paths"].get("glomap", ""))

        # Restore video and metadata.
        self._video_file = data["input_video"]
        if self._video_file is not None:
            self._load_video(self._video_file)
        else:
            self._video_info = {}
            self._video_preview.clear_session()
            self._update_metadata()
        self._fps_spin.setValue(settings.get("fps", 24))

        # Restore tracking data.
        self._project_dir = data.get("project_dir")
        self._project_file = Path(project_file_str)
        self._workspace_dir = data.get("workspace_dir")
        self._sparse_dir = data["sparse_dir"]
        self._frames_dir = data["frames_dir"]
        self._raw_track = data["raw_track"]
        self._raw_track_colmap = data["raw_track_colmap"]
        self._object_track = session.get("object_track")
        self._ar_preview.clear_session()
        self._oriented_track = data.get("oriented_track")
        self._orientation = data.get("orientation")
        self._point_cloud = data["point_cloud"]
        self._scale = data.get("scale", 1.0)
        self._z_rotation = data.get("z_rotation", 0.0)

        # Update previews.
        if self._frames_dir and self._raw_track_colmap:
            orientation = None
            if self._orientation is not None:
                orientation = (
                    np.asarray(self._orientation["rotation"]),
                    np.asarray(self._orientation["origin"]),
                )
            self._ar_preview.load_track(
                self._frames_dir,
                self._raw_track_colmap,
                self._point_cloud,
                orientation,
                z_rotation=self._z_rotation,
                object_track=self._object_track,
            )

        self._save_project_btn.setEnabled(True)
        self._save_as_btn.setEnabled(True)
        self._dirty = False
        self._update_export_buttons()
        self._tabs.setCurrentIndex(max(0, min(int(session.get("active_tab", 0)), self._tabs.count() - 1)))
        self._log.append(f"Project loaded from: {project_file_str}")

    def closeEvent(self, event) -> None:
        if self._dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved project",
                "The session has unsaved changes. Unsaved temporary data may be lost"
                " when OnTrack closes.\n\nSave before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save and not self._save_project():
                event.ignore()
                return
        if self._runner is not None and self._runner.is_running():
            self._runner.cancel()
            self._runner.wait(10000)
        try:
            self._roto.shutdown()
        except Exception:
            pass
        self._cleanup_temp()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    apply_stylesheet(app)
    install_file_dialog_memory()
    window = MainWindow()
    window.show()
    return app.exec()
