from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import Qt, QTimer
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
from core.paths import clear_workspace, default_binary_paths, default_temp_root
from core.pipeline import PipelineConfig, PipelineRunner
from core.pointcloud import parse_points3d_txt
from core.project import load_project, save_project
from core.video_info import probe_video
from ui.ar_preview_widget import ARPreviewWidget
from ui.dependencies_widget import DependenciesWidget, missing_dependencies
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
        self._dirty = False
        self._workspace_dir: Path | None = None

        self._build_ui()
        self._set_defaults()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(16, 16, 16, 16)

        # --- Menubar ---
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
        self._save_project_btn.setEnabled(False)
        self._save_project_btn.clicked.connect(lambda: self._save_project(save_as=False))
        btn_layout.addWidget(self._save_project_btn)

        self._save_as_btn = QPushButton("Save Project As...")
        self._save_as_btn.setMinimumHeight(42)
        self._save_as_btn.setEnabled(False)
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

        # Roto tab
        self._roto = RotoWidget(
            ffmpeg_getter=lambda: self._ffmpeg_path.text(),
            fps_getter=lambda: self._video_info.get("fps") or float(self._fps_spin.value()),
            video_getter=lambda: self._video_file,
        )
        self._roto.objects_changed.connect(self._refresh_mask_combos)
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

        self._matte_shrink_spin = QSpinBox()
        self._matte_shrink_spin.setRange(-200, 200)
        self._matte_shrink_spin.setValue(0)
        self._matte_shrink_spin.setToolTip(
            "Positive values grow the ignored matte (~1px per step), negative values shrink it."
            " Subject-edge features often need +10 to +30 to fully exclude the silhouette."
        )
        form.addRow("Matte grow/shrink (px):", self._matte_shrink_spin)
        page.addWidget(tune_group)

        opts_group = QGroupBox("Options")
        opts = QVBoxLayout(opts_group)
        self._gpu_check = QCheckBox("Use GPU for SIFT")
        self._gpu_check.setToolTip("Use the GPU for feature extraction/matching. The reference workflow uses CPU.")
        opts.addWidget(self._gpu_check)
        self._cleanup_check = QCheckBox("Clean workspace on exit")
        opts.addWidget(self._cleanup_check)
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
        self._deps_widget = DependenciesWidget()
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
        missing = missing_dependencies({
            "FFmpeg": self._ffmpeg_path.text(),
            "COLMAP": self._colmap_path.text(),
            "GLOMAP": self._glomap_path.text(),
        })
        if not missing:
            return
        answer = QMessageBox.question(
            self,
            "Missing dependencies",
            "Some dependencies aren't set up yet:\n\n• "
            + "\n• ".join(missing)
            + "\n\nOpen the Dependencies manager to review and install them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._show_advanced(1)

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
        idx = combo.findData(current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

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
                self._tabs.setCurrentWidget(self._roto)
                self._roto.open_editor(video)
                self._log.append("Video sent to the Roto tab — masks will appear in these dropdowns.")
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

    def _on_matte_ready(self, path: str, role: str) -> None:
        if role not in ("ignore", "object"):
            self._log.append(f"Roto matte exported: {path}")
            return
        target = self._matte_path if role == "ignore" else self._object_mask_path
        combo = self._matte_src if role == "ignore" else self._objmask_src
        target.setText(path)
        self._log.append(f"Roto matte ready for {'ignore matte' if role == 'ignore' else 'object mask'}: {path}")
        combo.blockSignals(True)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _set_defaults(self) -> None:
        bins = default_binary_paths()
        self._ffmpeg_path.setText(str(bins["ffmpeg"]))
        self._colmap_path.setText(str(bins["colmap"]))
        self._glomap_path.setText(str(bins["glomap"]))
        self._out_path.setText(str(default_temp_root()))
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
            output_dir=Path(self._out_path.text()),
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
            "smoothing_window": self._smoothing_window,
            "use_gpu": self._gpu_check.isChecked(),
            "binary_paths": {
                "ffmpeg": self._ffmpeg_path.text(),
                "colmap": self._colmap_path.text(),
                "glomap": self._glomap_path.text(),
            },
        }

    def _save_project(self, save_as: bool = False) -> bool:
        if not self._video_file:
            QMessageBox.warning(self, "Nothing to save", "Load a video or run tracking first.")
            return False

        source_workspace = self._workspace_dir
        if not source_workspace or not Path(source_workspace).exists():
            QMessageBox.warning(
                self, "Nothing to save", "The tracking workspace no longer exists. Run tracking or load a project first."
            )
            return False
        source_workspace = Path(source_workspace)

        if save_as or self._project_dir is None:
            default_dir = str(self._project_dir) if self._project_dir else ""
            chosen = QFileDialog.getExistingDirectory(self, "Save Project Folder", default_dir)
            if not chosen:
                return False
            project_dir_path = Path(chosen)
        else:
            project_dir_path = Path(self._project_dir)

        try:
            save_project(
                project_dir_path,
                self._video_file,
                workspace_dir=source_workspace,
                settings=self._gather_settings(),
                oriented_track=self._oriented_track,
                orientation=self._orientation,
                scale=self._scale,
                z_rotation=self._z_rotation,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            self._log.append(f"Save project failed: {exc}")
            return False

        self._project_dir = project_dir_path
        self._workspace_dir = project_dir_path / "workspace"
        self._dirty = False
        self._log.append(f"Project saved to: {project_dir_path}")
        return True

    def _load_project(self) -> None:
        project_file_str, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "GLOMAP Project (project.json)"
        )
        if not project_file_str:
            return

        try:
            data = load_project(Path(project_file_str))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self._log.append(f"Load project failed: {exc}")
            return

        if "raw_track" not in data:
            msg = f"No tracking data found in {data['sparse_dir']}. The project's workspace folder is missing or incomplete."
            QMessageBox.critical(self, "Load failed", msg)
            self._log.append(f"Load project failed: {msg}")
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
        self._ar_preview._smooth_window.setValue(settings.get("smoothing_window", 0))
        self._smoothing_window = settings.get("smoothing_window", 0)
        self._gpu_check.setChecked(settings.get("use_gpu", False))
        if "binary_paths" in settings:
            self._ffmpeg_path.setText(settings["binary_paths"].get("ffmpeg", ""))
            self._colmap_path.setText(settings["binary_paths"].get("colmap", ""))
            self._glomap_path.setText(settings["binary_paths"].get("glomap", ""))

        # Restore video and metadata.
        self._video_file = data["input_video"]
        self._load_video(self._video_file)

        # Restore tracking data.
        self._project_dir = data.get("project_dir")
        self._workspace_dir = data.get("workspace_dir")
        self._sparse_dir = data["sparse_dir"]
        self._frames_dir = data["frames_dir"]
        self._raw_track = data["raw_track"]
        self._raw_track_colmap = data["raw_track_colmap"]
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
        self._log.append(f"Project loaded from: {project_file_str}")

    def closeEvent(self, event) -> None:
        try:
            self._roto.shutdown()
        except Exception:
            pass
        if self._dirty and self._raw_track:
            answer = QMessageBox.question(
                self,
                "Unsaved project",
                "The project has unsaved changes. Save before closing?",
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
        if self._cleanup_check.isChecked() and self._runner and self._runner.workspace:
            try:
                clear_workspace(self._runner.workspace.base)
            except Exception:
                pass
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    apply_stylesheet(app)
    window = MainWindow()
    window.show()
    return app.exec()
