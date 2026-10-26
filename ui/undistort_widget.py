"""Undistort tab: calibrate a lens profile, then undistort footage for
tracking or redistort rendered footage for delivery."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QSizePolicy, QTextEdit,
    QVBoxLayout, QWidget, QMessageBox,
)

from core.undistort import LensProfile, VideoWarpWorker, calibrate


class CalibrateWorker(QThread):
    """Runs discorpy calibration off the UI thread."""

    log = Signal(str)
    finished = Signal(object, str)

    def __init__(self, image_path: Path, pattern: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.pattern = pattern

    def run(self) -> None:
        import traceback
        try:
            profile = calibrate(self.image_path, self.pattern,
                                log=lambda m: self.log.emit(m))
            self.finished.emit(profile, "")
        except Exception:  # noqa: BLE001
            self.finished.emit(None, traceback.format_exc())


class _PatternDialog(QDialog):
    """Fullscreen checkerboard pattern for photographing a calibration frame.
    Press Esc or click to close."""

    def __init__(self, parent=None, squares: int = 30):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self._squares = squares
        self._label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: black;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        self.close()

    def _render(self) -> None:
        w = max(self.width(), 100)
        h = max(self.height(), 100)
        cell = max(w // self._squares, 20)
        yy, xx = np.mgrid[0:h, 0:w]
        board = (((yy // cell) + (xx // cell)) % 2 * 255).astype(np.uint8)
        img = QImage(board.data, w, h, w, QImage.Format.Format_Grayscale8)
        self._label.setPixmap(QPixmap.fromImage(img.copy()))


class _CompareDialog(QDialog):
    """Shows original vs warped frame side by side."""

    def __init__(self, original: np.ndarray, warped: np.ndarray,
                 title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QHBoxLayout(self)
        self._imgs = (original, warped)
        self._labels = []
        for name in ("Original", "Warped"):
            col = QVBoxLayout()
            cap = QLabel(name, alignment=Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(1, 1)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                              QSizePolicy.Policy.Ignored)
            col.addWidget(cap)
            col.addWidget(lbl, 1)
            lay.addLayout(col)
            self._labels.append(lbl)
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.8), int(avail.height() * 0.8))
        else:
            self.resize(1280, 540)
        self._render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def _render(self) -> None:
        for lbl, arr in zip(self._labels, self._imgs):
            a = np.ascontiguousarray(arr)
            h, w = a.shape[:2]
            if a.ndim == 2:
                img = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                a = a[:, :, :3]
                a = np.ascontiguousarray(a)
                img = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(img.copy()).scaled(
                lbl.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(pix)


class UndistortWidget(QWidget):
    """Lens calibration + video undistort/redistort tab."""

    def __init__(self, ffmpeg_path_getter, parent=None):
        super().__init__(parent)
        self._ffmpeg_path_getter = ffmpeg_path_getter  # callable -> str
        self._profile: LensProfile | None = None
        self._worker: QThread | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ---- Calibration ----
        cal_group = QGroupBox("Lens Calibration")
        cal = QVBoxLayout(cal_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Calibration image:"))
        self._cal_img = QLineEdit()
        self._cal_img.setPlaceholderText(
            "Checkerboard or dot-pattern photo taken with the same lens")
        row.addWidget(self._cal_img, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._pick_cal_image)
        row.addWidget(btn)
        cal.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Pattern:"))
        self._pattern_combo = QComboBox()
        self._pattern_combo.addItems(["checkerboard", "dots"])
        row2.addWidget(self._pattern_combo)
        btn = QPushButton("Show Pattern Fullscreen")
        btn.setToolTip(
            "Display a checkerboard fullscreen — photograph it with the "
            "target lens, then load that photo as the calibration image.")
        btn.clicked.connect(self._show_pattern_fullscreen)
        row2.addWidget(btn)
        self._cal_btn = QPushButton("Calibrate")
        self._cal_btn.clicked.connect(self._run_calibration)
        row2.addWidget(self._cal_btn)
        row2.addStretch()
        cal.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Lens profile:"))
        self._profile_path = QLineEdit()
        self._profile_path.setPlaceholderText(
            "Saved after calibration — or load an existing .json profile")
        row3.addWidget(self._profile_path, 1)
        btn = QPushButton("Load")
        btn.clicked.connect(self._load_profile)
        row3.addWidget(btn)
        btn = QPushButton("Save As...")
        btn.clicked.connect(self._save_profile)
        row3.addWidget(btn)
        cal.addLayout(row3)

        self._profile_info = QLabel("No profile loaded")
        cal.addWidget(self._profile_info)
        layout.addWidget(cal_group)

        # ---- Video conversion ----
        conv_group = QGroupBox("Convert Video")
        conv = QVBoxLayout(conv_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Input video:"))
        self._in_video = QLineEdit()
        row.addWidget(self._in_video, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._pick_input_video)
        row.addWidget(btn)
        conv.addLayout(row)

        row = QHBoxLayout()
        self._mode_undistort = QPushButton("Undistort")
        self._mode_redistort = QPushButton("Redistort")
        for b, tip in ((self._mode_undistort,
                        "Fix lens distortion — run before tracking"),
                       (self._mode_redistort,
                        "Re-apply lens distortion to rendered/comped footage")):
            b.setCheckable(True)
            b.setMinimumHeight(44)
            b.setMinimumWidth(130)
            b.setToolTip(tip)
            row.addWidget(b)
        self._mode_undistort.setChecked(True)
        self._mode_undistort.clicked.connect(
            lambda: self._mode_redistort.setChecked(False))
        self._mode_redistort.clicked.connect(
            lambda: self._mode_undistort.setChecked(False))
        mode_style = (
            "QPushButton {"
            "  background-color: #555555;"
            "  color: #cccccc;"
            "  border: 2px solid #444444;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:hover { background-color: #666666; }"
            "QPushButton:checked {"
            "  background-color: #2e6cf6;"
            "  color: white;"
            "  font-weight: bold;"
            "  border: 2px solid #1a3fa0;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:checked:hover { background-color: #3d7bff; }")
        self._mode_undistort.setStyleSheet(mode_style)
        self._mode_redistort.setStyleSheet(mode_style)
        row.addWidget(QLabel("Output:"))
        self._out_video = QLineEdit()
        row.addWidget(self._out_video, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._pick_output_video)
        row.addWidget(btn)
        conv.addLayout(row)

        row = QHBoxLayout()
        btn = QPushButton("Export ST Map (EXR)")
        btn.setToolTip(
            "Save a 32-bit float ST map of the selected mode — "
            "apply with an STMap node in Nuke or RE:Map in AE")
        btn.clicked.connect(self._export_stmap)
        row.addWidget(btn)
        btn = QPushButton("Preview Frame")
        btn.setToolTip("Warp the first frame and show original vs result")
        btn.clicked.connect(self._preview_frame)
        row.addWidget(btn)
        self._run_btn = QPushButton("Process Video")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._run_video)
        row.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self._cancel_btn)
        row.addStretch()
        conv.addLayout(row)

        layout.addWidget(conv_group)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Log output will appear here...")
        layout.addWidget(self._log)

    def _mode(self) -> str:
        return ("redistort" if self._mode_redistort.isChecked()
                else "undistort")

    def _mode(self) -> str:
        return ("redistort" if self._mode_redistort.isChecked()
                else "undistort")

    # ---- pickers ----

    def _pick_cal_image(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration Image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if p:
            self._cal_img.setText(p)

    def _pick_input_video(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*.*)")
        if p:
            self._in_video.setText(p)
            if not self._out_video.text():
                stem = Path(p).with_suffix("")
                self._out_video.setText(
                    f"{stem}_{self._mode()}.mp4")

    def _pick_output_video(self) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "", "MP4 (*.mp4)")
        if p:
            self._out_video.setText(p)

    def _show_pattern_fullscreen(self) -> None:
        dlg = _PatternDialog(self)
        dlg.showFullScreen()
        dlg.exec()

    # ---- calibration ----

    def _run_calibration(self) -> None:
        img = self._cal_img.text().strip()
        if not img or not Path(img).exists():
            QMessageBox.warning(self, "Missing image",
                                "Select a valid calibration image.")
            return
        self._cal_btn.setEnabled(False)
        self._log.append(f"Calibrating from {img}...")
        self._worker = CalibrateWorker(Path(img),
                                       self._pattern_combo.currentText())
        self._worker.log.connect(self._log.append)
        self._worker.finished.connect(self._on_calibrated)
        self._worker.start()

    def _on_calibrated(self, profile: LensProfile | None, error: str) -> None:
        self._cal_btn.setEnabled(True)
        if profile is None:
            self._log.append(f"Calibration failed: {error}")
            QMessageBox.warning(self, "Calibration failed", error)
            return
        self._profile = profile
        self._profile_info.setText(
            f"Center: ({profile.xcenter:.1f}, {profile.ycenter:.1f})  "
            f"Image: {profile.width}x{profile.height}  "
            f"Coefs: {len(profile.undistort_coefs)}")
        self._log.append("Calibration complete. Saving profile...")
        self._save_profile()

    def _load_profile(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Load Lens Profile", "", "JSON (*.json)")
        if not p:
            return
        try:
            self._profile = LensProfile.load(Path(p))
            self._profile_path.setText(p)
            self._profile_info.setText(
                f"Center: ({self._profile.xcenter:.1f}, "
                f"{self._profile.ycenter:.1f})  "
                f"Image: {self._profile.width}x{self._profile.height}  "
                f"Coefs: {len(self._profile.undistort_coefs)}")
            self._log.append(f"Loaded profile: {p}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Load failed", str(e))

    def _save_profile(self) -> None:
        if self._profile is None:
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Lens Profile", "lens_profile.json", "JSON (*.json)")
        if not p:
            return
        self._profile.save(Path(p))
        self._profile_path.setText(p)
        self._log.append(f"Profile saved: {p}")

    # ---- video ----

    def _export_stmap(self) -> None:
        if self._profile is None:
            QMessageBox.warning(self, "No profile",
                                "Calibrate or load a lens profile first.")
            return
        mode = self._mode()
        p, _ = QFileDialog.getSaveFileName(
            self, f"Save {mode} ST Map", f"stmap_{mode}.exr",
            "EXR (*.exr)")
        if not p:
            return
        try:
            from core.undistort import stmap, save_stmap_exr
            # ST map is resolution-independent — export at the calibration
            # resolution; the consuming package resamples to the plate size.
            st = stmap(self._profile.width, self._profile.height,
                       self._profile, mode)
            save_stmap_exr(Path(p), st)
            self._log.append(
                f"ST map ({mode}) saved: {p} "
                f"[{self._profile.width}x{self._profile.height}]")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "ST map export failed", str(e))

    def _preview_frame(self) -> None:
        if self._profile is None:
            QMessageBox.warning(self, "No profile",
                                "Calibrate or load a lens profile first.")
            return
        in_path = Path(self._in_video.text().strip())
        if not in_path.exists():
            QMessageBox.warning(self, "Missing input", "Select a valid video.")
            return
        ffmpeg = Path(self._ffmpeg_path_getter())
        if not ffmpeg.exists():
            QMessageBox.warning(self, "Missing FFmpeg", str(ffmpeg))
            return
        try:
            import tempfile
            import subprocess
            import imageio.v3 as iio
            from core.undistort import undistort_frame, redistort_frame
            with tempfile.TemporaryDirectory() as td:
                frame_png = Path(td) / "f.png"
                subprocess.run(
                    [str(ffmpeg), "-loglevel", "error", "-y",
                     "-i", str(in_path), "-frames:v", "1", str(frame_png)],
                    check=True)
                img = np.asarray(iio.imread(frame_png))
            mode = self._mode()
            warped = (undistort_frame(img, self._profile)
                      if mode == "undistort"
                      else redistort_frame(img, self._profile))
            _CompareDialog(img, warped, f"Preview: {mode}", self).exec()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Preview failed", str(e))

    def _run_video(self) -> None:
        if self._profile is None:
            QMessageBox.warning(self, "No profile",
                                "Calibrate or load a lens profile first.")
            return
        in_path = Path(self._in_video.text().strip())
        out_path = Path(self._out_video.text().strip())
        if not in_path.exists():
            QMessageBox.warning(self, "Missing input", "Select a valid video.")
            return
        if not out_path.name:
            QMessageBox.warning(self, "Missing output", "Choose an output path.")
            return
        ffmpeg = Path(self._ffmpeg_path_getter())
        if not ffmpeg.exists():
            QMessageBox.warning(self, "Missing FFmpeg", str(ffmpeg))
            return

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progres

        self._worker = VideoWarpWorker(
            in_path, out_path, self._profile,
            self._mode_combo.currentText(), ffmpeg)
        self._worker.log.connect(self._log.append)
        self._worker.progress.connect(
            lambda pct, msg: (self._progress.setValue(pct), None))
        self._worker.finished.connect(self._on_video_done)
        self._worker.start()

    def _cancel(self) -> None:
        if isinstance(self._worker, VideoWarpWorker):
            self._worker.cancel()
            self._log.append("Cancelling...")

    def _on_video_done(self, ok: bool, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(100 if ok else 0)
        self._log.append(msg if ok else f"Failed: {msg}")

    def session_state(self) -> dict:
        return {
            "calibration_image": self._cal_img.text(),
            "pattern": self._pattern_combo.currentText(),
            "profile_path": self._profile_path.text(),
            "profile": asdict(self._profile) if self._profile is not None else None,
            "input_video": self._in_video.text(),
            "output_video": self._out_video.text(),
            "mode": self._mode(),
        }

    def restore_session(self, state: dict) -> None:
        self._cal_img.setText(state.get("calibration_image", ""))
        idx = self._pattern_combo.findText(state.get("pattern", ""))
        if idx >= 0:
            self._pattern_combo.setCurrentIndex(idx)
        self._profile_path.setText(state.get("profile_path", ""))
        self._in_video.setText(state.get("input_video", ""))
        self._out_video.setText(state.get("output_video", ""))
        redistort = state.get("mode") == "redistort"
        self._mode_redistort.setChecked(redistort)
        self._mode_undistort.setChecked(not redistort)
        profile = None
        data = state.get("profile")
        if isinstance(data, dict):
            try:
                profile = LensProfile(**data)
            except (TypeError, ValueError):
                profile = None
        if profile is None:
            path = self._profile_path.text().strip()
            if path and Path(path).exists():
                try:
                    profile = LensProfile.load(Path(path))
                except (OSError, TypeError, ValueError):
                    profile = None
        self._profile = profile
        if profile is not None:
            self._profile_info.setText(
                f"Center: ({profile.xcenter:.1f}, {profile.ycenter:.1f})  "
                f"Image: {profile.width}x{profile.height}  "
                f"Coefs: {len(profile.undistort_coefs)}")
        else:
            self._profile_info.setText("No profile loaded")
