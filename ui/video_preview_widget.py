from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.video_player import VideoSource


class VideoPreviewWidget(QWidget):
    """Simple frame-accurate video preview with play/pause and timeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source: VideoSource | None = None
        self._current_frame = 0
        self._is_playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._frame_label = QLabel("No video loaded")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(480, 270)
        self._frame_label.setStyleSheet("background-color: #0d0d0d; color: #888888;")
        layout.addWidget(self._frame_label, 1)

        controls = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self._play_btn)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setEnabled(False)
        self._frame_slider.valueChanged.connect(self._slider_changed)
        controls.addWidget(self._frame_slider, 1)

        self._frame_info = QLabel("0 / 0")
        controls.addWidget(self._frame_info)

        layout.addLayout(controls)

    def load_video(self, path: Path) -> None:
        try:
            if self._source is not None:
                self._source.close()
            self._source = VideoSource(path)
        except Exception as exc:
            self._frame_label.setText(f"Error loading video:\n{exc}")
            self._source = None
            return

        self._current_frame = 0
        self._frame_slider.setEnabled(True)
        self._frame_slider.setRange(0, max(0, self._source.frame_count - 1))
        self._frame_slider.setValue(0)
        self._play_btn.setEnabled(True)
        self._update_frame_info()
        self._show_frame(0)

    def _slider_changed(self, value: int) -> None:
        self._current_frame = value
        self._show_frame(value)
        self._update_frame_info()

    def _toggle_play(self) -> None:
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self._source is None or self._source.frame_count <= 1:
            return
        self._is_playing = True
        self._play_btn.setText("Pause")
        interval = max(1, int(1000.0 / self._source.fps))
        self._timer.start(interval)

    def _pause(self) -> None:
        self._is_playing = False
        self._play_btn.setText("Play")
        self._timer.stop()

    def _next_frame(self) -> None:
        if self._source is None:
            return
        self._current_frame += 1
        if self._current_frame >= self._source.frame_count:
            self._current_frame = 0
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(self._current_frame)
        self._frame_slider.blockSignals(False)
        self._show_frame(self._current_frame)
        self._update_frame_info()

    def _show_frame(self, index: int) -> None:
        if self._source is None:
            return
        rgb = self._source.read_rgb(index)
        if rgb is None:
            return
        pixmap = self._numpy_to_pixmap(rgb)
        scaled = pixmap.scaled(
            self._frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)

    def _update_frame_info(self) -> None:
        if self._source is None:
            self._frame_info.setText("0 / 0")
            return
        total = self._source.frame_count
        self._frame_info.setText(f"{self._current_frame + 1} / {total}")

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        if self._source is not None:
            self._show_frame(self._current_frame)

    @staticmethod
    def _numpy_to_pixmap(array: np.ndarray) -> QPixmap:
        h, w, ch = array.shape
        bytes_per_line = ch * w
        image = QImage(array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap
