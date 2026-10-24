from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.ar_preview import ARPreviewRenderer
from core.orientation import compute_scale


class ARPreviewWidget(QWidget):
    """Play the tracked video with a 3D object overlaid to test tracking quality."""

    orientation_applied = Signal(object, object, object)
    origin_set = Signal(object)
    scale_applied = Signal(float)
    z_rotation_changed = Signal(float)
    resolve_requested = Signal(list)  # list of bad points3D ids to prune + re-solve
    smoothing_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._renderer: ARPreviewRenderer | None = None
        self._current_index = 0
        self._is_playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        self._selected_origin: dict[str, Any] | None = None
        self._selected_floor: list[dict[str, Any]] = []
        self._selected_wall: list[dict[str, Any]] = []
        self._selected_scale: list[dict[str, Any]] = []
        self._last_selected: dict[str, Any] | None = None
        self._pending_selection: list[dict[str, Any]] = []

        self._drag_start: tuple[float, float] | None = None
        self._drag_current: tuple[float, float] | None = None
        self._is_dragging = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left: video overlay.
        left = QVBoxLayout()
        self._frame_label = QLabel("No tracked video loaded")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(640, 360)
        self._frame_label.setStyleSheet("background-color: #0d0d0d; color: #888888;")
        self._frame_label.setMouseTracking(True)
        self._frame_label.mousePressEvent = self._on_frame_press
        self._frame_label.mouseMoveEvent = self._on_frame_move
        self._frame_label.mouseReleaseEvent = self._on_frame_release
        left.addWidget(self._frame_label, 1)

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

        left.addLayout(controls)
        layout.addLayout(left, 1)

        # Right: object and orientation controls.
        right = QVBoxLayout()
        right.setSpacing(12)

        self._focal_info = QLabel("Focal length: -")
        self._focal_info.setStyleSheet("color: #a0a0a0;")
        right.addWidget(self._focal_info)

        help_label = QLabel(
            "Click a point-cloud dot, or click-and-drag a box to select multiple points. "
            "Then use Set Origin / Add to Floor / Add to Wall / Add to Scale to assign them. "
            "After picking the origin and 3+ floor points, click Apply Orientation."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #a0a0a0;")
        right.addWidget(help_label)

        right.addWidget(QLabel("Test object type:"))
        self._object_combo = QComboBox()
        self._object_combo.addItems(["cube", "axes", "grid"])
        self._object_combo.currentTextChanged.connect(self._update_object)
        right.addWidget(self._object_combo)

        right.addWidget(QLabel("Object size:"))
        self._size_spin = QDoubleSpinBox()
        self._size_spin.setRange(0.01, 1000.0)
        self._size_spin.setValue(1.0)
        self._size_spin.valueChanged.connect(self._update_object)
        right.addWidget(self._size_spin)

        right.addWidget(QLabel("Object position (X, Y, Z):"))
        pos_layout = QHBoxLayout()
        self._pos_x = QDoubleSpinBox()
        self._pos_y = QDoubleSpinBox()
        self._pos_z = QDoubleSpinBox()
        for spin in (self._pos_x, self._pos_y, self._pos_z):
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(3)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._update_object)
            pos_layout.addWidget(spin)
        self._pos_z.setValue(2.0)
        right.addLayout(pos_layout)

        right.addWidget(QLabel("Pending selection:"))
        self._pending_selection_info = QLabel("None")
        self._pending_selection_info.setStyleSheet("color: #a0a0a0;")
        right.addWidget(self._pending_selection_info)

        right.addWidget(QLabel("Assigned points:"))
        self._selection_info = QLabel("Origin: none | Floor: 0 | Wall: 0 | Scale: 0")
        self._selection_info.setStyleSheet("color: #a0a0a0;")
        right.addWidget(self._selection_info)

        sel_layout = QHBoxLayout()
        self._origin_btn = QPushButton("Set origin")
        self._origin_btn.setEnabled(False)
        self._origin_btn.clicked.connect(self._set_origin)
        sel_layout.addWidget(self._origin_btn)

        self._floor_btn = QPushButton("Add to floor")
        self._floor_btn.setEnabled(False)
        self._floor_btn.clicked.connect(self._add_floor)
        sel_layout.addWidget(self._floor_btn)

        self._wall_btn = QPushButton("Add to wall")
        self._wall_btn.setEnabled(False)
        self._wall_btn.clicked.connect(self._add_wall)
        sel_layout.addWidget(self._wall_btn)

        self._scale_btn = QPushButton("Add to scale")
        self._scale_btn.setEnabled(False)
        self._scale_btn.clicked.connect(self._add_scale)
        sel_layout.addWidget(self._scale_btn)
        right.addLayout(sel_layout)

        clear_layout = QHBoxLayout()
        self._clear_btn = QPushButton("Clear selections")
        self._clear_btn.setEnabled(True)
        self._clear_btn.clicked.connect(self._clear_selections)
        clear_layout.addWidget(self._clear_btn)
        right.addLayout(clear_layout)

        orient_layout = QHBoxLayout()
        self._apply_btn = QPushButton("Apply orientation")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply_orientation)
        orient_layout.addWidget(self._apply_btn)

        self._reset_btn = QPushButton("Reset orientation")
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._reset_orientation)
        orient_layout.addWidget(self._reset_btn)
        right.addLayout(orient_layout)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Known distance:"))
        self._scale_distance = QDoubleSpinBox()
        self._scale_distance.setRange(0.001, 10000.0)
        self._scale_distance.setValue(1.0)
        self._scale_distance.setDecimals(4)
        scale_layout.addWidget(self._scale_distance)
        self._apply_scale_btn = QPushButton("Apply scale")
        self._apply_scale_btn.setEnabled(False)
        self._apply_scale_btn.clicked.connect(self._apply_scale)
        scale_layout.addWidget(self._apply_scale_btn)
        right.addLayout(scale_layout)

        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z rotation (°):"))
        self._z_rotation = QDoubleSpinBox()
        self._z_rotation.setRange(-360.0, 360.0)
        self._z_rotation.setValue(0.0)
        self._z_rotation.setDecimals(2)
        self._z_rotation.setWrapping(True)
        self._z_rotation.valueChanged.connect(self._update_z_rotation)
        z_layout.addWidget(self._z_rotation)
        right.addLayout(z_layout)

        # --- Track health ---
        health_group = QGroupBox("Track health")
        health_layout = QVBoxLayout(health_group)
        self._error_stats = QLabel("Reprojection error: -")
        self._error_stats.setWordWrap(True)
        self._error_stats.setStyleSheet("color: #a0a0a0;")
        health_layout.addWidget(self._error_stats)

        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Error threshold (px):"))
        self._error_threshold = QDoubleSpinBox()
        self._error_threshold.setRange(0.0, 100.0)
        self._error_threshold.setValue(2.0)
        self._error_threshold.setDecimals(2)
        self._error_threshold.setToolTip(
            "Points whose mean reprojection error is ABOVE this value are shown red "
            "and can be pruned. Higher threshold = fewer bad points."
        )
        self._error_threshold.valueChanged.connect(self._on_error_threshold_changed)
        thresh_layout.addWidget(self._error_threshold)
        health_layout.addLayout(thresh_layout)

        tlen_layout = QHBoxLayout()
        tlen_layout.addWidget(QLabel("Min track length:"))
        self._min_track_len = QSpinBox()
        self._min_track_len.setRange(0, 100)
        self._min_track_len.setValue(0)
        self._min_track_len.setToolTip(
            "Points seen in fewer frames than this are treated as bad (shown red)."
            " Short 2-3 frame tracks are usually junk. 0 = disabled."
        )
        self._min_track_len.valueChanged.connect(self._on_error_threshold_changed)
        tlen_layout.addWidget(self._min_track_len)
        health_layout.addLayout(tlen_layout)

        prune_layout = QHBoxLayout()
        self._prune_threshold_btn = QPushButton("Prune + Re-solve")
        self._prune_threshold_btn.setEnabled(False)
        self._prune_threshold_btn.setToolTip(
            "Delete every point above the threshold and re-run GLOMAP."
        )
        prune_layout.addWidget(self._prune_threshold_btn)

        self._delete_selected_btn = QPushButton("Delete selected + Re-solve")
        self._delete_selected_btn.setEnabled(False)
        self._delete_selected_btn.setToolTip(
            "Delete the currently selected point-cloud points and re-run GLOMAP."
        )
        self._delete_selected_btn.clicked.connect(self._delete_selected_and_resolve)
        prune_layout.addWidget(self._delete_selected_btn)
        health_layout.addLayout(prune_layout)

        self._prune_threshold_btn.clicked.connect(self._prune_threshold_and_resolve)

        # --- Smoothing ---
        smooth_group = QGroupBox("Track smoothing")
        smooth_layout = QHBoxLayout(smooth_group)
        smooth_layout.addWidget(QLabel("Window (frames):"))
        self._smooth_window = QSpinBox()
        self._smooth_window.setRange(0, 61)
        self._smooth_window.setValue(0)
        self._smooth_window.setToolTip(
            "Savitzky-Golay temporal smoothing window for camera + object poses.\n"
            "0 = off. 7-15 frames removes micro-jitter; higher values smooth more\n"
            "but can lag fast deliberate motion."
        )
        self._smooth_window.valueChanged.connect(self.smoothing_changed)
        smooth_layout.addWidget(self._smooth_window)

        right.addWidget(smooth_group)
        right.addWidget(health_group)

        right.addStretch()
        layout.addLayout(right)

    def load_track(
        self,
        frames_dir: Path,
        track: list[dict[str, Any]],
        point_cloud: list[dict[str, Any]] | None = None,
        orientation: tuple[np.ndarray, np.ndarray] | None = None,
        z_rotation: float = 0.0,
        object_track: list[dict[str, Any]] | None = None,
    ) -> None:
        if not track:
            self._frame_label.setText("No track data")
            return
        self._renderer = ARPreviewRenderer(frames_dir, track, point_cloud, object_track=object_track)
        self._renderer.error_threshold = self._error_threshold.value()
        self._renderer.min_track_len = self._min_track_len.value()
        self._update_error_stats()
        if orientation is not None:
            self._renderer.set_orientation(*orientation)
            self._reset_btn.setEnabled(True)
        self._z_rotation.setValue(float(z_rotation))
        self._renderer.set_z_rotation(float(z_rotation))
        self._current_index = 0
        self._frame_slider.setEnabled(True)
        self._frame_slider.setRange(0, len(track) - 1)
        self._frame_slider.setValue(0)
        self._play_btn.setEnabled(True)
        self._clear_selections()
        self._update_object()
        self._update_error_stats()
        self._update_info()

    def _on_frame_press(self, event: QMouseEvent) -> None:
        if self._renderer is None:
            return
        self._drag_start = (event.position().x(), event.position().y())
        self._drag_current = self._drag_start
        self._is_dragging = False

    def _on_frame_move(self, event: QMouseEvent) -> None:
        if self._renderer is None or self._drag_start is None:
            return
        self._drag_current = (event.position().x(), event.position().y())
        dx = self._drag_current[0] - self._drag_start[0]
        dy = self._drag_current[1] - self._drag_start[1]
        if not self._is_dragging and (abs(dx) > 5 or abs(dy) > 5):
            self._is_dragging = True
        if self._is_dragging:
            self._render_current()

    def _on_frame_release(self, event: QMouseEvent) -> None:
        if self._renderer is None or self._drag_start is None:
            return
        end_pos = (event.position().x(), event.position().y())
        start_pos = self._drag_start
        self._drag_start = None
        self._drag_current = None

        if self._is_dragging:
            self._is_dragging = False
            self._box_select(start_pos, end_pos, event.modifiers())
        else:
            # Treat as a single click.
            img_coords = self._label_to_image_coords(end_pos[0], end_pos[1])
            if img_coords is None:
                return
            point = self._renderer.find_nearest_point(img_coords[0], img_coords[1], self._current_index)
            if point is None:
                return
            self._add_to_pending_selection([point], event.modifiers())
        self._render_current()

    def _label_to_image_coords(self, label_x: float, label_y: float) -> tuple[float, float] | None:
        if self._renderer is None or not self._renderer.track:
            return None
        entry = self._renderer.track[self._current_index]
        img_w = entry.get("width", 0)
        img_h = entry.get("height", 0)
        if img_w <= 0 or img_h <= 0:
            return None
        label_w = self._frame_label.width()
        label_h = self._frame_label.height()
        scale = min(label_w / img_w, label_h / img_h)
        scaled_w = img_w * scale
        scaled_h = img_h * scale
        offset_x = (label_w - scaled_w) / 2.0
        offset_y = (label_h - scaled_h) / 2.0
        x = (label_x - offset_x) / scale
        y = (label_y - offset_y) / scale
        if not (0 <= x < img_w and 0 <= y < img_h):
            return None
        return x, y

    def _add_to_pending_selection(
        self,
        points: list[dict[str, Any]],
        modifiers: Any,
    ) -> None:
        if not points:
            return
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._pending_selection.extend(points)
        else:
            self._pending_selection = list(points)
        self._last_selected = points[-1]
        self._update_pending_selection_info()
        self._update_button_states()

    def _box_select(
        self,
        label_start: tuple[float, float],
        label_end: tuple[float, float],
        modifiers: Any,
    ) -> None:
        if self._renderer is None:
            return
        start_img = self._label_to_image_coords(label_start[0], label_start[1])
        end_img = self._label_to_image_coords(label_end[0], label_end[1])
        if start_img is None or end_img is None:
            return
        x1, y1 = start_img
        x2, y2 = end_img
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        if right - left < 3 or bottom - top < 3:
            return

        entry = self._renderer.track[self._current_index]
        focal = entry.get("focal_length", 0.0)
        width = entry.get("width", 0)
        height = entry.get("height", 0)
        if focal <= 0:
            return
        from core.ar_preview import _quaternion_to_matrix, build_camera_matrix
        camera_matrix = build_camera_matrix(focal, width, height)
        R = _quaternion_to_matrix(entry["qw"], entry["qx"], entry["qy"], entry["qz"])
        t = np.array([entry["tx"], entry["ty"], entry["tz"]], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)

        selected: list[dict[str, Any]] = []
        for point in self._renderer.point_cloud:
            pos = np.asarray(point["position"], dtype=np.float64).reshape(1, 3)
            projected, _ = cv2.projectPoints(pos, rvec, t, camera_matrix, None)
            px, py = projected[0][0]
            if left <= px <= right and top <= py <= bottom:
                selected.append(point)

        self._add_to_pending_selection(selected, modifiers)

    def _set_origin(self) -> None:
        if len(self._pending_selection) == 1:
            picked = self._pending_selection[0]
            if self._renderer is not None:
                r = self._renderer
                origin_raw = r.to_original_frame(picked["position"])
                lists = (self._selected_floor, self._selected_wall, self._selected_scale)
                raws = [[r.to_original_frame(d["position"]) for d in lst] for lst in lists]
                r.set_origin(origin_raw)
                for lst, lst_raws in zip(lists, raws):
                    for i, raw in enumerate(lst_raws):
                        lst[i] = {**lst[i], "position": r.from_original_frame(raw)}
                self._selected_origin = {**picked, "position": np.zeros(3)}
                self._sync_object_position_from_renderer()
                self.origin_set.emit(origin_raw)
                self._render_current()
            else:
                self._selected_origin = picked
            self._clear_pending_selection()
            self._update_selection_info()
            self._update_button_states()

    def _add_floor(self) -> None:
        if not self._pending_selection:
            return
        self._selected_floor.extend(self._pending_selection)
        self._clear_pending_selection()
        self._update_selection_info()
        self._update_button_states()
        self._update_apply_state()

    def _update_apply_state(self) -> None:
        self._apply_btn.setEnabled(
            len(self._selected_floor) >= 3 or len(self._selected_wall) >= 3
        )

    def _add_wall(self) -> None:
        if not self._pending_selection:
            return
        self._selected_wall.extend(self._pending_selection)
        self._clear_pending_selection()
        self._update_selection_info()
        self._update_button_states()
        self._update_apply_state()

    def _add_scale(self) -> None:
        if not self._pending_selection:
            return
        room = 2 - len(self._selected_scale)
        if room <= 0:
            return
        to_add = self._pending_selection[:room]
        self._selected_scale.extend(to_add)
        self._clear_pending_selection()
        self._update_selection_info()
        self._update_button_states()
        self._apply_scale_btn.setEnabled(len(self._selected_scale) == 2)

    def _apply_scale(self) -> None:
        if len(self._selected_scale) < 2:
            return
        p1 = np.asarray(self._selected_scale[0]["position"])
        p2 = np.asarray(self._selected_scale[1]["position"])
        scale = compute_scale(p1, p2, self._scale_distance.value())
        self.scale_applied.emit(scale)
        self._selected_scale = []
        self._sync_object_position_from_renderer()

    def _update_z_rotation(self) -> None:
        if self._renderer is None:
            return
        self._renderer.set_z_rotation(self._z_rotation.value())
        self.z_rotation_changed.emit(self._z_rotation.value())
        self._render_current()

    def _clear_selections(self) -> None:
        self._selected_origin = None
        self._selected_floor = []
        self._selected_wall = []
        self._selected_scale = []
        self._last_selected = None
        self._clear_pending_selection()
        self._apply_btn.setEnabled(False)
        self._apply_scale_btn.setEnabled(False)
        self._update_selection_info()
        self._update_button_states()

    def _clear_pending_selection(self) -> None:
        self._pending_selection = []
        self._update_pending_selection_info()

    def _update_pending_selection_info(self) -> None:
        count = len(self._pending_selection)
        if count == 0:
            text = "None"
        elif count == 1:
            text = "1 point selected"
        else:
            text = f"{count} points selected"
        self._pending_selection_info.setText(text)

    def _update_button_states(self) -> None:
        has_pending = len(self._pending_selection) > 0
        self._origin_btn.setEnabled(has_pending and len(self._pending_selection) == 1)
        self._floor_btn.setEnabled(has_pending)
        self._wall_btn.setEnabled(has_pending)
        self._scale_btn.setEnabled(has_pending and len(self._selected_scale) < 2)
        self._delete_selected_btn.setEnabled(has_pending and any("id" in p for p in self._pending_selection))

    def _update_selection_info(self) -> None:
        origin = "yes" if self._selected_origin is not None else "none"
        self._selection_info.setText(
            f"Origin: {origin} | Floor: {len(self._selected_floor)} | Wall: {len(self._selected_wall)} | Scale: {len(self._selected_scale)}"
        )

    def _apply_orientation(self) -> None:
        if self._renderer is None:
            return
        if len(self._selected_floor) < 3 and len(self._selected_wall) < 3 and self._selected_origin is None:
            return
        to_orig = self._renderer.to_original_frame
        floor_positions = [to_orig(p["position"]) for p in self._selected_floor]
        wall_positions = [to_orig(p["position"]) for p in self._selected_wall]
        if self._selected_origin is not None:
            origin_position = to_orig(self._selected_origin["position"])
        elif self._renderer._orientation_origin is not None:
            origin_position = np.asarray(self._renderer._orientation_origin, dtype=np.float64)
        else:
            origin_position = None
        try:
            self._renderer.apply_orientation(floor_positions, wall_positions, origin_position)
            self._sync_object_position_from_renderer()
            self.orientation_applied.emit(floor_positions, wall_positions, origin_position)
            self._clear_selections()
            self._reset_btn.setEnabled(True)
            self._render_current()
        except Exception as exc:
            QMessageBox.warning(self, "Orientation failed", str(exc))

    def _reset_orientation(self) -> None:
        if self._renderer is None:
            return
        self._renderer.reset_orientation()
        self._sync_object_position_from_renderer()
        self._clear_selections()
        self._render_current()

    def _sync_object_position_from_renderer(self) -> None:
        if self._renderer is None:
            return
        x, y, z = self._renderer.object_position
        self._pos_x.blockSignals(True)
        self._pos_y.blockSignals(True)
        self._pos_z.blockSignals(True)
        self._pos_x.setValue(float(x))
        self._pos_y.setValue(float(y))
        self._pos_z.setValue(float(z))
        self._pos_x.blockSignals(False)
        self._pos_y.blockSignals(False)
        self._pos_z.blockSignals(False)

    def _update_object(self) -> None:
        if self._renderer is None:
            return
        pos = (self._pos_x.value(), self._pos_y.value(), self._pos_z.value())
        self._renderer.set_object(
            self._object_combo.currentText(),
            self._size_spin.value(),
            pos,
        )
        self._render_current()

    def _slider_changed(self, value: int) -> None:
        self._current_index = value
        self._render_current()
        self._update_info()

    def _toggle_play(self) -> None:
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self._renderer is None or len(self._renderer.track) <= 1:
            return
        self._is_playing = True
        self._play_btn.setText("Pause")
        self._timer.start(50)

    def _pause(self) -> None:
        self._is_playing = False
        self._play_btn.setText("Play")
        self._timer.stop()

    def _next_frame(self) -> None:
        if self._renderer is None:
            return
        self._current_index += 1
        if self._current_index >= len(self._renderer.track):
            self._current_index = 0
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(self._current_index)
        self._frame_slider.blockSignals(False)
        self._render_current()
        self._update_info()

    def _render_current(self) -> None:
        if self._renderer is None:
            return
        bgr = self._renderer.render(self._current_index)
        if bgr is None:
            return

        # Highlight selected points on this frame.
        self._draw_selections(bgr)
        if self._is_dragging and self._drag_start is not None and self._drag_current is not None:
            self._draw_selection_box(bgr)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pixmap = self._numpy_to_pixmap(rgb)
        scaled = pixmap.scaled(
            self._frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)

    def _draw_selection_box(self, frame: np.ndarray) -> None:
        if self._drag_start is None or self._drag_current is None:
            return
        start_img = self._label_to_image_coords(self._drag_start[0], self._drag_start[1])
        end_img = self._label_to_image_coords(self._drag_current[0], self._drag_current[1])
        if start_img is None or end_img is None:
            return
        x1, y1 = int(start_img[0]), int(start_img[1])
        x2, y2 = int(end_img[0]), int(end_img[1])
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_selections(self, frame: np.ndarray) -> None:
        if self._renderer is None:
            return
        entry = self._renderer.track[self._current_index]
        focal = entry.get("focal_length", 0.0)
        width = entry.get("width", frame.shape[1])
        height = entry.get("height", frame.shape[0])
        if focal <= 0:
            return
        from core.ar_preview import _quaternion_to_matrix, build_camera_matrix
        camera_matrix = build_camera_matrix(focal, width, height)
        R = _quaternion_to_matrix(entry["qw"], entry["qx"], entry["qy"], entry["qz"])
        t = np.array([entry["tx"], entry["ty"], entry["tz"]], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)

        def draw_point(point: dict[str, Any], color: tuple[int, int, int], radius: int) -> None:
            pos = np.asarray(point["position"], dtype=np.float64).reshape(1, 3)
            projected, _ = cv2.projectPoints(pos, rvec, t, camera_matrix, None)
            x, y = projected[0][0]
            x, y = int(x), int(y)
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), radius, color, 3, cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius - 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x - radius - 4, y), (x + radius + 4, y), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x, y - radius - 4), (x, y + radius + 4), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)

        if self._selected_origin is not None:
            draw_point(self._selected_origin, (0, 165, 255), 18)
        for p in self._selected_floor:
            draw_point(p, (0, 255, 0), 18)
        for p in self._selected_wall:
            draw_point(p, (0, 0, 255), 18)
        for p in self._selected_scale:
            draw_point(p, (255, 0, 255), 18)
        for p in self._pending_selection:
            draw_point(p, (255, 255, 0), 18)

    def _on_error_threshold_changed(self) -> None:
        if self._renderer is not None:
            self._renderer.error_threshold = self._error_threshold.value()
            self._renderer.min_track_len = self._min_track_len.value()
        self._update_error_stats()
        self._render_current()

    def _is_bad_point(self, p: dict[str, Any]) -> bool:
        threshold = self._error_threshold.value()
        min_len = self._min_track_len.value()
        if threshold > 0 and p.get("error", 0.0) > threshold:
            return True
        if min_len > 0 and len(p.get("track", [])) < min_len:
            return True
        return False

    def _update_error_stats(self) -> None:
        if self._renderer is None or not self._renderer.point_cloud:
            self._error_stats.setText("Reprojection error: -")
            return
        cloud = self._renderer.point_cloud
        errors = np.array([p.get("error", 0.0) for p in cloud], dtype=np.float64)
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            self._error_stats.setText("Reprojection error: -")
            return
        mean = float(np.mean(errors))
        median = float(np.median(errors))
        p90 = float(np.percentile(errors, 90))
        max_err = float(np.max(errors))
        bad = sum(1 for p in cloud if self._is_bad_point(p))
        self._error_stats.setText(
            f"Reprojection error — mean {mean:.3f}px, median {median:.3f}px, "
            f"p90 {p90:.3f}px, max {max_err:.3f}px.\n"
            f"{bad} / {len(cloud)} points flagged bad (shown red)."
        )
        self._prune_threshold_btn.setEnabled(bad > 0)

    def _prune_threshold_and_resolve(self) -> None:
        if self._renderer is None:
            return
        bad_ids = [p["id"] for p in self._renderer.point_cloud if self._is_bad_point(p)]
        if bad_ids:
            self.resolve_requested.emit(bad_ids)

    def _delete_selected_and_resolve(self) -> None:
        bad_ids = [p["id"] for p in self._pending_selection if "id" in p]
        if not bad_ids:
            QMessageBox.information(self, "Nothing selected", "Select point-cloud points first.")
            return
        self.resolve_requested.emit(bad_ids)
        self._clear_pending_selection()
        self._update_button_states()

    def _update_info(self) -> None:
        if self._renderer is None:
            self._frame_info.setText("0 / 0")
            self._focal_info.setText("Focal length: -")
            return
        total = len(self._renderer.track)
        self._frame_info.setText(f"{self._current_index + 1} / {total}")
        entry = self._renderer.track[self._current_index]
        focal = entry.get("focal_length", 0.0)
        self._focal_info.setText(f"Focal length: {focal:.4f} px")

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        if self._renderer is not None:
            self._render_current()

    @staticmethod
    def _numpy_to_pixmap(array: np.ndarray) -> QPixmap:
        h, w, ch = array.shape
        bytes_per_line = ch * w
        image = QImage(array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)
