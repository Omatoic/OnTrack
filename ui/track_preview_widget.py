from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class TrackPreviewWidget(QWidget):
    """Matplotlib-based 3D preview of a camera track."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._track: list[dict[str, Any]] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(5, 4), facecolor="#1e1e1e")
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)

    def set_track(self, track: list[dict[str, Any]]) -> None:
        self._track = track
        self._draw()

    def _draw(self) -> None:
        self._figure.clear()
        if not self._track:
            self._canvas.draw()
            return

        ax = self._figure.add_subplot(111, projection="3d")
        ax.set_facecolor("#1e1e1e")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#444444")
        ax.yaxis.pane.set_edgecolor("#444444")
        ax.zaxis.pane.set_edgecolor("#444444")
        ax.tick_params(colors="#a0a0a0")
        ax.title.set_color("#f0f0f0")

        positions = np.array([e["position"] for e in self._track])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=1)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=positions[:, 2], cmap="viridis", s=20)

        # Draw viewing directions for every 10th frame.
        step = max(1, len(self._track) // 20)
        for i in range(0, len(self._track), step):
            entry = self._track[i]
            pos = np.asarray(entry["position"])
            # Forward vector in local camera space is roughly +Z rotated by orientation.
            # Approximate using consecutive positions.
            if i + 1 < len(self._track):
                fwd = np.asarray(self._track[i + 1]["position"]) - pos
                norm = np.linalg.norm(fwd)
                if norm > 1e-6:
                    fwd = fwd / norm * np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) * 0.05
                    ax.quiver(pos[0], pos[1], pos[2], fwd[0], fwd[1], fwd[2], color="r", arrow_length_ratio=0.3)

        ax.set_xlabel("X", color="#a0a0a0")
        ax.set_ylabel("Y", color="#a0a0a0")
        ax.set_zlabel("Z", color="#a0a0a0")
        self._canvas.draw()
