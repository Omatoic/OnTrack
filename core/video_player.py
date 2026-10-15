from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoSource:
    """Lightweight OpenCV-based video source for frame-accurate preview."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.fps = 24.0
        self.width = 0
        self.height = 0
        self.duration = 0.0
        self._open()

    def _open(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self._cap = cap
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0

    def read_frame(self, index: int) -> np.ndarray | None:
        """Read a single BGR frame by index. Returns None on failure."""
        if self._cap is None:
            return None
        if index < 0:
            index = 0
        if index >= self.frame_count:
            index = max(0, self.frame_count - 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ret, frame = self._cap.read()
        return frame if ret else None

    def read_rgb(self, index: int) -> np.ndarray | None:
        """Read a single RGB frame by index."""
        frame = self.read_frame(index)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
