from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def parse_points3d_txt(path: Path) -> list[dict[str, Any]]:
    """Parse a COLMAP points3D.txt file into a list of point records."""
    points = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            # Remaining parts are (image_id, point2d_idx) pairs; ignore for basic use.
            points.append({
                "id": point_id,
                "position": np.array([x, y, z], dtype=np.float64),
                "color": (b, g, r),
            })
    return points
