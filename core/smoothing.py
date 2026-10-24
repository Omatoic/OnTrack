"""Temporal smoothing for camera and object tracks.

Micro-jitter between frames is a classic matchmoving problem: the solve is
per-frame independent, so tiny pose noise shows up as shimmer. The standard
fix (also used by AliceVision's SfMTemporalFiltering node) is a
Savitzky–Golay filter over positions and quaternion-smoothed rotations.

Positions and quaternions are filtered independently; the world->camera
translation is then recomputed as t = -R @ C so every frame stays a
consistent rigid pose.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from core.parser import matrix_to_euler_xyz, quaternion_to_matrix


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return (w, x, y, z) for rotation matrix R."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _sign_align(qs: np.ndarray) -> np.ndarray:
    """Flip quaternions so consecutive ones lie on the same hemisphere."""
    out = qs.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def _odd_window(requested: int, n: int) -> int:
    """Clamp a smoothing window to an odd length <= n; 0 means no smoothing."""
    if requested < 3 or n < 3:
        return 0
    w = min(requested, n if n % 2 == 1 else n - 1)
    return w if w % 2 == 1 else w - 1


def smooth_track(track: list[dict[str, Any]], window: int, polyorder: int = 2) -> list[dict[str, Any]]:
    """Return a smoothed copy of a camera track.

    ``window`` is the Savitzky–Golay window in frames (odd, >= 3).
    Each entry keeps its keys; ``position``, ``rotation_matrix``,
    ``orientation`` and ``tx/ty/tz`` are replaced with smoothed values.
    """
    w = _odd_window(window, len(track))
    if w == 0:
        return [dict(e) for e in track]

    positions = np.array([e["position"] for e in track], dtype=np.float64)
    quats = _sign_align(np.array([
        _matrix_to_quaternion(np.asarray(e["rotation_matrix"], dtype=np.float64))
        for e in track
    ]))

    poly = min(polyorder, w - 1)
    sm_pos = savgol_filter(positions, w, poly, axis=0)
    sm_quat = savgol_filter(quats, w, poly, axis=0)
    sm_quat /= np.linalg.norm(sm_quat, axis=1, keepdims=True)

    out = []
    for i, entry in enumerate(track):
        e = dict(entry)
        R = _quaternion_to_matrix(sm_quat[i])
        C = sm_pos[i]
        t = -R @ C
        e["position"] = C.tolist()
        e["rotation_matrix"] = R.tolist()
        e["orientation"] = matrix_to_euler_xyz(R.tolist())
        if "tx" in e:
            e["tx"], e["ty"], e["tz"] = float(t[0]), float(t[1]), float(t[2])
            # Keep the stored quaternion consistent with the smoothed matrix.
            e["qw"], e["qx"], e["qy"], e["qz"] = sm_quat[i].tolist()
        out.append(e)
    return out


def smooth_object_track(object_track: list[dict[str, Any]], window: int,
                        polyorder: int = 2) -> list[dict[str, Any]]:
    """Smooth an object track (same fields as object_track.json)."""
    w = _odd_window(window, len(object_track))
    if w == 0:
        return [dict(e) for e in object_track]

    positions = np.array([e["position"] for e in object_track], dtype=np.float64)
    quats = _sign_align(np.array([
        _matrix_to_quaternion(np.asarray(e["orientation"], dtype=np.float64))
        for e in object_track
    ]))

    poly = min(polyorder, w - 1)
    sm_pos = savgol_filter(positions, w, poly, axis=0)
    sm_quat = savgol_filter(quats, w, poly, axis=0)
    sm_quat /= np.linalg.norm(sm_quat, axis=1, keepdims=True)

    out = []
    for i, entry in enumerate(object_track):
        e = dict(entry)
        e["position"] = sm_pos[i].tolist()
        e["orientation"] = _quaternion_to_matrix(sm_quat[i]).tolist()
        out.append(e)
    return out
