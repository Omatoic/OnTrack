from __future__ import annotations

import math
from typing import Any

import numpy as np


# The world-up axis used for the ground plane. After Effects world space
# traditionally uses +Z as up, so floor points end up with Z ~= 0.
WORLD_UP = np.array([0.0, 0.0, 1.0])


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert XYZ Euler angles (degrees) to a 3x3 rotation matrix."""
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Extract XYZ Euler angles (degrees) from a rotation matrix."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion (qw, qx, qy, qz)."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (qw / norm, qx / norm, qy / norm, qz / norm)


def _rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula: rotation matrix from normalized axis and angle (radians)."""
    k = _normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


def estimate_world_up(track: list[dict[str, Any]]) -> np.ndarray | None:
    """
    Estimate the scene's true 'up' direction in the track's world frame as the mean
    camera up vector (negated camera +Y axis, since COLMAP cameras have +Y down).
    """
    ups = []
    for entry in track:
        R = entry.get("rotation_matrix")
        if R is None:
            continue
        ups.append(-np.asarray(R, dtype=np.float64)[1, :])
    if not ups:
        return None
    mean = np.mean(ups, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else None


def estimate_world_forward(track: list[dict[str, Any]]) -> np.ndarray | None:
    """Mean camera view direction (camera +Z axis) in the track's world frame."""
    fwd = []
    for entry in track:
        R = entry.get("rotation_matrix")
        if R is None:
            continue
        fwd.append(np.asarray(R, dtype=np.float64)[2, :])
    if not fwd:
        return None
    mean = np.mean(fwd, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else None


def resolve_floor(
    floor_points: list[np.ndarray],
    origin: np.ndarray | None = None,
    up_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the rigid transformation that aligns the floor plane with the world-up axis.

    Steps:
      1. Use SVD on all selected floor points to find the best-fit plane normal.
      2. Ensure the normal points upward (positive dot with WORLD_UP).
      3. Build an axis-angle rotation that maps the floor normal onto WORLD_UP.

    Returns
    -------
    rotation : 3x3 rotation matrix R_align
    origin : 3D point used as the new origin
    """
    if len(floor_points) < 3:
        raise ValueError("At least 3 floor points are required")

    pts = np.asarray(floor_points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    origin = centroid if origin is None else np.asarray(origin, dtype=np.float64)

    # SVD-based least-squares plane fit through the centered floor points.
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered)
    normal = _normalize(vt[-1])

    # Make sure the normal points upward. In the raw COLMAP frame +Z is the view
    # direction, not up, so prefer the cameras' real up direction when known.
    reference_up = WORLD_UP if up_hint is None else np.asarray(up_hint, dtype=np.float64)
    if np.dot(normal, reference_up) < 0.0:
        normal = -normal

    # If a user-selected origin was supplied, snap it vertically onto the floor
    # plane so that after rotation the floor lies at Z = 0 and the cube/object
    # placed at the origin sits on the floor.
    origin = origin - np.dot(origin - centroid, normal) * normal

    # Axis-angle rotation that aligns the floor normal with world up.
    dot = np.clip(np.dot(normal, WORLD_UP), -1.0, 1.0)
    angle = math.acos(dot)
    if abs(angle) < 1e-9:
        R_align = np.eye(3)
    else:
        axis = np.cross(normal, WORLD_UP)
        axis = _normalize(axis)
        R_align = _rotation_matrix_axis_angle(axis, angle)

    return R_align, origin


def _rotation_aligning(source: np.ndarray | None, target: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction ``source`` onto ``target`` (identity if unknown)."""
    if source is None:
        return np.eye(3)
    s = _normalize(np.asarray(source, dtype=np.float64))
    t = _normalize(np.asarray(target, dtype=np.float64))
    dot = float(np.clip(np.dot(s, t), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3)
    axis = np.cross(s, t)
    if np.linalg.norm(axis) < 1e-9:
        # Anti-parallel: rotate 180 degrees about any perpendicular axis.
        axis = np.cross(s, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(s, np.array([0.0, 1.0, 0.0]))
    return _rotation_matrix_axis_angle(axis, math.acos(dot))


def resolve_floor_and_wall(
    floor_points: list[np.ndarray],
    wall_points: list[np.ndarray],
    origin: np.ndarray | None = None,
    up_hint: np.ndarray | None = None,
    forward_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align the floor to the world-up plane and, optionally, rotate around world up
    so a selected wall plane becomes vertical (its normal is perpendicular to WORLD_UP).

    Without a wall, the heading is fixed by turning the mean camera view direction
    to point along +Y, so the result does not depend on an arbitrary yaw.
    """
    if len(floor_points) >= 3:
        R_align, origin = resolve_floor(floor_points, origin, up_hint)
    else:
        # No floor: level the scene using the cameras' up direction and keep the
        # user's origin exactly (no snapping onto a plane).
        R_align = _rotation_aligning(up_hint, WORLD_UP)
        if origin is None:
            origin = (
                np.mean(np.asarray(wall_points, dtype=np.float64), axis=0)
                if len(wall_points) > 0
                else np.zeros(3)
            )
        else:
            origin = np.asarray(origin, dtype=np.float64)

    if len(wall_points) < 3:
        if forward_hint is not None:
            f = R_align @ np.asarray(forward_hint, dtype=np.float64)
            if math.hypot(f[0], f[1]) > 1e-6:
                delta = math.pi / 2.0 - math.atan2(f[1], f[0])
                R_align = _rotation_matrix_axis_angle(WORLD_UP, delta) @ R_align
        return R_align, origin

    # Transform wall points into floor-aligned space.
    wall_aligned = np.array([R_align @ (np.asarray(p) - origin) for p in wall_points])
    wall_centroid = wall_aligned.mean(axis=0)
    centered = wall_aligned - wall_centroid
    _, _, vt = np.linalg.svd(centered)
    wall_normal = _normalize(vt[-1])

    # Project the wall normal onto the world-up plane and compute the rotation
    # around world up that makes the wall vertical.
    up = WORLD_UP
    n_proj = wall_normal - np.dot(wall_normal, up) * up
    n_len = np.linalg.norm(n_proj)
    if n_len < 1e-6:
        return R_align, origin

    n_proj = n_proj / n_len
    # We want the wall's normal to align with the +X axis in floor-aligned space.
    target_x = np.array([1.0, 0.0, 0.0])
    cos_a = np.clip(np.dot(n_proj, target_x), -1.0, 1.0)
    angle = math.acos(cos_a)
    cross = np.cross(n_proj, target_x)
    if np.dot(cross, up) < 0.0:
        angle = -angle

    if abs(angle) < 1e-9:
        return R_align, origin

    R_wall = _rotation_matrix_axis_angle(up, angle)
    return R_wall @ R_align, origin


def _make_4x4(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transformation matrix."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(rotation, dtype=np.float64)
    M[:3, 3] = np.asarray(translation, dtype=np.float64)
    return M


def apply_orientation(
    track: list[dict[str, Any]],
    rotation: np.ndarray,
    origin: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Apply a world-space rigid transformation to the camera track using 4x4 matrix composition.

    The world transform is: P_new = R_align @ (P_old - P_origin)
    Encoded as M_world = [R_align | -R_align @ P_origin; 0 | 1].

    For each camera we build its Camera-to-World matrix M_c2w_old and multiply
    by M_world on the left: M_c2w_new = M_world @ M_c2w_old.
    This guarantees that positions and orientations share the exact same pivot
    and rotation, eliminating parallax/sliding artifacts.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)

    # Single world rigid transformation.
    M_world = _make_4x4(rotation, -rotation @ origin)

    oriented = []
    for entry in track:
        pos = np.asarray(entry["position"], dtype=np.float64)
        R_w2c = np.asarray(
            entry.get("rotation_matrix", euler_xyz_to_matrix(*entry["orientation"])),
            dtype=np.float64,
        )

        # Camera-to-World from stored World-to-Camera data.
        R_c2w = R_w2c.T
        M_c2w_old = _make_4x4(R_c2w, pos)
        M_c2w_new = M_world @ M_c2w_old

        C_new = M_c2w_new[:3, 3]
        R_c2w_new = M_c2w_new[:3, :3]
        R_w2c_new = R_c2w_new.T

        new_entry: dict[str, Any] = {
            **entry,
            "position": C_new.tolist(),
            "rotation_matrix": R_w2c_new.tolist(),
            "orientation": matrix_to_euler_xyz(R_w2c_new),
        }

        # If the entry stores COLMAP-style raw quaternions and translations,
        # update them so AR-style projection continues to work.
        if {"qw", "qx", "qy", "qz", "tx", "ty", "tz"}.issubset(entry.keys()):
            qw, qx, qy, qz = _matrix_to_quaternion(R_w2c_new)
            t_new = -R_w2c_new @ C_new
            new_entry.update({
                "qw": float(qw),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "tx": float(t_new[0]),
                "ty": float(t_new[1]),
                "tz": float(t_new[2]),
            })

        oriented.append(new_entry)

    return oriented


def compute_scale(p1: np.ndarray, p2: np.ndarray, known_distance: float) -> float:
    """Return the scale factor that makes the distance between p1 and p2 equal to known_distance."""
    current_distance = np.linalg.norm(np.asarray(p2) - np.asarray(p1))
    if current_distance < 1e-9:
        raise ValueError("Selected points are too close to compute scale")
    return known_distance / current_distance


def apply_scale(
    track: list[dict[str, Any]],
    point_cloud: list[dict[str, Any]],
    scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply a uniform scale to a camera track and point cloud."""
    scaled_track = []
    for entry in track:
        pos = np.asarray(entry["position"])
        scaled_entry = {**entry, "position": (pos * scale).tolist()}
        if {"qw", "qx", "qy", "qz", "tx", "ty", "tz"}.issubset(entry.keys()):
            scaled_entry["tx"] = entry["tx"] * scale
            scaled_entry["ty"] = entry["ty"] * scale
            scaled_entry["tz"] = entry["tz"] * scale
        scaled_track.append(scaled_entry)

    scaled_cloud = []
    for p in point_cloud:
        pos = np.asarray(p["position"])
        scaled_cloud.append({**p, "position": (pos * scale).tolist()})

    return scaled_track, scaled_cloud
