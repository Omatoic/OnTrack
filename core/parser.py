from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    # Normalise first to be safe.
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("Quaternion has zero length")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = [
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ]
    return R


def matrix_to_euler_xyz(R: list[list[float]]) -> tuple[float, float, float]:
    """Extract XYZ Euler angles (in degrees) from a rotation matrix."""
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])
    else:
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], sy)
        z = 0.0

    return (math.degrees(x), math.degrees(y), math.degrees(z))


def colmap_to_ae_matrix(R: list[list[float]]) -> list[list[float]]:
    """
    Convert a COLMAP rotation matrix to After Effects world space.
    COLMAP/GLOMAP: +X right, +Y down, +Z forward (into scene).
    AE            : +X right, +Y down, +Z away / into screen.
    Since the axis conventions match, no rotation conversion is required.
    """
    return [list(row) for row in R]


def colmap_to_blender_matrix(R: list[list[float]]) -> list[list[float]]:
    """
    Convert a COLMAP rotation matrix to Blender world space.
    COLMAP/GLOMAP: +X right, +Y down, +Z forward (into scene).
    Blender       : +X right, +Y forward, +Z up.
    The axis conversion is M_conv = [[1,0,0],[0,0,1],[0,-1,0]].
    Equivalent rotation: M_conv * R * M_conv^T.
    """
    M = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    return [[
        M[i][0] * R[0][0] * M[j][0] + M[i][0] * R[0][1] * M[j][1] + M[i][0] * R[0][2] * M[j][2] +
        M[i][1] * R[1][0] * M[j][0] + M[i][1] * R[1][1] * M[j][1] + M[i][1] * R[1][2] * M[j][2] +
        M[i][2] * R[2][0] * M[j][0] + M[i][2] * R[2][1] * M[j][1] + M[i][2] * R[2][2] * M[j][2]
        for j in range(3)
    ] for i in range(3)]


def matrix_to_euler_zxy(R: list[list[float]]) -> tuple[float, float, float]:
    """Extract ZXY Euler angles (in degrees) from a rotation matrix for After Effects."""
    # sin(X) = -R[2][1]
    sx = -R[2][1]
    sx = max(-1.0, min(1.0, sx))
    x = math.asin(sx)
    cx = math.sqrt(max(0.0, 1.0 - sx * sx))
    if cx < 1e-6:
        z = 0.0
        y = math.atan2(-R[1][0], R[0][0])
    else:
        z = math.atan2(R[2][0], R[2][2])
        y = math.atan2(R[0][1], R[1][1])
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def _principal_point(model: str, width: int, height: int, params: list[float]) -> tuple[float, float]:
    """Extract the principal point (cx, cy) from COLMAP camera model parameters."""
    model = model.lower()
    if model in ("pinhole", "simple_radial", "radial", "opengl"):
        # params: [fx, fy, cx, cy] or [fx, fy, cx, cy, ...]
        if len(params) >= 4:
            return params[2], params[3]
    elif model in ("simple_pinhole",):
        # params: [f, cx, cy]
        if len(params) >= 3:
            return params[1], params[2]
    return width / 2.0, height / 2.0


def parse_cameras_txt(path: Path) -> dict[int, dict[str, Any]]:
    """
    Parse COLMAP cameras.txt and return a mapping of camera_id -> metadata.
    Supports models where the first parameter is the focal length in pixels.
    """
    cameras: dict[int, dict[str, Any]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            focal = params[0] if params else 0.0
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "focal_length": focal,
            }
    return cameras


def parse_images_txt(path: Path) -> list[dict[str, Any]]:
    """
    Parse COLMAP images.txt.
    Each image spans two lines: a header line and a (possibly empty) 2D points line.
    """
    images: list[dict[str, Any]] = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        tx, ty, tz = (float(parts[5]), float(parts[6]), float(parts[7]))
        camera_id = int(parts[8])
        name = parts[9] if len(parts) > 9 else ""

        # Consume the 2D points line.
        if i < len(lines):
            i += 1

        images.append(
            {
                "image_id": image_id,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "camera_id": camera_id,
                "name": name,
            }
        )
    return images


def _frame_number_from_name(name: str) -> int:
    """Attempt to extract a frame number from a filename like frame_00001.jpg."""
    match = re.search(r"\d+", Path(name).stem)
    return int(match.group()) if match else 0


def build_camera_track(
    images_path: Path,
    cameras_path: Path,
    target: str = "ae",
) -> list[dict[str, Any]]:
    """
    Build a unified camera track from COLMAP sparse model files.

    target can be 'ae', 'blender', or 'raw'. 'raw' keeps COLMAP world coordinates
    and includes the original quaternion/translation for AR-style projection.
    Each returned entry contains:
        - name, frame_number
        - position [x, y, z]
        - orientation [rx, ry, rz] in degrees
        - focal_length (zoom in pixels)
        - width, height
    """
    cameras = parse_cameras_txt(cameras_path)
    images = parse_images_txt(images_path)

    if not images:
        raise ValueError(f"No images parsed from {images_path}")

    # Sort by filename so keyframes are in video order.
    images.sort(key=lambda img: _frame_number_from_name(img["name"]))

    track: list[dict[str, Any]] = []
    for img in images:
        R = quaternion_to_matrix(img["qw"], img["qx"], img["qy"], img["qz"])
        t = [img["tx"], img["ty"], img["tz"]]

        # COLMAP camera center: C = -R^T * T
        Rt = list(zip(*R))
        C = [-(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
             -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
             -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2])]

        # Keep the track in native GLOMAP coordinates and let exporters handle
        # target-application conversion at write time.
        R_target = R
        C_target = C
        orientation = matrix_to_euler_xyz(R)

        camera = cameras.get(img["camera_id"], {})
        focal = camera.get("focal_length", 0.0)
        width = camera.get("width", 0)
        height = camera.get("height", 0)
        params = camera.get("params", [])
        model = camera.get("model", "")
        cx, cy = _principal_point(model, width, height, params)

        entry: dict[str, Any] = {
            "name": img["name"],
            "frame_number": _frame_number_from_name(img["name"]),
            "position": C_target,
            "principal_point": [cx, cy],
            "orientation": orientation,
            "rotation_matrix": R_target,
            "focal_length": focal,
            "width": width,
            "height": height,
        }
        if target == "raw":
            entry["qw"] = img["qw"]
            entry["qx"] = img["qx"]
            entry["qy"] = img["qy"]
            entry["qz"] = img["qz"]
            entry["tx"] = img["tx"]
            entry["ty"] = img["ty"]
            entry["tz"] = img["tz"]

        track.append(entry)

    return track
