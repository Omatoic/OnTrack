from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# GLOMAP/COLMAP world: +X right, +Y down, +Z forward (into scene).
# Blender world:        +X right, +Y forward, +Z up.
M_WORLD_SWAP: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# OpenCV/GLOMAP local camera: +X right, +Y down, +Z forward (view direction).
# Blender default camera:     +X right, +Y up,   -Z forward (view direction).
M_LOCAL_CAM: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# Oriented data (floor on Z=0, +Z up) is already in Blender's world frame.
M_ORIENTED_FLIP: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

DEFAULT_SENSOR_WIDTH_MM = 36.0


def _matmul_4x4(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two 4x4 matrices represented as nested lists."""
    return [[
        A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3] * B[3][j]
        for j in range(4)
    ] for i in range(4)]


def _matmul_4x4_vec3(M: list[list[float]], v: list[float]) -> list[float]:
    """Apply a 4x4 matrix to a 3D point (homogeneous w=1)."""
    x = M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2] + M[0][3]
    y = M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2] + M[1][3]
    z = M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2] + M[2][3]
    return [x, y, z]


def _transpose_3x3(R: list[list[float]]) -> list[list[float]]:
    return [[R[j][i] for j in range(3)] for i in range(3)]


def _matmul_3x3_vec3(R: list[list[float]], v: list[float]) -> list[float]:
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def _make_4x4(R: list[list[float]], t: list[float]) -> list[list[float]]:
    """Build a 4x4 homogeneous matrix from a 3x3 rotation and 3D translation."""
    return [
        [R[0][0], R[0][1], R[0][2], t[0]],
        [R[1][0], R[1][1], R[1][2], t[1]],
        [R[2][0], R[2][1], R[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _compute_camera_to_world_matrix(
    R_glomap: list[list[float]],
    T_glomap: list[float],
) -> list[list[float]]:
    """
    Convert GLOMAP World-to-Camera parameters to a Camera-to-World 4x4 matrix.

    GLOMAP: P_cam = R_glomap @ P_world + T_glomap
    Camera-to-World:
        R_world = R_glomap.T
        C_world = -R_world @ T_glomap
    """
    R_world = _transpose_3x3(R_glomap)
    C_world = _matmul_3x3_vec3(R_world, [-T_glomap[0], -T_glomap[1], -T_glomap[2]])
    return _make_4x4(R_world, C_world)


def _compute_blender_camera_matrix(
    R_glomap: list[list[float]],
    T_glomap: list[float],
    apply_world_swap: bool = True,
) -> list[list[float]]:
    """
    Build a Blender-ready camera matrix_world from raw GLOMAP parameters.

    When ``apply_world_swap`` is True, applies the world-axis swap to both
    rotation and translation (GLOMAP -> Blender Z-up). When the input is already
    in Blender's Z-up coordinate frame (e.g. after AR orientation), set this to
    False so only the local camera frame alignment is applied.
    """
    mat_world = _compute_camera_to_world_matrix(R_glomap, T_glomap)
    # M_LOCAL_CAM on the right converts the camera's local frame.
    mat_blender = _matmul_4x4(mat_world, M_LOCAL_CAM)
    world = M_WORLD_SWAP if apply_world_swap else M_ORIENTED_FLIP
    return _matmul_4x4(world, mat_blender)


def _transform_point_to_blender(p: list[float], apply_world_swap: bool = True) -> list[float]:
    """Map a world point into Blender Z-up world space."""
    world = M_WORLD_SWAP if apply_world_swap else M_ORIENTED_FLIP
    return _matmul_4x4_vec3(world, p)


def _focal_px_to_mm(focal_px: float, image_width_px: int, sensor_width_mm: float) -> float:
    """Convert focal length from pixels to millimeters."""
    if image_width_px <= 0:
        raise ValueError("image_width_px must be positive")
    return focal_px * (sensor_width_mm / image_width_px)


def _extract_glomap_params(entry: dict[str, Any]) -> tuple[list[list[float]], list[float]]:
    """
    Extract raw GLOMAP World-to-Camera rotation and translation from a track entry.

    The entry may store either the raw translation (tx, ty, tz) or the camera
    center in ``position``. When only the camera center is available, recover
    T_glomap = -R_glomap @ C.
    """
    R_glomap = entry.get("rotation_matrix", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if {"tx", "ty", "tz"}.issubset(entry.keys()):
        T_glomap = [float(entry["tx"]), float(entry["ty"]), float(entry["tz"])]
    else:
        C = [float(v) for v in entry["position"]]
        T_glomap = _matmul_3x3_vec3(R_glomap, [-C[0], -C[1], -C[2]])

    return R_glomap, T_glomap


def export_glomap_to_blender(
    scene_data: dict[str, Any],
    output_filepath: Path | str,
    world_space: str = "glomap",
) -> Path:
    """
    Export GLOMAP/COLMAP scene data to a self-contained Blender Python script.

    Parameters
    ----------
    scene_data : dict
        Must contain:
            - ``track``: list of frame entries with ``frame_number``,
              ``rotation_matrix`` (R_glomap), and either ``tx/ty/tz`` (T_glomap)
              or ``position`` (camera center C). Each entry should also provide
              ``focal_length`` and ``width`` for lens conversion.
            - ``point_cloud`` (optional): list of dicts with ``position`` keys.
            - ``camera_name`` (optional): name for the created camera object.
            - ``sensor_width_mm`` (optional): defaults to 36.0.
    output_filepath : Path or str
        Destination .py file path.
    world_space : str
        Either ``"glomap"`` (default) for raw GLOMAP/COLMAP coordinates, or
        ``"blender"`` if the input track and point cloud are already in Blender's
        Z-up coordinate frame (e.g. after AR orientation).

    Returns
    -------
    Path
        The written file path.
    """
    track: list[dict[str, Any]] = scene_data.get("track", [])
    if not track:
        raise ValueError("scene_data['track'] is empty")

    apply_world_swap = world_space == "glomap"

    point_cloud: list[dict[str, Any]] = scene_data.get("point_cloud") or []
    camera_name: str = scene_data.get("camera_name", "GLOMAP Camera")
    sensor_width_mm: float = float(scene_data.get("sensor_width_mm", DEFAULT_SENSOR_WIDTH_MM))

    first = track[0]
    img_width = int(first.get("width", 1920) or 1920)

    frame_data: list[dict[str, Any]] = []
    for entry in track:
        frame = entry["frame_number"]
        R_glomap, T_glomap = _extract_glomap_params(entry)
        matrix_world = _compute_blender_camera_matrix(R_glomap, T_glomap, apply_world_swap)
        focal_px = entry.get("focal_length") or 1.0
        lens_mm = _focal_px_to_mm(float(focal_px), img_width, sensor_width_mm)

        frame_data.append({
            "frame_number": frame,
            "matrix_world": matrix_world,
            "lens_mm": lens_mm,
        })

    cloud_points: list[list[float]] = []
    for p in point_cloud:
        cloud_points.append(_transform_point_to_blender(p["position"], apply_world_swap))

    payload: dict[str, Any] = {
        "camera_name": camera_name,
        "sensor_width_mm": sensor_width_mm,
        "frame_data": frame_data,
        "point_cloud": cloud_points,
    }

    script = _generate_blender_script(payload)

    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")
    return output_path


def _generate_blender_script(payload: dict[str, Any]) -> str:
    """Build a standalone Blender Python script from processed payload data."""
    data_json = json.dumps(payload, indent=2)

    return f"""import bpy
import mathutils
import json

DATA = json.loads({json.dumps(data_json)})

CAMERA_NAME = DATA["camera_name"]
SENSOR_WIDTH_MM = DATA["sensor_width_mm"]
FRAME_DATA = DATA["frame_data"]
POINT_CLOUD = DATA["point_cloud"]

scene = bpy.context.scene
scene.frame_set(1)

# Create camera
cam_data = bpy.data.cameras.new(name=CAMERA_NAME)
cam_data.sensor_width = SENSOR_WIDTH_MM
cam_data.sensor_fit = 'AUTO'
cam_obj = bpy.data.objects.new(name=CAMERA_NAME, object_data=cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.view_layer.objects.active = cam_obj

for frame_info in FRAME_DATA:
    frame = frame_info["frame_number"]
    matrix_world = frame_info["matrix_world"]
    lens_mm = frame_info["lens_mm"]

    scene.frame_set(frame)
    cam_obj.matrix_world = mathutils.Matrix(matrix_world)
    cam_data.lens = lens_mm
    cam_obj.keyframe_insert(data_path="location")
    cam_obj.keyframe_insert(data_path="rotation_euler")
    cam_data.keyframe_insert(data_path="lens")

# Point cloud as a real mesh object: one small closed tetrahedron per point
# (visible from every angle, no flat zero-thickness faces that shimmer).
if POINT_CLOUD:
    xs = [p[0] for p in POINT_CLOUD]
    ys = [p[1] for p in POINT_CLOUD]
    zs = [p[2] for p in POINT_CLOUD]
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6)
    s = extent * 0.003
    offsets = (
        (s, s, s),
        (s, -s, -s),
        (-s, s, -s),
        (-s, -s, s),
    )
    tet_faces = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
    verts = []
    faces = []
    for i, p in enumerate(POINT_CLOUD):
        base = 4 * i
        for o in offsets:
            verts.append((p[0] + o[0], p[1] + o[1], p[2] + o[2]))
        for f in tet_faces:
            faces.append((base + f[0], base + f[1], base + f[2]))
    mesh = bpy.data.meshes.new("GLOMAP_Point_Cloud")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    cloud_obj = bpy.data.objects.new("GLOMAP_Point_Cloud", mesh)
    bpy.context.collection.objects.link(cloud_obj)
    print(f"Imported {{len(POINT_CLOUD)}} point-cloud points")

origin_marker = bpy.data.objects.new("GLOMAP_Origin", None)
origin_marker.empty_display_type = 'ARROWS'
origin_marker.empty_display_size = 0.5
bpy.context.collection.objects.link(origin_marker)

scene.frame_set(1)
print("[debug] camera frame-1 location:", tuple(round(v, 4) for v in cam_obj.matrix_world.translation))
print("[debug] camera frame-1 forward (-Z) :", tuple(round(v, 3) for v in (cam_obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, -1)))))
if POINT_CLOUD:
    print("[debug] cloud min:", (round(min(xs), 3), round(min(ys), 3), round(min(zs), 3)))
    print("[debug] cloud max:", (round(max(xs), 3), round(max(ys), 3), round(max(zs), 3)))
print(f"GLOMAP track imported: {{len(FRAME_DATA)}} keyframes, {{len(POINT_CLOUD)}} cloud points")
"""


def generate_py(
    track: list[dict[str, Any]],
    fps: float,
    output_path: Path,
    camera_name: str = "GLOMAP Camera",
    point_cloud: list[dict[str, Any]] | None = None,
    world_space: str = "glomap",
) -> Path:
    """
    Legacy wrapper around :func:`export_glomap_to_blender`.

    ``fps`` is accepted for API compatibility but is not used in the Blender script
    (keyframes are placed at the original frame numbers).
    """
    scene_data: dict[str, Any] = {
        "track": track,
        "point_cloud": point_cloud,
        "camera_name": camera_name,
    }
    return export_glomap_to_blender(scene_data, output_path, world_space=world_space)
