from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from core.parser import matrix_to_euler_zxy


# Basis that maps our oriented world (+X right, +Y forward, +Z up)
# to After Effects world (+X right, +Y down, +Z forward).
# AE(x, y, z) = (world_x, -world_z, world_y)
_WORLD_TO_AE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


# OpenCV/GLOMAP local camera (+X right, +Y down, +Z forward) to
# Blender local camera (+X right, +Y up, -Z forward). Matches blender_py.M_LOCAL_CAM.
_M_LOCAL_CAM = np.diag([1.0, -1.0, -1.0])


def _js_array(values: list[Any]) -> str:
    """Serialize a list as a JavaScript literal array."""
    return json.dumps(values)


def _to_ae_position(pos: list[float] | np.ndarray, scale: float = 1.0) -> list[float]:
    """Convert a 3D position from oriented world space to AE world space.

    Oriented world: +X right, +Y forward, +Z up.
    After Effects:  +X right, +Y down, +Z forward (into screen).
    Therefore:
        AE X =  world X
        AE Y = -world Z
        AE Z =  world Y
    """
    p = np.asarray(pos, dtype=np.float64)
    return [float(p[0]) * scale, float(-p[2]) * scale, float(p[1]) * scale]


def _to_ae_rotation(R: list[list[float]] | np.ndarray) -> list[list[float]]:
    """Convert a 3x3 rotation from oriented world space to AE world space."""
    R = np.asarray(R, dtype=np.float64)
    R_ae = _WORLD_TO_AE @ R @ _WORLD_TO_AE.T
    return R_ae.tolist()


def _matrix_to_euler_zyx(R: list[list[float]] | np.ndarray) -> tuple[float, float, float]:
    """Extract ZYX Euler angles (in degrees) from a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    sy = math.sqrt(R[0][0] * R[0][0] + R[0][1] * R[0][1])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(-R[1][2], R[2][2])
        y = math.atan2(R[0][2], sy)
        z = math.atan2(-R[0][1], R[0][0])
    else:
        x = math.atan2(R[2][1], R[1][1])
        y = math.atan2(R[0][2], sy)
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def generate_jsx(
    track: list[dict[str, Any]],
    fps: float,
    output_path: Path,
    comp_name: str = "",
    point_cloud: list[dict[str, Any]] | None = None,
    pixel_scale: float = 1000.0,
    sensor_width_mm: float = 36.0,
) -> Path:
    """
    Generate an After Effects JSX script from a parsed camera track.

    Parameters
    ----------
    track : list[dict]
        Camera track in oriented world coordinates (+X right, +Y forward, +Z up).
    fps : float
        Frame rate used to convert frame numbers to composition time.
    output_path : Path
        Where to write the .jsx file.
    comp_name : str, optional
        If provided, activate a composition with this name instead of the active item.
    point_cloud : list[dict], optional
        Optional point cloud used only to determine a custom origin marker.
    pixel_scale : float
        Scale factor from world units to AE pixels.
    sensor_width_mm : float
        Camera sensor width in millimeters used to compute AE camera zoom.
    """
    if not track:
        raise ValueError("Track is empty")

    frame_times: list[float] = []
    positions: list[list[float]] = []
    orientations: list[list[float]] = []

    # Match Blender's timeline: frame 1 is time 0 (scene.frame_start = 1).
    frame_start = min(1, min(int(e["frame_number"]) for e in track))

    for entry in track:
        frame = entry["frame_number"]
        frame_times.append(int(frame) - frame_start)

        # Convert position from oriented world to AE and scale to pixels.
        pos = _to_ae_position(entry["position"], pixel_scale)
        positions.append(pos)

        # Rebuild the same Blender matrix_world rotation used by the Blender
        # exporter: R_cw @ M_LOCAL_CAM (oriented data is already Z-up).
        R_wc = entry.get("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        R_cw = np.asarray(list(zip(*R_wc)), dtype=np.float64)
        R_blender = R_cw @ _M_LOCAL_CAM
        # Proven Blender -> AE conversion (ZYX Euler, X-90, negate Y and Z).
        x, y, z = _matrix_to_euler_zyx(R_blender)
        rot = [x - 90.0, -y, -z]
        # Unwrap to stay continuous with the previous keyframe so AE never
        # interpolates the long way around a +/-180 degree jump.
        if orientations:
            prev = orientations[-1]
            rot = [r + 360.0 * round((p - r) / 360.0) for r, p in zip(rot, prev)]
        orientations.append(rot)

    first = track[0]
    width = int(first.get("width", 1920) or 1920)
    height = int(first.get("height", 1080) or 1080)
    comp_center = [width / 2.0, height / 2.0]

    # focal_px is relative to the tracked image width; JS rescales it to the
    # comp width (zoom = lens * comp_width / sensor, as in the proven script).
    focal_px = float(first.get("focal_length", 0.0))

    # Origin marker in AE space.
    origin = [0.0, 0.0, 0.0]
    if point_cloud and "origin" in point_cloud[0]:
        origin = _to_ae_position(point_cloud[0]["origin"], pixel_scale)

    comp_check = (
        f'var comp = app.project.itemByName({json.dumps(comp_name)});\n'
        if comp_name
        else "var comp = app.project.activeItem;\n"
    )

    jsx = f"""(function () {{
    app.beginUndoGroup("Import GLOMAP Track (Direct Camera)");

    {comp_check}
    if (!comp || !(comp instanceof CompItem)) {{
        alert("Please select or name a composition first.");
        return;
    }}

    var compWidth = comp.width;
    var compHeight = comp.height;
    var halfW = compWidth / 2;
    var halfH = compHeight / 2;

    // 1. World origin marker centered on the comp frame.
    var originSolid = comp.layers.addSolid([1.0, 0.0, 0.0], "GLOMAP_Origin", 300, 300, 1.0, comp.duration);
    originSolid.threeDLayer = true;
    originSolid.property("Transform").property("Position").setValue([halfW, halfH, 0]);
    originSolid.property("Transform").property("X Rotation").setValue(90);

    // 2. Create camera and set orientation directly.
    var camera = comp.layers.addCamera("GLOMAP 3D Camera", [halfW, halfH]);
    try {{ camera.autoOrient = AutoOrientType.NO_AUTO_ORIENT; }} catch (e) {{}}
    var posProp = camera.property("Transform").property("Position");
    var orientProp = camera.property("Transform").property("Orientation");
    var zoomProp = camera.property("Camera Options").property("Zoom");

    // Keyframes are placed by frame index on the comp's own timeline, so the
    // number of animated frames always matches the tracked frame count.
    var frameIndices = {_js_array(frame_times)};
    var positions = {_js_array(positions)};
    var orientations = {_js_array(orientations)};
    var imageWidth = {width};
    var focalPx = {focal_px};
    var correctZoom = focalPx * compWidth / imageWidth;

    for (var i = 0; i < frameIndices.length; i++) {{
        var t = frameIndices[i] * comp.frameDuration;
        var p = positions[i];
        var o = orientations[i];
        var pos = [p[0] + halfW, p[1] + halfH, p[2]];
        posProp.setValueAtTime(t, pos);
        orientProp.setValueAtTime(t, o);
        zoomProp.setValueAtTime(t, correctZoom);
    }}

    app.endUndoGroup();
}})();
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(jsx, encoding="utf-8")
    return output_path


def generate_diagnostic_jsx(
    track: list[dict[str, Any]],
    output_path: Path,
    pixel_scale: float = 1000.0,
) -> Path:
    """
    Generate a diagnostic JSX script that plots every frame as a 3D null.

    Use this to verify the position path is correct before applying rotation.
    """
    if not track:
        raise ValueError("Track is empty")

    positions: list[list[float]] = []
    for entry in track:
        positions.append(_to_ae_position(entry["position"], pixel_scale))

    comp_check = "var comp = app.project.activeItem;\n"

    jsx = f"""(function () {{
    app.beginUndoGroup("GLOMAP Position Diagnostic");

    {comp_check}
    if (!comp || !(comp instanceof CompItem)) {{
        alert("Please select a composition.");
        return;
    }}

    var halfW = comp.width / 2;
    var halfH = comp.height / 2;

    var originSolid = comp.layers.addSolid([1.0, 0.0, 0.0], "TARGET_ORIGIN", 100, 100, 1.0, comp.duration);
    originSolid.threeDLayer = true;
    originSolid.property("Transform").property("Position").setValue([halfW, halfH, 0]);

    var positions = {_js_array(positions)};

    for (var i = 0; i < positions.length; i++) {{
        var p = positions[i];
        var nullLayer = comp.layers.addNull();
        nullLayer.threeDLayer = true;
        nullLayer.name = "Pos_Frame_" + i;
        nullLayer.property("Transform").property("Position").setValue([p[0] + halfW, p[1] + halfH, p[2]]);
        nullLayer.inPoint = i * 0.041667;
        nullLayer.outPoint = (i + 1) * 0.041667;
    }}

    app.endUndoGroup();
}})();
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(jsx, encoding="utf-8")
    return output_path
