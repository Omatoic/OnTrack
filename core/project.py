from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from core.orientation import apply_orientation, apply_scale
from core.parser import build_camera_track
from core.pointcloud import parse_points3d_txt

PROJECT_VERSION = 1


def _to_native(obj: Any) -> Any:
    """Convert numpy arrays and Path objects to JSON-serializable natives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def save_project(
    project_dir: Path,
    input_video: Path | None,
    workspace_dir: Path | None = None,
    settings: dict[str, Any] | None = None,
    oriented_track: list[dict[str, Any]] | None = None,
    orientation: dict[str, Any] | None = None,
    scale: float = 1.0,
    z_rotation: float = 0.0,
    project_filename: str = "project.json",
    session: dict[str, Any] | None = None,
) -> Path:
    """Save a project by copying the workspace and writing metadata."""
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    workspace_name = "workspace" if project_filename == "project.json" else f"{Path(project_filename).stem}.workspace"

    if workspace_dir is not None:
        workspace_dir = Path(workspace_dir)
        saved_workspace = project_dir / workspace_name
        if not workspace_dir.exists():
            raise FileNotFoundError(f"Workspace folder not found: {workspace_dir}")
        if workspace_dir.resolve() != saved_workspace.resolve():
            staging = project_dir / f"{workspace_name}.saving"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(workspace_dir, staging)
            if saved_workspace.exists():
                shutil.rmtree(saved_workspace)
            staging.rename(saved_workspace)

    data: dict[str, Any] = {
        "version": PROJECT_VERSION,
        "input_video": str(input_video) if input_video else None,
        "settings": _to_native(settings or {}),
        "session": _to_native(session or {}),
        "workspace": workspace_name if workspace_dir is not None else None,
        "z_rotation": z_rotation,
        "scale": scale,
    }
    if oriented_track is not None:
        data["oriented_track"] = _to_native(oriented_track)
    if orientation is not None:
        data["orientation"] = _to_native(orientation)

    project_file = project_dir / project_filename
    with project_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return project_file


def load_project(project_file: Path) -> dict[str, Any]:
    """Load a project and return its metadata plus parsed tracks."""
    project_file = Path(project_file)
    with project_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    project_dir = project_file.parent
    workspace_name = data.get("workspace", "workspace")
    workspace_dir = project_dir / (workspace_name or "workspace")
    sparse_dir = workspace_dir / "03_sparse" / "0"
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    points3d_txt = sparse_dir / "points3D.txt"
    frames_dir = workspace_dir / "01_frames"

    result: dict[str, Any] = {
        "project_dir": project_dir,
        "workspace_dir": workspace_dir if workspace_name else None,
        "input_video": Path(data["input_video"]) if data.get("input_video") else None,
        "settings": data.get("settings") or {},
        "session": data.get("session") or {},
        "raw_track": None,
        "raw_track_colmap": None,
        "frames_dir": frames_dir if workspace_name else None,
        "sparse_dir": sparse_dir if workspace_name else None,
        "scale": data.get("scale", 1.0),
        "z_rotation": data.get("z_rotation", 0.0),
    }

    if not workspace_name:
        result["point_cloud"] = []
        return result

    if cameras_txt.exists() and images_txt.exists():
        raw_track_blender = build_camera_track(images_txt, cameras_txt, target="blender")
        raw_track_colmap = build_camera_track(images_txt, cameras_txt, target="raw")

        point_cloud = parse_points3d_txt(points3d_txt) if points3d_txt.exists() else []
        scale = float(data.get("scale", 1.0))
        if abs(scale - 1.0) > 1e-12:
            raw_track_blender, point_cloud = apply_scale(raw_track_blender, point_cloud, scale)
            raw_track_colmap, _ = apply_scale(raw_track_colmap, [], scale)

        orientation = data.get("orientation")
        result["raw_track"] = raw_track_blender
        result["raw_track_colmap"] = raw_track_colmap
        result["point_cloud"] = point_cloud
        result["orientation"] = orientation
        z_rotation = float(data.get("z_rotation", 0.0))
        if orientation is not None:
            R = np.asarray(orientation["rotation"])
            origin = np.asarray(orientation["origin"])
            if abs(z_rotation) > 1e-6:
                rad = math.radians(z_rotation)
                c, s = math.cos(rad), math.sin(rad)
                Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                R = Rz @ R
            result["oriented_track"] = apply_orientation(raw_track_blender, R, origin)
        else:
            result["oriented_track"] = None
    else:
        result["point_cloud"] = parse_points3d_txt(points3d_txt) if points3d_txt.exists() else []

    return result
