from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MAX_PREVIEW_POINTS = 5000


def _subsample_cloud(cloud: list[dict[str, Any]], max_points: int = MAX_PREVIEW_POINTS) -> list[dict[str, Any]]:
    """Return a uniformly random subset of the point cloud to keep rendering fast."""
    if len(cloud) <= max_points:
        return cloud
    indices = np.random.choice(len(cloud), max_points, replace=False)
    return [cloud[i] for i in indices]

from core.orientation import (
    apply_orientation as apply_orientation_to_track,
    apply_scale,
    estimate_world_forward,
    estimate_world_up,
    resolve_floor_and_wall,
)


def _quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ])
    return R


def build_camera_matrix(focal_length_px: float, width: int, height: int) -> np.ndarray:
    """Build a 3x3 COLMAP/OpenCV camera matrix from focal length and image size."""
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [focal_length_px, 0.0, cx],
        [0.0, focal_length_px, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def cube_object(size: float = 1.0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return cube vertices and edges with the origin at the bottom face center."""
    s = size / 2.0
    vertices = np.array([
        [-s, -s, 0.0],
        [ s, -s, 0.0],
        [ s,  s, 0.0],
        [-s,  s, 0.0],
        [-s, -s, size],
        [ s, -s, size],
        [ s,  s, size],
        [-s,  s, size],
    ], dtype=np.float64)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return vertices, edges


def axes_object(length: float = 1.0) -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int, int]]]]:
    """Return axes vertices and colored edges."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
        [0.0, 0.0, length],
    ], dtype=np.float64)
    edges = [
        (0, 1, (0, 0, 255)),   # X = red
        (0, 2, (0, 255, 0)),   # Y = green
        (0, 3, (255, 0, 0)),   # Z = blue
    ]
    return vertices, edges


def grid_object(size: float = 10.0, divisions: int = 10) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return a flat grid on the XY plane, useful as a floor reference (Z-up world)."""
    half = size / 2.0
    step = size / divisions
    vertices: list[list[float]] = []
    edges: list[tuple[int, int]] = []

    # Lines parallel to X axis.
    for i in range(divisions + 1):
        y = -half + i * step
        vertices.append([-half, y, 0.0])
        vertices.append([half, y, 0.0])
        edges.append((len(vertices) - 2, len(vertices) - 1))

    # Lines parallel to Y axis.
    for i in range(divisions + 1):
        x = -half + i * step
        vertices.append([x, -half, 0.0])
        vertices.append([x, half, 0.0])
        edges.append((len(vertices) - 2, len(vertices) - 1))

    return np.array(vertices, dtype=np.float64), edges


def project_points(
    points_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray | None:
    """Project 3D points to 2D image coordinates using OpenCV."""
    if camera_matrix[0, 0] <= 0:
        return None
    projected, _ = cv2.projectPoints(
        points_3d,
        rvec,
        tvec,
        camera_matrix,
        None,
    )
    return projected.reshape(-1, 2)


class ARPreviewRenderer:
    """Render a 3D object and point cloud overlaid on tracked frames."""

    def __init__(
        self,
        frames_dir: Path,
        track: list[dict[str, Any]],
        point_cloud: list[dict[str, Any]] | None = None,
        object_track: list[dict[str, Any]] | None = None,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self._original_track = track
        self.track = track
        self._original_point_cloud = _subsample_cloud(point_cloud or [])
        self.point_cloud = list(self._original_point_cloud)
        self._original_object_track = object_track or []
        self.object_track = list(self._original_object_track)
        self.object_type: str = "cube"
        self.object_size: float = 1.0
        self.object_position: tuple[float, float, float] = (0.0, 0.0, 2.0)
        self._original_object_position: tuple[float, float, float] = (0.0, 0.0, 2.0)
        self._orientation_rotation: np.ndarray | None = None
        self._orientation_origin: np.ndarray | None = None
        self._z_rotation: float = 0.0  # Extra Z-axis rotation in degrees
        self.error_threshold: float = 0.0  # 0 = no error coloring
        self.min_track_len: int = 0        # points with shorter tracks are bad too

        # Caches to avoid recomputing every frame.
        self._frame_cache: dict[int, np.ndarray] = {}
        self._cloud_projection_cache: dict[int, np.ndarray] = {}

    def set_object(self, object_type: str, size: float, position: tuple[float, float, float]) -> None:
        self.object_type = object_type
        self.object_size = size
        self.object_position = position
        # Update the base object position so orientation/scaling does not revert
        # a deliberate user placement (e.g. snapping the object to the selected origin).
        self._original_object_position = position

    def _get_vertices(self) -> np.ndarray:
        if self.object_type == "axes":
            vertices, _ = axes_object(self.object_size)
        elif self.object_type == "grid":
            vertices, _ = grid_object(self.object_size)
        else:
            vertices, _ = cube_object(self.object_size)
        return vertices + np.array(self.object_position)

    def _get_edges(self) -> list[tuple[int, int]] | list[tuple[int, int, tuple[int, int, int]]]:
        if self.object_type == "axes":
            _, edges = axes_object(self.object_size)
            return edges
        if self.object_type == "grid":
            _, edges = grid_object(self.object_size)
            return edges
        _, edges = cube_object(self.object_size)
        return edges

    def project_point_cloud(self, entry_index: int) -> np.ndarray | None:
        """Return projected 2D points for every visible point-cloud point."""
        if not self.point_cloud:
            return None
        if entry_index in self._cloud_projection_cache:
            return self._cloud_projection_cache[entry_index]

        entry = self.track[entry_index]
        focal = entry.get("focal_length", 0.0)
        width = entry.get("width", 0)
        height = entry.get("height", 0)
        if focal <= 0:
            return None
        camera_matrix = build_camera_matrix(focal, width, height)

        R = _quaternion_to_matrix(entry["qw"], entry["qx"], entry["qy"], entry["qz"])
        t = np.array([entry["tx"], entry["ty"], entry["tz"]], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)

        positions = np.array([p["position"] for p in self.point_cloud], dtype=np.float64)
        projected, _ = cv2.projectPoints(positions, rvec, t, camera_matrix, None)
        result = projected.reshape(-1, 2)
        self._cloud_projection_cache[entry_index] = result
        return result

    def find_nearest_point(self, screen_x: float, screen_y: float, entry_index: int) -> dict[str, Any] | None:
        """Find the point-cloud point nearest to the given screen coordinate."""
        projected = self.project_point_cloud(entry_index)
        if projected is None:
            return None
        coords = np.array([screen_x, screen_y])
        distances = np.linalg.norm(projected - coords, axis=1)
        idx = int(np.argmin(distances))
        # Only pick if reasonably close to a visible dot.
        if distances[idx] > 20:
            return None
        return self.point_cloud[idx]

    def apply_orientation(
        self,
        floor_points: list[np.ndarray],
        wall_points: list[np.ndarray],
        origin_point: np.ndarray | None = None,
    ) -> None:
        """Reorient the track, point cloud, and object so the floor is flat and a wall is vertical."""
        rotation, origin = resolve_floor_and_wall(
            floor_points,
            wall_points,
            origin_point,
            up_hint=estimate_world_up(self._original_track),
            forward_hint=estimate_world_forward(self._original_track),
        )
        self.set_orientation(rotation, origin)

    def set_z_rotation(self, degrees: float) -> None:
        """Set an additional rotation around world Z applied after orientation."""
        self._z_rotation = float(degrees)
        self._cloud_projection_cache.clear()
        self._recompute_scene()

    def _z_rotation_matrix(self) -> np.ndarray:
        angle = math.radians(self._z_rotation)
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _recompute_scene(self) -> None:
        """Recompute oriented track/cloud/object from raw data and current Z rotation."""
        if self._orientation_rotation is None or self._orientation_origin is None:
            R = self._z_rotation_matrix()
            self.track = apply_orientation_to_track(self._original_track, R, np.zeros(3))
            base_cloud = [{**p, "position": R @ np.asarray(p["position"])} for p in self._original_point_cloud]
            base_obj = [
                {**o, "position": (R @ np.asarray(o["position"])).tolist()}
                for o in self._original_object_track
            ]
        else:
            R = self._z_rotation_matrix() @ self._orientation_rotation
            self.track = apply_orientation_to_track(self._original_track, R, self._orientation_origin)
            base_cloud = [
                {**p, "position": R @ (np.asarray(p["position"]) - self._orientation_origin)}
                for p in self._original_point_cloud
            ]
            base_obj = [
                {
                    **o,
                    "position": (R @ (np.asarray(o["position"]) - self._orientation_origin)).tolist(),
                }
                for o in self._original_object_track
            ]
        self.point_cloud = base_cloud
        self.object_track = base_obj

    def set_orientation(self, rotation: np.ndarray, origin: np.ndarray) -> None:
        """Apply a known rotation/origin to the original (raw-frame) data."""
        rotation = np.asarray(rotation, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64)
        self._orientation_rotation = rotation
        self._orientation_origin = origin
        self._cloud_projection_cache.clear()
        self._recompute_scene()
        self.object_position = (0.0, 0.0, 0.0)

    def to_original_frame(self, position: np.ndarray) -> np.ndarray:
        """Map a point picked in the current (possibly oriented) frame back to the original frame."""
        p = np.asarray(position, dtype=np.float64)
        Rz = self._z_rotation_matrix()
        if self._orientation_rotation is None or self._orientation_origin is None:
            return Rz.T @ p
        R = Rz @ self._orientation_rotation
        return R.T @ p + self._orientation_origin

    def from_original_frame(self, position: np.ndarray) -> np.ndarray:
        """Map a raw-frame point into the current (possibly oriented) frame."""
        p = np.asarray(position, dtype=np.float64)
        Rz = self._z_rotation_matrix()
        if self._orientation_rotation is None or self._orientation_origin is None:
            return Rz @ p
        R = Rz @ self._orientation_rotation
        return R @ (p - self._orientation_origin)

    def set_origin(self, origin_raw: np.ndarray) -> None:
        """Make a raw-frame point the world origin, keeping the current rotation."""
        origin_raw = np.asarray(origin_raw, dtype=np.float64)
        if self._orientation_rotation is not None:
            rotation = self._orientation_rotation
        else:
            rotation, _ = resolve_floor_and_wall(
                [],
                [],
                origin_raw,
                up_hint=estimate_world_up(self._original_track),
                forward_hint=estimate_world_forward(self._original_track),
            )
        self.set_orientation(rotation, origin_raw)

    def apply_scale(self, scale: float) -> None:
        """Uniformly scale the track, point cloud, and object position."""
        self._cloud_projection_cache.clear()
        if self._orientation_origin is not None:
            self._orientation_origin = np.asarray(self._orientation_origin) * scale
        self._original_track, self._original_point_cloud = apply_scale(
            self._original_track, self._original_point_cloud, scale
        )
        self._recompute_scene()
        self._original_object_position = tuple((np.array(self._original_object_position) * scale).tolist())
        self.object_position = tuple((np.array(self.object_position) * scale).tolist())

    def reset_orientation(self) -> None:
        """Reset to the original COLMAP coordinate system."""
        self._cloud_projection_cache.clear()
        self._orientation_rotation = None
        self._orientation_origin = None
        self._z_rotation = 0.0
        self._cloud_projection_cache.clear()
        self.track = self._original_track
        self.point_cloud = list(self._original_point_cloud)
        self.object_track = list(self._original_object_track)
        self.object_position = self._original_object_position

    def _draw_tracked_object(self, frame: np.ndarray, entry_index: int,
                              camera_matrix: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Overlay the solved rigid object transform as a small axes/cube."""
        if not self.object_track:
            return
        # Find the object frame matching this camera frame number.
        cam_entry = self.track[entry_index]
        cam_frame = cam_entry.get("frame_number", entry_index + 1)
        obj_entry = next((o for o in self.object_track if o.get("frame_number") == cam_frame), None)
        if obj_entry is None:
            return
        R_obj = np.asarray(obj_entry["orientation"], dtype=np.float64)
        t_obj = np.asarray(obj_entry["position"], dtype=np.float64)

        # Object-local small axes.
        length = self.object_size * 0.2
        local_pts = np.array([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length],
        ], dtype=np.float64)
        world_pts = (R_obj @ local_pts.T).T + t_obj
        projected, _ = cv2.projectPoints(world_pts, rvec, tvec, camera_matrix, None)
        projected = projected.reshape(-1, 2)
        if projected is None or len(projected) < 4:
            return
        origin = tuple(projected[0].astype(int))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X, Y, Z
        for i, color in enumerate(colors, start=1):
            pt = tuple(projected[i].astype(int))
            cv2.line(frame, origin, pt, color, 2, cv2.LINE_AA)
        cv2.circle(frame, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def render(self, entry_index: int) -> np.ndarray | None:
        """Return a BGR frame with the 3D object overlaid."""
        if entry_index < 0 or entry_index >= len(self.track):
            return None
        entry = self.track[entry_index]

        if entry_index in self._frame_cache:
            frame = self._frame_cache[entry_index].copy()
        else:
            frame_path = self.frames_dir / entry["name"]
            if not frame_path.exists():
                return None
            loaded = cv2.imread(str(frame_path))
            if loaded is None:
                return None
            self._frame_cache[entry_index] = loaded
            frame = loaded.copy()

        # Build camera matrix from the track focal length.
        focal = entry.get("focal_length", 0.0)
        width = entry.get("width", frame.shape[1])
        height = entry.get("height", frame.shape[0])
        if focal <= 0:
            return frame
        camera_matrix = build_camera_matrix(focal, width, height)

        # COLMAP pose: world -> camera rotation and translation.
        R = _quaternion_to_matrix(entry["qw"], entry["qx"], entry["qy"], entry["qz"])
        t = np.array([entry["tx"], entry["ty"], entry["tz"]], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)

        vertices = self._get_vertices()
        projected = project_points(vertices, rvec, t, camera_matrix, (width, height))
        if projected is None:
            return frame

        # Draw solved object track axes.
        self._draw_tracked_object(frame, entry_index, camera_matrix, rvec, t)

        edges = self._get_edges()
        # Draw point cloud.
        cloud_projected = self.project_point_cloud(entry_index)
        if cloud_projected is not None:
            for pt, p in zip(cloud_projected, self.point_cloud):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    color = p.get("color", (255, 255, 255))
                    bad = (self.error_threshold > 0 and p.get("error", 0.0) > self.error_threshold) or \
                          (self.min_track_len > 0 and len(p.get("track", [])) < self.min_track_len)
                    if bad:
                        color = (0, 0, 255)  # red = flagged bad
                    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw object.
        if self.object_type == "axes":
            for i, j, color in edges:
                pt1 = tuple(projected[i].astype(int))
                pt2 = tuple(projected[j].astype(int))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
        elif self.object_type == "grid":
            for i, j in edges:
                pt1 = tuple(projected[i].astype(int))
                pt2 = tuple(projected[j].astype(int))
                cv2.line(frame, pt1, pt2, (0, 200, 200), 1, cv2.LINE_AA)
        else:
            for i, j in edges:
                pt1 = tuple(projected[i].astype(int))
                pt2 = tuple(projected[j].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        return frame
