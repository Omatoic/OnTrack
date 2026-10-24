"""Model-free rigid object tracking with primitive proxies.

Uses the solved camera track, an object mask video, and learned features to
estimate a rigid 6DoF object transform on every frame. The object is treated as
a stationary cloud of points in its own coordinate system; the per-frame camera
pose relative to the object is solved, then composed with the known world camera
pose to recover the object's world-space animation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from core.learned_features import LearnedFeatureEngine
from core.parser import build_camera_track, parse_cameras_txt, parse_images_txt, quaternion_to_matrix


@dataclass
class ObjectTrackFrame:
    frame_number: int
    name: str
    position: np.ndarray          # world-space object origin
    orientation: np.ndarray       # world-space rotation matrix (3,3)
    scale: float                  # estimated metric scale
    confidence: float
    inferred: bool = False        # true if interpolated


class PrimitiveGeometry:
    """Distance and sample functions for basic primitive proxies."""

    def __init__(self, kind: str, size: tuple[float, float, float]):
        self.kind = kind
        self.size = size  # (x, y, z)

    def sample_surface_points(self, count: int = 64) -> np.ndarray:
        """Return 3D points on the primitive surface in object-local space."""
        if self.kind == "sphere":
            r = self.size[0] / 2.0
            phi = np.linspace(0, math.pi, int(math.sqrt(count)))
            theta = np.linspace(0, 2 * math.pi, int(math.sqrt(count)))
            pts = []
            for p in phi:
                for t in theta:
                    pts.append([
                        r * math.sin(p) * math.cos(t),
                        r * math.sin(p) * math.sin(t),
                        r * math.cos(p),
                    ])
            return np.asarray(pts, dtype=np.float32)
        if self.kind == "box":
            w, h, d = self.size
            # corners + face centers + edge centers
            pts = []
            for x in (-w / 2, 0, w / 2):
                for y in (-h / 2, 0, h / 2):
                    for z in (-d / 2, 0, d / 2):
                        pts.append([x, y, z])
            return np.asarray(pts, dtype=np.float32)
        if self.kind == "cylinder":
            r = self.size[0] / 2.0
            h = self.size[1]
            pts = []
            for z in (-h / 2, 0, h / 2):
                for a in np.linspace(0, 2 * math.pi, count // 3, endpoint=False):
                    pts.append([r * math.cos(a), r * math.sin(a), z])
            return np.asarray(pts, dtype=np.float32)
        if self.kind == "capsule":
            r = self.size[0] / 2.0
            h = max(0.0, self.size[1] - 2 * r)
            pts = []
            for z in (-h / 2 - r, -h / 2, 0, h / 2, h / 2 + r):
                for a in np.linspace(0, 2 * math.pi, count // 5, endpoint=False):
                    pts.append([r * math.cos(a), r * math.sin(a), z])
            return np.asarray(pts, dtype=np.float32)
        if self.kind == "plane":
            w, h, _ = self.size
            return np.array([
                [-w / 2, -h / 2, 0], [w / 2, -h / 2, 0],
                [w / 2, h / 2, 0], [-w / 2, h / 2, 0],
                [0, 0, 0],
            ], dtype=np.float32)
        raise ValueError(f"Unknown primitive: {self.kind}")

    def surface_distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed-ish distance from points to the primitive surface."""
        if self.kind == "sphere":
            r = self.size[0] / 2.0
            return np.abs(np.linalg.norm(pts, axis=1) - r)
        if self.kind == "box":
            w, h, d = self.size
            dx = np.maximum(np.abs(pts[:, 0]) - w / 2, 0)
            dy = np.maximum(np.abs(pts[:, 1]) - h / 2, 0)
            dz = np.maximum(np.abs(pts[:, 2]) - d / 2, 0)
            return np.sqrt(dx * dx + dy * dy + dz * dz)
        if self.kind == "cylinder":
            r = self.size[0] / 2.0
            h = self.size[1]
            radial = np.abs(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) - r)
            axial = np.maximum(np.abs(pts[:, 2]) - h / 2, 0)
            return np.sqrt(radial * radial + axial * axial)
        if self.kind == "capsule":
            r = self.size[0] / 2.0
            h = max(0.0, self.size[1] - 2 * r)
            radial = np.abs(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) - r)
            axial = np.maximum(np.abs(pts[:, 2]) - h / 2 - r, 0)
            return np.sqrt(radial * radial + axial * axial)
        if self.kind == "plane":
            return np.abs(pts[:, 2])
        raise ValueError(f"Unknown primitive: {self.kind}")


def _extract_object_masks(ffmpeg: Path, mask_video: Path, output_dir: Path,
                          max_size: int | None, frame_skip: int) -> list[Path]:
    """Extract object masks to white-on-black PNGs matching frame names."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_pattern = output_dir / "frame_%06d.png"

    filters = []
    if frame_skip > 0:
        filters.append(f"select='not(mod(n,{frame_skip + 1}))'")
    filters.append("format=gray")
    filters.append("lut=y='if(gt(val,127),255,0)'")
    if max_size:
        filters.append(f"scale='min({max_size},iw)':-1")

    cmd = [
        str(ffmpeg), "-y", "-loglevel", "error", "-stats",
        "-i", str(mask_video),
        "-vf", ",".join(filters),
        str(mask_pattern),
    ]
    import subprocess
    subprocess.run(cmd, check=True)
    return sorted(output_dir.glob("frame_*.png"))


def _build_tracks(matches: list[tuple[int, int, np.ndarray]]) -> dict[int, dict[int, int]]:
    """Merge pairwise matches into persistent feature tracks.

    Returns: track_id -> {frame_index: keypoint_index}
    """
    union: dict[int, int] = {}

    def find(x: int) -> int:
        while union.get(x, x) != x:
            union[x] = union.get(union[x], union[x])
            x = union[x]
        return x

    def union_sets(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            union[rx] = ry

    # Each observation is encoded as frame_index << 20 | kp_index.
    SHIFT = 20
    for i, j, m in matches:
        for a, b in m:
            obs_a = (i << SHIFT) | int(a)
            obs_b = (j << SHIFT) | int(b)
            union_sets(obs_a, obs_b)

    groups: dict[int, dict[int, int]] = {}
    for obs in union:
        root = find(obs)
        frame = obs >> SHIFT
        kp = obs & ((1 << SHIFT) - 1)
        groups.setdefault(root, {})[frame] = kp

    # Re-index tracks sequentially.
    return {idx: frames for idx, frames in enumerate(groups.values())}


def _triangulate_point(observations: list[tuple[int, np.ndarray]],
                         camera_poses: list[np.ndarray],
                         K: np.ndarray) -> np.ndarray | None:
    """Triangulate a 3D point from 2D observations using DLT."""
    if len(observations) < 2:
        return None
    A = []
    for frame_idx, uv in observations:
        P = camera_poses[frame_idx]  # 3x4 projection matrix
        x, y = uv
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    if not A:
        return None
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X[:3] / X[3]
    return X.astype(np.float32)


def _project(K: np.ndarray, Rt: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project object-space points to image coordinates using camera matrix."""
    P = K @ Rt[:, :3]
    t = K @ Rt[:, 3]
    proj = (P @ pts.T).T + t
    z = proj[:, 2:3]
    z[z < 1e-6] = 1e-6
    return proj[:, :2] / z


def _compose_camera_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 3x4 projection matrix [R|t]."""
    return np.hstack([R, t.reshape(3, 1)])


class ObjectTracker:
    """First-version rigid object tracker using learned features and primitive prior."""

    # Object tracking can use fewer keypoints and a tighter window than camera tracking.
    DEFAULT_OBJ_MAX_KEYPOINTS = 768
    DEFAULT_OBJ_MAX_SIZE = 1024
    DEFAULT_OBJ_OVERLAP = 6

    def __init__(self, config, workspace, log=print, progress=None):
        self.cfg = config
        self.ws = workspace
        self._log = log
        self._progress = progress
        self.engine = LearnedFeatureEngine(
            max_keypoints=self.DEFAULT_OBJ_MAX_KEYPOINTS,
            max_size=self.DEFAULT_OBJ_MAX_SIZE,
            use_gpu=True,
            log=log,
        )

    def _emit_progress(self, percent: int, stage: str) -> None:
        if self._progress is not None:
            self._progress.emit(percent, stage)

    def _check_cancelled(self) -> None:
        if getattr(self.cfg, "_cancelled", False):
            raise Exception("Cancelled by user.")

    def run(self) -> list[ObjectTrackFrame]:
        frames = sorted(self.ws.frames.glob("*.jpg"))
        if len(frames) < 2:
            raise Exception("Need at least 2 frames for object tracking.")

        # Load camera solve.
        images_path = self.ws.sparse_model_dir() / "images.txt"
        cameras_path = self.ws.sparse_model_dir() / "cameras.txt"
        if not images_path.exists() or not cameras_path.exists():
            raise Exception("Camera solve not found. Run camera tracking first.")

        camera_track = build_camera_track(images_path, cameras_path, target="raw")
        cameras = parse_cameras_txt(cameras_path)

        # Build per-frame camera projection matrices.
        Ks = []
        world_Rts = []
        for entry in camera_track:
            R = np.asarray(entry["rotation_matrix"], dtype=np.float64)
            C = np.asarray(entry["position"], dtype=np.float64)
            t = -R @ C
            world_Rts.append(_compose_camera_matrix(R, t))

            cam = cameras.get(entry.get("camera_id", 1), {})
            focal = cam.get("focal_length", 1.0)
            w = cam.get("width", 1920)
            h = cam.get("height", 1080)
            cx = cam.get("params", [focal, focal, w / 2, h / 2])[2] if "params" in cam else w / 2
            cy = cam.get("params", [focal, focal, w / 2, h / 2])[3] if "params" in cam else h / 2
            Ks.append(np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64))

        # Extract object masks if provided as a video, or discover pre-rendered masks.
        mask_paths = sorted(self.ws.object_masks.glob("frame_*.png"))
        if self.cfg.object_mask_video is not None and not mask_paths:
            self._log("Extracting object masks...")
            mask_paths = _extract_object_masks(
                self.cfg.ffmpeg, self.cfg.object_mask_video,
                self.ws.object_masks, self.cfg.max_size, self.cfg.frame_skip,
            )

        if len(mask_paths) != len(frames):
            raise Exception(
                f"Object mask count ({len(mask_paths)}) does not match frame count ({len(frames)})."
            )

        # Extract features inside object masks.
        self._log("Extracting object features...")
        feats = []
        for i, frame in enumerate(frames):
            self._check_cancelled()
            f = self.engine.extract(frame, mask_path=mask_paths[i])
            feats.append(f)
            if i % 10 == 0 or i == len(frames) - 1:
                self._log(f"  object features {i + 1}/{len(frames)}: {len(f.kpts)} pts")
                self._emit_progress(10 + int(25 * (i + 1) / len(frames)), "Object features")

        # Match across sequential pairs (always sequential for speed).
        n = len(frames)
        overlap = min(self.cfg.learned_overlap or self.DEFAULT_OBJ_OVERLAP, self.DEFAULT_OBJ_OVERLAP)
        pair_count = n * overlap - (overlap * (overlap + 1)) // 2
        matches = []
        done = 0
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + overlap)):
                self._check_cancelled()
                m = self.engine.match(feats[i], feats[j], min_score=0.0)
                done += 1
                if len(m) >= 4:
                    matches.append((i, j, m))
                if done % 20 == 0 or done == pair_count:
                    self._emit_progress(35 + int(25 * done / pair_count), "Object matching")

        self._log(f"Built {len(matches)} pairwise matches.")

        # Build persistent tracks.
        tracks = _build_tracks(matches)
        self._log(f"Built {len(tracks)} persistent object feature tracks.")

        # Triangulate object-space points using camera solve.
        object_points: dict[int, np.ndarray] = {}
        track_obs: dict[int, list[tuple[int, np.ndarray]]] = {}
        for tid, frames_kp in tracks.items():
            obs = []
            for frame_idx, kp_idx in frames_kp.items():
                if frame_idx < len(camera_track):
                    uv = feats[frame_idx].kpts[kp_idx]
                    obs.append((frame_idx, uv))
            if len(obs) >= 2:
                X = _triangulate_point(obs, world_Rts, Ks[0])
                if X is not None and np.isfinite(X).all():
                    object_points[tid] = X
                    track_obs[tid] = obs

        self._log(f"Triangulated {len(object_points)} object-space points.")
        if len(object_points) < 3:
            raise Exception("Not enough object tracks to solve. Try a larger overlap or better mask.")

        # Solve per-frame object pose with PnP.
        primitive = PrimitiveGeometry(self.cfg.object_primitive, self.cfg.object_size)
        object_poses: list[ObjectTrackFrame | None] = [None] * n

        for frame_idx in range(n):
            self._check_cancelled()
            if frame_idx % 10 == 0:
                self._emit_progress(60 + int(15 * frame_idx / n), "Object PnP")
            pts_3d = []
            pts_2d = []
            for tid, obs in track_obs.items():
                for fi, uv in obs:
                    if fi == frame_idx:
                        pts_3d.append(object_points[tid])
                        pts_2d.append(uv)
                        break

            if len(pts_3d) < 4:
                continue

            pts_3d = np.asarray(pts_3d, dtype=np.float32)
            pts_2d = np.asarray(pts_2d, dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, Ks[frame_idx].astype(np.float32), None,
                reprojectionError=8.0, confidence=0.99, iterationsCount=500,
            )
            if not success or inliers is None or len(inliers) < 4:
                continue

            R_obj, _ = cv2.Rodrigues(rvec)
            R_obj = R_obj.astype(np.float64)
            t_obj = tvec.reshape(3)

            # Object-to-world: compose with camera-to-world.
            cam_Rt = world_Rts[frame_idx]
            cam_R = cam_Rt[:, :3]
            cam_t = cam_Rt[:, 3]

            # Object point in world = cam_R @ (R_obj @ P + t_obj) + cam_t
            # So object origin in world = cam_R @ t_obj + cam_t
            # Object world rotation = cam_R @ R_obj
            world_R = cam_R @ R_obj
            world_t = cam_R @ t_obj + cam_t

            object_poses[frame_idx] = ObjectTrackFrame(
                frame_number=camera_track[frame_idx]["frame_number"],
                name=camera_track[frame_idx]["name"],
                position=world_t.astype(np.float32),
                orientation=world_R.astype(np.float32),
                scale=1.0,
                confidence=min(1.0, len(inliers) / max(4, len(pts_3d) // 2)),
            )

        solved_count = sum(1 for p in object_poses if p is not None)
        self._log(f"Solved object pose for {solved_count}/{n} frames.")

        if solved_count == 0:
            raise Exception("Could not solve any object pose.")

        # Interpolate missing frames and smooth.
        self._emit_progress(75, "Object smoothing")
        object_poses = self._interpolate_and_smooth(object_poses, camera_track)

        # Optional: refine with temporal bundle adjustment.
        self._emit_progress(85, "Object refine")
        object_poses = self._refine_with_ba(
            object_poses, track_obs, object_points, primitive, Ks, world_Rts,
        )

        self._emit_progress(95, "Object track complete")
        return object_poses

    def _interpolate_and_smooth(self, poses: list[ObjectTrackFrame | None],
                                   camera_track: list[dict[str, Any]]) -> list[ObjectTrackFrame]:
        n = len(poses)
        out: list[ObjectTrackFrame] = []

        for i in range(n):
            p = poses[i]
            if p is not None:
                out.append(p)
                continue

            # Find nearest solved frames.
            left = next((j for j in range(i - 1, -1, -1) if poses[j] is not None), None)
            right = next((j for j in range(i + 1, n) if poses[j] is not None), None)

            if left is None and right is None:
                raise Exception("No object pose could be solved; tracking failed.")

            if left is None:
                ref = poses[right]
            elif right is None:
                ref = poses[left]
            else:
                lp = poses[left]
                rp = poses[right]
                alpha = (i - left) / (right - left)
                pos = (1 - alpha) * lp.position + alpha * rp.position
                # Spherical linear interpolation for rotation.
                q0 = self._matrix_to_quaternion(lp.orientation)
                q1 = self._matrix_to_quaternion(rp.orientation)
                q = self._slerp(q0, q1, alpha)
                R = self._quaternion_to_matrix(q)
                ref = ObjectTrackFrame(
                    frame_number=camera_track[i]["frame_number"],
                    name=camera_track[i]["name"],
                    position=pos.astype(np.float32),
                    orientation=R.astype(np.float32),
                    scale=(lp.scale + rp.scale) / 2,
                    confidence=min(lp.confidence, rp.confidence) * 0.5,
                    inferred=True,
                )
            out.append(ref)

        # Light temporal smoothing on position and rotation.
        window = 3
        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            weights = [out[j].confidence for j in range(start, end)]
            total = sum(weights) or 1.0
            smooth_pos = sum(out[j].position * (weights[j - start] / total) for j in range(start, end))
            # Average quaternions.
            q_sum = np.zeros(4)
            for j in range(start, end):
                q_sum += self._matrix_to_quaternion(out[j].orientation) * weights[j - start]
            q_sum /= np.linalg.norm(q_sum) + 1e-12
            out[i] = ObjectTrackFrame(
                frame_number=out[i].frame_number,
                name=out[i].name,
                position=smooth_pos.astype(np.float32),
                orientation=self._quaternion_to_matrix(q_sum).astype(np.float32),
                scale=out[i].scale,
                confidence=out[i].confidence,
                inferred=out[i].inferred,
            )

        return out

    def _refine_with_ba(self, poses: list[ObjectTrackFrame],
                        track_obs: dict[int, list[tuple[int, np.ndarray]]],
                        object_points: dict[int, np.ndarray],
                        primitive: PrimitiveGeometry,
                        Ks: list[np.ndarray],
                        world_Rts: list[np.ndarray]) -> list[ObjectTrackFrame]:
        """Light temporal refinement minimizing reprojection error."""
        # Per-frame parameters: [tx, ty, tz, qx, qy, qz] (qw derived).
        n = len(poses)
        x0 = []
        for p in poses:
            q = self._matrix_to_quaternion(p.orientation)
            x0.extend([p.position[0], p.position[1], p.position[2], q[1], q[2], q[3]])
        x0 = np.asarray(x0, dtype=np.float64)

        def residuals(x: np.ndarray) -> np.ndarray:
            res = []
            for frame_idx in range(n):
                base = frame_idx * 6
                t_obj = x[base:base + 3]
                qxyz = x[base + 3:base + 6]
                qw = math.sqrt(max(0.0, 1.0 - np.sum(qxyz ** 2)))
                q = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])
                R_obj = self._quaternion_to_matrix(q)

                cam_Rt = world_Rts[frame_idx]
                cam_R = cam_Rt[:, :3]
                cam_t = cam_Rt[:, 3]
                world_R = cam_R @ R_obj
                world_t = cam_R @ t_obj + cam_t

                K = Ks[frame_idx]
                for tid, obs in track_obs.items():
                    if tid not in object_points:
                        continue
                    for fi, uv in obs:
                        if fi != frame_idx:
                            continue
                        P_obj = object_points[tid]
                        P_world = world_R @ P_obj + world_t
                        P_cam = cam_R.T @ (P_world - cam_t)
                        if P_cam[2] <= 1e-6:
                            continue
                        p2d = K @ (P_cam / P_cam[2])
                        p2d = p2d[:2]
                        err = (p2d - uv) * poses[frame_idx].confidence
                        res.extend(err.tolist())
                        break
            return np.asarray(res, dtype=np.float64)

        try:
            result = least_squares(residuals, x0, method="lm", max_nfev=100)
            x = result.x
            for frame_idx in range(n):
                base = frame_idx * 6
                t_obj = x[base:base + 3]
                qxyz = x[base + 3:base + 6]
                qw = math.sqrt(max(0.0, 1.0 - np.sum(qxyz ** 2)))
                q = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])
                R_obj = self._quaternion_to_matrix(q)

                cam_Rt = world_Rts[frame_idx]
                cam_R = cam_Rt[:, :3]
                cam_t = cam_Rt[:, 3]
                poses[frame_idx] = ObjectTrackFrame(
                    frame_number=poses[frame_idx].frame_number,
                    name=poses[frame_idx].name,
                    position=(cam_R @ t_obj + cam_t).astype(np.float32),
                    orientation=(cam_R @ R_obj).astype(np.float32),
                    scale=poses[frame_idx].scale,
                    confidence=poses[frame_idx].confidence,
                    inferred=poses[frame_idx].inferred,
                )
            self._log(f"Refined object poses: {result.cost:.2f} final cost.")
        except Exception as exc:
            self._log(f"Object BA refinement skipped: {exc}")

        return poses

    @staticmethod
    def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
        if dot < 0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return (q0 + t * (q1 - q0)) / (np.linalg.norm(q0 + t * (q1 - q0)) + 1e-12)
        theta0 = math.acos(dot)
        theta = theta0 * t
        sin_theta = math.sin(theta)
        sin_theta0 = math.sin(theta0)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta0
        s1 = sin_theta / sin_theta0
        return q0 * s0 + q1 * s1


def save_object_track(path: Path, track: list[ObjectTrackFrame]) -> None:
    """Serialize object track to JSON for downstream exporters."""
    data = []
    for p in track:
        data.append({
            "frame_number": p.frame_number,
            "name": p.name,
            "position": p.position.tolist(),
            "orientation": p.orientation.tolist(),
            "scale": p.scale,
            "confidence": p.confidence,
            "inferred": p.inferred,
        })
    path.write_text(json.dumps(data, indent=2))
