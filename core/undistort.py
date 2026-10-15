"""Lens undistort/redistort integration via Discorpy.

Calibration produces a lens profile (center of distortion + polynomial
coefficients). The same profile is used to:
  - undistort footage before tracking (backward model, black = removed)
  - redistort rendered/comped footage afterward (backward-form coefficients
    derived from the forward model), so FX elements pick up the lens warp.

Discorpy is vendored, not installed - we add its folder to sys.path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.paths import project_root
from core.video_info import probe_video


def _ensure_discorpy() -> Path:
    root = project_root() / "EXTRA" / "Undistort" / "discorpy-1.7.0"
    if not root.exists():
        raise FileNotFoundError(f"Discorpy not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _mods():
    _ensure_discorpy()
    import discorpy.prep.preprocessing as prep
    import discorpy.prep.linepattern as lprep
    import discorpy.proc.processing as proc
    import discorpy.post.postprocessing as post
    return prep, lprep, proc, post


@dataclass
class LensProfile:
    """Persisted lens distortion profile (JSON)."""
    width: int
    height: int
    xcenter: float
    ycenter: float
    undistort_coefs: list[float]   # backward model: distorted -> undistorted
    redistort_coefs: list[float]   # backward form of forward model: undistorted -> distorted

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: Path) -> "LensProfile":
        return LensProfile(**json.loads(path.read_text()))


def _load_gray(path: Path) -> np.ndarray:
    import discorpy.losa.loadersaver as losa  # noqa - resolves via _mods
    _mods()
    return losa.load_image(str(path))


def calibrate(image_path: Path, pattern: str = "checkerboard",
              num_coef: int = 5, log=print) -> LensProfile:
    """Calibrate a lens profile from a single calibration image.

    pattern: "checkerboard" or "dots"
    """
    prep, lprep, proc, post = _mods()
    import discorpy.prep.linepattern as _lp  # noqa

    mat0 = _load_gray(image_path)
    (height, width) = mat0.shape
    log(f"Calibration image: {width}x{height}")

    if pattern == "checkerboard":
        # demo_06 workflow: chessboard -> line pattern -> cross points
        mat1 = lprep.convert_chessboard_to_linepattern(mat0)
        slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(
            mat1, radius=15, sensitive=0.5)
        slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(
            mat1, radius=15, sensitive=0.5)
        slope_hor = float(np.asarray(slope_hor).ravel()[0])
        slope_ver = float(np.asarray(slope_ver).ravel()[0])
        dist_hor = float(np.asarray(dist_hor).ravel()[0])
        dist_ver = float(np.asarray(dist_ver).ravel()[0])
        log(f"hor slope {slope_hor:.4f} dist {dist_hor:.1f} | "
            f"ver slope {slope_ver:.4f} dist {dist_ver:.1f}")
        list_points_hor = lprep.get_cross_points_hor_lines(
            mat1, slope_ver, dist_ver, ratio=0.3, norm=True, offset=450,
            bgr="bright", radius=15, sensitive=0.5, denoise=True, subpixel=True)
        list_points_ver = lprep.get_cross_points_ver_lines(
            mat1, slope_hor, dist_hor, ratio=0.3, norm=True, offset=150,
            bgr="bright", radius=15, sensitive=0.5, denoise=True, subpixel=True)
        if len(list_points_hor) == 0 or len(list_points_ver) == 0:
            raise ValueError("No reference points detected in checkerboard.")
        list_hor_lines = prep.group_dots_hor_lines(
            list_points_hor, slope_hor, dist_hor, ratio=0.1, num_dot_miss=2,
            accepted_ratio=0.8)
        list_ver_lines = prep.group_dots_ver_lines(
            list_points_ver, slope_ver, dist_ver, ratio=0.1, num_dot_miss=2,
            accepted_ratio=0.8)
    else:
        # demo_01 workflow: dot pattern
        mat1 = prep.binarization(mat0)
        dot_size, dot_dist = prep.calc_size_distance(mat1)
        hor_slope = float(np.asarray(hor_slope).ravel()[0])
        ver_slope = float(np.asarray(ver_slope).ravel()[0])
        dot_dist = float(np.asarray(dot_dist).ravel()[0])
        mat1 = prep.select_dots_based_size(mat1, dot_size)
        mat1 = prep.select_dots_based_ratio(mat1)
        hor_slope = prep.calc_hor_slope(mat1)
        ver_slope = prep.calc_ver_slope(mat1)
        log(f"hor slope {hor_slope:.4f} | ver slope {ver_slope:.4f} | "
            f"dot dist {dot_dist:.1f}")
        list_hor_lines = prep.group_dots_hor_lines(mat1, hor_slope, dot_dist)
        list_ver_lines = prep.group_dots_ver_lines(mat1, ver_slope, dot_dist)

    list_hor_lines = prep.remove_residual_dots_hor(
        list_hor_lines, slope_hor if pattern == "checkerboard" else hor_slope, 2.0)
    list_ver_lines = prep.remove_residual_dots_ver(
        list_ver_lines, slope_ver if pattern == "checkerboard" else ver_slope, 2.0)
    n_hor = sum(len(l) for l in list_hor_lines)
    n_ver = sum(len(l) for l in list_ver_lines)
    log(f"Grouped {len(list_hor_lines)} hor / {len(list_ver_lines)} ver lines "
        f"({n_hor}+{n_ver} points)")

    # Perspective correction pass (safe for near-frontal targets too)
    list_hor_lines, list_ver_lines = proc.regenerate_grid_points_parabola(
        list_hor_lines, list_ver_lines, perspective=True)

    xcenter, ycenter = proc.find_cod_coarse(list_hor_lines, list_ver_lines)
    log(f"Center of distortion: ({xcenter:.1f}, {ycenter:.1f})")

    bw = proc.calc_coef_backward(list_hor_lines, list_ver_lines,
                                 xcenter, ycenter, num_coef)
    # Backward-form coefficients of the forward model: applying the backward
    # unwarp with these to an undistorted image re-applies the distortion.
    _, bwff = proc.calc_coef_backward_from_forward(
        list_hor_lines, list_ver_lines, xcenter, ycenter, num_coef)
    log(f"Undistort coefs: {np.round(bw, 8).tolist()}")

    return LensProfile(
        width=int(width), height=int(height),
        xcenter=float(xcenter), ycenter=float(ycenter),
        undistort_coefs=[float(v) for v in bw],
        redistort_coefs=[float(v) for v in bwff],
    )


def scaled_coefs(coefs: list[float], scale: float) -> np.ndarray:
    """Rescale polynomial coefficients for a different resolution.

    If the profile was calibrated at resolution W and we warp at s*W,
    distances grow by s, so coef_i must be divided by s**i to keep
    F(r') = F(r'/s) equivalent.
    """
    exps = np.arange(len(coefs), dtype=np.float64)
    return np.asarray(coefs, dtype=np.float64) / np.power(scale, exps)


def resolution_scale(frame_shape, profile: LensProfile):
    """Return (scale, mismatched, nonuniform) vs the calibration resolution."""
    h, w = frame_shape[:2]
    sx = w / profile.width
    sy = h / profile.height
    return sx, abs(sx - 1.0) > 1e-6, abs(sx / sy - 1.0) > 1e-6


def _warp_frame(frame: np.ndarray, profile: LensProfile,
                coefs: list[float]) -> np.ndarray:
    """Warp an RGB uint8 frame with unwarp_image_backward per channel."""
    _, _, _, post = _mods()
    s, mismatched, _ = resolution_scale(frame.shape, profile)
    xc = profile.xcenter * frame.shape[1] / profile.width
    yc = profile.ycenter * frame.shape[0] / profile.height
    c = scaled_coefs(coefs, s) if mismatched else np.asarray(coefs)
    out = np.empty_like(frame)
    if frame.ndim == 3:
        for i in range(frame.shape[-1]):
            out[:, :, i] = post.unwarp_image_backward(
                frame[:, :, i], xc, yc, c)
    else:
        out = post.unwarp_image_backward(frame, xc, yc, c)
    return np.clip(out, 0, 255).astype(np.uint8)


def undistort_frame(frame: np.ndarray, profile: LensProfile) -> np.ndarray:
    return _warp_frame(frame, profile, profile.undistort_coefs)


def redistort_frame(frame: np.ndarray, profile: LensProfile) -> np.ndarray:
    return _warp_frame(frame, profile, profile.redistort_coefs)


def stmap(width: int, height: int, profile: LensProfile,
          mode: str = "undistort") -> np.ndarray:
    """Return an HxWx2 float32 ST map (R=u, G=v, normalized 0-1, V flipped).

    Encodes the same backward lookup unwarp_image_backward performs, so an
    STMap node in Nuke/AE reproduces the identical warp.
    """
    coefs = (profile.undistort_coefs if mode == "undistort"
             else profile.redistort_coefs)
    s = width / profile.width
    xc = profile.xcenter * s
    yc = profile.ycenter * (height / profile.height)
    c = scaled_coefs(coefs, s)
    xu_list = np.arange(width, dtype=np.float64) - xc
    yu_list = np.arange(height, dtype=np.float64) - yc
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)
    fact_mat = np.sum(np.asarray(
        [f * ru_mat ** i for i, f in enumerate(c)]), axis=0)
    xd = np.clip(xc + fact_mat * xu_mat, 0, width - 1)
    yd = np.clip(yc + fact_mat * yu_mat, 0, height - 1)
    u = xd / (width - 1)
    v = 1.0 - yd / (height - 1)  # STMap convention: V origin at bottom
    out = np.zeros((height, width, 3), dtype=np.float32)
    out[:, :, 0] = u
    out[:, :, 1] = v
    return out


def save_stmap_exr(path: Path, st: np.ndarray) -> None:
    """Write an HxWx3 float32 ST map to EXR (R=u, G=v, B=0)."""
    import OpenEXR
    path.parent.mkdir(parents=True, exist_ok=True)
    st = np.ascontiguousarray(st.astype(np.float32))
    header = {"compression": OpenEXR.ZIP_COMPRESSION,
              "type": OpenEXR.scanlineimage}
    channels = {"R": st[:, :, 0], "G": st[:, :, 1],
                "B": np.zeros(st.shape[:2], np.float32)}
    OpenEXR.File(header, channels).write(str(path))


# ---------------------------------------------------------------------------
# Video processing worker
# ---------------------------------------------------------------------------


class VideoWarpWorker(QThread):
    """Undistort or redistort every frame of a video, preserving audio."""

    log = Signal(str)
    progress = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(self, input_video: Path, output_video: Path,
                 profile: LensProfile, mode: str, ffmpeg: Path,
                 parent=None):
        super().__init__(parent)
        self.input_video = input_video
        self.output_video = output_video
        self.profile = profile
        self.mode = mode  # "undistort" | "redistort"
        self.ffmpeg = ffmpeg
        self._cancelled = False
        self._warned_mismatch = False

    def cancel(self) -> None:
        self._cancelled = True

    def _log(self, msg: str) -> None:
        self.log.emit(msg)

    def run(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="warp_"))
        frames_in = tmp / "in"
        frames_out = tmp / "out"
        frames_in.mkdir()
        frames_out.mkdir()
        try:
            coefs = (self.profile.undistort_coefs if self.mode == "undistort"
                     else self.profile.redistort_coefs)
            info = probe_video(self.input_video,
                               self.ffmpeg.parent / "ffprobe.exe")

            # 1) extract frames
            self._log("Extracting frames...")
            cmd = [str(self.ffmpeg), "-loglevel", "error", "-i",
                   str(self.input_video),
                   str(frames_in / "f_%06d.png")]
            subprocess.run(cmd, check=True)
            files = sorted(frames_in.glob("*.png"))
            if not files:
                raise RuntimeError("FFmpeg produced no frames")

            # 2) warp each frame
            _, _, _, post = _mods()
            for i, f in enumerate(files, 1):
                if self._cancelled:
                    raise RuntimeError("Cancelled")
                img = np.asarray(iio.imread(f))
                warped = self._warp(img, coefs)
                iio.imwrite(frames_out / f.name, warped)
                self.progress.emit(
                    int(90 * i / len(files)),
                    f"{self.mode} {i}/{len(files)}")

            # 3) re-encode, copying the original audio/subtitles if present
            self._log("Encoding output video...")
            fps = info.get("fps") or 24.0
            cmd = [
                str(self.ffmpeg), "-loglevel", "error", "-y",
                "-framerate", str(fps),
                "-i", str(frames_out / "f_%06d.png"),
                "-i", str(self.input_video),
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "16", "-c:a", "copy", "-shortest",
                str(self.output_video),
            ]
            subprocess.run(cmd, check=True)
            self.progress.emit(100, "Done")
            self.finished.emit(True, f"Saved: {self.output_video}")
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _warp(self, img: np.ndarray, coefs: list[float]) -> np.ndarray:
        _, _, _, post = _mods()
        s, mismatched, nonuniform = resolution_scale(img.shape, self.profile)
        if mismatched and not self._warned_mismatch:
            self._warned_mismatch = True
            warn = (f"WARNING: frame {img.shape[1]}x{img.shape[0]} vs profile "
                    f"{self.profile.width}x{self.profile.height} — "
                    f"coefficients scaled by {s:.4f}")
            if nonuniform:
                warn += " (aspect ratio differs — result may be inaccurate)"
            self._log(warn)
        xc = self.profile.xcenter * img.shape[1] / self.profile.width
        yc = self.profile.ycenter * img.shape[0] / self.profile.height
        c = scaled_coefs(coefs, s) if mismatched else np.asarray(coefs)
        if img.ndim == 3 and img.shape[-1] >= 3:
            out = np.empty(img.shape[:2] + (3,), dtype=np.uint8)
            for ch in range(3):
                out[:, :, ch] = np.clip(
                    post.unwarp_image_backward(
                        img[:, :, ch].astype(np.float64), xc, yc, c),
                    0, 255).astype(np.uint8)
            return out
        return np.clip(
            post.unwarp_image_backward(img.astype(np.float64), xc, yc, c),
            0, 255).astype(np.uint8)
