"""Bridge between OnTrack and the vendored Sammie-Roto editor.

The editor is embedded in-process in the Roto tab with ``roto/`` as the
working directory; its session scratch folder (``roto_temp()``, by default
``%TEMP%/OnTrack/roto``) holds:

    session_settings.conf   - JSON: video path, fps, total_frames
    frames/{f:05d}.png      - decoded frames (0-based)
    masks/{f:05d}/{id}.png  - per-object binary masks (white = object)
    matting/{f:05d}/{id}.png - refined mattes (if matting was run)

OnTrack consumes the masks by merging them into a full-length white-on-black
matte video (FFV1/MKV) which the tracking pipeline already understands.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.paths import project_root, roto_temp_root

# The vendored editor reads this to place its scratch folder (see
# roto/sammie/core.py) in the OS temp dir instead of roto/temp.
os.environ.setdefault("ONTRACK_ROTO_TEMP", str(roto_temp_root()))


def roto_root() -> Path:
    return project_root() / "roto"


def roto_temp() -> Path:
    return Path(os.environ["ONTRACK_ROTO_TEMP"])


def masks_dir() -> Path:
    return roto_temp() / "masks"


def matting_dir() -> Path:
    return roto_temp() / "matting"


def session_file() -> Path:
    return roto_temp() / "session_settings.conf"


def session_info() -> dict | None:
    """Return the roto editor's current session metadata, if any."""
    try:
        data = json.loads(session_file().read_text())
    except Exception:
        return None
    if not data.get("video_file_path"):
        return None
    return data


@dataclass
class MaskObject:
    object_id: int
    name: str
    frames: list[int] = field(default_factory=list)
    refined_frames: list[int] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def _mask_file_map() -> dict[tuple[int, int], Path]:
    """Map (frame, object_id) -> mask PNG, preferring the refined matting
    version whenever it exists for that frame/object."""
    files: dict[tuple[int, int], Path] = {}
    for root in (masks_dir(), matting_dir()):  # matting second = wins
        if not root.exists():
            continue
        for png in root.glob("*/*.png"):
            try:
                files[(int(png.parent.name), int(png.stem))] = png
            except ValueError:
                continue
    return files


def list_mask_objects() -> list[MaskObject]:
    """Enumerate roto objects by scanning the mask/matting folders."""
    session = session_info() or {}
    names = session.get("object_names") or {}
    matting_root = matting_dir()

    objects: dict[int, MaskObject] = {}
    for (frame_idx, oid), png in sorted(_mask_file_map().items()):
        obj = objects.setdefault(
            oid, MaskObject(oid, str(names.get(str(oid), f"Object {oid}")))
        )
        obj.frames.append(frame_idx)
        if png.is_relative_to(matting_root):
            obj.refined_frames.append(frame_idx)
    return sorted(objects.values(), key=lambda o: o.object_id)


def build_merged_masks(
    object_ids: list[int] | None,
    total_frames: int,
    dest_dir: Path,
    progress=None,
    binary: bool = True,
) -> int:
    """Merge per-object mask PNGs into one binary matte per frame.

    ``object_ids=None`` merges every object. Missing frames get a black mask.
    Returns the number of non-empty frames written.
    """
    session = session_info() or {}
    file_map = _mask_file_map()
    dest_dir.mkdir(parents=True, exist_ok=True)

    size = None
    if file_map:
        probe = cv2.imread(str(next(iter(file_map.values()))), cv2.IMREAD_GRAYSCALE)
        if probe is not None:
            size = (probe.shape[1], probe.shape[0])
    if size is None:
        w = session.get("video_width") or 1920
        h = session.get("video_height") or 1080
        size = (int(w), int(h))

    ids = object_ids
    by_frame: dict[int, list[Path]] = {}
    for (fidx, oid), png in file_map.items():
        if ids is not None and oid not in ids:
            continue
        by_frame.setdefault(fidx, []).append(png)

    nonempty = 0
    for frame_idx in range(total_frames):
        if progress:
            progress(frame_idx, total_frames)
        matte = None
        for png in by_frame.get(frame_idx, ()):
            mask = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[1] != size[0] or mask.shape[0] != size[1]:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            if binary:
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            matte = mask if matte is None else np.maximum(matte, mask)
        if matte is None:
            matte = np.zeros((size[1], size[0]), dtype=np.uint8)
        else:
            nonempty += 1
        cv2.imwrite(str(dest_dir / f"{frame_idx:05d}.png"), matte)
    return nonempty


def encode_matte_video(
    merged_dir: Path,
    output: Path,
    ffmpeg: Path,
    fps: float,
    log=print,
) -> Path:
    """Encode merged matte PNGs as a lossless grayscale video."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg), "-y",
        "-loglevel", "error",
        "-framerate", f"{fps:g}",
        "-i", str(merged_dir / "%05d.png"),
        "-c:v", "prores_ks",
        "-profile:v", "3",
        "-pix_fmt", "yuv422p10le",
        "-movflags", "+faststart",
        str(output),
    ]
    log(f"Encoding matte video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"ffmpeg matte encode failed: {result.stderr.strip()}")
    return output
