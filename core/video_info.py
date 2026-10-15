from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def probe_video(video_path: Path, ffprobe_path: Path | None = None) -> dict[str, Any]:
    """
    Probe a video file and return metadata including width, height, fps,
    duration, and estimated frame count.
    """
    if ffprobe_path is None:
        ffprobe_path = Path("ffprobe")

    cmd = [
        str(ffprobe_path),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return {}

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )

    width = video_stream.get("width")
    height = video_stream.get("height")
    duration = video_stream.get("duration") or data.get("format", {}).get("duration")
    nb_frames = video_stream.get("nb_frames")

    # Parse frame rate (e.g. "24000/1001").
    fps = None
    r_frame_rate = video_stream.get("r_frame_rate")
    if r_frame_rate and "/" in r_frame_rate:
        try:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = None

    frame_count = None
    if nb_frames and str(nb_frames).isdigit():
        frame_count = int(nb_frames)
    elif duration and fps:
        try:
            frame_count = int(float(duration) * fps)
        except ValueError:
            pass

    return {
        "path": str(video_path),
        "filename": video_path.name,
        "width": width,
        "height": height,
        "duration": float(duration) if duration else None,
        "fps": fps,
        "frame_count": frame_count,
    }


def extract_thumbnail(
    video_path: Path,
    ffmpeg_path: Path,
    output_path: Path | None = None,
    time_sec: float = 0.0,
    width: int = 320,
) -> Path | None:
    """
    Extract a single thumbnail frame from a video using FFmpeg.
    Returns the path to the saved thumbnail, or None on failure.
    """
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / f"glomap_thumb_{video_path.stem}.jpg"

    cmd = [
        str(ffmpeg_path),
        "-ss", str(time_sec),
        "-i", str(video_path),
        "-vf", f"scale={width}:-1",
        "-vframes", "1",
        "-q:v", "2",
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
    except subprocess.CalledProcessError:
        pass
    return None
