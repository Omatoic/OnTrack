from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from core.colmap_db import inspect_database
from core.paths import WorkspacePaths, clear_workspace, make_session_workspace


@dataclass
class PipelineConfig:
    input_video: Path
    output_dir: Path | None = None
    mask_video: Path | None = None
    mask_shrink: int = 0
    ffmpeg: Path | None = None
    colmap: Path | None = None
    glomap: Path | None = None
    fps: float = 24.0
    max_size: int | None = None
    frame_skip: int = 0
    matcher_type: str = "sequential"
    use_gpu: bool = False
    max_image_size: int = 4096
    sequential_overlap: int = 15
    cleanup: bool = False


class PipelineWorker(QObject):
    log = Signal(str)
    progress = Signal(int, str)
    finished = Signal(bool, str)
    export_ready = Signal(str)

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self._cancelled = False
        self._current_proc: subprocess.Popen | None = None
        self.workspace: WorkspacePaths | None = None

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise Exception("Cancelled by user.")

    def _log(self, message: str) -> None:
        self.log.emit(message)

    def _run_command(
        self,
        name: str,
        binary: Path,
        args: list[Any],
        cwd: Path | None = None,
        progress_start: int = 0,
        progress_end: int = 100,
        progress_regex: re.Pattern | None = None,
        total_items: int | None = None,
    ) -> None:
        """Run an external command, streaming stdout to the log and progress bar."""
        self._check_cancelled()

        env = dict(os.environ)
        bin_dir = str(binary.parent)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        if cwd is None:
            cwd = binary.parent

        cmd = [str(binary)] + [str(a) for a in args]
        self._log(f"> {' '.join(cmd)}")

        self._current_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
        )

        current = progress_start
        self.progress.emit(current, name)

        for line in self._current_proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            self._log(line)

            # Optional granular progress parsing.
            if progress_regex:
                match = progress_regex.search(line)
                if match and total_items and total_items > 0:
                    try:
                        done = int(match.group(1))
                        pct = progress_start + int((done / total_items) * (progress_end - progress_start))
                        self.progress.emit(pct, name)
                        current = pct
                    except (ValueError, IndexError):
                        pass

        self._current_proc.wait()
        if self._cancelled:
            raise Exception("Cancelled by user.")
        if self._current_proc.returncode != 0:
            raise Exception(f"{name} failed with exit code {self._current_proc.returncode}")

        self.progress.emit(progress_end, name)
        self._current_proc = None

    def _probe_total_frames(self) -> int | None:
        """Use ffprobe to estimate the total number of frames."""
        ffprobe = self.config.ffmpeg.parent / "ffprobe.exe" if self.config.ffmpeg else None
        if not ffprobe or not ffprobe.exists():
            return None
        try:
            result = subprocess.run(
                [str(ffprobe), "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames", "-of",
                 "default=noprint_wrappers=1", str(self.config.input_video)],
                capture_output=True, text=True, check=False,
            )
            for line in result.stdout.splitlines():
                if line.startswith("nb_frames="):
                    value = line.split("=", 1)[1]
                    if value != "N/A":
                        return int(value)
        except Exception:
            return None
        return None

    def _extract_frames(self) -> None:
        self._log("Step 1/4: Extracting frames with FFmpeg...")
        frames_dir = self.workspace.frames
        frame_pattern = frames_dir / "frame_%06d.jpg"

        filters = []
        if self.config.max_size:
            filters.append(f"scale='min({self.config.max_size},iw)':-1")
        if self.config.frame_skip > 0:
            filters.append(f"select='not(mod(n,{self.config.frame_skip + 1}))'")

        cmd = [
            "-loglevel", "error",
            "-stats",
            "-i", str(self.config.input_video),
            "-q:v", "2",
        ]
        if filters:
            cmd += ["-vf", ",".join(filters)]

        cmd += [str(frame_pattern)]

        total_frames = self._probe_total_frames()
        frame_regex = re.compile(r"frame=\s*(\d+)", re.IGNORECASE)

        self._run_command(
            "FFmpeg",
            self.config.ffmpeg,
            cmd,
            progress_start=0,
            progress_end=25,
            progress_regex=frame_regex,
            total_items=total_frames,
        )

    def _extract_masks(self) -> None:
        """Convert the user matte video into per-frame COLMAP masks.

        The matte convention is white = ignore, black = keep. COLMAP expects
        black = ignore, so the frames are inverted here. Mask filenames are
        ``<image_name>.png`` as required by --ImageReader.mask_path.
        """
        self._log("Step 2/5: Generating tracking masks from matte...")
        masks_dir = self.workspace.masks
        mask_pattern = masks_dir / "frame_%06d.jpg.png"

        # Apply the same select/scale transforms used for the frames so masks
        # line up 1:1, then invert (white matte -> black = ignored region).
        filters = []
        if self.config.frame_skip > 0:
            filters.append(f"select='not(mod(n,{self.config.frame_skip + 1}))'")
        filters.append("negate")
        # COLMAP requires single-channel grayscale masks; an RGB PNG is not
        # applied and silently behaves like an all-white mask.
        filters.append("format=gray")
        # Optional: shrink the ignored region. Each dilation pass grows the
        # white (keep) area into the black (ignore) area by ~1px.
        filters.extend(["dilation"] * max(0, int(self.config.mask_shrink)))
        # Snap to pure 0/255: video decoding leaves matte "black" at ~4-6,
        # and COLMAP keeps any nonzero pixel, so a threshold is required.
        filters.append("lut=y='if(gt(val,127),255,0)'")
        if self.config.max_size:
            filters.append(f"scale='min({self.config.max_size},iw)':-1")

        cmd = [
            "-loglevel", "error",
            "-stats",
            "-i", str(self.config.mask_video),
            "-vf", ",".join(filters),
            str(mask_pattern),
        ]

        total_frames = self._probe_total_frames()
        frame_regex = re.compile(r"frame=\s*(\d+)", re.IGNORECASE)

        self._run_command(
            "FFmpeg Masks",
            self.config.ffmpeg,
            cmd,
            progress_start=12,
            progress_end=25,
            progress_regex=frame_regex,
            total_items=total_frames,
        )

    def _colmap_feature_extractor(self) -> None:
        self._log("Step 3/5: COLMAP feature extraction...")
        args = [
            "feature_extractor",
            "--database_path", str(self.workspace.database_file),
            "--image_path", str(self.workspace.frames),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_image_size", str(self.config.max_image_size),
        ]
        if self.config.mask_video is not None:
            args.extend(["--ImageReader.mask_path", str(self.workspace.masks)])
        if not self.config.use_gpu:
            args.extend(["--SiftExtraction.use_gpu", "0"])
        self._run_command(
            "COLMAP Feature Extractor",
            self.config.colmap,
            args,
            progress_start=25,
            progress_end=50,
        )

    def _colmap_matcher(self) -> None:
        matcher = self.config.matcher_type
        self._log(f"Step 4/5: COLMAP {matcher} matching...")
        args = [
            f"{matcher}_matcher",
            "--database_path", str(self.workspace.database_file),
        ]
        if matcher == "sequential":
            args.extend(["--SequentialMatching.overlap", str(self.config.sequential_overlap)])
        if not self.config.use_gpu:
            args.extend(["--SiftMatching.use_gpu", "0"])
        self._run_command(
            "COLMAP Matcher",
            self.config.colmap,
            args,
            progress_start=50,
            progress_end=75,
        )

        # Verify that the database actually contains matches.
        counts = inspect_database(self.workspace.database_file)
        self._log(f"Database contents: {counts}")
        if counts.get("matches", 0) == 0 and counts.get("two_view_geometries", 0) == 0:
            raise Exception("COLMAP matcher produced no pairs. Try a different matcher or check image quality.")

    def _glomap_mapper(self) -> None:
        self._log("Step 5/5: GLOMAP global mapping...")
        self.workspace.sparse.mkdir(parents=True, exist_ok=True)

        # Attempt to count images for granular progress.
        image_files = list(self.workspace.frames.glob("*.jpg")) + list(self.workspace.frames.glob("*.png"))
        total_images = len(image_files)
        progress_regex = re.compile(r"Processing image \[(\d+)/(\d+)\]", re.IGNORECASE)

        self._run_command(
            "GLOMAP Mapper",
            self.config.glomap,
            [
                "mapper",
                "--database_path", str(self.workspace.database_file),
                "--image_path", str(self.workspace.frames),
                "--output_path", str(self.workspace.sparse),
            ],
            progress_start=75,
            progress_end=95,
            progress_regex=progress_regex,
            total_items=total_images,
        )

    def _export_txt_model(self) -> None:
        r"""Convert the binary GLOMAP model to COLMAP TXT format in both sparse\0 and sparse\."""
        model_dir = self.workspace.sparse_model_dir()
        if not model_dir.exists():
            return
        self._log("Exporting sparse model to COLMAP TXT format...")

        # TXT inside the model folder (same as the bat).
        self._run_command(
            "COLMAP Model Converter",
            self.config.colmap,
            [
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(model_dir),
                "--output_type", "TXT",
            ],
            progress_start=95,
            progress_end=97,
        )

        # TXT in the parent sparse folder for Blender auto-detect (same as the bat).
        self._run_command(
            "COLMAP Model Converter",
            self.config.colmap,
            [
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(self.workspace.sparse),
                "--output_type", "TXT",
            ],
            progress_start=97,
            progress_end=100,
        )

    def cancel(self) -> None:
        self._cancelled = True
        if self._current_proc is not None:
            self._current_proc.terminate()

    def run(self) -> None:
        try:
            base = make_session_workspace(self.config.output_dir)
            self.workspace = WorkspacePaths(base)
            self._log(f"Workspace: {self.workspace.base}")

            self._extract_frames()
            if self.config.mask_video is not None:
                self._extract_masks()
            self._colmap_feature_extractor()
            self._colmap_matcher()
            self._glomap_mapper()
            self._export_txt_model()

            sparse_dir = self.workspace.sparse_model_dir()
            if not (sparse_dir / "cameras.txt").exists() or not (sparse_dir / "images.txt").exists():
                raise Exception(f"GLOMAP did not produce a valid sparse model in {sparse_dir}")

            self.export_ready.emit(str(sparse_dir))
            self.finished.emit(True, "Tracking complete.")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self.finished.emit(False, str(exc))


class PipelineRunner:
    """Threaded wrapper around PipelineWorker."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._thread = QThread()
        self._worker = PipelineWorker(config)
        self._worker.moveToThread(self._thread)

        self.log = self._worker.log
        self.progress = self._worker.progress
        self.finished = self._worker.finished
        self.export_ready = self._worker.export_ready

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._worker.cancel()

    @property
    def workspace(self) -> WorkspacePaths | None:
        return self._worker.workspace
