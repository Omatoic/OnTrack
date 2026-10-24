import os
import shutil
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """Return the project root (parent of the core/ package)."""
    return Path(__file__).resolve().parent.parent


def default_dependency_root() -> Path:
    """Return the default Dependencies folder inside the project root."""
    return project_root() / "Dependencies"


def _find_binary(directory: Path, name: str) -> Path:
    """Look for an executable in a directory or its bin subdirectory."""
    for sub in (directory, directory / "bin"):
        candidate = sub / name
        if candidate.exists():
            return candidate
    # Fall back to the direct location even if it does not exist yet.
    return directory / name


def default_binary_paths() -> dict[str, Path]:
    """Return default absolute paths to the bundled binaries."""
    dep = default_dependency_root()
    return {
        "ffmpeg": _find_binary(dep / "03 FFMPEG", "ffmpeg.exe"),
        "ffprobe": _find_binary(dep / "03 FFMPEG", "ffprobe.exe"),
        "colmap": _find_binary(dep / "01 GLOMAP", "colmap.exe"),
        "glomap": _find_binary(dep / "01 GLOMAP", "glomap.exe"),
    }


def default_temp_root() -> Path:
    """Return the default temporary workspace root."""
    return project_root() / "temp_workspace"


def make_session_workspace(root: Path | None = None) -> Path:
    """Create a fresh per-session workspace and return its path."""
    if root is None:
        root = default_temp_root()
    session_dir = root / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    (session_dir / "01_frames").mkdir(parents=True, exist_ok=True)
    (session_dir / "02_database").mkdir(parents=True, exist_ok=True)
    (session_dir / "03_sparse").mkdir(parents=True, exist_ok=True)
    (session_dir / "04_masks").mkdir(parents=True, exist_ok=True)
    (session_dir / "05_object_masks").mkdir(parents=True, exist_ok=True)
    (session_dir / "06_object").mkdir(parents=True, exist_ok=True)
    return session_dir


def clear_workspace(workspace: Path) -> None:
    """Remove the workspace directory tree if it exists."""
    if workspace.exists():
        shutil.rmtree(workspace)


class WorkspacePaths:
    """Convenience class for the folder layout used by the pipeline."""

    def __init__(self, base: Path):
        self.base = base
        self.frames = base / "01_frames"
        self.database = base / "02_database"
        self.sparse = base / "03_sparse"
        self.masks = base / "04_masks"
        self.object_masks = base / "05_object_masks"
        self.object_dir = base / "06_object"
        self.database_file = self.database / "database.db"

    def ensure_dirs(self) -> None:
        self.frames.mkdir(parents=True, exist_ok=True)
        self.database.mkdir(parents=True, exist_ok=True)
        self.sparse.mkdir(parents=True, exist_ok=True)
        self.masks.mkdir(parents=True, exist_ok=True)
        self.object_masks.mkdir(parents=True, exist_ok=True)
        self.object_dir.mkdir(parents=True, exist_ok=True)

    def sparse_model_dir(self) -> Path:
        """Return the standard COLMAP sparse model subfolder sparse/0."""
        return self.sparse / "0"

    def cameras_txt(self) -> Path:
        return self.sparse_model_dir() / "cameras.txt"

    def images_txt(self) -> Path:
        return self.sparse_model_dir() / "images.txt"
