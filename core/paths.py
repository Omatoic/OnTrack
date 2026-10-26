import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """Return the project root (parent of the core/ package)."""
    return Path(__file__).resolve().parent.parent


def default_dependency_root() -> Path:
    """Return the default Dependencies folder inside the project root."""
    return project_root() / "Dependencies"


def _find_binary(directories: list[Path], name: str) -> Path:
    """Look for an executable in each directory or its bin subdirectory."""
    for directory in directories:
        for sub in (directory, directory / "bin"):
            candidate = sub / name
            if candidate.exists():
                return candidate
    # Fall back to where the downloader installs it, even if it doesn't exist yet.
    return directories[-1] / "bin" / name


# Where the in-app downloader installs each binary (relative to Dependencies/).
BINARY_INSTALL_DIRS = {
    "ffmpeg": "03 FFMPEG",
    "colmap": "01 GLOMAP/colmap",
    "glomap": "01 GLOMAP/glomap",
}


def default_binary_paths() -> dict[str, Path]:
    """Return default absolute paths to the bundled binaries. A flat
    '01 GLOMAP' folder (hand-assembled) is honoured before the downloader's
    per-tool subfolders."""
    dep = default_dependency_root()
    ffmpeg_dirs = [dep / BINARY_INSTALL_DIRS["ffmpeg"]]
    return {
        "ffmpeg": _find_binary(ffmpeg_dirs, "ffmpeg.exe"),
        "ffprobe": _find_binary(ffmpeg_dirs, "ffprobe.exe"),
        "colmap": _find_binary([dep / "01 GLOMAP", dep / BINARY_INSTALL_DIRS["colmap"]], "colmap.exe"),
        "glomap": _find_binary([dep / "01 GLOMAP", dep / BINARY_INSTALL_DIRS["glomap"]], "glomap.exe"),
    }


def default_temp_root() -> Path:
    """Scratch root in the OS temp folder. Everything under it is disposable:
    it is wiped on exit and stale leftovers are purged on startup. Saving a
    project copies what's needed out of here."""
    return Path(tempfile.gettempdir()) / "OnTrack"


def session_temp_root() -> Path:
    return default_temp_root() / "sessions"


def matte_temp_root() -> Path:
    return default_temp_root() / "mattes"


def roto_temp_root() -> Path:
    return default_temp_root() / "roto"


def is_temp_path(path: Path | str | None) -> bool:
    if not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(default_temp_root().resolve())
    except OSError:
        return False


def purge_stale_temp(max_age_hours: float = 24.0) -> None:
    """Delete session/matte leftovers older than max_age_hours (e.g. from a
    crash). The age guard keeps a concurrently running instance safe."""
    cutoff = time.time() - max_age_hours * 3600
    for root in (session_temp_root(), matte_temp_root()):
        if not root.exists():
            continue
        for entry in root.iterdir():
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
            except OSError:
                pass


def make_session_workspace(root: Path | None = None) -> Path:
    """Create a fresh per-session workspace and return its path."""
    if root is None:
        root = session_temp_root()
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
