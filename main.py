import sys
import traceback
from pathlib import Path


def _ensure_roto_env() -> None:
    """Re-exec under the Sammie-Roto venv if the roto deps are missing.

    The Roto tab embeds Sammie-Roto in-process, so OnTrack needs a Python
    with torch/av/PySide6. If the current interpreter lacks them, hand off
    to the Sammie venv so `python main.py` works from anywhere.
    """
    import importlib.util
    import os
    import subprocess

    if os.environ.get("ONTRACK_NO_REEXEC"):
        return
    if all(importlib.util.find_spec(m) for m in ("av", "torch")):
        return

    candidates = [
        os.environ.get("ONTRACK_ROTO_PYTHON"),
        str(Path(__file__).parent / "roto" / ".venv" / "Scripts" / "python.exe"),
        r"P:\Program Files\Sammie Roto\Sammie-Roto-2\.venv\Scripts\python.exe",
    ]
    for c in candidates:
        if not c:
            continue
        alt = Path(c)
        if alt.exists() and alt.resolve() != Path(sys.executable).resolve():
            print(f"[ontrack] roto deps missing; relaunching with {alt}")
            sys.exit(subprocess.call([str(alt), str(Path(__file__).resolve()), *sys.argv[1:]]))


if __name__ == "__main__":
    _ensure_roto_env()
    from ui.main_window import main

    log_path = Path(__file__).parent / "startup.log"
    try:
        sys.exit(main())
    except Exception as exc:
        msg = f"FATAL: {exc}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        try:
            log_path.write_text(msg, encoding="utf-8")
        except OSError:
            pass
        sys.exit(1)
