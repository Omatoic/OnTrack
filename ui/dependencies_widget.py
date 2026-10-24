"""Dependencies panel: model checkpoints, ONNX tracking models, and the
roto editor's runtime status. Used as a page inside the Settings dialog."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core import roto
from core.paths import default_dependency_root, project_root

LIGHTGLUE_RELEASE = "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3"

# (filename, label, url | None = generated locally)
ONNX_MODELS = [
    ("superpoint.onnx", "SuperPoint features", f"{LIGHTGLUE_RELEASE}/superpoint.onnx"),
    ("superpoint_lightglue.onnx", "SuperPoint + LightGlue matcher",
     f"{LIGHTGLUE_RELEASE}/superpoint_lightglue.onnx"),
    ("superpoint_lowthr.onnx", "SuperPoint low-threshold (recommended)", None),
]

# (registry key, checkpoint filename, label)
ROTO_MODELS = [
    ("Base", "sam2.1_hiera_base_plus.pt", "SAM2 Base+ (fast)"),
    ("Large", "sam2.1_hiera_large.pt", "SAM2 Large (best quality)"),
    ("Efficient", "efficienttam_s_512x512.pt", "EfficientTAM (lightweight)"),
    ("Anime", "sam2.1_anime_v1.pt", "SAM2 Anime"),
    ("matanyone", "matanyone.pth", "MatAnyone (refine matte)"),
    ("matanyone2", "matanyone2.pth", "MatAnyone2 (refine matte)"),
]


class _DownloadWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, url: str, dest: Path, parent=None):
        super().__init__(parent)
        self._url = url
        self._dest = dest

    def run(self) -> None:
        try:
            import requests
            self._dest.parent.mkdir(parents=True, exist_ok=True)
            part = self._dest.with_suffix(self._dest.suffix + ".part")
            with requests.get(self._url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            self.progress.emit(
                                f"{self._dest.name}: {done/1e6:.0f}/{total/1e6:.0f} MB"
                            )
            part.replace(self._dest)
            self.finished.emit(True, f"Downloaded {self._dest.name}")
        except Exception as exc:
            self.finished.emit(False, f"Download failed: {exc}")


def _status(ok: bool, detail: str = "") -> QLabel:
    lbl = QLabel(("✔ " if ok else "✖ ") + (detail or ("found" if ok else "missing")))
    lbl.setStyleSheet(f"color: {'#28a745' if ok else '#dc3545'}; font-weight: bold;")
    return lbl


class DependenciesWidget(QWidget):
    """Lists every downloadable dependency with status + install actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workers: list[QThread] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(12)

        # --- Interpreter ---
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout(env_group)
        self._env_label = QLabel()
        self._env_label.setWordWrap(True)
        env_layout.addWidget(self._env_label)
        layout.addWidget(env_group)

        # --- Tracking models ---
        onnx_group = QGroupBox("Tracking models — needed for Learned / Object Track")
        self._onnx_layout = QVBoxLayout(onnx_group)
        for fname, label, url in ONNX_MODELS:
            self._onnx_layout.addLayout(self._model_row(label, fname, url))
        layout.addWidget(onnx_group)

        # --- Roto checkpoints ---
        roto_group = QGroupBox("Roto models — SAM2 & matting (roto/checkpoints)")
        self._roto_layout = QVBoxLayout(roto_group)
        for key, fname, label in ROTO_MODELS:
            self._roto_layout.addLayout(self._roto_row(key, label, fname))
        note = QLabel(
            "Downloads run through the roto model manager with resume + checksum "
            "verification. Large is ~900 MB, Base+ ~320 MB."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #a0a0a0; font-size: 9pt;")
        self._roto_layout.addWidget(note)
        layout.addWidget(roto_group)

        # --- ffmpeg binaries ---
        ff_group = QGroupBox("FFmpeg (needed if not bundled)")
        ff_layout = QHBoxLayout(ff_group)
        self._ffmpeg_status = QLabel()
        ff_layout.addWidget(self._ffmpeg_status)
        ff_layout.addStretch()
        dl = QPushButton("Download FFmpeg build…")
        dl.clicked.connect(lambda: os.startfile("https://www.gyan.dev/ffmpeg/builds/"))
        ff_layout.addWidget(dl)
        layout.addWidget(ff_group)

        open_btn = QPushButton("Open Dependencies folder")
        open_btn.clicked.connect(
            lambda: os.startfile(default_dependency_root())
            if default_dependency_root().exists() else None
        )
        layout.addWidget(open_btn, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()

        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- Rows ----

    def _model_row(self, label: str, fname: str, url: str | None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label), 2)
        status = QLabel()
        status.setObjectName(f"status_{fname}")
        row.addWidget(status, 1)
        if url:
            btn = QPushButton("Download")
            btn.clicked.connect(lambda _c, u=url, f=fname: self._download(u, f))
        else:
            btn = QPushButton("Generate")
            btn.setToolTip("Runs tools/patch_superpoint.py to derive this model from superpoint.onnx")
            btn.clicked.connect(self._generate_lowthr)
        row.addWidget(btn)
        return row

    def _roto_row(self, key: str, label: str, fname: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label), 2)
        status = QLabel()
        status.setObjectName(f"status_ckpt_{fname}")
        row.addWidget(status, 1)
        btn = QPushButton("Download")
        btn.clicked.connect(lambda _c, k=key: self._download_roto(k))
        row.addWidget(btn)
        return row

    # ---- Status ----

    def refresh(self) -> None:
        deps = default_dependency_root()
        models = deps / "06 MODELS"
        for fname, _label, _url in ONNX_MODELS:
            lbl = self.findChild(QLabel, f"status_{fname}")
            if lbl:
                ok = (models / fname).exists()
                lbl.setText(("✔ found" if ok else "✖ missing"))
                lbl.setStyleSheet(f"color: {'#28a745' if ok else '#dc3545'}; font-weight: bold;")

        ckpts = roto.roto_root() / "checkpoints"
        for _key, fname, _label in ROTO_MODELS:
            lbl = self.findChild(QLabel, f"status_ckpt_{fname}")
            if lbl:
                ok = (ckpts / fname).exists()
                lbl.setText(("✔ found" if ok else "✖ missing"))
                lbl.setStyleSheet(f"color: {'#28a745' if ok else '#dc3545'}; font-weight: bold;")

        ffmpeg = deps / "03 FFMPEG" / "bin" / "ffmpeg.exe"
        self._ffmpeg_status.setText(
            "✔ bundled ffmpeg found" if ffmpeg.exists() else "✖ bundled ffmpeg missing"
        )
        self._ffmpeg_status.setStyleSheet(
            f"color: {'#28a745' if ffmpeg.exists() else '#dc3545'}; font-weight: bold;"
        )

        have = all(importlib.util.find_spec(m) for m in ("av", "torch"))
        self._env_label.setText(
            f"Python: {sys.executable}\n"
            f"Roto runtime (torch + av): {'available' if have else 'MISSING — the app will relaunch under a suitable interpreter'}"
        )

    # ---- Actions ----

    def _download(self, url: str, fname: str) -> None:
        dest = default_dependency_root() / "06 MODELS" / fname
        worker = _DownloadWorker(url, dest, self)
        worker.progress.connect(self._set_status_line)
        worker.finished.connect(self._on_download_done)
        self._workers.append(worker)
        self._set_status_line(f"Downloading {fname}…")
        worker.start()

    def _generate_lowthr(self) -> None:
        script = project_root() / "tools" / "patch_superpoint.py"
        if not script.exists():
            QMessageBox.warning(self, "Missing tool", f"Not found: {script}")
            return
        base = default_dependency_root() / "06 MODELS" / "superpoint.onnx"
        if not base.exists():
            QMessageBox.warning(self, "Missing model", "Download superpoint.onnx first.")
            return
        self._set_status_line("Generating superpoint_lowthr.onnx…")
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self._set_status_line(bytes(proc.readAllStandardOutput()).decode(errors="replace").strip().splitlines()[-1])
        )
        proc.finished.connect(lambda _c, _s: (self._set_status_line("Done."), self.refresh()))

        if importlib.util.find_spec("onnx") is None:
            self._set_status_line("Installing onnx package…")
            proc.start(sys.executable, ["-m", "pip", "install", "onnx"])
            proc.finished.connect(lambda *_: proc.start(sys.executable, [str(script)]))
        else:
            proc.start(sys.executable, [str(script)])

    def _download_roto(self, key: str) -> None:
        try:
            os.chdir(roto.roto_root())
            root = str(roto.roto_root())
            if root not in sys.path:
                sys.path.insert(0, root)
            from sammie.model_downloader import ensure_models
        except Exception as exc:
            QMessageBox.warning(
                self, "Downloader unavailable",
                f"Could not load the roto model downloader:\n{exc}"
            )
            return
        ensure_models(key, parent=self, title="Downloading Roto Model")
        self.refresh()

    def _set_status_line(self, text: str) -> None:
        self._env_label.setText(self._env_label.text().split("\n\n")[0] + f"\n\n{text}")

    def _on_download_done(self, ok: bool, message: str) -> None:
        self._set_status_line(message)
        if not ok:
            QMessageBox.warning(self, "Download failed", message)
        self.refresh()


def missing_dependencies(binaries: dict[str, Path]) -> list[str]:
    """Human-readable list of anything that's absent. For the first-run prompt."""
    missing = []
    for name, path in binaries.items():
        if not Path(path).exists():
            missing.append(f"{name} binary")
    models = default_dependency_root() / "06 MODELS"
    if not (models / "superpoint.onnx").exists() or not (models / "superpoint_lightglue.onnx").exists():
        missing.append("learned-tracking ONNX models (optional — needed for Learned/Object Track)")
    ckpts = roto.roto_root() / "checkpoints"
    if not any((ckpts / f).exists() for _k, f, _l in ROTO_MODELS[:4]):
        missing.append("SAM2 checkpoint (needed by the Roto tab)")
    return missing
