"""Dependencies panel: model checkpoints, ONNX tracking models, and the
roto editor's runtime status. Used as a page inside the Settings dialog."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import zipfile
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
from core.paths import BINARY_INSTALL_DIRS, default_binary_paths, default_dependency_root, project_root

LIGHTGLUE_RELEASE = "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3"

# (filename, label, url | None = generated locally)
ONNX_MODELS = [
    ("superpoint.onnx", "SuperPoint features", f"{LIGHTGLUE_RELEASE}/superpoint.onnx"),
    ("superpoint_lightglue.onnx", "SuperPoint + LightGlue matcher",
     f"{LIGHTGLUE_RELEASE}/superpoint_lightglue.onnx"),
    ("superpoint_lowthr.onnx", "SuperPoint low-threshold (recommended)", None),
]

# (key, label, zip url, approx MB). Pinned to COLMAP 3.12.x / GLOMAP 1.2.0:
# COLMAP 4.x renamed CLI options the pipeline relies on. CPU ("nocuda")
# builds run everywhere; SIFT runs on CPU by default anyway.
BINARY_DOWNLOADS = [
    ("ffmpeg", "FFmpeg", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", 115),
    ("colmap", "COLMAP 3.12.6",
     "https://github.com/colmap/colmap/releases/download/3.12.6/colmap-x64-windows-nocuda.zip", 88),
    ("glomap", "GLOMAP 1.2.0",
     "https://github.com/colmap/glomap/releases/download/1.2.0/glomap-x64-windows-nocuda.zip", 18),
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

    def __init__(self, url: str, dest: Path, parent=None, extract_to: Path | None = None):
        """Download url to dest. With extract_to, dest is a temporary zip that
        is unpacked into extract_to (a single top-level folder is flattened)
        and then deleted."""
        super().__init__(parent)
        self._url = url
        self._dest = dest
        self._extract_to = extract_to

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
            if self._extract_to is not None:
                self.progress.emit(f"Extracting {self._dest.name}…")
                _extract_flat(self._dest, self._extract_to)
                self._dest.unlink()
            self.finished.emit(True, f"Installed {self._dest.name}")
        except Exception as exc:
            self.finished.emit(False, f"Download failed: {exc}")


def _extract_flat(archive: Path, target: Path) -> None:
    staging = target.with_name(target.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)
    entries = list(staging.iterdir())
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() and entries[0].name != "bin" else staging
    target.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        dest = target / item.name
        if dest.is_dir():
            shutil.rmtree(dest)
        elif dest.exists():
            dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(staging, ignore_errors=True)


def _status(ok: bool, detail: str = "") -> QLabel:
    lbl = QLabel(("✔ " if ok else "✖ ") + (detail or ("found" if ok else "missing")))
    lbl.setStyleSheet(f"color: {'#28a745' if ok else '#dc3545'}; font-weight: bold;")
    return lbl


class DependenciesWidget(QWidget):
    """Lists every downloadable dependency with status + install actions."""

    binaries_installed = Signal(list)   # keys from BINARY_DOWNLOADS

    def __init__(self, parent=None, binary_paths_getter=None):
        super().__init__(parent)
        # Returns {"ffmpeg"|"colmap"|"glomap": configured path}; lets the
        # status reflect paths the user browsed to, not just the defaults.
        self._binary_paths_getter = binary_paths_getter or default_binary_paths
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
        self._essentials_btn = QPushButton("Download all essentials")
        self._essentials_btn.setToolTip(
            "SuperPoint/LightGlue tracking models, SAM2 Base+ and MatAnyone (~500 MB total)"
        )
        self._essentials_btn.clicked.connect(self.install_essentials)
        env_layout.addWidget(self._essentials_btn, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(env_group)

        # --- Programs ---
        bin_group = QGroupBox("Programs — FFmpeg, COLMAP, GLOMAP (required)")
        bin_layout = QVBoxLayout(bin_group)
        for key, label, url, mb in BINARY_DOWNLOADS:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label} (~{mb} MB)"), 2)
            status = QLabel()
            status.setObjectName(f"status_bin_{key}")
            row.addWidget(status, 1)
            btn = QPushButton("Download")
            btn.clicked.connect(lambda _c, k=key: self._download_binary(k))
            row.addWidget(btn)
            bin_layout.addLayout(row)
        bin_note = QLabel(f"Installed into {default_dependency_root()}.")
        bin_note.setWordWrap(True)
        bin_note.setStyleSheet("color: #a0a0a0; font-size: 9pt;")
        bin_layout.addWidget(bin_note)
        layout.addWidget(bin_group)

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

        configured = self._binary_paths_getter()
        for key, _label, _url, _mb in BINARY_DOWNLOADS:
            lbl = self.findChild(QLabel, f"status_bin_{key}")
            if lbl:
                ok = Path(configured.get(key, "")).is_file()
                lbl.setText(("✔ found" if ok else "✖ missing"))
                lbl.setStyleSheet(f"color: {'#28a745' if ok else '#dc3545'}; font-weight: bold;")

        have = all(importlib.util.find_spec(m) for m in ("av", "torch"))
        self._env_label.setText(
            f"Python: {sys.executable}\n"
            f"Roto runtime (torch + av): {'available' if have else 'MISSING — the app will relaunch under a suitable interpreter'}"
        )

    # ---- Actions ----

    def install_essentials(self) -> None:
        """Download every missing essential. Programs and ONNX tracking models
        download one after another in the background (then the low-threshold
        SuperPoint model is derived); roto checkpoints use the roto model
        manager's own progress dialog meanwhile."""
        models = default_dependency_root() / "06 MODELS"
        configured = self._binary_paths_getter()
        jobs = [
            lambda done, k=key: self._download_binary(k, on_done=done)
            for key, _l, _u, _mb in BINARY_DOWNLOADS
            if not Path(configured.get(key, "")).is_file()
        ] + [
            lambda done, u=url, f=fname: self._download(u, f, on_done=done)
            for fname, _l, url in ONNX_MODELS
            if url and not (models / fname).exists()
        ]
        gen_lowthr = not (models / "superpoint_lowthr.onnx").exists()

        def next_job(ok: bool = True, _msg: str = "") -> None:
            if jobs:
                jobs.pop(0)(next_job)   # keep going even if one item failed
            elif gen_lowthr and (models / "superpoint.onnx").exists():
                self._generate_lowthr()

        next_job()
        if roto_keys := missing_roto_keys():
            self._download_roto(roto_keys)

    def _download_binary(self, key: str, on_done=None) -> None:
        _k, label, url, _mb = next(b for b in BINARY_DOWNLOADS if b[0] == key)
        target = default_dependency_root() / BINARY_INSTALL_DIRS[key]
        archive = default_dependency_root() / "_downloads" / Path(url).name

        def finished(ok: bool, _msg: str) -> None:
            if ok:
                self.binaries_installed.emit([key])

        self._download(url, archive.name, on_done=on_done, dest=archive,
                       extract_to=target, before_done=finished)

    def _download(self, url: str, fname: str, on_done=None, dest: Path | None = None,
                  extract_to: Path | None = None, before_done=None) -> None:
        dest = dest or default_dependency_root() / "06 MODELS" / fname
        worker = _DownloadWorker(url, dest, self, extract_to=extract_to)
        if before_done is not None:
            worker.finished.connect(before_done)
        worker.progress.connect(self._set_status_line)
        worker.finished.connect(self._on_download_done)
        if on_done is not None:
            worker.finished.connect(on_done)
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

    def _download_roto(self, key: str | list[str]) -> None:
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


def missing_roto_keys() -> list[str]:
    ckpts = roto.roto_root() / "checkpoints"
    keys = []
    if not any((ckpts / f).exists() for _k, f, _l in ROTO_MODELS[:4]):
        keys.append("Base")
    if not any((ckpts / f).exists() for _k, f, _l in ROTO_MODELS[4:]):
        keys.append("matanyone")
    return keys


def missing_downloads(binaries: dict[str, str | Path]) -> list[tuple[str, int]]:
    """(label, approx MB) for every essential that isn't installed yet.
    ``binaries`` maps "ffmpeg"/"colmap"/"glomap" to the configured paths."""
    models = default_dependency_root() / "06 MODELS"
    missing = [
        (label, mb) for key, label, _u, mb in BINARY_DOWNLOADS
        if not Path(binaries.get(key, "")).is_file()
    ]
    missing += [
        (label, 50 if "LightGlue" in label else 5)
        for fname, label, url in ONNX_MODELS if url and not (models / fname).exists()
    ]
    sizes = {"Base": ("SAM2 Base+ segmentation model (Roto tab)", 320),
             "matanyone": ("MatAnyone matte refinement model (Roto tab)", 140)}
    missing += [sizes[k] for k in missing_roto_keys()]
    return missing
