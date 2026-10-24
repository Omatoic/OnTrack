"""Learned feature extraction + matching (SuperPoint + LightGlue) via ONNX Runtime.

Runs on any DirectX 12 GPU (AMD/NVIDIA/Intel) through the DirectML provider and
falls back to CPU. Results are fed into COLMAP's database so GLOMAP can solve
shots where SIFT fails (motion blur, low texture).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.paths import default_dependency_root

SUPERPOINT_MODEL = "superpoint.onnx"
SUPERPOINT_LOWTHR_MODEL = "superpoint_lowthr.onnx"  # see tools/patch_superpoint.py
LIGHTGLUE_MODEL = "superpoint_lightglue.onnx"


def default_model_dir() -> Path:
    return default_dependency_root() / "06 MODELS"


def models_available(model_dir: Path | None = None) -> bool:
    model_dir = model_dir or default_model_dir()
    return all((model_dir / m).exists() for m in (SUPERPOINT_MODEL, LIGHTGLUE_MODEL))


def learned_unavailable_reason(model_dir: Path | None = None) -> str | None:
    """Return why learned tracking can't run, or None if it can."""
    from importlib.util import find_spec

    if find_spec("onnxruntime") is None:
        return "onnxruntime is not installed. Run: .venv\\Scripts\\pip install onnxruntime-directml"
    if not models_available(model_dir):
        return f"SuperPoint/LightGlue ONNX models not found in {model_dir or default_model_dir()}"
    return None


@dataclass
class Features:
    kpts: np.ndarray      # (N, 2) float32, COLMAP pixel coords (centre of first pixel = 0.5)
    desc: np.ndarray      # (N, 256) float32
    size: tuple[int, int]  # (width, height) of the original image


_ZERO_DESC = " ".join(["0"] * 128)


def write_feature_txt(path: Path, feats: Features) -> None:
    """Write keypoints in COLMAP's text feature format for `feature_importer`.

    COLMAP requires 128-dim SIFT-style descriptors; they are unused downstream because
    matches are supplied explicitly, so zeros are written to keep the file simple.
    """
    lines = [f"{len(feats.kpts)} 128"]
    lines += [f"{x:.3f} {y:.3f} 1 0 {_ZERO_DESC}" for x, y in feats.kpts]
    path.write_text("\n".join(lines) + "\n")


def write_match_list(path: Path, entries: list[tuple[str, str, np.ndarray]]) -> None:
    """Write raw matches in COLMAP's `matches_importer` format."""
    with open(path, "w") as f:
        for name_a, name_b, m in entries:
            f.write(f"{name_a} {name_b}\n")
            f.writelines(f"{i} {j}\n" for i, j in m.tolist())
            f.write("\n")


def sequential_pairs(n: int, overlap: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + overlap))]


class LearnedFeatureEngine:
    def __init__(self, model_dir: Path | None = None, max_keypoints: int = 1536,
                 max_size: int = 1024, use_gpu: bool = True, log=print):
        import onnxruntime as ort

        model_dir = model_dir or default_model_dir()
        if not models_available(model_dir):
            raise FileNotFoundError(f"SuperPoint/LightGlue ONNX models not found in {model_dir}")
        self.max_keypoints = max_keypoints
        self.max_size = max_size

        available = ort.get_available_providers()
        providers = (["DmlExecutionProvider"] if use_gpu and "DmlExecutionProvider" in available else []) + ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        if providers[0] == "DmlExecutionProvider":
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.provider = providers[0]
        sp_name = SUPERPOINT_LOWTHR_MODEL if (model_dir / SUPERPOINT_LOWTHR_MODEL).exists() else SUPERPOINT_MODEL
        self._sp = ort.InferenceSession(str(model_dir / sp_name), opts, providers=providers)
        self._lg = ort.InferenceSession(str(model_dir / LIGHTGLUE_MODEL), opts, providers=providers)
        log(f"Learned features: {self.provider}, max {max_keypoints} keypoints @ {max_size}px")

    def extract(self, image_path: Path, mask_path: Path | None = None) -> Features:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image {image_path}")
        h, w = gray.shape
        scale = min(1.0, self.max_size / max(h, w))
        rw, rh = max(8, int(round(w * scale / 8)) * 8), max(8, int(round(h * scale / 8)) * 8)
        small = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA) if (rw, rh) != (w, h) else gray
        inp = (small.astype(np.float32) / 255.0)[None, None]

        kpts, scores, desc = self._sp.run(None, {"image": inp})
        kpts, scores, desc = kpts[0].astype(np.float32), scores[0], desc[0]

        if mask_path is not None and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
                keep = mask[kpts[:, 1].astype(int).clip(0, rh - 1), kpts[:, 0].astype(int).clip(0, rw - 1)] > 127
                kpts, scores, desc = kpts[keep], scores[keep], desc[keep]

        if len(scores) > self.max_keypoints:
            top = np.argsort(-scores)[: self.max_keypoints]
            kpts, desc = kpts[top], desc[top]

        kpts = (kpts + 0.5) * np.array([w / rw, h / rh], dtype=np.float32)
        return Features(kpts=kpts.astype(np.float32), desc=desc.astype(np.float32), size=(w, h))

    def match(self, a: Features, b: Features, min_score: float = 0.0) -> np.ndarray:
        """Match two feature sets. Returns (M, 2) index pairs into a.kpts / b.kpts.

        DirectML recompiles for every new input shape, so keep keypoint counts constant
        (the low-threshold SuperPoint model + top-K gives exactly `max_keypoints`).
        """
        if len(a.kpts) < 2 or len(b.kpts) < 2:
            return np.zeros((0, 2), dtype=np.uint32)

        def norm(f: Features) -> np.ndarray:
            w, h = f.size
            return ((f.kpts - np.array([w, h], dtype=np.float32) / 2) / (max(w, h) / 2))[None].astype(np.float32)

        m0, _, s0, _ = self._lg.run(None, {
            "kpts0": norm(a), "kpts1": norm(b), "desc0": a.desc[None], "desc1": b.desc[None],
        })
        m0, s0 = m0[0], s0[0]
        idx = np.nonzero((m0 > -1) & (s0 >= min_score))[0]
        return np.stack([idx, m0[idx]], axis=1).astype(np.uint32)
