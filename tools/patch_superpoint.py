"""One-off: lower SuperPoint's detection threshold so blurry / low-texture frames still yield
enough candidates for a fixed top-K (fixed shapes keep DirectML from recompiling per pair).

Needs the `onnx` package (not an app dependency):
    pip install onnx
    python tools/patch_superpoint.py
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MODELS = Path(__file__).resolve().parent.parent / "Dependencies" / "06 MODELS"
THRESHOLD = 1e-5

m = onnx.load(str(MODELS / "superpoint.onnx"))
patched = 0
for node in m.graph.node:
    if node.op_type == "Constant" and node.attribute and node.attribute[0].t.dims == []:
        arr = numpy_helper.to_array(node.attribute[0].t)
        if arr.dtype == np.float32 and abs(float(arr) - 0.0005) < 1e-9:
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array(THRESHOLD, dtype=np.float32)))
            patched += 1
if patched != 1:
    raise SystemExit(f"Expected to patch exactly one threshold constant, patched {patched}")
onnx.save(m, str(MODELS / "superpoint_lowthr.onnx"))
print("wrote superpoint_lowthr.onnx")
