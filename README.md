# OnTrack

A PySide6 desktop wrapper around FFmpeg → COLMAP → GLOMAP that extracts a 3D camera track from a video and exports it to After Effects (`.jsx`) or Blender (`.py`).

Also includes a lens calibration tab (Discorpy) for undistorting footage before tracking and redistorting rendered footage afterward, with optional garbage-matte masking of moving subjects and ST map (EXR) export.

## Folder Layout

```
OnTrack/
├── main.py
├── requirements.txt
├── core/
│   ├── paths.py                 # Workspace & binary path helpers
│   ├── parser.py                # COLMAP sparse model parser
│   ├── pipeline.py              # FFmpeg → COLMAP → GLOMAP runner
│   └── exporters/
│       ├── ae_jsx.py            # After Effects script generator
│   ├   └── blender_py.py        # Blender script generato
│   └── undistort_widget.py      # Lens calibration / undistort tab
├── EXTRA/Undistort/             # Vendored Discorpy libraryr
├── ui/
│   └── main_window.py           # Main application window
├── Dependencies/                # Your bundled binaries
│   ├── 01 GLOMAP/
│   │   ├── colmap.exe
│   │   ├── glomap.exe
│   │   └── *.dll
│   └── 03 FFMPEG/bin/
│       └── ffmpeg.exe
└── temp_workspace/              # Created automatically per session
```

## Setup

1. Open a terminal in this folder.
2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

> **Note:** the tracking-only `.venv` (any Python 3.10+) runs everything except the Roto tab. For the embedded roto editor you need an interpreter with torch/av/PySide6 — see below.

## Launching

**Easiest: double-click `run.bat`.** It picks the best interpreter automatically.

Or from a terminal / your IDE, pick the interpreter directly — the app checks for the roto editor's dependencies (torch, av, PySide6) and re-launches under a suitable Python if needed:

```powershell
roto\.venv\Scripts\python.exe main.py   # full app incl. Roto tab (recommended)
.venv\Scripts\python.exe main.py        # everything except the Roto tab
```

Interpreter priority (`run.bat` and the `main.py` bootstrap both use it):

1. `ONTRACK_ROTO_PYTHON` env var (override)
2. `roto\.venv` — dedicated env with Sammie + OnTrack deps (recommended)
3. `P:\Program Files\Sammie Roto\Sammie-Roto-2\.venv` — an existing Sammie install
4. `.venv` — plain OnTrack env (Roto tab shows an error placeholder)

If you see `[ontrack] roto deps missing; relaunching with …`, that's the bootstrap fixing the interpreter choice on the fly — the app still starts normally.

## Dependencies

The **Settings → Dependencies…** dialog lists every dependency with found/missing status and install buttons:

- **Binaries** — FFmpeg / COLMAP / GLOMAP paths (auto-detected from `Dependencies/`).
- **Tracking models** — SuperPoint/LightGlue ONNX downloads for Learned/Object Track, and `superpoint_lowthr.onnx` generation.
- **Roto models** — SAM2 checkpoints + MatAnyone weights, downloaded into `roto/checkpoints` via the roto model manager (resume + checksum).

On first launch the app prompts if anything is missing.

## Usage

1. Add a video by clicking **Add Video** or dragging and dropping a video file onto the window.
2. Review the video preview and metadata (resolution, duration, FPS, frame count). The FPS field auto-populates from the video.
3. Set FPS, frame skip, and track type. Binary paths and tuning knobs live in **Settings → Tracking Settings…**.
4. Click **Run Tracking**.
5. When complete, export to **After Effects (.jsx)** or **Blender (.py)** from the Export tab.

## Track Type

The Track tab has a **Track type** selector:

- **Classic (SIFT)** - COLMAP SIFT features + matching. Best for sharp, textured footage.
- **Learned (SuperPoint + LightGlue)** - neural features/matching run through ONNX Runtime + DirectML, so they work on any DirectX 12 GPU (AMD/NVIDIA/Intel) with no CUDA. Matches are imported into the COLMAP database and GLOMAP solves as normal. More robust on blurry / low-texture shots, but slower (~65 ms per frame pair on an RX 7900 XT) and sub-pixel precision is lower (higher reprojection error), so keep Classic for clean shots.

Learned tracking needs `onnxruntime-directml` (in `requirements.txt`) and these models in `Dependencies/06 MODELS/`:

- `superpoint.onnx` and `superpoint_lightglue.onnx` from the `v0.1.3` release of `fabio-sim/LightGlue-ONNX`
- `superpoint_lowthr.onnx` (optional, strongly recommended): generate with `pip install onnx` then `python tools/patch_superpoint.py`. It lowers SuperPoint's detection threshold so blurry frames still give a fixed keypoint count, which keeps DirectML from recompiling per frame pair.

Note: SuperPoint's weights are licensed for non-commercial use.

## Roto Tab

The **Roto** tab embeds a vendored copy of Sammie-Roto (SAM2-based roto, in `roto/`) directly inside OnTrack — no separate window. The embedded editor loads lazily the first time you open the tab (torch + SAM2 import takes a moment) and resumes its previous session.

Because the editor runs in-process, OnTrack must run on a Python environment that has Sammie's dependencies (torch, av, PySide6). `run.bat` prefers `roto\.venv` — a dedicated Python 3.12 environment containing the full Sammie + OnTrack dependency set — and falls back to the Sammie-Roto install's venv, then the plain `.venv` (where everything works except the embedded roto editor). `main.py` auto-relaunches under the roto venv when needed, so any launch method works.

Model checkpoints are shared via the `roto/checkpoints` junction; the editor's session lives in `roto/temp/` (masks: `roto/temp/masks/<frame>/<object>.png`).

Workflow:
1. Load a video in the Track tab.
2. In the Track tab's mask dropdowns (next to *Ignore matte* / *Object mask*), choose **+ Create new mask…** — the Roto tab opens with the video loaded in the editor.
3. Segment objects (and optionally run **Refine Matte** for soft edges); masks appear live in the Roto tab's mask list (right panel) and in the Track tab dropdowns (`Roto: <object>` / `Roto: All objects`). Refined mattes are preferred automatically when present.
4. Selecting a roto mask merges the per-object PNGs into a lossless matte video (`roto/exports/`) and fills in the path automatically — the tracking pipeline consumes it as usual.

**Export Matte Video…** in the Roto panel writes a standalone matte (soft alpha edges preserved) for the selected objects — handy for compositing.

## Notes / Known Limitations

- **Coordinate conversion**: The AE orientation is currently derived using XYZ Euler angles. Depending on your footage you may need to adjust the rotation order or add a null target. This is the area most likely to need tweaking.
- **Cleanup**: Enable *Clean workspace on exit* to delete the temporary frames/database/sparse folder when the app closes.
- **Progress**: GLOMAP progress is parsed from `Processing image [X/Y]` style output; if your build prints differently, the bar will use stage-based increments instead.

## License

Use at your own risk. Bundled binaries (FFmpeg, COLMAP, GLOMAP) belong to their respective projects.
