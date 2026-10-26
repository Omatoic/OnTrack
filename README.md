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
```

Scratch data (tracking workspaces, roto mattes, the roto editor's session) lives in `%TEMP%\OnTrack\`, not in the project folder — see [Temporary Files](#temporary-files).

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

- **Programs** — FFmpeg / COLMAP / GLOMAP, auto-detected from `Dependencies/` (or browse to your own copies). Missing ones download and unpack automatically: FFmpeg (gyan.dev essentials) → `Dependencies/03 FFMPEG/`, COLMAP 3.12.6 → `Dependencies/01 GLOMAP/colmap/`, GLOMAP 1.2.0 → `Dependencies/01 GLOMAP/glomap/`. Versions are pinned (CPU builds) because COLMAP 4.x renamed CLI options the pipeline uses.
- **Tracking models** — SuperPoint/LightGlue ONNX downloads for Learned/Object Track, and `superpoint_lowthr.onnx` generation.
- **Roto models** — SAM2 checkpoints + MatAnyone weights, downloaded into `roto/checkpoints` via the roto model manager (resume + checksum).

On launch, if anything is missing, the app lists it with sizes and offers to **Download now** (everything on a fresh install: FFmpeg, COLMAP, GLOMAP, SuperPoint/LightGlue, SAM2 Base+, MatAnyone — ~720 MB). Tick *Don't ask again* to silence it; **Download all essentials** in the Dependencies page does the same thing at any time.

## Temporary Files

Unless you save a project, OnTrack writes only to `%TEMP%\OnTrack\`:

- `sessions\` — tracking workspaces (frames, COLMAP database, sparse model). Each new successful run deletes the previous unsaved one, and all of them are deleted on exit.
- `mattes\` — matte videos generated from roto masks for the Track tab. Deleted on exit.
- `roto\` — the roto editor's working session (frames + masks). Kept between launches so the editor can resume; replaced whenever you load a new video.

Leftover `sessions\`/`mattes\` older than 24 h (e.g. after a crash) are purged on startup.

**Save Project** copies the workspace into the project folder and copies any temp matte videos into `<project>\mattes\`, so saved projects never depend on temp files. Setting an explicit **Output folder** in the Track tab keeps every run's workspace there permanently instead.

## Usage

1. Add a video by clicking **Add Video** or dragging and dropping a video file onto the window.
2. Review the video preview and metadata (resolution, duration, FPS, frame count). The FPS field auto-populates from the video.
3. Set FPS, frame skip, track type and (if using an ignore matte) **Matte grow/shrink**. Binary paths and remaining tuning knobs live in **Settings → Tracking Settings…**.
4. Click **Run Tracking**.
5. When complete, export to **After Effects (.jsx)** or **Blender (.py)** from the Export tab.

File browsers (including the Roto editor's) reopen in the last folder you used for that kind of file — videos, images, executables, etc. are remembered separately.

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

Model checkpoints are shared via the `roto/checkpoints` junction; the editor's session lives in `%TEMP%\OnTrack\roto\` (masks: `masks\<frame>\<object>.png`).

Workflow:
1. Load a video in the Track tab.
2. In the Track tab's mask dropdowns (next to *Ignore matte* / *Object mask*), choose **+ Create new mask…** — the Roto tab opens with the video loaded in the editor.
3. Segment objects (and optionally run **Refine Matte** for soft edges). When you switch back to the Track tab, the new mask is loaded into the field you started from automatically (a single object, or *All objects* if you made several). Masks are also listed in the Roto tab's right panel and in the Track tab dropdowns (`Roto: <object>` / `Roto: All objects`) for picking manually. Refined mattes are preferred automatically when present.
4. Selecting a roto mask merges the per-object PNGs into a ProRes HQ MOV matte video (`%TEMP%\OnTrack\mattes\`) and fills in the path automatically — the tracking pipeline consumes it as usual.

**Export Matte Video…** in the Roto panel writes a standalone QuickTime `.mov` using ProRes 422 HQ for the selected objects. Refined soft edges are retained in the black-and-white matte; use it as a luma/track matte in your NLE (it is not an embedded alpha channel).

**File → Save** (`Ctrl+S`) and **File → Save As…** (`Ctrl+Shift+S`) are available from every tab, even before tracking or loading a video. Save As lets you choose a project filename such as `My Shot.json`. Sessions include tracking settings, the active tab, Undistort inputs/options and the calibrated lens profile. If present, tracking data is kept in `My Shot.workspace`, and Roto frames, masks, refined mattes, points and editor settings are stored in `My Shot.roto.zip`. Opening the JSON restores the session without requiring tracking results. Keep these companion files/folders and any saved `mattes` folder together when moving a project; original source media remains externally referenced. Wait for active tracking/Roto processing to finish before saving or opening another session. Existing `project.json` projects remain supported.

## Notes / Known Limitations

- **Coordinate conversion**: The AE orientation is currently derived using XYZ Euler angles. Depending on your footage you may need to adjust the rotation order or add a null target. This is the area most likely to need tweaking.
- **Progress**: GLOMAP progress is parsed from `Processing image [X/Y]` style output; if your build prints differently, the bar will use stage-based increments instead.

## License

Use at your own risk. Bundled binaries (FFmpeg, COLMAP, GLOMAP) belong to their respective projects.
