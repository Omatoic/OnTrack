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

1. Make sure Python 3.10+ is installed.
2. Open a terminal in this folder.
3. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python main.py
```

Or double-click `run.bat`.

## Usage

1. Add a video by clicking **Add Video** or dragging and dropping a video file onto the window.
2. Review the video preview and metadata (resolution, duration, FPS, frame count). The FPS field auto-populates from the video.
3. Confirm the FFmpeg, COLMAP, and GLOMAP binary paths.
4. Set FPS, optional max frame size, and frame skip.
5. Click **Run Tracking**.
6. When complete, export to **After Effects (.jsx)** or **Blender (.py)**.

## Notes / Known Limitations

- **Coordinate conversion**: The AE orientation is currently derived using XYZ Euler angles. Depending on your footage you may need to adjust the rotation order or add a null target. This is the area most likely to need tweaking.
- **Cleanup**: Enable *Clean workspace on exit* to delete the temporary frames/database/sparse folder when the app closes.
- **Progress**: GLOMAP progress is parsed from `Processing image [X/Y]` style output; if your build prints differently, the bar will use stage-based increments instead.

## License

Use at your own risk. Bundled binaries (FFmpeg, COLMAP, GLOMAP) belong to their respective projects.
