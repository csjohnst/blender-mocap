# Blender Motion Capture

A Blender 4.5 LTS addon that captures human motion from a webcam using MediaPipe pose estimation and applies it to a Rigify armature in real-time. Record animations with synchronized audio and export to .blend, FBX, BVH, and WAV.

## Features

- **Live preview** — see your pose applied to a Rigify rig in the viewport as you move
- **Real-time pose estimation** — MediaPipe detects 33 body landmarks from a single webcam
- **Rigify integration** — automatic mapping to Rigify armature bones including foot/ankle orientation
- **Audio capture** — record synchronized audio from any microphone alongside motion
- **Jitter reduction** — one-euro adaptive filter for smooth real-time motion, plus post-capture F-curve smoothing
- **Multiple export formats** — .blend (Action), FBX, BVH, and WAV
- **Subprocess architecture** — capture runs in an isolated process so camera issues won't crash Blender

## Requirements

- Blender 4.5 LTS
- Linux (tested on Arch Linux)
- System Python 3.10–3.12
- Webcam
- PortAudio (`sudo pacman -S portaudio` on Arch)

## Quick Start

```bash
# Build the addon
./scripts/build_addon.sh
```

1. In Blender: **Edit → Preferences → Add-ons → Install** → select `dist/blender_mocap.zip`
2. Enable **"Motion Capture"** (first run installs dependencies automatically)
3. Add a Rigify armature: **Add → Armature → Human (Meta-Rig)** → **Generate Rig**
4. Open the **N-panel** → **Motion Capture** tab
5. Select your camera and armature, click **Start Preview**
6. Click **Record** to capture, **Stop** to finish
7. Export in your preferred format

See [`docs/quickstart.md`](docs/quickstart.md) for the full user guide.

## Architecture

Two-process design for performance and stability:

```
┌─────────────────────────┐       Unix Socket       ┌──────────────────────────┐
│     BLENDER ADDON       │◄──── JSON messages ────►│    CAPTURE SERVER        │
│                         │                          │    (subprocess)          │
│  • UI Panel             │                          │  • OpenCV camera         │
│  • Rigify bone mapping  │                          │  • MediaPipe pose        │
│  • Recording / baking   │                          │  • Preview window        │
│  • Export               │                          │  • One-euro smoothing    │
│                         │                          │  • Audio recording       │
└─────────────────────────┘                          └──────────────────────────┘
```

The capture server runs in its own Python venv (`~/.blender-mocap/venv/`) with MediaPipe, OpenCV, and sounddevice — no contamination of Blender's bundled Python.

## Development

```bash
# Run tests (requires numpy in your environment)
python3 -m pytest tests/ --ignore=tests/test_addon_load.py -v

# Run Blender integration test
blender --background --python tests/test_addon_load.py

# Build installable zip
./scripts/build_addon.sh
```

## Project Structure

```
blender_mocap/                  # Blender addon
├── __init__.py                 # Registration
├── properties.py               # UI properties
├── panels.py                   # N-panel layout
├── operators.py                # Blender operators
├── ipc_client.py               # Socket client
├── rigify_mapper.py            # Landmark → bone rotation
├── recording.py                # Frame buffer + action baking
├── export.py                   # .blend/FBX/BVH/WAV export
├── subprocess_manager.py       # Venv + process lifecycle
└── capture_server/             # Runs as subprocess
    ├── __main__.py             # Entry point
    ├── camera.py               # OpenCV capture
    ├── pose_estimator.py       # MediaPipe wrapper
    ├── preview.py              # Skeleton overlay window
    ├── ipc_server.py           # Socket server
    ├── smoothing.py            # One-euro filter
    └── audio.py                # Audio recording
```

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

Blender addons that use the `bpy` module are considered derivative works of Blender, which is licensed under GPL v2+. GPL v3 is the standard license for Blender addons.
