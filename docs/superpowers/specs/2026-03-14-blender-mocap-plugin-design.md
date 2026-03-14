# Blender Motion Capture Plugin — Design Spec

**Date:** 2026-03-14
**Target:** Blender 4.5 LTS on Arch Linux
**Status:** Approved

## Overview

A Blender addon that captures human motion from a webcam using MediaPipe pose estimation and applies it to a Rigify armature in real-time. Supports live preview, recording, post-capture smoothing, and export to .blend, FBX, and BVH formats.

## Architecture

### Subprocess Model

Two-process architecture for isolation and performance:

1. **Blender Addon** (`blender_mocap/`) — runs inside Blender's Python. Handles UI, armature manipulation, recording, and export. Communicates with the capture process via a Unix socket.

2. **Capture Server** (`blender_mocap/capture_server/`) — runs as a separate Python subprocess with its own venv. Handles webcam capture (OpenCV), pose estimation (MediaPipe), preview window, and jitter smoothing. Sends landmark data as JSON over the socket.

### Why Subprocess

- Avoids GIL contention — capture runs at full speed independent of Blender
- No need to install MediaPipe/OpenCV into Blender's bundled Python
- Crash isolation — camera failure won't take down Blender
- Clean dependency boundary

### IPC Protocol

- **Transport:** Unix socket at `/tmp/blender-mocap-{pid}.sock`
- **Format:** JSON, one message per line
- **Messages:**
  - `{"type": "pose", "landmarks": [...], "timestamp": float}` — 33 MediaPipe landmarks, each with x, y, z, visibility
  - `{"type": "status", "state": "ready|capturing|error", "message": str}` — status updates
  - `{"type": "error", "message": str}` — error reporting
- **Direction:** Capture server → Blender addon (unidirectional data flow). Blender sends control commands (start/stop) via subprocess signals.

## Rigify Bone Mapping

MediaPipe Pose produces 33 landmarks. These map to Rigify bones as follows:

| MediaPipe Landmarks | Rigify Bone(s) | Method |
|---|---|---|
| 11, 12 (shoulders) | spine, spine.001–.003 | Torso orientation from shoulder/hip vectors |
| 23, 24 (hips) | torso (root) | Hip center position for root motion |
| 11 → 13 → 15 | upper_arm.L, forearm.L, hand.L | Joint chain rotation |
| 12 → 14 → 16 | upper_arm.R, forearm.R, hand.R | Joint chain rotation |
| 23 → 25 → 27 | thigh.L, shin.L, foot.L | Joint chain rotation |
| 24 → 26 → 28 | thigh.R, shin.R, foot.R | Joint chain rotation |
| 0 (nose), 7, 8 (ears) | spine.004–.006 (neck/head) | Head orientation |

### Rotation Calculation

For each bone chain: compute the vector between parent and child landmarks, convert to a quaternion rotation relative to the bone's rest pose. Applied as pose bone rotations.

### Depth

MediaPipe provides Z-depth per landmark relative to the hip midpoint. This gives 3D rotations from a single camera, though Z accuracy is lower than XY.

## Capture Modes

### Live Preview

- Capture server sends pose data continuously
- Blender timer callback (~30Hz) polls the socket and applies poses to the armature
- No keyframes inserted — armature moves in real-time
- OpenCV window shows camera feed with skeleton overlay

### Recording

- User clicks Record — addon starts buffering raw landmark data with timestamps
- Poses continue to be applied live during recording
- User clicks Stop — buffer is baked into a Blender Action:
  - Raw landmarks converted to bone rotations
  - Keyframes inserted at appropriate frames (camera FPS mapped to scene FPS)
  - Action auto-named `MoCap_001`, `MoCap_002`, etc.

### Post-Processing

- Optional smoothing pass on baked keyframes using Blender's F-curve smoothing operators
- Smoothing strength controlled by addon slider (0.0–1.0)

## Smoothing

### Real-Time (Capture Server)

One-euro filter applied to landmarks before sending to Blender. This is an adaptive low-pass filter that:
- Reduces jitter on slow movements (high smoothing)
- Preserves sharp movements (low smoothing)
- Industry standard for real-time motion data

### Post-Capture (Blender Addon)

Blender's built-in F-curve smooth operator applied to baked keyframes. User-adjustable strength.

## Export

Three export formats:

1. **`.blend` Action** — saves the Action data block. Can be appended/linked into other Blender projects via File → Append.
2. **FBX** — industry-standard exchange format. Exports armature + action.
3. **BVH** — motion capture standard. Exports skeletal animation data only.

Export operates on the selected recording from the Recordings list.

## UI Layout

Sidebar panel in 3D Viewport N-panel, category "Motion Capture". Four collapsible sections:

### Setup
- Camera dropdown — auto-detects `/dev/video*` devices
- Target Armature — object picker (filtered to Rigify armatures)
- Smoothing slider — real-time filter strength (0.0–1.0)

### Capture
- Start Preview button — launches capture subprocess, opens OpenCV window
- Record button — begins recording (disabled until preview is active)
- Stop button — stops recording or preview
- Status label — Idle / Previewing / Recording (frame count)

### Recordings
- List of captured Actions with frame counts
- Smooth button — apply post-capture smoothing to selected recording
- Delete button — remove selected recording

### Export
- Three buttons: .blend, FBX, BVH
- Opens file browser for save location

## Project Structure

```
blender-mocap/
├── blender_mocap/                  # Blender addon
│   ├── __init__.py                 # bl_info, register/unregister
│   ├── operators.py                # Start/Stop/Record/Export operators
│   ├── panels.py                   # N-panel UI
│   ├── properties.py               # Addon properties
│   ├── ipc_client.py               # Unix socket client
│   ├── rigify_mapper.py            # Landmarks → Rigify rotations
│   ├── recording.py                # Frame buffer, Action baking
│   ├── export.py                   # .blend/FBX/BVH export
│   ├── subprocess_manager.py       # Venv + subprocess lifecycle
│   └── capture_server/             # Runs as subprocess
│       ├── __main__.py             # Entry point
│       ├── camera.py               # OpenCV VideoCapture
│       ├── pose_estimator.py       # MediaPipe Pose
│       ├── preview.py              # OpenCV skeleton overlay window
│       ├── ipc_server.py           # Unix socket server
│       ├── smoothing.py            # One-euro filter
│       └── requirements.txt        # mediapipe, opencv-python, numpy
├── tests/
│   ├── test_rigify_mapper.py
│   ├── test_ipc.py
│   ├── test_smoothing.py
│   └── test_export.py
└── docs/
```

## Dependencies

### Capture Server (venv)
- `mediapipe` — pose estimation
- `opencv-python` — webcam capture and preview
- `numpy` — math operations

### Blender Addon
- Pure Python, no external dependencies
- Uses Blender's bundled `bpy`, `mathutils`

## Installation

1. User installs addon .zip via Edit → Preferences → Add-ons → Install
2. On first enable, addon checks for venv at `~/.blender-mocap/venv/`
3. If missing, creates venv using system `python3` and installs `capture_server/requirements.txt`
4. Panel appears in 3D Viewport N-sidebar

### Requirements
- Blender 4.5 LTS
- System Python 3 (pre-installed on Arch Linux)
- Webcam accessible at `/dev/video*`

## Error Handling

- **No camera found:** Panel shows warning, Start Preview disabled
- **Venv creation fails:** Error dialog with instructions to install python3-venv
- **Capture process crashes:** Addon detects socket disconnect, shows error in status, cleans up
- **No armature selected:** Record disabled, prompt to select Rigify armature
- **MediaPipe fails to detect pose:** Preview continues, armature holds last known pose
