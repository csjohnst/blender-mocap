# Blender Motion Capture Plugin — Design Spec

**Date:** 2026-03-14
**Target:** Blender 4.5 LTS on Arch Linux
**Platform scope:** Linux only (no Windows/macOS support planned)
**Status:** Approved

## Overview

A Blender addon that captures human motion from a webcam using MediaPipe pose estimation and applies it to a Rigify armature in real-time. Supports live preview, recording, post-capture smoothing, export to .blend/FBX/BVH, and synchronized audio capture from a webcam mic or system microphone.

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

- **Transport:** Unix socket at `/tmp/blender-mocap-{blender_pid}.sock` (uses Blender's PID for uniqueness). On startup, the subprocess manager removes any stale socket file at that path before binding.
- **Format:** JSON, one message per line (newline-delimited)
- **Handshake:** On connection, the server sends `{"type": "hello", "protocol_version": 1}`. The client checks the version and disconnects with an error if it doesn't match.
- **Server → Client messages:**
  - `{"type": "pose", "landmarks": [...], "timestamp": float}` — 33 MediaPipe landmarks, each with x, y, z, visibility
  - `{"type": "status", "state": "ready|capturing|error", "message": str}` — status updates
  - `{"type": "error", "message": str}` — error reporting
- **Client → Server messages:**
  - `{"type": "command", "action": "start_preview"}` — begin camera capture and pose estimation
  - `{"type": "command", "action": "stop_preview"}` — stop capture, close preview window
  - `{"type": "command", "action": "shutdown"}` — terminate the capture process cleanly
  - `{"type": "command", "action": "start_recording"}` — begin buffering audio (capture server starts writing audio to file)
  - `{"type": "command", "action": "stop_recording"}` — stop audio recording, finalize file
- **Direction:** Bidirectional. Data flows server→client, commands flow client→server. The addon sends `shutdown` on Stop or when Blender closes; the subprocess also exits if the socket disconnects (Blender crash cleanup).
- **Backpressure:** When the Blender timer drains the socket, it reads all available messages and discards all but the most recent pose. This ensures the addon always uses the freshest data and doesn't fall behind.
- **Liveness:** The capture server sends `{"type": "heartbeat"}` every 2 seconds when idle (no pose data flowing). The Blender addon considers the server dead if no message of any type is received for 5 seconds, at which point it cleans up and shows an error. The capture server detects Blender disconnect by a broken pipe or empty read on the socket and exits.

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
| 27, 29, 31 (ankle, heel, foot index) | foot.L | Foot orientation from heel→toe vector |
| 28, 30, 32 (ankle, heel, foot index) | foot.R | Foot orientation from heel→toe vector |
| 0 (nose), 7, 8 (ears) | spine.004–.006 (neck/head) | Head orientation |

**Foot/ankle orientation:** The foot bone rotation is derived from the heel-to-foot-index vector (landmarks 29→31 for left, 30→32 for right), giving natural toe direction during walking. The ankle pitch (dorsiflexion/plantarflexion) comes from the shin-to-ankle-to-toe angle. This prevents the "dangling feet" artifact common in basic pose-to-rig mappings.

**Out of scope:** Finger tracking and shoulder roll are not mapped in this version. Full body only — no hands or face.

### Coordinate System Transform

MediaPipe outputs landmarks in normalized image coordinates:
- X: 0.0 (left) to 1.0 (right)
- Y: 0.0 (top) to 1.0 (bottom)
- Z: depth relative to hip midpoint (negative = closer to camera)

Blender uses a right-handed coordinate system (X-right, Y-forward, Z-up). The transform:
- Blender X = MediaPipe X (scaled and centered)
- Blender Y = -MediaPipe Z (depth becomes forward/back)
- Blender Z = -MediaPipe Y (flip vertical, Y-up becomes Z-up)

### Rotation Calculation

For each bone chain:
1. Compute the direction vector between parent and child landmarks in Blender space
2. Get the bone's rest-pose direction vector from `bone.vector` (bone-local space)
3. Compute the rotation quaternion that transforms the rest vector to the target vector
4. Apply as `pose_bone.rotation_quaternion` in bone-local space (rotation mode set to QUATERNION)

All rotations are applied in bone-local space, which means each bone rotates relative to its parent — matching how Rigify expects pose data.

### Depth

MediaPipe provides Z-depth per landmark relative to the hip midpoint. This gives 3D rotations from a single camera, though Z accuracy is lower than XY.

## Capture Modes

### Live Preview

- Capture server sends pose data continuously
- Blender timer callback (~30Hz) polls the socket and applies poses to the armature
- No keyframes inserted — armature moves in real-time
- OpenCV window shows camera feed with skeleton overlay

### Recording

- User clicks Record — addon starts buffering raw landmark data with timestamps and sends `start_recording` command to capture server to begin audio capture
- Poses continue to be applied live during recording
- User clicks Stop — sends `stop_recording` to capture server (finalizes audio file), then buffer is baked into a Blender Action:
  - Raw landmarks converted to bone rotations
  - Keyframes inserted with FPS resampling: timestamps are converted to scene frames using `frame = timestamp * scene_fps`. If camera FPS differs from scene FPS, landmark data is linearly interpolated to align with scene frame boundaries. Linear interpolation is acceptable for the initial version; the post-capture F-curve smoothing pass addresses any resulting stiffness.
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

The UI exposes a single "Smoothing" slider (0.0–1.0). This maps to the one-euro filter's `min_cutoff` parameter: slider 0.0 = min_cutoff 1.0 (no smoothing), slider 1.0 = min_cutoff 0.05 (heavy smoothing). The `beta` (speed coefficient) is fixed at 0.5, which provides good responsiveness for human motion.

### Post-Capture (Blender Addon)

Blender's built-in F-curve smooth operator applied to baked keyframes. User-adjustable strength.

## Audio Capture

Audio is recorded alongside motion capture for later use in rendering.

### Audio Source Selection

- UI provides a dropdown of available audio input devices (populated via `sounddevice.query_devices()` filtered to inputs)
- Options include webcam microphones, USB mics, and system audio inputs
- Default: the system's default input device

### Recording

- Audio recording starts/stops in sync with motion recording via the `start_recording`/`stop_recording` IPC commands
- The capture server records audio on a separate thread using `sounddevice` (PortAudio bindings)
- Audio is saved as a WAV file alongside the motion data: `~/.blender-mocap/recordings/MoCap_001.wav`
- Sample rate: 44100 Hz, 16-bit, mono (sufficient for voice/dialogue)

### Synchronization

- Audio and pose recording are started by the same command, so they share a common start timestamp
- On bake, the addon reports the audio file path and its time offset relative to the Action's first frame
- The user can import the WAV into Blender's Video Sequence Editor or use it externally during render

### Audio File Management

- Audio files are listed alongside their corresponding Action in the Recordings panel
- Delete removes both the Action and the associated audio file
- Audio files persist at `~/.blender-mocap/recordings/` and survive Blender restarts

## Export

Three export formats for motion, plus audio:

1. **`.blend` Action** — uses `bpy.data.libraries.write()` to save the Action data block to an external `.blend` file. Can be appended/linked into other Blender projects via File → Append.
2. **FBX** — industry-standard exchange format. Exports armature + action.
3. **BVH** — motion capture standard. Exports skeletal animation data only.

4. **Audio (WAV)** — exports the associated audio file. Included automatically when exporting motion if an audio file exists for that recording.

Export operates on the selected recording from the Recordings list.

## UI Layout

Sidebar panel in 3D Viewport N-panel, category "Motion Capture". Four collapsible sections:

### Setup
- Camera dropdown — auto-detects `/dev/video*` devices
- Target Armature — object picker filtered to armatures that have a `rig_id` custom property (set by Rigify's Generate Rig operator)
- Audio Source dropdown — auto-detects input devices (mics), default: system default input
- Smoothing slider — real-time filter strength (0.0–1.0)

### Capture
- Start Preview button — launches capture subprocess, opens OpenCV window
- Record button — begins recording (disabled until preview is active)
- Stop button — stops recording or preview
- Status label — Idle / Previewing / Recording (frame count)

### Recordings
- List of captured Actions with frame counts and audio indicator
- Smooth button — apply post-capture smoothing to selected recording
- Delete button — remove selected recording (and associated audio file)

### Export
- Four buttons: .blend, FBX, BVH, Audio (WAV)
- Opens file browser for save location
- Audio file automatically copied alongside motion export when available

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
│       ├── audio.py                # Audio capture via sounddevice
│       └── requirements.txt        # mediapipe, opencv-python, numpy, sounddevice
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
- `sounddevice` — audio capture (PortAudio bindings)

### Blender Addon
- Pure Python, no external dependencies
- Uses Blender's bundled `bpy`, `mathutils`

## Installation

1. User installs addon .zip via Edit → Preferences → Add-ons → Install
2. On first enable, addon checks for venv at `~/.blender-mocap/venv/`
3. If missing, creates venv using system `python3` and installs `capture_server/requirements.txt`

**Subprocess launch command:** The addon launches the capture server as:
```
~/.blender-mocap/venv/bin/python -m blender_mocap.capture_server --socket /tmp/blender-mocap-{pid}.sock --camera {device_index}
```
The `capture_server/` directory is located via `os.path.dirname(__file__)` from the addon, and its parent is added to `PYTHONPATH` for the subprocess.

**Venv upgrades:** The addon writes a version marker file (`~/.blender-mocap/venv/.addon-version`) containing the addon version string. On enable, if the marker doesn't match the current addon version, the venv is recreated.
4. Panel appears in 3D Viewport N-sidebar

### Requirements
- Blender 4.5 LTS
- System Python 3.10–3.12 (required by MediaPipe). On first run, the addon checks `python3 --version` and shows an error if the version is outside this range.
- Webcam accessible at `/dev/video*`

## Error Handling

- **No camera found:** Panel shows warning, Start Preview disabled
- **Venv creation fails:** Error dialog with instructions to install python3-venv
- **Capture process crashes:** Addon detects socket disconnect, shows error in status, cleans up
- **No armature selected:** Record disabled, prompt to select Rigify armature
- **MediaPipe fails to detect pose:** Preview continues, armature holds last known pose
- **No recording selected on export:** Export buttons disabled when no recording is selected
- **Addon unregister/Blender close:** `unregister()` sends shutdown command to capture server and reaps the subprocess. If subprocess doesn't exit within 3 seconds, it is killed via SIGKILL.
