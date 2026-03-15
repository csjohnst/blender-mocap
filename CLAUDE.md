# Blender Motion Capture Plugin — Development Guide

## Project Overview

Blender addon for real-time webcam motion capture using MediaPipe pose estimation, targeting Rigify armatures. Runs on Linux (Arch), Blender 4.5+ LTS.

**Repo:** https://github.com/csjohnst/blender-mocap
**License:** GPL v3 (required for Blender addons using bpy)

## Architecture

Two-process subprocess model:

```
BLENDER PROCESS                          CAPTURE SUBPROCESS
┌─────────────────────┐   Unix Socket   ┌──────────────────────┐
│ blender_mocap/       │◄──── JSON ────►│ capture_server/      │
│  operators.py        │                │  __main__.py         │
│  rigify_mapper.py    │                │  camera.py (OpenCV)  │
│  recording.py        │                │  pose_estimator.py   │
│  panels.py           │                │  preview.py          │
│  properties.py       │                │  smoothing.py        │
│  ipc_client.py       │                │  ipc_server.py       │
│  export.py           │                │  audio.py            │
│  subprocess_manager.py│                │  requirements.txt    │
└─────────────────────┘                └──────────────────────┘
```

**Why subprocess:** Blender's bundled Python can't easily install MediaPipe/OpenCV. The capture server runs in its own venv at `~/.blender-mocap/venv/`. Crash isolation — camera issues don't take down Blender. No GIL contention.

## Key Technical Decisions & Lessons Learned

### Bone Rotation Approach (FK, not IK)

**Current approach (v0.27.0+):** World-space direction delta.
- For each bone, store its direction at calibration
- Each frame: `rotation_quaternion = calib_dir.rotation_difference(current_dir)`
- No bone-local conversion — the calibration delta approach cancels the rest orientation

**What was tried and didn't work:**
1. `bone.matrix_local` conversion (v0.5-v0.6): Correct for root bones but wrong for chain children — the conversion frame doesn't account for parent rotation
2. `rest_local` (parent^-1 @ bone) conversion: Maps elbow bend to forearm twist axis
3. Chain-aware with tracked parent matrices (v0.7): Over-complicated, stale parent matrices from no depsgraph update between bone sets
4. `pose_bone.matrix` direct set (v0.6): Blender uses stale parent `pose_mat` — child decomposition is wrong
5. Joint angle approach (v0.15-v0.16): Conjugation mapped bend to wrong local axis
6. IK-based positioning (v0.24): Reverted — user preferred FK approach

**The fundamental issue with FK chains:** `rotation_quaternion` for a child bone is relative to the parent's CURRENT orientation. Without a depsgraph update between setting parent and child, the parent's `pose_mat` is stale. The simple world-space delta approach sidesteps this entirely.

### Depth Estimation

MediaPipe's Z (depth) from a single camera is unreliable. Two techniques are used:

1. **Trigonometric depth for bone directions (v0.26.0):** At calibration, store each bone's 2D image-plane length. Each frame, if apparent length < calibration length, the bone rotated out-of-plane: `depth = sqrt(calib_length² - apparent_length²)`. Sign from MediaPipe Z (direction is OK, magnitude isn't).

2. **Forward lean from torso foreshortening (v0.25.0):** `cos(lean_angle) = current_torso_height / calib_torso_height`. When leaning forward, torso appears shorter in the image.

3. **Torso size ratio for root depth:** `depth_ratio = calib_torso_size / current_torso_size`. Closer = bigger apparent size.

### Body Rotation

**Critical lesson:** Do NOT use MediaPipe Z for body rotation sign. The Z values flip suddenly when the head turns, causing instant 180° body rotation. Use ONLY the image-plane shoulder/hip line angle.

Per-bone velocity limits prevent unnatural motion:
- Torso: 15°/frame, 80% smoothing, 10° dead zone
- Head: 30°/frame, 60% smoothing
- Limbs: 90°/frame, user-controlled smoothing

### Calibration System

User stands in A-pose, clicks "Calibrate / Reset":
- Stores bone directions, body angle, torso metrics, 2D bone lengths
- All rotations computed as DELTA from calibration
- Model at rest when matching calibration pose
- Auto-calibrates on first frame if not manually calibrated

### Rigify Integration

**Generated rig bone names (NOT metarig names):**
- Arms: `upper_arm_fk.L`, `forearm_fk.L`, `hand_fk.L`
- Legs: `thigh_fk.L`, `shin_fk.L`, `foot_fk.L`
- IK/FK switch: property `IK_FK` on `upper_arm_parent.L`, `thigh_parent.L` etc.
- Spine: `chest`, `head`, `torso`

**FK mode required:** Set `IK_FK = 1.0` on parent bones to switch from IK to FK.

### MediaPipe API

MediaPipe removed `mediapipe.solutions` in v0.10.30. Uses the Tasks API (`PoseLandmarker`). Model file auto-downloaded to `~/.blender-mocap/pose_landmarker_lite.task`.

## Build & Test

```bash
# Run standalone tests (no Blender needed)
python3 -m venv /tmp/test-venv
/tmp/test-venv/bin/pip install pytest numpy
/tmp/test-venv/bin/python -m pytest tests/ --ignore=tests/test_addon_load.py -v

# Run Blender integration test
blender --background --python tests/test_addon_load.py

# Build installable zip
./scripts/build_addon.sh
# Output: dist/blender_mocap.zip
```

## Version Management

**CRITICAL:** Always bump version in BOTH files when building a new zip:
- `blender_mocap/__init__.py` → `bl_info["version"]`
- `blender_mocap/subprocess_manager.py` → `ADDON_VERSION`

The version bump triggers venv recreation (picks up dependency changes) and is shown in the Blender panel header for verification. Blender caches `.pyc` files — the `__init__.py` clears `__pycache__` on import to prevent stale code.

## Coordinate System

```
MediaPipe:                    Blender (character faces -Y):
  X: 0→1 (left to right)       X: right (+) / left (-)
  Y: 0→1 (top to bottom)       Y: forward (-) / back (+)
  Z: depth (neg=closer)        Z: up (+) / down (-)

Transform:
  bx = lm.x - 0.5              (center at 0)
  by = lm.z                    (closer = -Y = toward camera)
  bz = -(lm.y - 0.5)           (flip vertical)
```

The character faces -Y (toward the default camera). Webcam is NOT mirrored (OpenCV default). MediaPipe "left" = subject's left = camera's right = positive X.

## IPC Protocol

Bidirectional JSON over Unix socket at `/tmp/blender-mocap-{pid}.sock`:
- Server→Client: `pose`, `status`, `error`, `heartbeat`, `hello`
- Client→Server: `command` (start_preview, stop_preview, start_recording, stop_recording, shutdown)
- Handshake: server sends `{"type": "hello", "protocol_version": 1}`
- Backpressure: client drains all messages, keeps only latest pose
- Liveness: 2s heartbeat interval, 5s timeout

## File Structure

```
blender_mocap/
├── __init__.py              # bl_info, register/unregister, __pycache__ cleanup
├── operators.py             # Blender operators (preview, record, calibrate, export)
├── panels.py                # N-panel UI (setup, capture, recordings, export)
├── properties.py            # Scene properties, camera/audio device enumeration
├── rigify_mapper.py         # MediaPipe→Rigify rotation mapping (THE HARD PART)
├── recording.py             # Frame buffer, FPS resampling, action baking
├── export.py                # .blend/FBX/BVH/WAV export
├── ipc_client.py            # Unix socket client
├── subprocess_manager.py    # Venv bootstrap, subprocess lifecycle
└── capture_server/
    ├── __init__.py
    ├── __main__.py           # CLI entry point, main capture loop (30fps throttled)
    ├── camera.py             # OpenCV VideoCapture with V4L2 device detection
    ├── pose_estimator.py     # MediaPipe Tasks PoseLandmarker
    ├── preview.py            # OpenCV skeleton overlay window
    ├── ipc_server.py         # Unix socket server
    ├── smoothing.py          # One-euro adaptive filter
    ├── audio.py              # sounddevice WAV recorder
    └── requirements.txt      # mediapipe>=0.10.30, opencv-python, numpy, sounddevice
```

## Known Issues & Ongoing Work

1. **Elbow/knee bending:** World-space delta approach produces visible bending but accuracy varies. The calibration-relative delta partially double-counts parent rotation for chain children.

2. **Root motion:** Uses hip midpoint X (lateral) and torso size ratio (depth). Vertical from lowest foot. Walking moves the root but one-leg stands may drift slightly.

3. **Recording/baking:** Uses the legacy `compute_limb_rotations` function which has simpler rotation math. Baked actions may not perfectly match live preview.

4. **Camera compatibility:** V4L2 `device_caps` check filters metadata nodes. Camera opened by path (`/dev/videoN`) not index. Permission checks suggest `usermod -aG video`.

5. **Blender 5.0:** Tested on user's machine running Blender 5.0, despite targeting 4.5 LTS. No API issues found.

## Common Debugging

- **Check version:** Panel header shows version number. If wrong, remove addon, restart Blender, reinstall.
- **Capture server crashes:** Check Blender console — `[MoCap]` prefix shows stderr.
- **FK switch not working:** Console shows `[MoCap] === FK SWITCH ===` with property names found.
- **Bone cache:** Console shows `[MoCap] Bone cache: N bones found` with each bone's rest vector.
- **Calibration:** Console shows `[MoCap] === CALIBRATION ===` with all bone directions.
