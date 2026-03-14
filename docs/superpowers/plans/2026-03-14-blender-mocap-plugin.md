# Blender Motion Capture Plugin Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Blender 4.5 LTS addon that captures webcam video, estimates human pose via MediaPipe, maps it to a Rigify armature in real-time, records animations with synchronized audio, and exports to .blend/FBX/BVH/WAV.

**Architecture:** Two-process model — a Blender addon handles UI, armature, recording, and export, while a subprocess (in its own venv) handles webcam capture, MediaPipe pose estimation, audio recording, and preview. They communicate via a bidirectional Unix socket with JSON messages.

**Tech Stack:** Python, Blender 4.5 LTS API (bpy), MediaPipe, OpenCV, sounddevice, numpy. Tests run via pytest (capture server) and blender --background --python (addon).

---

## File Structure

```
blender-mocap/
├── blender_mocap/                     # Blender addon (installed via .zip)
│   ├── __init__.py                    # bl_info, register/unregister, version
│   ├── properties.py                  # Scene/addon property definitions
│   ├── panels.py                      # N-panel UI layout
│   ├── operators.py                   # Blender operators (preview, record, export, smooth)
│   ├── ipc_client.py                  # Unix socket client — connects to capture server
│   ├── rigify_mapper.py               # MediaPipe landmarks → Rigify bone rotations
│   ├── recording.py                   # Frame buffer, Action baking, FPS resampling
│   ├── export.py                      # .blend/FBX/BVH/WAV export logic
│   ├── subprocess_manager.py          # Venv bootstrap, subprocess launch/kill
│   └── capture_server/                # Subprocess package (bundled in addon)
│       ├── __init__.py                # Package marker
│       ├── __main__.py                # CLI entry point, arg parsing, main loop
│       ├── camera.py                  # OpenCV VideoCapture wrapper
│       ├── pose_estimator.py          # MediaPipe Pose wrapper
│       ├── preview.py                 # OpenCV window with skeleton overlay
│       ├── ipc_server.py              # Unix socket server — sends/receives JSON
│       ├── smoothing.py               # One-euro filter
│       ├── audio.py                   # Audio capture via sounddevice
│       └── requirements.txt           # mediapipe, opencv-python, numpy, sounddevice
├── tests/
│   ├── test_smoothing.py              # One-euro filter unit tests
│   ├── test_ipc.py                    # IPC protocol integration tests
│   ├── test_rigify_mapper.py          # Coordinate transform + rotation tests
│   ├── test_recording.py             # Frame buffer + FPS resampling tests
│   ├── test_export.py                 # Export format tests
│   ├── test_audio.py                  # Audio capture tests
│   └── test_subprocess_manager.py     # Venv + lifecycle tests
├── .gitignore
└── docs/
```

---

## Chunk 1: Foundation — IPC, Smoothing, Camera, Pose Estimation

These are the core building blocks that everything else depends on. All are standalone modules testable without Blender.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `blender_mocap/__init__.py`
- Create: `blender_mocap/capture_server/__init__.py`
- Create: `blender_mocap/capture_server/requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.superpowers/
*.blend1
*.blend2
.venv/
venv/
```

- [ ] **Step 2: Create addon __init__.py with bl_info stub**

```python
bl_info = {
    "name": "Motion Capture",
    "author": "Chris",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Motion Capture",
    "description": "Webcam motion capture with MediaPipe pose estimation",
    "category": "Animation",
}


def register():
    pass


def unregister():
    pass
```

- [ ] **Step 3: Create capture_server __init__.py**

Empty file — package marker only.

- [ ] **Step 4: Create requirements.txt**

```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
sounddevice>=0.4.6
```

- [ ] **Step 5: Create tests/__init__.py**

Empty file.

- [ ] **Step 6: Commit**

```bash
git add .gitignore blender_mocap/ tests/
git commit -m "feat: project scaffolding with bl_info and capture server package"
```

---

### Task 2: One-Euro Smoothing Filter

**Files:**
- Create: `blender_mocap/capture_server/smoothing.py`
- Create: `tests/test_smoothing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_smoothing.py
import math
import pytest
from blender_mocap.capture_server.smoothing import OneEuroFilter, LandmarkSmoother


class TestOneEuroFilter:
    def test_first_value_passes_through(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        result = f(0.0, 1.0)
        assert result == 1.0

    def test_constant_input_converges(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        for t in range(100):
            result = f(t * 0.033, 5.0)
        assert abs(result - 5.0) < 0.01

    def test_smoothing_reduces_jitter(self):
        f = OneEuroFilter(min_cutoff=0.05, beta=0.5)
        # Feed noisy signal around 1.0
        import random
        random.seed(42)
        values = [1.0 + random.gauss(0, 0.1) for _ in range(100)]
        raw_var = sum((v - 1.0) ** 2 for v in values) / len(values)
        filtered = []
        for i, v in enumerate(values):
            filtered.append(f(i * 0.033, v))
        filt_var = sum((v - 1.0) ** 2 for v in filtered[10:]) / len(filtered[10:])
        assert filt_var < raw_var * 0.5  # At least 50% variance reduction

    def test_fast_movement_preserved(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        f(0.0, 0.0)
        f(0.033, 0.0)
        # Sudden jump
        result = f(0.066, 10.0)
        # Should follow quickly (not lag behind significantly)
        assert result > 5.0


class TestLandmarkSmoother:
    def test_smooths_landmark_list(self):
        smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.5, num_landmarks=2)
        lm1 = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0},
               {"x": 0.3, "y": 0.7, "z": 0.1, "visibility": 0.9}]
        result = smoother(0.0, lm1)
        assert len(result) == 2
        # First call passes through
        assert result[0]["x"] == pytest.approx(0.5)
        assert result[1]["y"] == pytest.approx(0.7)
        # Visibility is not smoothed
        assert result[0]["visibility"] == 1.0

    def test_updates_min_cutoff(self):
        smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.5, num_landmarks=1)
        smoother.update_min_cutoff(0.05)
        lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}]
        result = smoother(0.0, lm)
        assert result[0]["x"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_smoothing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the one-euro filter**

```python
# blender_mocap/capture_server/smoothing.py
"""One-Euro Filter for real-time landmark smoothing.

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""
import math


class OneEuroFilter:
    """Adaptive low-pass filter that reduces jitter while preserving fast movements."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-6

        # Derivative estimation
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat


class LandmarkSmoother:
    """Applies one-euro filtering to a list of MediaPipe landmarks (x, y, z each)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, num_landmarks: int = 33):
        self._num_landmarks = num_landmarks
        self._min_cutoff = min_cutoff
        self._beta = beta
        # 3 filters per landmark (x, y, z)
        self._filters: list[list[OneEuroFilter]] = [
            [OneEuroFilter(min_cutoff, beta) for _ in range(3)]
            for _ in range(num_landmarks)
        ]

    def update_min_cutoff(self, min_cutoff: float) -> None:
        self._min_cutoff = min_cutoff
        for landmark_filters in self._filters:
            for f in landmark_filters:
                f.min_cutoff = min_cutoff

    def __call__(self, t: float, landmarks: list[dict]) -> list[dict]:
        result = []
        for i, lm in enumerate(landmarks):
            fx, fy, fz = self._filters[i]
            result.append({
                "x": fx(t, lm["x"]),
                "y": fy(t, lm["y"]),
                "z": fz(t, lm["z"]),
                "visibility": lm["visibility"],  # Not smoothed
            })
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_smoothing.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add blender_mocap/capture_server/smoothing.py tests/test_smoothing.py
git commit -m "feat: one-euro smoothing filter with landmark smoother"
```

---

### Task 3: IPC Protocol — Server and Client

**Files:**
- Create: `blender_mocap/capture_server/ipc_server.py`
- Create: `blender_mocap/ipc_client.py`
- Create: `tests/test_ipc.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ipc.py
import json
import os
import tempfile
import threading
import time
import pytest
from blender_mocap.capture_server.ipc_server import IPCServer
from blender_mocap.ipc_client import IPCClient


@pytest.fixture
def socket_path():
    path = os.path.join(tempfile.mkdtemp(), "test.sock")
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestIPCProtocol:
    def test_handshake(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            hello = client.read_message()
            assert hello["type"] == "hello"
            assert hello["protocol_version"] == 1
            client.close()
        finally:
            server.stop()

    def test_server_sends_pose(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            pose = {"type": "pose", "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}], "timestamp": 1.0}
            server.send(pose)
            msg = client.read_message()
            assert msg["type"] == "pose"
            assert msg["landmarks"][0]["x"] == 0.5
            client.close()
        finally:
            server.stop()

    def test_client_sends_command(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            client.send_command("start_preview")
            msg = server.read_command(timeout=2.0)
            assert msg["type"] == "command"
            assert msg["action"] == "start_preview"
            client.close()
        finally:
            server.stop()

    def test_backpressure_keeps_latest(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            # Send multiple poses rapidly
            for i in range(10):
                server.send({"type": "pose", "landmarks": [], "timestamp": float(i)})
            time.sleep(0.1)  # Let them buffer

            # Drain should return only the latest pose, plus other messages
            latest, others = client.drain_latest_pose()
            assert latest is not None
            assert latest["timestamp"] == 9.0
            client.close()
        finally:
            server.stop()

    def test_stale_socket_cleanup(self, socket_path):
        # Create a stale socket file
        os.makedirs(os.path.dirname(socket_path), exist_ok=True)
        with open(socket_path, "w") as f:
            f.write("stale")
        # Server should remove it and bind successfully
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            hello = client.read_message()
            assert hello["type"] == "hello"
            client.close()
        finally:
            server.stop()

    def test_heartbeat(self, socket_path):
        server = IPCServer(socket_path)
        server.start()
        try:
            client = IPCClient(socket_path)
            client.connect()
            client.read_message()  # handshake

            server.send_heartbeat()
            msg = client.read_message()
            assert msg["type"] == "heartbeat"
            client.close()
        finally:
            server.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_ipc.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement IPC server**

```python
# blender_mocap/capture_server/ipc_server.py
"""Unix socket server for the capture process. Sends pose data, receives commands."""
import json
import os
import socket
import select
import threading
import queue


PROTOCOL_VERSION = 1


class IPCServer:
    """Newline-delimited JSON server over a Unix socket."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._running = False
        self._accept_thread: threading.Thread | None = None
        self._command_queue: queue.Queue = queue.Queue()
        self._recv_buffer = ""

    def start(self) -> None:
        # Remove stale socket
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen(1)
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        self._server_sock.settimeout(1.0)
        while self._running:
            try:
                client, _ = self._server_sock.accept()
                self._client_sock = client
                self._client_sock.setblocking(False)
                # Send handshake
                self._raw_send({"type": "hello", "protocol_version": PROTOCOL_VERSION})
                self._read_commands_loop()
            except socket.timeout:
                continue
            except OSError:
                break

    def _read_commands_loop(self) -> None:
        while self._running and self._client_sock:
            try:
                ready, _, _ = select.select([self._client_sock], [], [], 0.5)
                if ready:
                    data = self._client_sock.recv(4096)
                    if not data:
                        break  # Client disconnected
                    self._recv_buffer += data.decode("utf-8")
                    while "\n" in self._recv_buffer:
                        line, self._recv_buffer = self._recv_buffer.split("\n", 1)
                        if line.strip():
                            msg = json.loads(line)
                            self._command_queue.put(msg)
            except (OSError, ConnectionError):
                break

    def _raw_send(self, msg: dict) -> None:
        if self._client_sock:
            data = json.dumps(msg) + "\n"
            self._client_sock.sendall(data.encode("utf-8"))

    def send(self, msg: dict) -> None:
        self._raw_send(msg)

    def send_heartbeat(self) -> None:
        self.send({"type": "heartbeat"})

    def read_command(self, timeout: float = 0.0) -> dict | None:
        try:
            return self._command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_client(self) -> bool:
        return self._client_sock is not None

    def stop(self) -> None:
        self._running = False
        if self._client_sock:
            try:
                self._client_sock.close()
            except OSError:
                pass
            self._client_sock = None
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        if self._accept_thread:
            self._accept_thread.join(timeout=3.0)
```

- [ ] **Step 4: Implement IPC client**

```python
# blender_mocap/ipc_client.py
"""Unix socket client for the Blender addon. Reads pose data, sends commands."""
import json
import socket
import select


class IPCClient:
    """Connects to the capture server's Unix socket."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._sock: socket.socket | None = None
        self._buffer = ""

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)
        self._sock.setblocking(False)

    def read_message(self, timeout: float = 5.0) -> dict | None:
        """Read a single JSON message. Blocks up to timeout."""
        if not self._sock:
            return None
        deadline = timeout
        while deadline > 0:
            # Check buffer first
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    return json.loads(line)
            ready, _, _ = select.select([self._sock], [], [], min(deadline, 0.5))
            if ready:
                data = self._sock.recv(8192)
                if not data:
                    return None  # Server disconnected
                self._buffer += data.decode("utf-8")
            deadline -= 0.5
        return None

    def drain_latest_pose(self) -> tuple[dict | None, list[dict]]:
        """Read all available messages. Returns (latest_pose, other_messages).

        other_messages includes heartbeats, status, errors — needed for liveness tracking.
        """
        if not self._sock:
            return None, []
        # Read all available data
        while True:
            ready, _, _ = select.select([self._sock], [], [], 0)
            if not ready:
                break
            data = self._sock.recv(65536)
            if not data:
                break
            self._buffer += data.decode("utf-8")

        # Parse all messages, keep latest pose, collect others
        latest_pose = None
        other_messages = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                msg = json.loads(line)
                if msg.get("type") == "pose":
                    latest_pose = msg
                else:
                    other_messages.append(msg)

        return latest_pose, other_messages

    def send_command(self, action: str) -> None:
        if self._sock:
            msg = json.dumps({"type": "command", "action": action}) + "\n"
            self._sock.sendall(msg.encode("utf-8"))

    def is_connected(self) -> bool:
        return self._sock is not None

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_ipc.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add blender_mocap/capture_server/ipc_server.py blender_mocap/ipc_client.py tests/test_ipc.py
git commit -m "feat: IPC protocol with handshake, backpressure, and heartbeat"
```

---

### Task 4: Camera Handler

**Files:**
- Create: `blender_mocap/capture_server/camera.py`

- [ ] **Step 1: Implement camera wrapper**

```python
# blender_mocap/capture_server/camera.py
"""OpenCV VideoCapture wrapper for webcam access."""
import cv2
import numpy as np


class Camera:
    """Wraps OpenCV VideoCapture with device enumeration."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at /dev/video{self._device_index}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    @property
    def fps(self) -> float:
        if self._cap:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    @staticmethod
    def list_devices() -> list[int]:
        """Probe /dev/video* devices that can actually open."""
        devices = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append(i)
                cap.release()
        return devices
```

- [ ] **Step 2: Commit**

No unit tests — this is a thin hardware wrapper. Tested via integration.

```bash
git add blender_mocap/capture_server/camera.py
git commit -m "feat: OpenCV camera wrapper with device enumeration"
```

---

### Task 5: Pose Estimator

**Files:**
- Create: `blender_mocap/capture_server/pose_estimator.py`

- [ ] **Step 1: Implement pose estimator wrapper**

```python
# blender_mocap/capture_server/pose_estimator.py
"""MediaPipe Pose wrapper that extracts landmarks from camera frames."""
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """Wraps MediaPipe Pose for single-person pose estimation."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame_rgb: np.ndarray) -> list[dict] | None:
        """Process a frame and return 33 landmarks as dicts, or None if no pose detected."""
        results = self._pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            })
        return landmarks

    def close(self) -> None:
        self._pose.close()
```

- [ ] **Step 2: Commit**

No unit tests — thin wrapper over MediaPipe. Tested via integration with a real camera.

```bash
git add blender_mocap/capture_server/pose_estimator.py
git commit -m "feat: MediaPipe pose estimator wrapper"
```

---

### Task 6: Preview Window

**Files:**
- Create: `blender_mocap/capture_server/preview.py`

- [ ] **Step 1: Implement preview window**

```python
# blender_mocap/capture_server/preview.py
"""OpenCV window showing camera feed with skeleton overlay."""
import cv2
import numpy as np

# MediaPipe Pose landmark connections for skeleton drawing
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (11, 23), (12, 24),  # Torso
    (23, 24),            # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
    (0, 7), (0, 8),      # Head
]

LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (0, 200, 0)
LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2


class PreviewWindow:
    """Displays camera frames with optional skeleton overlay."""

    WINDOW_NAME = "Motion Capture Preview"

    def __init__(self):
        self._open = False

    def open(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._open = True

    def update(self, frame_bgr: np.ndarray, landmarks: list[dict] | None = None) -> bool:
        """Show frame with optional skeleton. Returns False if window was closed."""
        if not self._open:
            return False

        if landmarks:
            self._draw_skeleton(frame_bgr, landmarks)

        cv2.imshow(self.WINDOW_NAME, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        return True

    def _draw_skeleton(self, frame: np.ndarray, landmarks: list[dict]) -> None:
        h, w = frame.shape[:2]
        points = []
        for lm in landmarks:
            px = int(lm["x"] * w)
            py = int(lm["y"] * h)
            points.append((px, py))

        # Draw connections
        for start, end in POSE_CONNECTIONS:
            if start < len(points) and end < len(points):
                if landmarks[start]["visibility"] > 0.5 and landmarks[end]["visibility"] > 0.5:
                    cv2.line(frame, points[start], points[end], CONNECTION_COLOR, CONNECTION_THICKNESS)

        # Draw landmarks
        for i, (px, py) in enumerate(points):
            if landmarks[i]["visibility"] > 0.5:
                cv2.circle(frame, (px, py), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._open = False
```

- [ ] **Step 2: Commit**

```bash
git add blender_mocap/capture_server/preview.py
git commit -m "feat: OpenCV preview window with skeleton overlay"
```

---

### Task 7: Audio Capture

**Files:**
- Create: `blender_mocap/capture_server/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio.py
import os
import tempfile
import wave
import pytest
from blender_mocap.capture_server.audio import AudioRecorder


class TestAudioRecorder:
    def test_list_devices_returns_list(self):
        devices = AudioRecorder.list_input_devices()
        assert isinstance(devices, list)
        # Each device should have an index and name
        for dev in devices:
            assert "index" in dev
            assert "name" in dev

    def test_stop_without_start_returns_none(self):
        recorder = AudioRecorder(device_index=None)
        result = recorder.stop()
        assert result is None

    def test_start_stop_writes_wav(self):
        """Test recording with a mock sounddevice stream."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            recorder = AudioRecorder(device_index=None)
            # Simulate recording by injecting frames directly
            recorder._output_path = path
            recorder._recording = True
            # Add some fake audio data
            import numpy as np
            for _ in range(10):
                recorder._frames.append(np.zeros((1024, 1), dtype=np.int16))
            recorder._recording = False
            result = recorder.stop()
            assert result == path
            assert os.path.exists(path)
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 44100
                assert wf.getnframes() == 10 * 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement audio recorder**

```python
# blender_mocap/capture_server/audio.py
"""Audio capture via sounddevice for synchronized recording."""
import os
import threading
import wave
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioRecorder:
    """Records audio from an input device to a WAV file."""

    SAMPLE_RATE = 44100
    CHANNELS = 1
    DTYPE = "int16"

    def __init__(self, device_index: int | None = None):
        self._device_index = device_index
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self, output_path: str) -> None:
        """Begin recording audio to the specified WAV file path."""
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        self._output_path = output_path
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            device=self._device_index,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())

    def stop(self) -> str | None:
        """Stop recording and write WAV file. Returns the file path."""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                return None
            audio_data = np.concatenate(self._frames, axis=0)
            self._frames = []

        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with wave.open(self._output_path, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        return self._output_path

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Return list of available audio input devices."""
        if sd is None:
            return []
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                inputs.append({"index": i, "name": dev["name"]})
        return inputs

    def _write_test_wav(self, path: str, duration_sec: float = 0.1) -> None:
        """Write a silent WAV file for testing."""
        num_frames = int(self.SAMPLE_RATE * duration_sec)
        silence = np.zeros(num_frames, dtype=np.int16)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(silence.tobytes())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_audio.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add blender_mocap/capture_server/audio.py tests/test_audio.py
git commit -m "feat: audio capture with sounddevice and WAV output"
```

---

### Task 8: Capture Server Main Loop

**Files:**
- Create: `blender_mocap/capture_server/__main__.py`

- [ ] **Step 1: Implement the main entry point**

```python
# blender_mocap/capture_server/__main__.py
"""Capture server entry point. Run as: python -m blender_mocap.capture_server --socket <path> --camera <index>"""
import argparse
import os
import signal
import sys
import time

import cv2

from .camera import Camera
from .pose_estimator import PoseEstimator
from .preview import PreviewWindow
from .ipc_server import IPCServer
from .smoothing import LandmarkSmoother
from .audio import AudioRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motion capture server")
    parser.add_argument("--socket", required=True, help="Unix socket path")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--audio-device", type=int, default=None, help="Audio device index")
    parser.add_argument("--smoothing", type=float, default=0.3, help="Smoothing strength 0.0-1.0")
    return parser.parse_args()


def smoothing_to_min_cutoff(strength: float) -> float:
    """Map slider 0.0-1.0 to min_cutoff: 0.0 -> 1.0 (none), 1.0 -> 0.05 (heavy)."""
    return 1.0 - strength * 0.95


def main() -> None:
    args = parse_args()

    ipc = IPCServer(args.socket)
    ipc.start()

    camera = Camera(args.camera)
    estimator = PoseEstimator()
    preview = PreviewWindow()
    smoother = LandmarkSmoother(
        min_cutoff=smoothing_to_min_cutoff(args.smoothing),
        beta=0.5,
        num_landmarks=33,
    )
    audio = AudioRecorder(device_index=args.audio_device)

    running = True
    previewing = False
    recording = False
    last_heartbeat = time.time()
    heartbeat_interval = 2.0

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Wait for client connection
    while running and not ipc.has_client():
        time.sleep(0.1)

    while running:
        # Process commands
        cmd = ipc.read_command(timeout=0)
        if cmd:
            action = cmd.get("action", "")
            if action == "start_preview":
                if not previewing:
                    camera.open()
                    preview.open()
                    previewing = True
                    ipc.send({"type": "status", "state": "ready", "message": "Preview started"})
            elif action == "stop_preview":
                if previewing:
                    preview.close()
                    camera.close()
                    previewing = False
                if recording:
                    audio.stop()
                    recording = False
                ipc.send({"type": "status", "state": "ready", "message": "Preview stopped"})
            elif action == "start_recording":
                if previewing and not recording:
                    recordings_dir = os.path.expanduser("~/.blender-mocap/recordings")
                    # Find next recording number by scanning for highest existing
                    max_num = 0
                    if os.path.exists(recordings_dir):
                        for f in os.listdir(recordings_dir):
                            if f.startswith("MoCap_") and f.endswith(".wav"):
                                try:
                                    num = int(f[6:-4])
                                    max_num = max(max_num, num)
                                except ValueError:
                                    pass
                    wav_path = os.path.join(recordings_dir, f"MoCap_{max_num + 1:03d}.wav")
                    audio.start(wav_path)
                    recording = True
                    ipc.send({"type": "status", "state": "capturing", "message": f"Recording to {wav_path}"})
            elif action == "stop_recording":
                if recording:
                    wav_path = audio.stop()
                    recording = False
                    ipc.send({"type": "status", "state": "ready", "message": f"Audio saved to {wav_path}"})
            elif action == "shutdown":
                running = False
                continue

        # Capture and process frame
        if previewing:
            ret, frame = camera.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = estimator.estimate(frame_rgb)

                if landmarks:
                    t = time.time()
                    smoothed = smoother(t, landmarks)
                    ipc.send({"type": "pose", "landmarks": smoothed, "timestamp": t})
                    if not preview.update(frame, smoothed):
                        running = False
                else:
                    if not preview.update(frame):
                        running = False
            else:
                time.sleep(0.001)
        else:
            # Not previewing — send heartbeats
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                try:
                    ipc.send_heartbeat()
                    last_heartbeat = now
                except (OSError, BrokenPipeError):
                    running = False
            time.sleep(0.1)

    # Cleanup
    if recording:
        audio.stop()
    preview.close()
    camera.close()
    estimator.close()
    ipc.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add blender_mocap/capture_server/__main__.py
git commit -m "feat: capture server main loop with camera, pose, preview, audio, IPC"
```

---

## Chunk 2: Blender Addon — Rigify Mapping, Recording, UI, Export

These tasks build the Blender-side components. Some tests require `bpy` and must run via `blender --background --python-expr`.

---

### Task 9: Rigify Bone Mapper

**Files:**
- Create: `blender_mocap/rigify_mapper.py`
- Create: `tests/test_rigify_mapper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rigify_mapper.py
import math
import pytest
from blender_mocap.rigify_mapper import (
    mediapipe_to_blender_coords,
    compute_bone_rotation,
    RIGIFY_BONE_MAP,
)


class TestCoordinateTransform:
    def test_center_point(self):
        lm = {"x": 0.5, "y": 0.5, "z": 0.0}
        bx, by, bz = mediapipe_to_blender_coords(lm)
        assert bx == pytest.approx(0.0)  # Centered
        assert bz == pytest.approx(0.0)  # Centered
        assert by == pytest.approx(0.0)  # No depth

    def test_x_mapping(self):
        lm_left = {"x": 0.0, "y": 0.5, "z": 0.0}
        lm_right = {"x": 1.0, "y": 0.5, "z": 0.0}
        bx_l, _, _ = mediapipe_to_blender_coords(lm_left)
        bx_r, _, _ = mediapipe_to_blender_coords(lm_right)
        assert bx_l < bx_r  # Left is negative X, right is positive X

    def test_y_is_negative_z(self):
        lm_top = {"x": 0.5, "y": 0.0, "z": 0.0}
        lm_bottom = {"x": 0.5, "y": 1.0, "z": 0.0}
        _, _, bz_top = mediapipe_to_blender_coords(lm_top)
        _, _, bz_bottom = mediapipe_to_blender_coords(lm_bottom)
        assert bz_top > bz_bottom  # Top of image = higher Z in Blender

    def test_depth_mapping(self):
        lm_near = {"x": 0.5, "y": 0.5, "z": -0.5}
        lm_far = {"x": 0.5, "y": 0.5, "z": 0.5}
        _, by_near, _ = mediapipe_to_blender_coords(lm_near)
        _, by_far, _ = mediapipe_to_blender_coords(lm_far)
        assert by_near > by_far  # Closer to camera = forward (+Y)


class TestBoneRotation:
    def test_identity_rotation(self):
        # Vector pointing in same direction as rest should give identity quaternion
        rest_vec = (0.0, 0.0, 1.0)
        target_vec = (0.0, 0.0, 1.0)
        q = compute_bone_rotation(rest_vec, target_vec)
        # Identity quaternion: (1, 0, 0, 0) or close
        assert q[0] == pytest.approx(1.0, abs=0.01)
        assert abs(q[1]) < 0.01
        assert abs(q[2]) < 0.01
        assert abs(q[3]) < 0.01

    def test_90_degree_rotation(self):
        rest_vec = (0.0, 0.0, 1.0)
        target_vec = (1.0, 0.0, 0.0)
        q = compute_bone_rotation(rest_vec, target_vec)
        # Should produce a 90-degree rotation
        angle = 2 * math.acos(min(abs(q[0]), 1.0))
        assert angle == pytest.approx(math.pi / 2, abs=0.1)


class TestBoneMap:
    def test_has_required_bones(self):
        required = ["upper_arm.L", "forearm.L", "upper_arm.R", "forearm.R",
                     "thigh.L", "shin.L", "thigh.R", "shin.R",
                     "foot.L", "foot.R"]
        for bone in required:
            assert bone in RIGIFY_BONE_MAP, f"Missing mapping for {bone}"

    def test_mapping_has_landmark_indices(self):
        for bone_name, mapping in RIGIFY_BONE_MAP.items():
            assert "parent_idx" in mapping or "indices" in mapping, \
                f"Bone {bone_name} missing landmark indices"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_rigify_mapper.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rigify mapper**

```python
# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

Coordinate transform: MediaPipe (image coords) → Blender (right-handed, Z-up).
Rotation calculation: compute direction vectors between landmarks, convert to
quaternion rotations relative to each bone's rest pose.
"""
import math


def mediapipe_to_blender_coords(lm: dict) -> tuple[float, float, float]:
    """Convert a MediaPipe landmark to Blender world coordinates.

    MediaPipe: X right [0,1], Y down [0,1], Z depth (neg=closer).
    Blender: X right, Y forward, Z up.
    """
    bx = lm["x"] - 0.5       # Center: 0.5 -> 0.0
    by = -lm["z"]             # Depth: neg closer -> positive forward
    bz = -(lm["y"] - 0.5)    # Flip Y: 0=top -> positive Z
    return bx, by, bz


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length < 1e-8:
        return (0.0, 0.0, 1.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def compute_bone_rotation(
    rest_vec: tuple[float, float, float],
    target_vec: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Compute quaternion (w, x, y, z) that rotates rest_vec to target_vec."""
    rest_n = _normalize(rest_vec)
    target_n = _normalize(target_vec)

    dot = _dot(rest_n, target_n)
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9999:
        return (1.0, 0.0, 0.0, 0.0)

    if dot < -0.9999:
        # 180-degree rotation — pick an arbitrary perpendicular axis
        if abs(rest_n[0]) < 0.9:
            perp = _normalize(_cross(rest_n, (1.0, 0.0, 0.0)))
        else:
            perp = _normalize(_cross(rest_n, (0.0, 1.0, 0.0)))
        return (0.0, perp[0], perp[1], perp[2])

    axis = _normalize(_cross(rest_n, target_n))
    half_angle = math.acos(dot) / 2.0
    s = math.sin(half_angle)
    w = math.cos(half_angle)
    return (w, axis[0] * s, axis[1] * s, axis[2] * s)


# Mapping: Rigify bone name -> landmark indices used to compute its rotation
# parent_idx/child_idx: direction vector from parent to child landmark
RIGIFY_BONE_MAP = {
    # Arms
    "upper_arm.L": {"parent_idx": 11, "child_idx": 13},
    "forearm.L":   {"parent_idx": 13, "child_idx": 15},
    "hand.L":      {"parent_idx": 15, "child_idx": 19},  # wrist to index finger tip approx
    "upper_arm.R": {"parent_idx": 12, "child_idx": 14},
    "forearm.R":   {"parent_idx": 14, "child_idx": 16},
    "hand.R":      {"parent_idx": 16, "child_idx": 20},
    # Legs
    "thigh.L":     {"parent_idx": 23, "child_idx": 25},
    "shin.L":      {"parent_idx": 25, "child_idx": 27},
    "thigh.R":     {"parent_idx": 24, "child_idx": 26},
    "shin.R":      {"parent_idx": 26, "child_idx": 28},
    # Feet — use heel-to-toe vector
    "foot.L":      {"indices": [29, 31], "type": "foot"},  # heel to foot index
    "foot.R":      {"indices": [30, 32], "type": "foot"},  # heel to foot index
}

# Spine and head use composite calculations (multiple landmarks)
SPINE_LANDMARKS = {
    "shoulders": (11, 12),
    "hips": (23, 24),
}

HEAD_LANDMARKS = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
}


def compute_limb_rotations(
    landmarks: list[dict],
    bone_rest_vectors: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float, float]]:
    """Compute rotation quaternions for all mapped bones.

    Args:
        landmarks: 33 MediaPipe landmarks with x, y, z, visibility.
        bone_rest_vectors: dict of bone_name -> rest pose direction vector.

    Returns:
        dict of bone_name -> (w, x, y, z) quaternion.
    """
    coords = [mediapipe_to_blender_coords(lm) for lm in landmarks]
    rotations = {}

    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in bone_rest_vectors:
            continue

        rest_vec = bone_rest_vectors[bone_name]

        if mapping.get("type") == "foot":
            # Foot: heel to toe vector
            heel_idx, toe_idx = mapping["indices"]
            target_vec = (
                coords[toe_idx][0] - coords[heel_idx][0],
                coords[toe_idx][1] - coords[heel_idx][1],
                coords[toe_idx][2] - coords[heel_idx][2],
            )
        else:
            parent = coords[mapping["parent_idx"]]
            child = coords[mapping["child_idx"]]
            target_vec = (
                child[0] - parent[0],
                child[1] - parent[1],
                child[2] - parent[2],
            )

        rotations[bone_name] = compute_bone_rotation(rest_vec, target_vec)

    # Torso orientation from shoulders/hips
    if "spine" in bone_rest_vectors:
        l_shoulder = coords[11]
        r_shoulder = coords[12]
        l_hip = coords[23]
        r_hip = coords[24]
        # Spine direction: hip midpoint to shoulder midpoint
        mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2, (l_hip[2] + r_hip[2]) / 2)
        mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2, (l_shoulder[2] + r_shoulder[2]) / 2)
        spine_vec = (mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1], mid_shoulder[2] - mid_hip[2])
        rotations["spine"] = compute_bone_rotation(bone_rest_vectors["spine"], spine_vec)

    # Head orientation from nose and ears
    if "spine.006" in bone_rest_vectors:
        nose = coords[0]
        l_ear = coords[7]
        r_ear = coords[8]
        ear_mid = ((l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2, (l_ear[2] + r_ear[2]) / 2)
        head_vec = (nose[0] - ear_mid[0], nose[1] - ear_mid[1], nose[2] - ear_mid[2])
        rotations["spine.006"] = compute_bone_rotation(bone_rest_vectors["spine.006"], head_vec)

    # Root position (hip midpoint)
    l_hip = coords[23]
    r_hip = coords[24]
    rotations["_root_position"] = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2,
        (l_hip[2] + r_hip[2]) / 2,
    )

    return rotations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_rigify_mapper.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add blender_mocap/rigify_mapper.py tests/test_rigify_mapper.py
git commit -m "feat: Rigify bone mapper with coordinate transform and rotation calc"
```

---

### Task 10: Recording Manager (Frame Buffer + Action Baking)

**Files:**
- Create: `blender_mocap/recording.py`
- Create: `tests/test_recording.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_recording.py
import pytest
from blender_mocap.recording import FrameBuffer


class TestFrameBuffer:
    def test_add_and_count(self):
        buf = FrameBuffer()
        buf.add(0.0, [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}])
        buf.add(0.033, [{"x": 0.6, "y": 0.5, "z": 0.0, "visibility": 1.0}])
        assert buf.frame_count == 2

    def test_clear(self):
        buf = FrameBuffer()
        buf.add(0.0, [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}])
        buf.clear()
        assert buf.frame_count == 0

    def test_duration(self):
        buf = FrameBuffer()
        buf.add(0.0, [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}])
        buf.add(1.0, [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}])
        assert buf.duration == pytest.approx(1.0)

    def test_resample_to_fps(self):
        buf = FrameBuffer()
        # Add frames at ~30fps for 1 second
        for i in range(30):
            buf.add(i / 30.0, [{"x": float(i) / 30.0, "y": 0.5, "z": 0.0, "visibility": 1.0}])
        # Resample to 24fps
        resampled = buf.resample(target_fps=24.0)
        assert len(resampled) == 24  # 1 second at 24fps
        # First frame should match
        assert resampled[0]["landmarks"][0]["x"] == pytest.approx(0.0, abs=0.05)

    def test_empty_resample(self):
        buf = FrameBuffer()
        resampled = buf.resample(target_fps=24.0)
        assert len(resampled) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_recording.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement frame buffer and recording logic**

```python
# blender_mocap/recording.py
"""Frame buffer for recording landmark data and baking to Blender Actions."""


class FrameBuffer:
    """Stores timestamped landmark frames and resamples to target FPS."""

    def __init__(self):
        self._frames: list[dict] = []  # {"timestamp": float, "landmarks": list}

    def add(self, timestamp: float, landmarks: list[dict]) -> None:
        self._frames.append({"timestamp": timestamp, "landmarks": landmarks})

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def duration(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1]["timestamp"] - self._frames[0]["timestamp"]

    def clear(self) -> None:
        self._frames = []

    def resample(self, target_fps: float) -> list[dict]:
        """Resample frames to target FPS using linear interpolation.

        Returns list of {"frame": int, "landmarks": list} dicts.
        """
        if not self._frames:
            return []

        start_t = self._frames[0]["timestamp"]
        end_t = self._frames[-1]["timestamp"]
        duration = end_t - start_t
        if duration <= 0:
            return [{"frame": 0, "landmarks": self._frames[0]["landmarks"]}]

        num_frames = int(duration * target_fps)
        if num_frames <= 0:
            return [{"frame": 0, "landmarks": self._frames[0]["landmarks"]}]

        result = []
        src_idx = 0

        for out_frame in range(num_frames):
            target_t = start_t + out_frame / target_fps

            # Find surrounding source frames
            while src_idx < len(self._frames) - 1 and self._frames[src_idx + 1]["timestamp"] < target_t:
                src_idx += 1

            if src_idx >= len(self._frames) - 1:
                result.append({"frame": out_frame, "landmarks": self._frames[-1]["landmarks"]})
                continue

            f0 = self._frames[src_idx]
            f1 = self._frames[src_idx + 1]
            dt = f1["timestamp"] - f0["timestamp"]
            if dt <= 0:
                alpha = 0.0
            else:
                alpha = (target_t - f0["timestamp"]) / dt

            # Linear interpolation of landmarks
            interp_lm = []
            for lm0, lm1 in zip(f0["landmarks"], f1["landmarks"]):
                interp_lm.append({
                    "x": lm0["x"] + alpha * (lm1["x"] - lm0["x"]),
                    "y": lm0["y"] + alpha * (lm1["y"] - lm0["y"]),
                    "z": lm0["z"] + alpha * (lm1["z"] - lm0["z"]),
                    "visibility": lm0["visibility"],  # No interp on visibility
                })
            result.append({"frame": out_frame, "landmarks": interp_lm})

        return result


def next_action_name(existing_names: list[str]) -> str:
    """Generate next MoCap_NNN name."""
    max_num = 0
    for name in existing_names:
        if name.startswith("MoCap_"):
            try:
                num = int(name.split("_")[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                pass
    return f"MoCap_{max_num + 1:03d}"


def bake_to_action(
    armature,  # bpy.types.Object
    resampled_frames: list[dict],
    bone_rest_vectors: dict[str, tuple],
    action_name: str,
) -> None:
    """Bake resampled landmark frames into a Blender Action.

    Must be called from Blender's Python context.
    """
    import bpy
    from mathutils import Quaternion as MQuaternion
    from .rigify_mapper import compute_limb_rotations

    action = bpy.data.actions.new(name=action_name)
    armature.animation_data_create()
    armature.animation_data.action = action

    for frame_data in resampled_frames:
        frame_num = frame_data["frame"] + 1  # Blender frames start at 1
        landmarks = frame_data["landmarks"]
        rotations = compute_limb_rotations(landmarks, bone_rest_vectors)

        for bone_name, quat in rotations.items():
            if bone_name == "_root_position":
                continue
            if bone_name not in armature.pose.bones:
                continue
            pb = armature.pose.bones[bone_name]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = MQuaternion(quat)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)

        # Root position
        if "_root_position" in rotations and "torso" in armature.pose.bones:
            pos = rotations["_root_position"]
            pb = armature.pose.bones["torso"]
            pb.location = pos
            pb.keyframe_insert(data_path="location", frame=frame_num)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_recording.py -v`
Expected: All PASS (only FrameBuffer tests — bake_to_action requires bpy)

- [ ] **Step 5: Commit**

```bash
git add blender_mocap/recording.py tests/test_recording.py
git commit -m "feat: frame buffer with FPS resampling and action baking"
```

---

### Task 11: Export Module

**Files:**
- Create: `blender_mocap/export.py`
- Create: `tests/test_export.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_export.py
import os
import shutil
import tempfile
import pytest
from blender_mocap.export import copy_audio_file


class TestAudioExport:
    def test_copy_audio_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "source.wav")
            with open(src, "w") as f:
                f.write("fake wav data")
            dst = os.path.join(tmpdir, "dest.wav")
            copy_audio_file(src, dst)
            assert os.path.exists(dst)
            with open(dst) as f:
                assert f.read() == "fake wav data"

    def test_copy_audio_file_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "nonexistent.wav")
            dst = os.path.join(tmpdir, "dest.wav")
            # Should not raise, just return None
            result = copy_audio_file(src, dst)
            assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement export module**

```python
# blender_mocap/export.py
"""Export motion capture data to .blend, FBX, BVH, and WAV formats.

The .blend, FBX, and BVH exports require bpy and must be called from Blender.
Audio export is a simple file copy.
"""
import os
import shutil


def copy_audio_file(src_path: str, dst_path: str) -> str | None:
    """Copy an audio WAV file. Returns destination path or None if source missing."""
    if not os.path.exists(src_path):
        return None
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def export_blend_action(action_name: str, filepath: str) -> None:
    """Export a Blender Action to a .blend file using bpy.data.libraries.write()."""
    import bpy
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise ValueError(f"Action '{action_name}' not found")
    action.use_fake_user = True
    bpy.data.libraries.write(filepath, {action})


def export_fbx(armature_name: str, action_name: str, filepath: str) -> None:
    """Export armature + action as FBX."""
    import bpy
    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise ValueError(f"Armature '{armature_name}' not found")
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise ValueError(f"Action '{action_name}' not found")

    # Set the action as active
    armature.animation_data_create()
    armature.animation_data.action = action

    # Select only the armature
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        object_types={"ARMATURE"},
        add_leaf_bones=False,
        bake_anim=True,
    )


def export_bvh(armature_name: str, action_name: str, filepath: str) -> None:
    """Export action as BVH."""
    import bpy
    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise ValueError(f"Armature '{armature_name}' not found")
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise ValueError(f"Action '{action_name}' not found")

    armature.animation_data_create()
    armature.animation_data.action = action

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_anim.bvh(filepath=filepath)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_export.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add blender_mocap/export.py tests/test_export.py
git commit -m "feat: export module for .blend, FBX, BVH, and WAV"
```

---

### Task 12: Subprocess Manager

**Files:**
- Create: `blender_mocap/subprocess_manager.py`
- Create: `tests/test_subprocess_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_subprocess_manager.py
import os
import tempfile
import pytest
from blender_mocap.subprocess_manager import (
    get_venv_path,
    get_socket_path,
    check_python_version,
    needs_venv_update,
    ADDON_VERSION,
)


class TestSubprocessManager:
    def test_venv_path(self):
        path = get_venv_path()
        assert path.endswith("venv")
        assert ".blender-mocap" in path

    def test_socket_path(self):
        path = get_socket_path(12345)
        assert "blender-mocap-12345" in path
        assert path.startswith("/tmp/")

    def test_check_python_version_valid(self):
        # Current system Python should work (or not — either way it shouldn't crash)
        result = check_python_version()
        assert isinstance(result, bool)

    def test_needs_venv_update_no_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert needs_venv_update(tmpdir) is True

    def test_needs_venv_update_matching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = os.path.join(tmpdir, ".addon-version")
            with open(marker, "w") as f:
                f.write(ADDON_VERSION)
            assert needs_venv_update(tmpdir) is False

    def test_needs_venv_update_outdated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = os.path.join(tmpdir, ".addon-version")
            with open(marker, "w") as f:
                f.write("0.0.0")
            assert needs_venv_update(tmpdir) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_subprocess_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement subprocess manager**

```python
# blender_mocap/subprocess_manager.py
"""Manages the capture server subprocess: venv creation, launch, shutdown."""
import os
import shutil
import signal
import subprocess
import sys


ADDON_VERSION = "0.1.0"
VENV_DIR = os.path.expanduser("~/.blender-mocap/venv")
RECORDINGS_DIR = os.path.expanduser("~/.blender-mocap/recordings")


def get_venv_path() -> str:
    return VENV_DIR


def get_recordings_path() -> str:
    return RECORDINGS_DIR


def get_socket_path(pid: int) -> str:
    return f"/tmp/blender-mocap-{pid}.sock"


def check_python_version() -> bool:
    """Check if system python3 is in the 3.10-3.12 range."""
    try:
        result = subprocess.run(
            ["python3", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        version_str = result.stdout.strip().split()[-1]
        major, minor = int(version_str.split(".")[0]), int(version_str.split(".")[1])
        return major == 3 and 10 <= minor <= 12
    except (subprocess.SubprocessError, ValueError, IndexError):
        return False


def needs_venv_update(venv_path: str) -> bool:
    """Check if the venv needs to be recreated based on version marker."""
    marker_path = os.path.join(venv_path, ".addon-version")
    if not os.path.exists(marker_path):
        return True
    with open(marker_path) as f:
        return f.read().strip() != ADDON_VERSION


def create_venv(venv_path: str) -> None:
    """Create a venv and install capture server requirements."""
    if os.path.exists(venv_path):
        shutil.rmtree(venv_path)

    subprocess.run(
        ["python3", "-m", "venv", venv_path],
        check=True, timeout=30,
    )

    # Install requirements
    req_path = os.path.join(
        os.path.dirname(__file__), "capture_server", "requirements.txt"
    )
    pip_path = os.path.join(venv_path, "bin", "pip")
    subprocess.run(
        [pip_path, "install", "-r", req_path],
        check=True, timeout=300,
    )

    # Write version marker
    marker_path = os.path.join(venv_path, ".addon-version")
    with open(marker_path, "w") as f:
        f.write(ADDON_VERSION)


def ensure_venv() -> str:
    """Ensure the venv exists and is up-to-date. Returns python path."""
    venv_path = get_venv_path()
    if needs_venv_update(venv_path):
        create_venv(venv_path)
    return os.path.join(venv_path, "bin", "python")


class CaptureProcess:
    """Manages the capture server subprocess lifecycle."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._socket_path: str | None = None

    def start(self, camera_index: int, audio_device: int | None = None,
              smoothing: float = 0.3, pid: int | None = None) -> str:
        """Launch capture server. Returns socket path."""
        python_path = ensure_venv()

        if pid is None:
            pid = os.getpid()
        self._socket_path = get_socket_path(pid)

        # Remove stale socket
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        addon_dir = os.path.dirname(os.path.dirname(__file__))
        env = os.environ.copy()
        env["PYTHONPATH"] = addon_dir + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            python_path, "-m", "blender_mocap.capture_server",
            "--socket", self._socket_path,
            "--camera", str(camera_index),
            "--smoothing", str(smoothing),
        ]
        if audio_device is not None:
            cmd.extend(["--audio-device", str(audio_device)])

        self._process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self._socket_path

    @property
    def socket_path(self) -> str | None:
        return self._socket_path

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the capture server gracefully, then force kill if needed."""
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)

        self._process = None

        if self._socket_path and os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/csjohnst/Documents/Development/Personal/blender-mocap && python -m pytest tests/test_subprocess_manager.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add blender_mocap/subprocess_manager.py tests/test_subprocess_manager.py
git commit -m "feat: subprocess manager with venv bootstrap and lifecycle"
```

---

### Task 13: Addon Properties

**Files:**
- Create: `blender_mocap/properties.py`

- [ ] **Step 1: Implement addon properties**

```python
# blender_mocap/properties.py
"""Blender addon property definitions for the Motion Capture panel."""
import bpy
from bpy.props import (
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
    CollectionProperty,
    BoolProperty,
)
from bpy.types import PropertyGroup


def get_camera_devices(self, context):
    """Enumerate available video devices."""
    items = []
    import glob
    devices = sorted(glob.glob("/dev/video*"))
    for i, dev in enumerate(devices):
        idx = dev.replace("/dev/video", "")
        items.append((idx, f"Camera {idx} ({dev})", f"Use {dev}"))
    if not items:
        items.append(("NONE", "No cameras found", ""))
    return items


def get_audio_devices(self, context):
    """Enumerate available audio input devices."""
    items = [("DEFAULT", "System Default", "Use system default input device")]
    # Audio device list is populated on first preview start from capture server
    return items


class MocapRecordingItem(PropertyGroup):
    name: StringProperty(name="Name")
    frame_count: IntProperty(name="Frames")
    audio_path: StringProperty(name="Audio Path")
    has_audio: BoolProperty(name="Has Audio", default=False)


class MocapProperties(PropertyGroup):
    camera_device: EnumProperty(
        name="Camera",
        description="Webcam device to use",
        items=get_camera_devices,
    )
    target_armature: PointerProperty(
        name="Armature",
        description="Rigify armature to animate",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE" and "rig_id" in obj.data,
    )
    audio_device: EnumProperty(
        name="Audio Source",
        description="Audio input device",
        items=get_audio_devices,
    )
    smoothing: FloatProperty(
        name="Smoothing",
        description="Real-time smoothing strength (0=none, 1=heavy)",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    status: StringProperty(
        name="Status",
        default="Idle",
    )
    is_previewing: BoolProperty(default=False)
    is_recording: BoolProperty(default=False)
    recording_index: IntProperty(name="Active Recording", default=-1)
    recordings: CollectionProperty(type=MocapRecordingItem)


def register():
    bpy.utils.register_class(MocapRecordingItem)
    bpy.utils.register_class(MocapProperties)
    bpy.types.Scene.mocap = PointerProperty(type=MocapProperties)


def unregister():
    del bpy.types.Scene.mocap
    bpy.utils.unregister_class(MocapProperties)
    bpy.utils.unregister_class(MocapRecordingItem)
```

- [ ] **Step 2: Verify via Task 17 integration test**

Properties require `bpy` and cannot be tested with pytest. Verification is deferred to Task 17's Blender headless integration test, which asserts all properties are registered.

- [ ] **Step 3: Commit**

```bash
git add blender_mocap/properties.py
git commit -m "feat: addon properties for camera, audio, armature, recordings"
```

---

### Task 14: Addon Operators

**Files:**
- Create: `blender_mocap/operators.py`

- [ ] **Step 1: Implement operators**

```python
# blender_mocap/operators.py
"""Blender operators for motion capture: preview, record, smooth, export."""
import os
import time
import bpy
from bpy.types import Operator
from .ipc_client import IPCClient
from .subprocess_manager import CaptureProcess, get_recordings_path
from .recording import FrameBuffer, bake_to_action, next_action_name
from .rigify_mapper import compute_limb_rotations
from .export import export_blend_action, export_fbx, export_bvh, copy_audio_file

# Global state (persists across operator invocations)
_capture_process = CaptureProcess()
_ipc_client: IPCClient | None = None
_frame_buffer = FrameBuffer()
_last_message_time = 0.0
_bone_rest_vectors: dict = {}


def _get_bone_rest_vectors(armature) -> dict:
    """Extract rest-pose direction vectors from a Rigify armature."""
    from .rigify_mapper import RIGIFY_BONE_MAP
    vectors = {}
    for bone_name in list(RIGIFY_BONE_MAP.keys()) + ["spine", "spine.006"]:
        if bone_name in armature.data.bones:
            bone = armature.data.bones[bone_name]
            vec = bone.vector.normalized()
            vectors[bone_name] = (vec.x, vec.y, vec.z)
    return vectors


class MOCAP_OT_start_preview(Operator):
    bl_idname = "mocap.start_preview"
    bl_label = "Start Preview"
    bl_description = "Launch webcam preview with pose estimation"

    def execute(self, context):
        global _ipc_client, _last_message_time, _bone_rest_vectors

        props = context.scene.mocap
        if props.camera_device == "NONE":
            self.report({"ERROR"}, "No camera found")
            return {"CANCELLED"}
        if props.target_armature is None:
            self.report({"ERROR"}, "Select a Rigify armature first")
            return {"CANCELLED"}

        camera_idx = int(props.camera_device)
        audio_dev = None if props.audio_device == "DEFAULT" else int(props.audio_device)

        socket_path = _capture_process.start(
            camera_index=camera_idx,
            audio_device=audio_dev,
            smoothing=props.smoothing,
        )

        # Wait for server to create socket
        for _ in range(50):
            if os.path.exists(socket_path):
                break
            time.sleep(0.1)
        else:
            self.report({"ERROR"}, "Capture server failed to start")
            _capture_process.stop()
            return {"CANCELLED"}

        client = IPCClient(socket_path)
        client.connect()
        _ipc_client = client

        # Read handshake
        hello = _ipc_client.read_message(timeout=5.0)
        if not hello or hello.get("protocol_version") != 1:
            self.report({"ERROR"}, "Protocol version mismatch")
            _ipc_client.close()
            _capture_process.stop()
            return {"CANCELLED"}

        _ipc_client.send_command("start_preview")
        _last_message_time = time.time()

        # Cache rest vectors
        _bone_rest_vectors = _get_bone_rest_vectors(props.target_armature)

        props.is_previewing = True
        props.status = "Previewing"

        # Register timer
        bpy.app.timers.register(_poll_poses, first_interval=0.033)

        return {"FINISHED"}


class MOCAP_OT_stop_preview(Operator):
    bl_idname = "mocap.stop_preview"
    bl_label = "Stop"
    bl_description = "Stop preview and/or recording"

    def execute(self, context):
        global _frame_buffer
        props = context.scene.mocap

        if props.is_recording:
            _ipc_client.send_command("stop_recording")
            # Bake recorded frames
            if _frame_buffer.frame_count > 0:
                scene_fps = context.scene.render.fps
                resampled = _frame_buffer.resample(target_fps=scene_fps)
                existing = [r.name for r in props.recordings]
                action_name = next_action_name(existing)
                bake_to_action(props.target_armature, resampled, _bone_rest_vectors, action_name)

                # Add to recordings list
                rec = props.recordings.add()
                rec.name = action_name
                rec.frame_count = len(resampled)
                # Check for audio file
                recordings_dir = get_recordings_path()
                wav_path = os.path.join(recordings_dir, f"{action_name}.wav")
                if os.path.exists(wav_path):
                    rec.audio_path = wav_path
                    rec.has_audio = True

            _frame_buffer.clear()
            props.is_recording = False

        if props.is_previewing:
            _ipc_client.send_command("stop_preview")
            _ipc_client.send_command("shutdown")
            _ipc_client.close()
            _capture_process.stop()
            props.is_previewing = False

        props.status = "Idle"
        return {"FINISHED"}


class MOCAP_OT_start_recording(Operator):
    bl_idname = "mocap.start_recording"
    bl_label = "Record"
    bl_description = "Start recording motion capture"

    def execute(self, context):
        global _frame_buffer
        props = context.scene.mocap
        if not props.is_previewing:
            self.report({"ERROR"}, "Start preview first")
            return {"CANCELLED"}

        _frame_buffer.clear()
        _ipc_client.send_command("start_recording")
        props.is_recording = True
        props.status = "Recording"
        return {"FINISHED"}


class MOCAP_OT_smooth_recording(Operator):
    bl_idname = "mocap.smooth_recording"
    bl_label = "Smooth"
    bl_description = "Apply smoothing to selected recording"

    def execute(self, context):
        props = context.scene.mocap
        if props.recording_index < 0 or props.recording_index >= len(props.recordings):
            self.report({"ERROR"}, "No recording selected")
            return {"CANCELLED"}

        rec = props.recordings[props.recording_index]
        action = bpy.data.actions.get(rec.name)
        if action is None:
            self.report({"ERROR"}, f"Action '{rec.name}' not found")
            return {"CANCELLED"}

        # Apply F-curve smoothing
        for fcurve in action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = "BEZIER"
            fcurve.update()
            # Use Blender's smooth operator via override
            for _ in range(int(props.smoothing * 10)):
                for i in range(1, len(fcurve.keyframe_points) - 1):
                    prev_val = fcurve.keyframe_points[i - 1].co[1]
                    next_val = fcurve.keyframe_points[i + 1].co[1]
                    curr_val = fcurve.keyframe_points[i].co[1]
                    fcurve.keyframe_points[i].co[1] = curr_val * 0.5 + (prev_val + next_val) * 0.25

        self.report({"INFO"}, f"Smoothed '{rec.name}'")
        return {"FINISHED"}


class MOCAP_OT_delete_recording(Operator):
    bl_idname = "mocap.delete_recording"
    bl_label = "Delete"
    bl_description = "Delete selected recording"

    def execute(self, context):
        props = context.scene.mocap
        if props.recording_index < 0 or props.recording_index >= len(props.recordings):
            return {"CANCELLED"}

        rec = props.recordings[props.recording_index]
        # Remove action
        action = bpy.data.actions.get(rec.name)
        if action:
            bpy.data.actions.remove(action)
        # Remove audio file
        if rec.has_audio and os.path.exists(rec.audio_path):
            os.unlink(rec.audio_path)
        # Remove from list
        props.recordings.remove(props.recording_index)
        props.recording_index = min(props.recording_index, len(props.recordings) - 1)
        return {"FINISHED"}


class MOCAP_OT_export_blend(Operator):
    bl_idname = "mocap.export_blend"
    bl_label = "Export .blend"
    bl_description = "Export selected recording as .blend action"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return 0 <= props.recording_index < len(props.recordings)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        export_blend_action(rec.name, self.filepath)
        if rec.has_audio:
            audio_dst = os.path.splitext(self.filepath)[0] + ".wav"
            copy_audio_file(rec.audio_path, audio_dst)
        self.report({"INFO"}, f"Exported to {self.filepath}")
        return {"FINISHED"}


class MOCAP_OT_export_fbx(Operator):
    bl_idname = "mocap.export_fbx"
    bl_label = "Export FBX"
    bl_description = "Export selected recording as FBX"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return 0 <= props.recording_index < len(props.recordings)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        export_fbx(props.target_armature.name, rec.name, self.filepath)
        if rec.has_audio:
            audio_dst = os.path.splitext(self.filepath)[0] + ".wav"
            copy_audio_file(rec.audio_path, audio_dst)
        self.report({"INFO"}, f"Exported to {self.filepath}")
        return {"FINISHED"}


class MOCAP_OT_export_bvh(Operator):
    bl_idname = "mocap.export_bvh"
    bl_label = "Export BVH"
    bl_description = "Export selected recording as BVH"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return 0 <= props.recording_index < len(props.recordings)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        export_bvh(props.target_armature.name, rec.name, self.filepath)
        if rec.has_audio:
            audio_dst = os.path.splitext(self.filepath)[0] + ".wav"
            copy_audio_file(rec.audio_path, audio_dst)
        self.report({"INFO"}, f"Exported to {self.filepath}")
        return {"FINISHED"}


class MOCAP_OT_export_audio(Operator):
    bl_idname = "mocap.export_audio"
    bl_label = "Export Audio"
    bl_description = "Export audio WAV for selected recording"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return (0 <= props.recording_index < len(props.recordings)
                and props.recordings[props.recording_index].has_audio)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        if not rec.has_audio:
            self.report({"ERROR"}, "No audio for this recording")
            return {"CANCELLED"}
        copy_audio_file(rec.audio_path, self.filepath)
        self.report({"INFO"}, f"Audio exported to {self.filepath}")
        return {"FINISHED"}


def _poll_poses() -> float | None:
    """Timer callback — polls IPC for new pose data and applies to armature."""
    global _last_message_time, _frame_buffer

    scene = bpy.context.scene
    if not hasattr(scene, "mocap"):
        return None
    props = scene.mocap

    if not props.is_previewing:
        return None

    # Check liveness
    now = time.time()
    if now - _last_message_time > 5.0:
        props.status = "Error: Server not responding"
        props.is_previewing = False
        _ipc_client.close()
        _capture_process.stop()
        return None

    if _ipc_client is None:
        return None

    pose, other_msgs = _ipc_client.drain_latest_pose()

    # Any message (pose, heartbeat, status) resets liveness timer
    if pose is not None or other_msgs:
        _last_message_time = now

    if pose is None:
        return 0.033  # Continue polling
    landmarks = pose["landmarks"]
    timestamp = pose["timestamp"]

    # Buffer if recording
    if props.is_recording:
        _frame_buffer.add(timestamp, landmarks)
        props.status = f"Recording ({_frame_buffer.frame_count} frames)"

    # Apply to armature
    if props.target_armature and _bone_rest_vectors:
        rotations = compute_limb_rotations(landmarks, _bone_rest_vectors)
        from mathutils import Quaternion as MQuaternion

        for bone_name, quat in rotations.items():
            if bone_name == "_root_position":
                continue
            if bone_name in props.target_armature.pose.bones:
                pb = props.target_armature.pose.bones[bone_name]
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = MQuaternion(quat)

        if "_root_position" in rotations and "torso" in props.target_armature.pose.bones:
            pos = rotations["_root_position"]
            props.target_armature.pose.bones["torso"].location = pos

        # Force viewport update
        props.target_armature.update_tag()
        bpy.context.view_layer.update()

    return 0.033  # ~30Hz


class MOCAP_OT_select_recording(Operator):
    bl_idname = "mocap.select_recording"
    bl_label = "Select Recording"
    bl_options = {"INTERNAL"}
    index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.mocap.recording_index = self.index
        return {"FINISHED"}


CLASSES = [
    MOCAP_OT_start_preview,
    MOCAP_OT_stop_preview,
    MOCAP_OT_start_recording,
    MOCAP_OT_smooth_recording,
    MOCAP_OT_delete_recording,
    MOCAP_OT_export_blend,
    MOCAP_OT_export_fbx,
    MOCAP_OT_export_bvh,
    MOCAP_OT_export_audio,
    MOCAP_OT_select_recording,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    # Cleanup subprocess on unregister
    if _capture_process.is_running():
        try:
            _ipc_client.send_command("shutdown")
        except Exception:
            pass
        _capture_process.stop()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
```

- [ ] **Step 2: Verify via Task 17 integration test**

Operators require `bpy` and cannot be tested with pytest. Verification is deferred to Task 17's Blender headless integration test, which asserts all operators are registered.

- [ ] **Step 3: Commit**

```bash
git add blender_mocap/operators.py
git commit -m "feat: Blender operators for preview, record, smooth, export"
```

---

### Task 15: UI Panel

**Files:**
- Create: `blender_mocap/panels.py`

- [ ] **Step 1: Implement the N-panel**

```python
# blender_mocap/panels.py
"""Blender N-panel UI for the Motion Capture addon."""
import bpy
from bpy.types import Panel


class MOCAP_PT_main(Panel):
    bl_label = "Motion Capture"
    bl_idname = "MOCAP_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"

    def draw(self, context):
        pass  # Sub-panels handle drawing


class MOCAP_PT_setup(Panel):
    bl_label = "Setup"
    bl_idname = "MOCAP_PT_setup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        layout.prop(props, "camera_device")
        layout.prop(props, "target_armature")
        layout.prop(props, "audio_device")
        layout.prop(props, "smoothing", slider=True)


class MOCAP_PT_capture(Panel):
    bl_label = "Capture"
    bl_idname = "MOCAP_PT_capture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        if not props.is_previewing:
            layout.operator("mocap.start_preview", icon="PLAY")
        else:
            row = layout.row(align=True)
            if not props.is_recording:
                row.operator("mocap.start_recording", icon="REC")
            row.operator("mocap.stop_preview", icon="SNAP_FACE")

        layout.label(text=f"Status: {props.status}")


class MOCAP_PT_recordings(Panel):
    bl_label = "Recordings"
    bl_idname = "MOCAP_PT_recordings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        if len(props.recordings) == 0:
            layout.label(text="No recordings yet")
            return

        for i, rec in enumerate(props.recordings):
            row = layout.row()
            icon = "SOUND" if rec.has_audio else "ACTION"
            is_active = i == props.recording_index
            row.operator(
                "mocap.select_recording",
                text=f"{rec.name} ({rec.frame_count}f)",
                icon=icon,
                depress=is_active,
            ).index = i

        if props.recording_index >= 0 and props.recording_index < len(props.recordings):
            row = layout.row(align=True)
            row.operator("mocap.smooth_recording")
            row.operator("mocap.delete_recording")


class MOCAP_PT_export(Panel):
    bl_label = "Export"
    bl_idname = "MOCAP_PT_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        has_selection = 0 <= props.recording_index < len(props.recordings)

        row = layout.row(align=True)
        row.enabled = has_selection
        row.operator("mocap.export_blend", text=".blend")
        row.operator("mocap.export_fbx", text="FBX")
        row.operator("mocap.export_bvh", text="BVH")

        if has_selection and props.recordings[props.recording_index].has_audio:
            layout.operator("mocap.export_audio", text="Audio (WAV)")


CLASSES = [
    MOCAP_PT_main,
    MOCAP_PT_setup,
    MOCAP_PT_capture,
    MOCAP_PT_recordings,
    MOCAP_PT_export,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
```

- [ ] **Step 2: Commit**

```bash
git add blender_mocap/panels.py
git commit -m "feat: N-panel UI with setup, capture, recordings, export sections"
```

---

### Task 16: Wire Up __init__.py — Full Registration

**Files:**
- Modify: `blender_mocap/__init__.py`

- [ ] **Step 1: Update __init__.py to register all modules**

```python
# blender_mocap/__init__.py
bl_info = {
    "name": "Motion Capture",
    "author": "Chris",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Motion Capture",
    "description": "Webcam motion capture with MediaPipe pose estimation",
    "category": "Animation",
}

from . import properties
from . import operators
from . import panels


def register():
    properties.register()
    operators.register()
    panels.register()


def unregister():
    panels.unregister()
    operators.unregister()
    properties.unregister()
```

- [ ] **Step 2: Commit**

```bash
git add blender_mocap/__init__.py
git commit -m "feat: wire up addon registration for properties, operators, panels"
```

---

### Task 17: Integration Test — End-to-End Addon Load

**Files:**
- Create: `tests/test_addon_load.py`

- [ ] **Step 1: Write integration test script for Blender headless**

```python
# tests/test_addon_load.py
"""Integration test — verifies addon loads in Blender without errors.

Run with: blender --background --python tests/test_addon_load.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import bpy

def test_addon_loads():
    # Register addon
    from blender_mocap import register, unregister

    register()

    # Verify properties exist
    assert hasattr(bpy.context.scene, "mocap"), "mocap properties not registered"
    props = bpy.context.scene.mocap
    assert hasattr(props, "camera_device")
    assert hasattr(props, "target_armature")
    assert hasattr(props, "smoothing")
    assert hasattr(props, "recordings")

    # Verify operators exist
    assert hasattr(bpy.ops.mocap, "start_preview")
    assert hasattr(bpy.ops.mocap, "stop_preview")
    assert hasattr(bpy.ops.mocap, "start_recording")
    assert hasattr(bpy.ops.mocap, "export_blend")

    # Clean unregister
    unregister()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    try:
        test_addon_loads()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
```

- [ ] **Step 2: Run integration test**

Run: `blender --background --python tests/test_addon_load.py`
Expected: "ALL TESTS PASSED" in output

- [ ] **Step 3: Commit**

```bash
git add tests/test_addon_load.py
git commit -m "test: integration test for addon load/unload in Blender headless"
```

---

### Task 18: Package as Installable Zip

**Files:**
- Create: `scripts/build_addon.sh`

- [ ] **Step 1: Create build script**

```bash
#!/bin/bash
# scripts/build_addon.sh — Package the addon as a .zip for Blender installation
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/dist"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create zip with the blender_mocap/ directory at the root
cd "$PROJECT_DIR"
zip -r "$BUILD_DIR/blender_mocap.zip" blender_mocap/ \
    -x "blender_mocap/__pycache__/*" \
    -x "blender_mocap/capture_server/__pycache__/*"

echo "Built: $BUILD_DIR/blender_mocap.zip"
```

- [ ] **Step 2: Make executable and test**

Run: `chmod +x scripts/build_addon.sh && scripts/build_addon.sh`
Expected: `dist/blender_mocap.zip` created

- [ ] **Step 3: Commit**

```bash
git add scripts/build_addon.sh
git commit -m "feat: build script to package addon as installable .zip"
```
