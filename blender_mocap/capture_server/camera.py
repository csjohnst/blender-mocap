# blender_mocap/capture_server/camera.py
"""OpenCV VideoCapture wrapper for webcam access."""
import cv2
import numpy as np


class Camera:
    """Wraps OpenCV VideoCapture with device enumeration."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._device_path = f"/dev/video{device_index}"
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        import os
        import stat
        path = self._device_path

        # Check device exists and is accessible before OpenCV attempt
        if not os.path.exists(path):
            raise RuntimeError(f"Camera device {path} does not exist")
        try:
            st = os.stat(path)
            if not stat.S_ISCHR(st.st_mode):
                raise RuntimeError(f"{path} is not a character device")
            # Check read/write access
            if not os.access(path, os.R_OK | os.W_OK):
                import getpass
                user = getpass.getuser()
                raise RuntimeError(
                    f"Permission denied on {path} — "
                    f"add user '{user}' to the 'video' group: "
                    f"sudo usermod -aG video {user}"
                )
        except OSError as e:
            raise RuntimeError(f"Cannot access {path}: {e}")

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at {path} — device may be in use by another application")

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
    def get_device_name(index: int) -> str:
        """Get human-readable name for a camera device index."""
        import os
        name_path = f"/sys/class/video4linux/video{index}/name"
        try:
            with open(name_path) as f:
                return f.read().strip()
        except OSError:
            pass
        return f"Camera {index}"

    @staticmethod
    def list_devices() -> list[int]:
        """Probe camera devices (indices 0-9) that can actually open."""
        devices = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append(i)
                cap.release()
        return devices

    @staticmethod
    def list_devices_with_names() -> list[tuple[int, str]]:
        """Return list of (index, name) for available camera devices."""
        result = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                name = Camera.get_device_name(i)
                result.append((i, name))
        return result
