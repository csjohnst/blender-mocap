# blender_mocap/subprocess_manager.py
"""Manages the capture server subprocess: venv creation, launch, shutdown."""
import os
import shutil
import signal
import subprocess
import sys


ADDON_VERSION = "0.10.0"
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

    def get_stderr(self) -> str:
        """Read any available stderr from the subprocess (non-blocking).

        Useful for diagnosing why the capture server crashed.
        """
        if self._process is None or self._process.stderr is None:
            return ""
        # If process has exited, read all remaining stderr
        if self._process.poll() is not None:
            try:
                return self._process.stderr.read().decode("utf-8", errors="replace")
            except (OSError, ValueError):
                return ""
        # Process still running — try non-blocking read
        import select
        try:
            ready, _, _ = select.select([self._process.stderr], [], [], 0)
            if ready:
                data = self._process.stderr.read1(4096)
                return data.decode("utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass
        return ""

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
