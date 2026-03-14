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
        # Current system Python should work (or not -- either way it shouldn't crash)
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
