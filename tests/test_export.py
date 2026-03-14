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
