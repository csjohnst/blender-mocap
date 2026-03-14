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
