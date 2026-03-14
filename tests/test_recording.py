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
        # Add frames at ~30fps for 1 second (31 frames: 0/30 to 30/30 inclusive)
        for i in range(31):
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
