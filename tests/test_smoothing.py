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
