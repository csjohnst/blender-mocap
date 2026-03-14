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
