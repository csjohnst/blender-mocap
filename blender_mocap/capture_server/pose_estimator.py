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
