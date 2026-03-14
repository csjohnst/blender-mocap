# blender_mocap/capture_server/pose_estimator.py
"""MediaPipe Pose wrapper that extracts landmarks from camera frames.

Uses the MediaPipe Tasks API (PoseLandmarker) which replaces the
deprecated mediapipe.solutions interface removed in v0.10.30.
"""
import os
import urllib.request

import mediapipe as mp
import numpy as np


# Model download URL and local cache path
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_MODEL_DIR = os.path.expanduser("~/.blender-mocap")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")


def _ensure_model() -> str:
    """Download the pose landmarker model if not cached. Returns model path."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"[MoCap] Downloading pose landmarker model to {_MODEL_PATH}...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[MoCap] Model download complete.")
    return _MODEL_PATH


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker (Tasks API) for single-person pose estimation."""

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        model_path = _ensure_model()

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def estimate(self, frame_rgb: np.ndarray) -> list[dict] | None:
        """Process a frame and return 33 landmarks as dicts, or None if no pose detected."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_timestamp_ms += 33  # ~30fps increment
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = []
        for lm in result.pose_landmarks[0]:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            })
        return landmarks

    def close(self) -> None:
        self._landmarker.close()
