# blender_mocap/capture_server/preview.py
"""OpenCV window showing camera feed with skeleton overlay."""
import cv2
import numpy as np

# MediaPipe Pose landmark connections for skeleton drawing
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),            # Shoulders
    (11, 23), (12, 24),  # Torso
    (23, 24),            # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
    (0, 7), (0, 8),      # Head
]

LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (0, 200, 0)
LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2


class PreviewWindow:
    """Displays camera frames with optional skeleton overlay."""

    WINDOW_NAME = "Motion Capture Preview"

    def __init__(self):
        self._open = False
        self._frame_count = 0
        self._display_interval = 3  # Show every Nth frame to reduce CPU load

    def open(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._open = True

    def update(self, frame_bgr: np.ndarray, landmarks: list[dict] | None = None) -> bool:
        """Show frame with optional skeleton. Returns False if window was closed."""
        if not self._open:
            return False

        self._frame_count += 1

        # Only render preview every Nth frame — skeleton drawing and imshow
        # are CPU-intensive and the preview is just for visual confirmation
        if self._frame_count % self._display_interval == 0:
            if landmarks:
                self._draw_skeleton(frame_bgr, landmarks)
            cv2.imshow(self.WINDOW_NAME, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        return True

    def _draw_skeleton(self, frame: np.ndarray, landmarks: list[dict]) -> None:
        h, w = frame.shape[:2]
        points = []
        for lm in landmarks:
            px = int(lm["x"] * w)
            py = int(lm["y"] * h)
            points.append((px, py))

        # Draw connections
        for start, end in POSE_CONNECTIONS:
            if start < len(points) and end < len(points):
                if landmarks[start]["visibility"] > 0.5 and landmarks[end]["visibility"] > 0.5:
                    cv2.line(frame, points[start], points[end], CONNECTION_COLOR, CONNECTION_THICKNESS)

        # Draw landmarks
        for i, (px, py) in enumerate(points):
            if landmarks[i]["visibility"] > 0.5:
                cv2.circle(frame, (px, py), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._open = False
