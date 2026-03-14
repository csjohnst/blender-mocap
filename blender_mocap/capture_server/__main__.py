# blender_mocap/capture_server/__main__.py
"""Capture server entry point. Run as: python -m blender_mocap.capture_server --socket <path> --camera <index>"""
import argparse
import os
import signal
import sys
import time

import cv2

from .camera import Camera
from .pose_estimator import PoseEstimator
from .preview import PreviewWindow
from .ipc_server import IPCServer
from .smoothing import LandmarkSmoother
from .audio import AudioRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motion capture server")
    parser.add_argument("--socket", required=True, help="Unix socket path")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--audio-device", type=int, default=None, help="Audio device index")
    parser.add_argument("--smoothing", type=float, default=0.3, help="Smoothing strength 0.0-1.0")
    return parser.parse_args()


def smoothing_to_min_cutoff(strength: float) -> float:
    """Map slider 0.0-1.0 to min_cutoff: 0.0 -> 1.0 (none), 1.0 -> 0.05 (heavy)."""
    return 1.0 - strength * 0.95


def main() -> None:
    args = parse_args()

    ipc = IPCServer(args.socket)
    ipc.start()

    camera = Camera(args.camera)
    estimator = PoseEstimator()
    preview = PreviewWindow()
    smoother = LandmarkSmoother(
        min_cutoff=smoothing_to_min_cutoff(args.smoothing),
        beta=0.5,
        num_landmarks=33,
    )
    audio = AudioRecorder(device_index=args.audio_device)

    running = True
    previewing = False
    recording = False
    last_heartbeat = time.time()
    heartbeat_interval = 2.0

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Wait for client connection
    while running and not ipc.has_client():
        time.sleep(0.1)

    while running:
        # Process commands
        cmd = ipc.read_command(timeout=0)
        if cmd:
            action = cmd.get("action", "")
            if action == "start_preview":
                if not previewing:
                    try:
                        camera.open()
                    except RuntimeError as e:
                        ipc.send({"type": "error", "message": str(e)})
                        running = False
                        continue
                    preview.open()
                    previewing = True
                    ipc.send({"type": "status", "state": "ready", "message": "Preview started"})
            elif action == "stop_preview":
                if previewing:
                    preview.close()
                    camera.close()
                    previewing = False
                if recording:
                    audio.stop()
                    recording = False
                ipc.send({"type": "status", "state": "ready", "message": "Preview stopped"})
            elif action == "start_recording":
                if previewing and not recording:
                    recordings_dir = os.path.expanduser("~/.blender-mocap/recordings")
                    # Find next recording number by scanning for highest existing
                    max_num = 0
                    if os.path.exists(recordings_dir):
                        for f in os.listdir(recordings_dir):
                            if f.startswith("MoCap_") and f.endswith(".wav"):
                                try:
                                    num = int(f[6:-4])
                                    max_num = max(max_num, num)
                                except ValueError:
                                    pass
                    wav_path = os.path.join(recordings_dir, f"MoCap_{max_num + 1:03d}.wav")
                    audio.start(wav_path)
                    recording = True
                    ipc.send({"type": "status", "state": "capturing", "message": f"Recording to {wav_path}"})
            elif action == "stop_recording":
                if recording:
                    wav_path = audio.stop()
                    recording = False
                    ipc.send({"type": "status", "state": "ready", "message": f"Audio saved to {wav_path}"})
            elif action == "shutdown":
                running = False
                continue

        # Capture and process frame
        if previewing:
            ret, frame = camera.read()
            pose_sent = False
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = estimator.estimate(frame_rgb)

                if landmarks:
                    t = time.time()
                    smoothed = smoother(t, landmarks)
                    ipc.send({"type": "pose", "landmarks": smoothed, "timestamp": t})
                    last_heartbeat = t
                    pose_sent = True
                    if not preview.update(frame, smoothed):
                        running = False
                else:
                    if not preview.update(frame):
                        running = False
            else:
                time.sleep(0.001)
            # Send heartbeat when no pose data (camera failure or no landmarks)
            if not pose_sent:
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    try:
                        ipc.send_heartbeat()
                        last_heartbeat = now
                    except (OSError, BrokenPipeError):
                        running = False
        else:
            # Not previewing — send heartbeats
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                try:
                    ipc.send_heartbeat()
                    last_heartbeat = now
                except (OSError, BrokenPipeError):
                    running = False
            time.sleep(0.1)

    # Cleanup
    if recording:
        audio.stop()
    preview.close()
    camera.close()
    estimator.close()
    ipc.stop()


if __name__ == "__main__":
    main()
