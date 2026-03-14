# blender_mocap/recording.py
"""Frame buffer for recording landmark data and baking to Blender Actions."""


class FrameBuffer:
    """Stores timestamped landmark frames and resamples to target FPS."""

    def __init__(self):
        self._frames: list[dict] = []  # {"timestamp": float, "landmarks": list}

    def add(self, timestamp: float, landmarks: list[dict]) -> None:
        self._frames.append({"timestamp": timestamp, "landmarks": landmarks})

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def duration(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1]["timestamp"] - self._frames[0]["timestamp"]

    def clear(self) -> None:
        self._frames = []

    def resample(self, target_fps: float) -> list[dict]:
        """Resample frames to target FPS using linear interpolation.

        Returns list of {"frame": int, "landmarks": list} dicts.
        """
        if not self._frames:
            return []

        start_t = self._frames[0]["timestamp"]
        end_t = self._frames[-1]["timestamp"]
        duration = end_t - start_t
        if duration <= 0:
            return [{"frame": 0, "landmarks": self._frames[0]["landmarks"]}]

        num_frames = int(duration * target_fps)
        if num_frames <= 0:
            return [{"frame": 0, "landmarks": self._frames[0]["landmarks"]}]

        result = []
        src_idx = 0

        for out_frame in range(num_frames):
            target_t = start_t + out_frame / target_fps

            # Find surrounding source frames
            while src_idx < len(self._frames) - 1 and self._frames[src_idx + 1]["timestamp"] < target_t:
                src_idx += 1

            if src_idx >= len(self._frames) - 1:
                result.append({"frame": out_frame, "landmarks": self._frames[-1]["landmarks"]})
                continue

            f0 = self._frames[src_idx]
            f1 = self._frames[src_idx + 1]
            dt = f1["timestamp"] - f0["timestamp"]
            if dt <= 0:
                alpha = 0.0
            else:
                alpha = (target_t - f0["timestamp"]) / dt

            # Linear interpolation of landmarks
            interp_lm = []
            for lm0, lm1 in zip(f0["landmarks"], f1["landmarks"]):
                interp_lm.append({
                    "x": lm0["x"] + alpha * (lm1["x"] - lm0["x"]),
                    "y": lm0["y"] + alpha * (lm1["y"] - lm0["y"]),
                    "z": lm0["z"] + alpha * (lm1["z"] - lm0["z"]),
                    "visibility": lm0["visibility"],  # No interp on visibility
                })
            result.append({"frame": out_frame, "landmarks": interp_lm})

        return result


def next_action_name(existing_names: list[str]) -> str:
    """Generate next MoCap_NNN name."""
    max_num = 0
    for name in existing_names:
        if name.startswith("MoCap_"):
            try:
                num = int(name.split("_")[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                pass
    return f"MoCap_{max_num + 1:03d}"


def bake_to_action(
    armature,  # bpy.types.Object
    resampled_frames: list[dict],
    bone_rest_vectors: dict[str, tuple],
    action_name: str,
) -> None:
    """Bake resampled landmark frames into a Blender Action.

    Must be called from Blender's Python context.
    """
    import bpy
    from mathutils import Quaternion as MQuaternion
    from .rigify_mapper import compute_limb_rotations

    action = bpy.data.actions.new(name=action_name)
    armature.animation_data_create()
    armature.animation_data.action = action

    for frame_data in resampled_frames:
        frame_num = frame_data["frame"] + 1  # Blender frames start at 1
        landmarks = frame_data["landmarks"]
        rotations = compute_limb_rotations(landmarks, bone_rest_vectors)

        for bone_name, quat in rotations.items():
            if bone_name == "_root_position":
                continue
            if bone_name not in armature.pose.bones:
                continue
            pb = armature.pose.bones[bone_name]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = MQuaternion(quat)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)

        # Root position
        if "_root_position" in rotations and "torso" in armature.pose.bones:
            pos = rotations["_root_position"]
            pb = armature.pose.bones["torso"]
            pb.location = pos
            pb.keyframe_insert(data_path="location", frame=frame_num)
