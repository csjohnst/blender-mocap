# blender_mocap/operators.py
"""Blender operators for motion capture: preview, record, smooth, export."""
import os
import time
import bpy
from bpy.types import Operator
from .ipc_client import IPCClient
from .subprocess_manager import CaptureProcess, get_recordings_path
from .recording import FrameBuffer, bake_to_action, next_action_name
from .rigify_mapper import compute_limb_rotations
from .export import export_blend_action, export_fbx, export_bvh, copy_audio_file

# Global state (persists across operator invocations)
_capture_process = CaptureProcess()
_ipc_client: IPCClient | None = None
_frame_buffer = FrameBuffer()
_last_message_time = 0.0
_bone_rest_vectors: dict = {}


def _get_bone_rest_vectors(armature) -> dict:
    """Extract rest-pose direction vectors from a Rigify armature."""
    from .rigify_mapper import RIGIFY_BONE_MAP
    vectors = {}
    for bone_name in list(RIGIFY_BONE_MAP.keys()) + ["spine", "spine.006"]:
        if bone_name in armature.data.bones:
            bone = armature.data.bones[bone_name]
            vec = bone.vector.normalized()
            vectors[bone_name] = (vec.x, vec.y, vec.z)
    return vectors


class MOCAP_OT_start_preview(Operator):
    bl_idname = "mocap.start_preview"
    bl_label = "Start Preview"
    bl_description = "Launch webcam preview with pose estimation"

    def execute(self, context):
        global _ipc_client, _last_message_time, _bone_rest_vectors

        props = context.scene.mocap
        if props.camera_device == "NONE":
            self.report({"ERROR"}, "No camera found")
            return {"CANCELLED"}
        if props.target_armature is None:
            self.report({"ERROR"}, "Select a Rigify armature first")
            return {"CANCELLED"}

        camera_idx = int(props.camera_device)
        audio_dev = None if props.audio_device == "DEFAULT" else int(props.audio_device)

        socket_path = _capture_process.start(
            camera_index=camera_idx,
            audio_device=audio_dev,
            smoothing=props.smoothing,
        )

        # Wait for server to create socket
        for _ in range(50):
            if os.path.exists(socket_path):
                break
            time.sleep(0.1)
        else:
            self.report({"ERROR"}, "Capture server failed to start")
            _capture_process.stop()
            return {"CANCELLED"}

        client = IPCClient(socket_path)
        client.connect()
        _ipc_client = client

        # Read handshake
        try:
            hello = _ipc_client.read_message(timeout=5.0)
        except OSError:
            hello = None
        if hello is None:
            self.report({"ERROR"}, "Capture server failed to start — check that OpenCV and MediaPipe are installed")
            _ipc_client.close()
            _capture_process.stop()
            return {"CANCELLED"}
        if hello.get("protocol_version") != 1:
            self.report({"ERROR"}, "Protocol version mismatch")
            _ipc_client.close()
            _capture_process.stop()
            return {"CANCELLED"}

        _ipc_client.send_command("start_preview")
        _last_message_time = time.time()

        # Cache rest vectors
        _bone_rest_vectors = _get_bone_rest_vectors(props.target_armature)

        props.is_previewing = True
        props.status = "Previewing"

        # Register timer
        bpy.app.timers.register(_poll_poses, first_interval=0.033)

        return {"FINISHED"}


class MOCAP_OT_stop_preview(Operator):
    bl_idname = "mocap.stop_preview"
    bl_label = "Stop"
    bl_description = "Stop preview and/or recording"

    def execute(self, context):
        global _frame_buffer
        props = context.scene.mocap

        if props.is_recording:
            _ipc_client.send_command("stop_recording")
            # Bake recorded frames
            if _frame_buffer.frame_count > 0:
                scene_fps = context.scene.render.fps
                resampled = _frame_buffer.resample(target_fps=scene_fps)
                existing = [r.name for r in props.recordings]
                action_name = next_action_name(existing)
                bake_to_action(props.target_armature, resampled, _bone_rest_vectors, action_name)

                # Add to recordings list
                rec = props.recordings.add()
                rec.name = action_name
                rec.frame_count = len(resampled)
                # Check for audio file
                recordings_dir = get_recordings_path()
                wav_path = os.path.join(recordings_dir, f"{action_name}.wav")
                if os.path.exists(wav_path):
                    rec.audio_path = wav_path
                    rec.has_audio = True

            _frame_buffer.clear()
            props.is_recording = False

        if props.is_previewing:
            _ipc_client.send_command("stop_preview")
            _ipc_client.send_command("shutdown")
            _ipc_client.close()
            _capture_process.stop()
            props.is_previewing = False

        props.status = "Idle"
        return {"FINISHED"}


class MOCAP_OT_start_recording(Operator):
    bl_idname = "mocap.start_recording"
    bl_label = "Record"
    bl_description = "Start recording motion capture"

    def execute(self, context):
        global _frame_buffer
        props = context.scene.mocap
        if not props.is_previewing:
            self.report({"ERROR"}, "Start preview first")
            return {"CANCELLED"}

        _frame_buffer.clear()
        _ipc_client.send_command("start_recording")
        props.is_recording = True
        props.status = "Recording"
        return {"FINISHED"}


class MOCAP_OT_smooth_recording(Operator):
    bl_idname = "mocap.smooth_recording"
    bl_label = "Smooth"
    bl_description = "Apply smoothing to selected recording"

    def execute(self, context):
        props = context.scene.mocap
        if props.recording_index < 0 or props.recording_index >= len(props.recordings):
            self.report({"ERROR"}, "No recording selected")
            return {"CANCELLED"}

        rec = props.recordings[props.recording_index]
        action = bpy.data.actions.get(rec.name)
        if action is None:
            self.report({"ERROR"}, f"Action '{rec.name}' not found")
            return {"CANCELLED"}

        # Apply F-curve smoothing
        for fcurve in action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = "BEZIER"
            fcurve.update()
            # Use Blender's smooth operator via override
            for _ in range(int(props.smoothing * 10)):
                for i in range(1, len(fcurve.keyframe_points) - 1):
                    prev_val = fcurve.keyframe_points[i - 1].co[1]
                    next_val = fcurve.keyframe_points[i + 1].co[1]
                    curr_val = fcurve.keyframe_points[i].co[1]
                    fcurve.keyframe_points[i].co[1] = curr_val * 0.5 + (prev_val + next_val) * 0.25

        self.report({"INFO"}, f"Smoothed '{rec.name}'")
        return {"FINISHED"}


class MOCAP_OT_delete_recording(Operator):
    bl_idname = "mocap.delete_recording"
    bl_label = "Delete"
    bl_description = "Delete selected recording"

    def execute(self, context):
        props = context.scene.mocap
        if props.recording_index < 0 or props.recording_index >= len(props.recordings):
            return {"CANCELLED"}

        rec = props.recordings[props.recording_index]
        # Remove action
        action = bpy.data.actions.get(rec.name)
        if action:
            bpy.data.actions.remove(action)
        # Remove audio file
        if rec.has_audio and os.path.exists(rec.audio_path):
            os.unlink(rec.audio_path)
        # Remove from list
        props.recordings.remove(props.recording_index)
        props.recording_index = min(props.recording_index, len(props.recordings) - 1)
        return {"FINISHED"}


class MOCAP_OT_export_blend(Operator):
    bl_idname = "mocap.export_blend"
    bl_label = "Export .blend"
    bl_description = "Export selected recording as .blend action"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return 0 <= props.recording_index < len(props.recordings)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        export_blend_action(rec.name, self.filepath)
        if rec.has_audio:
            audio_dst = os.path.splitext(self.filepath)[0] + ".wav"
            copy_audio_file(rec.audio_path, audio_dst)
        self.report({"INFO"}, f"Exported to {self.filepath}")
        return {"FINISHED"}


class MOCAP_OT_export_fbx(Operator):
    bl_idname = "mocap.export_fbx"
    bl_label = "Export FBX"
    bl_description = "Export selected recording as FBX"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return 0 <= props.recording_index < len(props.recordings)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        export_fbx(props.target_armature.name, rec.name, self.filepath)
        if rec.has_audio:
            audio_dst = os.path.splitext(self.filepath)[0] + ".wav"
            copy_audio_file(rec.audio_path, audio_dst)
        self.report({"INFO"}, f"Exported to {self.filepath}")
        return {"FINISHED"}


class MOCAP_OT_export_bvh(Operator):
    bl_idname = "mocap.export_bvh"
    bl_label = "Export BVH"
    bl_description = "Export selected recording as BVH"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return 0 <= props.recording_index < len(props.recordings)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        export_bvh(props.target_armature.name, rec.name, self.filepath)
        if rec.has_audio:
            audio_dst = os.path.splitext(self.filepath)[0] + ".wav"
            copy_audio_file(rec.audio_path, audio_dst)
        self.report({"INFO"}, f"Exported to {self.filepath}")
        return {"FINISHED"}


class MOCAP_OT_export_audio(Operator):
    bl_idname = "mocap.export_audio"
    bl_label = "Export Audio"
    bl_description = "Export audio WAV for selected recording"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        props = context.scene.mocap
        return (0 <= props.recording_index < len(props.recordings)
                and props.recordings[props.recording_index].has_audio)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.mocap
        rec = props.recordings[props.recording_index]
        if not rec.has_audio:
            self.report({"ERROR"}, "No audio for this recording")
            return {"CANCELLED"}
        copy_audio_file(rec.audio_path, self.filepath)
        self.report({"INFO"}, f"Audio exported to {self.filepath}")
        return {"FINISHED"}


def _poll_poses() -> float | None:
    """Timer callback -- polls IPC for new pose data and applies to armature."""
    global _last_message_time, _frame_buffer

    scene = bpy.context.scene
    if not hasattr(scene, "mocap"):
        return None
    props = scene.mocap

    if not props.is_previewing:
        return None

    if _ipc_client is None:
        return None

    # Check liveness
    now = time.time()
    if now - _last_message_time > 5.0:
        props.status = "Error: Server not responding"
        props.is_previewing = False
        _ipc_client.close()
        _capture_process.stop()
        return None

    pose, other_msgs = _ipc_client.drain_latest_pose()

    # Any message (pose, heartbeat, status) resets liveness timer
    if pose is not None or other_msgs:
        _last_message_time = now

    # Check for server-reported errors
    for msg in other_msgs:
        if msg.get("type") == "error":
            props.status = f"Error: {msg.get('message', 'Unknown error')}"
            props.is_previewing = False
            _ipc_client.close()
            _capture_process.stop()
            return None

    if pose is None:
        return 0.033  # Continue polling
    landmarks = pose["landmarks"]
    timestamp = pose["timestamp"]

    # Buffer if recording
    if props.is_recording:
        _frame_buffer.add(timestamp, landmarks)
        props.status = f"Recording ({_frame_buffer.frame_count} frames)"

    # Apply to armature
    if props.target_armature and _bone_rest_vectors:
        rotations = compute_limb_rotations(landmarks, _bone_rest_vectors)
        from mathutils import Quaternion as MQuaternion

        for bone_name, quat in rotations.items():
            if bone_name == "_root_position":
                continue
            if bone_name in props.target_armature.pose.bones:
                pb = props.target_armature.pose.bones[bone_name]
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = MQuaternion(quat)

        if "_root_position" in rotations and "torso" in props.target_armature.pose.bones:
            pos = rotations["_root_position"]
            props.target_armature.pose.bones["torso"].location = pos

        # Force viewport update
        props.target_armature.update_tag()
        bpy.context.view_layer.update()

    return 0.033  # ~30Hz


class MOCAP_OT_select_recording(Operator):
    bl_idname = "mocap.select_recording"
    bl_label = "Select Recording"
    bl_options = {"INTERNAL"}
    index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.mocap.recording_index = self.index
        return {"FINISHED"}


CLASSES = [
    MOCAP_OT_start_preview,
    MOCAP_OT_stop_preview,
    MOCAP_OT_start_recording,
    MOCAP_OT_smooth_recording,
    MOCAP_OT_delete_recording,
    MOCAP_OT_export_blend,
    MOCAP_OT_export_fbx,
    MOCAP_OT_export_bvh,
    MOCAP_OT_export_audio,
    MOCAP_OT_select_recording,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    # Cleanup subprocess on unregister
    if _capture_process.is_running():
        try:
            _ipc_client.send_command("shutdown")
        except Exception:
            pass
        _capture_process.stop()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
