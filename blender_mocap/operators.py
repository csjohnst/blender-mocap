# blender_mocap/operators.py
"""Blender operators for motion capture: preview, record, smooth, export."""
import os
import time
import bpy
from bpy.types import Operator
from .ipc_client import IPCClient
from .subprocess_manager import CaptureProcess, get_recordings_path
from .recording import FrameBuffer, bake_to_action, next_action_name
from .rigify_mapper import (
    apply_pose_to_armature, compute_limb_rotations,
    calibrate, clear_calibration, is_calibrated,
    store_latest_landmarks, get_latest_landmarks,
    clear_bone_cache, _ensure_bone_cache,
)


def _ensure_bone_cache_from_armature(armature):
    """Ensure bone cache is populated for calibration."""
    _ensure_bone_cache(armature)
from .export import export_blend_action, export_fbx, export_bvh, copy_audio_file

# Global state (persists across operator invocations)
_capture_process = CaptureProcess()
_ipc_client: IPCClient | None = None
_frame_buffer = FrameBuffer()
_last_message_time = 0.0
_bone_rest_vectors: dict = {}
_initial_root_position: tuple | None = None
_root_scale: float = 5.0


def _get_bone_rest_vectors(armature) -> dict:
    """Extract rest-pose direction vectors from a Rigify generated armature."""
    from .rigify_mapper import RIGIFY_BONE_MAP, CHEST_BONE, HEAD_BONE
    vectors = {}
    all_bones = list(RIGIFY_BONE_MAP.keys()) + [CHEST_BONE, HEAD_BONE]
    for bone_name in all_bones:
        if bone_name in armature.data.bones:
            bone = armature.data.bones[bone_name]
            vec = bone.vector.normalized()
            vectors[bone_name] = (vec.x, vec.y, vec.z)
    return vectors


def _switch_to_fk(armature) -> None:
    """Switch all Rigify limbs to FK mode so we can set rotations directly."""
    from .rigify_mapper import IK_FK_SWITCH_BONES
    print("[MoCap] === FK SWITCH ===")
    for ik_bone_name in IK_FK_SWITCH_BONES:
        if ik_bone_name in armature.pose.bones:
            pb = armature.pose.bones[ik_bone_name]
            if "IK_FK" in pb:
                pb["IK_FK"] = 1.0
                print(f"  {ik_bone_name}: IK_FK set to 1.0 (FK mode)")
            else:
                # Try to find any IK/FK property
                found = False
                for key in pb.keys():
                    if key.startswith("_"):
                        continue
                    if "ik" in key.lower() or "fk" in key.lower():
                        print(f"  {ik_bone_name}: found property '{key}' = {pb[key]}")
                        pb[key] = 1.0
                        found = True
                if not found:
                    print(f"  WARNING: {ik_bone_name} has no IK_FK property! Custom props: {list(pb.keys())}")
        else:
            print(f"  WARNING: {ik_bone_name} not found! Available bones with 'parent': "
                  f"{[b.name for b in armature.pose.bones if 'parent' in b.name]}")


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

        # Wait for server to create socket and be ready to accept
        # The socket file may exist briefly before the server crashes,
        # so we also need to check the process is still alive
        for _ in range(50):
            if os.path.exists(socket_path):
                break
            if not _capture_process.is_running():
                break
            time.sleep(0.1)

        if not _capture_process.is_running():
            stderr = _capture_process.get_stderr()
            error_detail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
            self.report({"ERROR"}, f"Capture server crashed: {error_detail}")
            print(f"[MoCap] Capture server stderr:\n{stderr}")
            _capture_process.stop()
            return {"CANCELLED"}

        if not os.path.exists(socket_path):
            self.report({"ERROR"}, "Capture server failed to start — socket not created")
            _capture_process.stop()
            return {"CANCELLED"}

        # Connect with retry — server may still be initializing after socket bind
        client = IPCClient(socket_path)
        for attempt in range(10):
            try:
                client.connect()
                break
            except (ConnectionRefusedError, OSError):
                if not _capture_process.is_running():
                    stderr = _capture_process.get_stderr()
                    error_detail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
                    self.report({"ERROR"}, f"Capture server crashed: {error_detail}")
                    print(f"[MoCap] Capture server stderr:\n{stderr}")
                    _capture_process.stop()
                    return {"CANCELLED"}
                time.sleep(0.2)
        else:
            stderr = _capture_process.get_stderr()
            error_detail = stderr.strip().splitlines()[-1] if stderr.strip() else "connection refused"
            self.report({"ERROR"}, f"Cannot connect to capture server: {error_detail}")
            print(f"[MoCap] Capture server stderr:\n{stderr}")
            _capture_process.stop()
            return {"CANCELLED"}

        _ipc_client = client

        # Read handshake
        try:
            hello = _ipc_client.read_message(timeout=5.0)
        except OSError:
            hello = None
        if hello is None:
            stderr = _capture_process.get_stderr()
            error_detail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
            self.report({"ERROR"}, f"Capture server failed to start: {error_detail}")
            print(f"[MoCap] Capture server stderr:\n{stderr}")
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

        # Switch Rigify limbs to FK mode, reset calibration
        _switch_to_fk(props.target_armature)
        clear_calibration()
        clear_bone_cache()
        _bone_rest_vectors = _get_bone_rest_vectors(props.target_armature)
        global _initial_root_position
        _initial_root_position = None  # Reset — will be set from first pose frame

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
            if _ipc_client is not None:
                try:
                    _ipc_client.send_command("stop_preview")
                    _ipc_client.send_command("shutdown")
                    _ipc_client.close()
                except Exception:
                    pass
            _capture_process.stop()
            props.is_previewing = False

        props.status = "Idle"
        return {"FINISHED"}


_countdown_remaining = 0


def _countdown_tick() -> float | None:
    """Timer callback for recording countdown."""
    global _countdown_remaining, _frame_buffer, _initial_root_position

    scene = bpy.context.scene
    if not hasattr(scene, "mocap"):
        return None
    props = scene.mocap

    if not props.is_previewing:
        props.status = "Previewing"
        return None

    if _countdown_remaining > 0:
        props.status = f"Recording in {_countdown_remaining}..."
        _countdown_remaining -= 1
        # Force UI redraw so countdown is visible
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return 1.0  # Next tick in 1 second

    # Countdown finished — start recording
    _frame_buffer.clear()
    _initial_root_position = None  # Reset root so recording starts from origin
    _ipc_client.send_command("start_recording")
    props.is_recording = True
    props.status = "Recording"
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
    return None


class MOCAP_OT_start_recording(Operator):
    bl_idname = "mocap.start_recording"
    bl_label = "Record"
    bl_description = "Start recording motion capture (5 second countdown)"

    def execute(self, context):
        global _countdown_remaining
        props = context.scene.mocap
        if not props.is_previewing:
            self.report({"ERROR"}, "Start preview first")
            return {"CANCELLED"}

        _countdown_remaining = 5
        props.status = "Recording in 5..."
        # Force immediate UI redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        bpy.app.timers.register(_countdown_tick, first_interval=1.0)
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
        stderr = _capture_process.get_stderr()
        if stderr.strip():
            error_detail = stderr.strip().splitlines()[-1]
            props.status = f"Error: {error_detail}"
            print(f"[MoCap] Capture server stderr:\n{stderr}")
        else:
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

    # Store for calibration
    store_latest_landmarks(landmarks)

    # Buffer if recording
    if props.is_recording:
        _frame_buffer.add(timestamp, landmarks)
        props.status = f"Recording ({_frame_buffer.frame_count} frames)"

    # Apply to armature using proper bone-local rotations
    if props.target_armature:
        result = apply_pose_to_armature(landmarks, props.target_armature)

        # Apply root motion — track position relative to first frame
        if "_root_position" in result and "root" in props.target_armature.pose.bones:
            global _initial_root_position
            pos = result["_root_position"]
            if _initial_root_position is None:
                _initial_root_position = pos
            dx = (pos[0] - _initial_root_position[0]) * _root_scale
            dy = (pos[1] - _initial_root_position[1]) * _root_scale
            dz = (pos[2] - _initial_root_position[2]) * _root_scale
            props.target_armature.pose.bones["root"].location = (dx, dy, dz)

        # Force viewport update
        props.target_armature.update_tag()
        bpy.context.view_layer.update()

    return 0.033  # ~30Hz


class MOCAP_OT_reset_pose(Operator):
    bl_idname = "mocap.reset_pose"
    bl_label = "Calibrate / Reset"
    bl_description = "Reset armature to rest pose and calibrate: your current pose becomes the A-pose reference"

    def execute(self, context):
        global _initial_root_position
        props = context.scene.mocap
        if props.target_armature is None:
            self.report({"ERROR"}, "No armature selected")
            return {"CANCELLED"}

        # Reset armature to rest pose
        armature = props.target_armature
        for pb in armature.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)

        _initial_root_position = None

        # Calibrate using latest landmarks if preview is active
        latest = get_latest_landmarks()
        if latest is not None:
            clear_bone_cache()
            _ensure_bone_cache_from_armature(armature)
            calibrate(latest)
            self.report({"INFO"}, "Calibrated — your current pose is now the reference")
        else:
            clear_calibration()
            self.report({"INFO"}, "Pose reset (start preview to calibrate)")

        armature.update_tag()
        bpy.context.view_layer.update()
        return {"FINISHED"}


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
    MOCAP_OT_reset_pose,
    MOCAP_OT_select_recording,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    # Cleanup subprocess on unregister
    if _capture_process.is_running():
        if _ipc_client is not None:
            try:
                _ipc_client.send_command("shutdown")
            except Exception:
                pass
        _capture_process.stop()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
