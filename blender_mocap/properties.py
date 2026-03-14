# blender_mocap/properties.py
"""Blender addon property definitions for the Motion Capture panel."""
import bpy
from bpy.props import (
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
    CollectionProperty,
    BoolProperty,
)
from bpy.types import PropertyGroup


def _get_camera_name(index: int) -> str:
    """Get human-readable name for a camera device index via sysfs."""
    import os
    name_path = f"/sys/class/video4linux/video{index}/name"
    try:
        with open(name_path) as f:
            return f.read().strip()
    except OSError:
        return f"Camera {index}"


def get_camera_devices(self, context):
    """Enumerate available video devices by name."""
    import glob
    items = []
    devices = sorted(glob.glob("/dev/video*"))
    for dev in devices:
        idx = dev.replace("/dev/video", "")
        try:
            index = int(idx)
        except ValueError:
            continue
        friendly_name = _get_camera_name(index)
        items.append((idx, friendly_name, f"Use {dev}"))
    if not items:
        items.append(("NONE", "No cameras found", ""))
    return items


def get_audio_devices(self, context):
    """Enumerate available audio input devices."""
    items = [("DEFAULT", "System Default", "Use system default input device")]
    # Audio device list is populated on first preview start from capture server
    return items


class MocapRecordingItem(PropertyGroup):
    name: StringProperty(name="Name")
    frame_count: IntProperty(name="Frames")
    audio_path: StringProperty(name="Audio Path")
    has_audio: BoolProperty(name="Has Audio", default=False)


class MocapProperties(PropertyGroup):
    camera_device: EnumProperty(
        name="Camera",
        description="Webcam device to use",
        items=get_camera_devices,
    )
    target_armature: PointerProperty(
        name="Armature",
        description="Rigify armature to animate",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE" and "rig_id" in obj.data,
    )
    audio_device: EnumProperty(
        name="Audio Source",
        description="Audio input device",
        items=get_audio_devices,
    )
    smoothing: FloatProperty(
        name="Smoothing",
        description="Real-time smoothing strength (0=none, 1=heavy)",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    status: StringProperty(
        name="Status",
        default="Idle",
    )
    is_previewing: BoolProperty(default=False)
    is_recording: BoolProperty(default=False)
    recording_index: IntProperty(name="Active Recording", default=-1)
    recordings: CollectionProperty(type=MocapRecordingItem)


def register():
    bpy.utils.register_class(MocapRecordingItem)
    bpy.utils.register_class(MocapProperties)
    bpy.types.Scene.mocap = PointerProperty(type=MocapProperties)


def unregister():
    del bpy.types.Scene.mocap
    bpy.utils.unregister_class(MocapProperties)
    bpy.utils.unregister_class(MocapRecordingItem)
