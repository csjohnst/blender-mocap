# blender_mocap/panels.py
"""Blender N-panel UI for the Motion Capture addon."""
import bpy
from bpy.types import Panel


class MOCAP_PT_main(Panel):
    bl_label = "Motion Capture"
    bl_idname = "MOCAP_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"

    def draw(self, context):
        from . import bl_info
        version = ".".join(str(v) for v in bl_info["version"])
        self.layout.label(text=f"v{version}", icon="INFO")


class MOCAP_PT_setup(Panel):
    bl_label = "Setup"
    bl_idname = "MOCAP_PT_setup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        layout.prop(props, "camera_device")
        layout.prop(props, "target_armature")
        layout.prop(props, "audio_device")
        layout.prop(props, "smoothing", slider=True)


class MOCAP_PT_capture(Panel):
    bl_label = "Capture"
    bl_idname = "MOCAP_PT_capture"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        if not props.is_previewing:
            layout.operator("mocap.start_preview", icon="PLAY")
        else:
            row = layout.row(align=True)
            if not props.is_recording:
                row.operator("mocap.start_recording", icon="REC")
            row.operator("mocap.stop_preview", icon="SNAP_FACE")

        layout.label(text=f"Status: {props.status}")


class MOCAP_PT_recordings(Panel):
    bl_label = "Recordings"
    bl_idname = "MOCAP_PT_recordings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        if len(props.recordings) == 0:
            layout.label(text="No recordings yet")
            return

        for i, rec in enumerate(props.recordings):
            row = layout.row()
            icon = "SOUND" if rec.has_audio else "ACTION"
            is_active = i == props.recording_index
            row.operator(
                "mocap.select_recording",
                text=f"{rec.name} ({rec.frame_count}f)",
                icon=icon,
                depress=is_active,
            ).index = i

        if props.recording_index >= 0 and props.recording_index < len(props.recordings):
            row = layout.row(align=True)
            row.operator("mocap.smooth_recording")
            row.operator("mocap.delete_recording")


class MOCAP_PT_export(Panel):
    bl_label = "Export"
    bl_idname = "MOCAP_PT_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Capture"
    bl_parent_id = "MOCAP_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mocap

        has_selection = 0 <= props.recording_index < len(props.recordings)

        row = layout.row(align=True)
        row.enabled = has_selection
        row.operator("mocap.export_blend", text=".blend")
        row.operator("mocap.export_fbx", text="FBX")
        row.operator("mocap.export_bvh", text="BVH")

        if has_selection and props.recordings[props.recording_index].has_audio:
            layout.operator("mocap.export_audio", text="Audio (WAV)")


CLASSES = [
    MOCAP_PT_main,
    MOCAP_PT_setup,
    MOCAP_PT_capture,
    MOCAP_PT_recordings,
    MOCAP_PT_export,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
