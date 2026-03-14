# blender_mocap/export.py
"""Export motion capture data to .blend, FBX, BVH, and WAV formats.

The .blend, FBX, and BVH exports require bpy and must be called from Blender.
Audio export is a simple file copy.
"""
import os
import shutil


def copy_audio_file(src_path: str, dst_path: str) -> str | None:
    """Copy an audio WAV file. Returns destination path or None if source missing."""
    if not os.path.exists(src_path):
        return None
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def export_blend_action(action_name: str, filepath: str) -> None:
    """Export a Blender Action to a .blend file using bpy.data.libraries.write()."""
    import bpy
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise ValueError(f"Action '{action_name}' not found")
    action.use_fake_user = True
    bpy.data.libraries.write(filepath, {action})


def export_fbx(armature_name: str, action_name: str, filepath: str) -> None:
    """Export armature + action as FBX."""
    import bpy
    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise ValueError(f"Armature '{armature_name}' not found")
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise ValueError(f"Action '{action_name}' not found")

    # Set the action as active
    armature.animation_data_create()
    armature.animation_data.action = action

    # Select only the armature
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        object_types={"ARMATURE"},
        add_leaf_bones=False,
        bake_anim=True,
    )


def export_bvh(armature_name: str, action_name: str, filepath: str) -> None:
    """Export action as BVH."""
    import bpy
    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise ValueError(f"Armature '{armature_name}' not found")
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise ValueError(f"Action '{action_name}' not found")

    armature.animation_data_create()
    armature.animation_data.action = action

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.ops.export_anim.bvh(filepath=filepath)
