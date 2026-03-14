# tests/test_addon_load.py
"""Integration test -- verifies addon loads in Blender without errors.

Run with: blender --background --python tests/test_addon_load.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import bpy

def test_addon_loads():
    # Register addon
    from blender_mocap import register, unregister

    register()

    # Verify properties exist
    assert hasattr(bpy.context.scene, "mocap"), "mocap properties not registered"
    props = bpy.context.scene.mocap
    assert hasattr(props, "camera_device")
    assert hasattr(props, "target_armature")
    assert hasattr(props, "smoothing")
    assert hasattr(props, "recordings")

    # Verify operators exist
    assert hasattr(bpy.ops.mocap, "start_preview")
    assert hasattr(bpy.ops.mocap, "stop_preview")
    assert hasattr(bpy.ops.mocap, "start_recording")
    assert hasattr(bpy.ops.mocap, "export_blend")

    # Clean unregister
    unregister()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    try:
        test_addon_loads()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
