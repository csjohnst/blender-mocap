bl_info = {
    "name": "Motion Capture",
    "author": "Chris",
    "version": (0, 11, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Motion Capture",
    "description": "Webcam motion capture with MediaPipe pose estimation",
    "category": "Animation",
}

try:
    import bpy
    _HAS_BPY = True
except ImportError:
    _HAS_BPY = False


def register():
    from . import properties, operators, panels
    properties.register()
    operators.register()
    panels.register()


def unregister():
    from . import properties, operators, panels
    panels.unregister()
    operators.unregister()
    properties.unregister()
