bl_info = {
    "name": "Motion Capture",
    "author": "Chris",
    "version": (0, 30, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Motion Capture",
    "description": "Webcam motion capture with MediaPipe pose estimation",
    "category": "Animation",
}

import os
import shutil

# Clear stale __pycache__ on import to prevent cached .pyc from overriding new code
_addon_dir = os.path.dirname(__file__)
for _root, _dirs, _files in os.walk(_addon_dir):
    for _d in _dirs:
        if _d == "__pycache__":
            shutil.rmtree(os.path.join(_root, _d), ignore_errors=True)

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
