bl_info = {
    "name": "Motion Capture",
    "author": "Chris",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Motion Capture",
    "description": "Webcam motion capture with MediaPipe pose estimation",
    "category": "Animation",
}

from . import properties
from . import operators
from . import panels


def register():
    properties.register()
    operators.register()
    panels.register()


def unregister():
    panels.unregister()
    operators.unregister()
    properties.unregister()
