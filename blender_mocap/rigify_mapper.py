# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

Uses Blender's mathutils for proper bone-local rotation calculation.
Each bone's rotation is computed by:
1. Getting the world-space target direction from MediaPipe landmarks
2. Converting to the bone's local space using its rest-pose matrix
3. Computing the rotation from the bone's rest direction to the target

This correctly handles bone chains where child rotations are relative
to their parent's orientation.
"""
import math


def mediapipe_to_blender_coords(lm: dict) -> tuple[float, float, float]:
    """Convert a MediaPipe landmark to Blender world coordinates.

    MediaPipe: X right [0,1], Y down [0,1], Z depth (neg=closer).
    Blender: X right, Y forward, Z up.
    """
    bx = lm["x"] - 0.5       # Center: 0.5 -> 0.0
    by = -lm["z"]             # Depth: neg closer -> positive forward
    bz = -(lm["y"] - 0.5)    # Flip Y: 0=top -> positive Z
    return bx, by, bz


# Mapping: Rigify GENERATED rig FK control bone names -> MediaPipe landmark indices
RIGIFY_BONE_MAP = {
    # Arms (FK control bones)
    "upper_arm_fk.L": {"parent_idx": 11, "child_idx": 13},
    "forearm_fk.L":   {"parent_idx": 13, "child_idx": 15},
    "hand_fk.L":      {"parent_idx": 15, "child_idx": 19},
    "upper_arm_fk.R": {"parent_idx": 12, "child_idx": 14},
    "forearm_fk.R":   {"parent_idx": 14, "child_idx": 16},
    "hand_fk.R":      {"parent_idx": 16, "child_idx": 20},
    # Legs (FK control bones)
    "thigh_fk.L":     {"parent_idx": 23, "child_idx": 25},
    "shin_fk.L":      {"parent_idx": 25, "child_idx": 27},
    "thigh_fk.R":     {"parent_idx": 24, "child_idx": 26},
    "shin_fk.R":      {"parent_idx": 26, "child_idx": 28},
    # Feet (FK control bones) -- use heel-to-toe vector
    "foot_fk.L":      {"indices": [29, 31], "type": "foot"},
    "foot_fk.R":      {"indices": [30, 32], "type": "foot"},
}

# Bones that carry the IK_FK custom property (the limb parent/master bones)
IK_FK_SWITCH_BONES = [
    "upper_arm_parent.L",
    "upper_arm_parent.R",
    "thigh_parent.L",
    "thigh_parent.R",
]

# Spine/head control bone names
CHEST_BONE = "chest"
HEAD_BONE = "head"
TORSO_BONE = "torso"


def apply_pose_to_armature(landmarks: list[dict], armature) -> dict:
    """Apply MediaPipe landmarks to a Rigify armature using proper bone-local rotations.

    Uses Blender's mathutils to correctly handle bone hierarchies.
    Must be called from Blender's Python context.

    Args:
        landmarks: 33 MediaPipe landmarks with x, y, z, visibility.
        armature: bpy.types.Object (armature).

    Returns:
        dict with "_root_position" for root motion tracking.
    """
    from mathutils import Vector, Quaternion, Matrix

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]
    result = {}

    # Apply limb rotations
    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in armature.pose.bones:
            continue

        pb = armature.pose.bones[bone_name]
        rest_bone = armature.data.bones[bone_name]

        # Get target direction in world space
        if mapping.get("type") == "foot":
            heel_idx, toe_idx = mapping["indices"]
            target_world = (coords[toe_idx] - coords[heel_idx]).normalized()
        else:
            parent_pos = coords[mapping["parent_idx"]]
            child_pos = coords[mapping["child_idx"]]
            target_world = (child_pos - parent_pos).normalized()

        if target_world.length < 1e-6:
            continue

        # Get the bone's rest direction in world space (bone.vector is in armature space)
        rest_dir_armature = rest_bone.vector.normalized()

        # Transform target direction from world space into armature space
        # (for a basic setup these are the same, but handles armature transforms)
        armature_mat_inv = armature.matrix_world.inverted()
        target_armature = (armature_mat_inv.to_3x3() @ target_world).normalized()

        # Convert target direction into the bone's LOCAL space
        # The bone's rest matrix transforms from bone-local to armature space
        # So its inverse transforms from armature to bone-local space
        if rest_bone.parent:
            # Parent's world-space rest matrix
            parent_rest_mat = rest_bone.parent.matrix_local
            # Bone's own rest matrix relative to parent
            bone_rest_local = parent_rest_mat.inverted() @ rest_bone.matrix_local
        else:
            bone_rest_local = rest_bone.matrix_local

        # Transform target into the bone's local coordinate frame
        local_target = (bone_rest_local.inverted().to_3x3() @ target_armature).normalized()

        # The bone's rest direction in its own local space is always along Y (tail - head)
        local_rest = Vector((0, 1, 0))  # Blender bones point along +Y in local space

        # Compute rotation from local rest to local target
        rotation = local_rest.rotation_difference(local_target)

        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = rotation

    # Chest/spine orientation from shoulders/hips
    if CHEST_BONE in armature.pose.bones:
        pb = armature.pose.bones[CHEST_BONE]
        rest_bone = armature.data.bones[CHEST_BONE]

        l_shoulder = coords[11]
        r_shoulder = coords[12]
        l_hip = coords[23]
        r_hip = coords[24]
        mid_hip = (l_hip + r_hip) / 2
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        spine_dir = (mid_shoulder - mid_hip).normalized()

        if spine_dir.length > 1e-6:
            armature_mat_inv = armature.matrix_world.inverted()
            target_armature = (armature_mat_inv.to_3x3() @ spine_dir).normalized()

            if rest_bone.parent:
                parent_rest_mat = rest_bone.parent.matrix_local
                bone_rest_local = parent_rest_mat.inverted() @ rest_bone.matrix_local
            else:
                bone_rest_local = rest_bone.matrix_local

            local_target = (bone_rest_local.inverted().to_3x3() @ target_armature).normalized()
            local_rest = Vector((0, 1, 0))
            rotation = local_rest.rotation_difference(local_target)

            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rotation

    # Head orientation from nose and ears
    if HEAD_BONE in armature.pose.bones:
        pb = armature.pose.bones[HEAD_BONE]
        rest_bone = armature.data.bones[HEAD_BONE]

        nose = coords[0]
        l_ear = coords[7]
        r_ear = coords[8]
        ear_mid = (l_ear + r_ear) / 2
        head_dir = (nose - ear_mid).normalized()

        if head_dir.length > 1e-6:
            armature_mat_inv = armature.matrix_world.inverted()
            target_armature = (armature_mat_inv.to_3x3() @ head_dir).normalized()

            if rest_bone.parent:
                parent_rest_mat = rest_bone.parent.matrix_local
                bone_rest_local = parent_rest_mat.inverted() @ rest_bone.matrix_local
            else:
                bone_rest_local = rest_bone.matrix_local

            local_target = (bone_rest_local.inverted().to_3x3() @ target_armature).normalized()
            local_rest = Vector((0, 1, 0))
            rotation = local_rest.rotation_difference(local_target)

            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rotation

    # Root position (hip midpoint) for world-space translation
    l_hip = coords[23]
    r_hip = coords[24]
    hip_mid = (l_hip + r_hip) / 2
    result["_root_position"] = (hip_mid.x, hip_mid.y, hip_mid.z)

    return result
