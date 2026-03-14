# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

Uses pose_bone.matrix (armature-space) to set bone orientations directly.
Blender internally decomposes this into the correct parent-relative
rotation_quaternion, handling the bone chain hierarchy automatically.

This avoids the error-prone manual bone-local rotation math that breaks
on chains where child rotations compound with parent rotations.
"""


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
    """Apply MediaPipe landmarks to a Rigify armature.

    Sets each bone's armature-space matrix directly using pose_bone.matrix.
    Blender handles decomposition into parent-relative rotation automatically.

    Args:
        landmarks: 33 MediaPipe landmarks with x, y, z, visibility.
        armature: bpy.types.Object (armature).

    Returns:
        dict with "_root_position" for root motion tracking.
    """
    from mathutils import Vector, Matrix

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]
    result = {}

    # Apply limb rotations
    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in armature.pose.bones:
            continue

        pb = armature.pose.bones[bone_name]
        rest_bone = pb.bone

        # Get target direction in armature space
        if mapping.get("type") == "foot":
            heel_idx, toe_idx = mapping["indices"]
            target_dir = (coords[toe_idx] - coords[heel_idx]).normalized()
        else:
            parent_pos = coords[mapping["parent_idx"]]
            child_pos = coords[mapping["child_idx"]]
            target_dir = (child_pos - parent_pos).normalized()

        if target_dir.length < 1e-6:
            continue

        _set_bone_direction(pb, rest_bone, target_dir)

    # Chest orientation from shoulders/hips
    if CHEST_BONE in armature.pose.bones:
        pb = armature.pose.bones[CHEST_BONE]
        rest_bone = pb.bone
        mid_hip = (coords[23] + coords[24]) / 2
        mid_shoulder = (coords[11] + coords[12]) / 2
        spine_dir = (mid_shoulder - mid_hip).normalized()
        if spine_dir.length > 1e-6:
            _set_bone_direction(pb, rest_bone, spine_dir)

    # Head orientation from nose and ears
    if HEAD_BONE in armature.pose.bones:
        pb = armature.pose.bones[HEAD_BONE]
        rest_bone = pb.bone
        ear_mid = (coords[7] + coords[8]) / 2
        head_dir = (coords[0] - ear_mid).normalized()
        if head_dir.length > 1e-6:
            _set_bone_direction(pb, rest_bone, head_dir)

    # Root position (hip midpoint) for world-space translation
    hip_mid = (coords[23] + coords[24]) / 2
    result["_root_position"] = (hip_mid.x, hip_mid.y, hip_mid.z)

    return result


def _set_bone_direction(pose_bone, rest_bone, target_dir):
    """Set a bone to point in target_dir (armature space) while preserving roll.

    Uses pose_bone.matrix to set the armature-space orientation directly.
    Blender decomposes this into the correct parent-relative rotation.
    """
    from mathutils import Vector, Matrix

    # Bone's rest direction in armature space
    rest_dir = rest_bone.vector.normalized()

    # Rotation from rest direction to target direction
    rot = rest_dir.rotation_difference(target_dir)

    # Apply rotation to the bone's rest matrix, keeping head position
    rest_mat = rest_bone.matrix_local.copy()
    head_pos = Vector(rest_bone.head_local)

    # Rotate the rest matrix around the bone's head position
    new_mat = (
        Matrix.Translation(head_pos)
        @ rot.to_matrix().to_4x4()
        @ Matrix.Translation(-head_pos)
        @ rest_mat
    )

    # Set the pose bone's armature-space matrix
    # Blender automatically decomposes this into parent-relative transforms
    pose_bone.matrix = new_mat


# Keep compute_limb_rotations for backwards compat with recording.py bake
def compute_limb_rotations(landmarks, bone_rest_vectors):
    """Legacy function for recording bake — returns rotation dict."""
    import math

    def _normalize(v):
        length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if length < 1e-8:
            return (0, 0, 1)
        return (v[0]/length, v[1]/length, v[2]/length)

    def _cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def _dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def _bone_rot(rest, target):
        rn = _normalize(rest)
        tn = _normalize(target)
        d = max(-1, min(1, _dot(rn, tn)))
        if d > 0.9999:
            return (1, 0, 0, 0)
        if d < -0.9999:
            perp = _normalize(_cross(rn, (1,0,0) if abs(rn[0])<0.9 else (0,1,0)))
            return (0, perp[0], perp[1], perp[2])
        axis = _normalize(_cross(rn, tn))
        ha = math.acos(d) / 2
        s = math.sin(ha)
        return (math.cos(ha), axis[0]*s, axis[1]*s, axis[2]*s)

    coords = [mediapipe_to_blender_coords(lm) for lm in landmarks]
    rotations = {}

    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in bone_rest_vectors:
            continue
        rest_vec = bone_rest_vectors[bone_name]
        if mapping.get("type") == "foot":
            h, t = mapping["indices"]
            tv = (coords[t][0]-coords[h][0], coords[t][1]-coords[h][1], coords[t][2]-coords[h][2])
        else:
            p, c = coords[mapping["parent_idx"]], coords[mapping["child_idx"]]
            tv = (c[0]-p[0], c[1]-p[1], c[2]-p[2])
        rotations[bone_name] = _bone_rot(rest_vec, tv)

    l_hip, r_hip = coords[23], coords[24]
    rotations["_root_position"] = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2, (l_hip[2]+r_hip[2])/2)
    return rotations
