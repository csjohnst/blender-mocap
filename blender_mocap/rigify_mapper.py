# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

Processes bones in chain order (root→tip), tracking each bone's computed
armature-space matrix so children correctly account for parent rotations.

The key issue with naive approaches: Blender's pose_bone.matrix setter
and rotation_quaternion are both relative to the parent's current state.
Without a depsgraph update between parent and child, the parent's matrix
is stale. We solve this by computing everything ourselves and only
setting rotation_quaternion at the end.
"""


def mediapipe_to_blender_coords(lm: dict) -> tuple[float, float, float]:
    """Convert a MediaPipe landmark to Blender world coordinates.

    MediaPipe: X right [0,1], Y down [0,1], Z depth (neg=closer).
    Blender: X right, Y forward, Z up.
    """
    bx = lm["x"] - 0.5
    by = -lm["z"]
    bz = -(lm["y"] - 0.5)
    return bx, by, bz


# Rigify FK bone map: bone_name -> landmark indices
RIGIFY_BONE_MAP = {
    "upper_arm_fk.L": {"parent_idx": 11, "child_idx": 13},
    "forearm_fk.L":   {"parent_idx": 13, "child_idx": 15},
    "hand_fk.L":      {"parent_idx": 15, "child_idx": 19},
    "upper_arm_fk.R": {"parent_idx": 12, "child_idx": 14},
    "forearm_fk.R":   {"parent_idx": 14, "child_idx": 16},
    "hand_fk.R":      {"parent_idx": 16, "child_idx": 20},
    "thigh_fk.L":     {"parent_idx": 23, "child_idx": 25},
    "shin_fk.L":      {"parent_idx": 25, "child_idx": 27},
    "thigh_fk.R":     {"parent_idx": 24, "child_idx": 26},
    "shin_fk.R":      {"parent_idx": 26, "child_idx": 28},
    "foot_fk.L":      {"indices": [29, 31], "type": "foot"},
    "foot_fk.R":      {"indices": [30, 32], "type": "foot"},
}

# Process order: root bones first, then children, then leaves
# This ensures parent rotations are computed before children need them
CHAIN_ORDER = [
    "upper_arm_fk.L", "upper_arm_fk.R",
    "thigh_fk.L", "thigh_fk.R",
    "forearm_fk.L", "forearm_fk.R",
    "shin_fk.L", "shin_fk.R",
    "hand_fk.L", "hand_fk.R",
    "foot_fk.L", "foot_fk.R",
]

IK_FK_SWITCH_BONES = [
    "upper_arm_parent.L",
    "upper_arm_parent.R",
    "thigh_parent.L",
    "thigh_parent.R",
]

CHEST_BONE = "chest"
HEAD_BONE = "head"
TORSO_BONE = "torso"


def apply_pose_to_armature(landmarks: list[dict], armature) -> dict:
    """Apply MediaPipe landmarks to a Rigify armature.

    Processes bones root→tip, tracking computed matrices so children
    correctly account for parent rotations without needing depsgraph updates.

    Args:
        landmarks: 33 MediaPipe landmarks with x, y, z, visibility.
        armature: bpy.types.Object (armature).

    Returns:
        dict with "_root_position" for root motion tracking.
    """
    from mathutils import Vector, Matrix, Quaternion

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]

    # Track the armature-space matrix we've computed for each bone
    # so children can reference the parent's NEW orientation
    posed_matrices = {}

    # Process limbs in chain order
    for bone_name in CHAIN_ORDER:
        if bone_name not in armature.pose.bones:
            continue

        mapping = RIGIFY_BONE_MAP[bone_name]
        pb = armature.pose.bones[bone_name]
        rest_bone = pb.bone

        # Compute target direction from landmarks (armature space)
        if mapping.get("type") == "foot":
            heel_idx, toe_idx = mapping["indices"]
            target_dir = (coords[toe_idx] - coords[heel_idx]).normalized()
        else:
            parent_pos = coords[mapping["parent_idx"]]
            child_pos = coords[mapping["child_idx"]]
            target_dir = (child_pos - parent_pos).normalized()

        if target_dir.length < 1e-6:
            continue

        # Compute the effective parent matrix
        # If we've already posed the parent bone, use our computed matrix
        # Otherwise fall back to the rest pose
        if rest_bone.parent and rest_bone.parent.name in posed_matrices:
            parent_mat = posed_matrices[rest_bone.parent.name]
        elif rest_bone.parent:
            parent_mat = rest_bone.parent.matrix_local
        else:
            parent_mat = Matrix.Identity(4)

        # Bone's rest transform relative to parent
        rest_local = parent_mat.inverted() @ rest_bone.matrix_local

        # The bone's effective armature-space matrix (before our rotation)
        effective_mat = parent_mat @ rest_local  # = bone.matrix_local when parent at rest

        # Convert target direction into the bone's local frame
        local_target = (effective_mat.to_3x3().inverted() @ target_dir).normalized()

        # Bone's rest direction in its local frame is always Y
        local_rest = Vector((0, 1, 0))

        # Compute the local rotation
        local_rot = local_rest.rotation_difference(local_target)

        # Apply
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = local_rot

        # Compute and store this bone's new armature-space matrix for children
        rot_mat = local_rot.to_matrix().to_4x4()
        posed_matrices[bone_name] = effective_mat @ rot_mat

    # Chest/spine orientation
    if CHEST_BONE in armature.pose.bones:
        pb = armature.pose.bones[CHEST_BONE]
        rest_bone = pb.bone

        mid_hip = (coords[23] + coords[24]) / 2
        mid_shoulder = (coords[11] + coords[12]) / 2
        target_dir = (mid_shoulder - mid_hip).normalized()

        if target_dir.length > 1e-6:
            _apply_rotation(pb, rest_bone, target_dir, posed_matrices)

    # Head orientation
    if HEAD_BONE in armature.pose.bones:
        pb = armature.pose.bones[HEAD_BONE]
        rest_bone = pb.bone

        ear_mid = (coords[7] + coords[8]) / 2
        target_dir = (coords[0] - ear_mid).normalized()

        if target_dir.length > 1e-6:
            _apply_rotation(pb, rest_bone, target_dir, posed_matrices)

    # Root position
    hip_mid = (coords[23] + coords[24]) / 2
    return {"_root_position": (hip_mid.x, hip_mid.y, hip_mid.z)}


def _apply_rotation(pose_bone, rest_bone, target_dir, posed_matrices):
    """Apply a rotation to make a bone point in target_dir, accounting for parent chain."""
    from mathutils import Vector, Matrix

    if rest_bone.parent and rest_bone.parent.name in posed_matrices:
        parent_mat = posed_matrices[rest_bone.parent.name]
    elif rest_bone.parent:
        parent_mat = rest_bone.parent.matrix_local
    else:
        parent_mat = Matrix.Identity(4)

    rest_local = parent_mat.inverted() @ rest_bone.matrix_local
    effective_mat = parent_mat @ rest_local

    local_target = (effective_mat.to_3x3().inverted() @ target_dir).normalized()
    local_rest = Vector((0, 1, 0))
    local_rot = local_rest.rotation_difference(local_target)

    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = local_rot

    rot_mat = local_rot.to_matrix().to_4x4()
    posed_matrices[rest_bone.name] = effective_mat @ rot_mat


# Legacy function for recording.py bake
def compute_limb_rotations(landmarks, bone_rest_vectors):
    """Compute rotations for baking — uses simplified world-space approach."""
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
            perp = _normalize(_cross(rn, (1, 0, 0) if abs(rn[0]) < 0.9 else (0, 1, 0)))
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
    rotations["_root_position"] = (
        (l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2, (l_hip[2]+r_hip[2])/2
    )
    return rotations
