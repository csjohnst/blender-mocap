# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

Processes bones root→tip with tracked parent matrices. Computes
rotation_quaternion in bone-local space for each bone.
"""

_debug_frame_count = 0
_DEBUG_FRAMES = 3  # Print debug info for first N frames


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
HIPS_BONE = "hips"


def apply_pose_to_armature(landmarks: list[dict], armature) -> dict:
    """Apply MediaPipe landmarks to a Rigify armature.

    Args:
        landmarks: 33 MediaPipe landmarks with x, y, z, visibility.
        armature: bpy.types.Object (armature).

    Returns:
        dict with "_root_position" for root motion tracking.
    """
    from mathutils import Vector, Matrix, Quaternion

    global _debug_frame_count
    debug = _debug_frame_count < _DEBUG_FRAMES
    _debug_frame_count += 1

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]

    if debug:
        print(f"\n[MoCap DEBUG] Frame {_debug_frame_count}")
        print(f"  Hip L (23): {coords[23]:.3f}")
        print(f"  Knee L (25): {coords[25]:.3f}")
        print(f"  Shoulder L (11): {coords[11]:.3f}")
        print(f"  Elbow L (13): {coords[13]:.3f}")
        print(f"  Nose (0): {coords[0]:.3f}")

    # --- TORSO ROTATION (body turning left/right) ---
    # Compute from the shoulder line orientation
    if TORSO_BONE in armature.pose.bones:
        pb = armature.pose.bones[TORSO_BONE]
        l_shoulder = coords[11]
        r_shoulder = coords[12]
        l_hip = coords[23]
        r_hip = coords[24]

        # Shoulder line direction (left to right in subject's frame)
        shoulder_vec = (r_shoulder - l_shoulder).normalized()
        hip_vec = (r_hip - l_hip).normalized()

        # Average the shoulder and hip orientation for body yaw
        # The Rigify "torso" bone controls overall body orientation
        # We only apply Y-axis rotation (yaw) to avoid fighting the spine
        # Compute the angle of the shoulder line around Z axis
        import math
        body_angle = math.atan2(shoulder_vec.y, shoulder_vec.x)
        # Rest pose shoulder line is along X (left-right), so angle should be ~0
        # Apply as rotation around Z
        yaw_quat = Quaternion(Vector((0, 0, 1)), -body_angle)

        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = yaw_quat

        if debug:
            print(f"  [torso] shoulder_vec: {shoulder_vec:.3f}, angle: {math.degrees(body_angle):.1f}°")

    # --- CHEST (spine direction) ---
    if CHEST_BONE in armature.pose.bones:
        pb = armature.pose.bones[CHEST_BONE]
        rest_bone = pb.bone

        mid_hip = (coords[23] + coords[24]) / 2
        mid_shoulder = (coords[11] + coords[12]) / 2
        target_dir = (mid_shoulder - mid_hip).normalized()

        if target_dir.length > 1e-6:
            _apply_bone_rotation(pb, rest_bone, target_dir, {}, debug, "chest")

    # --- HEAD (face-up direction, NOT face-forward!) ---
    if HEAD_BONE in armature.pose.bones:
        pb = armature.pose.bones[HEAD_BONE]
        rest_bone = pb.bone

        # The head bone points UPWARD (neck to top of skull)
        # Compute the head's up vector from the face plane:
        # face_right × face_forward = face_up
        l_ear = coords[7]
        r_ear = coords[8]
        nose = coords[0]
        ear_mid = (l_ear + r_ear) / 2

        face_right = (r_ear - l_ear).normalized()
        face_forward = (nose - ear_mid).normalized()
        face_up = face_right.cross(face_forward).normalized()

        if face_up.length > 1e-6:
            # face_up should point upward — if it points down, flip it
            if face_up.z < 0:
                face_up = -face_up
            _apply_bone_rotation(pb, rest_bone, face_up, {}, debug, "head")

        if debug:
            print(f"  [head] face_right: {face_right:.3f}, face_fwd: {face_forward:.3f}, face_up: {face_up:.3f}")

    # --- LIMBS (chain-aware) ---
    posed_matrices = {}

    for bone_name in CHAIN_ORDER:
        if bone_name not in armature.pose.bones:
            if debug:
                print(f"  [SKIP] {bone_name} not found in armature")
            continue

        mapping = RIGIFY_BONE_MAP[bone_name]
        pb = armature.pose.bones[bone_name]
        rest_bone = pb.bone

        # Compute target direction from landmarks
        if mapping.get("type") == "foot":
            heel_idx, toe_idx = mapping["indices"]
            target_dir = (coords[toe_idx] - coords[heel_idx]).normalized()
        else:
            parent_pos = coords[mapping["parent_idx"]]
            child_pos = coords[mapping["child_idx"]]
            target_dir = (child_pos - parent_pos).normalized()

        if target_dir.length < 1e-6:
            continue

        _apply_bone_rotation(pb, rest_bone, target_dir, posed_matrices,
                             debug and bone_name in ("thigh_fk.L", "upper_arm_fk.L", "shin_fk.L"),
                             bone_name)

    # Root position
    hip_mid = (coords[23] + coords[24]) / 2
    return {"_root_position": (hip_mid.x, hip_mid.y, hip_mid.z)}


def _apply_bone_rotation(pose_bone, rest_bone, target_dir, posed_matrices, debug=False, name=""):
    """Compute and apply rotation for a single bone.

    Uses the bone's matrix_local to convert armature-space target direction
    into bone-local space, then sets rotation_quaternion.
    """
    from mathutils import Vector, Matrix

    # Get effective parent matrix
    if rest_bone.parent and rest_bone.parent.name in posed_matrices:
        parent_mat = posed_matrices[rest_bone.parent.name]
    elif rest_bone.parent:
        parent_mat = rest_bone.parent.matrix_local
    else:
        parent_mat = Matrix.Identity(4)

    # Bone's rest transform relative to parent
    rest_local = parent_mat.inverted() @ rest_bone.matrix_local

    # Effective armature-space matrix for this bone
    effective_mat = parent_mat @ rest_local

    # Bone's rest direction in armature space (Y axis of effective matrix)
    rest_dir_armature = (effective_mat.to_3x3() @ Vector((0, 1, 0))).normalized()

    # Convert target direction into bone's local frame
    local_target = (effective_mat.to_3x3().inverted() @ target_dir).normalized()

    # Bone's rest direction in local frame is always Y
    local_rest = Vector((0, 1, 0))

    # Compute rotation
    local_rot = local_rest.rotation_difference(local_target)

    if debug:
        print(f"  [{name}] rest_dir_armature: {rest_dir_armature:.3f}")
        print(f"  [{name}] target_dir: {target_dir:.3f}")
        print(f"  [{name}] local_target: {local_target:.3f}")
        print(f"  [{name}] rotation: {local_rot:.3f}")
        print(f"  [{name}] bone.vector: {rest_bone.vector.normalized():.3f}")
        print(f"  [{name}] parent: {rest_bone.parent.name if rest_bone.parent else 'None'}")
        if rest_bone.parent:
            print(f"  [{name}] parent in posed_matrices: {rest_bone.parent.name in posed_matrices}")

    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = local_rot

    # Track this bone's posed matrix for children
    rot_mat = local_rot.to_matrix().to_4x4()
    posed_matrices[rest_bone.name] = effective_mat @ rot_mat


def reset_debug_counter():
    """Reset debug frame counter (call when preview starts)."""
    global _debug_frame_count
    _debug_frame_count = 0


# Legacy function for recording.py bake
def compute_limb_rotations(landmarks, bone_rest_vectors):
    """Compute rotations for baking."""
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
