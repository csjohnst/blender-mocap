# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

Uses a calibration-relative approach:
1. User stands in A-pose and clicks "Calibrate"
2. The landmark directions in that pose become the reference
3. All subsequent rotations are the DIFFERENCE between current and reference
4. When the user matches their calibration pose, the model is at rest

This eliminates offset errors from the person's natural pose not matching
the Rigify rest pose.
"""
import math

_debug_frame_count = 0
_DEBUG_FRAMES = 3

# Calibration data: stores bone directions from the calibration frame
_calibration_dirs = {}  # bone_name -> Vector direction in blender coords
_calibration_body_angle = 0.0  # shoulder line angle at calibration
_is_calibrated = False


def mediapipe_to_blender_coords(lm: dict) -> tuple[float, float, float]:
    """Convert a MediaPipe landmark to Blender world coordinates.

    MediaPipe: X right [0,1], Y down [0,1], Z depth (neg=closer).
    Blender: X right, Y forward, Z up.

    The character faces -Y (toward camera), so closer to camera = -Y.
    """
    bx = lm["x"] - 0.5
    by = lm["z"]              # depth: neg=closer to camera = -Y (forward for character)
    bz = -(lm["y"] - 0.5)    # flip Y: 0=top -> positive Z
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


def calibrate(landmarks: list[dict]) -> None:
    """Store the current pose as calibration reference (A-pose).

    After calibration, all rotations are computed as the difference
    between the current frame and this reference frame.
    """
    from mathutils import Vector

    global _calibration_dirs, _calibration_body_angle, _is_calibrated

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]

    # Store bone directions at calibration
    _calibration_dirs = {}
    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if mapping.get("type") == "foot":
            heel_idx, toe_idx = mapping["indices"]
            direction = (coords[toe_idx] - coords[heel_idx]).normalized()
        else:
            parent_pos = coords[mapping["parent_idx"]]
            child_pos = coords[mapping["child_idx"]]
            direction = (child_pos - parent_pos).normalized()
        _calibration_dirs[bone_name] = direction

    # Store spine direction
    mid_hip = (coords[23] + coords[24]) / 2
    mid_shoulder = (coords[11] + coords[12]) / 2
    _calibration_dirs["_spine"] = (mid_shoulder - mid_hip).normalized()

    # Store head up direction
    l_ear, r_ear, nose = coords[7], coords[8], coords[0]
    ear_mid = (l_ear + r_ear) / 2
    face_right = (r_ear - l_ear).normalized()
    face_forward = (nose - ear_mid).normalized()
    face_up = face_right.cross(face_forward).normalized()
    if face_up.z < 0:
        face_up = -face_up
    _calibration_dirs["_head"] = face_up

    # Store body angle
    shoulder_vec = (coords[12] - coords[11]).normalized()
    _calibration_body_angle = math.atan2(shoulder_vec.y, shoulder_vec.x)

    _is_calibrated = True
    print(f"[MoCap] Calibrated with {len(_calibration_dirs)} bone directions")


def is_calibrated() -> bool:
    return _is_calibrated


def clear_calibration() -> None:
    global _is_calibrated
    _is_calibrated = False


def apply_pose_to_armature(landmarks: list[dict], armature) -> dict:
    """Apply MediaPipe landmarks to a Rigify armature.

    If calibrated, rotations are relative to the calibration pose.
    If not calibrated, auto-calibrates on first frame.
    """
    from mathutils import Vector, Matrix, Quaternion

    global _debug_frame_count, _is_calibrated
    debug = _debug_frame_count < _DEBUG_FRAMES
    _debug_frame_count += 1

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]

    # Auto-calibrate on first frame if not already calibrated
    if not _is_calibrated:
        calibrate(landmarks)

    if debug:
        print(f"\n[MoCap DEBUG] Frame {_debug_frame_count}")
        print(f"  Calibrated: {_is_calibrated}")
        print(f"  Hip L (23): {coords[23]:.3f}")
        print(f"  Knee L (25): {coords[25]:.3f}")
        print(f"  Shoulder L (11): {coords[11]:.3f}")

    # --- TORSO ROTATION (relative to calibration) ---
    if TORSO_BONE in armature.pose.bones:
        pb = armature.pose.bones[TORSO_BONE]
        shoulder_vec = (coords[12] - coords[11]).normalized()
        current_angle = math.atan2(shoulder_vec.y, shoulder_vec.x)
        # Rotation relative to calibration
        delta_angle = current_angle - _calibration_body_angle
        yaw_quat = Quaternion(Vector((0, 0, 1)), -delta_angle)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = yaw_quat

        if debug:
            print(f"  [torso] delta_angle: {math.degrees(delta_angle):.1f}°")

    # --- CHEST (spine tilt, relative to calibration) ---
    if CHEST_BONE in armature.pose.bones and "_spine" in _calibration_dirs:
        pb = armature.pose.bones[CHEST_BONE]
        rest_bone = pb.bone

        mid_hip = (coords[23] + coords[24]) / 2
        mid_shoulder = (coords[11] + coords[12]) / 2
        current_spine = (mid_shoulder - mid_hip).normalized()
        calib_spine = _calibration_dirs["_spine"]

        # Rotation from calibration spine to current spine
        delta_rot = calib_spine.rotation_difference(current_spine)
        # Convert to bone-local
        bone_orient = rest_bone.matrix_local.to_3x3()
        local_rot = (bone_orient.inverted() @ delta_rot.to_matrix() @ bone_orient).to_quaternion()

        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = local_rot

        if debug:
            print(f"  [chest] calib: {calib_spine:.3f} curr: {current_spine:.3f}")

    # --- HEAD (relative to calibration) ---
    if HEAD_BONE in armature.pose.bones and "_head" in _calibration_dirs:
        pb = armature.pose.bones[HEAD_BONE]
        rest_bone = pb.bone

        l_ear, r_ear, nose = coords[7], coords[8], coords[0]
        ear_mid = (l_ear + r_ear) / 2
        face_right = (r_ear - l_ear).normalized()
        face_forward = (nose - ear_mid).normalized()
        current_up = face_right.cross(face_forward).normalized()
        if current_up.z < 0:
            current_up = -current_up

        calib_up = _calibration_dirs["_head"]
        delta_rot = calib_up.rotation_difference(current_up)
        bone_orient = rest_bone.matrix_local.to_3x3()
        local_rot = (bone_orient.inverted() @ delta_rot.to_matrix() @ bone_orient).to_quaternion()

        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = local_rot

        if debug:
            print(f"  [head] calib_up: {calib_up:.3f} curr_up: {current_up:.3f}")

    # --- LIMBS (relative to calibration, chain-aware) ---
    posed_matrices = {}

    for bone_name in CHAIN_ORDER:
        if bone_name not in armature.pose.bones:
            continue
        if bone_name not in _calibration_dirs:
            continue

        mapping = RIGIFY_BONE_MAP[bone_name]
        pb = armature.pose.bones[bone_name]
        rest_bone = pb.bone

        # Current direction from landmarks
        if mapping.get("type") == "foot":
            heel_idx, toe_idx = mapping["indices"]
            current_dir = (coords[toe_idx] - coords[heel_idx]).normalized()
        else:
            parent_pos = coords[mapping["parent_idx"]]
            child_pos = coords[mapping["child_idx"]]
            current_dir = (child_pos - parent_pos).normalized()

        if current_dir.length < 1e-6:
            continue

        # Calibration direction for this bone
        calib_dir = _calibration_dirs[bone_name]

        # Delta rotation: from calibration direction to current direction
        # This is the actual movement the person made since calibration
        delta_rot = calib_dir.rotation_difference(current_dir)

        # Convert armature-space delta rotation to bone-local
        bone_orient = rest_bone.matrix_local.to_3x3()
        local_rot = (bone_orient.inverted() @ delta_rot.to_matrix() @ bone_orient).to_quaternion()

        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = local_rot

        if debug and bone_name in ("thigh_fk.L", "upper_arm_fk.L"):
            print(f"  [{bone_name}] calib: {calib_dir:.3f} curr: {current_dir:.3f}")
            print(f"  [{bone_name}] delta_rot: {delta_rot:.3f}")
            print(f"  [{bone_name}] local_rot: {local_rot:.3f}")

    # Root position
    hip_mid = (coords[23] + coords[24]) / 2
    return {"_root_position": (hip_mid.x, hip_mid.y, hip_mid.z)}


def reset_debug_counter():
    global _debug_frame_count
    _debug_frame_count = 0


# Legacy function for recording.py bake
def compute_limb_rotations(landmarks, bone_rest_vectors):
    """Compute rotations for baking."""
    coords = [mediapipe_to_blender_coords(lm) for lm in landmarks]
    rotations = {}

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
