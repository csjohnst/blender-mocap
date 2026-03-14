# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

SIMPLE APPROACH:
For each bone, compute an "absolute" rotation from rest to target using
the bone's rest_local matrix. Store the calibration-frame absolute rotation.
At each frame: rotation_quaternion = calib_absolute^-1 @ current_absolute.
This gives identity at calibration and the correct delta for movement.
"""
import math

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

IK_FK_SWITCH_BONES = [
    "upper_arm_parent.L",
    "upper_arm_parent.R",
    "thigh_parent.L",
    "thigh_parent.R",
]

CHEST_BONE = "chest"
HEAD_BONE = "head"
TORSO_BONE = "torso"

# FK chain parent relationships: child -> parent
# Used to subtract parent's world rotation from child's delta
FK_CHAIN_PARENT = {
    "forearm_fk.L": "upper_arm_fk.L",
    "forearm_fk.R": "upper_arm_fk.R",
    "hand_fk.L":    "forearm_fk.L",
    "hand_fk.R":    "forearm_fk.R",
    "shin_fk.L":    "thigh_fk.L",
    "shin_fk.R":    "thigh_fk.R",
    "foot_fk.L":    "shin_fk.L",
    "foot_fk.R":    "shin_fk.R",
}

# Calibration state
_calib_rotations = {}  # bone_name -> Quaternion (absolute rotation at calibration)
_calib_dirs = {}       # bone_name -> Vector (direction at calibration, for chain correction)
_calib_body_angle = 0.0
_is_calibrated = False
_latest_landmarks = None


def mediapipe_to_blender_coords(lm: dict) -> tuple[float, float, float]:
    bx = lm["x"] - 0.5
    by = lm["z"]              # closer to camera = -Y (character faces -Y)
    bz = -(lm["y"] - 0.5)    # top of image = +Z
    return bx, by, bz


def _get_target_dir(coords, mapping):
    """Get target direction from landmark coordinates."""
    from mathutils import Vector
    if mapping.get("type") == "foot":
        heel_idx, toe_idx = mapping["indices"]
        return (coords[toe_idx] - coords[heel_idx]).normalized()
    else:
        return (coords[mapping["child_idx"]] - coords[mapping["parent_idx"]]).normalized()


def _compute_absolute_rotation(rest_bone, target_dir):
    """Compute the absolute rotation for a bone to point at target_dir.

    Returns a Quaternion that would make the bone point from its rest
    direction to target_dir. Computed in the bone's rest_local space.
    """
    from mathutils import Vector

    # rest_local: the bone's rest matrix relative to its parent
    if rest_bone.parent:
        rest_local = rest_bone.parent.matrix_local.inverted() @ rest_bone.matrix_local
    else:
        rest_local = rest_bone.matrix_local

    # Convert target direction into the bone's rest_local frame
    local_target = (rest_local.to_3x3().inverted() @ target_dir).normalized()

    # In rest_local frame, the bone points along Y
    local_rest = Vector((0, 1, 0))

    # Rotation from rest to target in the bone's local frame
    return local_rest.rotation_difference(local_target)


def calibrate(landmarks: list[dict]) -> None:
    """Store the current pose as calibration reference."""
    from mathutils import Vector
    global _calib_rotations, _calib_dirs, _calib_body_angle, _is_calibrated

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]
    _calib_rotations = {}
    _calib_dirs = {}

    print("[MoCap] === CALIBRATION ===")

    # Calibrate limb bones
    for bone_name in _bone_cache:
        if bone_name not in RIGIFY_BONE_MAP:
            continue
        mapping = RIGIFY_BONE_MAP[bone_name]
        rest_bone = _bone_cache[bone_name]
        target_dir = _get_target_dir(coords, mapping)
        if target_dir.length < 1e-6:
            continue
        abs_rot = _compute_absolute_rotation(rest_bone, target_dir)
        _calib_rotations[bone_name] = abs_rot
        _calib_dirs[bone_name] = target_dir.copy()
        print(f"  {bone_name}: target=({target_dir.x:.3f}, {target_dir.y:.3f}, {target_dir.z:.3f})")

    # Calibrate spine
    mid_hip = (coords[23] + coords[24]) / 2
    mid_shoulder = (coords[11] + coords[12]) / 2
    spine_dir = (mid_shoulder - mid_hip).normalized()
    if CHEST_BONE in _bone_cache and spine_dir.length > 1e-6:
        abs_rot = _compute_absolute_rotation(_bone_cache[CHEST_BONE], spine_dir)
        _calib_rotations[CHEST_BONE] = abs_rot
        print(f"  {CHEST_BONE}: target=({spine_dir.x:.3f}, {spine_dir.y:.3f}, {spine_dir.z:.3f})")

    # Calibrate head
    l_ear, r_ear, nose = coords[7], coords[8], coords[0]
    ear_mid = (l_ear + r_ear) / 2
    face_right = (r_ear - l_ear).normalized()
    face_forward = (nose - ear_mid).normalized()
    face_up = face_right.cross(face_forward).normalized()
    if face_up.z < 0:
        face_up = -face_up
    if HEAD_BONE in _bone_cache and face_up.length > 1e-6:
        abs_rot = _compute_absolute_rotation(_bone_cache[HEAD_BONE], face_up)
        _calib_rotations[HEAD_BONE] = abs_rot
        print(f"  {HEAD_BONE}: face_up=({face_up.x:.3f}, {face_up.y:.3f}, {face_up.z:.3f})")

    # Calibrate body angle (average of shoulder and hip lines)
    shoulder_vec = (coords[12] - coords[11]).normalized()
    hip_vec = (coords[24] - coords[23]).normalized()
    _calib_body_angle = (math.atan2(shoulder_vec.y, shoulder_vec.x) +
                         math.atan2(hip_vec.y, hip_vec.x)) / 2
    print(f"  body_angle: {math.degrees(_calib_body_angle):.1f}°")

    _is_calibrated = True
    print(f"[MoCap] Calibrated {len(_calib_rotations)} bones")


def is_calibrated() -> bool:
    return _is_calibrated


def clear_calibration() -> None:
    global _is_calibrated
    _is_calibrated = False


def store_latest_landmarks(landmarks):
    global _latest_landmarks
    _latest_landmarks = landmarks


def get_latest_landmarks():
    return _latest_landmarks


# Cache of bone rest data (populated on first apply)
_bone_cache = {}  # bone_name -> rest bone


def _ensure_bone_cache(armature):
    """Cache bone rest data from armature."""
    global _bone_cache
    if _bone_cache:
        return

    all_bones = list(RIGIFY_BONE_MAP.keys()) + [CHEST_BONE, HEAD_BONE]
    for bone_name in all_bones:
        if bone_name in armature.data.bones:
            _bone_cache[bone_name] = armature.data.bones[bone_name]

    print(f"[MoCap] Bone cache: {len(_bone_cache)} bones found")
    for name in all_bones:
        if name not in _bone_cache:
            print(f"[MoCap] WARNING: bone '{name}' not found in armature!")
        else:
            bone = _bone_cache[name]
            v = bone.vector.normalized()
            print(f"[MoCap]   {name}: vector=({v.x:.3f}, {v.y:.3f}, {v.z:.3f}) parent={bone.parent.name if bone.parent else 'None'}")


def clear_bone_cache():
    global _bone_cache
    _bone_cache = {}


def apply_pose_to_armature(landmarks: list[dict], armature) -> dict:
    """Apply MediaPipe landmarks to a Rigify armature."""
    from mathutils import Vector, Quaternion

    _ensure_bone_cache(armature)

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]

    # Auto-calibrate on first frame
    if not _is_calibrated:
        calibrate(landmarks)

    # --- TORSO ROTATION ---
    if TORSO_BONE in armature.pose.bones:
        pb = armature.pose.bones[TORSO_BONE]
        # Average shoulder and hip lines for more stable body rotation
        shoulder_vec = (coords[12] - coords[11]).normalized()
        hip_vec = (coords[24] - coords[23]).normalized()
        avg_angle = (math.atan2(shoulder_vec.y, shoulder_vec.x) +
                     math.atan2(hip_vec.y, hip_vec.x)) / 2
        delta_angle = avg_angle - _calib_body_angle
        # Dead zone: ignore tiny rotations (< 5 degrees) to prevent leg crosstalk
        if abs(delta_angle) < math.radians(5):
            delta_angle = 0.0
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion(Vector((0, 0, 1)), -delta_angle)

    # --- CHEST ---
    if CHEST_BONE in _bone_cache and CHEST_BONE in _calib_rotations:
        pb = armature.pose.bones[CHEST_BONE]
        rest_bone = _bone_cache[CHEST_BONE]
        mid_hip = (coords[23] + coords[24]) / 2
        mid_shoulder = (coords[11] + coords[12]) / 2
        spine_dir = (mid_shoulder - mid_hip).normalized()
        if spine_dir.length > 1e-6:
            current_abs = _compute_absolute_rotation(rest_bone, spine_dir)
            calib_abs = _calib_rotations[CHEST_BONE]
            delta = calib_abs.inverted() @ current_abs
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = delta

    # --- HEAD ---
    if HEAD_BONE in _bone_cache and HEAD_BONE in _calib_rotations:
        pb = armature.pose.bones[HEAD_BONE]
        rest_bone = _bone_cache[HEAD_BONE]
        l_ear, r_ear, nose = coords[7], coords[8], coords[0]
        ear_mid = (l_ear + r_ear) / 2
        face_right = (r_ear - l_ear).normalized()
        face_forward = (nose - ear_mid).normalized()
        face_up = face_right.cross(face_forward).normalized()
        if face_up.z < 0:
            face_up = -face_up
        if face_up.length > 1e-6:
            current_abs = _compute_absolute_rotation(rest_bone, face_up)
            calib_abs = _calib_rotations[HEAD_BONE]
            delta = calib_abs.inverted() @ current_abs
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = delta

    # --- LIMBS ---
    # Compute all current directions first
    current_dirs = {}
    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in _bone_cache:
            continue
        target_dir = _get_target_dir(coords, mapping)
        if target_dir.length > 1e-6:
            current_dirs[bone_name] = target_dir

    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in armature.pose.bones:
            continue
        if bone_name not in _calib_rotations:
            continue
        if bone_name not in current_dirs:
            continue

        pb = armature.pose.bones[bone_name]
        rest_bone = _bone_cache[bone_name]
        target_dir = current_dirs[bone_name]
        fk_parent = FK_CHAIN_PARENT.get(bone_name)

        if fk_parent and fk_parent in _calib_dirs and fk_parent in current_dirs:
            # CHAIN CHILD: compute JOINT ANGLE change
            # Joint angle = rotation from parent direction to child direction
            # This directly captures the elbow/knee bend

            parent_calib = _calib_dirs[fk_parent]
            parent_current = current_dirs[fk_parent]
            child_calib = _calib_dirs[bone_name]

            # Joint configuration at calibration and now
            calib_joint = parent_calib.rotation_difference(child_calib)
            current_joint = parent_current.rotation_difference(target_dir)

            # Change in joint angle since calibration
            joint_delta = calib_joint.inverted() @ current_joint

            # Convert world-space joint delta to bone-local frame
            if rest_bone.parent:
                rest_local = rest_bone.parent.matrix_local.inverted() @ rest_bone.matrix_local
            else:
                rest_local = rest_bone.matrix_local
            bone_orient = rest_local.to_3x3()
            local_rot = (bone_orient.inverted() @ joint_delta.to_matrix() @ bone_orient).to_quaternion()

            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = local_rot
        else:
            # ROOT BONE: use calibration delta approach
            current_abs = _compute_absolute_rotation(rest_bone, target_dir)
            calib_abs = _calib_rotations[bone_name]
            delta = calib_abs.inverted() @ current_abs

            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = delta

    # Root position: hips for XY, lowest foot for Z (jump detection)
    hip_mid = (coords[23] + coords[24]) / 2
    # Use the lowest foot landmark (ankles 27, 28) for vertical reference
    # If both feet go up, it's a jump. If only hips drop, feet stay grounded.
    lowest_foot_z = min(coords[27].z, coords[28].z)
    return {
        "_root_position": (hip_mid.x, hip_mid.y, hip_mid.z),
        "_lowest_foot_z": lowest_foot_z,
    }


# Legacy function for recording.py bake
def compute_limb_rotations(landmarks, bone_rest_vectors):
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
