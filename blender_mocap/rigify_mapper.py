# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature.

APPROACH:
- LIMBS: IK mode — position hand_ik/foot_ik targets at landmark positions.
  Blender's IK solver handles elbow/knee bending automatically.
- SPINE/HEAD: FK calibration-delta rotation approach.
- ROOT: hip midpoint X (lateral), lowest foot Z (vertical).

IK eliminates all the FK rotation chain math that was causing
incorrect elbow/knee/arm behavior.
"""
import math


def mediapipe_to_blender_coords(lm: dict) -> tuple[float, float, float]:
    bx = lm["x"] - 0.5
    by = lm["z"]              # closer to camera = -Y
    bz = -(lm["y"] - 0.5)    # top of image = +Z
    return bx, by, bz


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

# IK target bones and their landmark mappings
IK_TARGETS = {
    "hand_ik.L": {"landmark": 15, "shoulder": 11},   # left wrist, left shoulder
    "hand_ik.R": {"landmark": 16, "shoulder": 12},   # right wrist, right shoulder
    "foot_ik.L": {"landmark": 27, "hip": 23},        # left ankle, left hip
    "foot_ik.R": {"landmark": 28, "hip": 24},        # right ankle, right hip
}

# IK pole targets (elbow/knee direction)
IK_POLES = {
    "upper_arm_ik_target.L": 13,  # left elbow
    "upper_arm_ik_target.R": 14,  # right elbow
    "thigh_ik_target.L": 25,      # left knee
    "thigh_ik_target.R": 26,      # right knee
}

# Bones to ensure are in IK mode (IK_FK = 0.0)
IK_FK_SWITCH_BONES = [
    "upper_arm_parent.L",
    "upper_arm_parent.R",
    "thigh_parent.L",
    "thigh_parent.R",
]

CHEST_BONE = "chest"
HEAD_BONE = "head"
TORSO_BONE = "torso"

FK_CHAIN_PARENT = {}  # Not used in IK approach but kept for compat

# Calibration state
_calib_rotations = {}
_calib_body_angle = 0.0
_calib_torso_size = 0.0
_calib_shoulder_width = 0.0
_calib_arm_scale = 1.0     # rig_arm_length / person_arm_length
_calib_leg_scale = 1.0     # rig_leg_length / person_leg_length
_calib_ik_rest = {}        # IK target rest positions
_is_calibrated = False
_latest_landmarks = None
_prev_rotations = {}
_smoothing_factor = 0.4
_max_angular_velocity = math.radians(120)
_prev_ik_positions = {}    # Smoothing for IK targets

# Bone cache
_bone_cache = {}


def _compute_torso_metrics(landmarks_raw):
    ls = landmarks_raw[11]
    rs = landmarks_raw[12]
    lh = landmarks_raw[23]
    rh = landmarks_raw[24]
    shoulder_w = math.sqrt((rs["x"] - ls["x"])**2 + (rs["y"] - ls["y"])**2)
    hip_w = math.sqrt((rh["x"] - lh["x"])**2 + (rh["y"] - lh["y"])**2)
    sm_x, sm_y = (ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2
    hm_x, hm_y = (lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2
    torso_h = math.sqrt((sm_x - hm_x)**2 + (sm_y - hm_y)**2)
    torso_size = (shoulder_w + hip_w + torso_h) / 3
    return torso_size, shoulder_w


def _compute_absolute_rotation(rest_bone, target_dir):
    from mathutils import Vector
    local_target = (rest_bone.matrix_local.to_3x3().inverted() @ target_dir).normalized()
    local_rest = Vector((0, 1, 0))
    return local_rest.rotation_difference(local_target)


def _smooth_rotation(bone_name, new_rot):
    from mathutils import Quaternion
    if bone_name not in _prev_rotations:
        _prev_rotations[bone_name] = new_rot.copy()
        return new_rot
    prev = _prev_rotations[bone_name]
    angle = prev.rotation_difference(new_rot).angle
    if angle > _max_angular_velocity:
        clamped_t = _max_angular_velocity / angle
        new_rot = prev.slerp(new_rot, clamped_t)
    blend = 1.0 - _smoothing_factor
    smoothed = prev.slerp(new_rot, blend)
    _prev_rotations[bone_name] = smoothed.copy()
    return smoothed


def _smooth_position(name, pos, factor=0.3):
    """Smooth a 3D position with linear interpolation."""
    if name not in _prev_ik_positions:
        _prev_ik_positions[name] = pos
        return pos
    prev = _prev_ik_positions[name]
    blend = 1.0 - factor
    smoothed = (
        prev[0] + (pos[0] - prev[0]) * blend,
        prev[1] + (pos[1] - prev[1]) * blend,
        prev[2] + (pos[2] - prev[2]) * blend,
    )
    _prev_ik_positions[name] = smoothed
    return smoothed


def set_smoothing(value):
    global _smoothing_factor
    _smoothing_factor = max(0.0, min(0.95, value))


def calibrate(landmarks):
    from mathutils import Vector
    global _calib_rotations, _calib_body_angle, _is_calibrated
    global _calib_torso_size, _calib_shoulder_width
    global _calib_arm_scale, _calib_leg_scale, _calib_ik_rest

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]
    _calib_rotations = {}
    _calib_ik_rest = {}

    _calib_torso_size, _calib_shoulder_width = _compute_torso_metrics(landmarks)

    print("[MoCap] === CALIBRATION ===")
    print(f"  torso_size: {_calib_torso_size:.4f}")

    # Compute arm and leg scale factors
    # Person's arm length in our coords (shoulder to wrist)
    person_arm_l = (coords[15] - coords[11]).length
    person_arm_r = (coords[16] - coords[12]).length
    person_arm = (person_arm_l + person_arm_r) / 2

    person_leg_l = (coords[27] - coords[23]).length
    person_leg_r = (coords[28] - coords[24]).length
    person_leg = (person_leg_l + person_leg_r) / 2

    # Rig's arm/leg length from bone data
    rig_arm = 0.0
    rig_leg = 0.0
    if "upper_arm_fk.L" in _bone_cache and "forearm_fk.L" in _bone_cache and "hand_fk.L" in _bone_cache:
        rig_arm = (_bone_cache["upper_arm_fk.L"].vector.length +
                   _bone_cache["forearm_fk.L"].vector.length +
                   _bone_cache["hand_fk.L"].vector.length)
    if "thigh_fk.L" in _bone_cache and "shin_fk.L" in _bone_cache:
        rig_leg = (_bone_cache["thigh_fk.L"].vector.length +
                   _bone_cache["shin_fk.L"].vector.length)

    if person_arm > 1e-6 and rig_arm > 1e-6:
        _calib_arm_scale = rig_arm / person_arm
    if person_leg > 1e-6 and rig_leg > 1e-6:
        _calib_leg_scale = rig_leg / person_leg

    print(f"  arm_scale: {_calib_arm_scale:.2f} (rig={rig_arm:.3f} person={person_arm:.3f})")
    print(f"  leg_scale: {_calib_leg_scale:.2f} (rig={rig_leg:.3f} person={person_leg:.3f})")

    # Store IK target rest positions
    for ik_name in IK_TARGETS:
        if ik_name in _bone_cache:
            bone = _bone_cache[ik_name]
            _calib_ik_rest[ik_name] = Vector(bone.head_local)
            print(f"  IK rest {ik_name}: ({bone.head_local.x:.3f}, {bone.head_local.y:.3f}, {bone.head_local.z:.3f})")

    # Store calibration landmark positions for delta computation
    _calib_ik_rest["_coords"] = [c.copy() for c in coords]

    # Calibrate spine/head (FK)
    mid_hip = (coords[23] + coords[24]) / 2
    mid_shoulder = (coords[11] + coords[12]) / 2
    spine_dir = (mid_shoulder - mid_hip).normalized()
    if CHEST_BONE in _bone_cache and spine_dir.length > 1e-6:
        _calib_rotations[CHEST_BONE] = _compute_absolute_rotation(_bone_cache[CHEST_BONE], spine_dir)

    l_ear, r_ear, nose = coords[7], coords[8], coords[0]
    ear_mid = (l_ear + r_ear) / 2
    face_right = (r_ear - l_ear).normalized()
    face_forward = (nose - ear_mid).normalized()
    face_up = face_right.cross(face_forward).normalized()
    if face_up.z < 0:
        face_up = -face_up
    if HEAD_BONE in _bone_cache and face_up.length > 1e-6:
        _calib_rotations[HEAD_BONE] = _compute_absolute_rotation(_bone_cache[HEAD_BONE], face_up)

    shoulder_vec = (coords[12] - coords[11]).normalized()
    hip_vec = (coords[24] - coords[23]).normalized()
    _calib_body_angle = (math.atan2(shoulder_vec.y, shoulder_vec.x) +
                         math.atan2(hip_vec.y, hip_vec.x)) / 2

    _is_calibrated = True
    print(f"[MoCap] Calibration complete")


def is_calibrated():
    return _is_calibrated


def clear_calibration():
    global _is_calibrated, _prev_rotations, _prev_ik_positions
    _is_calibrated = False
    _prev_rotations = {}
    _prev_ik_positions = {}


def store_latest_landmarks(landmarks):
    global _latest_landmarks
    _latest_landmarks = landmarks


def get_latest_landmarks():
    return _latest_landmarks


def _ensure_bone_cache(armature):
    global _bone_cache
    if _bone_cache:
        return
    all_bones = (list(RIGIFY_BONE_MAP.keys()) + [CHEST_BONE, HEAD_BONE] +
                 list(IK_TARGETS.keys()) + list(IK_POLES.keys()))
    for bone_name in all_bones:
        if bone_name in armature.data.bones:
            _bone_cache[bone_name] = armature.data.bones[bone_name]

    print(f"[MoCap] Bone cache: {len(_bone_cache)} bones found")
    for name in list(IK_TARGETS.keys()) + list(IK_POLES.keys()):
        if name in _bone_cache:
            print(f"[MoCap]   IK: {name} ✓")
        else:
            print(f"[MoCap]   IK: {name} ✗ (not found)")


def clear_bone_cache():
    global _bone_cache
    _bone_cache = {}


def apply_pose_to_armature(landmarks: list[dict], armature) -> dict:
    """Apply MediaPipe landmarks to a Rigify armature using IK for limbs."""
    from mathutils import Vector, Quaternion

    _ensure_bone_cache(armature)

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]

    if not _is_calibrated:
        calibrate(landmarks)

    calib_coords = _calib_ik_rest.get("_coords", coords)

    # --- TORSO ROTATION ---
    current_torso_size, current_shoulder_w = _compute_torso_metrics(landmarks)

    shoulder_vec = (coords[12] - coords[11]).normalized()
    hip_vec = (coords[24] - coords[23]).normalized()
    avg_angle = (math.atan2(shoulder_vec.y, shoulder_vec.x) +
                 math.atan2(hip_vec.y, hip_vec.x)) / 2
    delta_angle = avg_angle - _calib_body_angle
    if abs(delta_angle) < math.radians(5):
        delta_angle = 0.0

    if TORSO_BONE in armature.pose.bones:
        pb = armature.pose.bones[TORSO_BONE]
        pb.rotation_mode = "QUATERNION"
        raw = Quaternion(Vector((0, 0, 1)), -delta_angle)
        pb.rotation_quaternion = _smooth_rotation(TORSO_BONE, raw)

    # --- CHEST (FK) ---
    if CHEST_BONE in _bone_cache and CHEST_BONE in _calib_rotations:
        pb = armature.pose.bones[CHEST_BONE]
        rest_bone = _bone_cache[CHEST_BONE]
        mid_hip = (coords[23] + coords[24]) / 2
        mid_shoulder = (coords[11] + coords[12]) / 2
        spine_dir = (mid_shoulder - mid_hip).normalized()
        if spine_dir.length > 1e-6:
            current_abs = _compute_absolute_rotation(rest_bone, spine_dir)
            delta = _calib_rotations[CHEST_BONE].inverted() @ current_abs
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = _smooth_rotation(CHEST_BONE, delta)

    # --- HEAD (FK) ---
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
            delta = _calib_rotations[HEAD_BONE].inverted() @ current_abs
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = _smooth_rotation(HEAD_BONE, delta)

    # --- LIMBS (IK targets) ---
    # Position IK target bones at scaled landmark positions
    # The IK solver automatically bends elbows and knees correctly
    for ik_name, info in IK_TARGETS.items():
        if ik_name not in armature.pose.bones:
            continue
        if ik_name not in _calib_ik_rest:
            continue

        pb = armature.pose.bones[ik_name]
        lm_idx = info["landmark"]

        # Current and calibration landmark positions
        current_pos = coords[lm_idx]

        # Origin: shoulder for arms, hip for legs
        if "shoulder" in info:
            origin_idx = info["shoulder"]
            scale = _calib_arm_scale
        else:
            origin_idx = info["hip"]
            scale = _calib_leg_scale

        current_origin = coords[origin_idx]
        calib_origin = calib_coords[origin_idx]
        calib_target = calib_coords[lm_idx]

        # Delta from calibration, scaled to rig proportions
        delta = current_pos - current_origin - (calib_target - calib_origin)
        scaled_delta = delta * scale

        # Apply as offset from rest position
        rest_pos = _calib_ik_rest[ik_name]
        new_pos = (rest_pos.x + scaled_delta.x,
                   rest_pos.y + scaled_delta.y,
                   rest_pos.z + scaled_delta.z)

        smoothed = _smooth_position(ik_name, new_pos, _smoothing_factor * 0.8)
        pb.location = (smoothed[0] - rest_pos.x,
                       smoothed[1] - rest_pos.y,
                       smoothed[2] - rest_pos.z)

    # --- IK POLE TARGETS (elbow/knee direction) ---
    for pole_name, lm_idx in IK_POLES.items():
        if pole_name not in armature.pose.bones:
            continue
        if pole_name not in _bone_cache:
            continue

        pb = armature.pose.bones[pole_name]
        pole_rest = _bone_cache[pole_name].head_local

        current_pos = coords[lm_idx]
        calib_pos = calib_coords[lm_idx]

        # Determine scale (arms or legs)
        scale = _calib_arm_scale if "arm" in pole_name else _calib_leg_scale

        delta = (current_pos - calib_pos) * scale
        new_pos = (pole_rest.x + delta.x,
                   pole_rest.y + delta.y,
                   pole_rest.z + delta.z)

        smoothed = _smooth_position(pole_name, new_pos, _smoothing_factor * 0.5)
        pb.location = (smoothed[0] - pole_rest.x,
                       smoothed[1] - pole_rest.y,
                       smoothed[2] - pole_rest.z)

    # Root position: hip X (lateral), lowest foot Z (vertical)
    hip_mid = (coords[23] + coords[24]) / 2
    lowest_foot_z = min(coords[27].z, coords[28].z)

    return {
        "_root_xy": (hip_mid.x, 0.0),  # No depth — too noisy from single camera
        "_root_z": lowest_foot_z,
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
