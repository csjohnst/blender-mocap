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
_calib_torso_size = 0.0    # Average torso metric at calibration (for depth)
_calib_shoulder_width = 0.0  # Shoulder width at calibration (for rotation)
_is_calibrated = False
_latest_landmarks = None
_prev_rotations = {}   # bone_name -> Quaternion (previous frame, for smoothing)
_smoothing_factor = 0.4  # 0 = no smoothing (instant), 1 = frozen. 0.4 = natural

# Per-bone angular velocity limits (degrees per frame at ~30fps)
# A human can't rotate their torso 180° instantly — limit to ~30°/s
# Limbs can move faster but still have physical limits
_BONE_MAX_ANGULAR_VELOCITY = {
    TORSO_BONE: math.radians(15),   # ~450°/s — slow, deliberate body turns only
    CHEST_BONE: math.radians(20),   # ~600°/s — spine doesn't whip around
    HEAD_BONE:  math.radians(30),   # ~900°/s — head turns faster than body
}
_DEFAULT_MAX_ANGULAR_VELOCITY = math.radians(120)  # ~3600°/s — limbs need fast response


def _compute_torso_metrics(landmarks_raw: list[dict]) -> tuple[float, float, float]:
    """Compute torso size and rotation from the 4 torso landmarks (raw MediaPipe coords).

    Uses image-plane measurements (X, Y only — not depth Z) which are reliable.

    Returns: (torso_size, shoulder_width_xy, body_rotation_rad)

    torso_size: average of shoulder width, hip width, and torso height in image coords.
                Inversely proportional to depth (closer = bigger).
    shoulder_width_xy: apparent shoulder width in image plane.
    body_rotation_rad: estimated body rotation from shoulder width foreshortening.
    """
    # Use RAW MediaPipe coords (image plane) for size — these are reliable
    ls = landmarks_raw[11]  # left shoulder
    rs = landmarks_raw[12]  # right shoulder
    lh = landmarks_raw[23]  # left hip
    rh = landmarks_raw[24]  # right hip

    # Shoulder width in image plane
    shoulder_w = math.sqrt((rs["x"] - ls["x"])**2 + (rs["y"] - ls["y"])**2)

    # Hip width in image plane
    hip_w = math.sqrt((rh["x"] - lh["x"])**2 + (rh["y"] - lh["y"])**2)

    # Torso height: shoulder midpoint to hip midpoint
    sm_x = (ls["x"] + rs["x"]) / 2
    sm_y = (ls["y"] + rs["y"]) / 2
    hm_x = (lh["x"] + rh["x"]) / 2
    hm_y = (lh["y"] + rh["y"]) / 2
    torso_h = math.sqrt((sm_x - hm_x)**2 + (sm_y - hm_y)**2)

    # Average size metric (stable — combines 3 measurements)
    torso_size = (shoulder_w + hip_w + torso_h) / 3

    return torso_size, shoulder_w, 0.0


def mediapipe_to_blender_coords(lm: dict) -> tuple[float, float, float]:
    bx = lm["x"] - 0.5
    by = lm["z"]              # closer to camera = -Y (character faces -Y)
    bz = -(lm["y"] - 0.5)    # top of image = +Z
    return bx, by, bz


# Calibration 2D lengths for each bone (image-plane distance at calibration)
_calib_2d_lengths = {}


def _reconstruct_3d_direction(parent_lm, child_lm, calib_2d_length):
    """Reconstruct 3D bone direction from 2D image projection + length constraint.

    Uses reliable image-plane X,Y and computes depth via trigonometry:
    - If the bone appears shorter in the image than at calibration,
      it must have rotated out of the image plane
    - depth² = calib_length² - apparent_length² (Pythagorean theorem)
    - Sign of depth from MediaPipe Z (approximate direction is fine)

    This is FAR more accurate than using MediaPipe's Z directly.
    """
    # 2D components from image plane (reliable)
    dx = child_lm["x"] - parent_lm["x"]       # lateral
    dy_img = child_lm["y"] - parent_lm["y"]   # vertical in image

    # 2D projected length
    length_2d = math.sqrt(dx**2 + dy_img**2)

    # Depth from foreshortening
    if length_2d < calib_2d_length and calib_2d_length > 1e-6:
        depth = math.sqrt(max(0, calib_2d_length**2 - length_2d**2))
    else:
        depth = 0.0

    # Sign of depth from MediaPipe Z (direction is OK even if magnitude isn't)
    z_parent = parent_lm["z"]
    z_child = child_lm["z"]
    if z_child > z_parent:  # child farther from camera
        depth = -depth  # negative Y in Blender = away from camera

    # Convert to Blender coordinates
    bx = dx                # image X → Blender X
    by = depth             # computed depth → Blender Y
    bz = -dy_img           # image Y inverted → Blender Z (up)

    length = math.sqrt(bx**2 + by**2 + bz**2)
    if length < 1e-8:
        return (0.0, 0.0, -1.0)  # default: pointing down
    return (bx / length, by / length, bz / length)


def _get_target_dir(coords, mapping, landmarks_raw=None):
    """Get target direction from landmark coordinates.

    If landmarks_raw and calibration 2D lengths are available, uses
    trigonometric depth reconstruction. Otherwise falls back to
    coordinate-based direction.
    """
    from mathutils import Vector

    if mapping.get("type") == "foot":
        heel_idx, toe_idx = mapping["indices"]
        if landmarks_raw and f"foot_{heel_idx}" in _calib_2d_lengths:
            d = _reconstruct_3d_direction(
                landmarks_raw[heel_idx], landmarks_raw[toe_idx],
                _calib_2d_lengths[f"foot_{heel_idx}"])
            return Vector(d)
        return (coords[toe_idx] - coords[heel_idx]).normalized()
    else:
        parent_idx = mapping["parent_idx"]
        child_idx = mapping["child_idx"]
        key = f"{parent_idx}_{child_idx}"
        if landmarks_raw and key in _calib_2d_lengths:
            d = _reconstruct_3d_direction(
                landmarks_raw[parent_idx], landmarks_raw[child_idx],
                _calib_2d_lengths[key])
            return Vector(d)
        return (coords[child_idx] - coords[parent_idx]).normalized()


def _compute_absolute_rotation(rest_bone, target_dir):
    """Compute the absolute rotation for a bone to point at target_dir.

    Returns a Quaternion in the bone's own local frame.

    IMPORTANT: Uses bone.matrix_local (bone-local → armature), NOT rest_local
    (bone-local → parent-local). target_dir is in armature space, so we need
    bone.matrix_local^-1 to convert it to bone-local. Using rest_local would
    map the rotation to the wrong axis for child bones (e.g., elbow bend
    becomes forearm twist).
    """
    from mathutils import Vector

    # Convert armature-space target direction into bone's own local frame
    local_target = (rest_bone.matrix_local.to_3x3().inverted() @ target_dir).normalized()

    # In bone-local frame, the bone points along Y
    local_rest = Vector((0, 1, 0))

    # Rotation from rest to target in the bone's local frame
    return local_rest.rotation_difference(local_target)


def calibrate(landmarks: list[dict]) -> None:
    """Store the current pose as calibration reference."""
    from mathutils import Vector
    global _calib_rotations, _calib_dirs, _calib_body_angle, _is_calibrated
    global _calib_torso_size, _calib_shoulder_width

    coords = [Vector(mediapipe_to_blender_coords(lm)) for lm in landmarks]
    _calib_rotations = {}
    _calib_dirs = {}

    # Store torso metrics for depth estimation
    _calib_torso_size, _calib_shoulder_width, _ = _compute_torso_metrics(landmarks)
    print(f"  torso_size: {_calib_torso_size:.4f}, shoulder_width: {_calib_shoulder_width:.4f}")

    print("[MoCap] === CALIBRATION ===")

    # Store 2D bone lengths at calibration for trigonometric depth reconstruction
    global _calib_2d_lengths
    _calib_2d_lengths = {}
    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if mapping.get("type") == "foot":
            heel_idx, toe_idx = mapping["indices"]
            dx = landmarks[toe_idx]["x"] - landmarks[heel_idx]["x"]
            dy = landmarks[toe_idx]["y"] - landmarks[heel_idx]["y"]
            _calib_2d_lengths[f"foot_{heel_idx}"] = math.sqrt(dx**2 + dy**2)
        else:
            p_idx, c_idx = mapping["parent_idx"], mapping["child_idx"]
            dx = landmarks[c_idx]["x"] - landmarks[p_idx]["x"]
            dy = landmarks[c_idx]["y"] - landmarks[p_idx]["y"]
            _calib_2d_lengths[f"{p_idx}_{c_idx}"] = math.sqrt(dx**2 + dy**2)

    print(f"  2D lengths stored: {len(_calib_2d_lengths)}")

    # Calibrate limb bones using trig-reconstructed directions
    for bone_name in _bone_cache:
        if bone_name not in RIGIFY_BONE_MAP:
            continue
        mapping = RIGIFY_BONE_MAP[bone_name]
        rest_bone = _bone_cache[bone_name]
        target_dir = _get_target_dir(coords, mapping, landmarks)
        if target_dir.length < 1e-6:
            continue
        abs_rot = _compute_absolute_rotation(rest_bone, target_dir)
        _calib_rotations[bone_name] = abs_rot
        _calib_dirs[bone_name] = target_dir.copy()
        print(f"  {bone_name}: target=({target_dir.x:.3f}, {target_dir.y:.3f}, {target_dir.z:.3f})")

    # Calibrate spine (store torso height for lean detection)
    sm_y = (landmarks[11]["y"] + landmarks[12]["y"]) / 2
    hm_y = (landmarks[23]["y"] + landmarks[24]["y"]) / 2
    calibrate._calib_torso_h = abs(hm_y - sm_y)
    sm_x = (landmarks[11]["x"] + landmarks[12]["x"]) / 2
    hm_x = (landmarks[23]["x"] + landmarks[24]["x"]) / 2
    calibrate._calib_lateral = sm_x - hm_x
    print(f"  chest: torso_h={calibrate._calib_torso_h:.4f} lateral={calibrate._calib_lateral:.4f}")

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
    global _is_calibrated, _prev_rotations
    _is_calibrated = False
    _prev_rotations = {}


def set_smoothing(value: float) -> None:
    """Set rotation smoothing factor. 0 = instant, 1 = frozen."""
    global _smoothing_factor
    _smoothing_factor = max(0.0, min(0.95, value))


def _smooth_rotation(bone_name, new_rot):
    """Apply temporal smoothing and angular velocity clamping to a rotation.

    Uses per-bone velocity limits — torso/chest/head are much slower
    than limbs to prevent unrealistic instant body flips.
    """
    from mathutils import Quaternion

    if bone_name not in _prev_rotations:
        _prev_rotations[bone_name] = new_rot.copy()
        return new_rot

    prev = _prev_rotations[bone_name]

    # Per-bone angular velocity clamp
    max_vel = _BONE_MAX_ANGULAR_VELOCITY.get(bone_name, _DEFAULT_MAX_ANGULAR_VELOCITY)
    angle = prev.rotation_difference(new_rot).angle
    if angle > max_vel:
        clamped_t = max_vel / angle
        new_rot = prev.slerp(new_rot, clamped_t)

    # Per-bone smoothing: torso/chest get heavier smoothing
    if bone_name in (TORSO_BONE, CHEST_BONE):
        # Torso: very heavy smoothing — 80% previous, 20% new
        blend = 0.2
    elif bone_name == HEAD_BONE:
        # Head: moderate smoothing
        blend = 0.4
    else:
        # Limbs: use user's smoothing setting
        blend = 1.0 - _smoothing_factor

    smoothed = prev.slerp(new_rot, blend)
    _prev_rotations[bone_name] = smoothed.copy()
    return smoothed


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

    # --- DEPTH from torso geometry ---
    current_torso_size, current_shoulder_w, _ = _compute_torso_metrics(landmarks)

    if current_torso_size > 1e-6 and _calib_torso_size > 1e-6:
        depth_ratio = _calib_torso_size / current_torso_size
    else:
        depth_ratio = 1.0

    # --- TORSO ROTATION ---
    # Use ONLY the image-plane shoulder/hip line angle (stable and reliable).
    # Do NOT use MediaPipe Z for rotation sign — Z is noisy and causes
    # instant 180° body flips when head movement confuses the depth estimate.
    # The model should essentially stay facing the camera unless the person
    # deliberately turns their body.
    if TORSO_BONE in armature.pose.bones:
        pb = armature.pose.bones[TORSO_BONE]

        shoulder_vec = (coords[12] - coords[11]).normalized()
        hip_vec = (coords[24] - coords[23]).normalized()
        avg_angle = (math.atan2(shoulder_vec.y, shoulder_vec.x) +
                     math.atan2(hip_vec.y, hip_vec.x)) / 2
        delta_angle = avg_angle - _calib_body_angle

        # Soft dead zone: fade rotation in between 3° and 8°
        # Prevents sudden jumps when crossing the threshold
        abs_angle = abs(delta_angle)
        dead_min = math.radians(3)
        dead_max = math.radians(8)
        if abs_angle < dead_min:
            delta_angle = 0.0
        elif abs_angle < dead_max:
            # Gradual fade-in from 0 to full
            fade = (abs_angle - dead_min) / (dead_max - dead_min)
            delta_angle = delta_angle * fade

        pb.rotation_mode = "QUATERNION"
        raw = Quaternion(Vector((0, 0, 1)), -delta_angle)
        # Heavy smoothing + low velocity limit prevents instant flips
        pb.rotation_quaternion = _smooth_rotation(TORSO_BONE, raw)

    # --- CHEST (spine lean) ---
    # Forward lean computed from IMAGE PLANE torso height (reliable)
    # When leaning forward, torso appears shorter due to foreshortening
    # cos(lean_angle) = current_torso_height / calib_torso_height
    if CHEST_BONE in armature.pose.bones:
        pb = armature.pose.bones[CHEST_BONE]

        # Torso height in image plane (raw MediaPipe Y coords)
        sm_y = (landmarks[11]["y"] + landmarks[12]["y"]) / 2
        hm_y = (landmarks[23]["y"] + landmarks[24]["y"]) / 2
        current_torso_h = abs(hm_y - sm_y)

        # Lateral tilt: shoulder midpoint X offset from hip midpoint X
        sm_x = (landmarks[11]["x"] + landmarks[12]["x"]) / 2
        hm_x = (landmarks[23]["x"] + landmarks[24]["x"]) / 2
        lateral_offset = sm_x - hm_x  # positive = leaning right

        # Store calibration torso height on first calibration
        if not hasattr(calibrate, '_calib_torso_h'):
            calibrate._calib_torso_h = current_torso_h
            calibrate._calib_lateral = sm_x - hm_x

        calib_h = getattr(calibrate, '_calib_torso_h', current_torso_h)
        calib_lat = getattr(calibrate, '_calib_lateral', 0.0)

        # Forward lean angle from foreshortening
        if calib_h > 1e-6:
            cos_lean = min(1.0, current_torso_h / calib_h)
            lean_angle = math.acos(cos_lean)
            # Determine lean direction from shoulder Z vs hip Z (depth)
            shoulder_depth = (landmarks[11]["z"] + landmarks[12]["z"]) / 2
            hip_depth = (landmarks[23]["z"] + landmarks[24]["z"]) / 2
            if shoulder_depth < hip_depth:  # shoulders closer to camera = leaning forward
                lean_angle = -lean_angle
        else:
            lean_angle = 0.0

        # Lateral tilt delta
        lateral_delta = lateral_offset - calib_lat

        # Apply as X rotation (forward/back lean) and Y rotation (side tilt)
        lean_quat = Quaternion(Vector((1, 0, 0)), lean_angle)
        tilt_quat = Quaternion(Vector((0, 1, 0)), lateral_delta * 3.0)  # scale for visibility
        chest_rot = lean_quat @ tilt_quat

        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = _smooth_rotation(CHEST_BONE, chest_rot)

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
            pb.rotation_quaternion = _smooth_rotation(HEAD_BONE, delta)

    # --- LIMBS ---
    # SIMPLE APPROACH: For every bone, compute world-space direction delta
    # from calibration and set directly. No bone-local conversion — the
    # calibration delta cancels out the bone's rest orientation so
    # rotation_quaternion receives the correct relative change.
    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in armature.pose.bones:
            continue
        if bone_name not in _calib_dirs:
            continue

        pb = armature.pose.bones[bone_name]
        target_dir = _get_target_dir(coords, mapping, landmarks)
        if target_dir.length < 1e-6:
            continue

        calib_dir = _calib_dirs[bone_name]

        # World-space rotation from calibration direction to current direction
        # This IS the movement the person made — no conversion needed
        delta = calib_dir.rotation_difference(target_dir)

        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = _smooth_rotation(bone_name, delta)

    # Root position:
    # X: hip midpoint X from image plane (lateral movement — reliable)
    # Y: depth from torso size ratio (trigonometric)
    # Z: lowest foot from image plane (vertical/jumping)
    hip_mid_x = (landmarks[23]["x"] + landmarks[24]["x"]) / 2 - 0.5  # centered
    lowest_foot_z = min(coords[27].z, coords[28].z)
    computed_depth_y = -(depth_ratio - 1.0) if depth_ratio != 1.0 else 0.0

    return {
        "_root_xy": (hip_mid_x, computed_depth_y),
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
