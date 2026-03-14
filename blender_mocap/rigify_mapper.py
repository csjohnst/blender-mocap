# blender_mocap/rigify_mapper.py
"""Maps MediaPipe pose landmarks to Rigify armature bone rotations.

Coordinate transform: MediaPipe (image coords) -> Blender (right-handed, Z-up).
Rotation calculation: compute direction vectors between landmarks, convert to
quaternion rotations relative to each bone's rest pose.
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


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length < 1e-8:
        return (0.0, 0.0, 1.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def compute_bone_rotation(
    rest_vec: tuple[float, float, float],
    target_vec: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Compute quaternion (w, x, y, z) that rotates rest_vec to target_vec."""
    rest_n = _normalize(rest_vec)
    target_n = _normalize(target_vec)

    dot = _dot(rest_n, target_n)
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9999:
        return (1.0, 0.0, 0.0, 0.0)

    if dot < -0.9999:
        # 180-degree rotation -- pick an arbitrary perpendicular axis
        if abs(rest_n[0]) < 0.9:
            perp = _normalize(_cross(rest_n, (1.0, 0.0, 0.0)))
        else:
            perp = _normalize(_cross(rest_n, (0.0, 1.0, 0.0)))
        return (0.0, perp[0], perp[1], perp[2])

    axis = _normalize(_cross(rest_n, target_n))
    half_angle = math.acos(dot) / 2.0
    s = math.sin(half_angle)
    w = math.cos(half_angle)
    return (w, axis[0] * s, axis[1] * s, axis[2] * s)


# Mapping: Rigify bone name -> landmark indices used to compute its rotation
# parent_idx/child_idx: direction vector from parent to child landmark
RIGIFY_BONE_MAP = {
    # Arms
    "upper_arm.L": {"parent_idx": 11, "child_idx": 13},
    "forearm.L":   {"parent_idx": 13, "child_idx": 15},
    "hand.L":      {"parent_idx": 15, "child_idx": 19},  # wrist to index finger tip approx
    "upper_arm.R": {"parent_idx": 12, "child_idx": 14},
    "forearm.R":   {"parent_idx": 14, "child_idx": 16},
    "hand.R":      {"parent_idx": 16, "child_idx": 20},
    # Legs
    "thigh.L":     {"parent_idx": 23, "child_idx": 25},
    "shin.L":      {"parent_idx": 25, "child_idx": 27},
    "thigh.R":     {"parent_idx": 24, "child_idx": 26},
    "shin.R":      {"parent_idx": 26, "child_idx": 28},
    # Feet -- use heel-to-toe vector
    "foot.L":      {"indices": [29, 31], "type": "foot"},  # heel to foot index
    "foot.R":      {"indices": [30, 32], "type": "foot"},  # heel to foot index
}

# Spine and head use composite calculations (multiple landmarks)
SPINE_LANDMARKS = {
    "shoulders": (11, 12),
    "hips": (23, 24),
}

HEAD_LANDMARKS = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
}


def compute_limb_rotations(
    landmarks: list[dict],
    bone_rest_vectors: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float, float]]:
    """Compute rotation quaternions for all mapped bones.

    Args:
        landmarks: 33 MediaPipe landmarks with x, y, z, visibility.
        bone_rest_vectors: dict of bone_name -> rest pose direction vector.

    Returns:
        dict of bone_name -> (w, x, y, z) quaternion.
    """
    coords = [mediapipe_to_blender_coords(lm) for lm in landmarks]
    rotations = {}

    for bone_name, mapping in RIGIFY_BONE_MAP.items():
        if bone_name not in bone_rest_vectors:
            continue

        rest_vec = bone_rest_vectors[bone_name]

        if mapping.get("type") == "foot":
            # Foot: heel to toe vector
            heel_idx, toe_idx = mapping["indices"]
            target_vec = (
                coords[toe_idx][0] - coords[heel_idx][0],
                coords[toe_idx][1] - coords[heel_idx][1],
                coords[toe_idx][2] - coords[heel_idx][2],
            )
        else:
            parent = coords[mapping["parent_idx"]]
            child = coords[mapping["child_idx"]]
            target_vec = (
                child[0] - parent[0],
                child[1] - parent[1],
                child[2] - parent[2],
            )

        rotations[bone_name] = compute_bone_rotation(rest_vec, target_vec)

    # Torso orientation from shoulders/hips
    if "spine" in bone_rest_vectors:
        l_shoulder = coords[11]
        r_shoulder = coords[12]
        l_hip = coords[23]
        r_hip = coords[24]
        # Spine direction: hip midpoint to shoulder midpoint
        mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2, (l_hip[2] + r_hip[2]) / 2)
        mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2, (l_shoulder[2] + r_shoulder[2]) / 2)
        spine_vec = (mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1], mid_shoulder[2] - mid_hip[2])
        rotations["spine"] = compute_bone_rotation(bone_rest_vectors["spine"], spine_vec)

    # Head orientation from nose and ears
    if "spine.006" in bone_rest_vectors:
        nose = coords[0]
        l_ear = coords[7]
        r_ear = coords[8]
        ear_mid = ((l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2, (l_ear[2] + r_ear[2]) / 2)
        head_vec = (nose[0] - ear_mid[0], nose[1] - ear_mid[1], nose[2] - ear_mid[2])
        rotations["spine.006"] = compute_bone_rotation(bone_rest_vectors["spine.006"], head_vec)

    # Root position (hip midpoint)
    l_hip = coords[23]
    r_hip = coords[24]
    rotations["_root_position"] = (
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2,
        (l_hip[2] + r_hip[2]) / 2,
    )

    return rotations
