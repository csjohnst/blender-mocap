# tests/test_rigify_mapper.py
import math
import pytest
from blender_mocap.rigify_mapper import (
    mediapipe_to_blender_coords,
    compute_bone_rotation,
    RIGIFY_BONE_MAP,
)


class TestCoordinateTransform:
    def test_center_point(self):
        lm = {"x": 0.5, "y": 0.5, "z": 0.0}
        bx, by, bz = mediapipe_to_blender_coords(lm)
        assert bx == pytest.approx(0.0)  # Centered
        assert bz == pytest.approx(0.0)  # Centered
        assert by == pytest.approx(0.0)  # No depth

    def test_x_mapping(self):
        lm_left = {"x": 0.0, "y": 0.5, "z": 0.0}
        lm_right = {"x": 1.0, "y": 0.5, "z": 0.0}
        bx_l, _, _ = mediapipe_to_blender_coords(lm_left)
        bx_r, _, _ = mediapipe_to_blender_coords(lm_right)
        assert bx_l < bx_r  # Left is negative X, right is positive X

    def test_y_is_negative_z(self):
        lm_top = {"x": 0.5, "y": 0.0, "z": 0.0}
        lm_bottom = {"x": 0.5, "y": 1.0, "z": 0.0}
        _, _, bz_top = mediapipe_to_blender_coords(lm_top)
        _, _, bz_bottom = mediapipe_to_blender_coords(lm_bottom)
        assert bz_top > bz_bottom  # Top of image = higher Z in Blender

    def test_depth_mapping(self):
        lm_near = {"x": 0.5, "y": 0.5, "z": -0.5}
        lm_far = {"x": 0.5, "y": 0.5, "z": 0.5}
        _, by_near, _ = mediapipe_to_blender_coords(lm_near)
        _, by_far, _ = mediapipe_to_blender_coords(lm_far)
        assert by_near > by_far  # Closer to camera = forward (+Y)


class TestBoneRotation:
    def test_identity_rotation(self):
        # Vector pointing in same direction as rest should give identity quaternion
        rest_vec = (0.0, 0.0, 1.0)
        target_vec = (0.0, 0.0, 1.0)
        q = compute_bone_rotation(rest_vec, target_vec)
        # Identity quaternion: (1, 0, 0, 0) or close
        assert q[0] == pytest.approx(1.0, abs=0.01)
        assert abs(q[1]) < 0.01
        assert abs(q[2]) < 0.01
        assert abs(q[3]) < 0.01

    def test_90_degree_rotation(self):
        rest_vec = (0.0, 0.0, 1.0)
        target_vec = (1.0, 0.0, 0.0)
        q = compute_bone_rotation(rest_vec, target_vec)
        # Should produce a 90-degree rotation
        angle = 2 * math.acos(min(abs(q[0]), 1.0))
        assert angle == pytest.approx(math.pi / 2, abs=0.1)


class TestBoneMap:
    def test_has_required_bones(self):
        required = ["upper_arm.L", "forearm.L", "upper_arm.R", "forearm.R",
                     "thigh.L", "shin.L", "thigh.R", "shin.R",
                     "foot.L", "foot.R"]
        for bone in required:
            assert bone in RIGIFY_BONE_MAP, f"Missing mapping for {bone}"

    def test_mapping_has_landmark_indices(self):
        for bone_name, mapping in RIGIFY_BONE_MAP.items():
            assert "parent_idx" in mapping or "indices" in mapping, \
                f"Bone {bone_name} missing landmark indices"
