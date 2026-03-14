# tests/test_rigify_mapper.py
import math
import pytest
from blender_mocap.rigify_mapper import (
    mediapipe_to_blender_coords,
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
        assert by_near < by_far  # Closer to camera = -Y (character faces camera)


class TestBoneMap:
    def test_has_required_bones(self):
        required = ["upper_arm_fk.L", "forearm_fk.L", "upper_arm_fk.R", "forearm_fk.R",
                     "thigh_fk.L", "shin_fk.L", "thigh_fk.R", "shin_fk.R",
                     "foot_fk.L", "foot_fk.R"]
        for bone in required:
            assert bone in RIGIFY_BONE_MAP, f"Missing mapping for {bone}"

    def test_mapping_has_landmark_indices(self):
        for bone_name, mapping in RIGIFY_BONE_MAP.items():
            assert "parent_idx" in mapping or "indices" in mapping, \
                f"Bone {bone_name} missing landmark indices"

    def test_limb_bones_have_parent_child(self):
        for bone_name, mapping in RIGIFY_BONE_MAP.items():
            if mapping.get("type") != "foot":
                assert "parent_idx" in mapping and "child_idx" in mapping
                assert isinstance(mapping["parent_idx"], int)
                assert isinstance(mapping["child_idx"], int)

    def test_foot_bones_have_indices(self):
        for bone_name, mapping in RIGIFY_BONE_MAP.items():
            if mapping.get("type") == "foot":
                assert "indices" in mapping
                assert len(mapping["indices"]) == 2
