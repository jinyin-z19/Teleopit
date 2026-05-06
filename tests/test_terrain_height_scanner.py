"""Tests for teleopit.sim.terrain_height_scanner — TerrainHeightScanner."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_mujoco


# A minimal MuJoCo model with a plane floor for ray casting tests.
_MINIMAL_SCENE_XML = """<mujoco model="terrain_test">
  <worldbody>
    <geom name="floor" size="10 10 0.1" pos="0 0 0" type="plane"/>
  </worldbody>
</mujoco>"""

# A scene with a static box on the ground to test non-zero hit heights.
_BOX_SCENE_XML = """<mujoco model="terrain_test_box">
  <worldbody>
    <geom name="floor" size="10 10 0.1" pos="0 0 0" type="plane"/>
    <geom name="box" type="box" size="0.5 0.5 0.1" pos="0.5 0 0.1"/>
  </worldbody>
</mujoco>"""


def _make_model_data(xml: str):
    """Create a MuJoCo model + data pair from an XML string."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@requires_mujoco
class TestTerrainHeightScannerInit:
    """Test TerrainHeightScanner initialization."""

    def test_init_with_default_probes(self):
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)
        scanner = TerrainHeightScanner(model, data)

        assert scanner.num_points == 25  # default 5x5 grid
        assert len(scanner.probe_offsets) == 25

    def test_init_with_custom_probes(self):
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)
        custom = [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0)]
        scanner = TerrainHeightScanner(model, data, probe_offsets=custom)

        assert scanner.num_points == 3
        assert scanner.probe_offsets == [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0)]

    def test_init_with_custom_ray_params(self):
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)
        scanner = TerrainHeightScanner(
            model, data,
            ray_start_height=2.0,
            ray_length=10.0,
            geom_group_filter=None,
        )

        assert scanner.num_points == 25

    def test_probe_offsets_property_returns_copy(self):
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)
        scanner = TerrainHeightScanner(model, data)
        offsets = scanner.probe_offsets
        offsets.append((99.0, 99.0))
        assert len(scanner.probe_offsets) == 25  # original unchanged


@requires_mujoco
class TestTerrainHeightScannerScan:
    """Test scan() method."""

    def test_scan_on_plane_returns_zeros(self):
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)

        base_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        scanner = TerrainHeightScanner(model, data)
        heights = scanner.scan(base_pos, base_quat)

        assert heights.shape == (25,)
        assert heights.dtype == np.float32
        # Plane at z=0, all rays should hit near 0
        np.testing.assert_allclose(heights, 0.0, atol=0.1)

    def test_scan_with_base_rotation(self):
        """Rays should hit floor even when base is pitched/rolled."""
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)

        base_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # 30° pitch around Y axis
        angle = np.pi / 6
        base_quat = np.array(
            [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0],
            dtype=np.float64,
        )

        scanner = TerrainHeightScanner(model, data, probe_offsets=[(0.0, 0.0)])
        heights = scanner.scan(base_pos, base_quat)

        assert heights.shape == (1,)
        # The center probe (0,0) rotated by pitch — still roughly hits at z=0
        np.testing.assert_allclose(heights[0], 0.0, atol=0.1)

    def test_scan_with_translated_base(self):
        """Heights should still be ~0 when base is shifted."""
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)

        base_pos = np.array([5.0, -3.0, 2.0], dtype=np.float64)
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        scanner = TerrainHeightScanner(model, data, probe_offsets=[(0.0, 0.0)])
        heights = scanner.scan(base_pos, base_quat)

        np.testing.assert_allclose(heights[0], 0.0, atol=0.1)

    def test_scan_hits_box_above_plane(self):
        """When a ray hits a static box above the floor, return the box's top face Z."""
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_BOX_SCENE_XML)

        # Static box geom at pos (0.5, 0, 0.1), size (0.5, 0.5, 0.1)
        # Box top face is at z = 0.1 + 0.1 = 0.2
        base_pos = np.array([0.5, 0.0, 1.0], dtype=np.float64)
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Probe directly under base center
        scanner = TerrainHeightScanner(
            model, data,
            probe_offsets=[(0.0, 0.0), (0.5, 0.0)],
        )
        heights = scanner.scan(base_pos, base_quat)

        # (0.0, 0.0) probe: base at (0.5, 0, 1.0), offset rotates to world (0.5, 0, 1.0),
        # ray down hits box top ~0.4
        assert heights[0] > 0.0, f"Expected hit on box, got {heights[0]}"
        assert heights[0] < 1.0, f"Expected hit below base, got {heights[0]}"

    def test_scan_ray_miss_returns_zero(self):
        """When no geometry is hit, return 0.0."""
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)

        # Start ray very high, but it should still hit the plane.
        # To test miss, use a very short ray_length that won't reach the floor.
        base_pos = np.array([0.0, 0.0, 100.0], dtype=np.float64)
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        scanner = TerrainHeightScanner(
            model, data,
            probe_offsets=[(0.0, 0.0)],
            ray_length=0.1,  # very short — won't reach the floor from z=100
        )
        heights = scanner.scan(base_pos, base_quat)

        assert heights[0] == 0.0

    def test_scan_custom_num_points(self):
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)

        base_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for n in [1, 4, 9]:
            offsets = [(float(i) * 0.1, 0.0) for i in range(n)]
            scanner = TerrainHeightScanner(model, data, probe_offsets=offsets)
            heights = scanner.scan(base_pos, base_quat)
            assert heights.shape == (n,)
            assert heights.dtype == np.float32

    def test_scan_returns_float32_zero_copy_friendly(self):
        from teleopit.sim.terrain_height_scanner import TerrainHeightScanner
        model, data = _make_model_data(_MINIMAL_SCENE_XML)

        base_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        scanner = TerrainHeightScanner(model, data)
        heights = scanner.scan(base_pos, base_quat)

        assert heights.dtype == np.float32
        assert np.all(np.isfinite(heights))


@requires_mujoco
class TestTerrainHeightScannerQuatRotate:
    """Test the internal _quat_rotate_vec helper."""

    def test_identity_quaternion(self):
        from teleopit.sim.terrain_height_scanner import _quat_rotate_vec
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _quat_rotate_vec(q, v)
        np.testing.assert_allclose(result, v, atol=1e-10)

    def test_rotate_z_90(self):
        from teleopit.sim.terrain_height_scanner import _quat_rotate_vec
        # 90° around Z
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], dtype=np.float64)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        result = _quat_rotate_vec(q, v)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_rotate_x_90(self):
        from teleopit.sim.terrain_height_scanner import _quat_rotate_vec
        # 90° around X
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0], dtype=np.float64)
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        result = _quat_rotate_vec(q, v)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-6)

    def test_non_unit_quaternion_normalized(self):
        from teleopit.sim.terrain_height_scanner import _quat_rotate_vec
        q = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)  # non-unit
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        result = _quat_rotate_vec(q, v)
        np.testing.assert_allclose(result, v, atol=1e-10)
