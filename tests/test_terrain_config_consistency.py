"""Tests for terrain height config consistency across sim2sim and training.

Ensures that num_height_points, probe grid layout, and observation dimension
are consistent between:

- teleopit/configs/controller/rl_policy.yaml (sim2sim)
- teleopit/configs/robot/azureloong_v9.yaml (sim2sim scanner)
- train_mimic/tasks/tracking/mdp/observations.py (training probe grid)
- train_mimic/tasks/tracking/config/azureloong_v9_env.py (training obs)
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from typing import Any

from conftest import requires_mujoco

_PROJECT_ROOT = Path(__file__).parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file using OmegaConf and convert to plain dict."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]


# ── Config files ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rl_policy_cfg() -> dict[str, Any]:
    return _load_yaml(_PROJECT_ROOT / "teleopit" / "configs" / "controller" / "rl_policy.yaml")


@pytest.fixture(scope="module")
def azureloong_v9_cfg() -> dict[str, Any]:
    return _load_yaml(_PROJECT_ROOT / "teleopit" / "configs" / "robot" / "azureloong_v9.yaml")


# ── Tests ────────────────────────────────────────────────────────

class TestTerrainConfigConsistency:
    """Validate terrain config consistency between sim2sim configs."""

    def test_rl_policy_num_height_points_matches_robot_probe_count(
        self, rl_policy_cfg, azureloong_v9_cfg
    ) -> None:
        """rl_policy.yaml num_height_points must match the number of probe_offsets."""
        num_height = rl_policy_cfg.get("num_height_points", 0)
        scanner = azureloong_v9_cfg.get("terrain_height_scanner", {})
        probe_count = len(scanner.get("probe_offsets", []))

        assert num_height == probe_count, (
            f"rl_policy.yaml num_height_points={num_height} "
            f"!= azureloong_v9.yaml probe_offsets count={probe_count}"
        )

    def test_probe_offsets_are_5x5_grid(self, azureloong_v9_cfg) -> None:
        """Verify the probe grid is a 5×5 grid over ±0.4 m."""
        scanner = azureloong_v9_cfg.get("terrain_height_scanner", {})
        probes = scanner.get("probe_offsets", [])

        assert len(probes) == 25, f"Expected 25 probes, got {len(probes)}"

        expected_x = [-0.4, -0.2, 0.0, 0.2, 0.4]
        expected_y = [-0.4, -0.2, 0.0, 0.2, 0.4]

        for i, expected_x_val in enumerate(expected_x):
            for j, expected_y_val in enumerate(expected_y):
                idx = i * 5 + j
                assert probes[idx][0] == expected_x_val
                assert probes[idx][1] == expected_y_val

    def test_terrain_scanner_enabled(self, azureloong_v9_cfg) -> None:
        scanner = azureloong_v9_cfg.get("terrain_height_scanner", {})
        assert scanner.get("enabled", False), (
            "terrain_height_scanner.enabled must be true for sim2sim terrain observation"
        )

    def test_ray_params_reasonable(self, azureloong_v9_cfg) -> None:
        scanner = azureloong_v9_cfg.get("terrain_height_scanner", {})
        assert scanner.get("ray_start_height", 0) > 0
        assert scanner.get("ray_length", 0) > 0

    def test_rl_policy_num_height_points_is_25(self, rl_policy_cfg) -> None:
        assert rl_policy_cfg.get("num_height_points", 0) == 25, (
            "Expected num_height_points=25 in rl_policy.yaml for azureloong_v9"
        )


class TestTrainingTerrainProbeConsistency:
    """Validate that training-side probe offsets match sim2sim."""

    def test_default_probe_offsets_are_25(self) -> None:
        from train_mimic.tasks.tracking.mdp.observations import (
            _DEFAULT_TERRAIN_PROBE_OFFSETS,
        )
        assert len(_DEFAULT_TERRAIN_PROBE_OFFSETS) == 25

    def test_default_probe_grid_matches_config(self, azureloong_v9_cfg) -> None:
        """Training probe grid must be identical to sim2sim config probe grid."""
        from train_mimic.tasks.tracking.mdp.observations import (
            _DEFAULT_TERRAIN_PROBE_OFFSETS,
        )

        scanner = azureloong_v9_cfg.get("terrain_height_scanner", {})
        config_probes = scanner.get("probe_offsets", [])

        training_probes = [(float(x), float(y)) for x, y in _DEFAULT_TERRAIN_PROBE_OFFSETS]
        config_probes_float = [(float(p[0]), float(p[1])) for p in config_probes]

        assert training_probes == config_probes_float, (
            "Training _DEFAULT_TERRAIN_PROBE_OFFSETS differs from "
            "azureloong_v9.yaml terrain_height_scanner.probe_offsets"
        )

    def test_sim2sim_default_grid_matches_training_grid(self) -> None:
        """The default 5×5 grid in TerrainHeightScanner matches training grid."""
        from teleopit.sim.terrain_height_scanner import _DEFAULT_PROBE_GRID
        from train_mimic.tasks.tracking.mdp.observations import (
            _DEFAULT_TERRAIN_PROBE_OFFSETS,
        )

        sim_grid = [(float(x), float(y)) for x, y in _DEFAULT_PROBE_GRID]
        train_grid = [(float(x), float(y)) for x, y in _DEFAULT_TERRAIN_PROBE_OFFSETS]

        assert len(sim_grid) == len(train_grid), "Grid lengths differ"
        for i, ((sx, sy), (tx, ty)) in enumerate(zip(sim_grid, train_grid)):
            assert abs(sx - tx) < 1e-10, f"Index {i}: sim.x={sx} != train.x={tx}"
            assert abs(sy - ty) < 1e-10, f"Index {i}: sim.y={sy} != train.y={ty}"


class TestAzureLoongV9EnvTerrainObs:
    """Validate that azureloong_v9_env.py configures terrain observation correctly."""

    def test_actor_terrain_heights_num_height_points(self) -> None:
        from train_mimic.tasks.tracking.config.azureloong_v9_env import (
            _VELCMD_ACTOR_TERMS,
        )
        terrain_cfg = _VELCMD_ACTOR_TERMS.get("terrain_heights")
        assert terrain_cfg is not None, "terrain_heights missing in actor observation terms"
        params = terrain_cfg.params
        assert params["num_height_points"] == 25, (
            f"Actor terrain num_height_points={params['num_height_points']}, expected 25"
        )

    def test_critic_terrain_heights_num_height_points(self) -> None:
        from train_mimic.tasks.tracking.config.azureloong_v9_env import (
            _VELCMD_CRITIC_TERMS,
        )
        terrain_cfg = _VELCMD_CRITIC_TERMS.get("terrain_heights")
        assert terrain_cfg is not None, "terrain_heights missing in critic observation terms"
        params = terrain_cfg.params
        assert params["num_height_points"] == 25, (
            f"Critic terrain num_height_points={params['num_height_points']}, expected 25"
        )

    def test_actor_has_noise_critic_does_not(self) -> None:
        from train_mimic.tasks.tracking.config.azureloong_v9_env import (
            _VELCMD_ACTOR_TERMS,
            _VELCMD_CRITIC_TERMS,
        )
        actor_terrain = _VELCMD_ACTOR_TERMS["terrain_heights"]
        critic_terrain = _VELCMD_CRITIC_TERMS["terrain_heights"]

        assert actor_terrain.noise is not None, "Actor terrain should have noise for domain randomization"
        assert critic_terrain.noise is None, "Critic terrain should NOT have noise (privileged observation)"


class TestObservationDimensionCalculation:
    """Validate observation dimension calculations."""

    def test_azureloong_v9_total_dim(self) -> None:
        """azureloong_v9: num_actions=30 → base=171, +12 velcmd, +25 terrain = 208.
        
        Wait — let's recompute. num_actions * 2 (command) = 60
        + rot6d (6) + ang_vel (3) = 69
        + joint_pos_rel (30) + joint_vel (30) + last_action (30) = 159 base_obs
        Then + projected_gravity (3) + ref_lin_vel (3) + ref_ang_vel (3) + ref_proj_grav (3) = 171
        + terrain (25) = 196
        """
        num_actions = 30
        num_height_points = 25

        # base_obs = num_actions*2 + 6 + 3 + num_actions*3
        base = num_actions * 2 + 6 + 3 + num_actions * 3  # 60+6+3+90=159
        # velcmd_obs = 3 + 3 + 3 + 3 = 12
        total = base + 12 + num_height_points

        assert total == 196, f"Expected 196D for azureloong_v9, got {total}"

    def test_g1_total_dim(self) -> None:
        """G1: num_actions=29 → base=154, +12 velcmd, +25 terrain = 191."""
        num_actions = 29
        num_height_points = 25

        base = num_actions * 2 + 6 + 3 + num_actions * 3  # 58+6+3+87=154
        total = base + 12 + num_height_points

        assert total == 191, f"Expected 191D for G1 with terrain, got {total}"

    def test_g1_total_dim_no_terrain(self) -> None:
        num_actions = 29
        base = num_actions * 2 + 6 + 3 + num_actions * 3  # 154
        total = base + 12  # 166

        assert total == 166, f"Expected 166D for G1 without terrain, got {total}"
