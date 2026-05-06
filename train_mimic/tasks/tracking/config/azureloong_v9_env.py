"""Environment builder for the General-Tracking-AzureloongV9 task."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from train_mimic.tasks.tracking import mdp
from train_mimic.tasks.tracking.config.azureloong_v9_constants import (
    AZURELOONG_V9_ACTION_SCALE,
    build_terrain_cfg,
    get_azureloong_v9_robot_cfg,
)
from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE
from train_mimic.tasks.tracking.mdp import MotionCommandCfg
from train_mimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# Tracking body names for azureloong_v9 (all 31 bodies except world).
# These are the bodies whose position/orientation is tracked for motion imitation.
_TRACKING_BODY_NAMES = (
    "base_link",
    "link_arm_l_01",
    "link_arm_l_02",
    "link_arm_l_03",
    "link_arm_l_04",
    "link_arm_l_05",
    "link_arm_l_06",
    "link_arm_l_07",
    "link_arm_r_01",
    "link_arm_r_02",
    "link_arm_r_03",
    "link_arm_r_04",
    "link_arm_r_05",
    "link_arm_r_06",
    "link_arm_r_07",
    "link_head_yaw",
    "link_head_pitch",
    "link_waist_roll",
    "link_waist_yaw",
    "link_hip_l_pitch",
    "link_hip_l_roll",
    "link_hip_l_yaw",
    "link_knee_l_pitch",
    "link_ankle_l_pitch",
    "link_ankle_l_roll",
    "link_hip_r_pitch",
    "link_hip_r_roll",
    "link_hip_r_yaw",
    "link_knee_r_pitch",
    "link_ankle_r_pitch",
    "link_ankle_r_roll",
)


def _apply_play_mode_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"


def _add_history_obs_groups(
    cfg: ManagerBasedRlEnvCfg, history_length: int = 10
) -> None:
    cfg.observations["actor_history"] = ObservationGroupCfg(
        terms=deepcopy(cfg.observations["actor"].terms),
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
        history_length=history_length,
        flatten_history_dim=False,
    )
    cfg.observations["critic_history"] = ObservationGroupCfg(
        terms=deepcopy(cfg.observations["critic"].terms),
        concatenate_terms=True,
        enable_corruption=False,
        history_length=history_length,
        flatten_history_dim=False,
    )


_VELCMD_ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ref_base_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_base_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_base_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_base_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
    "terrain_heights": ObservationTermCfg(
        func=mdp.terrain_heights,
        params={"command_name": "motion", "num_height_points": 25},
        noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
}

_VELCMD_CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "ref_base_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_base_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_base_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_base_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
    "terrain_heights": ObservationTermCfg(
        func=mdp.terrain_heights,
        params={"command_name": "motion", "num_height_points": 25},
    ),
}


def make_azureloong_v9_tracking_env_cfg(
    *, play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the General-Tracking-AzureloongV9 training env."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_azureloong_v9_robot_cfg()}
    cfg.scene.terrain = build_terrain_cfg()
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = AZURELOONG_V9_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "base_link"
    motion_cmd.body_names = _TRACKING_BODY_NAMES
    motion_cmd.motion_file = DEFAULT_TRAIN_MOTION_FILE
    motion_cmd.sampling_mode = "uniform"
    motion_cmd.window_steps = (0,)

    # azureloong_v9 uses default MuJoCo frame sensors (baselink-*).
    # Override IMU sensor names to match the available sensor names.
    _AZ_IMU_ANG_VEL_SENSOR = "robot/baselink-gyro"
    _AZ_IMU_LIN_VEL_SENSOR = "robot/baselink-velocity"

    # Fix IMU sensor references in observation terms.
    # base_ang_vel is in both actor and critic observations.
    if "base_ang_vel" in cfg.observations["actor"].terms:
        cfg.observations["actor"].terms["base_ang_vel"].params[
            "sensor_name"
        ] = _AZ_IMU_ANG_VEL_SENSOR
    if "base_ang_vel" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["base_ang_vel"].params[
            "sensor_name"
        ] = _AZ_IMU_ANG_VEL_SENSOR
    if "base_lin_vel" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["base_lin_vel"].params[
            "sensor_name"
        ] = _AZ_IMU_LIN_VEL_SENSOR

    # azureloong_v9 has minimal collision geoms: base_link_collision and ankle collisions
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^link_ankle_[lr]_roll_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "link_ankle_l_roll",
        "link_ankle_r_roll",
        "link_arm_l_07",
        "link_arm_r_07",
    )
    cfg.terminations["anchor_pos"].params["threshold"] = 0.4
    cfg.terminations["anchor_ori"].params["threshold"] = 1.0
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.4
    cfg.viewer.body_name = "base_link"
    cfg.episode_length_s = 10.0
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    actor_terms = {
        key: value
        for key, value in cfg.observations["actor"].terms.items()
        if key not in {"motion_anchor_pos_b", "base_lin_vel"}
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
    )

    cfg.observations["actor"].terms.update(deepcopy(_VELCMD_ACTOR_TERMS))
    cfg.observations["critic"].terms.update(deepcopy(_VELCMD_CRITIC_TERMS))

    _add_history_obs_groups(cfg)

    if play:
        _apply_play_mode_overrides(cfg)
        cfg.observations["actor_history"].enable_corruption = False
        cfg.observations["critic_history"].enable_corruption = False

    return cfg
