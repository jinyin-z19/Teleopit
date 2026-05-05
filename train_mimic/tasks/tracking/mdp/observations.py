from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
    yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# Velocity-command observation terms: reference velocities and projected
# gravity for the VelCmd task variant.
# ---------------------------------------------------------------------------


def ref_base_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor linear velocity in the robot's body frame. (N, 3)"""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return quat_apply(quat_inv(command.robot_anchor_quat_w), command.anchor_lin_vel_w)


def ref_base_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor angular velocity in the robot's body frame. (N, 3)"""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return quat_apply(quat_inv(command.robot_anchor_quat_w), command.anchor_ang_vel_w)


def ref_projected_gravity_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Gravity direction in the reference anchor's body frame. (N, 3)

    Encodes the reference body tilt — analogous to ``projected_gravity`` but
    for the motion reference rather than the robot.
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset = env.scene[command.cfg.entity_name]
    return quat_apply(quat_inv(command.anchor_quat_w), asset.data.gravity_vec_w)


def ref_base_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor height (z-coordinate). (N, 1) — critic privileged."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return command.anchor_pos_w[:, 2:3]


# ---------------------------------------------------------------------------
# Yaw-only variants: use yaw_quat(robot_anchor_quat_w) to decouple
# roll/pitch from the coordinate transform, matching the TWIST2 approach.
# ---------------------------------------------------------------------------


def motion_anchor_pos_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        yaw_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        yaw_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    robot_yaw = yaw_quat(command.robot_anchor_quat_w)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_yaw[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    robot_yaw = yaw_quat(command.robot_anchor_quat_w)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_yaw[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# Pure-ref windowed observation term
# ---------------------------------------------------------------------------


def ref_window_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Pure-ref windowed observations: ref_projected_gravity + ref_joint_pos + ref_joint_vel.

    Returns (B, W, 61) — only quantities that depend solely on the reference
    trajectory (no robot state), suitable for future-frame encoding.
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset = env.scene[command.cfg.entity_name]
    gravity_w = asset.data.gravity_vec_w  # (B, 3)

    # ref_projected_gravity_b for each window frame: (B, W, 3)
    win_quat = command.window_anchor_quat_w  # (B, W, 4)
    B, W, _ = win_quat.shape
    win_quat_inv = quat_inv(win_quat.reshape(-1, 4)).reshape(B, W, 4)
    win_proj_grav = quat_apply(
        win_quat_inv.reshape(-1, 4),
        gravity_w[:, None, :].expand(B, W, 3).reshape(-1, 3),
    ).reshape(B, W, 3)

    # ref joint pos/vel: (B, W, 29)
    win_joint_pos = command.window_joint_pos
    win_joint_vel = command.window_joint_vel

    return torch.cat([win_proj_grav, win_joint_pos, win_joint_vel], dim=-1)


# ---------------------------------------------------------------------------
# Terrain height observation — scans terrain heights at probe points in the
# robot body frame using MuJoCo ray casting (hfield geoms).  The probe grid
# matches TerrainHeightScanner._DEFAULT_PROBE_GRID (5×5 over ±0.4 m).
# ---------------------------------------------------------------------------

# Default probe point offsets in the robot body frame (X-forward, Y-left).
_DEFAULT_TERRAIN_PROBE_OFFSETS: list[tuple[float, float]] = [
    (x, y)
    for x in [-0.4, -0.2, 0.0, 0.2, 0.4]
    for y in [-0.4, -0.2, 0.0, 0.2, 0.4]
]


def terrain_heights(
    env: ManagerBasedRlEnv,
    command_name: str,
    num_height_points: int = 25,
    ray_start_height: float = 1.0,
    ray_length: float = 5.0,
) -> torch.Tensor:
    """Terrain heights at probe points in the robot body frame.

    Casts rays downward (world -Z) from probe offsets defined in the robot
    base frame and returns the world-Z coordinate of the hit point.

    For plane terrain this returns zeros.  For hfield / generator terrain
    the function uses MuJoCo ``mj_ray`` per environment to sample heights.

    Parameters
    ----------
    env : ManagerBasedRlEnv
        The training environment (batched).
    command_name : str
        Name of the motion command (unused, kept for interface consistency).
    num_height_points : int
        Number of probe points.  Must match the probe_offsets count.
    ray_start_height : float
        Height (m) above the base origin from which rays are cast downward.
    ray_length : float
        Maximum ray length (m).  Misses default to 0.0 (ground plane).

    Returns
    -------
    heights : torch.Tensor of shape (num_envs, num_height_points)
        World-frame Z coordinates at each probe point.
    """
    import math

    import mujoco
    import numpy as np

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    robot = env.scene[command.cfg.entity_name]
    num_envs = env.num_envs

    # Use the first num_height_points probe offsets (default: 5×5 grid).
    probe_offsets = _DEFAULT_TERRAIN_PROBE_OFFSETS[:num_height_points]
    if len(probe_offsets) < num_height_points:
        raise ValueError(
            f"num_height_points={num_height_points} exceeds default probe grid "
            f"size ({len(_DEFAULT_TERRAIN_PROBE_OFFSETS)})."
        )

    # Robot base state in world frame — batched tensors on env.device.
    # root_state_w shape: (num_envs, 13) — [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    root_state = robot.data.root_state_w  # (num_envs, 13)
    base_pos_w = root_state[:, 0:3]        # (num_envs, 3)
    base_quat_w = root_state[:, 3:7]       # (num_envs, 4)  w,x,y,z

    # Move tensors to CPU for MuJoCo ray calls.
    base_pos = base_pos_w.cpu().numpy()
    base_quat = base_quat_w.cpu().numpy()

    # Access the MuJoCo model from the first environment's physics.
    # ManagerBasedRlEnv stores the shared model in env.physics.model (or similar).
    physics = getattr(env, "physics", None)
    if physics is None:
        # Fallback: access via the scene's first environment physics.
        physics = getattr(env.scene, "_physics", None)
        if physics is not None and hasattr(physics, "__getitem__"):
            physics = physics[0]
    if physics is None or not hasattr(physics, "model") or not hasattr(physics, "data"):
        # No MuJoCo access — return zeros (plane terrain assumption).
        return torch.zeros(num_envs, num_height_points, dtype=torch.float32, device=env.device)

    mj_model = physics.model
    mj_data = physics.data

    heights_np = np.zeros((num_envs, num_height_points), dtype=np.float32)

    for e in range(num_envs):
        bp = base_pos[e]  # (3,)
        bq = base_quat[e]  # (4,) w,x,y,z

        # Normalize quaternion
        qn = bq / max(np.linalg.norm(bq), 1e-8)
        w, x, y, z = float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])

        for i, (ox, oy) in enumerate(probe_offsets):
            # Rotate local offset (ox, oy, ray_start_height) by quaternion.
            # Use Rodrigues-like formula for efficiency.
            lx, ly, lz = ox, oy, ray_start_height

            # q * v * q_conj
            tx = 2.0 * (y * lz - z * ly)
            ty = 2.0 * (z * lx - x * lz)
            tz = 2.0 * (x * ly - y * lx)

            rx = lx + w * tx + (y * tz - z * ty)
            ry = ly + w * ty + (z * tx - x * tz)
            rz = lz + w * tz + (x * ty - y * tx)

            ray_start = np.array([bp[0] + rx, bp[1] + ry, bp[2] + rz], dtype=np.float64)
            ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

            distance = mujoco.mj_ray(
                mj_model,
                mj_data,
                ray_start,
                ray_dir,
                None,   # geomgroup
                -1,     # flg_static
                -1,     # bodyexclude
                None,   # geomid output
            )

            if distance >= 0.0 and distance <= ray_length:
                heights_np[e, i] = np.float32(float(ray_start[2] - distance))
            else:
                heights_np[e, i] = np.float32(0.0)

    return torch.from_numpy(heights_np).to(device=env.device, dtype=torch.float32)
