#!/usr/bin/env python3
"""Play back a trained tracking policy in simulation.

Modes (fastest → slowest):
  headless -- no rendering at all, pure sim stepping (fastest)
  video    -- render to rgb_array + encode video
  viser    -- browser-based 3D viewer at http://localhost:8012
  native   -- MuJoCo native window (default, requires display, slowest)

Usage:
    # Headless (fastest, no rendering)
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets/twist2_full/val \
        --headless

    # Native window
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets/twist2_full/val

    # Browser viewer (no display required)
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets/twist2_full/val \
        --viewer viser

    # Record video (no window)
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets/twist2_full/val \
        --video
"""

from __future__ import annotations

import argparse
import os
import time

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from train_mimic.app import (
    DEFAULT_TASK,
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    resolve_device,
    validate_checkpoint_path,
    validate_motion_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play trained G1 tracking policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to motion shard directory")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--viewer", type=str, default="native", choices=["native", "viser"],
        help="native: MuJoCo window (requires display); viser: browser at localhost:8012",
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run without any rendering (fastest, pure sim stepping)")
    parser.add_argument("--video", action="store_true", help="Record video instead of interactive viewer")
    parser.add_argument("--steps_num", type=int, default=500,
                        help="Number of steps in headless mode (default: 500)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Task id to play (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── GPU rendering for video mode (must be set BEFORE any MuJoCo/GL init) ──
    if args.video and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("[INFO] --video enabled, MUJOCO_GL not set. Defaulting to MUJOCO_GL=egl.")
    if args.video and "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("[INFO] --video enabled, PYOPENGL_PLATFORM not set. Defaulting to PYOPENGL_PLATFORM=egl.")

    (
        torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        _load_env_cfg,
        _load_rl_cfg,
        _load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()

    try:
        validate_checkpoint_path(args.checkpoint)
        validate_motion_file(args.motion_file)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    configure_torch_backends()

    # Load configs (play=True disables corruption, push_robot, etc.)
    task_name, env_cfg, agent_cfg, runner_cls = load_task_components(
        args.task,
        play=True,
        load_env_cfg=_load_env_cfg,
        load_rl_cfg=_load_rl_cfg,
        load_runner_cls=_load_runner_cls,
    )

    # Override for playback
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands["motion"].motion_file = args.motion_file

    device = resolve_device(args.device, torch)

    # render_mode: rgb_array for video recording, None for headless/viewer
    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if args.video:
        from mjlab.utils.wrappers import VideoRecorder
        log_dir = os.path.dirname(args.checkpoint)
        env = VideoRecorder(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args.steps_num,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy (force tensorboard to avoid wandb init during playback).
    log_dir = os.path.dirname(args.checkpoint)
    agent_dict = build_runner_cfg_dict(agent_cfg, force_tensorboard=True)
    RunnerCls = runner_cls or MjlabOnPolicyRunner
    runner = RunnerCls(env, agent_dict, log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    if args.headless:
        # Pure sim stepping, no rendering overhead — fastest mode
        print(f"Running headless for {args.steps_num} steps...")
        obs = env.get_observations()
        t_start = time.time()
        for i in range(args.steps_num):
            t0 = time.time()
            with torch.no_grad():
                actions = policy(obs)
            t_policy = time.time()
            obs, _, _, _ = env.step(actions)
            t_step = time.time()
            if i < 5 or i % 100 == 0:
                print(f"  step {i:4d}/{args.steps_num}  "
                      f"policy={t_policy-t0:.3f}s  step={t_step-t_policy:.3f}s  "
                      f"total={t_step-t0:.3f}s")
        elapsed = time.time() - t_start
        print(f"Done in {elapsed:.1f}s ({args.steps_num/elapsed:.1f} steps/s)")
    elif args.video:
        # Run a fixed number of steps then close
        print(f"Recording {args.steps_num} steps to video...")
        obs = env.get_observations()
        t_start = time.time()
        for i in range(args.steps_num):
            t0 = time.time()
            with torch.no_grad():
                actions = policy(obs)
            t_policy = time.time()
            obs, _, _, _ = env.step(actions)
            t_step = time.time()
            print(f"  step {i:4d}/{args.steps_num}  "
                    f"policy={t_policy-t0:.3f}s  step={t_step-t_policy:.3f}s  "
                    f"total={t_step-t0:.3f}s")
        elapsed = time.time() - t_start
        print(f"Simulation done in {elapsed:.1f}s ({args.steps_num/elapsed:.1f} steps/s)")
        print("Encoding video... (may take a while)")
        t_enc = time.time()
        env.close()  # triggers _finish_recording → media.write_video
        print(f"Video saved in {time.time()-t_enc:.1f}s")
    elif args.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
