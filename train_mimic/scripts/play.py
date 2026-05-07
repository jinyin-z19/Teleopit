#!/usr/bin/env python3
from __future__ import annotations

# ── Must run BEFORE any mujoco import (GL platform selection) ─────
import os as _os, sys as _sys
if "--video" in _sys.argv:
    _os.environ.setdefault("MUJOCO_GL", "egl")
    _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

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

import argparse
import logging
import os
import time
import re
from glob import glob

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

_logger = logging.getLogger(__name__)


def _attach_terrain_probe_renderer(env) -> None:
    """Attach a terrain-probe debug callback to *env* so that
    ``env.update_visualizers`` draws coloured height-scan spheres
    **in addition to** the existing ghost / command / sensor visualizers.

    Must be called before wrapping *env* with ``VideoRecorder`` or
    ``RslRlVecEnvWrapper``.
    """
    try:
        from teleopit.sim.terrain_probe_drawer import TerrainProbeDrawer
        from train_mimic.tasks.tracking.mdp.observations import (
            _DEFAULT_TERRAIN_PROBE_OFFSETS,
        )
    except ImportError:
        return

    probe_offsets = list(_DEFAULT_TERRAIN_PROBE_OFFSETS[:25])
    drawer = TerrainProbeDrawer(probe_offsets, ray_start_height=1.0)
    terrain_cb = drawer.make_update_callback(env)

    # Chain with the existing update_visualizers method (ghost robot, etc.).
    _original = env.update_visualizers

    def _chained(visualizer) -> None:
        _original(visualizer)
        terrain_cb(visualizer)

    env.update_visualizers = _chained
    print(f"[play] Terrain probe renderer attached ({drawer.num_points} points)")


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

    # ── Terrain probe renderer ──────────────────────────────────────
    _attach_terrain_probe_renderer(env)

    if args.video:
        from mjlab.utils.wrappers import VideoRecorder
        log_dir = os.path.dirname(args.checkpoint)
        video_folder = os.path.join(log_dir, "videos", "play")

        # Auto-increment: find max N in rl-video-step-*.mp4 → use N+1.
        _max_n = -1
        for p in glob(os.path.join(video_folder, "rl-video-step-*.mp4")):
            m = re.search(r"rl-video-step-(\d+)", os.path.basename(p))
            if m:
                _max_n = max(_max_n, int(m.group(1)))
        _video_idx = _max_n + 1

        env = VideoRecorder(
            env,
            video_folder=video_folder,
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

        # ── Rename to auto-incremented index ───────────────────────
        _default_path = os.path.join(video_folder, "rl-video-step-0.mp4")
        _target_path = os.path.join(video_folder, f"rl-video-step-{_video_idx}.mp4")
        if _video_idx != 0 and os.path.isfile(_default_path):
            os.rename(_default_path, _target_path)
            print(f"[play] Renamed: rl-video-step-0.mp4 → rl-video-step-{_video_idx}.mp4")
    elif args.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
