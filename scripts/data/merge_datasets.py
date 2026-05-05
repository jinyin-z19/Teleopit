#!/usr/bin/env python3
"""Merge pre-built AzureLoong V9 datasets into one combined dataset.

Supports two modes:
  1. CLI list mode (quick ad-hoc):
     python scripts/data/merge_datasets.py \
         --datasets lafan1_v1 CMU_v1 \
         --output lafan_CMU_v1

  2. YAML spec mode (reproducible, recommended for config tracking):
     python scripts/data/merge_datasets.py \
         --spec train_mimic/configs/datasets/lafan_CMU_v1.yaml

     YAML format:
         output: lafan_CMU_v1
         datasets:
           - lafan1_v1
           - CMU_v1

     python scripts/data/merge_datasets.py \
         --spec train_mimic/configs/datasets/lafan_CMU_v1.yaml \
         --force
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

# Keys that are concatenated along the time (frame) axis (axis=0).
TIME_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}

# Keys that are concatenated along the clip axis (axis=0).
CLIP_KEYS = {
    "clip_lengths",
    "clip_fps",
    "clip_weights",
}


def _validate_split(
    datasets: list[str],
    split: str,
    base_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Collect all shards for a split across datasets and validate compatibility."""
    all_shards: list[Path] = []
    ref_body_names: np.ndarray | None = None
    ref_fps: int | None = None
    ref_num_actions: int | None = None

    for ds_name in datasets:
        split_dir = base_dir / ds_name / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split: {split_dir}")
        shards = sorted(split_dir.glob("shard_*.npz"))
        if not shards:
            raise ValueError(f"no shards in {split_dir}")

        for shard in shards:
            data = np.load(shard, allow_pickle=True)
            if ref_body_names is None:
                ref_body_names = data["body_names"]
                ref_fps = int(data["fps"])
                ref_num_actions = data["joint_pos"].shape[1]
            else:
                if not np.array_equal(data["body_names"], ref_body_names):
                    raise ValueError(
                        f"body_names mismatch: {shard}\n"
                        f"  expected: {list(ref_body_names)}\n"
                        f"  got: {list(data['body_names'])}"
                    )
                if int(data["fps"]) != ref_fps:
                    raise ValueError(f"fps mismatch in {shard}: {data['fps']} vs {ref_fps}")
                if data["joint_pos"].shape[1] != ref_num_actions:
                    raise ValueError(
                        f"DOF mismatch in {shard}: {data['joint_pos'].shape[1]} vs {ref_num_actions}"
                    )
            all_shards.append(shard)

    return all_shards, {
        "body_names": ref_body_names,
        "fps": ref_fps,
        "num_actions": ref_num_actions,
    }


def _merge_shards(
    shards: list[Path],
    output_dir: Path,
    ref: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Merge all shards into new shard files and return shard metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_infos: list[dict[str, Any]] = []

    # Collect all frames and clips linearly
    frame_offset = 0
    all_clips: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for shard in shards:
        data = np.load(shard, allow_pickle=True)
        jp = data["joint_pos"]
        jv = data["joint_vel"]
        bp = data["body_pos_w"]
        bq = data["body_quat_w"]
        blv = data["body_lin_vel_w"]
        bav = data["body_ang_vel_w"]
        cl = data["clip_lengths"]
        cs = data["clip_starts"]
        cfps = data["clip_fps"]
        cw = data["clip_weights"]

        # Adjust clip_starts for this shard
        adj_cs = cs + frame_offset
        frame_offset += len(jp)

        all_clips.append((jp, jv, bp, bq, blv, bav, cl, adj_cs, cfps, cw))

    # Concat all data
    joint_pos = np.concatenate([c[0] for c in all_clips], axis=0).astype(np.float32)
    joint_vel = np.concatenate([c[1] for c in all_clips], axis=0).astype(np.float32)
    body_pos_w = np.concatenate([c[2] for c in all_clips], axis=0).astype(np.float32)
    body_quat_w = np.concatenate([c[3] for c in all_clips], axis=0).astype(np.float32)
    body_lin_vel_w = np.concatenate([c[4] for c in all_clips], axis=0).astype(np.float32)
    body_ang_vel_w = np.concatenate([c[5] for c in all_clips], axis=0).astype(np.float32)
    clip_lengths = np.concatenate([c[6] for c in all_clips]).astype(np.int64)
    clip_starts = np.concatenate([c[7] for c in all_clips]).astype(np.int64)
    clip_fps = np.concatenate([c[8] for c in all_clips]).astype(np.int64)
    clip_weights = np.concatenate([c[9] for c in all_clips]).astype(np.float64)

    total_frames = len(joint_pos)
    total_clips = len(clip_lengths)

    # Write a single merged shard
    shard_path = output_dir / "shard_000.npz"
    np.savez(
        shard_path,
        fps=ref["fps"],
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        body_names=ref["body_names"],
        clip_starts=clip_starts,
        clip_lengths=clip_lengths,
        clip_fps=clip_fps,
        clip_weights=clip_weights,
    )

    duration_s = total_frames / ref["fps"]
    print(f"  [{split}] {total_clips} clips, {total_frames} frames ({duration_s:.1f}s) -> {shard_path.name}")

    shard_infos.append({
        "path": str(shard_path),
        "clip_lengths": clip_lengths.tolist(),
    })
    return shard_infos


def merge_datasets(
    dataset_names: list[str],
    output_name: str,
    *,
    base_dir: Path | None = None,
    force: bool = False,
) -> None:
    base_dir = base_dir or DEFAULT_DATASETS_DIR
    output_dir = base_dir / output_name

    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"output dataset already exists: {output_dir}\n"
                f"Use --force to overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging datasets: {', '.join(dataset_names)} -> {output_name}")
    print(f"Source dir: {base_dir}")

    all_stats: dict[str, Any] = {
        "name": output_name,
        "datasets_merged": dataset_names,
        "splits": {},
    }

    for split in ("train", "val"):
        shards, ref = _validate_split(dataset_names, split, base_dir)
        split_dir = output_dir / split
        shard_infos = _merge_shards(shards, split_dir, ref, split)

        total_clips = sum(
            sum(info["clip_lengths"]) > 0 for info in shard_infos
        )
        # Actually count clips properly
        total_clip_count = 0
        total_frames_count = 0
        for info in shard_infos:
            total_clip_count += len(info["clip_lengths"])
            total_frames_count += sum(info["clip_lengths"])

        all_stats["splits"][split] = {
            "output": str(split_dir),
            "shards": len(shard_infos),
            "clips": total_clip_count,
            "frames": total_frames_count,
            "fps": ref["fps"],
            "duration_s": total_frames_count / ref["fps"],
        }

    # Write build_info.json
    build_info_path = output_dir / "build_info.json"
    build_info_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print(f"\n[DONE] Merged dataset: {output_dir}")
    print(f"  train: {all_stats['splits']['train']['clips']} clips, "
          f"{all_stats['splits']['train']['frames']} frames "
          f"({all_stats['splits']['train']['duration_s']:.1f}s)")
    print(f"  val:   {all_stats['splits']['val']['clips']} clips, "
          f"{all_stats['splits']['val']['frames']} frames "
          f"({all_stats['splits']['val']['duration_s']:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge pre-built datasets into one combined dataset."
    )
    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        help="YAML merge spec (output name + dataset list). "
             "Alternative to --datasets/--output.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Dataset names to merge (e.g. lafan1_v1 CMU_v1). "
             "Use with --output.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Name of the merged output dataset. Required with --datasets.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base datasets directory (default: data/datasets)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output dataset",
    )

    args = parser.parse_args()

    # Resolve spec mode vs CLI mode
    if args.spec:
        if args.datasets or args.output:
            parser.error("--spec is exclusive with --datasets/--output")
        spec_path = Path(args.spec).expanduser().resolve()
        if not spec_path.is_file():
            parser.error(f"merge spec not found: {spec_path}")
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            parser.error(f"merge spec must be a YAML mapping: {spec_path}")
        output_name = str(payload.get("output", "")).strip()
        raw_datasets = payload.get("datasets")
        if not output_name:
            parser.error(f"merge spec missing non-empty 'output': {spec_path}")
        if not isinstance(raw_datasets, list) or not raw_datasets:
            parser.error(f"merge spec must contain a non-empty 'datasets' list: {spec_path}")
        dataset_names = [str(d).strip() for d in raw_datasets]
        if not all(dataset_names):
            parser.error(f"merge spec contains empty dataset name: {spec_path}")
    else:
        if not args.datasets or not args.output:
            parser.error("either --spec or both --datasets and --output are required")
        output_name = args.output
        dataset_names = list(args.datasets)

    if len(dataset_names) < 2:
        parser.error("need at least 2 datasets to merge")

    base_dir = Path(args.base_dir) if args.base_dir else None
    merge_datasets(dataset_names, output_name, base_dir=base_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
