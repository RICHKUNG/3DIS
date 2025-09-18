#!/usr/bin/env python3
"""End-to-end pipeline: Semantic-SAM candidate generation → SAM2 tracking.

This orchestrator reuses the modular stage implementations from
``generate_candidates.py`` and ``track_from_candidates.py`` so the outputs
match what ``run_experiment.sh`` produces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from generate_candidates import (
    DEFAULT_SEMANTIC_SAM_CKPT,
    run_generation as run_candidate_generation,
)
from track_from_candidates import (
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CKPT,
    run_tracking as run_candidate_tracking,
)


DEFAULT_DATA_PATH = "/media/public_dataset2/multiscan/scene_00065_00/outputs/color"
DEFAULT_OUTPUT_ROOT = "/media/Pluto/richkung/My3DIS/outputs/scene_00065_00"


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_stage(label: str, func, *args, dry_run: bool = False, dry_run_details: str | None = None, **kwargs):
    print(f"→ {label}")
    if dry_run:
        if dry_run_details:
            print(f"  dry-run: {dry_run_details}")
        else:
            print("  dry-run: skipping execution")
        return None, 0.0
    start = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception:
        duration = time.time() - start
        print(f"  failed after {format_duration(duration)}", file=sys.stderr)
        raise
    duration = time.time() - start
    print(f"  completed in {format_duration(duration)}")
    return result, duration


def main():
    parser = argparse.ArgumentParser(
        description="Semantic-SAM multi-level → SAM2 tracking pipeline"
    )
    parser.add_argument(
        '--data-path',
        default=DEFAULT_DATA_PATH,
        help='Folder containing all frames (JPG/PNG)',
    )
    parser.add_argument('--levels', default='2,4,6', help='Comma-separated levels, default 2,4,6')
    parser.add_argument('--frames', default='1200:1600:20', help='Range as start:end:step (end exclusive)')
    parser.add_argument('--sam-ckpt', default=DEFAULT_SEMANTIC_SAM_CKPT,
                        help='Semantic-SAM checkpoint path')
    parser.add_argument('--sam2-cfg', default=DEFAULT_SAM2_CFG,
                        help='SAM2 config YAML or Hydra path')
    parser.add_argument('--sam2-ckpt', default=DEFAULT_SAM2_CKPT,
                        help='SAM2 checkpoint path')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_ROOT, help='Output root inside My3DIS')
    parser.add_argument('--min-area', type=int, default=300)
    parser.add_argument('--stability', '--stability-threshold', dest='stability_threshold',
                        type=float, default=0.9, help='Minimum stability_score (default: 0.9)')
    parser.add_argument('--no-timestamp', action='store_true', help='Do not append timestamp folder to output root')
    parser.add_argument('--dry-run', action='store_true', help='Print the planned stages without executing')
    args = parser.parse_args()

    script_start = time.time()
    stage_summary = []

    expected_run_dir = args.output if args.no_timestamp else os.path.join(args.output, '<timestamp>')

    stage1_result, stage1_duration = run_stage(
        "Stage 1: Semantic-SAM candidate generation",
        run_candidate_generation,
        data_path=args.data_path,
        levels=args.levels,
        frames=args.frames,
        sam_ckpt=args.sam_ckpt,
        output=args.output,
        min_area=args.min_area,
        stability_threshold=args.stability_threshold,
        add_gaps=False,
        no_timestamp=args.no_timestamp,
        dry_run=args.dry_run,
        dry_run_details="invoke generate_candidates.run_generation(...)",
    )
    stage1_manifest = None
    if args.dry_run:
        run_dir = expected_run_dir
    else:
        if isinstance(stage1_result, tuple):
            if len(stage1_result) != 2:
                raise ValueError(
                    "generate_candidates.run_generation() should return (run_dir, manifest)"
                )
            run_dir, stage1_manifest = stage1_result
        elif isinstance(stage1_result, str):
            run_dir = stage1_result
        else:
            raise TypeError(
                "Unexpected return type from generate_candidates.run_generation(): "
                f"{type(stage1_result)!r}"
            )
        stage_summary.append(("Stage 1", stage1_duration))

    print(f"Outputs located at: {run_dir}")
    if stage1_manifest is not None:
        print("  Stage 1 manifest:")
        print(json.dumps(stage1_manifest, indent=2, sort_keys=True))

    _, stage2_duration = run_stage(
        "Stage 2: SAM2 tracking",
        run_candidate_tracking,
        data_path=args.data_path,
        candidates_root=run_dir,
        output=run_dir,
        levels=args.levels,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        dry_run=args.dry_run,
        dry_run_details="invoke track_from_candidates.run_tracking(...)",
    )
    if not args.dry_run:
        stage_summary.append(("Stage 2", stage2_duration))

    total_elapsed = time.time() - script_start
    print("\nTiming summary:")
    if args.dry_run:
        print("  - Stage 1: <dry-run>")
        print("  - Stage 2: <dry-run>")
        print("  - Total: <dry-run>")
    else:
        for label, duration in stage_summary:
            print(f"  - {label}: {format_duration(duration)}")
        print(f"  - Total: {format_duration(total_elapsed)}")
    print(f"\nDone. Final artifacts under: {run_dir}")


if __name__ == '__main__':
    main()
