#!/usr/bin/env python3
"""End-to-end pipeline: Semantic-SAM candidate generation → SAM2 tracking."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from common_utils import format_duration, parse_levels
from pipeline_defaults import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEMANTIC_SAM_CKPT,
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CKPT,
    expand_default,
)
from run_workflow import execute_workflow


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    levels = parse_levels(args.levels)
    experiment: Dict[str, Any] = {
        'data_path': args.data_path,
        'output_root': args.output,
        'levels': levels,
        'sam_ckpt': args.sam_ckpt,
        'sam2_cfg': args.sam2_cfg,
        'sam2_ckpt': args.sam2_ckpt,
    }

    stages: Dict[str, Any] = {
        'ssam': {
            'enabled': True,
            'levels': levels,
            'frames': args.frames,
            'min_area': args.min_area,
            'fill_area': args.fill_area,
            'stability_threshold': args.stability_threshold,
            'append_timestamp': not args.no_timestamp,
        },
        'filter': {
            'enabled': True,
            'min_area': args.min_area,
            'stability_threshold': args.stability_threshold,
        },
        'tracker': {
            'enabled': True,
            'levels': levels,
            'sam2_cfg': args.sam2_cfg,
            'sam2_ckpt': args.sam2_ckpt,
        },
        'report': {
            'enabled': True,
        },
    }

    return {
        'experiment': experiment,
        'stages': stages,
    }


def _print_summary(summaries: List[Dict[str, Any]]) -> None:
    for summary in summaries:
        run_dir = summary.get('run_dir')
        if run_dir:
            print(f"Outputs located at: {run_dir}")

        stages = summary.get('stages', {})
        if not isinstance(stages, dict):
            continue

        durations = []
        for name, meta in stages.items():
            if not isinstance(meta, dict):
                continue
            try:
                duration = float(meta.get('duration_sec', 0.0))
            except (TypeError, ValueError):
                duration = 0.0
            durations.append((name, duration))

        if durations:
            print('\nTiming summary:')
            total = 0.0
            for name, duration in durations:
                print(f"  - {name}: {format_duration(duration)}")
                total += duration
            print(f"  - Total: {format_duration(total)}")
        if run_dir:
            print(f"\nDone. Final artifacts under: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic-SAM multi-level → SAM2 tracking pipeline",
    )
    parser.add_argument(
        '--data-path',
        default=str(expand_default(DEFAULT_DATA_PATH)),
        help='Folder containing all frames (JPG/PNG)',
    )
    parser.add_argument('--levels', default='2,4,6', help='Comma-separated levels, default 2,4,6')
    parser.add_argument('--frames', default='1200:1600:20', help='Range as start:end:step (end exclusive)')
    parser.add_argument(
        '--sam-ckpt',
        default=str(expand_default(DEFAULT_SEMANTIC_SAM_CKPT)),
        help='Semantic-SAM checkpoint path',
    )
    parser.add_argument(
        '--sam2-cfg',
        default=str(expand_default(DEFAULT_SAM2_CFG)),
        help='SAM2 config YAML or Hydra path',
    )
    parser.add_argument(
        '--sam2-ckpt',
        default=str(expand_default(DEFAULT_SAM2_CKPT)),
        help='SAM2 checkpoint path',
    )
    parser.add_argument(
        '--output',
        default=str(expand_default(DEFAULT_OUTPUT_ROOT)),
        help='Output root inside My3DIS',
    )
    parser.add_argument('--min-area', type=int, default=300)
    parser.add_argument(
        '--fill-area',
        type=int,
        default=None,
        help='Minimum area for SSAM gap-fill masks (default: min-area)',
    )
    parser.add_argument(
        '--stability',
        '--stability-threshold',
        dest='stability_threshold',
        type=float,
        default=0.9,
        help='Minimum stability_score (default: 0.9)',
    )
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Do not append timestamp folder to output root',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned stages without executing',
    )

    args = parser.parse_args()

    config = _build_config(args)

    if args.dry_run:
        print('Dry-run configuration:')
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0

    summaries = execute_workflow(config)
    _print_summary(summaries)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
