#!/usr/bin/env python3
"""Generate per-scene workflow configs for the MultiScan dataset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit('PyYAML is required to generate configs (pip install pyyaml)') from exc


def count_frames(color_dir: Path) -> int:
    if not color_dir.exists():
        return 0
    patterns = ('*.png', '*.jpg', '*.jpeg')
    total = 0
    for pattern in patterns:
        total += sum(1 for _ in color_dir.glob(pattern))
    return total


def build_stage_template(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        'gpu': args.default_gpu,
        'ssam': {
            'enabled': True,
            'ssam_freq': args.ssam_freq,
            'min_area': args.min_area,
            'stability_threshold': args.stability_threshold,
            'persist_raw': args.persist_raw,
            'skip_filtering': args.skip_filtering,
            'add_gaps': args.add_gaps,
            'fill_area': args.fill_area,
            'append_timestamp': not args.no_timestamp,
        },
        'filter': {
            'enabled': args.enable_filter,
            'min_area': args.filter_min_area,
            'stability_threshold': args.filter_stability,
            'update_manifest': True,
        },
        'tracker': {
            'enabled': True,
            'prompt_mode': args.prompt_mode,
            'max_propagate': args.max_propagate,
            'iou_threshold': args.iou_threshold,
            'downscale_masks': args.downscale_masks,
            'downscale_ratio': args.downscale_ratio,
        },
        'report': {
            'enabled': True,
            'name': args.report_name,
            'max_width': args.report_max_width,
            'record_timings': True,
            'timing_output': 'stage_timings.json',
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare YAML configs for MultiScan scenes')
    parser.add_argument('--dataset-root', default='/media/public_dataset2/multiscan', help='MultiScan dataset root')
    parser.add_argument('--project-root', default='.', help='Project root that contains the outputs directory')
    parser.add_argument('--output-dir', default='configs/scenes', help='Directory to write generated YAML files (per-scene)')
    parser.add_argument('--output-base', help='Base directory for experiment outputs (default: <project-root>/outputs)')
    parser.add_argument('--scenes', nargs='*', help='Subset of scenes to generate (defaults to all)')
    parser.add_argument('--dry-run', action='store_true', help='Preview generated configs without writing files')
    parser.add_argument('--skip-existing', action='store_true', help='Skip generating configs that already exist')
    parser.add_argument('--default-gpu', type=int, default=0, help='Default GPU id recorded in configs')
    parser.add_argument('--frame-step', type=int, default=50, help='Frame sampling step for SSAM stage')
    parser.add_argument('--ssam-freq', type=int, default=2, help='Semantic-SAM frequency')
    parser.add_argument('--min-area', type=int, default=0, help='Minimum area passed to SSAM stage')
    parser.add_argument('--fill-area', type=int, default=10000, help='Gap fill area threshold')
    parser.add_argument('--stability-threshold', type=float, default=1.0, help='Stability threshold for SSAM candidates')
    parser.add_argument('--persist-raw', action='store_true', help='Enable raw candidate persistence in SSAM stage')
    parser.add_argument('--skip-filtering', action='store_true', help='Disable inline filtering after SSAM stage')
    parser.add_argument('--add-gaps', action='store_true', help='Enable gap-fill mask synthesis on the first level')
    parser.add_argument('--no-timestamp', action='store_true', help='Disable timestamped run directories in SSAM stage')
    parser.add_argument('--enable-filter', action='store_true', help='Enable filter stage by default')
    parser.add_argument('--filter-min-area', type=int, default=0, help='Filter stage minimum area threshold')
    parser.add_argument('--filter-stability', type=float, default=0.0, help='Filter stage stability threshold')
    parser.add_argument('--prompt-mode', default='all_mask', help='Tracker prompt mode (all_mask/lt_bbox/all_bbox)')
    parser.add_argument('--max-propagate', type=int, help='Tracker max propagation frames (None for unlimited)')
    parser.add_argument('--iou-threshold', type=float, default=0.6, help='Tracker IoU threshold override')
    parser.add_argument('--downscale-masks', dest='downscale_masks', action='store_true', help='Persist masks at reduced resolution (default)')
    parser.add_argument('--no-downscale-masks', dest='downscale_masks', action='store_false', help='Disable storing downscaled masks in tracker stage')
    parser.add_argument('--downscale-ratio', type=float, default=0.3, help='Mask downscale ratio when enabled')
    parser.add_argument('--report-name', default='report.md', help='Report filename')
    parser.add_argument('--report-max-width', type=int, default=640, help='Max width for report previews')
    parser.add_argument('--tag', default='multiscan', help='Default experiment tag stored in configs')
    parser.add_argument('--levels', default='2,4,6', help='Comma-separated levels for experiment.defaults')
    parser.add_argument('--summary-path', default='configs/index/multiscan_scene_index.json', help='Path to write summary JSON')

    parser.set_defaults(downscale_masks=True)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_base = Path(args.output_base).expanduser().resolve() if args.output_base else project_root / 'outputs'
    output_base.mkdir(parents=True, exist_ok=True)

    scenes = args.scenes
    if scenes:
        requested = {scene if scene.startswith('scene_') else f'scene_{scene}' for scene in scenes}
    else:
        requested = None

    stage_template = build_stage_template(args)
    experiment_defaults = {
        'frames': {'step': args.frame_step},
    }
    levels = [int(x) for x in str(args.levels).split(',') if x.strip()]

    entries: List[Dict[str, Any]] = []

    if not dataset_root.exists():
        raise SystemExit(f'Dataset root {dataset_root} does not exist')

    for scene_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith('scene_')):
        if requested and scene_dir.name not in requested:
            continue

        color_dir = scene_dir / 'outputs' / 'color'
        frame_count = count_frames(color_dir)

        output_root = output_base / scene_dir.name
        config_payload = {
            'experiment': {
                'name': scene_dir.name,
                'data_path': str(color_dir),
                'output_root': str(output_root),
                'frames': {'step': args.frame_step},
                'levels': levels,
                'tag': args.tag,
            },
            'stages': stage_template,
        }

        config_path = output_dir / f'{scene_dir.name}.yaml'
        if args.skip_existing and config_path.exists():
            action = 'skipped'
        elif args.dry_run:
            action = 'preview'
        else:
            with config_path.open('w', encoding='utf-8') as f:
                yaml.safe_dump(config_payload, f, sort_keys=False)
            action = 'written'

        entries.append(
            {
                'scene': scene_dir.name,
                'data_path': str(color_dir),
                'output_root': str(output_root),
                'frames': frame_count,
                'config_path': str(config_path),
                'action': action,
            }
        )

        print(f'{scene_dir.name}: {action} -> {config_path}')

    if not args.dry_run:
        summary_path = Path(args.summary_path).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open('w', encoding='utf-8') as f:
            payload = {
                'generated_at': datetime.utcnow().isoformat(timespec='seconds'),
                'dataset_root': str(dataset_root),
                'output_dir': str(output_dir),
                'output_base': str(output_base),
                'levels': levels,
                'stage_defaults': stage_template,
                'experiment_defaults': experiment_defaults,
                'scenes': entries,
            }
            json.dump(payload, f, indent=2)
        print(f'Summary written to {summary_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
