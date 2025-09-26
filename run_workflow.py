#!/usr/bin/env python3
"""High-level workflow orchestrator driven by YAML config."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

from generate_candidates import (
    RAW_DIR_NAME,
    DEFAULT_SEMANTIC_SAM_CKPT,
    run_generation as run_candidate_generation,
)
from filter_candidates import run_filtering
from track_from_candidates import run_tracking as run_candidate_tracking
from generate_report import build_report


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open('r') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Config file {path} must define a mapping at the top level')
    return data


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(minutes):02d}:{int(secs):02d}"


@contextmanager
def using_gpu(gpu: Optional[Any]):
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu is None:
        yield
        return
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        yield
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous


class StageRecorder:
    def __init__(self, summary: Dict[str, Any], name: str, gpu: Optional[Any]) -> None:
        self.summary = summary
        self.name = name
        self.gpu = None if gpu is None else str(gpu)
        self.start = None
        self.end = None
        self.duration = None

    def __enter__(self):
        self.start = time.perf_counter()
        now_iso = datetime.utcnow().isoformat(timespec='seconds')
        self.summary.setdefault('order', []).append(self.name)
        self.summary.setdefault('stages', {})[self.name] = {
            'started_at': now_iso,
            'gpu': self.gpu,
        }
        return self

    def __exit__(self, exc_type, exc, tb):
        end_time = time.perf_counter()
        duration = end_time - (self.start or end_time)
        now_iso = datetime.utcnow().isoformat(timespec='seconds')
        stage_entry = self.summary['stages'].setdefault(self.name, {})
        stage_entry['ended_at'] = now_iso
        stage_entry['duration_sec'] = duration
        stage_entry['duration_text'] = format_duration(duration)
        if self.gpu is not None:
            stage_entry['gpu'] = self.gpu
        if exc is not None:
            stage_entry['error'] = str(exc)


def list_to_csv(values: Optional[List[Any]]) -> str:
    if not values:
        return ''
    return ','.join(str(v) for v in values)


def resolve_levels(stage_cfg: Dict[str, Any], manifest: Optional[Dict[str, Any]], fallback: Optional[List[int]]) -> List[int]:
    if 'levels' in stage_cfg and stage_cfg['levels'] is not None:
        try:
            return [int(x) for x in stage_cfg['levels']]
        except (TypeError, ValueError):
            raise ValueError(f"Invalid levels in config: {stage_cfg['levels']}")
    if manifest and isinstance(manifest.get('levels'), list):
        try:
            return [int(x) for x in manifest['levels']]
        except (TypeError, ValueError):
            pass
    if fallback:
        return [int(x) for x in fallback]
    raise ValueError('Unable to determine levels for stage')


def stage_frames_string(stage_cfg: Dict[str, Any]) -> str:
    frames_cfg = stage_cfg.get('frames', {}) or {}
    start_raw = frames_cfg.get('start', frames_cfg.get('from'))
    end_raw = frames_cfg.get('end', frames_cfg.get('to'))
    step = int(
        frames_cfg.get('step')
        or frames_cfg.get('freq')
        or frames_cfg.get('stride')
        or stage_cfg.get('freq')
        or 1
    )
    start = int(start_raw) if start_raw is not None else 0
    # Use -1 as sentinel to indicate "until end" when end not provided.
    end = int(end_raw) if end_raw is not None else -1
    return f"{start}:{end}:{step}"


def resolve_stage_gpu(stage_cfg: Optional[Dict[str, Any]], default_gpu: Optional[Any]) -> Optional[Any]:
    if isinstance(stage_cfg, dict) and 'gpu' in stage_cfg:
        return stage_cfg.get('gpu')
    return default_gpu


def update_summary_config(summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    summary['config_snapshot'] = config


def load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open('r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Run workflow defined by a YAML config')
    parser.add_argument('--config', required=True, help='Path to YAML workflow config')
    parser.add_argument('--override-output', help='Override output root directory')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)

    experiment_cfg = config.get('experiment', {})
    if not isinstance(experiment_cfg, dict):
        raise SystemExit('`experiment` section must be a mapping')

    data_path = experiment_cfg.get('data_path')
    if not data_path:
        raise SystemExit('experiment.data_path is required')
    data_path = str(Path(data_path).expanduser())

    output_root = args.override_output or experiment_cfg.get('output_root')
    if not output_root:
        raise SystemExit('experiment.output_root is required (or override via --override-output)')
    output_root = str(Path(output_root).expanduser())

    stages_cfg = config.get('stages', {})
    if not isinstance(stages_cfg, dict):
        raise SystemExit('`stages` section must be a mapping')
    default_stage_gpu = stages_cfg.get('gpu')

    summary: Dict[str, Any] = {
        'config_path': str(config_path),
        'invoked_at': datetime.utcnow().isoformat(timespec='seconds'),
    }
    update_summary_config(summary, config)

    manifest: Optional[Dict[str, Any]] = None
    run_dir: Optional[Path] = None

    # Stage: Semantic-SAM generation
    ssam_cfg = stages_cfg.get('ssam', {}) if isinstance(stages_cfg.get('ssam'), dict) else {}
    ssam_enabled = ssam_cfg.get('enabled', True)

    if ssam_enabled:
        levels = resolve_levels(ssam_cfg, None, experiment_cfg.get('levels'))
        ssam_levels_csv = list_to_csv(levels)
        frames_str = stage_frames_string(ssam_cfg)
        ssam_freq = int(ssam_cfg.get('ssam_freq', 1))
        min_area = int(ssam_cfg.get('min_area', 300))
        fill_area_cfg = ssam_cfg.get('fill_area')
        if fill_area_cfg is None:
            fill_area = min_area
        else:
            try:
                fill_area = int(fill_area_cfg)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f'invalid stages.ssam.fill_area: {fill_area_cfg!r}') from exc
        fill_area = max(0, fill_area)
        stability = float(ssam_cfg.get('stability_threshold', 0.9))
        persist_raw = bool(ssam_cfg.get('persist_raw', True))
        skip_filtering = bool(ssam_cfg.get('skip_filtering', False))
        add_gaps = bool(ssam_cfg.get('add_gaps', False))
        append_timestamp = ssam_cfg.get('append_timestamp', True)
        experiment_tag = ssam_cfg.get('experiment_tag') or experiment_cfg.get('tag')
        sam_ckpt_cfg = ssam_cfg.get('sam_ckpt') or experiment_cfg.get('sam_ckpt')
        if sam_ckpt_cfg:
            sam_ckpt_path = Path(str(sam_ckpt_cfg)).expanduser()
        else:
            sam_ckpt_path = Path(DEFAULT_SEMANTIC_SAM_CKPT)
        if not sam_ckpt_path.exists():
            raise SystemExit(
                f'Semantic-SAM checkpoint not found at {sam_ckpt_path}. '
                'Set stages.ssam.sam_ckpt or experiment.sam_ckpt to a valid file path.'
            )
        sam_ckpt = str(sam_ckpt_path)

        ssam_gpu = resolve_stage_gpu(ssam_cfg, default_stage_gpu)

        print('Stage SSAM: Semantic-SAM 採樣與候選輸出')
        with StageRecorder(summary, 'ssam', ssam_gpu), using_gpu(ssam_gpu):
            run_root, manifest = run_candidate_generation(
                data_path=data_path,
                levels=ssam_levels_csv,
                frames=frames_str,
                sam_ckpt=sam_ckpt,
                output=output_root,
                min_area=min_area,
                fill_area=fill_area,
                stability_threshold=stability,
                add_gaps=add_gaps,
                no_timestamp=not append_timestamp,
                ssam_freq=ssam_freq,
                sam2_max_propagate=ssam_cfg.get('sam2_max_propagate'),
                experiment_tag=experiment_tag,
                persist_raw=persist_raw,
                skip_filtering=skip_filtering,
            )
            run_dir = Path(run_root)
            summary['stages']['ssam'].update(
                {
                    'params': {
                        'levels': levels,
                        'frames': frames_str,
                        'ssam_freq': ssam_freq,
                        'min_area': min_area,
                        'fill_area': fill_area,
                        'stability_threshold': stability,
                        'persist_raw': persist_raw,
                        'skip_filtering': skip_filtering,
                    }
                }
            )
    else:
        reuse_run_dir = experiment_cfg.get('run_dir')
        if not reuse_run_dir:
            raise SystemExit('SSAM stage disabled but experiment.run_dir not provided')
        run_dir = Path(reuse_run_dir).expanduser()
        if not run_dir.exists():
            raise SystemExit(f'Provided run_dir {run_dir} does not exist')
        manifest = load_manifest(run_dir)

    if run_dir is None:
        raise SystemExit('Failed to determine run directory from SSAM stage')

    manifest = manifest or load_manifest(run_dir)
    manifest_path = run_dir / 'manifest.json'

    # Stage: Filtering
    filter_cfg = stages_cfg.get('filter', {}) if isinstance(stages_cfg.get('filter'), dict) else {}
    filter_enabled = filter_cfg.get('enabled', True)
    if filter_enabled:
        levels = resolve_levels(filter_cfg, manifest, experiment_cfg.get('levels'))
        filter_gpu = resolve_stage_gpu(filter_cfg, default_stage_gpu)
        min_area = int(filter_cfg.get('min_area', ssam_cfg.get('min_area', 300)))
        stability = float(filter_cfg.get('stability_threshold', ssam_cfg.get('stability_threshold', 0.9)))
        update_manifest_flag = filter_cfg.get('update_manifest', True)
        filter_quiet = bool(filter_cfg.get('quiet', False))
        raw_available = any(
            (run_dir / f'level_{lvl}' / RAW_DIR_NAME).exists() for lvl in levels
        )
        if not raw_available:
            print('Stage Filter: 找不到 raw 候選資料，略過此階段')
            summary.setdefault('order', []).append('filter')
            summary.setdefault('stages', {}).setdefault('filter', {})['skipped'] = 'missing_raw'
        else:
            print('Stage Filter: 重新套用候選遮罩篩選條件')
            with StageRecorder(summary, 'filter', filter_gpu), using_gpu(filter_gpu):
                run_filtering(
                    root=str(run_dir),
                    levels=levels,
                    min_area=min_area,
                    stability_threshold=stability,
                    update_manifest=update_manifest_flag,
                    quiet=filter_quiet,
                )
                summary['stages']['filter'].update(
                    {
                        'params': {
                            'levels': levels,
                            'min_area': min_area,
                            'stability_threshold': stability,
                            'update_manifest': update_manifest_flag,
                        }
                    }
                )
            manifest = load_manifest(run_dir)

    # Stage: Tracking
    tracker_cfg = stages_cfg.get('tracker', {}) if isinstance(stages_cfg.get('tracker'), dict) else {}
    tracker_enabled = tracker_cfg.get('enabled', True)
    if tracker_enabled:
        levels = resolve_levels(tracker_cfg, manifest, experiment_cfg.get('levels'))
        tracker_gpu = resolve_stage_gpu(tracker_cfg, default_stage_gpu)
        max_propagate = tracker_cfg.get('max_propagate')
        iou_threshold = float(tracker_cfg.get('iou_threshold', 0.6))
        prompt_mode = tracker_cfg.get('prompt_mode', 'none')
        prompt_mode = str(prompt_mode).lower()
        if prompt_mode not in {'none', 'long_tail', 'all'}:
            raise SystemExit(f'Unknown tracker.prompt_mode={prompt_mode}')
        all_box = prompt_mode == 'all'
        long_tail_box = prompt_mode == 'long_tail'

        downscale_enabled = bool(tracker_cfg.get('downscale_masks', False))
        downscale_ratio = tracker_cfg.get('downscale_ratio', 0.3)
        try:
            downscale_ratio = float(downscale_ratio)
        except (TypeError, ValueError):
            raise SystemExit(f'Invalid tracker.downscale_ratio={downscale_ratio!r}')
        mask_scale_ratio = downscale_ratio if downscale_enabled else 1.0
        if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
            raise SystemExit('tracker.downscale_ratio must be in (0, 1] when downscale_masks is true')

        sam2_cfg = tracker_cfg.get('sam2_cfg') or experiment_cfg.get('sam2_cfg')
        sam2_ckpt = tracker_cfg.get('sam2_ckpt') or experiment_cfg.get('sam2_ckpt')

        print('Stage Tracker: SAM2 追蹤與遮罩匯出')
        with StageRecorder(summary, 'tracker', tracker_gpu), using_gpu(tracker_gpu):
            run_candidate_tracking(
                data_path=data_path,
                candidates_root=str(run_dir),
                output=str(run_dir),
                levels=list_to_csv(levels),
                sam2_cfg=sam2_cfg,
                sam2_ckpt=sam2_ckpt,
                sam2_max_propagate=max_propagate,
                iou_threshold=iou_threshold,
                long_tail_box_prompt=long_tail_box,
                all_box_prompt=all_box,
                mask_scale_ratio=mask_scale_ratio,
            )
            summary['stages']['tracker'].update(
                {
                    'params': {
                        'levels': levels,
                        'max_propagate': max_propagate,
                        'iou_threshold': iou_threshold,
                        'prompt_mode': prompt_mode,
                        'downscale_masks': downscale_enabled,
                        'downscale_ratio': mask_scale_ratio,
                    }
                }
            )

    # Stage: Report
    report_cfg = stages_cfg.get('report', {}) if isinstance(stages_cfg.get('report'), dict) else {}
    report_enabled = report_cfg.get('enabled', True)
    report_name = report_cfg.get('name', 'report.md')
    max_width = int(report_cfg.get('max_width', 960))

    if report_enabled:
        print('Stage Report: 生成 Markdown 紀錄')
        report_gpu = resolve_stage_gpu(report_cfg, default_stage_gpu)
        with StageRecorder(summary, 'report', report_gpu), using_gpu(report_gpu):
            build_report(run_dir, report_name=report_name, max_preview_width=max_width)
            summary['stages']['report'].setdefault('params', {})['max_width'] = max_width
            summary['stages']['report']['params']['report_name'] = report_name

    summary['generated_at'] = datetime.utcnow().isoformat(timespec='seconds')
    summary['run_dir'] = str(run_dir)

    with (run_dir / 'workflow_summary.json').open('w') as f:
        json.dump(summary, f, indent=2)

    print(f'Workflow finished. 輸出路徑：{run_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
