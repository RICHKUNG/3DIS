#!/usr/bin/env python3
"""High-level workflow orchestrator driven by YAML config."""

from __future__ import annotations

import argparse
import csv
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


def export_stage_timings(summary: Dict[str, Any], output_path: Path) -> None:
    """Persist stage timing metadata as a JSON artifact."""
    stages = summary.get('stages')
    if not stages:
        return

    ordered_names: List[str] = []
    order = summary.get('order')
    if isinstance(order, list):
        ordered_names.extend(name for name in order if name in stages)

    for name in stages:
        if name not in ordered_names:
            ordered_names.append(name)

    records: List[Dict[str, Any]] = []
    total_duration = 0.0
    for name in ordered_names:
        meta = stages.get(name, {})
        duration_raw = meta.get('duration_sec')
        try:
            duration = float(duration_raw) if duration_raw is not None else 0.0
        except (TypeError, ValueError):
            duration = 0.0
        total_duration += duration
        records.append(
            {
                'stage': name,
                'duration_sec': duration,
                'duration_text': meta.get('duration_text'),
                'started_at': meta.get('started_at'),
                'ended_at': meta.get('ended_at'),
                'gpu': meta.get('gpu'),
            }
        )

    payload = {
        'generated_at': datetime.utcnow().isoformat(timespec='seconds'),
        'total_duration_sec': total_duration,
        'total_duration_text': format_duration(total_duration),
        'stages': records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def expand_output_path_template(path_value: Any, experiment_cfg: Dict[str, Any]) -> str:
    if path_value is None:
        raise SystemExit('experiment.output_root is required (or override via --override-output)')

    path_str = str(path_value)
    if '{name}' in path_str:
        experiment_name = experiment_cfg.get('name')
        if not experiment_name:
            raise SystemExit(
                'experiment.output_root 使用了 {name} 但未提供 experiment.name'
            )
        path_str = path_str.replace('{name}', str(experiment_name))

    return path_str


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


def derive_scene_metadata(data_path: str) -> Dict[str, Optional[str]]:
    path = Path(str(data_path)).expanduser()
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path.absolute()

    scene_name: Optional[str] = None
    scene_root: Optional[Path] = None
    dataset_root: Optional[Path] = None
    for candidate in [resolved] + list(resolved.parents):
        if candidate.name.startswith('scene_'):
            scene_name = candidate.name
            scene_root = candidate
            dataset_root = candidate.parent
            break

    return {
        'scene': scene_name,
        'scene_root': str(scene_root) if scene_root else None,
        'dataset_root': str(dataset_root) if dataset_root else None,
    }


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


def append_run_history(
    summary: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    history_root: Optional[Path] = None,
) -> None:
    history_root = history_root or Path(__file__).resolve().parent / 'logs'
    try:
        history_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    history_path = history_root / 'workflow_history.csv'
    is_new = not history_path.exists()

    experiment = summary.get('experiment', {}) if isinstance(summary.get('experiment'), dict) else {}
    stages = summary.get('stages', {}) if isinstance(summary.get('stages'), dict) else {}
    ssam_params = stages.get('ssam', {}).get('params', {}) if isinstance(stages.get('ssam'), dict) else {}

    def _flatten_levels(levels_val: Any) -> str:
        if isinstance(levels_val, (list, tuple)):
            return ','.join(str(v) for v in levels_val)
        return str(levels_val) if levels_val is not None else ''

    entry = {
        'timestamp': summary.get('generated_at') or datetime.utcnow().isoformat(timespec='seconds'),
        'scene': experiment.get('scene') or '',
        'experiment_name': experiment.get('name') or '',
        'config_path': summary.get('config_path') or '',
        'data_path': experiment.get('data_path') or '',
        'output_root': experiment.get('output_root') or '',
        'run_dir': summary.get('run_dir') or '',
        'levels': _flatten_levels(experiment.get('levels')),
        'ssam_freq': ssam_params.get('ssam_freq') or '',
        'frames_total': '',
        'frames_selected': '',
        'frames_ssam': '',
        'parent_experiment': experiment.get('parent_experiment') or '',
        'experiment_root': experiment.get('experiment_root') or '',
        'scene_output_root': experiment.get('scene_output_root') or '',
        'scene_index': '' if experiment.get('scene_index') is None else str(experiment.get('scene_index')),
    }

    if manifest:
        entry['frames_total'] = manifest.get('frames_total', '')
        entry['frames_selected'] = manifest.get('frames_selected', '')
        entry['frames_ssam'] = manifest.get('frames_ssam', '')

    fieldnames = [
        'timestamp',
        'scene',
        'experiment_name',
        'config_path',
        'data_path',
        'output_root',
        'run_dir',
        'levels',
        'ssam_freq',
        'frames_total',
        'frames_selected',
        'frames_ssam',
        'parent_experiment',
        'experiment_root',
        'scene_output_root',
        'scene_index',
    ]

    try:
        with history_path.open('a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(entry)
    except OSError:
        return


def _run_scene_workflow(
    *,
    config: Dict[str, Any],
    experiment_cfg: Dict[str, Any],
    stages_cfg: Dict[str, Any],
    default_stage_gpu: Optional[Any],
    data_path: str,
    output_root: str,
    config_path: Optional[Path],
    parent_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data_path = str(Path(data_path).expanduser())
    output_root = str(Path(output_root).expanduser())

    summary: Dict[str, Any] = {
        'config_path': str(config_path) if config_path else None,
        'invoked_at': datetime.utcnow().isoformat(timespec='seconds'),
    }
    update_summary_config(summary, config)

    scene_meta = derive_scene_metadata(data_path)
    experiment_name = (
        experiment_cfg.get('name')
        or (parent_meta.get('name') if parent_meta else None)
        or scene_meta.get('scene')
    )
    summary['experiment'] = {
        'name': experiment_name,
        'scene': scene_meta.get('scene'),
        'scene_root': scene_meta.get('scene_root'),
        'dataset_root': scene_meta.get('dataset_root'),
        'data_path': data_path,
        'output_root': output_root,
        'levels': experiment_cfg.get('levels'),
        'tag': experiment_cfg.get('tag'),
        'scene_output_root': output_root,
    }
    if parent_meta:
        summary['experiment']['parent_experiment'] = parent_meta.get('name')
        summary['experiment']['experiment_root'] = parent_meta.get('experiment_root')
        summary['experiment']['scene_index'] = parent_meta.get('index')
        if parent_meta.get('scenes') is not None:
            summary['experiment']['scene_list'] = parent_meta.get('scenes')

    manifest: Optional[Dict[str, Any]] = None
    run_dir: Optional[Path] = None

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

    tracker_cfg = stages_cfg.get('tracker', {}) if isinstance(stages_cfg.get('tracker'), dict) else {}
    tracker_enabled = tracker_cfg.get('enabled', True)
    if tracker_enabled:
        levels = resolve_levels(tracker_cfg, manifest, experiment_cfg.get('levels'))
        tracker_gpu = resolve_stage_gpu(tracker_cfg, default_stage_gpu)
        max_propagate = tracker_cfg.get('max_propagate')
        iou_threshold = float(tracker_cfg.get('iou_threshold', 0.6))
        prompt_mode_raw = str(tracker_cfg.get('prompt_mode', 'all_mask')).lower()
        prompt_aliases = {
            'none': 'all_mask',
            'all_mask': 'all_mask',
            'long_tail': 'lt_bbox',
            'lt_bbox': 'lt_bbox',
            'all': 'all_bbox',
            'all_bbox': 'all_bbox',
        }
        if prompt_mode_raw not in prompt_aliases:
            raise SystemExit(f'Unknown tracker.prompt_mode={prompt_mode_raw}')
        prompt_mode = prompt_aliases[prompt_mode_raw]
        all_box = prompt_mode == 'all_bbox'
        long_tail_box = prompt_mode == 'lt_bbox'

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

        record_timings_flag = bool(report_cfg.get('record_timings'))
        timing_output_name = report_cfg.get('timing_output')
        if record_timings_flag or timing_output_name:
            output_name = timing_output_name or 'stage_timings.json'
            timings_path = run_dir / output_name
            export_stage_timings(summary, timings_path)
            summary['stages']['report'].setdefault('artifacts', {})['timings'] = str(timings_path)

    summary['generated_at'] = datetime.utcnow().isoformat(timespec='seconds')
    summary['run_dir'] = str(run_dir)

    with (run_dir / 'workflow_summary.json').open('w') as f:
        json.dump(summary, f, indent=2)

    append_run_history(summary, manifest)

    print(f'Workflow finished. 輸出路徑：{run_dir}')
    return summary


def execute_workflow(
    config: Dict[str, Any],
    *,
    override_output: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    experiment_cfg = config.get('experiment', {})
    if not isinstance(experiment_cfg, dict):
        raise SystemExit('`experiment` section must be a mapping')

    stages_cfg = config.get('stages', {})
    if not isinstance(stages_cfg, dict):
        raise SystemExit('`stages` section must be a mapping')
    default_stage_gpu = stages_cfg.get('gpu')

    scenes = experiment_cfg.get('scenes')
    if scenes:
        if not isinstance(scenes, (list, tuple)):
            raise SystemExit('experiment.scenes must be a list when provided')
        dataset_root_raw = experiment_cfg.get('dataset_root')
        if not dataset_root_raw:
            raise SystemExit('experiment.dataset_root is required when experiment.scenes is provided')
        dataset_root = Path(dataset_root_raw).expanduser()
        output_root_base_raw = override_output or experiment_cfg.get('output_root')
        output_root_base_resolved = expand_output_path_template(output_root_base_raw, experiment_cfg)
        output_root_base = Path(output_root_base_resolved).expanduser()
        parent_meta = {
            'name': experiment_cfg.get('name'),
            'scenes': [str(s) for s in scenes],
            'experiment_root': str(output_root_base),
        }

        summaries: List[Dict[str, Any]] = []
        for index, scene_name in enumerate(parent_meta['scenes']):
            scene_data_path = dataset_root / scene_name / 'outputs' / 'color'
            if not scene_data_path.exists():
                raise SystemExit(f'data path not found for scene {scene_name}: {scene_data_path}')
            scene_output_root = output_root_base / scene_name
            scene_experiment_cfg = dict(experiment_cfg)
            scene_experiment_cfg['name'] = experiment_cfg.get('name')
            scene_experiment_cfg['data_path'] = str(scene_data_path)
            scene_experiment_cfg['output_root'] = str(scene_output_root)
            parent_meta_scene = dict(parent_meta)
            parent_meta_scene['index'] = index
            parent_meta_scene['scene'] = scene_name
            summary = _run_scene_workflow(
                config=config,
                experiment_cfg=scene_experiment_cfg,
                stages_cfg=stages_cfg,
                default_stage_gpu=default_stage_gpu,
                data_path=str(scene_data_path),
                output_root=str(scene_output_root),
                config_path=config_path,
                parent_meta=parent_meta_scene,
            )
            summaries.append(summary)
        return summaries

    data_path_raw = experiment_cfg.get('data_path')
    if not data_path_raw:
        raise SystemExit('experiment.data_path is required')
    output_root_raw = expand_output_path_template(override_output or experiment_cfg.get('output_root'), experiment_cfg)

    summary = _run_scene_workflow(
        config=config,
        experiment_cfg=experiment_cfg,
        stages_cfg=stages_cfg,
        default_stage_gpu=default_stage_gpu,
        data_path=data_path_raw,
        output_root=output_root_raw,
        config_path=config_path,
        parent_meta=None,
    )
    return [summary]


def main() -> int:
    parser = argparse.ArgumentParser(description='Run workflow defined by a YAML config')
    parser.add_argument('--config', required=True, help='Path to YAML workflow config')
    parser.add_argument('--override-output', help='Override output root directory')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)

    execute_workflow(
        config,
        override_output=args.override_output,
        config_path=config_path,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
