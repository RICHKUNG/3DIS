#!/usr/bin/env python3
"""High-level workflow orchestrator driven by YAML config."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Set, Tuple

try:  # pragma: no cover - platform specific
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

try:  # pragma: no cover - platform specific
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None  # type: ignore

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

from my3dis.common_utils import RAW_DIR_NAME, format_duration, list_to_csv
from my3dis.generate_candidates import (
    DEFAULT_SEMANTIC_SAM_CKPT,
    run_generation as run_candidate_generation,
)
from my3dis.filter_candidates import run_filtering
from my3dis.track_from_candidates import run_tracking as run_candidate_tracking
from my3dis.generate_report import build_report
from oom_monitor import memory_watch_context
from oom_monitor.notifier import create_notifier


class WorkflowError(Exception):
    """Base class for workflow related failures."""


class WorkflowConfigError(WorkflowError):
    """Raised when the provided configuration is invalid."""


class WorkflowRuntimeError(WorkflowError):
    """Raised when runtime side-effects (files, IO, etc.) fail."""


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open('r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise WorkflowConfigError(f'Config file not found: {path}') from exc
    if not isinstance(data, dict):
        raise WorkflowConfigError(f'Config file {path} must define a mapping at the top level')
    return data


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


def _now_local_iso() -> str:
    """Return ISO timestamp using the system local timezone."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _now_local_stamp() -> str:
    """Return folder-friendly timestamp using local time."""
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')


def _build_completion_email_payload(
    status: str,
    *,
    config_path: Path,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    run_summaries: Optional[List[Dict[str, Any]]],
    error_message: Optional[str] = None,
    traceback_text: Optional[str] = None,
) -> Tuple[str, str]:
    status_upper = status.upper()
    config_display = str(config_path)
    duration_text = format_duration(duration_seconds)
    subject = f'[My3DIS] Workflow {status_upper}: {Path(config_display).name}'

    lines = [
        f'Status: {status_upper}',
        f'Config: {config_display}',
        f'Started: {started_at}',
        f'Finished: {finished_at}',
        f'Duration: {duration_text} ({duration_seconds:.1f}s)',
    ]

    if run_summaries:
        lines.append('')
        lines.append('Runs:')
        for index, summary in enumerate(run_summaries, start=1):
            experiment_meta = summary.get('experiment') if isinstance(summary, dict) else None
            scene_name = None
            if isinstance(experiment_meta, dict):
                scene_name = experiment_meta.get('scene') or experiment_meta.get('name')
            run_dir = summary.get('run_dir') if isinstance(summary, dict) else None
            label = scene_name or f'Run #{index}'
            if run_dir:
                lines.append(f'- {label}: {run_dir}')
            else:
                lines.append(f'- {label}')

    if error_message:
        lines.append('')
        lines.append(f'Error: {error_message}')

    if traceback_text:
        lines.append('')
        lines.append('Traceback:')
        lines.append(traceback_text)

    body = '\n'.join(lines)
    return subject, body


def _send_completion_notification(notifier, subject: str, body: str) -> None:
    if notifier is None:
        return
    try:
        notifier.send(subject, body)
    except Exception as exc:  # pragma: no cover - notification best effort
        print(f'Failed to send notification email: {exc}', file=sys.stderr)


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
        now_iso = _now_local_iso()
        self.summary.setdefault('order', []).append(self.name)
        self.summary.setdefault('stages', {})[self.name] = {
            'started_at': now_iso,
            'gpu': self.gpu,
        }
        return self

    def __exit__(self, exc_type, exc, tb):
        end_time = time.perf_counter()
        duration = end_time - (self.start or end_time)
        now_iso = _now_local_iso()
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
        'generated_at': _now_local_iso(),
        'total_duration_sec': total_duration,
        'total_duration_text': format_duration(total_duration),
        'stages': records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def expand_output_path_template(path_value: Any, experiment_cfg: Dict[str, Any]) -> str:
    if path_value is None:
        raise WorkflowConfigError('experiment.output_root is required (or override via --override-output)')

    path_str = str(path_value)
    if '{name}' in path_str:
        experiment_name = experiment_cfg.get('name')
        if not experiment_name:
            raise WorkflowConfigError(
                'experiment.output_root 使用了 {name} 但未提供 experiment.name'
            )
        path_str = path_str.replace('{name}', str(experiment_name))

    return path_str


_ALL_SCENE_TOKENS = {'all', '*', '__all__'}


def discover_scene_names(dataset_root: Path) -> List[str]:
    if not dataset_root.exists():
        raise WorkflowConfigError(f'experiment.dataset_root does not exist: {dataset_root}')

    try:
        entries = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
    except OSError as exc:
        raise WorkflowRuntimeError(f'failed to list dataset_root={dataset_root}: {exc}') from exc

    preferred = [name for name in entries if name.startswith('scene_')]
    return preferred or entries


def normalize_scene_list(
    raw_scenes: Any,
    dataset_root: Path,
    *,
    scene_start: Optional[str] = None,
    scene_end: Optional[str] = None,
) -> List[str]:
    discovered_cache: Optional[List[str]] = None

    def ensure_discovered() -> List[str]:
        nonlocal discovered_cache
        if discovered_cache is None:
            discovered_cache = discover_scene_names(dataset_root)
        return discovered_cache

    if raw_scenes is None:
        scenes_iterable: List[str] = ensure_discovered()
    elif isinstance(raw_scenes, str):
        token = raw_scenes.strip()
        if token.lower() in _ALL_SCENE_TOKENS:
            scenes_iterable = ensure_discovered()
        else:
            scenes_iterable = [token]
    elif isinstance(raw_scenes, (list, tuple)):
        result: List[str] = []
        seen: Set[str] = set()
        for entry in raw_scenes:
            token = str(entry).strip()
            if not token:
                continue
            if token.lower() in _ALL_SCENE_TOKENS:
                for name in ensure_discovered():
                    if name not in seen:
                        result.append(name)
                        seen.add(name)
                continue
            if token not in seen:
                result.append(token)
                seen.add(token)
        scenes_iterable = result
    else:
        raise WorkflowConfigError('experiment.scenes must be a list, string, or null when provided')

    if not scenes_iterable:
        raise WorkflowConfigError('No scenes resolved for experiment (empty list after processing)')

    missing = [scene for scene in scenes_iterable if not (dataset_root / scene).exists()]
    if missing:
        raise WorkflowConfigError(
            'The following scenes were not found under dataset_root '
            f"{dataset_root}: {', '.join(missing)}"
        )

    if scene_start is not None or scene_end is not None:
        ordered = ensure_discovered()
        order_map = {name: idx for idx, name in enumerate(ordered)}

        if scene_start is not None:
            start_token = str(scene_start).strip()
            if start_token not in order_map:
                raise WorkflowConfigError(
                    f'scene_start {scene_start!r} not found under dataset_root {dataset_root}'
                )
            start_idx = order_map[start_token]
        else:
            start_idx = 0

        if scene_end is not None:
            end_token = str(scene_end).strip()
            if end_token not in order_map:
                raise WorkflowConfigError(
                    f'scene_end {scene_end!r} not found under dataset_root {dataset_root}'
                )
            end_idx = order_map[end_token]
        else:
            end_idx = len(ordered) - 1

        if end_idx < start_idx:
            raise WorkflowConfigError('scene_end occurs before scene_start; please provide a valid range')

        allowed: Set[str] = set(scenes_iterable)
        sliced = [name for name in ordered[start_idx : end_idx + 1] if name in allowed]
        if not sliced:
            raise WorkflowConfigError(
                'Scene range selection produced an empty set; adjust scene_start or scene_end'
            )
        scenes_iterable = sliced

    return scenes_iterable

def resolve_levels(stage_cfg: Dict[str, Any], manifest: Optional[Dict[str, Any]], fallback: Optional[List[int]]) -> List[int]:
    if 'levels' in stage_cfg and stage_cfg['levels'] is not None:
        try:
            return [int(x) for x in stage_cfg['levels']]
        except (TypeError, ValueError):
            raise WorkflowConfigError(f"Invalid levels in config: {stage_cfg['levels']}")
    if manifest and isinstance(manifest.get('levels'), list):
        try:
            return [int(x) for x in manifest['levels']]
        except (TypeError, ValueError):
            pass
    if fallback:
        try:
            return [int(x) for x in fallback]
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f'Invalid fallback levels: {fallback}') from exc
    raise WorkflowConfigError('Unable to determine levels for stage')


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


def _lock_file_handle(handle: IO[Any]) -> None:
    try:
        if fcntl is not None:  # pragma: no cover - platform specific
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - platform specific
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 0x7FFFFFFF)
    except OSError:
        # Best effort locking; fall back to unlocked writes when not supported.
        pass


def _unlock_file_handle(handle: IO[Any]) -> None:
    try:
        if fcntl is not None:  # pragma: no cover - platform specific
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - platform specific
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 0x7FFFFFFF)
    except OSError:
        pass


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

    experiment = summary.get('experiment', {}) if isinstance(summary.get('experiment'), dict) else {}
    stages = summary.get('stages', {}) if isinstance(summary.get('stages'), dict) else {}
    ssam_params = stages.get('ssam', {}).get('params', {}) if isinstance(stages.get('ssam'), dict) else {}

    def _flatten_levels(levels_val: Any) -> str:
        if isinstance(levels_val, (list, tuple)):
            return ','.join(str(v) for v in levels_val)
        return str(levels_val) if levels_val is not None else ''

    entry = {
        'timestamp': summary.get('generated_at') or _now_local_iso(),
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
        with history_path.open('a+', newline='') as csvfile:
            _lock_file_handle(csvfile)
            try:
                csvfile.seek(0, os.SEEK_END)
                need_header = csvfile.tell() == 0
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if need_header:
                    writer.writeheader()
                writer.writerow(entry)
                csvfile.flush()
                try:
                    os.fsync(csvfile.fileno())
                except OSError:
                    pass
            finally:
                _unlock_file_handle(csvfile)
    except OSError:
        return


def _move_file(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst = src.replace(dst)
    except OSError as exc:
        raise WorkflowRuntimeError(f'Failed to move {src} → {dst}: {exc}') from exc
    return dst


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open('r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def apply_scene_level_layout(
    run_dir: Path,
    summary: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    manifest = manifest or {}
    levels_raw = manifest.get('levels') or summary.get('experiment', {}).get('levels')
    if not levels_raw:
        return None

    resolved_levels: List[int] = []
    for lvl in levels_raw:
        try:
            resolved_levels.append(int(lvl))
        except (TypeError, ValueError):
            continue

    if not resolved_levels:
        return None

    resolved_levels = sorted(set(resolved_levels))
    aggregated_levels: Dict[str, Dict[str, Any]] = {}

    tracking_artifacts = manifest.get('tracking_artifacts') or {}

    for lvl in resolved_levels:
        level_dir = run_dir / f'level_{lvl}'
        if not level_dir.exists():
            continue

        level_label = f'L{lvl:02d}'
        track_dir = level_dir / 'tracking'
        object_rel: Optional[str] = None
        video_rel: Optional[str] = None

        if track_dir.exists():
            object_candidates = sorted(track_dir.glob('object_segments*.npz'))
            object_src = object_candidates[0] if object_candidates else None
            if object_src:
                object_dst = level_dir / f'object_segments_{level_label}.npz'
                moved = _move_file(object_src, object_dst)
                if moved:
                    object_rel = str(moved.relative_to(run_dir))

            video_candidates = sorted(track_dir.glob('video_segments*.npz'))
            video_src = video_candidates[0] if video_candidates else None
            if video_src:
                video_dst = level_dir / f'video_segments_{level_label}.npz'
                moved = _move_file(video_src, video_dst)
                if moved:
                    video_rel = str(moved.relative_to(run_dir))

            # Remove tracking directory if empty after moves
            try:
                if track_dir.exists() and not any(track_dir.iterdir()):
                    track_dir.rmdir()
            except OSError:
                pass

        report_dir = level_dir / 'report'
        report_rel_paths: List[str] = []
        if report_dir.exists():
            image_paths = sorted(
                [p for p in report_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
            )
            for idx, image in enumerate(image_paths, start=1):
                dst_name = image.name
                # Avoid collisions by prefixing with level label when necessary
                if not dst_name.lower().startswith(f'{level_label.lower()}_'):
                    dst_name = f'{level_label}_{idx:02d}_{dst_name}'
                dest = level_dir / dst_name
                moved = _move_file(image, dest)
                if moved:
                    report_rel_paths.append(str(moved.relative_to(run_dir)))

            try:
                if report_dir.exists() and not any(report_dir.iterdir()):
                    report_dir.rmdir()
            except OSError:
                pass

        aggregated_levels[str(lvl)] = {
            'object_segments': object_rel,
            'video_segments': video_rel,
            'report_images': report_rel_paths,
        }

        manifest_key = f'level_{lvl}'
        tracking_artifacts.setdefault(manifest_key, {})
        if object_rel:
            tracking_artifacts[manifest_key]['object_segments'] = object_rel
        if video_rel:
            tracking_artifacts[manifest_key]['video_segments'] = video_rel

        # Clean up visualization directory when present (optional artifacts)
        viz_dir = level_dir / 'viz'
        if viz_dir.exists():
            try:
                shutil.rmtree(viz_dir)
            except OSError:
                pass

    manifest['tracking_artifacts'] = tracking_artifacts
    manifest_path = run_dir / 'manifest.json'
    try:
        with manifest_path.open('w') as f:
            json.dump(manifest, f, indent=2)
    except OSError:
        pass

    timings_payload = _load_json_if_exists(run_dir / 'stage_timings.json')

    summary_payload = {
        'generated_at': summary.get('generated_at'),
        'scene': summary.get('experiment', {}).get('scene'),
        'experiment': summary.get('experiment'),
        'config': summary.get('config_snapshot'),
        'stages': summary.get('stages'),
        'manifest': manifest,
        'levels': aggregated_levels,
        'timings': timings_payload,
    }

    summary_path = run_dir / 'summary.json'
    try:
        with summary_path.open('w') as f:
            json.dump(summary_payload, f, indent=2)
    except OSError:
        pass

    summary.setdefault('artifacts', {})['scene_layout'] = aggregated_levels
    summary['summary_json'] = str(summary_path)

    return summary_payload


@dataclass
class SceneContext:
    config: Dict[str, Any]
    experiment_cfg: Dict[str, Any]
    stages_cfg: Dict[str, Any]
    default_stage_gpu: Optional[Any]
    data_path: str
    output_root: str
    config_path: Optional[Path]
    parent_meta: Optional[Dict[str, Any]] = None


class SceneWorkflow:
    def __init__(self, context: SceneContext) -> None:
        self.config = context.config
        self.experiment_cfg = context.experiment_cfg
        self.stages_cfg = context.stages_cfg
        self.default_stage_gpu = context.default_stage_gpu
        self.data_path = str(Path(context.data_path).expanduser())
        self.output_root = str(Path(context.output_root).expanduser())
        self.config_path = context.config_path
        self.parent_meta = context.parent_meta

        self.summary: Dict[str, Any] = {
            'config_path': str(self.config_path) if self.config_path else None,
            'invoked_at': _now_local_iso(),
        }
        update_summary_config(self.summary, self.config)

        self.output_layout_mode = self._determine_layout_mode()
        self.run_dir: Optional[Path] = None
        self.manifest: Optional[Dict[str, Any]] = None

        self._populate_experiment_metadata()

    def run(self) -> Dict[str, Any]:
        self._run_ssam_stage()
        self._run_filter_stage()
        self._run_tracker_stage()
        self._run_report_stage()
        self._finalize()
        return self.summary

    def _stage_cfg(self, name: str) -> Dict[str, Any]:
        raw = self.stages_cfg.get(name)
        return raw if isinstance(raw, dict) else {}

    def _stage_summary(self, name: str) -> Dict[str, Any]:
        return self.summary.setdefault('stages', {}).setdefault(name, {})

    def _determine_layout_mode(self) -> Optional[str]:
        raw = self.experiment_cfg.get('output_layout')
        if isinstance(raw, str):
            value = raw.strip().lower()
            return value or None
        return None

    def _populate_experiment_metadata(self) -> None:
        scene_meta = derive_scene_metadata(self.data_path)
        experiment_name = (
            self.experiment_cfg.get('name')
            or (self.parent_meta.get('name') if self.parent_meta else None)
            or scene_meta.get('scene')
        )
        experiment_summary = {
            'name': experiment_name,
            'scene': scene_meta.get('scene'),
            'scene_root': scene_meta.get('scene_root'),
            'dataset_root': scene_meta.get('dataset_root'),
            'data_path': self.data_path,
            'output_root': self.output_root,
            'levels': self.experiment_cfg.get('levels'),
            'tag': self.experiment_cfg.get('tag'),
            'scene_output_root': self.output_root,
        }
        if self.parent_meta:
            experiment_summary['parent_experiment'] = self.parent_meta.get('name')
            experiment_summary['experiment_root'] = self.parent_meta.get('experiment_root')
            experiment_summary['scene_index'] = self.parent_meta.get('index')
            if self.parent_meta.get('scenes') is not None:
                experiment_summary['scene_list'] = self.parent_meta.get('scenes')
            if self.parent_meta.get('scene_start') is not None:
                experiment_summary['scene_start'] = self.parent_meta.get('scene_start')
            if self.parent_meta.get('scene_end') is not None:
                experiment_summary['scene_end'] = self.parent_meta.get('scene_end')
        self.summary['experiment'] = experiment_summary

    def _ensure_run_dir(self) -> Path:
        if self.run_dir is None:
            raise WorkflowRuntimeError('Run directory not resolved; ensure SSAM stage executed properly.')
        return self.run_dir

    def _ensure_manifest(self) -> Optional[Dict[str, Any]]:
        if self.manifest is None and self.run_dir is not None:
            self.manifest = load_manifest(self.run_dir)
        return self.manifest

    def _run_ssam_stage(self) -> None:
        stage_cfg = self._stage_cfg('ssam')
        if not stage_cfg.get('enabled', True):
            reuse_run_dir = self.experiment_cfg.get('run_dir')
            if not reuse_run_dir:
                raise WorkflowConfigError('SSAM stage disabled but experiment.run_dir not provided')
            run_dir = Path(str(reuse_run_dir)).expanduser()
            if not run_dir.exists():
                raise WorkflowConfigError(f'Provided run_dir {run_dir} does not exist')
            self.run_dir = run_dir
            self.manifest = load_manifest(run_dir)
            return

        levels = resolve_levels(stage_cfg, None, self.experiment_cfg.get('levels'))
        frames_str = stage_frames_string(stage_cfg)
        try:
            ssam_freq = int(stage_cfg.get('ssam_freq', 1))
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f"invalid stages.ssam.ssam_freq: {stage_cfg.get('ssam_freq')!r}") from exc
        min_area = int(stage_cfg.get('min_area', 300))
        fill_area_cfg = stage_cfg.get('fill_area')
        if fill_area_cfg is None:
            fill_area = min_area
        else:
            try:
                fill_area = int(fill_area_cfg)
            except (TypeError, ValueError) as exc:
                raise WorkflowConfigError(f'invalid stages.ssam.fill_area: {fill_area_cfg!r}') from exc
        fill_area = max(0, fill_area)
        stability = float(stage_cfg.get('stability_threshold', 0.9))
        persist_raw = bool(stage_cfg.get('persist_raw', True))
        skip_filtering = bool(stage_cfg.get('skip_filtering', False))
        add_gaps = bool(stage_cfg.get('add_gaps', False))
        append_timestamp = stage_cfg.get('append_timestamp', True)
        experiment_tag = stage_cfg.get('experiment_tag') or self.experiment_cfg.get('tag')
        ssam_downscale_enabled = bool(stage_cfg.get('downscale_masks', False))
        ssam_downscale_ratio = stage_cfg.get('downscale_ratio', stage_cfg.get('mask_scale_ratio', 1.0))
        try:
            ssam_downscale_ratio = float(ssam_downscale_ratio)
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(
                f'invalid stages.ssam.downscale_ratio: {ssam_downscale_ratio!r}'
            ) from exc

        sam_ckpt_cfg = stage_cfg.get('sam_ckpt') or self.experiment_cfg.get('sam_ckpt')
        if sam_ckpt_cfg:
            sam_ckpt_path = Path(str(sam_ckpt_cfg)).expanduser()
        else:
            sam_ckpt_path = Path(DEFAULT_SEMANTIC_SAM_CKPT)
        if not sam_ckpt_path.exists():
            raise WorkflowConfigError(
                f'Semantic-SAM checkpoint not found at {sam_ckpt_path}. '
                'Set stages.ssam.sam_ckpt or experiment.sam_ckpt to a valid file path.'
            )
        sam_ckpt = str(sam_ckpt_path)

        ssam_gpu = resolve_stage_gpu(stage_cfg, self.default_stage_gpu)

        print('Stage SSAM: Semantic-SAM 採樣與候選輸出')
        with StageRecorder(self.summary, 'ssam', ssam_gpu), using_gpu(ssam_gpu):
            run_root, manifest = run_candidate_generation(
                data_path=self.data_path,
                levels=list_to_csv(levels),
                frames=frames_str,
                sam_ckpt=sam_ckpt,
                output=self.output_root,
                min_area=min_area,
                fill_area=fill_area,
                stability_threshold=stability,
                add_gaps=add_gaps,
                no_timestamp=not append_timestamp,
                ssam_freq=ssam_freq,
                sam2_max_propagate=stage_cfg.get('sam2_max_propagate'),
                experiment_tag=experiment_tag,
                persist_raw=persist_raw,
                skip_filtering=skip_filtering,
                downscale_masks=ssam_downscale_enabled,
                mask_scale_ratio=ssam_downscale_ratio,
            )
        self.run_dir = Path(run_root)
        self.manifest = manifest if isinstance(manifest, dict) else manifest

        mask_meta = self.manifest.get('mask_downscale', {}) if isinstance(self.manifest, dict) else {}
        try:
            actual_ratio = float(self.manifest.get('mask_scale_ratio', 1.0)) if isinstance(self.manifest, dict) else 1.0
        except (TypeError, ValueError):
            actual_ratio = 1.0
        actual_enabled = bool(mask_meta.get('enabled', actual_ratio < 1.0))

        stage_summary = self._stage_summary('ssam')
        stage_summary.update(
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
                    'downscale_masks': actual_enabled,
                    'downscale_ratio': actual_ratio,
                }
            }
        )

    def _run_filter_stage(self) -> None:
        stage_cfg = self._stage_cfg('filter')
        if not stage_cfg.get('enabled', True):
            return

        manifest = self._ensure_manifest()
        levels = resolve_levels(stage_cfg, manifest, self.experiment_cfg.get('levels'))
        filter_gpu = resolve_stage_gpu(stage_cfg, self.default_stage_gpu)
        ssam_cfg = self._stage_cfg('ssam')
        min_area = int(stage_cfg.get('min_area', ssam_cfg.get('min_area', 300)))
        stability = float(stage_cfg.get('stability_threshold', ssam_cfg.get('stability_threshold', 0.9)))
        update_manifest_flag = stage_cfg.get('update_manifest', True)
        filter_quiet = bool(stage_cfg.get('quiet', False))

        run_dir = self._ensure_run_dir()
        raw_available = any((run_dir / f'level_{lvl}' / RAW_DIR_NAME).exists() for lvl in levels)
        if not raw_available:
            print('Stage Filter: 找不到 raw 候選資料，略過此階段')
            self.summary.setdefault('order', []).append('filter')
            self._stage_summary('filter')['skipped'] = 'missing_raw'
            return

        print('Stage Filter: 重新套用候選遮罩篩選條件')
        with StageRecorder(self.summary, 'filter', filter_gpu), using_gpu(filter_gpu):
            run_filtering(
                root=str(run_dir),
                levels=levels,
                min_area=min_area,
                stability_threshold=stability,
                update_manifest=update_manifest_flag,
                quiet=filter_quiet,
            )

        self.manifest = load_manifest(run_dir)
        stage_summary = self._stage_summary('filter')
        stage_summary.update(
            {
                'params': {
                    'levels': levels,
                    'min_area': min_area,
                    'stability_threshold': stability,
                    'update_manifest': update_manifest_flag,
                }
            }
        )

    def _run_tracker_stage(self) -> None:
        stage_cfg = self._stage_cfg('tracker')
        if not stage_cfg.get('enabled', True):
            return

        manifest = self._ensure_manifest() or {}
        levels = resolve_levels(stage_cfg, manifest, self.experiment_cfg.get('levels'))
        tracker_gpu = resolve_stage_gpu(stage_cfg, self.default_stage_gpu)
        max_propagate = stage_cfg.get('max_propagate')
        iou_threshold = float(stage_cfg.get('iou_threshold', 0.6))
        prompt_mode_raw = str(stage_cfg.get('prompt_mode', 'all_mask')).lower()
        prompt_aliases = {
            'none': 'all_mask',
            'all_mask': 'all_mask',
            'long_tail': 'lt_bbox',
            'lt_bbox': 'lt_bbox',
            'all': 'all_bbox',
            'all_bbox': 'all_bbox',
        }
        if prompt_mode_raw not in prompt_aliases:
            raise WorkflowConfigError(f'Unknown tracker.prompt_mode={prompt_mode_raw}')
        prompt_mode = prompt_aliases[prompt_mode_raw]
        all_box = prompt_mode == 'all_bbox'
        long_tail_box = prompt_mode == 'lt_bbox'

        try:
            mask_ratio_default = float(manifest.get('mask_scale_ratio', 1.0)) if manifest else 1.0
        except (TypeError, ValueError):
            mask_ratio_default = 1.0
        downscale_enabled = bool(stage_cfg.get('downscale_masks', mask_ratio_default < 1.0))
        downscale_ratio = stage_cfg.get(
            'downscale_ratio', mask_ratio_default if mask_ratio_default < 1.0 else 0.3
        )
        try:
            downscale_ratio = float(downscale_ratio)
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigError(f'Invalid tracker.downscale_ratio={downscale_ratio!r}') from exc
        mask_scale_ratio = downscale_ratio if downscale_enabled else 1.0
        if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
            raise WorkflowConfigError('tracker.downscale_ratio must be in (0, 1] when downscale_masks is true')

        sam2_cfg = stage_cfg.get('sam2_cfg') or self.experiment_cfg.get('sam2_cfg')
        sam2_ckpt = stage_cfg.get('sam2_ckpt') or self.experiment_cfg.get('sam2_ckpt')
        render_viz = bool(stage_cfg.get('render_viz', True))

        run_dir = self._ensure_run_dir()
        print('Stage Tracker: SAM2 追蹤與遮罩匯出')
        with StageRecorder(self.summary, 'tracker', tracker_gpu), using_gpu(tracker_gpu):
            run_candidate_tracking(
                data_path=self.data_path,
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
                render_viz=render_viz,
            )

        stage_summary = self._stage_summary('tracker')
        stage_summary.update(
            {
                'params': {
                    'levels': levels,
                    'max_propagate': max_propagate,
                    'iou_threshold': iou_threshold,
                    'prompt_mode': prompt_mode,
                    'downscale_masks': downscale_enabled,
                    'downscale_ratio': mask_scale_ratio,
                    'render_viz': render_viz,
                }
            }
        )

    def _run_report_stage(self) -> None:
        stage_cfg = self._stage_cfg('report')
        if not stage_cfg.get('enabled', True):
            return

        report_name = stage_cfg.get('name', 'report.md')
        max_width = int(stage_cfg.get('max_width', 960))
        report_gpu = resolve_stage_gpu(stage_cfg, self.default_stage_gpu)
        run_dir = self._ensure_run_dir()

        print('Stage Report: 生成 Markdown 紀錄')
        with StageRecorder(self.summary, 'report', report_gpu), using_gpu(report_gpu):
            build_report(run_dir, report_name=report_name, max_preview_width=max_width)
            report_summary = self._stage_summary('report').setdefault('params', {})
            report_summary['max_width'] = max_width
            report_summary['report_name'] = report_name

        record_timings_flag = bool(stage_cfg.get('record_timings'))
        timing_output_name = stage_cfg.get('timing_output')
        if record_timings_flag or timing_output_name:
            output_name = timing_output_name or 'stage_timings.json'
            timings_path = run_dir / output_name
            export_stage_timings(self.summary, timings_path)
            artifacts_entry = self._stage_summary('report').setdefault('artifacts', {})
            artifacts_entry['timings'] = str(timings_path)

    def _finalize(self) -> None:
        run_dir = self._ensure_run_dir()
        manifest = self._ensure_manifest()

        self.summary['generated_at'] = _now_local_iso()
        self.summary['run_dir'] = str(run_dir)

        if self.output_layout_mode == 'scene_level':
            aggregated_payload = apply_scene_level_layout(run_dir, self.summary, manifest)
            if aggregated_payload is not None:
                self.summary['scene_level_summary'] = aggregated_payload

        with (run_dir / 'workflow_summary.json').open('w') as f:
            json.dump(self.summary, f, indent=2)

        append_run_history(self.summary, manifest)

        print(f'Workflow finished. 輸出路徑：{run_dir}')


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
    context = SceneContext(
        config=config,
        experiment_cfg=experiment_cfg,
        stages_cfg=stages_cfg,
        default_stage_gpu=default_stage_gpu,
        data_path=data_path,
        output_root=output_root,
        config_path=config_path,
        parent_meta=parent_meta,
    )
    return SceneWorkflow(context).run()


def execute_workflow(
    config: Dict[str, Any],
    *,
    override_output: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    experiment_cfg = config.get('experiment', {})
    if not isinstance(experiment_cfg, dict):
        raise WorkflowConfigError('`experiment` section must be a mapping')

    stages_cfg = config.get('stages', {})
    if not isinstance(stages_cfg, dict):
        raise WorkflowConfigError('`stages` section must be a mapping')
    default_stage_gpu = stages_cfg.get('gpu')

    scenes_cfg = experiment_cfg.get('scenes')
    scene_start_cfg = experiment_cfg.get('scene_start')
    scene_end_cfg = experiment_cfg.get('scene_end')
    scenes_list: Optional[List[str]] = None
    dataset_root_raw = experiment_cfg.get('dataset_root')
    if scenes_cfg is not None or scene_start_cfg is not None or scene_end_cfg is not None:
        if not dataset_root_raw:
            raise WorkflowConfigError('experiment.dataset_root is required when selecting scenes')
        dataset_root = Path(dataset_root_raw).expanduser()
        scenes_list = normalize_scene_list(
            scenes_cfg,
            dataset_root,
            scene_start=scene_start_cfg,
            scene_end=scene_end_cfg,
        )
    else:
        dataset_root = Path(dataset_root_raw).expanduser() if dataset_root_raw else None

    if scenes_list:
        output_root_base_raw = override_output or experiment_cfg.get('output_root')
        output_root_base_resolved = expand_output_path_template(output_root_base_raw, experiment_cfg)
        output_root_base = Path(output_root_base_resolved).expanduser()

        aggregate_output = bool(experiment_cfg.get('aggregate_output'))
        run_timestamp = experiment_cfg.get('run_timestamp')
        if aggregate_output:
            experiment_stamp = (
                str(run_timestamp)
                if run_timestamp
                else _now_local_stamp()
            )
            experiment_root = output_root_base / experiment_stamp
        else:
            experiment_root = output_root_base
            experiment_stamp = str(run_timestamp) if run_timestamp else None

        experiment_root.mkdir(parents=True, exist_ok=True)

        parent_meta = {
            'name': experiment_cfg.get('name'),
            'scenes': scenes_list,
            'experiment_root': str(experiment_root),
            'timestamp': experiment_stamp,
            'aggregate_output': aggregate_output,
            'scene_start': scene_start_cfg,
            'scene_end': scene_end_cfg,
        }

        summaries: List[Dict[str, Any]] = []
        experiment_scene_records: List[Dict[str, Any]] = []

        for index, scene_name in enumerate(scenes_list):
            scene_data_path = dataset_root / scene_name / 'outputs' / 'color'
            if not scene_data_path.exists():
                raise WorkflowConfigError(f'data path not found for scene {scene_name}: {scene_data_path}')
            if aggregate_output:
                scene_output_root = experiment_root / scene_name
            else:
                scene_output_root = output_root_base / scene_name
            scene_output_root.mkdir(parents=True, exist_ok=True)

            scene_experiment_cfg = dict(experiment_cfg)
            scene_experiment_cfg['name'] = experiment_cfg.get('name')
            scene_experiment_cfg['data_path'] = str(scene_data_path)
            scene_experiment_cfg['output_root'] = str(scene_output_root)
            scene_experiment_cfg['aggregate_output'] = aggregate_output
            scene_experiment_cfg['run_timestamp'] = experiment_stamp
            scene_experiment_cfg.pop('scenes', None)

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

            summary_json_path = summary.get('summary_json')
            summary_json_rel = None
            if summary_json_path:
                try:
                    summary_json_rel = str(Path(summary_json_path).relative_to(experiment_root))
                except ValueError:
                    summary_json_rel = summary_json_path

            run_dir_path = summary.get('run_dir')
            run_dir_rel = None
            if run_dir_path:
                try:
                    run_dir_rel = str(Path(run_dir_path).relative_to(experiment_root))
                except ValueError:
                    run_dir_rel = run_dir_path

            scene_record = {
                'scene': scene_name,
                'run_dir': run_dir_rel,
                'summary_json': summary_json_rel,
                'levels': summary.get('scene_level_summary', {}).get('levels')
                if summary.get('scene_level_summary')
                else None,
            }
            experiment_scene_records.append(scene_record)

        if aggregate_output:
            experiment_summary = {
                'generated_at': _now_local_iso(),
                'experiment': {
                    'name': experiment_cfg.get('name'),
                    'timestamp': experiment_stamp,
                    'scenes': scenes_list,
                    'output_root': str(experiment_root),
                },
                'runs': experiment_scene_records,
            }
            experiment_summary_path = experiment_root / 'experiment_summary.json'
            try:
                with experiment_summary_path.open('w') as f:
                    json.dump(experiment_summary, f, indent=2)
            except OSError:
                pass

        return summaries

    data_path_raw = experiment_cfg.get('data_path')
    if not data_path_raw:
        raise WorkflowConfigError('experiment.data_path is required')
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
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the parsed config (and override hints) without executing stages',
    )
    parser.add_argument(
        '--no-oom-watch',
        action='store_true',
        help='Disable automatic OOM monitoring (enabled by default)',
    )
    parser.add_argument(
        '--oom-watch-poll',
        type=float,
        default=5.0,
        help='Polling interval in seconds for the background OOM watcher',
    )
    parser.add_argument(
        '--oom-log',
        type=Path,
        default=Path('logs/oom_monitor.log'),
        help='Log file used to store OOM watcher events',
    )
    parser.add_argument(
        '--oom-email',
        default='kunghsiangyu@gapp.nthu.edu.tw',
        help='Email for OOM notifications (requires SMTP environment configuration)',
    )
    parser.add_argument(
        '--notify-email',
        default='kunghsiangyu@gapp.nthu.edu.tw',
        help='Email for completion notifications (defaults to --oom-email; requires SMTP environment configuration)',
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_yaml(config_path)
    except WorkflowError as exc:
        print(f'Workflow configuration error: {exc}', file=sys.stderr)
        return 1

    if args.dry_run:
        preview = config
        if args.override_output:
            preview = dict(config)
            experiment_cfg = dict(preview.get('experiment', {}))
            experiment_cfg['output_root'] = args.override_output
            preview['experiment'] = experiment_cfg
        print('# My3DIS workflow dry run')
        print(f'# config: {config_path}')
        if args.override_output:
            print(f'# override_output: {args.override_output}')
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    monitor_enabled = not args.no_oom_watch
    oom_log_path = Path(args.oom_log).expanduser()
    oom_email = args.oom_email or None
    completion_email = args.notify_email if args.notify_email is not None else args.oom_email
    completion_email = completion_email or None
    completion_notifier = create_notifier(completion_email) if completion_email else None

    monitor_context = (
        memory_watch_context(
            poll_interval=float(max(0.5, args.oom_watch_poll)),
            log_path=oom_log_path,
            email=oom_email,
        )
        if monitor_enabled
        else nullcontext([])
    )

    if completion_email:
        print(f'Completion notifications will target {completion_email}.', file=sys.stderr)

    workflow_started_at = _now_local_iso()
    started_monotonic = time.perf_counter()
    run_summaries: Optional[List[Dict[str, Any]]] = None

    try:
        with monitor_context as watched_paths:
            if monitor_enabled:
                if not watched_paths:
                    print(
                        'OOM monitor: no accessible memory.events files found; running without background watcher.',
                        file=sys.stderr,
                )
            else:
                print(
                    (
                        f'OOM monitor active on {len(watched_paths)} cgroup files; '
                        f'logging to {oom_log_path}.'
                    ),
                    file=sys.stderr,
                )
                if oom_email:
                    print(f'OOM notifications will target {oom_email}.', file=sys.stderr)

            run_summaries = execute_workflow(
                config,
                override_output=args.override_output,
                config_path=config_path,
            )

    except BaseException as exc:
        finished_at = _now_local_iso()
        duration_seconds = time.perf_counter() - started_monotonic
        if completion_notifier:
            subject, body = _build_completion_email_payload(
                'failure',
                config_path=config_path,
                started_at=workflow_started_at,
                finished_at=finished_at,
                duration_seconds=duration_seconds,
                run_summaries=run_summaries,
                error_message=str(exc),
                traceback_text=traceback.format_exc(),
            )
            _send_completion_notification(completion_notifier, subject, body)
        if isinstance(exc, WorkflowError):
            print(f'Workflow error: {exc}', file=sys.stderr)
            return 1
        raise
    else:
        finished_at = _now_local_iso()
        duration_seconds = time.perf_counter() - started_monotonic
        if completion_notifier:
            subject, body = _build_completion_email_payload(
                'success',
                config_path=config_path,
                started_at=workflow_started_at,
                finished_at=finished_at,
                duration_seconds=duration_seconds,
                run_summaries=run_summaries,
            )
            _send_completion_notification(completion_notifier, subject, body)

    return 0


if __name__ == '__main__':
    sys.exit(main())
