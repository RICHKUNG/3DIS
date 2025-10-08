"""執行摘要與輸出相關的工具。"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, List, Optional

try:  # pragma: no cover - platform specific
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

try:  # pragma: no cover - platform specific
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None  # type: ignore

from my3dis.common_utils import format_duration

from .utils import now_local_iso


@dataclass
class StageRecorder:
    """記錄每個 workflow stage 的執行時間與 GPU 資訊。"""

    summary: Dict[str, Any]
    name: str
    gpu: Optional[Any]

    def __post_init__(self) -> None:
        self.gpu = None if self.gpu is None else str(self.gpu)
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self) -> 'StageRecorder':
        self.start = time.perf_counter()
        now_iso = now_local_iso()
        self.summary.setdefault('order', []).append(self.name)
        self.summary.setdefault('stages', {})[self.name] = {
            'started_at': now_iso,
            'gpu': self.gpu,
        }
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end_time = time.perf_counter()
        duration = end_time - (self.start or end_time)
        now_iso = now_local_iso()
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
        'generated_at': now_local_iso(),
        'total_duration_sec': total_duration,
        'total_duration_text': format_duration(total_duration),
        'stages': records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def update_summary_config(summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    summary['config_snapshot'] = config


def load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open('r') as handle:
            return json.load(handle)
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
        'timestamp': summary.get('generated_at') or now_local_iso(),
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
        from .errors import WorkflowRuntimeError

        raise WorkflowRuntimeError(f'Failed to move {src} → {dst}: {exc}') from exc
    return dst


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open('r') as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def apply_scene_level_layout(
    run_dir: Path,
    summary: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """整理場景層級輸出，並產生 summary.json。"""
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
        with manifest_path.open('w') as handle:
            json.dump(manifest, handle, indent=2)
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
        with summary_path.open('w') as handle:
            json.dump(summary_payload, handle, indent=2)
    except OSError:
        pass

    summary.setdefault('artifacts', {})['scene_layout'] = aggregated_levels
    summary['summary_json'] = str(summary_path)

    return summary_payload


__all__ = [
    'StageRecorder',
    'export_stage_timings',
    'update_summary_config',
    'load_manifest',
    'append_run_history',
    'apply_scene_level_layout',
]
