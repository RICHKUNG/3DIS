"""執行摘要與輸出相關的工具。"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - platform specific
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

try:  # pragma: no cover - platform specific
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil not installed
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch not installed
    torch = None  # type: ignore

from my3dis.common_utils import format_duration

from .utils import normalise_gpu_spec, now_local_iso, serialise_gpu_spec


LOGGER = logging.getLogger(__name__)
_BYTES_PER_MIB = 1024 * 1024


def _bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / float(_BYTES_PER_MIB), 2)


def _normalise_gpu_indices(spec: Any) -> List[int]:
    return normalise_gpu_spec(spec)


class StageResourceMonitor:
    """Background sampler that captures peak CPU/GPU usage for a stage."""

    def __init__(self, stage_name: str, gpu_spec: Any, poll_interval: float = 0.5) -> None:
        self._stage_name = stage_name
        self._gpu_indices = _normalise_gpu_indices(gpu_spec)
        self._poll_interval = float(poll_interval)
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._cpu_peak_bytes: Optional[int] = None
        self._cpu_samples: int = 0
        self._cpu_process: Optional[Any] = None
        self._gpu_metrics: Dict[int, Dict[str, Optional[int]]] = {}
        self._torch_devices: List[Any] = []

    def start(self) -> None:
        self._setup_cpu_monitor()
        self._setup_gpu_monitor()
        if self._cpu_process is not None:
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._poll_cpu_usage,
                name=f'stage-resource-{self._stage_name}',
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._update_cpu_peak()  # final sample

        gpu_summary = self._finalise_gpu_metrics()
        cpu_summary = self._build_cpu_summary()
        payload: Dict[str, Any] = {}
        if cpu_summary is not None:
            payload['cpu'] = cpu_summary
        if gpu_summary is not None:
            payload['gpu'] = gpu_summary
        if payload:
            payload['poll_interval_sec'] = self._poll_interval
        return payload

    def _setup_cpu_monitor(self) -> None:
        if psutil is None:
            LOGGER.debug('psutil not available; CPU memory sampling disabled')
            return
        try:
            self._cpu_process = psutil.Process(os.getpid())
        except (psutil.Error, OSError) as exc:  # type: ignore[attr-defined]
            LOGGER.debug('Unable to initialise psutil.Process: %s', exc)
            self._cpu_process = None
            return
        self._cpu_peak_bytes = 0
        self._cpu_samples = 0
        # Take an initial measurement so we do not miss instantaneous spikes.
        self._update_cpu_peak()

    def _collect_process_rss(self) -> Optional[int]:
        if self._cpu_process is None or psutil is None:
            return None
        total_rss = 0
        try:
            total_rss += int(self._cpu_process.memory_info().rss)
        except (psutil.Error, OSError):  # type: ignore[attr-defined]
            return None
        try:
            children = self._cpu_process.children(recursive=True)
        except (psutil.Error, OSError):  # type: ignore[attr-defined]
            children = []
        for child in children:
            try:
                total_rss += int(child.memory_info().rss)
            except (psutil.Error, OSError):  # type: ignore[attr-defined]
                continue
        return total_rss

    def _update_cpu_peak(self) -> None:
        rss = self._collect_process_rss()
        if rss is None:
            return
        self._cpu_samples += 1
        if self._cpu_peak_bytes is None or rss > self._cpu_peak_bytes:
            self._cpu_peak_bytes = rss

    def _poll_cpu_usage(self) -> None:
        if self._stop_event is None:
            return
        while not self._stop_event.is_set():
            self._update_cpu_peak()
            self._stop_event.wait(self._poll_interval)

    def _setup_gpu_monitor(self) -> None:
        if torch is None or not torch.cuda.is_available():  # type: ignore[attr-defined]
            return
        try:
            device_count = torch.cuda.device_count()  # type: ignore[attr-defined]
        except Exception:
            device_count = 0
        if device_count <= 0:
            return
        requested = [idx for idx in self._gpu_indices if idx >= 0]
        if not requested:
            requested = list(range(device_count))
        elif any(idx >= device_count for idx in requested):
            requested = list(range(min(len(requested), device_count)))
        devices: List[Any] = []
        for idx in requested:
            try:
                device = torch.device('cuda', idx)  # type: ignore[attr-defined]
            except Exception:
                continue
            devices.append(device)
        if not devices:
            return
        self._gpu_indices = [int(device.index) for device in devices]
        self._torch_devices = devices
        for device in devices:
            try:
                torch.cuda.reset_peak_memory_stats(device)  # type: ignore[attr-defined]
            except Exception:
                LOGGER.debug('Failed to reset CUDA peak stats for %s', device, exc_info=True)
            self._gpu_metrics[int(device.index)] = {
                'initial_reserved_bytes': self._safe_cuda_call(torch.cuda.memory_reserved, device),  # type: ignore[attr-defined]
                'initial_allocated_bytes': self._safe_cuda_call(torch.cuda.memory_allocated, device),  # type: ignore[attr-defined]
                'peak_reserved_bytes': None,
                'peak_allocated_bytes': None,
            }

    @staticmethod
    def _safe_cuda_call(fn, device) -> Optional[int]:
        try:
            return int(fn(device))
        except Exception:
            return None

    def _finalise_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        if not self._torch_devices:
            return None
        devices_summary: Dict[str, Dict[str, Optional[float]]] = {}
        total_reserved_bytes = 0
        total_allocated_bytes = 0
        for device in self._torch_devices:
            try:
                torch.cuda.synchronize(device)  # type: ignore[attr-defined]
            except Exception:
                pass
            idx = int(device.index)
            peak_reserved = self._safe_cuda_call(torch.cuda.max_memory_reserved, device)  # type: ignore[attr-defined]
            peak_allocated = self._safe_cuda_call(torch.cuda.max_memory_allocated, device)  # type: ignore[attr-defined]
            if peak_reserved is not None:
                total_reserved_bytes += peak_reserved
            if peak_allocated is not None:
                total_allocated_bytes += peak_allocated
            devices_summary[f'cuda:{idx}'] = {
                'peak_reserved_bytes': peak_reserved,
                'peak_reserved_mib': _bytes_to_mib(peak_reserved),
                'peak_allocated_bytes': peak_allocated,
                'peak_allocated_mib': _bytes_to_mib(peak_allocated),
            }
            metrics_entry = self._gpu_metrics.get(idx)
            if metrics_entry is not None:
                metrics_entry['peak_reserved_bytes'] = peak_reserved
                metrics_entry['peak_allocated_bytes'] = peak_allocated
        return {
            'devices': devices_summary,
            'total_reserved_bytes': total_reserved_bytes if total_reserved_bytes > 0 else None,
            'total_reserved_mib': _bytes_to_mib(total_reserved_bytes) if total_reserved_bytes > 0 else None,
            'total_allocated_bytes': total_allocated_bytes if total_allocated_bytes > 0 else None,
            'total_allocated_mib': _bytes_to_mib(total_allocated_bytes) if total_allocated_bytes > 0 else None,
        }

    def _build_cpu_summary(self) -> Optional[Dict[str, Any]]:
        if self._cpu_peak_bytes is None:
            return None
        return {
            'peak_rss_bytes': self._cpu_peak_bytes,
            'peak_rss_mib': _bytes_to_mib(self._cpu_peak_bytes),
            'samples': self._cpu_samples,
        }


def _gather_git_snapshot() -> Optional[Dict[str, Any]]:
    git_path = shutil.which('git')
    if not git_path:
        return None
    try:
        repo_root = Path(__file__).resolve().parents[3]
    except IndexError:
        return None
    env = os.environ.copy()
    env.setdefault('LC_ALL', 'C')
    env.setdefault('LANG', 'C')
    try:
        commit = subprocess.check_output(
            [git_path, 'rev-parse', 'HEAD'],
            cwd=str(repo_root),
            env=env,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    snapshot: Dict[str, Any] = {'commit': commit}
    try:
        branch = subprocess.check_output(
            [git_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=str(repo_root),
            env=env,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if branch:
            snapshot['branch'] = branch
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            [git_path, 'status', '--short'],
            cwd=str(repo_root),
            env=env,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if status:
            snapshot['dirty'] = True
        else:
            snapshot['dirty'] = False
    except Exception:
        pass
    return snapshot


def collect_environment_snapshot() -> Dict[str, Any]:
    """收集目前環境資訊（Python/torch/CUDA 等）供追蹤使用。"""

    snapshot: Dict[str, Any] = {
        'generated_at': now_local_iso(),
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'executable': sys.executable,
        },
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
    }

    env_hints = {}
    for key in ['CUDA_VISIBLE_DEVICES', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV']:
        value = os.environ.get(key)
        if value:
            env_hints[key] = value
    if env_hints:
        snapshot['environment'] = env_hints

    if torch is not None:
        torch_info: Dict[str, Any] = {
            'version': torch.__version__,  # type: ignore[attr-defined]
            'cuda_available': bool(torch.cuda.is_available()),  # type: ignore[attr-defined]
        }
        cuda_version = getattr(getattr(torch, 'version', None), 'cuda', None)
        if cuda_version:
            torch_info['cuda_version'] = cuda_version
        try:
            cudnn_version = torch.backends.cudnn.version()  # type: ignore[attr-defined]
            if cudnn_version:
                torch_info['cudnn_version'] = cudnn_version
        except Exception:
            pass
        if torch_info['cuda_available']:
            try:
                device_count = torch.cuda.device_count()  # type: ignore[attr-defined]
            except Exception:
                device_count = 0
            torch_info['device_count'] = device_count
            devices: List[Dict[str, Any]] = []
            for idx in range(device_count):
                device_entry: Dict[str, Any] = {'index': idx}
                try:
                    device_entry['name'] = torch.cuda.get_device_name(idx)  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    props = torch.cuda.get_device_properties(idx)  # type: ignore[attr-defined]
                    device_entry['total_memory_mib'] = round(props.total_memory / float(_BYTES_PER_MIB), 2)
                    device_entry['multi_processor_count'] = props.multi_processor_count
                    device_entry['capability'] = f"{props.major}.{props.minor}"
                except Exception:
                    pass
                devices.append(device_entry)
            if devices:
                torch_info['devices'] = devices
        snapshot['torch'] = torch_info

    try:
        import numpy as _np  # type: ignore

        snapshot['numpy'] = {'version': _np.__version__}
    except ImportError:
        pass

    if psutil is not None:
        snapshot['psutil'] = {'version': getattr(psutil, '__version__', None)}

    git_snapshot = _gather_git_snapshot()
    if git_snapshot:
        snapshot['git'] = git_snapshot

    return snapshot


@dataclass
class StageRecorder:
    """記錄每個 workflow stage 的執行時間與 GPU 資訊。"""

    summary: Dict[str, Any]
    name: str
    gpu: Optional[Any]

    def __post_init__(self) -> None:
        self._raw_gpu = self.gpu
        self.gpu = serialise_gpu_spec(self.gpu)
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.duration: Optional[float] = None
        self._resource_monitor: Optional[StageResourceMonitor] = None

    def __enter__(self) -> 'StageRecorder':
        self.start = time.perf_counter()
        now_iso = now_local_iso()
        self.summary.setdefault('order', []).append(self.name)
        self.summary.setdefault('stages', {})[self.name] = {
            'started_at': now_iso,
            'gpu': self.gpu,
        }
        try:
            monitor = StageResourceMonitor(self.name, self._raw_gpu)
            monitor.start()
            self._resource_monitor = monitor
        except Exception:
            LOGGER.debug('StageResourceMonitor initialisation failed for %s', self.name, exc_info=True)
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
        if self._resource_monitor is not None:
            try:
                resources = self._resource_monitor.stop()
            except Exception:
                LOGGER.debug('StageResourceMonitor shutdown failed for %s', self.name, exc_info=True)
                resources = {}
            if resources:
                stage_entry['resources'] = resources
            self._resource_monitor = None


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
        record: Dict[str, Any] = {
            'stage': name,
            'duration_sec': duration,
            'duration_text': meta.get('duration_text'),
            'started_at': meta.get('started_at'),
            'ended_at': meta.get('ended_at'),
            'gpu': meta.get('gpu'),
        }
        resources = meta.get('resources') if isinstance(meta.get('resources'), dict) else {}
        cpu_meta = resources.get('cpu') if isinstance(resources.get('cpu'), dict) else {}
        gpu_meta = resources.get('gpu') if isinstance(resources.get('gpu'), dict) else {}

        cpu_peak_bytes = cpu_meta.get('peak_rss_bytes')
        if cpu_peak_bytes is not None:
            record['cpu_peak_rss_bytes'] = cpu_peak_bytes
            record['cpu_peak_rss_mib'] = _bytes_to_mib(cpu_peak_bytes)
            if cpu_meta.get('samples') is not None:
                record['cpu_samples'] = int(cpu_meta['samples'])

        gpu_reserved_bytes = gpu_meta.get('total_reserved_bytes')
        if gpu_reserved_bytes is not None:
            record['gpu_peak_reserved_bytes'] = gpu_reserved_bytes
            record['gpu_peak_reserved_mib'] = _bytes_to_mib(gpu_reserved_bytes)
        gpu_allocated_bytes = gpu_meta.get('total_allocated_bytes')
        if gpu_allocated_bytes is not None:
            record['gpu_peak_allocated_bytes'] = gpu_allocated_bytes
            record['gpu_peak_allocated_mib'] = _bytes_to_mib(gpu_allocated_bytes)
        if gpu_meta.get('devices'):
            record['gpu_devices'] = gpu_meta['devices']

        poll_interval = resources.get('poll_interval_sec')
        if poll_interval is not None:
            record['resource_poll_interval_sec'] = float(poll_interval)

        records.append(record)

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
    comparison_manifest = manifest.get('comparison_summary')
    if not isinstance(comparison_manifest, dict):
        comparison_manifest = {}

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

        manifest_key = f'level_{lvl}'
        manifest_entry = tracking_artifacts.setdefault(manifest_key, {})
        if object_rel:
            manifest_entry['object_segments'] = object_rel
        if video_rel:
            manifest_entry['video_segments'] = video_rel

        compare_src_dir = level_dir / 'viz' / 'compare'
        if compare_src_dir.exists():
            compare_dest_dir = level_dir / 'compare'
            for item in compare_src_dir.iterdir():
                _move_file(item, compare_dest_dir / item.name)
            try:
                compare_src_dir.rmdir()
            except OSError:
                pass

        def _rewrite_compare_rel(path_value: Optional[str]) -> Optional[str]:
            if not path_value:
                return None
            rel_path = Path(str(path_value))
            parts = rel_path.parts
            if len(parts) >= 3 and parts[1] == 'viz' and parts[2] == 'compare':
                rel_path = Path(parts[0]) / 'compare' / Path(*parts[3:])
            return str(rel_path)

        comparison_summary_rel = _rewrite_compare_rel(manifest_entry.get('comparison_summary'))
        comparison_images_rel_raw = manifest_entry.get('comparison_images')
        comparison_images_rel: Optional[List[str]] = None
        if isinstance(comparison_images_rel_raw, list):
            comparison_images_rel = []
            for rel in comparison_images_rel_raw:
                rewritten = _rewrite_compare_rel(rel)
                if rewritten:
                    comparison_images_rel.append(rewritten)
        comparison_fallback_rel = _rewrite_compare_rel(manifest_entry.get('comparison_fallback'))

        if comparison_summary_rel:
            manifest_entry['comparison_summary'] = comparison_summary_rel
        elif 'comparison_summary' in manifest_entry:
            manifest_entry.pop('comparison_summary')
        if comparison_images_rel is not None:
            manifest_entry['comparison_images'] = comparison_images_rel
        if comparison_fallback_rel:
            manifest_entry['comparison_fallback'] = comparison_fallback_rel

        level_entry: Dict[str, Any] = {
            'object_segments': manifest_entry.get('object_segments'),
            'video_segments': manifest_entry.get('video_segments'),
            'report_images': report_rel_paths,
        }
        if comparison_summary_rel:
            level_entry['comparison_summary'] = comparison_summary_rel
        if comparison_images_rel:
            level_entry['comparison_images'] = comparison_images_rel
        if comparison_fallback_rel:
            level_entry['comparison_fallback'] = comparison_fallback_rel
        aggregated_levels[str(lvl)] = level_entry

        comp_entry = comparison_manifest.get(manifest_key)
        if isinstance(comp_entry, dict):
            summary_rel = _rewrite_compare_rel(comp_entry.get('summary'))
            if summary_rel:
                comp_entry['summary'] = summary_rel
            elif 'summary' in comp_entry:
                comp_entry.pop('summary')
            if isinstance(comp_entry.get('rendered_images'), list):
                comp_entry['rendered_images'] = [
                    rel for rel in (_rewrite_compare_rel(rel) for rel in comp_entry['rendered_images']) if rel
                ]
            fallback_rel = _rewrite_compare_rel(comp_entry.get('fallback'))
            if fallback_rel:
                comp_entry['fallback'] = fallback_rel
            elif 'fallback' in comp_entry:
                comp_entry.pop('fallback')

        # Clean up visualization directory when present (optional artifacts)
        viz_dir = level_dir / 'viz'
        if viz_dir.exists():
            try:
                shutil.rmtree(viz_dir)
            except OSError:
                pass

    manifest['tracking_artifacts'] = tracking_artifacts
    if comparison_manifest:
        manifest['comparison_summary'] = comparison_manifest
    else:
        manifest.pop('comparison_summary', None)
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
    'collect_environment_snapshot',
]
