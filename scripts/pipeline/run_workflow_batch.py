#!/usr/bin/env python3
"""Run multiple workflow configs sequentially for multi-scene experiments."""

from __future__ import annotations

if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


import argparse
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from oom_monitor import memory_watch_context

def discover_configs(config_dir: Path) -> List[Path]:
    if not config_dir.exists():
        return []
    return sorted(p for p in config_dir.glob('*.yaml') if p.is_file())


def filter_by_scene(configs: List[Path], scenes: List[str]) -> List[Path]:
    if not scenes:
        return configs

    normalized = {scene if scene.endswith('.yaml') else f'{scene}.yaml' for scene in scenes}
    normalized_stems = {scene for scene in scenes if not scene.endswith('.yaml')}

    selected: List[Path] = []
    for path in configs:
        if path.name in normalized or path.stem in normalized_stems:
            selected.append(path)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description='Batch runner for multi-scene workflows')
    parser.add_argument(
        '--config-dir',
        default='configs/scenes',
        help='Directory containing per-scene YAML configs (default: configs/scenes)',
    )
    parser.add_argument(
        '--configs',
        nargs='*',
        help='Explicit list of config paths to run (overrides --config-dir discovery)',
    )
    parser.add_argument(
        '--scenes',
        nargs='*',
        help='Filter discovered configs by scene name (e.g., scene_00065_00)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Process at most N configs after filtering',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List the configs that would run without executing them',
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Abort the batch as soon as a run fails',
    )
    parser.add_argument(
        '--batch-log-dir',
        default='logs/batch',
        help='Directory to store batch summary JSON files',
    )
    parser.add_argument(
        '--override-output',
        help='Override output root for all configs (passes --override-output to each run)',
    )
    parser.add_argument(
        '--no-oom-watch',
        action='store_true',
        help='Disable automatic OOM monitoring (enabled by default)',
    )
    parser.add_argument(
        '--oom-watch-poll',
        type=float,
        default=2.0,
        help='Polling interval in seconds for the background OOM watcher',
    )
    parser.add_argument(
        '--oom-log',
        type=Path,
        default=Path('logs/oom_monitor.log'),
        help='Log file used to store OOM watcher events',
    )

    args = parser.parse_args()

    monitor_enabled = not args.no_oom_watch
    oom_log_path = Path(args.oom_log).expanduser()
    monitor_context = (
        memory_watch_context(
            poll_interval=float(max(0.5, args.oom_watch_poll)),
            log_path=oom_log_path,
        )
        if monitor_enabled
        else nullcontext([])
    )

    with monitor_context as watched_paths:
        if monitor_enabled:
            if not watched_paths:
                print(
                    'OOM monitor: no accessible memory.events files found; running batch without background watcher.',
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

        config_paths: List[Path]
        if args.configs:
            config_paths = [Path(cfg).expanduser().resolve() for cfg in args.configs]
        else:
            config_dir = Path(args.config_dir).expanduser().resolve()
            config_paths = discover_configs(config_dir)

        if args.scenes:
            config_paths = filter_by_scene(config_paths, args.scenes)

        if args.limit is not None and args.limit >= 0:
            config_paths = config_paths[: args.limit]

        if not config_paths:
            print('No configs selected for execution.', file=sys.stderr)
            return 1

        if args.dry_run:
            print('Dry-run: would execute the following configs in order:')
            for path in config_paths:
                print(f'  - {path}')
            return 0

        from my3dis.run_workflow import execute_workflow, load_yaml  # noqa: WPS433 (local import)

        batch_records: List[Dict[str, Any]] = []
        batch_started = datetime.utcnow().isoformat(timespec='seconds')

        for idx, config_path in enumerate(config_paths, start=1):
            print(f'[{idx}/{len(config_paths)}] Running {config_path}')
            try:
                config = load_yaml(config_path)
            except Exception as exc:  # pragma: no cover - config parsing error surface
                record = {
                    'config_path': str(config_path),
                    'status': 'load_failed',
                    'error': str(exc),
                }
                batch_records.append(record)
                print(f'  Failed to load config: {exc}', file=sys.stderr)
                if args.stop_on_error:
                    break
                continue

            start_time = time.perf_counter()
            start_iso = datetime.utcnow().isoformat(timespec='seconds')

            try:
                summaries = execute_workflow(
                    config,
                    override_output=args.override_output,
                    config_path=config_path,
                )
            except SystemExit as exc:
                duration = time.perf_counter() - start_time
                record = {
                    'config_path': str(config_path),
                    'status': 'failed',
                    'error': str(exc),
                    'duration_sec': duration,
                    'started_at': start_iso,
                    'ended_at': datetime.utcnow().isoformat(timespec='seconds'),
                }
                print(f'  Workflow exited with error: {exc}', file=sys.stderr)
                batch_records.append(record)
                if args.stop_on_error:
                    break
                continue
            except Exception as exc:  # pragma: no cover - unexpected failure
                duration = time.perf_counter() - start_time
                record = {
                    'config_path': str(config_path),
                    'status': 'failed',
                    'error': repr(exc),
                    'duration_sec': duration,
                    'started_at': start_iso,
                    'ended_at': datetime.utcnow().isoformat(timespec='seconds'),
                }
                print(f'  Unexpected error: {exc}', file=sys.stderr)
                batch_records.append(record)
                if args.stop_on_error:
                    break
                continue
            else:
                if not isinstance(summaries, list):
                    summaries = [summaries]

                def _summary_duration(payload: Dict[str, Any]) -> float:
                    total = 0.0
                    stages_meta = payload.get('stages', {}) if isinstance(payload.get('stages'), dict) else {}
                    for meta in stages_meta.values():
                        try:
                            total += float(meta.get('duration_sec') or 0.0)
                        except (TypeError, ValueError):
                            continue
                    return total

                end_perf = time.perf_counter()
                end_iso = datetime.utcnow().isoformat(timespec='seconds')
                for summary in summaries:
                    experiment = summary.get('experiment', {}) if isinstance(summary.get('experiment'), dict) else {}
                    summary_duration = _summary_duration(summary)
                    if summary_duration <= 0:
                        summary_duration = end_perf - start_time
                    record = {
                        'config_path': str(config_path),
                        'status': 'completed',
                        'started_at': start_iso,
                        'ended_at': end_iso,
                        'duration_sec': summary_duration,
                        'summary_path': str(Path(summary['run_dir']) / 'workflow_summary.json') if summary.get('run_dir') else '',
                        'run_dir': summary.get('run_dir'),
                        'scene': experiment.get('scene'),
                        'levels': experiment.get('levels'),
                        'parent_experiment': experiment.get('parent_experiment'),
                    }
                    batch_records.append(record)

    log_dir = Path(args.batch_log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'batch_{timestamp}.json'
    payload = {
        'generated_at': datetime.utcnow().isoformat(timespec='seconds'),
        'batch_started_at': batch_started,
        'config_dir': str(Path(args.config_dir).expanduser().resolve()),
        'override_output': args.override_output,
        'stop_on_error': bool(args.stop_on_error),
        'runs': batch_records,
    }
    with log_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'Batch summary written to {log_path}')

    failed = [r for r in batch_records if r.get('status') != 'completed']
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
