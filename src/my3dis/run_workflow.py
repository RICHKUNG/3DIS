#!/usr/bin/env python3
"""命令列入口，負責調用 My3DIS 工作流程模組。"""
from __future__ import annotations

if __package__ is None or __package__ == '':
    import sys as _sys
    import pathlib as _pathlib

    _project_root = _pathlib.Path(__file__).resolve().parents[2]
    _src_path = _project_root / 'src'
    if str(_src_path) not in _sys.path:
        _sys.path.insert(0, str(_src_path))
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))

import argparse
import json
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

from oom_monitor import memory_watch_context

from my3dis.workflow import (
    WorkflowConfigError,
    WorkflowError,
    WorkflowRuntimeError,
    StageRecorder,
    SceneContext,
    SceneWorkflow,
    append_run_history,
    apply_scene_level_layout,
    build_completion_log_entry,
    derive_scene_metadata,
    discover_scene_names,
    execute_workflow,
    export_stage_timings,
    expand_output_path_template,
    load_manifest,
    load_yaml,
    log_completion_event,
    normalize_scene_list,
    now_local_iso,
    now_local_stamp,
    resolve_levels,
    resolve_stage_gpu,
    stage_frames_string,
    update_summary_config,
    using_gpu,
)

# 與舊版介面相容的別名
_now_local_iso = now_local_iso
_now_local_stamp = now_local_stamp
_build_completion_log_entry = build_completion_log_entry
_log_completion_event = log_completion_event

__all__ = [
    'WorkflowError',
    'WorkflowConfigError',
    'WorkflowRuntimeError',
    'load_yaml',
    'using_gpu',
    'now_local_iso',
    'now_local_stamp',
    'build_completion_log_entry',
    'log_completion_event',
    '_now_local_iso',
    '_now_local_stamp',
    '_build_completion_log_entry',
    '_log_completion_event',
    'StageRecorder',
    'export_stage_timings',
    'expand_output_path_template',
    'discover_scene_names',
    'normalize_scene_list',
    'resolve_levels',
    'stage_frames_string',
    'resolve_stage_gpu',
    'derive_scene_metadata',
    'update_summary_config',
    'load_manifest',
    'append_run_history',
    'apply_scene_level_layout',
    'SceneContext',
    'SceneWorkflow',
    'execute_workflow',
    'main',
]


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

    monitor_context = (
        memory_watch_context(
            poll_interval=float(max(0.5, args.oom_watch_poll)),
            log_path=oom_log_path,
        )
        if monitor_enabled
        else nullcontext([])
    )

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

            run_summaries = execute_workflow(
                config,
                override_output=args.override_output,
                config_path=config_path,
            )

    except BaseException as exc:
        finished_at = _now_local_iso()
        duration_seconds = time.perf_counter() - started_monotonic
        message = _build_completion_log_entry(
            'failure',
            config_path=config_path,
            started_at=workflow_started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            run_summaries=run_summaries,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
        )
        _log_completion_event('failure', config_path, message)
        if isinstance(exc, WorkflowError):
            print(f'Workflow error: {exc}', file=sys.stderr)
            return 1
        raise
    else:
        finished_at = _now_local_iso()
        duration_seconds = time.perf_counter() - started_monotonic
        message = _build_completion_log_entry(
            'success',
            config_path=config_path,
            started_at=workflow_started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            run_summaries=run_summaries,
        )
        _log_completion_event('success', config_path, message)

    return 0


if __name__ == '__main__':
    sys.exit(main())
