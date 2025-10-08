"""工作流程執行的紀錄與摘要輸出。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from my3dis.common_utils import format_duration

from .utils import now_local_iso


def build_completion_log_entry(
    status: str,
    *,
    config_path: Path,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    run_summaries: Optional[List[Dict[str, Any]]],
    error_message: Optional[str] = None,
    traceback_text: Optional[str] = None,
) -> str:
    """將完成狀態組合成可寫入日誌的字串。"""
    status_upper = status.upper()
    config_display = str(config_path)
    duration_text = format_duration(duration_seconds)

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

    return '\n'.join(lines)


def log_completion_event(status: str, config_path: Path, message: str) -> None:
    """Append a workflow completion record to the notification log."""
    log_path = Path('logs/workflow_notifications.log').expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = now_local_iso()
    header = f'[{timestamp}] status={status.upper()} config={config_path}'
    try:
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(header + '\n')
            handle.write(message + '\n\n')
    except OSError as exc:  # pragma: no cover - best effort logging
        print(f'Failed to write completion log: {exc}', file=sys.stderr)


__all__ = ['build_completion_log_entry', 'log_completion_event']
