from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .memory_events import MemoryEventsReader, OOMEvent


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    start_time: datetime
    end_time: datetime
    memory_events_before: Dict[Path, Dict[str, int]]
    memory_events_after: Dict[Path, Dict[str, int]]
    oom_events: List[OOMEvent]

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def killed_by_oom(self) -> bool:
        if self.returncode in (137, 134):
            return True
        return any(event.delta > 0 for event in self.oom_events)


def run_command(
    command: List[str],
    memory_event_paths: Iterable[Path],
    log_path: Optional[Path] = None,
) -> CommandResult:
    readers = [MemoryEventsReader(path) for path in memory_event_paths]
    before = {reader.path: reader.read() for reader in readers}
    start = datetime.utcnow()
    result = subprocess.run(command, check=False)
    end = datetime.utcnow()
    after = {reader.path: reader.read() for reader in readers}
    events: List[OOMEvent] = []
    for reader in readers:
        events.extend(reader.detect_oom_events(before.get(reader.path), after.get(reader.path, {})))
    record = CommandResult(
        command=command,
        returncode=result.returncode,
        start_time=start,
        end_time=end,
        memory_events_before=before,
        memory_events_after=after,
        oom_events=events,
    )
    if log_path:
        _append_log(record, log_path)
    return record


def _append_log(result: CommandResult, log_path: Path) -> None:
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = result.end_time.isoformat(timespec="seconds")
    lines = [
        f"[{timestamp}] command={' '.join(result.command)} returncode={result.returncode} duration={result.duration_seconds:.2f}s",
    ]
    if result.killed_by_oom:
        lines.append("status=oom_detected")
    else:
        lines.append("status=ok")
    for event in result.oom_events:
        lines.append(
            f"  {event.path}: {event.field} {event.previous} -> {event.current} (delta {event.delta})",
        )
    if not result.oom_events:
        for path, snapshot in result.memory_events_after.items():
            oom_value = snapshot.get("oom", 0)
            kill_value = snapshot.get("oom_kill", 0)
            lines.append(f"  {path}: oom={oom_value} oom_kill={kill_value}")
    lines.append("")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
