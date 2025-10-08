from __future__ import annotations

import sys
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from .memory_events import MemoryEventsReader, OOMEvent, detect_user_memory_event_files


def watch_memory_events(
    paths: Iterable[Path] | None,
    *,
    poll_interval: float = 5.0,
    log_path: Path = Path("logs/oom_monitor.log"),
    stop_event: Optional[threading.Event] = None,
    announce: bool = True,
) -> None:
    resolved_paths = list(paths) if paths else detect_user_memory_event_files()
    if not resolved_paths:
        print("No memory.events files to watch", file=sys.stderr)
        return
    readers = [MemoryEventsReader(path) for path in resolved_paths]
    previous: Dict[Path, Dict[str, int]] = {reader.path: reader.read() for reader in readers}
    if announce:
        print("Watching for OOM events. Press Ctrl+C to stop.")
    interrupted = False
    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            for reader in readers:
                current = reader.read()
                events = reader.detect_oom_events(previous.get(reader.path), current)
                if events:
                    for event in events:
                        _log_event(event, log_path)
                        _emit_console(event)
                previous[reader.path] = current
            if stop_event and stop_event.is_set():
                break
            time.sleep(max(0.5, poll_interval))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if announce:
            if interrupted:
                print("Stopped watching.")
            else:
                print("Stopping OOM watch.")


def _log_event(event: OOMEvent, log_path: Path) -> None:
    log_path = log_path.expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    line = (
        f"[{timestamp}] watch path={event.path} field={event.field}"
        f" previous={event.previous} current={event.current} delta={event.delta}"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _emit_console(event: OOMEvent) -> None:
    print(
        f"OOM event detected: {event.path} {event.field} {event.previous} -> {event.current}",
        file=sys.stderr,
    )


@contextmanager
def memory_watch_context(
    paths: Iterable[Path] | None = None,
    *,
    poll_interval: float = 5.0,
    log_path: Path = Path("logs/oom_monitor.log"),
):
    resolved_paths = list(paths) if paths else detect_user_memory_event_files()
    if not resolved_paths:
        yield resolved_paths
        return
    stop_event = threading.Event()
    thread = threading.Thread(
        target=watch_memory_events,
        args=(resolved_paths,),
        kwargs={
            "poll_interval": poll_interval,
            "log_path": log_path,
            "stop_event": stop_event,
            "announce": False,
        },
        daemon=True,
    )
    thread.start()
    try:
        yield resolved_paths
    finally:
        stop_event.set()
        thread.join(timeout=max(1.0, poll_interval * 2))
