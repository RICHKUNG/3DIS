from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from .memory_events import MemoryEventsReader, detect_user_memory_event_files
from .runner import run_command


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cgroup memory event helpers")
    subparsers = parser.add_subparsers(dest="action", required=True)

    events_parser = subparsers.add_parser("events", help="Show memory.events snapshots")
    events_parser.add_argument(
        "--paths",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit memory.events files to read",
    )

    run_parser = subparsers.add_parser("run", help="Run a command and log OOM evidence")
    run_parser.add_argument("--log", type=Path, default=Path("logs/oom_monitor.log"))
    run_parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute")

    watch_parser = subparsers.add_parser("watch", help="Watch memory.events for OOM changes")
    watch_parser.add_argument("--paths", nargs="*", type=Path, default=None)
    watch_parser.add_argument("--poll", type=float, default=5.0, help="Polling interval in seconds")
    watch_parser.add_argument("--log", type=Path, default=Path("logs/oom_monitor.log"))
    watch_parser.add_argument("--email", default=None, help="Email address for notifications")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.action == "events":
        return _cmd_events(args.paths)
    if args.action == "run":
        return _cmd_run(args.cmd, args.log)
    if args.action == "watch":
        from .watcher import watch_memory_events

        paths = _resolve_paths(args.paths)
        watch_memory_events(paths, poll_interval=args.poll, log_path=args.log, email=args.email)
        return 0
    return 1


def _cmd_events(paths: Iterable[Path] | None) -> int:
    resolved = _resolve_paths(paths)
    if not resolved:
        print("No memory.events files detected", file=sys.stderr)
        return 1
    for path in resolved:
        reader = MemoryEventsReader(path)
        snapshot = reader.read()
        print(str(path))
        if not snapshot:
            print("  (unavailable)")
            continue
        for key, value in sorted(snapshot.items()):
            print(f"  {key}: {value}")
    return 0


def _cmd_run(command: List[str], log: Path) -> int:
    if not command:
        print("No command specified after --", file=sys.stderr)
        return 2
    if command[0] == "--":
        command = command[1:]
        if not command:
            print("No command specified after --", file=sys.stderr)
            return 2
    paths = _resolve_paths(None)
    result = run_command(command, paths, log_path=log)
    if result.killed_by_oom:
        print("OOM detected; see log for details", file=sys.stderr)
    return result.returncode


def _resolve_paths(paths: Iterable[Path] | None) -> List[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]
    return detect_user_memory_event_files()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
