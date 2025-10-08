from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

MEMORY_EVENTS_FILENAME = "memory.events"
MEMORY_OOM_CONTROL_FILENAME = "memory.oom_control"
OOM_FIELDS = {"oom", "oom_kill"}


@dataclass(frozen=True)
class OOMEvent:
    path: Path
    field: str
    previous: int
    current: int

    @property
    def delta(self) -> int:
        return self.current - self.previous


class MemoryEventsReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> Dict[str, int]:
        data: Dict[str, int] = {}
        try:
            content = self.path.read_text(encoding="ascii", errors="replace")
        except FileNotFoundError:
            return data
        for line in content.splitlines():
            if not line.strip():
                continue
            name, _, value = line.partition(" ")
            name = name.strip()
            if not name:
                continue
            try:
                data[name] = int(value.strip())
            except ValueError:
                continue
        return data

    def detect_oom_events(
        self,
        previous: Optional[Dict[str, int]],
        current: Dict[str, int],
    ) -> List[OOMEvent]:
        if not previous:
            return []
        events: List[OOMEvent] = []
        for field in OOM_FIELDS:
            prior = previous.get(field)
            now = current.get(field)
            if prior is None or now is None:
                continue
            if now > prior:
                events.append(OOMEvent(path=self.path, field=field, previous=prior, current=now))
        return events


def detect_user_memory_event_files(user_id: Optional[int] = None) -> List[Path]:
    uid = user_id if user_id is not None else _safe_getuid()

    base_candidates = [
        Path("/sys/fs/cgroup"),
        Path("/sys/fs/cgroup/unified"),
        Path("/sys/fs/cgroup/systemd"),
        Path("/sys/fs/cgroup/memory"),
    ]

    candidates: List[Path] = []
    for base in base_candidates:
        user_root = base / "user.slice" / f"user-{uid}.slice"
        if user_root.is_dir():
            candidates.extend(_iter_memory_event_files(user_root))

        legacy_root = base / f"user.slice/user-{uid}.slice"
        if legacy_root.is_dir() and legacy_root != user_root:
            candidates.extend(_iter_memory_event_files(legacy_root))

        scope_parent = base / "user.slice" / f"user-{uid}.slice"
        if scope_parent.is_dir():
            for scope in scope_parent.glob("*.scope"):
                candidates.extend(_iter_memory_event_files(scope))

    seen = set()
    unique: List[Path] = []
    for path in sorted(candidates):
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


TARGET_EVENT_FILENAMES = {MEMORY_EVENTS_FILENAME, MEMORY_OOM_CONTROL_FILENAME}


def _iter_memory_event_files(root: Path) -> Iterator[Path]:
    if root.is_file() and root.name in TARGET_EVENT_FILENAMES:
        yield root
        return
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_memory_event_files(child)
        elif child.name in TARGET_EVENT_FILENAMES:
            yield child


def _safe_getuid() -> int:
    try:
        return os.getuid()  # type: ignore[attr-defined]
    except AttributeError:
        # Windows is not relevant here but default to 0
        return 0


try:
    import os
except ImportError:  # pragma: no cover - highly unlikely
    os = None  # type: ignore
