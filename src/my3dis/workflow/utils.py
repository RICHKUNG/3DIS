"""通用工具函式。"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, List, Optional

_GPU_TOKEN_SPLIT = re.compile(r'[,\s]+')


def _collect_gpu_tokens(spec: Any) -> List[str]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple, set, frozenset)):
        collected: List[str] = []
        for item in spec:
            collected.extend(_collect_gpu_tokens(item))
        return collected
    text = str(spec).strip()
    if not text:
        return []
    if len(text) >= 2 and text[0] in '[({' and text[-1] in '])}':
        text = text[1:-1].strip()
        if not text:
            return []
    raw_tokens = _GPU_TOKEN_SPLIT.split(text)
    collected: List[str] = []
    for raw in raw_tokens:
        token = raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in {'none', 'null'}:
            continue
        if lowered.startswith('cuda:') or lowered.startswith('gpu:'):
            token = lowered.split(':', 1)[1].strip()
            if not token:
                continue
        collected.append(token)
    return collected


def normalise_gpu_spec(gpu: Optional[Any]) -> List[int]:
    tokens = _collect_gpu_tokens(gpu)
    resolved: List[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            index = int(token)
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        if index not in seen:
            seen.add(index)
            resolved.append(index)
    return resolved


def serialise_gpu_spec(gpu: Optional[Any]) -> Optional[str]:
    indices = normalise_gpu_spec(gpu)
    if not indices:
        return None
    return ','.join(str(idx) for idx in indices)


@contextmanager
def using_gpu(gpu: Optional[Any]) -> Iterator[None]:
    """暫時性設定 `CUDA_VISIBLE_DEVICES` 以選擇指定 GPU。"""
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_value = serialise_gpu_spec(gpu)
    if gpu_value is None:
        yield
        return
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_value
        yield
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous


def now_local_iso() -> str:
    """Return ISO timestamp using the system local timezone."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def now_local_stamp() -> str:
    """Return folder-friendly timestamp using local time."""
    return datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')


__all__ = ['using_gpu', 'now_local_iso', 'now_local_stamp', 'normalise_gpu_spec', 'serialise_gpu_spec']
