"""通用工具函式。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional


@contextmanager
def using_gpu(gpu: Optional[Any]) -> Iterator[None]:
    """暫時性設定 `CUDA_VISIBLE_DEVICES` 以選擇指定 GPU。"""
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu is None:
        yield
        return
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
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


__all__ = ['using_gpu', 'now_local_iso', 'now_local_stamp']
