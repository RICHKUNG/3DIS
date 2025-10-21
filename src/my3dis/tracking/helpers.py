"""追蹤流程中共用的輔助工具。"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from my3dis.common_utils import PACKED_ORIG_SHAPE_KEY, PACKED_SHAPE_KEY, is_packed_mask, unpack_binary_mask

__all__ = [
    'ProgressPrinter',
    'format_scale_suffix',
    'scaled_npz_path',
    'resize_mask_to_shape',
    'infer_relative_scale',
    'determine_mask_shape',
    'format_duration_precise',
    'TimingAggregator',
    'bbox_transform_xywh_to_xyxy',
    'bbox_scalar_fit',
    'compute_iou',
]


class ProgressPrinter:
    """簡易進度列，避免額外依賴 heavy UI 套件。"""

    def __init__(self, total: int) -> None:
        self.total = max(0, int(total))
        self._last_len = 0
        self._closed = False
        self._count = 0

    def update(self, index: int, abs_frame: Optional[int] = None) -> None:
        if self._closed or self.total == 0:
            return
        self._count = min(self._count + 1, self.total)
        pct = (self._count / self.total) * 100.0
        bars = int((self._count / self.total) * 20)
        bar = f"[{'#' * bars}{'.' * (20 - bars)}]"
        frame_info = f" → frame {abs_frame:05d}" if abs_frame is not None else ""
        msg = f"SAM2 tracking {bar} {self._count}/{self.total} frames ({pct:5.1f}%){frame_info}"
        pad = ' ' * max(0, self._last_len - len(msg))
        sys.stdout.write('\r' + msg + pad)
        sys.stdout.flush()
        self._last_len = len(msg)

    def close(self) -> None:
        if self._closed:
            return
        if self._last_len:
            sys.stdout.write('\n')
            sys.stdout.flush()
        self._closed = True


def format_scale_suffix(ratio: float) -> str:
    clean = f"{float(ratio):.3f}".rstrip('0').rstrip('.')
    return clean or "1"


def scaled_npz_path(path: str, ratio: float) -> str:
    if not (0.0 < float(ratio) < 1.0):
        return path
    directory, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    suffix = format_scale_suffix(ratio)
    new_name = f"{stem}_scale{suffix}x{ext}"
    return os.path.join(directory, new_name)


def resize_mask_to_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize boolean mask to target (H, W) using nearest neighbour."""
    if mask is None:
        return None
    if is_packed_mask(mask):
        arr = unpack_binary_mask(mask)
    else:
        arr = np.asarray(mask)
    arr = np.asarray(arr, dtype=np.bool_)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    target_h, target_w = target_shape
    if arr.shape == (target_h, target_w):
        return arr
    from PIL import Image

    img = Image.fromarray(arr.astype(np.uint8) * 255)
    resized = img.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.array(resized, dtype=np.uint8) >= 128)


def infer_relative_scale(mask_entry: Any) -> Optional[float]:
    """Return stored packed-to-original resolution ratio if available."""
    if not isinstance(mask_entry, dict):
        return None
    packed_shape = mask_entry.get(PACKED_SHAPE_KEY)
    orig_shape = mask_entry.get(PACKED_ORIG_SHAPE_KEY)
    if packed_shape is None or orig_shape is None:
        return None

    from my3dis.common_utils import normalize_shape_tuple

    packed = list(normalize_shape_tuple(packed_shape))
    original = list(normalize_shape_tuple(orig_shape))

    if len(packed) >= 2 and len(original) >= 2:
        h_ratio = packed[-2] / original[-2] if original[-2] else None
        w_ratio = packed[-1] / original[-1] if original[-1] else None
        ratios = [r for r in (h_ratio, w_ratio) if r is not None]
        if ratios:
            return float(sum(ratios) / len(ratios))
    return None


def determine_mask_shape(mask_entry: Any, fallback: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    """Best-effort retrieval of the full-resolution (H, W) for a mask entry."""
    if isinstance(mask_entry, dict):
        if PACKED_ORIG_SHAPE_KEY in mask_entry:
            orig = mask_entry[PACKED_ORIG_SHAPE_KEY]
            if isinstance(orig, np.ndarray):
                orig_list = orig.flatten().tolist()
            elif isinstance(orig, (list, tuple)):
                orig_list = list(orig)
            else:
                orig_list = [orig]
            if orig_list:
                if len(orig_list) >= 2:
                    return int(orig_list[-2]), int(orig_list[-1])
                val = int(orig_list[0])
                return val, val
    if is_packed_mask(mask_entry):
        arr = unpack_binary_mask(mask_entry)
        return arr.shape[-2], arr.shape[-1]
    if isinstance(mask_entry, np.ndarray):
        arr = np.asarray(mask_entry)
        if arr.ndim >= 2:
            return arr.shape[-2], arr.shape[-1]
    return fallback


def format_duration_precise(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1e-3:
        return "0m"
    minutes = seconds / 60.0
    if seconds < 60.0:
        return f"{minutes:.2f}m"
    hours = minutes / 60.0
    if seconds < 3600.0:
        return f"{minutes:.1f}m"
    return f"{hours:.1f}h"


class _TimingSection:
    def __init__(self, aggregator: "TimingAggregator", stage: str) -> None:
        self._aggregator = aggregator
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        duration = time.perf_counter() - self._start
        self._aggregator.add(self._stage, duration)
        return False


class TimingAggregator:
    """Collects and aggregates named timing spans while preserving insertion order."""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}
        self._order: List[str] = []

    def add(self, stage: str, duration: float) -> None:
        stage = str(stage)
        duration = float(duration)
        if stage not in self._totals:
            self._totals[stage] = 0.0
            self._order.append(stage)
        self._totals[stage] += max(0.0, duration)

    def track(self, stage: str) -> _TimingSection:
        return _TimingSection(self, stage)

    def total(self, stage: str) -> float:
        return self._totals.get(stage, 0.0)

    def total_prefix(self, prefix: str) -> float:
        return sum(self._totals[name] for name in self._totals if name.startswith(prefix))

    def total_all(self) -> float:
        return sum(self._totals.values())

    def items(self) -> List[Tuple[str, float]]:
        return [(stage, self._totals[stage]) for stage in self._order]

    def merge(self, other: "TimingAggregator") -> None:
        for stage, duration in other.items():
            self.add(stage, duration)

    def format_breakdown(self) -> str:
        if not self._order:
            return "n/a"
        parts = [f"{name}={format_duration_precise(duration)}" for name, duration in self.items()]
        return ", ".join(parts)


def bbox_transform_xywh_to_xyxy(bboxes: List[List[int]]) -> List[List[int]]:
    for bbox in bboxes:
        x, y, w, h = bbox
        bbox[0] = x
        bbox[1] = y
        bbox[2] = x + w
        bbox[3] = y + h
    return bboxes


def bbox_scalar_fit(bboxes: List[List[int]], scalar_x: float, scalar_y: float) -> List[List[int]]:
    for bbox in bboxes:
        bbox[0] = int(bbox[0] * scalar_x)
        bbox[1] = int(bbox[1] * scalar_y)
        bbox[2] = int(bbox[2] * scalar_x)
        bbox[3] = int(bbox[3] * scalar_y)
    return bboxes


def compute_iou(mask1: Any, mask2: Any) -> float:
    arr1 = unpack_binary_mask(mask1)
    arr2 = unpack_binary_mask(mask2)
    if arr1.shape == arr2.shape:
        inter = np.logical_and(arr1, arr2).sum()
        union = np.logical_or(arr1, arr2).sum()
        return float(inter) / float(union) if union else 0.0
    m1 = torch.from_numpy(arr1.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2 = torch.from_numpy(arr2.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2r = torch.nn.functional.interpolate(m2, size=m1.shape[-2:], mode='bilinear', align_corners=False)
    m2r = (m2r > 0.5).squeeze()
    m1 = m1.squeeze()
    inter = torch.logical_and(m1, m2r).sum().item()
    union = torch.logical_or(m1, m2r).sum().item()
    return float(inter) / float(union) if union else 0.0
