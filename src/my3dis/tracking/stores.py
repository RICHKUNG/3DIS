"""Storage helpers for SAM2 tracking, including deduplication caches and frame stores."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from my3dis.common_utils import unpack_binary_mask
from my3dis.tracking.helpers import resize_mask_to_shape
from my3dis.tracking.outputs import encode_packed_mask_for_json

__all__ = [
    'DedupStore',
    'FrameResultStore',
]


def _frame_entry_name(frame_idx: int) -> str:
    return f"frames/frame_{int(frame_idx):06d}.json"


@dataclass
class _DedupEntry:
    target_shape: Tuple[int, int]
    masks: List[np.ndarray]


class DedupStore:
    """Downscaled per-frame mask stacks used for IoU-based deduplication."""

    def __init__(self, *, max_dim: int = 256) -> None:
        self._max_dim = max(1, int(max_dim))
        self._frames: Dict[int, _DedupEntry] = {}

    def _compute_target_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        h, w = shape
        longest = max(h, w)
        if longest <= self._max_dim:
            return h, w
        ratio = self._max_dim / float(longest)
        new_h = max(1, int(round(h * ratio)))
        new_w = max(1, int(round(w * ratio)))
        return new_h, new_w

    def _ensure_entry(self, frame_idx: int, mask_shape: Tuple[int, int]) -> _DedupEntry:
        entry = self._frames.get(frame_idx)
        if entry is not None:
            return entry
        target_shape = self._compute_target_shape(mask_shape)
        entry = _DedupEntry(target_shape=target_shape, masks=[])
        self._frames[frame_idx] = entry
        return entry

    @staticmethod
    def _resize(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if mask.shape == target_shape:
            return mask.astype(np.bool_)
        return resize_mask_to_shape(mask, target_shape).astype(np.bool_)

    def _max_iou(self, entry: _DedupEntry, candidate: np.ndarray) -> float:
        """Compute maximum IoU between candidate and all existing masks (vectorized)."""
        if not entry.masks:
            return 0.0

        cand = self._resize(candidate, entry.target_shape)

        # Vectorized IoU computation: stack all masks and compute in batch
        # Shape: (num_masks, H, W)
        existing_stack = np.stack(entry.masks, axis=0)

        # Broadcast candidate to match stack shape: (1, H, W) → (num_masks, H, W)
        cand_broadcast = cand[np.newaxis, :, :]

        # Compute intersection and union for all masks at once
        # Shape: (num_masks,)
        inter = np.logical_and(existing_stack, cand_broadcast).sum(axis=(1, 2))
        union = np.logical_or(existing_stack, cand_broadcast).sum(axis=(1, 2))

        # Avoid division by zero
        valid = union > 0
        if not valid.any():
            return 0.0

        # Compute IoU only for valid masks
        ious = inter[valid].astype(float) / union[valid].astype(float)
        return float(ious.max())

    def has_overlap(self, frame_idx: int, mask: np.ndarray, threshold: float) -> bool:
        entry = self._frames.get(frame_idx)
        if entry is None or not entry.masks:
            return False
        return self._max_iou(entry, np.asarray(mask, dtype=np.bool_)) > float(threshold)

    def add_mask(self, frame_idx: int, mask: np.ndarray) -> None:
        arr = np.asarray(mask, dtype=np.bool_)
        entry = self._ensure_entry(frame_idx, arr.shape)
        entry.masks.append(self._resize(arr, entry.target_shape))

    def add_packed(self, frame_idx: int, payloads: Dict[int, Any]) -> None:
        if not payloads:
            return
        for packed in payloads.values():
            arr = unpack_binary_mask(packed)
            arr = np.asarray(arr, dtype=np.bool_)
            self.add_mask(frame_idx, arr)

    def filter_candidates(
        self,
        frame_idx: int,
        candidates: List["PromptCandidate"],
        threshold: float,
    ) -> List["PromptCandidate"]:
        accepted: List["PromptCandidate"] = []
        for cand in candidates:
            seg = cand.seg_for_iou
            if seg is not None and self.has_overlap(frame_idx, seg, threshold):
                continue
            accepted.append(cand)
            if seg is not None:
                self.add_mask(frame_idx, seg)
        return accepted


class FrameResultStore:
    """Disk-backed storage for frame-major SAM2 propagation results."""

    def __init__(self, *, prefix: str = "sam2_frames_") -> None:
        self._root = Path(tempfile.mkdtemp(prefix=prefix))
        self._index: Dict[int, Path] = {}

    def update(self, frame_idx: int, frame_name: Optional[str], frame_data: Dict[int, Any]) -> str:
        entry_name = _frame_entry_name(frame_idx)
        path = self._root / f"{frame_idx:06d}.json"
        if path.exists():
            with path.open('r', encoding='utf-8') as fh:
                existing = json.load(fh)
        else:
            existing = {'frame_index': int(frame_idx), 'frame_name': frame_name, 'objects': {}}

        serialised = {
            str(obj_id): encode_packed_mask_for_json(payload)
            for obj_id, payload in frame_data.items()
        }
        existing['frame_name'] = frame_name
        existing.setdefault('objects', {})
        existing['objects'].update(serialised)

        with path.open('w', encoding='utf-8') as fh:
            json.dump(existing, fh, ensure_ascii=False)

        self._index[frame_idx] = path
        return entry_name

    def iter_frames(self) -> Iterator[Dict[str, Any]]:
        for frame_idx in sorted(self._index.keys()):
            path = self._index[frame_idx]
            with path.open('r', encoding='utf-8') as fh:
                yield json.load(fh)

    def cleanup(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
