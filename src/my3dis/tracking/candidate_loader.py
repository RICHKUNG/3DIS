"""Utilities for loading filtered Semantic-SAM candidates from disk."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from my3dis.common_utils import downscale_binary_mask, unpack_binary_mask
from my3dis.tracking import infer_relative_scale

DEFAULT_PREVIEW_MAX_FRAMES = 12

__all__ = [
    'FrameCandidateBatch',
    'DEFAULT_PREVIEW_MAX_FRAMES',
    'iter_candidate_batches',
    'load_filtered_frame_by_index',
    'load_filtered_manifest',
    'select_preview_indices',
]


@dataclass
class FrameCandidateBatch:
    """Container for a single frame's filtered SSAM candidates."""

    local_index: int
    frame_index: int
    frame_name: str
    candidates: List[Dict[str, Any]]


def load_filtered_manifest(level_root: str) -> Dict[str, Any]:
    filt_dir = os.path.join(level_root, 'filtered')
    manifest_path = os.path.join(filt_dir, 'filtered.json')
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _load_frame_candidates(
    filt_dir: str,
    frame_meta: Dict[str, Any],
    *,
    mask_scale_ratio: float,
) -> List[Dict[str, Any]]:
    fidx = int(frame_meta['frame_idx'])
    seg_path = os.path.join(filt_dir, f'seg_frame_{fidx:05d}.npy')
    seg_stack = np.load(seg_path, mmap_mode='r') if os.path.exists(seg_path) else None
    try:
        items_meta = frame_meta.get('items', [])
        loaded: List[Dict[str, Any]] = []
        for j, it in enumerate(items_meta):
            d = dict(it)
            mask_payload = d.pop('mask', None)
            seg = None
            ratio_hint = None
            if mask_payload is not None:
                seg = unpack_binary_mask(mask_payload)
                ratio_hint = infer_relative_scale(mask_payload)
            elif seg_stack is not None and j < seg_stack.shape[0]:
                seg = seg_stack[j]

            seg_scaled = None
            if seg is not None:
                seg = np.asarray(seg, dtype=np.bool_)
                if mask_scale_ratio < 1.0:
                    eps = 1e-6
                    effective_ratio = ratio_hint or 1.0
                    if effective_ratio <= mask_scale_ratio + eps:
                        seg_scaled = seg
                    else:
                        relative_ratio = mask_scale_ratio / effective_ratio if effective_ratio else 0.0
                        if 0.0 < relative_ratio < 1.0 - eps:
                            seg_scaled = downscale_binary_mask(seg, relative_ratio)
                        else:
                            seg_scaled = seg
            d['segmentation'] = seg
            if mask_scale_ratio < 1.0:
                d['segmentation_scaled'] = seg_scaled if seg_scaled is not None else seg
            loaded.append(d)
        return loaded
    finally:
        if seg_stack is not None:
            del seg_stack


def iter_candidate_batches(
    level_root: str,
    frames_meta: List[Dict[str, Any]],
    *,
    mask_scale_ratio: float,
    local_indices: Optional[Sequence[int]] = None,
) -> Iterator[FrameCandidateBatch]:
    filt_dir = os.path.join(level_root, 'filtered')
    for local_idx, frame_meta in enumerate(frames_meta):
        fidx = int(frame_meta['frame_idx'])
        fname = frame_meta.get('frame_name')
        if fname is None:
            fname = f"{int(fidx):05d}.png"
        if local_indices is not None and local_idx < len(local_indices):
            effective_local = int(local_indices[local_idx])
        else:
            effective_local = local_idx
        candidates = _load_frame_candidates(filt_dir, frame_meta, mask_scale_ratio=mask_scale_ratio)
        yield FrameCandidateBatch(
            local_index=effective_local,
            frame_index=fidx,
            frame_name=str(fname),
            candidates=candidates,
        )


def load_filtered_frame_by_index(
    level_root: str,
    frames_meta: List[Dict[str, Any]],
    *,
    local_index: int,
    mask_scale_ratio: float,
) -> Optional[List[Dict[str, Any]]]:
    if local_index < 0 or local_index >= len(frames_meta):
        return None
    filt_dir = os.path.join(level_root, 'filtered')
    return _load_frame_candidates(
        filt_dir,
        frames_meta[local_index],
        mask_scale_ratio=mask_scale_ratio,
    )


def select_preview_indices(
    total_frames: int,
    *,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> List[int]:
    if total_frames <= 0:
        return []

    stride_val: Optional[int] = None
    if stride is not None:
        try:
            stride_val = max(1, int(stride))
        except (TypeError, ValueError):
            stride_val = None

    if max_samples is None:
        target: Optional[int] = DEFAULT_PREVIEW_MAX_FRAMES
    else:
        try:
            parsed_target = int(max_samples)
        except (TypeError, ValueError):
            parsed_target = DEFAULT_PREVIEW_MAX_FRAMES
        target = parsed_target if parsed_target > 0 else None

    indices: List[int]
    if stride_val:
        indices = list(range(0, total_frames, stride_val))
    else:
        if target is None or target >= total_frames:
            indices = list(range(total_frames))
        else:
            positions = np.linspace(0, total_frames - 1, num=target, dtype=int)
            indices = [int(pos) for pos in positions]

    if not indices:
        indices = [0]

    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)

    indices = sorted(set(idx for idx in indices if 0 <= idx < total_frames))
    if target is not None and len(indices) > target:
        positions = np.linspace(0, len(indices) - 1, num=target, dtype=int)
        reduced = [indices[int(pos)] for pos in positions]
        reduced[0] = 0
        reduced[-1] = total_frames - 1
        indices = sorted(set(reduced))
    return indices
