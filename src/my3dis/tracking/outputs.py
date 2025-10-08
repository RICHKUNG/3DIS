"""追蹤結果輸出與視覺化輔助。"""

from __future__ import annotations

import base64
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from my3dis.common_utils import (
    PACKED_MASK_B64_KEY,
    PACKED_MASK_KEY,
    PACKED_ORIG_SHAPE_KEY,
    PACKED_SHAPE_KEY,
    ensure_dir,
    unpack_binary_mask,
)

from .helpers import format_scale_suffix, resize_mask_to_shape, scaled_npz_path

__all__ = [
    'encode_packed_mask_for_json',
    'decode_packed_mask_from_json',
    'build_video_segments_archive',
    'build_object_segments_archive',
    'save_comparison_proposals',
]


def encode_packed_mask_for_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serialisable copy of a packed mask payload."""

    encoded: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == PACKED_MASK_KEY:
            arr = np.asarray(value, dtype=np.uint8)
            encoded[PACKED_MASK_B64_KEY] = base64.b64encode(arr.tobytes()).decode('ascii')
        elif isinstance(value, np.ndarray):
            encoded[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            encoded[key] = value.item()
        else:
            encoded[key] = value
    if PACKED_MASK_KEY in encoded:
        encoded.pop(PACKED_MASK_KEY, None)
    return encoded


def decode_packed_mask_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON-friendly packed mask back to numpy-compatible form."""

    decoded: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == PACKED_MASK_B64_KEY:
            decoded[PACKED_MASK_KEY] = np.frombuffer(
                base64.b64decode(value.encode('ascii')),
                dtype=np.uint8,
            )
        elif key in (PACKED_SHAPE_KEY, PACKED_ORIG_SHAPE_KEY) and isinstance(value, list):
            decoded[key] = tuple(int(v) for v in value)
        else:
            decoded[key] = value
    decoded.pop(PACKED_MASK_B64_KEY, None)
    return decoded


def _ensure_frames_dir(path: str) -> str:
    root_path = Path(path)
    root_path.parent.mkdir(parents=True, exist_ok=True)
    return str(root_path)


def _frame_entry_name(frame_idx: int) -> str:
    return f"frames/frame_{int(frame_idx):06d}.json"


def build_video_segments_archive(
    frames: Iterable[Dict[str, Any]],
    path: str,
    *,
    mask_scale_ratio: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Persist frame-major SAM2 results into a manifest-backed archive."""

    actual_path = scaled_npz_path(path, mask_scale_ratio)
    _ensure_frames_dir(actual_path)

    manifest_frames: List[Dict[str, Any]] = []
    meta = dict(metadata or {})
    meta['mask_scale_ratio'] = float(mask_scale_ratio)

    with zipfile.ZipFile(actual_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for frame in frames:
            frame_idx = int(frame['frame_index'])
            entry_name = _frame_entry_name(frame_idx)
            if 'objects' not in frame:
                frame['objects'] = {}
            manifest_frames.append(
                {
                    'frame_index': frame_idx,
                    'frame_name': frame.get('frame_name'),
                    'entry': entry_name,
                    'objects': sorted(frame['objects'].keys()),
                }
            )
            zf.writestr(entry_name, json.dumps(frame, ensure_ascii=False).encode('utf-8'))

        manifest = {
            'meta': meta,
            'frames': manifest_frames,
        }
        zf.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False).encode('utf-8'))

    return actual_path, manifest


def build_object_segments_archive(
    object_manifest: Dict[int, Dict[int, str]],
    path: str,
    *,
    mask_scale_ratio: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist an object→frame reference manifest mirroring the video archive."""

    actual_path = scaled_npz_path(path, mask_scale_ratio)
    _ensure_frames_dir(actual_path)

    meta = dict(metadata or {})
    meta['mask_scale_ratio'] = float(mask_scale_ratio)

    serialisable_objects: Dict[str, List[Dict[str, Any]]] = {}
    for obj_id, frame_map in object_manifest.items():
        entries: List[Dict[str, Any]] = []
        for frame_idx, entry_name in sorted(frame_map.items()):
            entries.append(
                {
                    'frame_index': int(frame_idx),
                    'frame_entry': entry_name,
                }
            )
        serialisable_objects[str(obj_id)] = entries

    payload = {
        'meta': meta,
        'objects': serialisable_objects,
    }

    with zipfile.ZipFile(actual_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('manifest.json', json.dumps(payload, ensure_ascii=False).encode('utf-8'))

    return actual_path


def save_comparison_proposals(
    viz_dir: str,
    base_frames_dir: str,
    filtered_per_frame: List[Optional[List[Dict[str, Any]]]],
    video_segments: Dict[int, Dict[int, Any]],
    level: int,
    frame_numbers: Optional[List[int]] = None,
    frames_to_save: Optional[List[int]] = None,
    frame_name_lookup: Optional[Dict[int, str]] = None,
    subset_dir: Optional[str] = None,
    subset_map: Optional[Dict[int, str]] = None,
) -> None:
    """輸出 SAM2 與 SSAM 的遮罩對照圖。"""
    from PIL import Image, ImageDraw

    out_dir = ensure_dir(os.path.join(viz_dir, 'compare'))

    if frame_numbers is None:
        frame_numbers = [idx for idx in range(len(filtered_per_frame))]
    frame_numbers = [int(fn) for fn in frame_numbers]
    frame_number_to_local = {fn: idx for idx, fn in enumerate(frame_numbers)}

    frames_to_render = sorted(frame_numbers)
    if frames_to_save is not None:
        frames_to_render = [f for f in frames_to_save if f in frame_number_to_local]

    if frames_to_render:
        total = len(frames_to_render)
        candidate_indices = [0, total // 2, total - 1]
        seen = set()
        ordered_frames: List[int] = []
        for idx in candidate_indices:
            if idx < 0 or idx >= total:
                continue
            frame_id = frames_to_render[idx]
            if frame_id in seen:
                continue
            ordered_frames.append(frame_id)
            seen.add(frame_id)
        frames_to_render = ordered_frames

    rng = np.random.default_rng(0)
    sam_color_map: Dict[int, Tuple[int, int, int]] = {}

    subset_lookup: Dict[int, str] = {}
    if subset_map:
        try:
            subset_lookup = {int(k): str(v) for k, v in subset_map.items()}
        except Exception:
            subset_lookup = {}

    def resolve_frame_path(frame_idx: int) -> Optional[str]:
        """Find the best effort frame path across naming conventions."""
        candidates: List[str] = []
        seen: set[str] = set()

        if frame_name_lookup:
            name = frame_name_lookup.get(frame_idx)
            if name:
                name = str(name)
                candidates.append(os.path.join(base_frames_dir, name))
                if subset_dir:
                    candidates.append(os.path.join(subset_dir, name))

        subset_name = subset_lookup.get(frame_idx)
        if subset_name and subset_dir:
            candidates.append(os.path.join(subset_dir, subset_name))

        base_names = [
            f"{frame_idx:05d}.png",
            f"{frame_idx}.png",
            f"{frame_idx:05d}.jpg",
            f"{frame_idx}.jpg",
        ]
        for name in base_names:
            candidates.append(os.path.join(base_frames_dir, name))
            if subset_dir:
                candidates.append(os.path.join(subset_dir, name))

        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.exists(path):
                return path
        return None

    def build_instance_map_img(target_size: Tuple[int, int], masks: List[np.ndarray]) -> Image.Image:
        H, W = target_size
        inst_map = np.zeros((H, W), dtype=np.int32)
        label = 0
        for seg in masks:
            if seg is None:
                continue
            if seg.shape[:2] != (H, W):
                seg_img = Image.fromarray((seg.astype(np.uint8) * 255))
                seg_img = seg_img.resize((W, H), resample=Image.NEAREST)
                seg = np.array(seg_img) > 127
            label += 1
            inst_map[(seg) & (inst_map == 0)] = label
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for idx in range(1, inst_map.max() + 1):
            color = tuple(rng.integers(50, 255, size=3).tolist())
            rgb[inst_map == idx] = color
        return Image.fromarray(rgb, 'RGB')

    def build_sam_instance_map_img(target_size: Tuple[int, int], masks_by_id: Dict[int, np.ndarray]) -> Image.Image:
        """Render SAM instance map keeping colors aligned with global object ids."""
        H, W = target_size
        inst_map = np.zeros((H, W), dtype=np.int32)
        for obj_id in sorted(masks_by_id.keys()):
            seg = masks_by_id[obj_id]
            if seg is None:
                continue
            seg_arr = np.asarray(seg)
            if seg_arr.ndim > 2:
                seg_arr = np.squeeze(seg_arr)
            seg_arr = seg_arr > 0
            if seg_arr.shape != (H, W):
                seg_img = Image.fromarray((seg_arr.astype(np.uint8) * 255))
                seg_img = seg_img.resize((W, H), resample=Image.NEAREST)
                seg_arr = np.array(seg_img) > 127
            inst_map[(seg_arr) & (inst_map == 0)] = obj_id
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        unique_obj_ids = np.unique(inst_map)
        for obj_id in unique_obj_ids:
            if obj_id == 0:
                continue
            if obj_id not in sam_color_map:
                sam_color_map[obj_id] = tuple(rng.integers(50, 255, size=3).tolist())
            rgb[inst_map == obj_id] = sam_color_map[obj_id]
        return Image.fromarray(rgb, 'RGB')

    rendered_count = 0
    for f_idx in frames_to_render:
        local_idx = frame_number_to_local.get(f_idx)
        if local_idx is None or local_idx >= len(filtered_per_frame):
            continue

        filtered_candidates = filtered_per_frame[local_idx] or []

        H = W = None
        if filtered_candidates:
            seg0 = filtered_candidates[0].get('segmentation')
            if isinstance(seg0, np.ndarray):
                H, W = seg0.shape[:2]

        sam_frame = video_segments.get(f_idx) or {}
        if H is None or W is None:
            if sam_frame:
                first_mask = next(iter(sam_frame.values()))
                if isinstance(first_mask, np.ndarray):
                    H, W = first_mask.shape[:2]

        frame_path = resolve_frame_path(f_idx)
        if frame_path is None or H is None or W is None:
            continue

        frame_img = Image.open(frame_path).convert('RGB')
        frame_img = frame_img.resize((W, H), resample=Image.BILINEAR)

        filtered_masks = [
            resize_mask_to_shape(item.get('segmentation'), (H, W))
            for item in filtered_candidates
            if item.get('segmentation') is not None
        ]
        filtered_masks = [m for m in filtered_masks if m is not None]
        filtered_img = build_instance_map_img((H, W), filtered_masks)

        sam_masks: Dict[int, np.ndarray] = {}
        for obj_id, payload in sam_frame.items():
            mask = payload
            if isinstance(payload, dict):
                mask = unpack_binary_mask(payload)
            arr = resize_mask_to_shape(mask, (H, W))
            if arr is None:
                continue
            sam_masks[int(obj_id)] = np.asarray(arr, dtype=np.bool_)

        sam_img = build_sam_instance_map_img((H, W), sam_masks)

        canvas_w = frame_img.width * 3
        canvas_h = frame_img.height
        canvas = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
        canvas.paste(frame_img, (0, 0))
        canvas.paste(filtered_img, (frame_img.width, 0))
        canvas.paste(sam_img, (frame_img.width * 2, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"Frame {f_idx}", fill=(255, 255, 255))
        draw.text((frame_img.width + 10, 10), "SSAM filtered", fill=(255, 255, 255))
        draw.text((frame_img.width * 2 + 10, 10), "SAM2 propagated", fill=(255, 255, 255))

        out_name = os.path.join(out_dir, f"L{level}_{f_idx:05d}.png")
        canvas.save(out_name)
        rendered_count += 1

    if rendered_count == 0:
        return
