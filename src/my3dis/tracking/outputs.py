"""追蹤結果輸出相關函式。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from my3dis.common_utils import ensure_dir

from .helpers import format_scale_suffix, resize_mask_to_shape, scaled_npz_path

__all__ = [
    'save_video_segments_npz',
    'reorganize_segments_by_object',
    'save_object_segments_npz',
    'save_comparison_proposals',
]


def save_video_segments_npz(
    segments: Dict[int, Dict[int, Any]],
    path: str,
    *,
    mask_scale_ratio: float = 1.0,
) -> str:
    actual_path = scaled_npz_path(path, mask_scale_ratio)
    packed = {
        str(frame_idx): {str(obj_id): mask for obj_id, mask in frame_data.items()}
        for frame_idx, frame_data in segments.items()
    }
    np.savez_compressed(actual_path, data=packed)
    return actual_path


def reorganize_segments_by_object(segments: Dict[int, Dict[int, Any]]) -> Dict[int, Dict[int, Any]]:
    """Re-index frame-major predictions into an object-major structure."""
    per_object: Dict[int, Dict[int, Any]] = {}
    for frame_idx, frame_data in segments.items():
        for obj_id_raw, mask in frame_data.items():
            obj_id = int(obj_id_raw)
            frame_id = int(frame_idx)
            if obj_id not in per_object:
                per_object[obj_id] = {}
            per_object[obj_id][frame_id] = mask
    return per_object


def save_object_segments_npz(
    segments: Dict[int, Dict[int, Any]],
    path: str,
    *,
    mask_scale_ratio: float = 1.0,
) -> str:
    """Persist object-major mask stacks mirroring the frame-major archive."""
    actual_path = scaled_npz_path(path, mask_scale_ratio)
    packed = {
        str(obj_id): {str(frame_idx): mask for frame_idx, mask in frames.items()}
        for obj_id, frames in segments.items()
    }
    np.savez_compressed(actual_path, data=packed)
    return actual_path


def save_comparison_proposals(
    viz_dir: str,
    base_frames_dir: str,
    filtered_per_frame: List[List[Dict[str, Any]]],
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
        frame_numbers = list(range(len(filtered_per_frame)))
    frame_numbers = [int(fn) for fn in frame_numbers]
    frame_number_to_local = {fn: idx for idx, fn in enumerate(frame_numbers)}

    frames_to_render = sorted(frame_numbers)
    if frames_to_save is not None:
        frames_to_render = [f for f in frames_to_save if f in set(frame_numbers)]

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

        # Backward compatibility fallbacks (zero-padded / plain indices)
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

        H = W = None
        if filtered_per_frame[local_idx]:
            seg0 = filtered_per_frame[local_idx][0].get('segmentation')
            if isinstance(seg0, np.ndarray):
                H, W = seg0.shape[:2]

        if H is None or W is None:
            sam_frame = video_segments.get(f_idx)
            if sam_frame:
                first_mask = next(iter(sam_frame.values()))
                if isinstance(first_mask, np.ndarray):
                    H, W = first_mask.shape[:2]

        frame_path = resolve_frame_path(f_idx)
        if frame_path is None:
            continue

        with Image.open(frame_path) as base_img:
            base_rgb = base_img.convert('RGB')
        if H is None or W is None:
            H, W = base_rgb.height, base_rgb.width

        sam_masks = [item.get('segmentation') for item in filtered_per_frame[local_idx]]
        sam_instance = build_instance_map_img((H, W), sam_masks)

        sam2_masks = {
            obj_id: resize_mask_to_shape(mask, (H, W))
            for obj_id, mask in video_segments.get(f_idx, {}).items()
        }
        sam2_instance = build_sam_instance_map_img((H, W), sam2_masks)

        gap = 20
        width = base_rgb.width + sam_instance.width + sam2_instance.width + gap * 2
        height = max(base_rgb.height, sam_instance.height, sam2_instance.height)

        canvas = Image.new('RGB', (width, height), color=(0, 0, 0))
        canvas.paste(base_rgb, (0, 0))
        canvas.paste(sam_instance, (base_rgb.width + gap, 0))
        canvas.paste(sam2_instance, (base_rgb.width + sam_instance.width + gap * 2, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f'Level {level} Frame {f_idx}', fill=(255, 255, 255))

        canvas.save(os.path.join(out_dir, f'compare_L{level:02d}_F{f_idx:05d}.png'))
        rendered_count += 1

    if rendered_count == 0:
        print('No comparison proposals rendered.')
