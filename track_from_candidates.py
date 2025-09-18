"""
Tracking-only stage: read filtered candidates saved by generate_candidates.py
and run SAM2 masklet propagation. Designed to run in a SAM2-capable env.
"""

import os
import sys
import json
import argparse
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np

DEFAULT_SAM2_ROOT = "/media/Pluto/richkung/SAM2"
if DEFAULT_SAM2_ROOT not in sys.path:
    sys.path.append(DEFAULT_SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor

import torch

LOGGER = logging.getLogger("my3dis.track_from_candidates")


# Masks are stored packed-bytes to keep peak RAM usage manageable while
# still allowing downstream code to transparently request dense arrays.
PACKED_MASK_KEY = "packed_bits"
PACKED_SHAPE_KEY = "shape"


def pack_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    bool_mask = np.asarray(mask, dtype=np.bool_, order="C")
    packed = np.packbits(bool_mask.ravel())
    return {
        PACKED_MASK_KEY: packed,
        PACKED_SHAPE_KEY: tuple(int(dim) for dim in bool_mask.shape),
    }


def is_packed_mask(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and PACKED_MASK_KEY in entry
        and PACKED_SHAPE_KEY in entry
    )


def unpack_binary_mask(entry: Any) -> np.ndarray:
    if is_packed_mask(entry):
        shape = entry[PACKED_SHAPE_KEY]
        if isinstance(shape, np.ndarray):
            shape = tuple(int(v) for v in shape.tolist())
        total = int(np.prod(shape))
        unpacked = np.unpackbits(entry[PACKED_MASK_KEY], count=total)
        return unpacked.reshape(shape).astype(np.bool_)
    array = np.asarray(entry)
    if array.dtype != np.bool_:
        array = array.astype(np.bool_)
    return array


DEFAULT_SAM2_CFG = os.path.join(
    DEFAULT_SAM2_ROOT,
    "sam2",
    "configs",
    "sam2.1",
    "sam2.1_hiera_l.yaml",
)
DEFAULT_SAM2_CKPT = os.path.join(
    DEFAULT_SAM2_ROOT,
    "checkpoints",
    "sam2.1_hiera_large.pt",
)

def resolve_sam2_config_path(config_arg: str) -> str:
    cfg_path = os.path.expanduser(config_arg)
    if os.path.isfile(cfg_path):
        base = os.path.join(DEFAULT_SAM2_ROOT, 'sam2')
        rel = os.path.relpath(cfg_path, base)
        rel = rel.replace(os.sep, '/')
        if rel.endswith('.yaml'):
            rel = rel[:-5]
        return rel
    return config_arg


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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


def build_subset_video(
    frames_dir: str,
    selected: List[str],
    selected_indices: List[int],
    out_root: str,
    folder_name: str = "selected_frames",
) -> Tuple[str, Dict[int, str]]:
    subset_dir = os.path.join(out_root, folder_name)
    os.makedirs(subset_dir, exist_ok=True)
    index_to_subset: Dict[int, str] = {}
    for i, (abs_idx, fname) in enumerate(zip(selected_indices, selected)):
        src = os.path.join(frames_dir, fname)
        dst_name = f"{i:06d}.jpg"
        dst = os.path.join(subset_dir, dst_name)
        index_to_subset[abs_idx] = dst_name
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                from shutil import copy2
                copy2(src, dst)
    return subset_dir, index_to_subset


def configure_logging(explicit_level: Optional[int] = None) -> int:
    if explicit_level is None:
        log_level_name = os.environ.get("MY3DIS_LOG_LEVEL", "INFO").upper()
        explicit_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=explicit_level, format="%(message)s")
    root_logger.setLevel(explicit_level)
    LOGGER.setLevel(explicit_level)
    return explicit_level


def load_filtered_candidates(level_root: str) -> List[List[Dict[str, Any]]]:
    filt_dir = os.path.join(level_root, 'filtered')
    with open(os.path.join(filt_dir, 'filtered.json'), 'r') as f:
        meta = json.load(f)
    frames_meta = meta.get('frames', [])
    per_frame = []
    for fm in frames_meta:
        fidx = fm['frame_idx']
        items = fm['items']
        seg_path = os.path.join(filt_dir, f'seg_frame_{fidx:05d}.npy')
        seg_stack = None
        if os.path.exists(seg_path):
            seg_stack = np.load(seg_path)
        lst = []
        for j, it in enumerate(items):
            seg = seg_stack[j] if seg_stack is not None and j < seg_stack.shape[0] else None
            d = dict(it)
            d['segmentation'] = seg
            lst.append(d)
        per_frame.append(lst)
    return per_frame


def sam2_tracking(
    frames_dir: str,
    predictor,
    mask_candidates: List[List[Dict[str, Any]]],
    frame_numbers: List[int],
    iou_threshold=0.6,
):
    with torch.inference_mode(), torch.autocast("cuda"):
        obj_count = 1
        final_video_segments: Dict[int, Dict[int, Any]] = {}

        # init state once and reuse it across iterations
        state = predictor.init_state(video_path=frames_dir)
        # get scaling
        h0 = mask_candidates[0][0]['segmentation'].shape[0]
        w0 = mask_candidates[0][0]['segmentation'].shape[1]
        sx = state['video_width'] / w0
        sy = state['video_height'] / h0

        local_to_abs = {i: frame_numbers[i] for i in range(len(frame_numbers))}

        for frame_idx, frame_masks in enumerate(mask_candidates):
            # clear prompts from the previous loop without reloading video frames
            predictor.reset_state(state)
            abs_idx = local_to_abs.get(frame_idx)
            if abs_idx is None:
                continue

            # choose objects to add
            to_add = []
            prev = final_video_segments.get(abs_idx, {})
            prev_masks = list(prev.values())
            if prev_masks:
                for m in frame_masks:
                    seg = m.get('segmentation')
                    if seg is None:
                        to_add.append(m)
                        continue
                    tracked = any(compute_iou(pm, seg) > iou_threshold for pm in prev_masks)
                    if not tracked:
                        to_add.append(m)
            else:
                to_add = list(frame_masks)

            if len(to_add) == 0:
                continue

            # Prefer mask prompts for higher fidelity on the annotated frame
            for m in to_add:
                seg = m.get('segmentation')
                if seg is not None:
                    predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=obj_count,
                        mask=seg.astype(bool),
                    )
                else:
                    bbox = m.get('bbox')
                    if bbox is None:
                        continue
                    xyxy = bbox_transform_xywh_to_xyxy([list(bbox)])[0]
                    xyxy = bbox_scalar_fit([xyxy], sx, sy)[0]
                    _ , _out_ids, _out_logits = predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=frame_idx, obj_id=obj_count, box=xyxy
                    )
                obj_count += 1

            segs = {}

            def collect(iterator):
                for out_fidx, out_obj_ids, out_mask_logits in iterator:
                    abs_out_idx = local_to_abs.get(out_fidx)
                    if abs_out_idx is None:
                        continue
                    frame_data = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask_arr = (out_mask_logits[i] > 0.0).cpu().numpy()
                        frame_data[int(out_obj_id)] = pack_binary_mask(mask_arr)
                    if abs_out_idx not in segs:
                        segs[abs_out_idx] = {}
                    segs[abs_out_idx].update(frame_data)

            collect(
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=frame_idx,
                    reverse=False,
                )
            )
            if frame_idx > 0:
                # Run an additional backward sweep so late discoveries can fill early frames.
                collect(
                    predictor.propagate_in_video(
                        state,
                        start_frame_idx=frame_idx,
                        reverse=True,
                    )
                )

            for abs_out_idx, frame_data in segs.items():
                if abs_out_idx not in final_video_segments:
                    final_video_segments[abs_out_idx] = {}
                final_video_segments[abs_out_idx].update(frame_data)

        return final_video_segments


def save_video_segments_npz(segments: Dict[int, Dict[int, Any]], path: str) -> None:
    packed = {
        str(frame_idx): {str(obj_id): mask for obj_id, mask in frame_data.items()}
        for frame_idx, frame_data in segments.items()
    }
    np.savez_compressed(path, data=packed)


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


def save_object_segments_npz(segments: Dict[int, Dict[int, Any]], path: str) -> None:
    """Persist object-major mask stacks mirroring the frame-major archive."""
    packed = {
        str(obj_id): {str(frame_idx): mask for frame_idx, mask in frames.items()}
        for obj_id, frames in segments.items()
    }
    np.savez_compressed(path, data=packed)


def store_output_masks(
    output_root: str,
    frames_dir: str,
    video_segments: Dict[int, Dict[int, Any]],
    level: Optional[int] = None,
    frame_lookup: Dict[int, str] = None,
    add_label: bool = True,
):
    """Persist per-object masked images alongside JSON metadata."""
    os.makedirs(output_root, exist_ok=True)
    objects_dir = ensure_dir(os.path.join(output_root, 'objects'))

    object_segments = reorganize_segments_by_object(video_segments)

    # Build frame lookup table for reverse mapping to filenames.
    frame_catalog: Dict[int, Dict[str, str]] = {}
    if frame_lookup:
        for raw_idx, fname in frame_lookup.items():
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            frame_path = os.path.join(frames_dir, fname)
            if os.path.exists(frame_path):
                frame_catalog[idx] = {'name': fname, 'path': frame_path}
    if not frame_catalog:
        valid_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        for idx, fname in enumerate(sorted(f for f in os.listdir(frames_dir) if f.endswith(valid_ext))):
            frame_catalog[idx] = {'name': fname, 'path': os.path.join(frames_dir, fname)}

    frame_cache: Dict[int, np.ndarray] = {}
    index_entries = []

    from PIL import Image, ImageDraw

    for obj_id in sorted(object_segments.keys()):
        folder_label = f"L{level}_ID{obj_id}" if level is not None else f"ID{obj_id}"
        obj_dir = ensure_dir(os.path.join(objects_dir, folder_label))
        meta_path = os.path.join(obj_dir, 'metadata.json')

        frames_meta = []
        for frame_idx in sorted(object_segments[obj_id].keys()):
            frame_info = frame_catalog.get(int(frame_idx))
            if frame_info is None:
                continue

            cache_entry = frame_cache.get(int(frame_idx))
            if cache_entry is None:
                with Image.open(frame_info['path']) as img:
                    frame_array = np.array(img.convert('RGB'))
                frame_cache[int(frame_idx)] = frame_array
            else:
                frame_array = cache_entry

            mask = object_segments[obj_id][frame_idx]
            mask_arr = np.squeeze(unpack_binary_mask(mask))
            if mask_arr.ndim != 2:
                continue
            if mask_arr.shape != frame_array.shape[:2]:
                mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255)
                mask_img = mask_img.resize((frame_array.shape[1], frame_array.shape[0]), resample=Image.NEAREST)
                mask_arr = np.array(mask_img) > 127

            base_name, _ = os.path.splitext(frame_info['name'])
            out_path = os.path.join(obj_dir, f"{base_name}.png")

            masked = frame_array * mask_arr[:, :, np.newaxis]
            out_img = Image.fromarray(masked.astype(np.uint8))
            if add_label:
                draw = ImageDraw.Draw(out_img)
                label = f"L{level} ID:{obj_id}" if level is not None else f"ID:{obj_id}"
                width = 6 * len(label) + 20
                draw.rectangle([5, 5, 5 + width, 28], fill=(0, 0, 0))
                draw.text((10, 8), label, fill=(255, 255, 255))
            out_img.save(out_path)

            ys, xs = np.nonzero(mask_arr)
            area = int(mask_arr.sum())
            bbox = None
            if ys.size > 0:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            frames_meta.append({
                'frame_idx': int(frame_idx),
                'frame_name': frame_info['name'],
                'area': area,
                'bbox_xyxy': bbox,
                'mask_path': os.path.relpath(out_path, obj_dir),
            })

        meta = {
            'level': level,
            'object_id': int(obj_id),
            'frame_count': len(frames_meta),
            'frames': frames_meta,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        index_entries.append({
            'object_id': int(obj_id),
            'level': level,
            'metadata_path': os.path.relpath(meta_path, objects_dir),
            'frame_count': len(frames_meta),
        })

    with open(os.path.join(objects_dir, 'index.json'), 'w') as f:
        json.dump({'level': level, 'objects': index_entries}, f, indent=2)


def save_viz_frames(
    viz_dir: str,
    frames_dir: str,
    video_segments: Dict[int, Dict[int, Any]],
    level: int,
    frame_lookup: Dict[int, str] = None,
):
    """Save per-frame overlays of all masks with colored fills and labels."""
    ensure_dir(viz_dir)
    from PIL import Image, ImageDraw
    if frame_lookup is not None:
        frame_items = [(idx, frame_lookup[idx]) for idx in sorted(frame_lookup.keys())]
    else:
        frame_filenames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        frame_items = list(enumerate(frame_filenames))
    rng = np.random.default_rng(0)
    color_map: Dict[int, Tuple[int,int,int]] = {}
    for frame_idx, frame_name in frame_items:
        img_path = os.path.join(frames_dir, frame_name)
        base = Image.open(img_path).convert('RGB')
        overlay = Image.new('RGBA', base.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                if int(obj_id) not in color_map:
                    color_map[int(obj_id)] = tuple(rng.integers(50, 255, size=3).tolist())
                color = color_map[int(obj_id)]
                m = np.squeeze(unpack_binary_mask(mask))
                # create colored alpha mask
                alpha = (m.astype(np.uint8)*120)
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
                rgba[...,0]=color[0]; rgba[...,1]=color[1]; rgba[...,2]=color[2]; rgba[...,3]=alpha
                overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, 'RGBA'))
                # label
                draw = ImageDraw.Draw(overlay)
                draw.text((8,8), f"L{level}", fill=(255,255,255,255))
        comp = Image.alpha_composite(base.convert('RGBA'), overlay)
        out_path = os.path.join(viz_dir, f"{os.path.splitext(frame_name)[0]}_L{level}.png")
        comp.convert('RGB').save(out_path)


def save_instance_maps(viz_dir: str, video_segments: Dict[int, Dict[int, Any]], level: int):
    """Save colored instance maps with consistent colors for an object across frames."""
    inst_dir = ensure_dir(os.path.join(viz_dir, 'instance_map'))
    from PIL import Image
    all_obj_ids = sorted({int(k) for frame in video_segments.values() for k in frame.keys()})
    rng = np.random.default_rng(0)
    color_map = {oid: tuple(rng.integers(50,255,size=3).tolist()) for oid in all_obj_ids}
    for frame_idx in sorted(video_segments.keys()):
        objs = video_segments[frame_idx]
        # find a reference shape
        first_mask = unpack_binary_mask(next(iter(objs.values())))
        H, W = first_mask.shape[-2], first_mask.shape[-1]
        inst_map = np.zeros((H, W), dtype=np.int32)
        # assign object ids directly so colors remain consistent across frames
        for obj_id_key in sorted(objs.keys(), key=lambda x: int(x)):
            obj_id = int(obj_id_key)
            m = np.squeeze(unpack_binary_mask(objs[obj_id_key]))
            inst_map[(m) & (inst_map == 0)] = obj_id
        # colorize with a global map per object id
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for obj_id, col in color_map.items():
            rgb[inst_map == obj_id] = col
        Image.fromarray(rgb, 'RGB').save(os.path.join(inst_dir, f"frame_{frame_idx:05d}_L{level}.png"))
        np.save(os.path.join(inst_dir, f"frame_{frame_idx:05d}_L{level}.npy"), inst_map)


def save_comparison_proposals(
    viz_dir: str,
    base_frames_dir: str,
    filtered_per_frame: List[List[Dict[str, Any]]],
    video_segments: Dict[int, Dict[int, Any]],
    level: int,
    frame_numbers: List[int] = None,
    frames_to_save: List[int] = None,
):
    """Save side-by-side comparisons using instance maps (no base image).

    Left: SemanticSAM instance map
    Right: SAM2 instance map
    """
    from PIL import Image, ImageDraw

    out_dir = ensure_dir(os.path.join(viz_dir, 'compare'))

    if frame_numbers is None:
        frame_numbers = list(range(len(filtered_per_frame)))
    frame_number_to_local = {fn: idx for idx, fn in enumerate(frame_numbers)}

    # Determine all frame indices to render
    max_f_filtered = len(filtered_per_frame) - 1
    frames_from_sam2 = sorted(list(video_segments.keys()))
    frames_from_sem = list(frame_numbers)
    all_frames = sorted(set(frames_from_sam2) | set(frames_from_sem))
    if frames_to_save is not None:
        all_frames = [f for f in frames_to_save if f in set(all_frames)]

    rng = np.random.default_rng(0)
    sam_color_map: Dict[int, Tuple[int, int, int]] = {}

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

    for f_idx in all_frames:
        # Determine target size
        H = W = None
        if f_idx in video_segments and len(video_segments[f_idx]) > 0:
            first_mask = unpack_binary_mask(next(iter(video_segments[f_idx].values())))
            H, W = first_mask.shape[-2], first_mask.shape[-1]
        else:
            local_idx = frame_number_to_local.get(f_idx)
            if local_idx is not None and 0 <= local_idx <= max_f_filtered and filtered_per_frame[local_idx]:
                seg0 = filtered_per_frame[local_idx][0].get('segmentation')
                if isinstance(seg0, np.ndarray):
                    H, W = seg0.shape[:2]
        if H is None or W is None:
            continue

        # Gather masks
        sem_masks = []
        local_idx = frame_number_to_local.get(f_idx)
        if local_idx is not None and 0 <= local_idx <= max_f_filtered:
            for m in filtered_per_frame[local_idx]:
                seg = m.get('segmentation')
                if isinstance(seg, np.ndarray):
                    sem_masks.append(seg.astype(bool))
        sam2_masks: Dict[int, np.ndarray] = {}
        if f_idx in video_segments:
            for obj_key, mask in video_segments[f_idx].items():
                obj_id = int(obj_key)
                sam2_masks[obj_id] = np.squeeze(unpack_binary_mask(mask))

        sem_img = build_instance_map_img((H, W), sem_masks)
        if sam2_masks:
            sam2_img = build_sam_instance_map_img((H, W), sam2_masks)
        else:
            sam2_img = build_instance_map_img((H, W), [])

        pad = 10
        canvas = Image.new('RGB', (W * 2 + pad, H), (0, 0, 0))
        canvas.paste(sem_img, (0, 0))
        canvas.paste(sam2_img, (W + pad, 0))

        draw = ImageDraw.Draw(canvas)
        def draw_label(x, y, text):
            tw = 8 * len(text) + 10
            draw.rectangle([x, y, x + tw, y + 22], fill=(0, 0, 0))
            draw.text((x + 5, y + 5), text, fill=(255, 255, 255))
        draw_label(5, 5, f"SemanticSAM Instances L{level}")
        draw_label(W + pad + 5, 5, f"SAM2 Instances L{level}")

        out_path = os.path.join(out_dir, f"frame_{f_idx:05d}_L{level}.png")
        canvas.save(out_path)

    if all_frames:
        rep_src = os.path.join(out_dir, f"frame_{all_frames[0]:05d}_L{level}.png")
        rep_dst = os.path.join(viz_dir, f"compare_L{level}.png")
        if os.path.exists(rep_src):
            from shutil import copy2
            copy2(rep_src, rep_dst)


def run_tracking(
    *,
    data_path: str,
    candidates_root: str,
    output: str,
    levels: Union[str, List[int]] = "2,4,6",
    sam2_cfg: str = DEFAULT_SAM2_CFG,
    sam2_ckpt: str = DEFAULT_SAM2_CKPT,
    log_level: Optional[int] = None,
) -> str:
    configure_logging(log_level)

    overall_start = time.perf_counter()
    if isinstance(levels, str):
        level_list = [int(x) for x in levels.split(',') if x.strip()]
    else:
        level_list = [int(x) for x in levels]

    LOGGER.info("SAM2 tracking started (levels=%s)", ",".join(str(x) for x in level_list))

    with open(os.path.join(candidates_root, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    selected = manifest.get('selected_frames', [])
    selected_indices = manifest.get('selected_indices')
    if selected_indices is None:
        selected_indices = list(range(len(selected)))
    else:
        selected_indices = [int(x) for x in selected_indices]
    subset_dir_manifest = manifest.get('subset_dir')

    try:
        os.chdir(DEFAULT_SAM2_ROOT)
    except Exception:
        pass
    sam2_cfg_resolved = resolve_sam2_config_path(sam2_cfg)
    predictor = build_sam2_video_predictor(sam2_cfg_resolved, sam2_ckpt)

    out_root = ensure_dir(output)
    rebuild_subset = False
    if subset_dir_manifest and os.path.isdir(subset_dir_manifest):
        subset_dir = subset_dir_manifest
        subset_map = manifest.get('subset_map', {})
        subset_map = {int(k): v for k, v in subset_map.items()}
        valid_imgs = [
            f
            for f in os.listdir(subset_dir)
            if os.path.splitext(f)[1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        if len(valid_imgs) == 0:
            rebuild_subset = True
    else:
        rebuild_subset = True

    if rebuild_subset:
        subset_dir, subset_map = build_subset_video(
            frames_dir=data_path,
            selected=selected,
            selected_indices=selected_indices,
            out_root=out_root,
        )
        manifest['subset_dir'] = subset_dir
        manifest['subset_map'] = subset_map
        with open(os.path.join(candidates_root, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
    LOGGER.info("Selected frames available at %s", subset_dir)
    frame_index_to_name = {idx: name for idx, name in zip(selected_indices, selected)}

    level_stats = []
    for level in level_list:
        level_start = time.perf_counter()
        level_root = os.path.join(candidates_root, f'level_{level}')
        track_dir = ensure_dir(os.path.join(out_root, f'level_{level}', 'tracking'))
        per_frame = load_filtered_candidates(level_root)
        track_start = time.perf_counter()
        segs = sam2_tracking(
            subset_dir,
            predictor,
            per_frame,
            frame_numbers=selected_indices,
            iou_threshold=0.6,
        )
        track_time = time.perf_counter() - track_start

        persist_start = time.perf_counter()
        save_video_segments_npz(segs, os.path.join(track_dir, 'video_segments.npz'))
        obj_segments = reorganize_segments_by_object(segs)
        save_object_segments_npz(obj_segments, os.path.join(track_dir, 'object_segments.npz'))
        persist_time = time.perf_counter() - persist_start

        render_start = time.perf_counter()
        store_output_masks(
            track_dir,
            data_path,
            segs,
            level=level,
            frame_lookup=frame_index_to_name,
        )
        viz_dir = os.path.join(out_root, f'level_{level}', 'viz')
        save_viz_frames(
            viz_dir,
            data_path,
            segs,
            level,
            frame_lookup=frame_index_to_name,
        )
        save_instance_maps(viz_dir, segs, level)
        save_comparison_proposals(
            viz_dir=viz_dir,
            base_frames_dir=data_path,
            filtered_per_frame=per_frame,
            video_segments=segs,
            level=level,
            frame_numbers=selected_indices,
            frames_to_save=None,
        )
        render_time = time.perf_counter() - render_start

        form_time = time.perf_counter() - level_start
        LOGGER.info(
            "  Timings → track=%s, persist=%s, render=%s, total=%s",
            format_seconds(track_time),
            format_seconds(persist_time),
            format_seconds(render_time),
            format_seconds(form_time),
        )
        level_stats.append(
            (
                level,
                len(obj_segments),
                len(segs),
                track_time,
                persist_time,
                render_time,
                form_time,
            )
        )
        LOGGER.info(
            "Level %d finished in %s (objects=%d, frames=%d)",
            level,
            format_seconds(form_time),
            len(obj_segments),
            len(segs),
        )

    if level_stats:
        summary = "; ".join(
            f"L{lvl}: {objs} objects / {frames} frames "
            f"(track={format_seconds(track)}, render={format_seconds(render)}, total={format_seconds(total)})"
            for lvl, objs, frames, track, _, render, total in level_stats
        )
        LOGGER.info("Tracking summary → %s", summary)
    LOGGER.info("Tracking results saved at %s", out_root)
    LOGGER.info(
        "Tracking completed in %s",
        format_seconds(time.perf_counter() - overall_start),
    )

    return out_root


def main():
    ap = argparse.ArgumentParser(description="SAM2 tracking from pre-generated candidates")
    ap.add_argument('--data-path', required=True, help='Original frames dir')
    ap.add_argument('--candidates-root', required=True, help='Root containing level_*/filtered')
    ap.add_argument('--sam2-cfg', default=DEFAULT_SAM2_CFG,
                    help='SAM2 config YAML or Hydra path (default: sam2.1_hiera_l)')
    ap.add_argument('--sam2-ckpt', default=DEFAULT_SAM2_CKPT,
                    help='SAM2 checkpoint path (default: sam2.1_hiera_large.pt)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--levels', default='2,4,6')
    args = ap.parse_args()

    run_tracking(
        data_path=args.data_path,
        candidates_root=args.candidates_root,
        output=args.output,
        levels=args.levels,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
    )


if __name__ == '__main__':
    main()
