"""
Bridge pipeline: Semantic-SAM (multi-level) → SAM2 tracking (Algorithm 1 variant)

This script follows My3DIS/Agent.md:
- Levels fixed to [2,4,6] unless overridden
- Frame range like 1200:1600:20 (end exclusive)
- Saves per-level candidate lists (raw and filtered), tracking results and simple visualizations

Note: This file does not execute on import. Use the CLI at bottom.
"""

import os
import sys
import json
import argparse
import time
import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

# --- Local repo paths (non-destructive) ---
DEFAULT_SEMANTIC_SAM_ROOT = "/media/Pluto/richkung/Semantic-SAM"
DEFAULT_SAM2_ROOT = "/media/Pluto/richkung/SAM2"

# Checkpoints are fixed for this deployment; keep them here so callers do not
# need to provide CLI overrides every run.
DEFAULT_SEMANTIC_SAM_CKPT = os.path.join(
    DEFAULT_SEMANTIC_SAM_ROOT,
    "checkpoints",
    "swinl_only_sam_many2many.pth",
)
DEFAULT_SAM2_CKPT = os.path.join(
    DEFAULT_SAM2_ROOT,
    "checkpoints",
    "sam2.1_hiera_large.pt",
)

if DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.append(DEFAULT_SEMANTIC_SAM_ROOT)
if DEFAULT_SAM2_ROOT not in sys.path:
    sys.path.append(DEFAULT_SAM2_ROOT)

from semantic_sam import (
    prepare_image,
    build_semantic_sam,
    SemanticSamAutomaticMaskGenerator,
)

# from auto_generation_inference import instance_map_to_anns  # unused import retained for reference

from sam2.build_sam import build_sam2_video_predictor

import torch


# ---------------------
# Path helpers
# ---------------------
def resolve_sam2_config_path(config_arg: str) -> str:
    """Allow passing either Hydra config name or absolute YAML path."""
    cfg_path = os.path.expanduser(config_arg)
    if os.path.isfile(cfg_path):
        base = os.path.join(DEFAULT_SAM2_ROOT, 'sam2')
        rel = os.path.relpath(cfg_path, base)
        rel = rel.replace(os.sep, '/')
        if rel.endswith('.yaml'):
            rel = rel[:-5]
        return rel
    return config_arg


# ---------------------
# Utilities
# ---------------------
def parse_levels(levels_str: str) -> List[int]:
    return [int(x) for x in str(levels_str).split(',') if str(x).strip()]


def parse_range(range_str: str) -> Tuple[int, int, int]:
    parts = str(range_str).split(':')
    if len(parts) != 3:
        raise ValueError("frames must be 'start:end:step'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def bbox_from_mask_xyxy(seg: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def xyxy_to_xywh(bbox_xyxy: Tuple[int, int, int, int]) -> List[int]:
    x1, y1, x2, y2 = bbox_xyxy
    return [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]


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


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    # If masks are already same size
    if mask1.shape == mask2.shape:
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(inter) / float(union) if union else 0.0

    # Resize mask2 to mask1 via torch (avoid cv2 dependency)
    m1 = torch.from_numpy(mask1.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2 = torch.from_numpy(mask2.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2r = torch.nn.functional.interpolate(m2, size=m1.shape[-2:], mode='bilinear', align_corners=False)
    m2r = (m2r > 0.5).squeeze()
    m1 = m1.squeeze()
    inter = torch.logical_and(m1, m2r).sum().item()
    union = torch.logical_or(m1, m2r).sum().item()
    return float(inter) / float(union) if union else 0.0


# ---------------------
# SAM2 tracking (Algorithm 1 core)
# ---------------------
def sam2_tracking(
    frames_dir: str,
    predictor,
    mask_candidates: List[List[Dict[str, Any]]],
    frame_numbers: List[int],
    iou_threshold: float = 0.6,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Run Algorithm 1 variant using SAM2 video predictor.

    mask_candidates: list over sampled-frame-index -> list of mask dicts (with 'bbox' XYWH, 'segmentation', ...)
    frame_numbers: absolute frame indices (relative to frames_dir ordering) matching mask_candidates order
    Returns: final per-frame dict: abs_frame_idx -> {obj_id: mask_bool_array}
    """
    with torch.inference_mode(), torch.autocast("cuda"):
        obj_count = 1
        final_video_segments: Dict[int, Dict[int, np.ndarray]] = {}

        # Initialize SAM2 video predictor state
        inference_state = predictor.init_state(video_path=frames_dir)

        # Scaling factors from candidate mask size to SAM2 input size
        # Use first candidate to infer H,W
        h0 = mask_candidates[0][0]["segmentation"].shape[0]
        w0 = mask_candidates[0][0]["segmentation"].shape[1]
        scalar_x = inference_state['video_width'] / w0
        scalar_y = inference_state['video_height'] / h0

        local_to_abs = {i: frame_numbers[i] for i in range(len(frame_numbers))}

        for local_idx, frame_masks in enumerate(mask_candidates):
            # re-init state so each iteration prompts fresh boxes
            inference_state = predictor.init_state(video_path=frames_dir)

            abs_idx = local_to_abs.get(local_idx)
            if abs_idx is None:
                continue

            existing_masks_map = final_video_segments.get(abs_idx, {})
            existing_masks = list(existing_masks_map.values())

            candidate_pool: List[Dict[str, Any]] = []
            if existing_masks:
                for m in frame_masks:
                    seg = m.get("segmentation")
                    if seg is None:
                        candidate_pool.append(m)
                        continue
                    tracked = any(compute_iou(pm, seg) > iou_threshold for pm in existing_masks)
                    if not tracked:
                        candidate_pool.append(m)
            else:
                candidate_pool = list(frame_masks)

            if not candidate_pool:
                continue

            # Prefer mask prompts when available; fallback to boxes
            # If any mask exists, use them directly for higher fidelity
            masks_used = False
            for m in candidate_pool:
                seg = m.get('segmentation') if isinstance(m, dict) else None
                if seg is not None:
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=local_idx,
                        obj_id=obj_count,
                        mask=seg.astype(bool),
                    )
                    obj_count += 1
                    masks_used = True
            if not masks_used:
                bboxes_xywh = [m.get('bbox') for m in candidate_pool if m.get('bbox') is not None]
                bboxes_xyxy = bbox_transform_xywh_to_xyxy([list(bb) for bb in bboxes_xywh])
                bboxes_xyxy = bbox_scalar_fit(bboxes_xyxy, scalar_x, scalar_y)
                for bbox in bboxes_xyxy:
                    ann_obj_id = obj_count
                    obj_count += 1
                    _ , _out_ids, _out_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=local_idx,
                        obj_id=ann_obj_id,
                        box=bbox,
                    )

            # Propagate in video
            video_segments = {}
            for out_local_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                abs_out_idx = local_to_abs.get(out_local_idx)
                if abs_out_idx is None:
                    continue
                video_segments[abs_out_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Merge segments from this iteration
            for abs_out_idx, frame_data in video_segments.items():
                if abs_out_idx not in final_video_segments:
                    final_video_segments[abs_out_idx] = {}
                final_video_segments[abs_out_idx].update(frame_data)

        return final_video_segments


# ---------------------
# Semantic-SAM per-level candidate generation
# ---------------------
def generate_candidates_per_level(
    frames_dir: str,
    selected_frames: List[str],
    frame_indices: List[int],
    semantic_sam,
    level: int,
    min_area: int,
    stability_thresh: float,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Returns (filtered_per_frame, raw_flattened_for_saving)

    filtered_per_frame: list over frame order -> list of dicts with keys ['bbox','segmentation','area','stability_score','level']
    raw_flattened_for_saving: flat list with frame metadata for JSON persistence
    """
    gen = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[level])
    per_frame_filtered: List[List[Dict[str, Any]]] = []
    raw_dump: List[Dict[str, Any]] = []

    for f_idx, fname in enumerate(selected_frames):
        abs_idx = frame_indices[f_idx]
        img_path = os.path.join(frames_dir, fname)
        _, tensor_img = prepare_image(img_path)
        masks = gen.generate(tensor_img)

        # enrich + compute bbox when missing
        filtered = []
        for m in masks:
            seg = m.get('segmentation')
            if seg is None:
                continue
            area = int(m.get('area', int(seg.sum())))
            stability = float(m.get('stability_score', 1.0))
            bbox = m.get('bbox')
            if bbox is None:
                x1, y1, x2, y2 = bbox_from_mask_xyxy(seg)
                bbox = xyxy_to_xywh((x1, y1, x2, y2))
            cand = {
                'frame_idx': f_idx,
                'frame_number': abs_idx,
                'frame_name': fname,
                'bbox': [int(b) for b in bbox],
                'area': area,
                'stability_score': stability,
                'level': level,
                'segmentation': seg,
            }
            raw_dump.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cand.items() if k != 'segmentation'})

            if area >= min_area and stability >= stability_thresh:
                filtered.append(cand)

        per_frame_filtered.append(filtered)

    return per_frame_filtered, raw_dump


# ---------------------
# Persist helpers
# ---------------------
def save_json(obj: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_video_segments_npz(segments: Dict[int, Dict[int, np.ndarray]], path: str) -> None:
    # pack as dict of dict of arrays
    packed = {str(k): {str(kk): vv.astype(np.uint8) for kk, vv in v.items()} for k, v in segments.items()}
    np.savez_compressed(path, data=packed)


def store_output_masks(
    output_root: str,
    frames_dir: str,
    video_segments: Dict[int, Dict[int, np.ndarray]],
    level: int = None,
    add_label: bool = True,
    frame_lookup: Dict[int, str] = None,
):
    os.makedirs(output_root, exist_ok=True)

    # Find all unique object ids
    all_obj_ids = set()
    for frame_masks in video_segments.values():
        all_obj_ids.update(int(k) for k in frame_masks.keys())

    # Create folders
    for oid in all_obj_ids:
        folder_name = f"L{level}_ID{oid}" if level is not None else str(oid)
        ensure_dir(os.path.join(output_root, 'objects', folder_name))

    # Save masked images for all frames
    if frame_lookup is not None:
        frame_items = [(idx, frame_lookup[idx]) for idx in sorted(frame_lookup.keys())]
    else:
        filenames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        frame_items = list(enumerate(filenames))

    for frame_idx, frame_name in frame_items:
        if frame_idx not in video_segments:
            continue
        img_path = os.path.join(frames_dir, frame_name)
        from PIL import Image, ImageDraw
        img_pil = Image.open(img_path).convert('RGB')
        img_arr = np.array(img_pil)
        frame_masks = video_segments[frame_idx]
        for obj_id, mask in frame_masks.items():
            bool_mask = np.squeeze(mask > 0)
            masked_arr = img_arr * bool_mask[:, :, np.newaxis]
            out_img = Image.fromarray(masked_arr.astype(np.uint8))
            if add_label:
                draw = ImageDraw.Draw(out_img)
                label = f"L{level} ID:{obj_id}" if level is not None else f"ID:{obj_id}"
                box = [5, 5, 5 + 8 * len(label) + 8, 28]
                draw.rectangle(box, fill=(0, 0, 0))
                draw.text((10, 8), label, fill=(255, 255, 255))
            folder_name = f"L{level}_ID{obj_id}" if level is not None else str(obj_id)
            out_path = os.path.join(output_root, 'objects', folder_name, os.path.splitext(frame_name)[0] + '.png')
            ensure_dir(os.path.dirname(out_path))
            out_img.save(out_path)


def save_viz_frames(
    viz_dir: str,
    frames_dir: str,
    video_segments: Dict[int, Dict[int, np.ndarray]],
    level: int,
    frame_lookup: Dict[int, str] = None,
):
    ensure_dir(viz_dir)
    from PIL import Image
    if frame_lookup is not None:
        frame_items = [(idx, frame_lookup[idx]) for idx in sorted(frame_lookup.keys())]
    else:
        filenames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
        frame_items = list(enumerate(filenames))
    rng = np.random.default_rng(0)
    color_map: Dict[int, Tuple[int,int,int]] = {}
    for frame_idx, frame_name in frame_items:
        base = Image.open(os.path.join(frames_dir, frame_name)).convert('RGB')
        overlay = Image.new('RGBA', base.size, (0,0,0,0))
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                if int(obj_id) not in color_map:
                    color_map[int(obj_id)] = tuple(rng.integers(50,255,size=3).tolist())
                color = color_map[int(obj_id)]
                m = np.squeeze(mask>0)
                alpha = (m.astype(np.uint8)*120)
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
                rgba[...,0]=color[0]; rgba[...,1]=color[1]; rgba[...,2]=color[2]; rgba[...,3]=alpha
                overlay = Image.alpha_composite(overlay, Image.fromarray(rgba,'RGBA'))
        comp = Image.alpha_composite(base.convert('RGBA'), overlay)
        comp.convert('RGB').save(os.path.join(viz_dir, f"{os.path.splitext(frame_name)[0]}_L{level}.png"))


def save_instance_maps(viz_dir: str, video_segments: Dict[int, Dict[int, np.ndarray]], level: int):
    inst_dir = ensure_dir(os.path.join(viz_dir, 'instance_map'))
    from PIL import Image
    all_obj_ids = sorted({int(k) for frame in video_segments.values() for k in frame.keys()})
    rng = np.random.default_rng(0)
    color_map = {oid: tuple(rng.integers(50,255,size=3).tolist()) for oid in all_obj_ids}
    for frame_idx in sorted(video_segments.keys()):
        objs = video_segments[frame_idx]
        first_mask = next(iter(objs.values()))
        H, W = first_mask.shape[-2], first_mask.shape[-1]
        inst_map = np.zeros((H, W), dtype=np.int32)
        for obj_id_key in sorted(objs.keys(), key=lambda x: int(x)):
            obj_id = int(obj_id_key)
            m = np.squeeze(objs[obj_id_key] > 0)
            inst_map[(m) & (inst_map == 0)] = obj_id
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for obj_id, col in color_map.items():
            rgb[inst_map == obj_id] = col
        Image.fromarray(rgb, 'RGB').save(os.path.join(inst_dir, f"frame_{frame_idx:05d}_L{level}.png"))
        np.save(os.path.join(inst_dir, f"frame_{frame_idx:05d}_L{level}.npy"), inst_map)


def save_comparison_proposals(
    viz_dir: str,
    base_frames_dir: str,
    filtered_per_frame: List[List[Dict[str, Any]]],
    video_segments: Dict[int, Dict[int, np.ndarray]],
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

    # Determine frames to render
    max_f_filtered = len(filtered_per_frame) - 1
    frames_from_sam2 = sorted(list(video_segments.keys()))
    frames_from_sem = list(frame_numbers)
    all_frames = sorted(set(frames_from_sam2) | set(frames_from_sem))
    if frames_to_save is not None:
        all_frames = [f for f in frames_to_save if f in set(all_frames)]

    rng = np.random.default_rng(0)

    def build_instance_map_img(target_size: Tuple[int, int], masks: List[np.ndarray]) -> Image.Image:
        H, W = target_size
        inst_map = np.zeros((H, W), dtype=np.int32)
        label = 0
        for seg in masks:
            if seg is None:
                continue
            # Resize if needed
            if seg.shape[:2] != (H, W):
                seg_img = Image.fromarray((seg.astype(np.uint8) * 255))
                seg_img = seg_img.resize((W, H), resample=Image.NEAREST)
                seg = np.array(seg_img) > 127
            label += 1
            inst_map[(seg) & (inst_map == 0)] = label
        # Colorize
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for idx in range(1, inst_map.max() + 1):
            color = tuple(rng.integers(50, 255, size=3).tolist())
            rgb[inst_map == idx] = color
        return Image.fromarray(rgb, 'RGB')

    for f_idx in all_frames:
        # Determine target size from SAM2 masks if available; else from first SSAM mask
        H = W = None
        if f_idx in video_segments and len(video_segments[f_idx]) > 0:
            first_mask = next(iter(video_segments[f_idx].values()))
            H, W = first_mask.shape[-2], first_mask.shape[-1]
        else:
            local_idx = frame_number_to_local.get(f_idx)
            if local_idx is not None and 0 <= local_idx <= max_f_filtered and filtered_per_frame[local_idx]:
                seg0 = filtered_per_frame[local_idx][0].get('segmentation')
                if isinstance(seg0, np.ndarray):
                    H, W = seg0.shape[:2]
        if H is None or W is None:
            continue

        # Collect masks
        sem_masks = []
        local_idx = frame_number_to_local.get(f_idx)
        if local_idx is not None and 0 <= local_idx <= max_f_filtered:
            for m in filtered_per_frame[local_idx]:
                seg = m.get('segmentation')
                if isinstance(seg, np.ndarray):
                    sem_masks.append(seg.astype(bool))
        sam2_masks = []
        if f_idx in video_segments:
            for _, mask in video_segments[f_idx].items():
                sam2_masks.append(np.squeeze(mask > 0))

        sem_img = build_instance_map_img((H, W), sem_masks)
        sam2_img = build_instance_map_img((H, W), sam2_masks)

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
        rep_path_src = os.path.join(out_dir, f"frame_{all_frames[0]:05d}_L{level}.png")
        rep_path_dst = os.path.join(viz_dir, f"compare_L{level}.png")
        if os.path.exists(rep_path_src):
            from shutil import copy2
            copy2(rep_path_src, rep_path_dst)


def build_subset_video(
    frames_dir: str,
    selected: List[str],
    selected_indices: List[int],
    out_root: str,
    folder_name: str = "selected_frames",
) -> Tuple[str, Dict[int, str]]:
    """Create a lightweight directory containing only the selected frames.

    Returns (subset_dir, index_to_subset_filename)
    """
    subset_dir = os.path.join(out_root, folder_name)
    os.makedirs(subset_dir, exist_ok=True)
    index_to_subset: Dict[int, str] = {}
    for i, (abs_idx, fname) in enumerate(zip(selected_indices, selected)):
        src = os.path.join(frames_dir, fname)
        subset_name = f"{i:06d}.jpg"
        dst = os.path.join(subset_dir, subset_name)
        index_to_subset[abs_idx] = subset_name
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                from shutil import copy2
                copy2(src, dst)
    return subset_dir, index_to_subset

def main():
    ap = argparse.ArgumentParser(description="Semantic-SAM multi-level → SAM2 tracking pipeline")
    ap.add_argument('--data-path', required=True, help='Folder containing all frames (JPG/PNG)')
    ap.add_argument('--levels', default='2,4,6', help='Comma-separated levels, default 2,4,6')
    ap.add_argument('--frames', default='1200:1600:20', help='Range as start:end:step (end exclusive)')
    ap.add_argument('--sam2-cfg', required=True)
    ap.add_argument('--output', required=True, help='Output root inside My3DIS')
    ap.add_argument('--min-area', type=int, default=300)
    ap.add_argument('--stability-threshold', type=float, default=0.9)
    ap.add_argument('--no-timestamp', action='store_true', help='Do not append timestamp folder to output root')
    args = ap.parse_args()

    frames_dir = args.data_path
    start, end, step = parse_range(args.frames)
    levels = parse_levels(args.levels)

    # Resolve frames
    all_frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    selected_indices = list(range(start, min(end, len(all_frames)), step))
    selected = [all_frames[i] for i in selected_indices]

    # Load models
    print("⏳ Loading Semantic-SAM...")
    semantic_sam = build_semantic_sam(model_type="L", ckpt=DEFAULT_SEMANTIC_SAM_CKPT)
    print("✅ Semantic-SAM ready")
    print("⏳ Loading SAM2 video predictor...")
    sam2_cfg = resolve_sam2_config_path(args.sam2_cfg)
    predictor = build_sam2_video_predictor(sam2_cfg, DEFAULT_SAM2_CKPT)
    print("✅ SAM2 ready")

    # Output structure
    # Timestamped experiment folder
    if args.no_timestamp:
        out_root = ensure_dir(args.output)
        ts = None
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = ensure_dir(os.path.join(args.output, ts))
    subset_dir, index_to_subset = build_subset_video(
        frames_dir=frames_dir,
        selected=selected,
        selected_indices=selected_indices,
        out_root=out_root,
    )
    manifest = {
        'levels': levels,
        'frames': args.frames,
        'min_area': args.min_area,
        'stability_threshold': args.stability_threshold,
        'data_path': frames_dir,
        'selected_frames': selected,
        'selected_indices': selected_indices,
        'subset_dir': subset_dir,
        'subset_map': index_to_subset,
        'sam_ckpt': DEFAULT_SEMANTIC_SAM_CKPT,
        'sam2_cfg': args.sam2_cfg,
        'sam2_ckpt': DEFAULT_SAM2_CKPT,
        'ts_epoch': int(time.time()),
        'timestamp': ts,
    }
    save_json(manifest, os.path.join(out_root, 'manifest.json'))

    print(f"🗂️ Selected frames copied to: {subset_dir}")
    frame_index_to_name = {idx: name for idx, name in zip(selected_indices, selected)}

    # Per-level processing
    for level in levels:
        print(f"\n=== Level {level} ===")
        level_root = ensure_dir(os.path.join(out_root, f'level_{level}'))
        cand_dir = ensure_dir(os.path.join(level_root, 'candidates'))
        filt_dir = ensure_dir(os.path.join(level_root, 'filtered'))
        track_dir = ensure_dir(os.path.join(level_root, 'tracking'))
        viz_dir = ensure_dir(os.path.join(level_root, 'viz'))

        # Generate candidates for this level
        filtered_per_frame, raw_dump = generate_candidates_per_level(
            frames_dir=frames_dir,
            selected_frames=selected,
            frame_indices=selected_indices,
            semantic_sam=semantic_sam,
            level=level,
            min_area=args.min_area,
            stability_thresh=args.stability_threshold,
        )

        # Persist raw and filtered lists
        save_json({'items': raw_dump}, os.path.join(cand_dir, 'candidates.json'))

        # For filtered, drop heavy masks for JSON; store minimal metadata in JSON, masks to npy per-frame
        filtered_json = []
        for f_idx, lst in enumerate(filtered_per_frame):
            meta_list = []
            seg_stack = []
            for j, m in enumerate(lst):
                meta = {k: v for k, v in m.items() if k != 'segmentation'}
                meta['id'] = j
                meta_list.append(meta)
                seg_stack.append(m['segmentation'].astype(np.uint8))
            filtered_json.append({'frame_idx': f_idx, 'count': len(meta_list), 'items': meta_list})
            if seg_stack:
                np.save(os.path.join(filt_dir, f'seg_frame_{f_idx:05d}.npy'), np.stack(seg_stack, axis=0))
        save_json({'frames': filtered_json}, os.path.join(filt_dir, 'filtered.json'))

        # Run SAM2 tracking on the filtered candidates
        print("⏳ Running SAM2 tracking...")
        video_segments = sam2_tracking(
            frames_dir=subset_dir,
            predictor=predictor,
            mask_candidates=filtered_per_frame,
            frame_numbers=selected_indices,
            iou_threshold=0.6,
        )
        print("✅ Tracking complete")

        save_video_segments_npz(video_segments, os.path.join(track_dir, 'video_segments.npz'))
        # Save masked images relative to the original frames_dir for consistent names
        store_output_masks(
            track_dir,
            frames_dir,
            video_segments,
            level=level,
            add_label=True,
            frame_lookup=frame_index_to_name,
        )
        save_viz_frames(
            viz_dir,
            frames_dir,
            video_segments,
            level,
            frame_lookup=frame_index_to_name,
        )
        save_instance_maps(viz_dir, video_segments, level)
        # Save comparison images between SemanticSAM proposals and SAM2 results
        save_comparison_proposals(
            viz_dir=viz_dir,
            base_frames_dir=frames_dir,
            filtered_per_frame=filtered_per_frame,
            video_segments=video_segments,
            level=level,
            frame_numbers=selected_indices,
            frames_to_save=None,
        )

    print(f"\n🎉 Done. Outputs at: {out_root}")


if __name__ == '__main__':
    main()
