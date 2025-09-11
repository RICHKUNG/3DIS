"""
Bridge pipeline: Semantic-SAM (multi-level) → SAM2 tracking (Algorithm 1 variant)

This script follows My3DIS/Agent.md:
- Levels fixed to [2,4,6] unless overridden
- Frame range like 1200:1600:20 (end exclusive)
- Demo can be limited by --max-frames 3
- Saves per-level candidate lists (raw and filtered), tracking results and simple visualizations

Note: This file does not execute on import. Use the CLI at bottom.
"""

import os
import sys
import json
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

# --- Local repo paths (non-destructive) ---
DEFAULT_SEMANTIC_SAM_ROOT = "/media/Pluto/richkung/Semantic-SAM"
DEFAULT_SAM2_ROOT = "/media/Pluto/richkung/SAM2"

if DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.append(DEFAULT_SEMANTIC_SAM_ROOT)
if DEFAULT_SAM2_ROOT not in sys.path:
    sys.path.append(DEFAULT_SAM2_ROOT)

from semantic_sam import (
    prepare_image,
    build_semantic_sam,
    SemanticSamAutomaticMaskGenerator,
)

from auto_generation_inference import instance_map_to_anns  # for optional conversions

from sam2.build_sam import build_sam2_video_predictor

import torch


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
    freq: int,
    iou_threshold: float = 0.6,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Run Algorithm 1 variant using SAM2 video predictor.

    mask_candidates: list over sampled-frame-index -> list of mask dicts (with 'bbox' XYWH, 'segmentation', ...)
    freq: stride between sampled frames so that ann_frame_idx = frame_idx * freq
    Returns: final per-frame dict: frame_idx -> {obj_id: mask_bool_array}
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

        for frame_idx, frame_masks in enumerate(mask_candidates):
            # re-init state so each iteration prompts fresh boxes
            inference_state = predictor.init_state(video_path=frames_dir)

            # Determine bboxes for this iteration (untracked only for frame>0)
            if frame_idx > 0:
                untracked = []
                prev_frame_abs = frame_idx * freq
                prev_masks_map = final_video_segments.get(prev_frame_abs, {})
                prev_masks = list(prev_masks_map.values())
                for m in frame_masks:
                    seg = m.get("segmentation")
                    if seg is None:
                        continue
                    tracked = any(compute_iou(pm, seg) > iou_threshold for pm in prev_masks)
                    if not tracked:
                        untracked.append(m)
                bboxes_xywh = [m.get('bbox') for m in untracked]
            else:
                bboxes_xywh = [m.get('bbox') for m in frame_masks]

            # Convert to xyxy then scale to SAM2 size
            bboxes_xyxy = bbox_transform_xywh_to_xyxy([list(bb) for bb in bboxes_xywh if bb is not None])
            bboxes_xyxy = bbox_scalar_fit(bboxes_xyxy, scalar_x, scalar_y)

            # Prompt SAM2 with each box
            for bbox in bboxes_xyxy:
                ann_frame_idx = frame_idx * freq
                ann_obj_id = obj_count
                obj_count += 1
                # add prompt
                _ , _out_ids, _out_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=bbox,
                )

            # Propagate in video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Merge segments from this iteration
            for fidx, frame_data in video_segments.items():
                if fidx not in final_video_segments:
                    final_video_segments[fidx] = {}
                final_video_segments[fidx].update(frame_data)

        return final_video_segments


# ---------------------
# Semantic-SAM per-level candidate generation
# ---------------------
def generate_candidates_per_level(
    frames_dir: str,
    selected_frames: List[str],
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


def store_output_masks(output_root: str, frames_dir: str, video_segments: Dict[int, Dict[int, np.ndarray]]):
    os.makedirs(output_root, exist_ok=True)
    frame_filenames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])

    # Find all unique object ids
    all_obj_ids = set()
    for frame_masks in video_segments.values():
        all_obj_ids.update(int(k) for k in frame_masks.keys())

    # Create folders
    for oid in all_obj_ids:
        ensure_dir(os.path.join(output_root, 'objects', str(oid)))

    # Save masked images for all frames
    for frame_idx, frame_name in enumerate(frame_filenames):
        if frame_idx not in video_segments:
            continue
        img_path = os.path.join(frames_dir, frame_name)
        img = Image.open(img_path).convert('RGB')
        img_arr = np.array(img)
        frame_masks = video_segments[frame_idx]
        for obj_id, mask in frame_masks.items():
            bool_mask = np.squeeze(mask > 0)
            masked_arr = img_arr * bool_mask[:, :, np.newaxis]
            out_img = Image.fromarray(masked_arr.astype(np.uint8))
            out_path = os.path.join(output_root, 'objects', str(obj_id), os.path.splitext(frame_name)[0] + '.png')
            ensure_dir(os.path.dirname(out_path))
            out_img.save(out_path)


# ---------------------
# Main CLI
# ---------------------
def build_subset_video(frames_dir: str, selected: List[str], out_root: str) -> str:
    """Create a lightweight video directory containing only selected frames.
    Uses symlinks when possible; falls back to copies.
    Returns the path to the subset directory.
    """
    subset_dir = os.path.join(out_root, "_video_subset")
    os.makedirs(subset_dir, exist_ok=True)
    for i, fname in enumerate(selected):
        src = os.path.join(frames_dir, fname)
        dst = os.path.join(subset_dir, f"{i:06d}.jpg")
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                # fallback to copy
                from shutil import copy2
                copy2(src, dst)
    return subset_dir


def main():
    ap = argparse.ArgumentParser(description="Semantic-SAM multi-level → SAM2 tracking pipeline")
    ap.add_argument('--data-path', required=True, help='Folder containing all frames (JPG/PNG)')
    ap.add_argument('--levels', default='2,4,6', help='Comma-separated levels, default 2,4,6')
    ap.add_argument('--frames', default='1200:1600:20', help='Range as start:end:step (end exclusive)')
    ap.add_argument('--sam-ckpt', required=True)
    ap.add_argument('--sam2-cfg', required=True)
    ap.add_argument('--sam2-ckpt', required=True)
    ap.add_argument('--output', required=True, help='Output root inside My3DIS')
    ap.add_argument('--max-frames', type=int, default=None, help='For demo, limit number of sampled frames')
    ap.add_argument('--min-area', type=int, default=300)
    ap.add_argument('--stability-threshold', type=float, default=0.9)
    args = ap.parse_args()

    frames_dir = args.data_path
    start, end, step = parse_range(args.frames)
    # Default: restrict SAM2 to a tiny subset for demo by re-building a small 'video' with only selected frames.
    # When using subset, freq is 1 since indices are local to the subset.
    subset_video = True if args.max_frames is not None else False
    levels = parse_levels(args.levels)

    # Resolve frames
    all_frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    selected = [all_frames[i] for i in range(start, min(end, len(all_frames)), step)]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    # Load models
    print("⏳ Loading Semantic-SAM...")
    semantic_sam = build_semantic_sam(model_type="L", ckpt=args.sam_ckpt)
    print("✅ Semantic-SAM ready")
    print("⏳ Loading SAM2 video predictor...")
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt)
    print("✅ SAM2 ready")

    # Output structure
    out_root = ensure_dir(args.output)
    manifest = {
        'levels': levels,
        'frames': args.frames,
        'min_area': args.min_area,
        'stability_threshold': args.stability_threshold,
        'data_path': frames_dir,
        'selected_frames': selected,
        'sam_ckpt': args.sam_ckpt,
        'sam2_cfg': args.sam2_cfg,
        'sam2_ckpt': args.sam2_ckpt,
        'timestamp': int(time.time()),
    }
    save_json(manifest, os.path.join(out_root, 'manifest.json'))

    # If subset enabled, build a minimal video directory for SAM2
    video_path_for_sam2 = frames_dir
    freq = step
    if subset_video:
        video_path_for_sam2 = build_subset_video(frames_dir, selected, out_root)
        freq = 1

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
            frames_dir=video_path_for_sam2,
            predictor=predictor,
            mask_candidates=filtered_per_frame,
            freq=freq,
            iou_threshold=0.6,
        )
        print("✅ Tracking complete")

        save_video_segments_npz(video_segments, os.path.join(track_dir, 'video_segments.npz'))
        # Save masked images relative to the original frames_dir for consistent names
        store_output_masks(track_dir, frames_dir, video_segments)

    print(f"\n🎉 Done. Outputs at: {out_root}")


if __name__ == '__main__':
    main()
