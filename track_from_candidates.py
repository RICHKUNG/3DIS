"""
Tracking-only stage: read filtered candidates saved by generate_candidates.py
and run SAM2 masklet propagation. Designed to run in a SAM2-capable env.
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any

import numpy as np

DEFAULT_SAM2_ROOT = "/media/Pluto/richkung/SAM2"
if DEFAULT_SAM2_ROOT not in sys.path:
    sys.path.append(DEFAULT_SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor

import torch


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


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
    if mask1.shape == mask2.shape:
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(inter) / float(union) if union else 0.0
    m1 = torch.from_numpy(mask1.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2 = torch.from_numpy(mask2.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2r = torch.nn.functional.interpolate(m2, size=m1.shape[-2:], mode='bilinear', align_corners=False)
    m2r = (m2r > 0.5).squeeze()
    m1 = m1.squeeze()
    inter = torch.logical_and(m1, m2r).sum().item()
    union = torch.logical_or(m1, m2r).sum().item()
    return float(inter) / float(union) if union else 0.0


def build_subset_video(frames_dir: str, selected: List[str], out_root: str) -> str:
    subset_dir = os.path.join(out_root, "_video_subset")
    os.makedirs(subset_dir, exist_ok=True)
    for i, fname in enumerate(selected):
        src = os.path.join(frames_dir, fname)
        # SAM2 image loader expects .jpg/.jpeg. Use .jpg extension for symlink/copy.
        dst = os.path.join(subset_dir, f"{i:06d}.jpg")
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                from shutil import copy2
                copy2(src, dst)
    return subset_dir


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


def sam2_tracking(frames_dir: str, predictor, mask_candidates: List[List[Dict[str, Any]]], iou_threshold=0.6):
    with torch.inference_mode(), torch.autocast("cuda"):
        obj_count = 1
        final_video_segments: Dict[int, Dict[int, np.ndarray]] = {}

        # init state
        state = predictor.init_state(video_path=frames_dir)
        # get scaling
        h0 = mask_candidates[0][0]['segmentation'].shape[0]
        w0 = mask_candidates[0][0]['segmentation'].shape[1]
        sx = state['video_width'] / w0
        sy = state['video_height'] / h0

        for frame_idx, frame_masks in enumerate(mask_candidates):
            state = predictor.init_state(video_path=frames_dir)
            if frame_idx > 0:
                untracked = []
                prev = final_video_segments.get(frame_idx, {})  # subset indices are contiguous
                prev_masks = list(prev.values())
                for m in frame_masks:
                    seg = m.get('segmentation')
                    if seg is None:
                        continue
                    tracked = any(compute_iou(pm, seg) > iou_threshold for pm in prev_masks)
                    if not tracked:
                        untracked.append(m)
                bboxes_xywh = [m['bbox'] for m in untracked]
            else:
                bboxes_xywh = [m['bbox'] for m in frame_masks]
            bboxes_xyxy = bbox_transform_xywh_to_xyxy([list(bb) for bb in bboxes_xywh if bb is not None])
            bboxes_xyxy = bbox_scalar_fit(bboxes_xyxy, sx, sy)

            for bbox in bboxes_xyxy:
                _ , _out_ids, _out_logits = predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=frame_idx, obj_id=obj_count, box=bbox
                )
                obj_count += 1

            segs = {}
            for out_fidx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                segs[out_fidx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            for fidx, frame_data in segs.items():
                if fidx not in final_video_segments:
                    final_video_segments[fidx] = {}
                final_video_segments[fidx].update(frame_data)

        return final_video_segments


def save_video_segments_npz(segments: Dict[int, Dict[int, np.ndarray]], path: str) -> None:
    packed = {str(k): {str(kk): vv.astype(np.uint8) for kk, vv in v.items()} for k, v in segments.items()}
    np.savez_compressed(path, data=packed)


def store_output_masks(output_root: str, frames_dir: str, video_segments: Dict[int, Dict[int, np.ndarray]]):
    os.makedirs(output_root, exist_ok=True)
    frame_filenames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    all_obj_ids = set()
    for frame_masks in video_segments.values():
        all_obj_ids.update(int(k) for k in frame_masks.keys())
    for oid in all_obj_ids:
        ensure_dir(os.path.join(output_root, 'objects', str(oid)))
    for frame_idx, frame_name in enumerate(frame_filenames):
        if frame_idx not in video_segments:
            continue
        img_path = os.path.join(frames_dir, frame_name)
        img = np.array((__import__('PIL').Image.open(img_path).convert('RGB')))
        frame_masks = video_segments[frame_idx]
        for obj_id, mask in frame_masks.items():
            bool_mask = np.squeeze(mask > 0)
            masked_arr = img * bool_mask[:, :, np.newaxis]
            out_img = __import__('PIL').Image.fromarray(masked_arr.astype(np.uint8))
            out_path = os.path.join(output_root, 'objects', str(obj_id), os.path.splitext(frame_name)[0] + '.png')
            ensure_dir(os.path.dirname(out_path))
            out_img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="SAM2 tracking from pre-generated candidates")
    ap.add_argument('--data-path', required=True, help='Original frames dir')
    ap.add_argument('--candidates-root', required=True, help='Root containing level_*/filtered')
    ap.add_argument('--sam2-cfg', required=True)
    ap.add_argument('--sam2-ckpt', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--levels', default='2,4,6')
    args = ap.parse_args()

    # Load manifest to get selected frames (for subset)
    with open(os.path.join(args.candidates_root, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    selected = manifest.get('selected_frames', [])

    # Ensure working dir for relative Hydra config paths
    try:
        os.chdir(DEFAULT_SAM2_ROOT)
    except Exception:
        pass
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt)

    out_root = ensure_dir(args.output)
    subset_dir = build_subset_video(args.data_path, selected, out_root)

    levels = [int(x) for x in str(args.levels).split(',') if str(x).strip()]
    for level in levels:
        print(f"\n=== Tracking level {level} ===")
        level_root = os.path.join(args.candidates_root, f'level_{level}')
        track_dir = ensure_dir(os.path.join(out_root, f'level_{level}', 'tracking'))
        per_frame = load_filtered_candidates(level_root)
        segs = sam2_tracking(subset_dir, predictor, per_frame, iou_threshold=0.6)
        save_video_segments_npz(segs, os.path.join(track_dir, 'video_segments.npz'))
        store_output_masks(track_dir, subset_dir, segs)

    print(f"\n🎉 Tracking results saved at: {out_root}")


if __name__ == '__main__':
    main()
