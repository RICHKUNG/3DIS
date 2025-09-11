"""
Generate per-level Semantic-SAM mask candidates for selected frames.
Runs only the Semantic-SAM part to allow using a different environment (e.g., Semantic-SAM env).
Outputs match the structure expected by run_pipeline/track_from_candidates.
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Tuple

import numpy as np

DEFAULT_SEMANTIC_SAM_ROOT = "/media/Pluto/richkung/Semantic-SAM"
if DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.append(DEFAULT_SEMANTIC_SAM_ROOT)

from semantic_sam import (
    prepare_image,
    build_semantic_sam,
    SemanticSamAutomaticMaskGenerator,
)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def parse_range(range_str: str):
    s, e, st = [int(x) for x in str(range_str).split(':')]
    return s, e, st


def parse_levels(levels_str: str):
    return [int(x) for x in str(levels_str).split(',') if str(x).strip()]


def bbox_from_mask_xyxy(seg: np.ndarray):
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]


def generate_candidates_per_level(
    frames_dir: str,
    selected_frames: List[str],
    semantic_sam,
    level: int,
    min_area: int,
    stability_thresh: float,
):
    gen = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[level])
    per_frame_filtered = []
    raw_dump = []

    for f_idx, fname in enumerate(selected_frames):
        img_path = os.path.join(frames_dir, fname)
        _, tensor_img = prepare_image(img_path)
        masks = gen.generate(tensor_img)

        filtered = []
        for m in masks:
            seg = m.get('segmentation')
            if seg is None:
                continue
            area = int(m.get('area', int(seg.sum())))
            stability = float(m.get('stability_score', 1.0))
            bbox = m.get('bbox')
            if bbox is None:
                bbox = xyxy_to_xywh(bbox_from_mask_xyxy(seg))
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


def main():
    ap = argparse.ArgumentParser(description="Generate Semantic-SAM candidates per level")
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--levels', default='2,4,6')
    ap.add_argument('--frames', default='1200:1600:20')
    ap.add_argument('--sam-ckpt', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--min-area', type=int, default=300)
    ap.add_argument('--stability-threshold', type=float, default=0.9)
    args = ap.parse_args()

    frames_dir = args.data_path
    levels = parse_levels(args.levels)
    s, e, st = parse_range(args.frames)

    all_frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    selected = [all_frames[i] for i in range(s, min(e, len(all_frames)), st)]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    # model
    print("⏳ Loading Semantic-SAM...")
    # Ensure working directory matches expected config relative paths
    try:
        os.chdir(DEFAULT_SEMANTIC_SAM_ROOT)
    except Exception:
        pass
    semantic_sam = build_semantic_sam(model_type="L", ckpt=args.sam_ckpt)
    print("✅ Semantic-SAM ready")

    out_root = ensure_dir(args.output)
    manifest = {
        'mode': 'candidates_only',
        'levels': levels,
        'frames': args.frames,
        'min_area': args.min_area,
        'stability_threshold': args.stability_threshold,
        'data_path': frames_dir,
        'selected_frames': selected,
        'sam_ckpt': args.sam_ckpt,
        'timestamp': int(time.time()),
    }
    with open(os.path.join(out_root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    for level in levels:
        print(f"\n=== Level {level} ===")
        level_root = ensure_dir(os.path.join(out_root, f'level_{level}'))
        cand_dir = ensure_dir(os.path.join(level_root, 'candidates'))
        filt_dir = ensure_dir(os.path.join(level_root, 'filtered'))
        ensure_dir(os.path.join(level_root, 'viz'))

        filtered_per_frame, raw_dump = generate_candidates_per_level(
            frames_dir=frames_dir,
            selected_frames=selected,
            semantic_sam=semantic_sam,
            level=level,
            min_area=args.min_area,
            stability_thresh=args.stability_threshold,
        )

        # save raw
        with open(os.path.join(cand_dir, 'candidates.json'), 'w') as f:
            json.dump({'items': raw_dump}, f, indent=2)

        # save filtered
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
        with open(os.path.join(filt_dir, 'filtered.json'), 'w') as f:
            json.dump({'frames': filtered_json}, f, indent=2)

    print(f"\n🎉 Candidates saved at: {out_root}")


if __name__ == '__main__':
    main()
