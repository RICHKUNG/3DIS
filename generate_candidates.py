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
import datetime
import logging
from typing import List, Dict, Any, Tuple

import numpy as np

DEFAULT_SEMANTIC_SAM_ROOT = "/media/Pluto/richkung/Semantic-SAM"
if DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.append(DEFAULT_SEMANTIC_SAM_ROOT)

from ssam_progressive_adapter import generate_with_progressive, ensure_dir


LOGGER = logging.getLogger("my3dis.generate_candidates")

DEFAULT_SEMANTIC_SAM_CKPT = os.path.join(
    DEFAULT_SEMANTIC_SAM_ROOT,
    "checkpoints",
    "swinl_only_sam_many2many.pth",
)


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


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_subset_video(
    frames_dir: str,
    selected: List[str],
    selected_indices: List[int],
    out_root: str,
    folder_name: str = "selected_frames",
):
    subset_dir = os.path.join(out_root, folder_name)
    os.makedirs(subset_dir, exist_ok=True)
    index_to_subset = {}
    for local_idx, (abs_idx, fname) in enumerate(zip(selected_indices, selected)):
        src = os.path.join(frames_dir, fname)
        dst_name = f"{local_idx:06d}.jpg"
        dst = os.path.join(subset_dir, dst_name)
        index_to_subset[abs_idx] = dst_name
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                from shutil import copy2
                copy2(src, dst)
    return subset_dir, index_to_subset

def main():
    ap = argparse.ArgumentParser(description="Generate Semantic-SAM candidates per level")
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--levels', default='2,4,6')
    ap.add_argument('--frames', default='1200:1600:20')
    ap.add_argument('--sam-ckpt', default=DEFAULT_SEMANTIC_SAM_CKPT,
                    help='Semantic-SAM checkpoint path (default: swinl_only_sam_many2many.pth)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-area', type=int, default=300)
    ap.add_argument('--stability-threshold', type=float, default=0.9)
    ap.add_argument('--add-gaps', action='store_true', help='Add uncovered area as a candidate per frame per level')
    ap.add_argument('--no-timestamp', action='store_true', help='Do not append a timestamp folder to output root')
    args = ap.parse_args()

    log_level_name = os.environ.get("MY3DIS_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")
    LOGGER.setLevel(log_level)
    # Silence verbose checkpoint load logs from Semantic-SAM internals.
    logging.getLogger("utils.model").setLevel(logging.WARNING)
    logging.getLogger("semantic_sam.utils.model").setLevel(logging.WARNING)

    start_time = time.perf_counter()

    frames_dir = args.data_path
    levels = parse_levels(args.levels)
    s, e, st = parse_range(args.frames)

    all_frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    selected_indices = list(range(s, min(e, len(all_frames)), st))
    selected = [all_frames[i] for i in selected_indices]

    LOGGER.info(
        "Semantic-SAM candidate generation started (levels=%s, frames=%s)",
        args.levels,
        args.frames,
    )

    # Timestamped experiment folder
    if args.no_timestamp:
        out_root = ensure_dir(args.output)
        ts = None
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = ensure_dir(os.path.join(args.output, ts))
    subset_dir, subset_map = build_subset_video(frames_dir, selected, selected_indices, out_root)
    manifest = {
        'mode': 'candidates_only',
        'levels': levels,
        'frames': args.frames,
        'min_area': args.min_area,
        'stability_threshold': args.stability_threshold,
        'data_path': frames_dir,
        'selected_frames': selected,
        'selected_indices': selected_indices,
        'subset_dir': subset_dir,
        'subset_map': subset_map,
        'sam_ckpt': args.sam_ckpt,
        'ts_epoch': int(time.time()),
        'timestamp': ts,
        'output_root': out_root,
    }
    with open(os.path.join(out_root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    LOGGER.info("Selected %d frames cached at %s", len(selected), subset_dir)

    # Run progressive refinement once per frame and collect per-level results
    per_level = generate_with_progressive(
        frames_dir=frames_dir,
        selected_frames=selected,
        sam_ckpt_path=args.sam_ckpt,
        levels=levels,
        min_area=args.min_area,
        save_root=os.path.join(out_root, '_progressive_tmp'),
    )

    # Persist in our standard layout (candidates + filtered are identical here
    # since progressive_refinement already applies area filtering).
    level_stats = []
    for level in levels:
        level_root = ensure_dir(os.path.join(out_root, f'level_{level}'))
        cand_dir = ensure_dir(os.path.join(level_root, 'candidates'))
        filt_dir = ensure_dir(os.path.join(level_root, 'filtered'))
        ensure_dir(os.path.join(level_root, 'viz'))

        # Flatten for raw dump without heavy masks
        raw_items = []
        filtered_json = []
        per_frame_list = per_level[level]
        frame_count = len(per_frame_list)
        total_masks = 0
        for f_idx, lst in enumerate(per_frame_list):
            # Optionally add uncovered area as a candidate
            if args.add_gaps:
                if lst:
                    H, W = lst[0]['segmentation'].shape
                    union = np.zeros((H, W), dtype=bool)
                    for m in lst:
                        union |= m['segmentation']
                    gap = ~union
                    gap_area = int(gap.sum())
                    if gap_area >= args.min_area:
                        ys, xs = np.where(gap)
                        x1, y1, x2, y2 = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        lst.append({
                            'frame_idx': f_idx,
                            'frame_name': f"gap_{f_idx:05d}",
                            'bbox': bbox,
                            'area': gap_area,
                            'stability_score': 1.0,
                            'level': level,
                            'segmentation': gap,
                        })
            meta_list = []
            seg_stack = []
            for j, m in enumerate(lst):
                raw_items.append({k: (v.tolist() if hasattr(v, 'tolist') else v)
                                  for k, v in m.items() if k != 'segmentation'})
                meta = {k: v for k, v in m.items() if k != 'segmentation'}
                meta['id'] = j
                meta_list.append(meta)
                seg_stack.append(m['segmentation'].astype(np.uint8))
            filtered_json.append({'frame_idx': f_idx, 'count': len(meta_list), 'items': meta_list})
            if seg_stack:
                np.save(os.path.join(filt_dir, f'seg_frame_{f_idx:05d}.npy'), np.stack(seg_stack, axis=0))
            total_masks += len(meta_list)

        with open(os.path.join(cand_dir, 'candidates.json'), 'w') as f:
            json.dump({'items': raw_items}, f, indent=2)
        with open(os.path.join(filt_dir, 'filtered.json'), 'w') as f:
            json.dump({'frames': filtered_json}, f, indent=2)

        level_stats.append((level, frame_count, total_masks))

    if level_stats:
        summary = "; ".join(
            f"L{lvl}: {masks} masks across {frames} frames"
            for lvl, frames, masks in level_stats
        )
        LOGGER.info("Persisted levels → %s", summary)
    LOGGER.info("Candidates saved at %s", out_root)
    LOGGER.info(
        "Candidate generation finished in %s",
        format_seconds(time.perf_counter() - start_time),
    )


if __name__ == '__main__':
    main()
