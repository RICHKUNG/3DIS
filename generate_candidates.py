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
from typing import List, Dict, Any, Tuple, Optional, Union

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


def configure_logging(explicit_level: Optional[int] = None) -> int:
    """Initialize logging once and return the effective level."""
    if explicit_level is None:
        log_level_name = os.environ.get("MY3DIS_LOG_LEVEL", "INFO").upper()
        explicit_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=explicit_level, format="%(message)s")
    root_logger.setLevel(explicit_level)
    LOGGER.setLevel(explicit_level)
    logging.getLogger("utils.model").setLevel(logging.WARNING)
    logging.getLogger("semantic_sam.utils.model").setLevel(logging.WARNING)
    return explicit_level


def run_generation(
    *,
    data_path: str,
    levels: Union[str, List[int]] = "2,4,6",
    frames: str = "1200:1600:20",
    sam_ckpt: str = DEFAULT_SEMANTIC_SAM_CKPT,
    output: str,
    min_area: int = 300,
    stability_threshold: float = 0.9,
    add_gaps: bool = False,
    no_timestamp: bool = False,
    log_level: Optional[int] = None,
    ssam_freq: int = 1,
    sam2_max_propagate: int = None,
) -> Tuple[str, Dict[str, Any]]:
    """Generate candidates and persist them in the standard layout.

    Returns (run_dir, manifest_dict).
    """

    configure_logging(log_level)
    start_time = time.perf_counter()

    frames_dir = data_path
    level_list = parse_levels(levels)
    start_idx, end_idx, step = parse_range(frames)

    try:
        ssam_freq = max(1, int(ssam_freq))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid ssam_freq=%r; defaulting to 1", ssam_freq)
        ssam_freq = 1

    all_frames = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    selected_indices = list(range(start_idx, min(end_idx, len(all_frames)), step))
    selected = [all_frames[i] for i in selected_indices]

    # 選擇需要進行 Semantic-SAM 分割的幀（按 ssam_freq 間隔）
    ssam_local_indices = list(range(0, len(selected), ssam_freq))
    ssam_frames = [selected[i] for i in ssam_local_indices]
    ssam_absolute_indices = [selected_indices[i] for i in ssam_local_indices]

    LOGGER.info(
        "Semantic-SAM candidate generation started (levels=%s, frames=%s, ssam_freq=%d)",
        ",".join(str(x) for x in level_list),
        frames,
        ssam_freq,
    )
    LOGGER.info(
        "Will run Semantic-SAM on %d frames (every %d frames from %d selected frames)",
        len(ssam_frames),
        ssam_freq,
        len(selected),
    )

    if no_timestamp:
        run_root = ensure_dir(output)
        timestamp = None
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = ensure_dir(os.path.join(output, timestamp))

    # 只為有 SSAM 分割的幀建立 subset
    subset_dir, subset_map = build_subset_video(
        frames_dir, ssam_frames, ssam_absolute_indices, run_root
    )
    manifest = {
        'mode': 'candidates_only',
        'levels': level_list,
        'frames': frames,
        'min_area': min_area,
        'stability_threshold': stability_threshold,
        'data_path': frames_dir,
        'selected_frames': selected,
        'selected_indices': selected_indices,
        'ssam_frames': ssam_frames,
        'ssam_indices': ssam_local_indices,
        'ssam_absolute_indices': ssam_absolute_indices,
        'ssam_freq': ssam_freq,
        'sam2_max_propagate': sam2_max_propagate,
        'subset_dir': subset_dir,
        'subset_map': subset_map,
        'sam_ckpt': sam_ckpt,
        'ts_epoch': int(time.time()),
        'timestamp': timestamp,
        'output_root': run_root,
    }
    with open(os.path.join(run_root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    LOGGER.info("Selected %d SSAM frames cached at %s", len(ssam_frames), subset_dir)

    # 只對選定的幀進行 Semantic-SAM 分割
    per_level = generate_with_progressive(
        frames_dir=frames_dir,
        selected_frames=ssam_frames,
        sam_ckpt_path=sam_ckpt,
        levels=level_list,
        min_area=min_area,
        save_root=os.path.join(run_root, '_progressive_tmp'),
    )

    level_stats = []
    for level in level_list:
        level_root = ensure_dir(os.path.join(run_root, f'level_{level}'))
        cand_dir = ensure_dir(os.path.join(level_root, 'candidates'))
        filt_dir = ensure_dir(os.path.join(level_root, 'filtered'))
        ensure_dir(os.path.join(level_root, 'viz'))

        raw_items: List[Dict[str, Any]] = []
        filtered_json: List[Dict[str, Any]] = []
        per_frame_list = per_level[level]
        frame_count = len(per_frame_list)
        total_masks = 0

        # 只處理有 SSAM 分割的幀
        for f_idx, lst in enumerate(per_frame_list):
            candidates = list(lst)
            # 記錄這是第幾個 SSAM 處理的幀
            ssam_frame_idx = ssam_absolute_indices[f_idx]

            if add_gaps and candidates:
                H, W = candidates[0]['segmentation'].shape
                union = np.zeros((H, W), dtype=bool)
                for m in candidates:
                    union |= m['segmentation']
                gap = ~union
                gap_area = int(gap.sum())
                if gap_area >= min_area:
                    ys, xs = np.where(gap)
                    x1, y1, x2, y2 = (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    )
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    candidates.append({
                        'frame_idx': ssam_frame_idx,
                        'frame_name': f"gap_{ssam_frame_idx:05d}",
                        'bbox': bbox,
                        'area': gap_area,
                        'stability_score': 1.0,
                        'level': level,
                        'segmentation': gap,
                    })

            meta_list = []
            seg_stack = []
            filtered_local_index = 0
            for m in candidates:
                # 更新 frame_idx 為在 selected 中的索引
                m['frame_idx'] = ssam_frame_idx
                
                # Persist raw metadata without heavy segmentation masks.
                raw_items.append({
                    k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in m.items()
                    if k != 'segmentation'
                })

                stability = float(m.get('stability_score', 1.0))
                area = int(m.get('area', 0))
                if area < min_area or stability < stability_threshold:
                    continue

                meta = {k: v for k, v in m.items() if k != 'segmentation'}
                meta['id'] = filtered_local_index
                meta_list.append(meta)
                seg_stack.append(m['segmentation'].astype(np.uint8))
                filtered_local_index += 1

            filtered_json.append({'frame_idx': ssam_frame_idx, 'count': len(meta_list), 'items': meta_list})
            if seg_stack:
                np.save(
                    os.path.join(filt_dir, f'seg_frame_{ssam_frame_idx:05d}.npy'),
                    np.stack(seg_stack, axis=0),
                )
            total_masks += len(meta_list)

        with open(os.path.join(cand_dir, 'candidates.json'), 'w') as f:
            json.dump({'items': raw_items}, f, indent=2)
        with open(os.path.join(filt_dir, 'filtered.json'), 'w') as f:
            json.dump({'frames': filtered_json}, f, indent=2)

        level_stats.append((level, frame_count, total_masks))

    if level_stats:
        summary = "; ".join(
            f"L{lvl}: {masks} masks across {frames} SSAM frames"
            for lvl, frames, masks in level_stats
        )
        LOGGER.info("Persisted levels → %s", summary)
    LOGGER.info("Candidates saved at %s", run_root)
    LOGGER.info(
        "Candidate generation finished in %s",
        format_seconds(time.perf_counter() - start_time),
    )

    return run_root, manifest


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
    ap.add_argument('--ssam-freq', type=int, default=1, 
                    help='Run Semantic-SAM every N frames (default: 1, means every frame)')
    ap.add_argument('--sam2-max-propagate', type=int, default=None,
                    help='Maximum number of frames to propagate in each direction for SAM2 (default: no limit)')
    args = ap.parse_args()

    run_generation(
        data_path=args.data_path,
        levels=args.levels,
        frames=args.frames,
        sam_ckpt=args.sam_ckpt,
        output=args.output,
        min_area=args.min_area,
        stability_threshold=args.stability_threshold,
        add_gaps=args.add_gaps,
        no_timestamp=args.no_timestamp,
        ssam_freq=args.ssam_freq,
        sam2_max_propagate=args.sam2_max_propagate,
    )


if __name__ == '__main__':
    main()
