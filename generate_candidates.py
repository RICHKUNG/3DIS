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
import base64
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

RAW_DIR_NAME = "raw"
RAW_META_TEMPLATE = "frame_{frame_idx:05d}.json"
RAW_MASK_TEMPLATE = "frame_{frame_idx:05d}.npz"

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
    parts = str(range_str).split(':')
    if len(parts) != 3:
        raise ValueError(f'Invalid range spec: {range_str!r}')
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else -1
    step = int(parts[2]) if parts[2] else 1
    if step <= 0:
        raise ValueError('step must be positive')
    return start, end, step


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


def encode_mask(mask: np.ndarray) -> Dict[str, Any]:
    """Serialize a boolean mask into a JSON-friendly packed representation."""
    bool_mask = np.asarray(mask, dtype=np.bool_, order='C')
    packed = np.packbits(bool_mask.reshape(-1))
    return {
        'shape': [int(dim) for dim in bool_mask.shape],
        'packed_bits_b64': base64.b64encode(packed.tobytes()).decode('ascii'),
    }


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def numeric_frame_sort_key(fname: str) -> Tuple[float, str]:
    """Ensure frames iterate in numerical order when filenames mix padding."""
    stem, _ = os.path.splitext(fname)
    match = re.search(r'\d+', stem)
    if match:
        try:
            return float(int(match.group())), fname
        except ValueError:
            pass
    return float('inf'), fname


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
        if os.path.lexists(dst):
            try:
                if os.path.samefile(src, dst):
                    continue
            except FileNotFoundError:
                os.unlink(dst)
            except OSError:
                os.unlink(dst)
            else:
                os.unlink(dst)
        try:
            os.symlink(src, dst)
        except OSError:
            from shutil import copy2, SameFileError

            try:
                copy2(src, dst)
            except SameFileError:
                # Already linked/copied from a previous run; nothing to do.
                continue
    return subset_dir, index_to_subset


def persist_raw_frame(
    *,
    level_root: str,
    frame_idx: int,
    frame_name: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Persist raw Semantic-SAM candidates for a given frame.

    Stores compact metadata (without segmentation masks) alongside an NPZ file
    containing the boolean mask stack so later stages can re-apply filtering
    without re-running Semantic-SAM.
    """

    raw_dir = ensure_dir(os.path.join(level_root, RAW_DIR_NAME))
    meta_path = os.path.join(raw_dir, RAW_META_TEMPLATE.format(frame_idx=frame_idx))
    mask_path = os.path.join(raw_dir, RAW_MASK_TEMPLATE.format(frame_idx=frame_idx))

    metadata_items: List[Dict[str, Any]] = []
    mask_arrays: List[Optional[np.ndarray]] = []
    has_mask_flags: List[bool] = []

    mask_shape = None
    pending_zero_fill: List[int] = []

    for local_idx, cand in enumerate(candidates):
        c_meta = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in cand.items()
            if k != 'segmentation'
        }
        c_meta['raw_index'] = local_idx
        c_meta.setdefault('frame_idx', frame_idx)
        c_meta.setdefault('frame_name', frame_name)
        metadata_items.append(c_meta)

        seg = cand.get('segmentation')
        if seg is None:
            has_mask_flags.append(False)
            if mask_shape is None:
                mask_arrays.append(None)
                pending_zero_fill.append(local_idx)
            else:
                mask_arrays.append(np.zeros(mask_shape, dtype=np.bool_))
            continue

        seg_arr = np.asarray(seg, dtype=np.bool_)
        if mask_shape is None:
            mask_shape = seg_arr.shape
            for idx in pending_zero_fill:
                mask_arrays[idx] = np.zeros(mask_shape, dtype=np.bool_)
            pending_zero_fill.clear()
        elif seg_arr.shape != mask_shape:
            # 如果尺寸不一致，將其重新調整至第一個遮罩的尺寸
            from PIL import Image

            ref_h, ref_w = mask_shape
            seg_img = Image.fromarray((seg_arr.astype(np.uint8) * 255))
            seg_img = seg_img.resize((ref_w, ref_h), resample=Image.NEAREST)
            seg_arr = np.array(seg_img) > 127

        mask_arrays.append(seg_arr.astype(np.bool_))
        has_mask_flags.append(True)

    if mask_shape is not None and pending_zero_fill:
        for idx in pending_zero_fill:
            mask_arrays[idx] = np.zeros(mask_shape, dtype=np.bool_)

    # 撰寫 JSON metadata
    frame_record = {
        'frame_idx': frame_idx,
        'frame_name': frame_name,
        'candidate_count': len(metadata_items),
        'candidates': metadata_items,
    }
    with open(meta_path, 'w') as f:
        json.dump(frame_record, f, indent=2)

    stored_masks = 0
    if mask_shape is not None:
        mask_stack = np.stack(mask_arrays, axis=0) if mask_arrays else np.zeros((0, *mask_shape), dtype=np.bool_)
        np.savez_compressed(
            mask_path,
            masks=mask_stack.astype(np.bool_),
            has_mask=np.asarray(has_mask_flags, dtype=np.bool_),
        )
        stored_masks = mask_stack.shape[0]
    elif os.path.exists(mask_path):
        os.remove(mask_path)

    return {
        'meta_count': len(metadata_items),
        'mask_count': stored_masks,
    }


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
    fill_area: Optional[int] = None,
    stability_threshold: float = 0.9,
    add_gaps: bool = False,
    no_timestamp: bool = False,
    log_level: Optional[int] = None,
    ssam_freq: int = 1,
    sam2_max_propagate: int = None,
    experiment_tag: str = None,
    persist_raw: bool = False,
    skip_filtering: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Generate candidates and persist them in the standard layout.

    Returns (run_dir, manifest_dict).
    """

    configure_logging(log_level)
    start_time = time.perf_counter()

    frames_dir = data_path
    if fill_area is None:
        fill_area = min_area
    try:
        fill_area = int(fill_area)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid fill_area=%r; defaulting to min_area=%d", fill_area, min_area)
        fill_area = int(min_area)
    fill_area = max(0, fill_area)
    level_list = parse_levels(levels)
    start_idx, end_idx, step = parse_range(frames)
    start_idx = max(0, start_idx)

    try:
        ssam_freq = max(1, int(ssam_freq))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid ssam_freq=%r; defaulting to 1", ssam_freq)
        ssam_freq = 1

    all_frames = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=numeric_frame_sort_key,
    )
    if end_idx < 0:
        range_end = len(all_frames)
    else:
        range_end = min(end_idx, len(all_frames))
    selected_indices = list(range(start_idx, range_end, step))
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

    # 建立自動化的資料夾名稱，包含重要參數
    def build_folder_name():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 重要參數
        levels_str = "L" + "_".join(str(x) for x in level_list)
        ssam_str = f"ssam{ssam_freq}"
        
        # 可選參數
        parts = [timestamp, levels_str, ssam_str]
        
        if sam2_max_propagate is not None:
            parts.append(f"propmax{sam2_max_propagate}")

        if min_area != 300:
            parts.append(f"area{min_area}")
        if fill_area != min_area:
            parts.append(f"fill{fill_area}")

        if add_gaps:
            parts.append("gaps")

        # 如果有自定義標籤，加到最後
        if experiment_tag:
            parts.append(experiment_tag)

        return "_".join(parts)

    if no_timestamp:
        if experiment_tag:
            run_root = ensure_dir(os.path.join(output, experiment_tag))
        else:
            run_root = ensure_dir(output)
        timestamp = None
    else:
        folder_name = build_folder_name()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = ensure_dir(os.path.join(output, folder_name))

    # 只為有 SSAM 分割的幀建立 subset
    subset_dir, subset_map = build_subset_video(
        frames_dir, ssam_frames, ssam_absolute_indices, run_root
    )
    frames_path = Path(frames_dir).expanduser()
    try:
        frames_path = frames_path.resolve()
    except FileNotFoundError:
        frames_path = frames_path.absolute()

    scene_name = None
    scene_root = None
    dataset_root = None
    for parent in [frames_path] + list(frames_path.parents):
        name = parent.name
        if name.startswith('scene_'):
            scene_name = name
            scene_root = parent
            dataset_root = parent.parent
            break

    manifest = {
        'mode': 'candidates_only',
        'levels': level_list,
        'frames': frames,
        'min_area': min_area,
        'fill_area': fill_area,
        'stability_threshold': stability_threshold,
        'data_path': frames_dir,
        'selected_frames': selected,
        'selected_indices': selected_indices,
        'ssam_frames': ssam_frames,
        'ssam_indices': ssam_local_indices,
        'ssam_absolute_indices': ssam_absolute_indices,
        'ssam_freq': ssam_freq,
        'sam2_max_propagate': sam2_max_propagate,
        'experiment_tag': experiment_tag,
        'subset_dir': subset_dir,
        'subset_map': subset_map,
        'sam_ckpt': sam_ckpt,
        'ts_epoch': int(time.time()),
        'timestamp': timestamp,
        'output_root': run_root,
        'gap_fill': {
            'fill_area': fill_area,
            'add_gaps': bool(add_gaps),
        },
        'filtering': {
            'applied': not skip_filtering,
            'min_area': None if skip_filtering else min_area,
            'stability_threshold': stability_threshold,
        },
        'raw_storage': {
            'enabled': bool(persist_raw),
            'format': 'frame_npz_v1',
            'dir_name': RAW_DIR_NAME if persist_raw else None,
        },
    }

    if scene_name:
        manifest['scene'] = scene_name
        if scene_root:
            manifest['scene_dir'] = str(scene_root)
        if dataset_root:
            manifest['dataset_root'] = str(dataset_root)

    manifest['frames_total'] = len(all_frames)
    manifest['frames_selected'] = len(selected)
    manifest['frames_ssam'] = len(ssam_frames)

    manifest_path = os.path.join(run_root, 'manifest.json')
    LOGGER.info("Selected %d SSAM frames cached at %s", len(ssam_frames), subset_dir)

    # 只對選定的幀進行 Semantic-SAM 分割
    per_level = generate_with_progressive(
        frames_dir=frames_dir,
        selected_frames=ssam_frames,
        sam_ckpt_path=sam_ckpt,
        levels=level_list,
        min_area=min_area,
        fill_area=fill_area,
        enable_gap_fill=bool(add_gaps),
    )

    level_stats: List[Dict[str, Any]] = []
    for level_idx, level in enumerate(level_list):
        level_root = ensure_dir(os.path.join(run_root, f'level_{level}'))
        cand_dir = ensure_dir(os.path.join(level_root, 'candidates'))
        filt_dir = ensure_dir(os.path.join(level_root, 'filtered'))
        ensure_dir(os.path.join(level_root, 'viz'))

        # 只讓第一個 level 執行 gap 填補（add_gaps=True 時）
        level_add_gaps = bool(add_gaps) and level_idx == 0

        raw_items: List[Dict[str, Any]] = []
        filtered_json: List[Dict[str, Any]] = [] if not skip_filtering else []
        per_frame_list = per_level[level]

        raw_candidate_total = 0
        filtered_total = 0

        for f_idx, lst in enumerate(per_frame_list):
            candidates = list(lst)
            ssam_frame_idx = int(ssam_absolute_indices[f_idx])
            frame_name = ssam_frames[f_idx]

            if level_add_gaps and candidates:
                H, W = candidates[0]['segmentation'].shape
                union = np.zeros((H, W), dtype=bool)
                for m in candidates:
                    union |= m['segmentation']
                gap = ~union
                gap_area = int(gap.sum())
                if gap_area >= fill_area:
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

            for m in candidates:
                m['frame_idx'] = ssam_frame_idx
                m.setdefault('frame_name', frame_name)

            raw_candidate_total += len(candidates)

            if persist_raw:
                persist_raw_frame(
                    level_root=level_root,
                    frame_idx=ssam_frame_idx,
                    frame_name=frame_name,
                    candidates=candidates,
                )

            for m in candidates:
                raw_items.append({
                    k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in m.items()
                    if k != 'segmentation'
                })

            if skip_filtering:
                continue

            meta_list = []
            filtered_local_index = 0
            for m in candidates:
                stability = float(m.get('stability_score', 1.0))
                area = int(m.get('area', 0))
                if area < min_area or stability < stability_threshold:
                    continue

                seg_data = m.get('segmentation')
                if seg_data is None:
                    continue

                meta = {k: v for k, v in m.items() if k != 'segmentation'}
                meta['id'] = filtered_local_index
                meta['mask'] = encode_mask(seg_data)
                meta_list.append(meta)
                filtered_local_index += 1

            filtered_json.append(
                {
                    'frame_idx': ssam_frame_idx,
                    'frame_name': frame_name,
                    'count': len(meta_list),
                    'items': meta_list,
                }
            )
            filtered_total += len(meta_list)

        with open(os.path.join(cand_dir, 'candidates.json'), 'w') as f:
            json.dump({'items': raw_items}, f, indent=2)

        if not skip_filtering:
            with open(os.path.join(filt_dir, 'filtered.json'), 'w') as f:
                json.dump({'frames': filtered_json}, f, indent=2)

        level_stats.append(
            {
                'level': level,
                'frame_count': len(per_frame_list),
                'raw_candidates': raw_candidate_total,
                'filtered_masks': filtered_total,
                'filtering_applied': not skip_filtering,
            }
        )

    manifest['generation_summary'] = level_stats
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    if level_stats:
        summary_parts = []
        for entry in level_stats:
            lvl = entry['level']
            raw_count = entry['raw_candidates']
            filtered_count = entry['filtered_masks']
            frame_count = entry['frame_count']
            if skip_filtering:
                summary_parts.append(f"L{lvl}: raw {raw_count} (frames {frame_count})")
            else:
                summary_parts.append(
                    f"L{lvl}: raw {raw_count} → filtered {filtered_count} (frames {frame_count})"
                )
        LOGGER.info("Persisted levels → %s", "; ".join(summary_parts))
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
    ap.add_argument('--fill-area', type=int, default=None,
                    help='Minimum area for SSAM gap-fill masks (default: min-area)')
    ap.add_argument('--stability-threshold', type=float, default=0.9)
    ap.add_argument('--add-gaps', action='store_true', help='Add uncovered area as a candidate per frame per level')
    ap.add_argument('--no-timestamp', action='store_true', help='Do not append a timestamp folder to output root')
    ap.add_argument('--ssam-freq', type=int, default=1, 
                    help='Run Semantic-SAM every N frames (default: 1, means every frame)')
    ap.add_argument('--sam2-max-propagate', type=int, default=None,
                    help='Maximum number of frames to propagate in each direction for SAM2 (default: no limit)')
    ap.add_argument('--experiment-tag', type=str, default=None,
                    help='Custom tag to append to timestamp for experiment identification')
    ap.add_argument('--persist-raw', action='store_true',
                    help='Store raw mask stacks for re-filtering later (default: disabled)')
    ap.add_argument('--skip-filtering', action='store_true',
                    help='Skip immediate filtering so it can be done as a separate stage')
    args = ap.parse_args()

    run_generation(
        data_path=args.data_path,
        levels=args.levels,
        frames=args.frames,
        sam_ckpt=args.sam_ckpt,
        output=args.output,
        min_area=args.min_area,
        fill_area=args.fill_area,
        stability_threshold=args.stability_threshold,
        add_gaps=args.add_gaps,
        no_timestamp=args.no_timestamp,
        ssam_freq=args.ssam_freq,
        sam2_max_propagate=args.sam2_max_propagate,
        experiment_tag=args.experiment_tag,
        persist_raw=args.persist_raw,
        skip_filtering=args.skip_filtering,
    )


if __name__ == '__main__':
    main()
