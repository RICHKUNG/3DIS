"""
Generate per-level Semantic-SAM mask candidates for selected frames.
Runs only the Semantic-SAM part to allow using a different environment (e.g., Semantic-SAM env).
Outputs match the structure expected by run_pipeline/track_from_candidates.
"""
if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))



import argparse
import datetime
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from my3dis.common_utils import (
    PACKED_MASK_B64_KEY,
    PACKED_MASK_KEY,
    PACKED_ORIG_SHAPE_KEY,
    PACKED_SHAPE_KEY,
    RAW_DIR_NAME,
    RAW_MASK_TEMPLATE,
    RAW_META_TEMPLATE,
    bbox_from_mask_xyxy,
    bbox_xyxy_to_xywh,
    build_subset_video,
    ensure_dir,
    encode_mask,
    format_duration,
    is_packed_mask,
    downscale_binary_mask,
    numeric_frame_sort_key,
    pack_binary_mask,
    parse_levels,
    parse_range,
    setup_logging,
    unpack_binary_mask,
)
from my3dis.pipeline_defaults import (
    DEFAULT_SEMANTIC_SAM_CKPT as _DEFAULT_SEMANTIC_SAM_CKPT_PATH,
    DEFAULT_SEMANTIC_SAM_ROOT as _DEFAULT_SEMANTIC_SAM_ROOT,
    expand_default,
)

_SEM_ROOT_STR = expand_default(_DEFAULT_SEMANTIC_SAM_ROOT)
if _SEM_ROOT_STR not in sys.path:
    sys.path.append(_SEM_ROOT_STR)

from .ssam_progressive_adapter import generate_with_progressive
from my3dis.raw_archive import ARCHIVE_FORMAT_TAG, RawCandidateArchiveWriter


LOGGER = logging.getLogger("my3dis.generate_candidates")

DEFAULT_SEMANTIC_SAM_CKPT = str(_DEFAULT_SEMANTIC_SAM_CKPT_PATH)


def _coerce_packed_mask(entry: Any) -> Optional[Dict[str, Any]]:
    if entry is None:
        return None

    if is_packed_mask(entry) and PACKED_MASK_KEY in entry:
        payload = dict(entry)
        payload[PACKED_MASK_KEY] = np.ascontiguousarray(
            np.asarray(payload[PACKED_MASK_KEY], dtype=np.uint8)
        )
        shape_entry = payload.get(PACKED_SHAPE_KEY)
        if isinstance(shape_entry, np.ndarray):
            payload[PACKED_SHAPE_KEY] = tuple(int(v) for v in shape_entry.tolist())
        elif isinstance(shape_entry, list):
            payload[PACKED_SHAPE_KEY] = tuple(int(v) for v in shape_entry)
        elif isinstance(shape_entry, tuple):
            payload[PACKED_SHAPE_KEY] = tuple(int(v) for v in shape_entry)
        elif shape_entry is not None:
            payload[PACKED_SHAPE_KEY] = (int(shape_entry),)
        payload.pop(PACKED_MASK_B64_KEY, None)
        return payload

    mask_bool = unpack_binary_mask(entry)
    return pack_binary_mask(mask_bool, full_resolution_shape=mask_bool.shape)


def _mask_to_bool(entry: Any) -> Optional[np.ndarray]:
    if entry is None:
        return None
    if isinstance(entry, np.ndarray):
        return np.asarray(entry, dtype=np.bool_)
    return unpack_binary_mask(entry)


def _coerce_union_shape(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Attempt to coerce a mask to ``target_shape`` for boolean union."""

    arr = np.asarray(mask, dtype=np.bool_)
    if arr.shape == target_shape:
        return arr

    if arr.ndim != 2:
        return None

    target_h, target_w = target_shape
    src_h, src_w = arr.shape

    if min(target_h, target_w, src_h, src_w) <= 0:
        return None

    ratio_h = target_h / src_h
    ratio_w = target_w / src_w

    # Only downscale when ratios are approximately equal and < 1
    if (
        ratio_h > 0.0
        and ratio_w > 0.0
        and ratio_h <= 1.0 + 1e-6
        and ratio_w <= 1.0 + 1e-6
        and abs(ratio_h - ratio_w) <= 0.02
    ):
        ratio = min(1.0, max(ratio_h, ratio_w))
        try:
            resized = downscale_binary_mask(arr, ratio)
        except ValueError:
            return None

        if resized.shape != target_shape:
            resized_h, resized_w = resized.shape
            trimmed = resized[:target_h, :target_w]
            if trimmed.shape == target_shape:
                return np.asarray(trimmed, dtype=np.bool_)
            padded = np.zeros(target_shape, dtype=bool)
            padded[: min(resized_h, target_h), : min(resized_w, target_w)] = resized[
                : min(resized_h, target_h), : min(resized_w, target_w)
            ]
            return padded

        return np.asarray(resized, dtype=np.bool_)

    return None


def persist_raw_frame(
    *,
    level_root: str,
    frame_idx: int,
    frame_name: str,
    candidates: List[Dict[str, Any]],
    chunk_writer: Optional[RawCandidateArchiveWriter] = None,
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
    packed_arrays: List[Optional[np.ndarray]] = []
    has_mask_flags: List[bool] = []

    mask_shape: Optional[Tuple[int, int]] = None
    packed_len: Optional[int] = None
    pending_zero_fill: List[int] = []

    for cand in candidates:
        c_meta = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in cand.items()
            if k != 'segmentation'
        }
        c_meta['raw_index'] = len(metadata_items)
        c_meta.setdefault('frame_idx', frame_idx)
        c_meta.setdefault('frame_name', frame_name)
        metadata_items.append(c_meta)

        seg_entry = cand.get('segmentation')

        array_slot = len(packed_arrays)
        packed_arrays.append(None)

        if seg_entry is None:
            has_mask_flags.append(False)
            pending_zero_fill.append(array_slot)
            continue

        packed_payload = _coerce_packed_mask(seg_entry)
        if packed_payload is None:
            has_mask_flags.append(False)
            pending_zero_fill.append(array_slot)
            cand['segmentation'] = None
            continue

        cand['segmentation'] = packed_payload
        seg_shape = tuple(int(v) for v in packed_payload[PACKED_SHAPE_KEY])
        seg_bits = np.ascontiguousarray(
            np.asarray(packed_payload[PACKED_MASK_KEY], dtype=np.uint8)
        )

        if mask_shape is None:
            mask_shape = seg_shape
            packed_len = seg_bits.size
            zero_template = np.zeros(packed_len, dtype=np.uint8)
            for idx in pending_zero_fill:
                packed_arrays[idx] = zero_template.copy()
            pending_zero_fill.clear()
        elif seg_shape != mask_shape:
            seg_bool = unpack_binary_mask(packed_payload)
            if seg_bool.shape != mask_shape:
                from PIL import Image

                ref_h, ref_w = mask_shape
                seg_img = Image.fromarray((seg_bool.astype(np.uint8) * 255))
                seg_img = seg_img.resize((ref_w, ref_h), resample=Image.NEAREST)
                seg_bool = np.array(seg_img) > 127

            packed_payload = pack_binary_mask(seg_bool, full_resolution_shape=seg_bool.shape)
            cand['segmentation'] = packed_payload
            seg_bits = np.ascontiguousarray(
                np.asarray(packed_payload[PACKED_MASK_KEY], dtype=np.uint8)
            )

        if packed_len is None:
            packed_len = seg_bits.size

        packed_arrays[array_slot] = seg_bits.copy()
        has_mask_flags.append(True)

    if mask_shape is not None and packed_len is not None:
        zero_template = np.zeros(packed_len, dtype=np.uint8)
        for idx in pending_zero_fill:
            packed_arrays[idx] = zero_template.copy()
        pending_zero_fill.clear()

    # 撰寫 JSON metadata
    frame_record = {
        'frame_idx': frame_idx,
        'frame_name': frame_name,
        'candidate_count': len(metadata_items),
        'candidates': metadata_items,
    }
    stored_masks = 0
    mask_bytes: Optional[bytes] = None

    if mask_shape is not None and packed_len is not None:
        packed_matrix = (
            np.stack(packed_arrays, axis=0)
            if packed_arrays
            else np.zeros((0, packed_len), dtype=np.uint8)
        )
        stored_masks = packed_matrix.shape[0]
        mask_buffer = io.BytesIO()
        np.savez_compressed(
            mask_buffer,
            packed_masks=packed_matrix,
            mask_shape=np.asarray(mask_shape, dtype=np.int32),
            has_mask=np.asarray(has_mask_flags, dtype=np.bool_),
        )
        mask_bytes = mask_buffer.getvalue()

    if chunk_writer is None:
        with open(meta_path, 'w') as f:
            json.dump(frame_record, f, indent=2)
        if mask_bytes is not None:
            with open(mask_path, 'wb') as fh:
                fh.write(mask_bytes)
        elif os.path.exists(mask_path):
            os.remove(mask_path)
    else:
        meta_bytes = json.dumps(frame_record, ensure_ascii=False).encode('utf-8')
        chunk_writer.add_frame(
            frame_idx=frame_idx,
            frame_name=frame_name,
            meta_bytes=meta_bytes,
            mask_bytes=mask_bytes,
            candidate_count=len(metadata_items),
            mask_count=stored_masks,
        )
        # 清理 legacy 檔案，避免新舊格式並存
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass
        if os.path.exists(mask_path):
            try:
                os.remove(mask_path)
            except OSError:
                pass

    return {
        'meta_count': len(metadata_items),
        'mask_count': stored_masks,
    }


def configure_logging(explicit_level: Optional[int] = None) -> int:
    """Initialize logging once and return the effective level."""
    level = setup_logging(
        explicit_level=explicit_level,
        logger_names_to_quiet=(
            "utils.model",
            "semantic_sam.utils.model",
        ),
    )
    LOGGER.setLevel(level)
    return level


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
    downscale_masks: bool = False,
    mask_scale_ratio: float = 1.0,
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

    try:
        mask_scale_ratio = float(mask_scale_ratio)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid mask_scale_ratio=%r; defaulting to 1.0", mask_scale_ratio)
        mask_scale_ratio = 1.0
    if not downscale_masks:
        mask_scale_ratio = 1.0
    elif mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
        raise ValueError('mask_scale_ratio must be within (0, 1] when downscale_masks is true')
    if mask_scale_ratio < 1.0:
        LOGGER.info("Downscaling SSAM masks by ratio %.3f before persistence", mask_scale_ratio)

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
    def build_folder_name(run_timestamp: str) -> str:
        # 重要參數
        levels_str = "L" + "_".join(str(x) for x in level_list)
        ssam_str = f"ssam{ssam_freq}"

        # 可選參數
        parts = [run_timestamp, levels_str, ssam_str]

        if sam2_max_propagate is not None:
            parts.append(f"propmax{sam2_max_propagate}")

        if min_area != 300:
            parts.append(f"area{min_area}")
        if fill_area != min_area:
            parts.append(f"fill{fill_area}")

        if add_gaps:
            parts.append("gaps")

        if downscale_masks and mask_scale_ratio < 1.0:
            ratio_str = f"{mask_scale_ratio:.3f}".rstrip('0').rstrip('.')
            parts.append(f"scale{ratio_str}x")

        # 如果有自定義標籤，加到最後
        if experiment_tag:
            parts.append(experiment_tag)

        return "_".join(parts)

    run_timestamp: Optional[str]
    if no_timestamp:
        if experiment_tag:
            run_root = ensure_dir(os.path.join(output, experiment_tag))
        else:
            run_root = ensure_dir(output)
        run_timestamp = None
    else:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = build_folder_name(run_timestamp)
        run_root = ensure_dir(os.path.join(output, folder_name))

    # 建立全量取樣的 subset，供 downstream / tracker 直接使用
    subset_dir, subset_map = build_subset_video(
        frames_dir, selected, selected_indices, run_root
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
        'timestamp': run_timestamp,
        'output_root': run_root,
        'gap_fill': {
            'fill_area': fill_area,
            'add_gaps': bool(add_gaps),
        },
        'mask_scale_ratio': mask_scale_ratio,
        'mask_downscale': {
            'enabled': bool(downscale_masks) and mask_scale_ratio < 1.0,
            'ratio': mask_scale_ratio,
        },
        'filtering': {
            'applied': not skip_filtering,
            'min_area': None if skip_filtering else min_area,
            'stability_threshold': stability_threshold,
        },
        'raw_storage': {
            'enabled': bool(persist_raw),
            'format': ARCHIVE_FORMAT_TAG if persist_raw else None,
            'dir_name': RAW_DIR_NAME if persist_raw else None,
            'archives': {},
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

    run_manifest_path = os.path.join(run_root, 'manifest.json')
    LOGGER.info(
        "Cached %d sampled frames at %s (SSAM cadence=%d → %d frames)",
        len(selected),
        subset_dir,
        ssam_freq,
        len(ssam_frames),
    )

    # 只對選定的幀進行 Semantic-SAM 分割，逐幀處理以避免一次佔用全部記憶體
    progressive_iter = generate_with_progressive(
        frames_dir=frames_dir,
        selected_frames=ssam_frames,
        sam_ckpt_path=sam_ckpt,
        levels=level_list,
        min_area=min_area,
        fill_area=fill_area,
        enable_gap_fill=bool(add_gaps),
        mask_scale_ratio=mask_scale_ratio,
    )

    level_infos: List[Dict[str, Any]] = []
    for level_idx, level in enumerate(level_list):
        level_root = ensure_dir(os.path.join(run_root, f'level_{level}'))
        cand_dir = ensure_dir(os.path.join(level_root, 'candidates'))
        filt_dir = ensure_dir(os.path.join(level_root, 'filtered'))
        ensure_dir(os.path.join(level_root, 'viz'))
        raw_writer = RawCandidateArchiveWriter(level_root) if persist_raw else None
        info: Dict[str, Any] = {
            'level': level,
            'index': level_idx,
            'level_root': level_root,
            'cand_dir': cand_dir,
            'filt_dir': filt_dir,
            'raw_items': [],
            'filtered_frames': [] if not skip_filtering else None,
            'raw_total': 0,
            'filtered_total': 0,
            'frame_count': 0,
            'raw_writer': raw_writer,
            'raw_manifest': None,
        }
        level_infos.append(info)

    for rel_idx, fname, level_payload in progressive_iter:
        ssam_frame_idx = int(ssam_absolute_indices[rel_idx])
        frame_name = ssam_frames[rel_idx]

        for info in level_infos:
            level = info['level']
            level_add_gaps = bool(add_gaps) and info['index'] == 0
            candidates_src = level_payload.get(level, [])
            if candidates_src is None:
                candidates = []
            else:
                candidates = list(candidates_src)

            info['frame_count'] += 1

            if level_add_gaps and candidates:
                first_seg_entry = candidates[0].get('segmentation')
                first_mask = _mask_to_bool(first_seg_entry)
                gap = None
                gap_area = 0
                ratio_hint = float(candidates[0].get('mask_scale_ratio', mask_scale_ratio))
                ratio_hint = max(0.0, min(1.0, ratio_hint)) or 1.0
                ratio_sq = ratio_hint * ratio_hint
                scaled_fill_area = fill_area
                if ratio_hint < 1.0:
                    scaled_fill_area = max(1, int(round(fill_area * ratio_sq)))
                if first_mask is not None:
                    H, W = first_mask.shape
                    mask_stack: List[np.ndarray] = []
                    for m in candidates:
                        seg_arr = _mask_to_bool(m.get('segmentation'))
                        if seg_arr is None:
                            continue
                        if seg_arr.shape != first_mask.shape:
                            coerced = _coerce_union_shape(seg_arr, first_mask.shape)
                            if coerced is None:
                                LOGGER.warning(
                                    "Gap-fill union skipped mask for frame %s due to shape %s (expected %s)",
                                    frame_name,
                                    tuple(seg_arr.shape),
                                    first_mask.shape,
                                )
                                continue
                            seg_arr = coerced
                        mask_stack.append(np.asarray(seg_arr, dtype=np.bool_))
                    if mask_stack:
                        mask_matrix = np.empty((len(mask_stack), H, W), dtype=np.bool_)
                        for idx, seg_arr in enumerate(mask_stack):
                            mask_matrix[idx] = seg_arr
                        union = np.any(mask_matrix, axis=0)
                    else:
                        union = np.zeros((H, W), dtype=np.bool_)
                    gap = np.logical_not(union)
                    gap_area = int(gap.sum())
                if gap is not None and gap_area >= scaled_fill_area:
                    ys, xs = np.where(gap)
                    x1, y1, x2, y2 = (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    )
                    bbox = bbox_xyxy_to_xywh((x1, y1, x2, y2))
                    approx_area = gap_area
                    if ratio_hint < 1.0 and ratio_sq > 0:
                        approx_area = int(round(gap_area / ratio_sq))
                    full_shape = None
                    if isinstance(first_seg_entry, dict) and PACKED_ORIG_SHAPE_KEY in first_seg_entry:
                        orig_shape = first_seg_entry[PACKED_ORIG_SHAPE_KEY]
                        if isinstance(orig_shape, np.ndarray):
                            full_shape = tuple(int(v) for v in orig_shape.flatten().tolist())
                        elif isinstance(orig_shape, (list, tuple)):
                            full_shape = tuple(int(v) for v in orig_shape)
                        elif orig_shape is not None:
                            full_shape = (int(orig_shape), int(orig_shape))
                    if full_shape is None:
                        if ratio_hint < 1.0:
                            full_shape = (
                                max(1, int(round(H / ratio_hint))),
                                max(1, int(round(W / ratio_hint))),
                            )
                        else:
                            full_shape = (int(H), int(W))
                    candidates.append({
                        'frame_idx': ssam_frame_idx,
                        'frame_name': f"gap_{ssam_frame_idx:05d}",
                        'bbox': bbox,
                        'area': approx_area,
                        'stability_score': 1.0,
                        'level': level,
                        'mask_scale_ratio': ratio_hint,
                        'segmentation': pack_binary_mask(gap, full_resolution_shape=full_shape),
                    })

            for m in candidates:
                m['frame_idx'] = ssam_frame_idx
                m.setdefault('frame_name', frame_name)

            info['raw_total'] += len(candidates)

            if persist_raw:
                persist_raw_frame(
                    level_root=info['level_root'],
                    frame_idx=ssam_frame_idx,
                    frame_name=frame_name,
                    candidates=candidates,
                    chunk_writer=info.get('raw_writer'),
                )

            for m in candidates:
                info['raw_items'].append({
                    k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in m.items()
                    if k != 'segmentation'
                })

            if skip_filtering:
                candidates.clear()
                continue

            meta_list: List[Dict[str, Any]] = []
            filtered_local_index = 0
            for m in candidates:
                stability = float(m.get('stability_score', 1.0))
                area = int(m.get('area', 0))
                if area < min_area or stability < stability_threshold:
                    continue

                seg_data = _mask_to_bool(m.get('segmentation'))
                if seg_data is None:
                    continue

                meta = {k: v for k, v in m.items() if k != 'segmentation'}
                meta['id'] = filtered_local_index
                meta['mask'] = encode_mask(seg_data)
                meta_list.append(meta)
                filtered_local_index += 1

            info['filtered_total'] += len(meta_list)
            if info['filtered_frames'] is not None:
                info['filtered_frames'].append(
                    {
                        'frame_idx': ssam_frame_idx,
                        'frame_name': frame_name,
                        'count': len(meta_list),
                        'items': meta_list,
                    }
                )

            candidates.clear()

        for cand_list in level_payload.values():
            if isinstance(cand_list, list):
                cand_list.clear()
        level_payload.clear()

    level_stats: List[Dict[str, Any]] = []
    for info in level_infos:
        level = info['level']
        cand_dir = info['cand_dir']
        filt_dir = info['filt_dir']
        writer: Optional[RawCandidateArchiveWriter] = info.get('raw_writer')
        if writer is not None:
            raw_manifest_path = writer.close()
            if raw_manifest_path is not None:
                info['raw_manifest'] = str(raw_manifest_path)

        with open(os.path.join(cand_dir, 'candidates.json'), 'w') as f:
            json.dump({'items': info['raw_items']}, f, indent=2)

        if not skip_filtering and info['filtered_frames'] is not None:
            with open(os.path.join(filt_dir, 'filtered.json'), 'w') as f:
                json.dump({'frames': info['filtered_frames']}, f, indent=2)

        level_stats.append(
            {
                'level': level,
                'frame_count': info['frame_count'],
                'raw_candidates': info['raw_total'],
                'filtered_masks': info['filtered_total'],
                'filtering_applied': not skip_filtering,
            }
        )

    raw_manifest_entries: Dict[str, str] = {}
    for info in level_infos:
        raw_manifest_path = info.get('raw_manifest')
        if raw_manifest_path:
            try:
                rel_path = os.path.relpath(raw_manifest_path, info['level_root'])
            except (KeyError, TypeError, ValueError):
                rel_path = raw_manifest_path
            raw_manifest_entries[str(info['level'])] = rel_path

    manifest['generation_summary'] = level_stats
    if persist_raw:
        storage_meta = manifest.setdefault('raw_storage', {})
        storage_meta.update(
            {
                'enabled': bool(raw_manifest_entries),
                'format': ARCHIVE_FORMAT_TAG,
                'dir_name': RAW_DIR_NAME,
                'archives': raw_manifest_entries,
            }
        )
    else:
        manifest['raw_storage'] = {'enabled': False}
    if raw_manifest_entries:
        manifest['raw_archives'] = raw_manifest_entries
    with open(run_manifest_path, 'w') as f:
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
        format_duration(time.perf_counter() - start_time),
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
    ap.add_argument('--downscale-masks', action='store_true',
                    help='Downscale SSAM masks before persistence')
    ap.add_argument('--mask-scale-ratio', type=float, default=1.0,
                    help='Scale ratio applied when --downscale-masks is provided (0 < r ≤ 1)')
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
        downscale_masks=args.downscale_masks,
        mask_scale_ratio=args.mask_scale_ratio,
    )


if __name__ == '__main__':
    main()
