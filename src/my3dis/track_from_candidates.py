"""
Tracking-only stage: read filtered candidates saved by generate_candidates.py
and run SAM2 masklet propagation. Designed to run in a SAM2-capable env.
"""
if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))



import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from my3dis.common_utils import (
    build_subset_video,
    downscale_binary_mask,
    ensure_dir,
    numeric_frame_sort_key,
    pack_binary_mask,
    setup_logging,
    unpack_binary_mask,
)
from my3dis.pipeline_defaults import (
    DEFAULT_SAM2_CFG as _DEFAULT_SAM2_CFG_PATH,
    DEFAULT_SAM2_CKPT as _DEFAULT_SAM2_CKPT_PATH,
    DEFAULT_SAM2_ROOT as _DEFAULT_SAM2_ROOT,
    expand_default,
)

_SAM2_ROOT_STR = expand_default(_DEFAULT_SAM2_ROOT)
if _SAM2_ROOT_STR not in sys.path:
    sys.path.insert(0, _SAM2_ROOT_STR)

from sam2.build_sam import build_sam2_video_predictor

from my3dis.tracking import (
    ProgressPrinter,
    TimingAggregator,
    bbox_scalar_fit,
    bbox_transform_xywh_to_xyxy,
    format_duration_precise,
    infer_relative_scale,
    reorganize_segments_by_object,
    resize_mask_to_shape,
    save_comparison_proposals,
    save_object_segments_npz,
    save_video_segments_npz,
)

LOGGER = logging.getLogger("my3dis.track_from_candidates")


@dataclass
class PromptCandidate:
    """Derived metadata attached to a filtered SSAM prompt candidate."""

    payload: Dict[str, Any]
    seg_prompt: Optional[np.ndarray]
    seg_for_iou: Optional[np.ndarray]
    area: Optional[int]
    bbox_xywh: Optional[Tuple[float, float, float, float]]


def _coerce_mask_bool(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if isinstance(mask, dict):
        mask = unpack_binary_mask(mask)
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        arr = arr.astype(np.bool_)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    return arr


def _prepare_prompt_candidates(frame_masks: List[Dict[str, Any]]) -> List[PromptCandidate]:
    prepared: List[PromptCandidate] = []
    for item in frame_masks:
        seg_prompt = _coerce_mask_bool(item.get('segmentation'))
        seg_for_iou = item.get('segmentation_scaled')
        seg_for_iou = _coerce_mask_bool(seg_for_iou) if seg_for_iou is not None else seg_prompt

        area_raw = item.get('area')
        if area_raw is None:
            bbox_val = item.get('bbox')
            if bbox_val is not None and len(bbox_val) == 4:
                area_raw = int(bbox_val[2]) * int(bbox_val[3])
        try:
            area_val = int(area_raw) if area_raw is not None else None
        except (TypeError, ValueError):
            area_val = None

        bbox = item.get('bbox')
        bbox_xywh: Optional[Tuple[float, float, float, float]]
        if bbox is None or len(bbox) != 4:
            bbox_xywh = None
        else:
            bbox_xywh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        prepared.append(
            PromptCandidate(
                payload=item,
                seg_prompt=seg_prompt,
                seg_for_iou=seg_for_iou,
                area=area_val,
                bbox_xywh=bbox_xywh,
            )
        )
    return prepared


def _build_existing_mask_stack(frame_masks: Dict[int, Any]) -> Optional[np.ndarray]:
    if not frame_masks:
        return None
    masks: List[np.ndarray] = []
    target_shape: Optional[Tuple[int, int]] = None
    for packed in frame_masks.values():
        arr = unpack_binary_mask(packed)
        arr = np.asarray(arr, dtype=np.bool_)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            continue
        if target_shape is None:
            target_shape = arr.shape
        elif arr.shape != target_shape:
            arr = resize_mask_to_shape(arr, target_shape)
        masks.append(arr)
    if not masks:
        return None
    return np.stack(masks, axis=0)


def _max_iou_with_stack(existing_stack: np.ndarray, candidate: np.ndarray) -> float:
    if existing_stack.size == 0:
        return 0.0
    target_shape = existing_stack.shape[1:]
    if candidate.shape != target_shape:
        candidate = resize_mask_to_shape(candidate, target_shape)
    cand = candidate.astype(np.bool_)
    stack = existing_stack.astype(np.bool_)
    inter = np.logical_and(stack, cand).sum(axis=(1, 2))
    union = np.logical_or(stack, cand).sum(axis=(1, 2))
    valid = union > 0
    if not np.any(valid):
        return 0.0
    ious = inter[valid] / union[valid]
    return float(ious.max()) if ious.size else 0.0


def _filter_new_candidates(
    candidates: List[PromptCandidate],
    existing_masks: Dict[int, Any],
    iou_threshold: float,
) -> List[PromptCandidate]:
    existing_stack = _build_existing_mask_stack(existing_masks)
    if existing_stack is not None:
        stack = existing_stack.copy()
    else:
        stack = None

    accepted: List[PromptCandidate] = []
    for cand in candidates:
        seg_for_iou = cand.seg_for_iou
        if seg_for_iou is not None and stack is not None:
            overlap = _max_iou_with_stack(stack, seg_for_iou)
            if overlap > iou_threshold:
                continue
        accepted.append(cand)
        if cand.seg_for_iou is not None:
            seg_bool = cand.seg_for_iou.astype(np.bool_)
            if stack is None:
                stack = seg_bool[None, ...]
            else:
                if seg_bool.shape != stack.shape[1:]:
                    seg_bool = resize_mask_to_shape(seg_bool, stack.shape[1:])
                stack = np.concatenate([stack, seg_bool[None, ...]], axis=0)
    return accepted


def _should_use_box_prompt(
    candidate: PromptCandidate,
    *,
    use_box_for_all: bool,
    use_box_for_small: bool,
    small_object_area_threshold: Optional[int],
) -> bool:
    if use_box_for_all:
        return True
    if candidate.seg_prompt is None:
        return True
    if not use_box_for_small or small_object_area_threshold is None:
        return False
    if candidate.area is None:
        return False
    try:
        return int(candidate.area) <= int(small_object_area_threshold)
    except (TypeError, ValueError):
        return False


def _add_prompts_to_predictor(
    predictor,
    state,
    frame_idx: int,
    candidates: List[PromptCandidate],
    *,
    obj_start: int,
    scale_x: float,
    scale_y: float,
    use_box_for_all: bool,
    use_box_for_small: bool,
    small_object_area_threshold: Optional[int],
) -> int:
    obj_id = obj_start
    for cand in candidates:
        if cand.seg_prompt is not None and not _should_use_box_prompt(
            cand,
            use_box_for_all=use_box_for_all,
            use_box_for_small=use_box_for_small,
            small_object_area_threshold=small_object_area_threshold,
        ):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=np.asarray(cand.seg_prompt, dtype=bool),
            )
            obj_id += 1
            continue
        bbox_xywh = cand.bbox_xywh
        if bbox_xywh is None:
            seg_prompt = cand.seg_prompt
            if seg_prompt is None:
                continue
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=np.asarray(seg_prompt, dtype=bool),
            )
        else:
            xyxy = bbox_transform_xywh_to_xyxy([list(bbox_xywh)])[0]
            xyxy = bbox_scalar_fit([xyxy], scale_x, scale_y)[0]
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=xyxy,
            )
        obj_id += 1
    return obj_id


def _propagate_frame_predictions(
    predictor,
    state,
    frame_idx: int,
    *,
    local_to_abs: Dict[int, int],
    total_frames: int,
    max_propagate: Optional[int],
    mask_scale_ratio: float,
) -> Dict[int, Dict[int, Any]]:
    segs: Dict[int, Dict[int, Any]] = {}

    def collect(iterator) -> None:
        for out_fidx, out_obj_ids, out_mask_logits in iterator:
            abs_out_idx = local_to_abs.get(out_fidx)
            if abs_out_idx is None:
                continue
            frame_data: Dict[int, Any] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_arr = (out_mask_logits[i] > 0.0).cpu().numpy()
                orig_shape = mask_arr.shape
                if mask_scale_ratio < 1.0:
                    mask_arr = downscale_binary_mask(mask_arr, mask_scale_ratio)
                    full_shape: Optional[Tuple[int, int]] = (orig_shape[0], orig_shape[1])
                else:
                    full_shape = None
                frame_data[int(out_obj_id)] = pack_binary_mask(
                    mask_arr, full_resolution_shape=full_shape
                )
            if frame_data:
                segs.setdefault(abs_out_idx, {}).update(frame_data)

    forward_budget = max(0, total_frames - frame_idx - 1)
    backward_budget = max(0, frame_idx)
    if max_propagate is not None:
        forward_budget = min(forward_budget, max_propagate)
        backward_budget = min(backward_budget, max_propagate)

    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):
        if forward_budget > 0:
            forward_kwargs = {'start_frame_idx': frame_idx, 'reverse': False}
            if max_propagate is not None:
                forward_kwargs['max_frame_num_to_track'] = forward_budget
            collect(predictor.propagate_in_video(state, **forward_kwargs))

        if backward_budget > 0:
            backward_kwargs = {'start_frame_idx': frame_idx, 'reverse': True}
            if max_propagate is not None:
                backward_kwargs['max_frame_num_to_track'] = backward_budget
            collect(predictor.propagate_in_video(state, **backward_kwargs))

    return segs


DEFAULT_SAM2_ROOT = _SAM2_ROOT_STR
DEFAULT_SAM2_CFG = str(_DEFAULT_SAM2_CFG_PATH)
DEFAULT_SAM2_CKPT = str(_DEFAULT_SAM2_CKPT_PATH)

def resolve_sam2_config_path(config_arg: str) -> str:
    cfg_path = os.path.expanduser(config_arg)
    if os.path.isfile(cfg_path):
        base = os.path.join(DEFAULT_SAM2_ROOT, 'sam2')
        rel = os.path.relpath(cfg_path, base)
        rel = rel.replace(os.sep, '/')
        if rel.endswith('.yaml'):
            rel = rel[:-5]
        return rel
def configure_logging(explicit_level: Optional[int] = None) -> int:
    level = setup_logging(explicit_level=explicit_level)
    LOGGER.setLevel(level)
    return level


def load_filtered_candidates(
    level_root: str,
    *,
    mask_scale_ratio: float = 1.0,
) -> Tuple[List[List[Dict[str, Any]]], List[int], List[str]]:
    """加載篩選後的候選項，返回候選項列表、幀索引與對應檔名"""
    filt_dir = os.path.join(level_root, 'filtered')
    with open(os.path.join(filt_dir, 'filtered.json'), 'r') as f:
        meta = json.load(f)
    frames_meta = meta.get('frames', [])
    per_frame = []
    frame_indices = []
    frame_names: List[str] = []
    
    for fm in frames_meta:
        fidx = fm['frame_idx']
        fname = fm.get('frame_name')
        if fname is None:
            fname = f"{int(fidx):05d}.png"
        frame_names.append(str(fname))
        items = fm['items']
        seg_path = os.path.join(filt_dir, f'seg_frame_{fidx:05d}.npy')
        seg_stack = None
        if os.path.exists(seg_path):
            seg_stack = np.load(seg_path, mmap_mode='r')
        lst = []
        for j, it in enumerate(items):
            d = dict(it)
            mask_payload = d.pop('mask', None)
            seg = None
            ratio_hint = None
            if mask_payload is not None:
                seg = unpack_binary_mask(mask_payload)
                ratio_hint = infer_relative_scale(mask_payload)
            elif seg_stack is not None and j < seg_stack.shape[0]:
                seg = seg_stack[j]
            seg_scaled = None
            if seg is not None:
                seg = np.asarray(seg, dtype=np.bool_)
                if mask_scale_ratio < 1.0:
                    eps = 1e-6
                    effective_ratio = ratio_hint or 1.0
                    if effective_ratio <= mask_scale_ratio + eps:
                        seg_scaled = seg
                    else:
                        relative_ratio = mask_scale_ratio / effective_ratio if effective_ratio else 0.0
                        if 0.0 < relative_ratio < 1.0 - eps:
                            seg_scaled = downscale_binary_mask(seg, relative_ratio)
                        else:
                            seg_scaled = seg
            d['segmentation'] = seg
            if mask_scale_ratio < 1.0:
                d['segmentation_scaled'] = seg_scaled if seg_scaled is not None else seg
            lst.append(d)
        per_frame.append(lst)
        frame_indices.append(fidx)
    
    return per_frame, frame_indices, frame_names


def sam2_tracking(
    frames_dir: str,
    predictor,
    mask_candidates: List[List[Dict[str, Any]]],
    frame_numbers: List[int],
    iou_threshold: float = 0.6,
    max_propagate: Optional[int] = None,
    use_box_for_small: bool = False,
    use_box_for_all: bool = False,
    small_object_area_threshold: Optional[int] = None,
    mask_scale_ratio: float = 1.0,
) -> Dict[int, Dict[int, Any]]:
    os.environ['TQDM_DISABLE'] = '1'
    try:  # pragma: no cover - optional dependency
        import tqdm

        tqdm.tqdm.disable = True
    except ImportError:  # pragma: no cover - tqdm not installed
        pass

    with torch.inference_mode(), torch.autocast("cuda"):
        final_video_segments: Dict[int, Dict[int, Any]] = {}
        state = predictor.init_state(video_path=frames_dir)

        first_seg = None
        for frame_masks in mask_candidates:
            for item in frame_masks:
                candidate_seg = item.get('segmentation')
                if isinstance(candidate_seg, np.ndarray):
                    first_seg = candidate_seg
                    break
            if first_seg is not None:
                break
        if first_seg is None:
            raise RuntimeError('No segmentation masks available to derive resolution')

        h0, w0 = first_seg.shape[:2]
        sx = state['video_width'] / max(1, w0)
        sy = state['video_height'] / max(1, h0)

        local_to_abs = {i: frame_numbers[i] for i in range(len(frame_numbers))}
        total_frames = len(frame_numbers)
        progress = ProgressPrinter(total_frames)

        if max_propagate is not None:
            try:
                max_propagate = max(0, int(max_propagate))
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Invalid max_propagate=%r supplied; disabling propagation limit",
                    max_propagate,
                )
                max_propagate = None

        if small_object_area_threshold is not None:
            try:
                small_object_area_threshold = int(small_object_area_threshold)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Invalid small_object_area_threshold=%r; disabling long-tail box mode",
                    small_object_area_threshold,
                )
                small_object_area_threshold = None

        obj_count = 1
        try:
            for frame_idx, raw_candidates in enumerate(mask_candidates):
                predictor.reset_state(state)
                abs_idx = local_to_abs.get(frame_idx)
                progress.update(frame_idx, abs_idx)
                if abs_idx is None:
                    continue

                prepared_candidates = _prepare_prompt_candidates(raw_candidates)
                existing_masks = final_video_segments.get(abs_idx, {})
                filtered_candidates = _filter_new_candidates(
                    prepared_candidates,
                    existing_masks,
                    iou_threshold=iou_threshold,
                )
                if not filtered_candidates:
                    continue

                obj_count = _add_prompts_to_predictor(
                    predictor,
                    state,
                    frame_idx,
                    filtered_candidates,
                    obj_start=obj_count,
                    scale_x=sx,
                    scale_y=sy,
                    use_box_for_all=use_box_for_all,
                    use_box_for_small=use_box_for_small,
                    small_object_area_threshold=small_object_area_threshold,
                )

                frame_segments = _propagate_frame_predictions(
                    predictor,
                    state,
                    frame_idx,
                    local_to_abs=local_to_abs,
                    total_frames=total_frames,
                    max_propagate=max_propagate,
                    mask_scale_ratio=mask_scale_ratio,
                )

                for abs_out_idx, frame_data in frame_segments.items():
                    final_video_segments.setdefault(abs_out_idx, {}).update(frame_data)
        finally:
            progress.close()

    return final_video_segments


def run_tracking(
    *,
    data_path: str,
    candidates_root: str,
    output: str,
    levels: Union[str, List[int]] = "2,4,6",
    sam2_cfg: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CFG,
    sam2_ckpt: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CKPT,
    sam2_max_propagate: Optional[int] = None,
    log_level: Optional[int] = None,
    iou_threshold: float = 0.6,
    long_tail_box_prompt: bool = False,
    all_box_prompt: bool = False,
    mask_scale_ratio: float = 1.0,
    render_viz: bool = True,
) -> str:
    if not sam2_cfg:
        sam2_cfg = DEFAULT_SAM2_CFG
    if not sam2_ckpt:
        sam2_ckpt = DEFAULT_SAM2_CKPT
    sam2_cfg = os.fspath(sam2_cfg) if isinstance(sam2_cfg, os.PathLike) else sam2_cfg
    sam2_ckpt = os.fspath(sam2_ckpt) if isinstance(sam2_ckpt, os.PathLike) else sam2_ckpt

    configure_logging(log_level)

    try:
        mask_scale_ratio = float(mask_scale_ratio)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid mask_scale_ratio={mask_scale_ratio!r}')
    if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
        raise ValueError('mask_scale_ratio must be within (0, 1]')

    overall_start = time.perf_counter()
    if isinstance(levels, str):
        level_list = [int(x) for x in levels.split(',') if x.strip()]
    else:
        level_list = [int(x) for x in levels]

    LOGGER.info("SAM2 tracking started (levels=%s)", ",".join(str(x) for x in level_list))

    with open(os.path.join(candidates_root, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    
    # 讀取相關參數
    selected = manifest.get('selected_frames', [])
    selected_indices = manifest.get('selected_indices')
    ssam_frames = manifest.get('ssam_frames', selected)  # 有 SSAM 分割的幀
    ssam_absolute_indices = manifest.get('ssam_absolute_indices', selected_indices)
    ssam_freq = manifest.get('ssam_freq', 1)
    manifest_max_propagate = manifest.get('sam2_max_propagate')
    
    if selected_indices is None:
        selected_indices = list(range(len(selected)))
    else:
        selected_indices = [int(x) for x in selected_indices]
    
    if ssam_absolute_indices is None:
        ssam_absolute_indices = selected_indices
    else:
        ssam_absolute_indices = [int(x) for x in ssam_absolute_indices]
        
    subset_dir_manifest = manifest.get('subset_dir')

    if sam2_max_propagate is None:
        sam2_max_propagate = manifest_max_propagate
    if sam2_max_propagate is not None:
        try:
            sam2_max_propagate = max(0, int(sam2_max_propagate))
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid sam2_max_propagate=%r; defaulting to unlimited",
                sam2_max_propagate,
            )
            sam2_max_propagate = None

    LOGGER.info(
        "Configuration: ssam_freq=%d, sam2_max_propagate=%s, ssam_frames=%d, iou_threshold=%.2f, mask_scale_ratio=%.3f, render_viz=%s",
        ssam_freq,
        sam2_max_propagate if sam2_max_propagate is not None else "unlimited",
        len(ssam_frames),
        float(iou_threshold),
        float(mask_scale_ratio),
        "yes" if render_viz else "no",
    )

    long_tail_area_threshold: Optional[int] = None
    if all_box_prompt:
        LOGGER.info("All mask candidates will be converted to SAM2 box prompts")
        if long_tail_box_prompt:
            LOGGER.info("Long-tail box prompt flag ignored because all-box prompt is active")
    elif long_tail_box_prompt:
        env_area = os.environ.get("MY3DIS_LONG_TAIL_AREA")
        if env_area:
            try:
                long_tail_area_threshold = max(1, int(env_area))
                LOGGER.info(
                    "Environment override: MY3DIS_LONG_TAIL_AREA=%d", long_tail_area_threshold
                )
            except (TypeError, ValueError):
                LOGGER.warning("Invalid MY3DIS_LONG_TAIL_AREA=%r; ignoring", env_area)
        if long_tail_area_threshold is None:
            manifest_min_area = manifest.get('min_area')
            try:
                manifest_min_area = int(manifest_min_area) if manifest_min_area is not None else None
            except (TypeError, ValueError):
                manifest_min_area = None
            if manifest_min_area is not None and manifest_min_area > 0:
                long_tail_area_threshold = max(manifest_min_area * 3, manifest_min_area + 1)
        if long_tail_area_threshold is None:
            long_tail_area_threshold = 1500
        LOGGER.info(
            "Long-tail box prompt enabled: masks with area ≤ %d px will use SAM2 box prompts",
            long_tail_area_threshold,
        )

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
        # 使用 SSAM 幀來重建 subset
        subset_dir, subset_map = build_subset_video(
            frames_dir=data_path,
            selected=ssam_frames,
            selected_indices=ssam_absolute_indices,
            out_root=out_root,
        )
        manifest['subset_dir'] = subset_dir
        manifest['subset_map'] = subset_map
        with open(os.path.join(candidates_root, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
    LOGGER.info("Selected frames available at %s", subset_dir)
    
    # 建立幀索引對應關係（針對有 SSAM 分割的幀）
    frame_index_to_name = {idx: name for idx, name in zip(ssam_absolute_indices, ssam_frames)}

    level_stats = []
    level_artifacts: Dict[int, Dict[str, str]] = {}
    overall_timer = TimingAggregator()
    for level in level_list:
        level_start = time.perf_counter()
        level_root = os.path.join(candidates_root, f'level_{level}')
        track_dir = ensure_dir(os.path.join(out_root, f'level_{level}', 'tracking'))
        
        # 加載候選項和對應的幀索引
        per_frame, frame_indices, frame_names = load_filtered_candidates(
            level_root, mask_scale_ratio=mask_scale_ratio
        )
        frame_name_lookup = {
            int(idx): name for idx, name in zip(frame_indices, frame_names)
        }
        if frame_index_to_name:
            for idx, name in frame_index_to_name.items():
                frame_name_lookup.setdefault(int(idx), str(name))
        
        LOGGER.info(f"Level {level}: Processing {len(per_frame)} frames with SAM2 tracking...")
        level_timer = TimingAggregator()

        with level_timer.track('track.sam2'):
            segs = sam2_tracking(
                subset_dir,
                predictor,
                per_frame,
                frame_numbers=frame_indices,  # 使用實際的幀索引
                iou_threshold=iou_threshold,
                max_propagate=sam2_max_propagate,
                use_box_for_small=(long_tail_box_prompt and not all_box_prompt),
                use_box_for_all=all_box_prompt,
                small_object_area_threshold=long_tail_area_threshold,
                mask_scale_ratio=mask_scale_ratio,
            )

        level_video_path = None
        level_object_path = None

        with level_timer.track('persist.video_segments'):
            level_video_path = save_video_segments_npz(
                segs,
                os.path.join(track_dir, 'video_segments.npz'),
                mask_scale_ratio=mask_scale_ratio,
            )

        with level_timer.track('persist.object_npz'):
            obj_segments = reorganize_segments_by_object(segs)
            level_object_path = save_object_segments_npz(
                obj_segments,
                os.path.join(track_dir, 'object_segments.npz'),
                mask_scale_ratio=mask_scale_ratio,
            )

        level_artifacts[level] = {
            'video_segments': level_video_path,
            'object_segments': level_object_path,
        }
        LOGGER.info(
            "Level %d artifacts saved (video=%s, object=%s)",
            level,
            os.path.basename(level_video_path) if level_video_path else 'n/a',
            os.path.basename(level_object_path) if level_object_path else 'n/a',
        )

        if render_viz:
            viz_dir = os.path.join(out_root, f'level_{level}', 'viz')
            with level_timer.track('viz.comparison'):
                save_comparison_proposals(
                    viz_dir=viz_dir,
                    base_frames_dir=data_path,
                    filtered_per_frame=per_frame,
                    video_segments=segs,
                    level=level,
                    frame_numbers=frame_indices,  # 使用實際的幀索引
                    frame_name_lookup=frame_name_lookup,
                    subset_dir=subset_dir,
                    subset_map=subset_map,
                    frames_to_save=None,
                )

        level_total = time.perf_counter() - level_start

        track_time = level_timer.total('track.sam2')
        persist_time = level_timer.total_prefix('persist.')
        viz_time = level_timer.total_prefix('viz.')
        render_time = viz_time

        LOGGER.info(
            "Level %d finished (%d objects / %d frames) → %s",
            level,
            len(obj_segments),
            len(segs),
            level_timer.format_breakdown(),
        )

        level_stats.append(
            (
                level,
                len(obj_segments),
                len(segs),
                track_time,
                persist_time,
                viz_time,
                render_time,
                level_total,
            )
        )

        overall_timer.merge(level_timer)

    if level_stats:
        summary = "; ".join(
            f"L{lvl}: {objs} objects / {frames} frames "
            f"(track={format_duration_precise(track)}, persist={format_duration_precise(persist)}, "
            f"viz={format_duration_precise(viz)}, render={format_duration_precise(render)}, "
            f"total={format_duration_precise(total)})"
            for (
                lvl,
                objs,
                frames,
                track,
                persist,
                viz,
                render,
                total,
            ) in level_stats
        )
        LOGGER.info("Tracking summary → %s", summary)

    if overall_timer.items():
        category_summary = []
        for label, prefix in [
            ("track", 'track.'),
            ("persist", 'persist.'),
            ("viz", 'viz.'),
        ]:
            total = overall_timer.total_prefix(prefix)
            if total > 0:
                category_summary.append(f"{label}={format_duration_precise(total)}")
        if category_summary:
            LOGGER.info("Aggregate timing by stage → %s", ", ".join(category_summary))
        LOGGER.debug("Aggregate timing breakdown → %s", overall_timer.format_breakdown())

    manifest['mask_scale_ratio'] = float(mask_scale_ratio)
    manifest['render_viz'] = bool(render_viz)
    tracking_artifacts: Dict[str, Dict[str, Optional[str]]] = {}
    for lvl, paths in level_artifacts.items():
        video_path = paths.get('video_segments') if paths else None
        object_path = paths.get('object_segments') if paths else None
        rel_video = os.path.relpath(video_path, out_root) if video_path else None
        rel_object = os.path.relpath(object_path, out_root) if object_path else None
        tracking_artifacts[f"level_{lvl}"] = {
            'video_segments': rel_video,
            'object_segments': rel_object,
        }
    manifest['tracking_artifacts'] = tracking_artifacts
    manifest_path = os.path.join(candidates_root, 'manifest.json')
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        LOGGER.warning('Failed to update manifest at %s', manifest_path, exc_info=True)

    LOGGER.info("Tracking results saved at %s", out_root)
    LOGGER.info(
        "Tracking completed in %s",
        format_duration_precise(time.perf_counter() - overall_start),
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
    ap.add_argument('--sam2-max-propagate', type=int, default=None,
                    help='Limit SAM2 propagation to N frames per direction (default: unlimited)')
    ap.add_argument('--iou-threshold', type=float, default=0.6,
                    help='IoU threshold for deduplicating SAM2 prompts (default: 0.6)')
    ap.add_argument('--long-tail-box-prompt', action='store_true',
                    help='Convert long-tail small objects to SAM2 box prompts')
    ap.add_argument('--all-box-prompt', action='store_true',
                    help='Convert all mask prompts to SAM2 box prompts')
    ap.add_argument('--mask-scale-ratio', type=float, default=1.0,
                    help='Downscale masks before persistence (e.g., 0.3 keeps 30% resolution)')
    ap.add_argument('--skip-viz', action='store_true',
                    help='Disable all additional visualization renders to keep outputs minimal')
    args = ap.parse_args()

    run_tracking(
        data_path=args.data_path,
        candidates_root=args.candidates_root,
        output=args.output,
        levels=args.levels,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        sam2_max_propagate=args.sam2_max_propagate,
        iou_threshold=args.iou_threshold,
        long_tail_box_prompt=args.long_tail_box_prompt,
        all_box_prompt=args.all_box_prompt,
        mask_scale_ratio=args.mask_scale_ratio,
        render_viz=not args.skip_viz,
    )


if __name__ == '__main__':
    main()
