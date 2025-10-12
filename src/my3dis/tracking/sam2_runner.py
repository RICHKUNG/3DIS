"""SAM2 tracking execution and prompt preparation helpers."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from my3dis.common_utils import downscale_binary_mask, pack_binary_mask, unpack_binary_mask
from my3dis.tracking.helpers import (
    ProgressPrinter,
    bbox_scalar_fit,
    bbox_transform_xywh_to_xyxy,
)
from my3dis.tracking.stores import DedupStore, FrameResultStore
from .candidate_loader import FrameCandidateBatch

LOGGER = logging.getLogger("my3dis.track_from_candidates")

__all__ = [
    'PromptCandidate',
    'TrackingArtifacts',
    'sam2_tracking',
]


@dataclass
class PromptCandidate:
    """Derived metadata attached to a filtered SSAM prompt candidate."""

    payload: Dict[str, Any]
    seg_prompt: Optional[np.ndarray]
    seg_for_iou: Optional[np.ndarray]
    area: Optional[int]
    bbox_xywh: Optional[Tuple[float, float, float, float]]


@dataclass
class TrackingArtifacts:
    """In-memory aggregates returned by the SAM2 tracking stage."""

    object_refs: Dict[int, Dict[int, str]]
    preview_segments: Dict[int, Dict[int, Any]]
    frames_with_predictions: Set[int]
    objects_seen: Set[int]


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
        if bbox is None or len(bbox) != 4:
            bbox_xywh: Optional[Tuple[float, float, float, float]] = None
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


def _filter_new_candidates(
    candidates: List[PromptCandidate],
    *,
    frame_idx: int,
    dedup_store: DedupStore,
    iou_threshold: float,
) -> List[PromptCandidate]:
    return dedup_store.filter_candidates(frame_idx, candidates, iou_threshold)


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


def sam2_tracking(
    frames_dir: str,
    predictor,
    candidate_batches: Iterable[FrameCandidateBatch],
    *,
    frame_numbers: List[int],
    frame_name_lookup: Dict[int, str],
    ssam_local_indices: Optional[Sequence[int]] = None,
    iou_threshold: float = 0.6,
    max_propagate: Optional[int] = None,
    use_box_for_small: bool = False,
    use_box_for_all: bool = False,
    small_object_area_threshold: Optional[int] = None,
    mask_scale_ratio: float = 1.0,
    preview_targets: Optional[Set[int]] = None,
    dedup_store: Optional[DedupStore] = None,
    result_store: Optional[FrameResultStore] = None,
) -> TrackingArtifacts:
    os.environ['TQDM_DISABLE'] = '1'
    try:  # pragma: no cover - optional dependency
        import tqdm

        tqdm.tqdm.disable = True
    except ImportError:  # pragma: no cover - tqdm not installed
        pass

    preview_targets = set(preview_targets or [])
    dedup_store = dedup_store or DedupStore()
    result_store = result_store or FrameResultStore()

    object_refs: Dict[int, Dict[int, str]] = defaultdict(dict)
    preview_segments: Dict[int, Dict[int, Any]] = {}
    frames_with_predictions: Set[int] = set()
    objects_seen: Set[int] = set()

    with torch.inference_mode(), torch.autocast("cuda"):
        state = predictor.init_state(video_path=frames_dir)
        sx: Optional[float] = None
        sy: Optional[float] = None

        local_to_abs = {i: frame_numbers[i] for i in range(len(frame_numbers))}
        if ssam_local_indices is not None:
            try:
                total_updates = max(1, len(ssam_local_indices))  # type: ignore[arg-type]
            except TypeError:
                ssam_local_indices = list(ssam_local_indices)
                total_updates = max(1, len(ssam_local_indices))
        else:
            total_updates = max(1, len(frame_numbers))
        progress = ProgressPrinter(total_updates)

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
            for batch in candidate_batches:
                frame_idx = batch.local_index
                abs_idx = local_to_abs.get(frame_idx, batch.frame_index)
                predictor.reset_state(state)
                progress.update(frame_idx, abs_idx)
                if abs_idx is None:
                    continue

                if sx is None or sy is None:
                    for item in batch.candidates:
                        candidate_seg = item.get('segmentation')
                        if isinstance(candidate_seg, np.ndarray):
                            h0, w0 = candidate_seg.shape[:2]
                            sx = state['video_width'] / max(1, w0)
                            sy = state['video_height'] / max(1, h0)
                            break
                    if sx is None or sy is None:
                        continue

                prepared_candidates = _prepare_prompt_candidates(batch.candidates)
                filtered_candidates = _filter_new_candidates(
                    prepared_candidates,
                    frame_idx=abs_idx,
                    dedup_store=dedup_store,
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
                    total_frames=len(frame_numbers),
                    max_propagate=max_propagate,
                    mask_scale_ratio=mask_scale_ratio,
                )

                for abs_out_idx, frame_data in frame_segments.items():
                    if not frame_data:
                        continue
                    frames_with_predictions.add(abs_out_idx)
                    dedup_store.add_packed(abs_out_idx, frame_data)
                    frame_name = frame_name_lookup.get(abs_out_idx)
                    entry_name = result_store.update(abs_out_idx, frame_name, frame_data)
                    for obj_id_raw, payload in frame_data.items():
                        obj_id = int(obj_id_raw)
                        object_refs[obj_id][abs_out_idx] = entry_name
                        objects_seen.add(obj_id)
                    if abs_out_idx in preview_targets and abs_out_idx not in preview_segments:
                        preview_segments[abs_out_idx] = {
                            int(obj_id): dict(payload)
                            for obj_id, payload in frame_data.items()
                        }
        finally:
            progress.close()

    if sx is None or sy is None:
        raise RuntimeError('No segmentation masks available to derive resolution')

    return TrackingArtifacts(
        object_refs=dict(object_refs),
        preview_segments=preview_segments,
        frames_with_predictions=frames_with_predictions,
        objects_seen=objects_seen,
    )
