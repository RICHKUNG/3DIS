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
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

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
    build_object_segments_archive,
    build_video_segments_archive,
    encode_packed_mask_for_json,
    format_duration_precise,
    infer_relative_scale,
    resize_mask_to_shape,
    save_comparison_proposals,
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


@dataclass
class FrameCandidateBatch:
    """Container for a single frame's filtered SSAM candidates."""

    local_index: int
    frame_index: int
    frame_name: str
    candidates: List[Dict[str, Any]]


@dataclass
class TrackingArtifacts:
    """In-memory aggregates returned by the SAM2 tracking stage."""

    object_refs: Dict[int, Dict[int, str]]
    preview_segments: Dict[int, Dict[int, Any]]
    frames_with_predictions: Set[int]
    objects_seen: Set[int]


@dataclass
class TrackingContext:
    """Pre-computed manifest context shared across level tracking runs."""

    manifest: Dict[str, Any]
    manifest_path: str
    level_list: List[int]
    selected_frames: List[str]
    selected_indices: List[int]
    ssam_frames: List[str]
    ssam_absolute_indices: List[int]
    ssam_freq: int
    subset_dir: Optional[str]
    subset_map: Dict[int, Any]
    sam2_max_propagate: Optional[int]


@dataclass
class LevelRunResult:
    """Outputs produced for a specific level during tracking."""

    level: int
    artifacts: Dict[str, Any]
    comparison: Optional[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    stats: Tuple[int, int, int, float, float, float, float, float]
    timer: TimingAggregator
    duration: float


@dataclass
class _DedupEntry:
    target_shape: Tuple[int, int]
    masks: List[np.ndarray]


class DedupStore:
    """Downscaled per-frame mask stacks used for IoU-based deduplication."""

    def __init__(self, *, max_dim: int = 256) -> None:
        self._max_dim = max(1, int(max_dim))
        self._frames: Dict[int, _DedupEntry] = {}

    def _compute_target_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        h, w = shape
        longest = max(h, w)
        if longest <= self._max_dim:
            return h, w
        ratio = self._max_dim / float(longest)
        new_h = max(1, int(round(h * ratio)))
        new_w = max(1, int(round(w * ratio)))
        return new_h, new_w

    def _ensure_entry(self, frame_idx: int, mask_shape: Tuple[int, int]) -> _DedupEntry:
        entry = self._frames.get(frame_idx)
        if entry is not None:
            return entry
        target_shape = self._compute_target_shape(mask_shape)
        entry = _DedupEntry(target_shape=target_shape, masks=[])
        self._frames[frame_idx] = entry
        return entry

    @staticmethod
    def _resize(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if mask.shape == target_shape:
            return mask.astype(np.bool_)
        return resize_mask_to_shape(mask, target_shape).astype(np.bool_)

    def _max_iou(self, entry: _DedupEntry, candidate: np.ndarray) -> float:
        if not entry.masks:
            return 0.0
        cand = self._resize(candidate, entry.target_shape)
        if not entry.masks:
            return 0.0
        ious: List[float] = []
        for existing in entry.masks:
            inter = np.logical_and(existing, cand).sum()
            union = np.logical_or(existing, cand).sum()
            if union == 0:
                continue
            ious.append(float(inter) / float(union))
        return max(ious) if ious else 0.0

    def has_overlap(self, frame_idx: int, mask: np.ndarray, threshold: float) -> bool:
        entry = self._frames.get(frame_idx)
        if entry is None or not entry.masks:
            return False
        return self._max_iou(entry, mask) > float(threshold)

    def add_mask(self, frame_idx: int, mask: np.ndarray) -> None:
        arr = np.asarray(mask, dtype=np.bool_)
        entry = self._ensure_entry(frame_idx, arr.shape)
        entry.masks.append(self._resize(arr, entry.target_shape))

    def add_packed(self, frame_idx: int, payloads: Dict[int, Any]) -> None:
        if not payloads:
            return
        for packed in payloads.values():
            arr = unpack_binary_mask(packed)
            arr = np.asarray(arr, dtype=np.bool_)
            self.add_mask(frame_idx, arr)

    def filter_candidates(
        self,
        frame_idx: int,
        candidates: List[PromptCandidate],
        threshold: float,
    ) -> List[PromptCandidate]:
        accepted: List[PromptCandidate] = []
        for cand in candidates:
            seg = cand.seg_for_iou
            if seg is not None and self.has_overlap(frame_idx, seg, threshold):
                continue
            accepted.append(cand)
            if seg is not None:
                self.add_mask(frame_idx, seg)
        return accepted


def _frame_entry_name(frame_idx: int) -> str:
    return f"frames/frame_{int(frame_idx):06d}.json"


class FrameResultStore:
    """Disk-backed storage for frame-major SAM2 propagation results."""

    def __init__(self, *, prefix: str = "sam2_frames_") -> None:
        self._root = Path(tempfile.mkdtemp(prefix=prefix))
        self._index: Dict[int, Path] = {}

    def update(self, frame_idx: int, frame_name: Optional[str], frame_data: Dict[int, Any]) -> str:
        entry_name = _frame_entry_name(frame_idx)
        path = self._root / f"{frame_idx:06d}.json"
        if path.exists():
            with path.open('r', encoding='utf-8') as fh:
                existing = json.load(fh)
        else:
            existing = {'frame_index': int(frame_idx), 'frame_name': frame_name, 'objects': {}}

        serialised = {
            str(obj_id): encode_packed_mask_for_json(payload)
            for obj_id, payload in frame_data.items()
        }
        existing['frame_name'] = frame_name
        existing.setdefault('objects', {})
        existing['objects'].update(serialised)

        with path.open('w', encoding='utf-8') as fh:
            json.dump(existing, fh, ensure_ascii=False)

        self._index[frame_idx] = path
        return entry_name

    def iter_frames(self) -> Iterator[Dict[str, Any]]:
        for frame_idx in sorted(self._index.keys()):
            path = self._index[frame_idx]
            with path.open('r', encoding='utf-8') as fh:
                yield json.load(fh)

    def cleanup(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)


def _load_filtered_manifest(level_root: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    filt_dir = os.path.join(level_root, 'filtered')
    manifest_path = os.path.join(filt_dir, 'filtered.json')
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        meta = json.load(fh)
    frames_meta = meta.get('frames', [])
    return meta, frames_meta


def _load_frame_candidates(
    filt_dir: str,
    frame_meta: Dict[str, Any],
    *,
    mask_scale_ratio: float,
) -> List[Dict[str, Any]]:
    fidx = int(frame_meta['frame_idx'])
    seg_path = os.path.join(filt_dir, f'seg_frame_{fidx:05d}.npy')
    seg_stack = np.load(seg_path, mmap_mode='r') if os.path.exists(seg_path) else None
    try:
        items_meta = frame_meta.get('items', [])
        loaded: List[Dict[str, Any]] = []
        for j, it in enumerate(items_meta):
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
            loaded.append(d)
        return loaded
    finally:
        if seg_stack is not None:
            del seg_stack


def iter_filtered_candidate_batches(
    level_root: str,
    frames_meta: List[Dict[str, Any]],
    *,
    mask_scale_ratio: float,
) -> Iterator[FrameCandidateBatch]:
    filt_dir = os.path.join(level_root, 'filtered')
    for local_idx, frame_meta in enumerate(frames_meta):
        fidx = int(frame_meta['frame_idx'])
        fname = frame_meta.get('frame_name')
        if fname is None:
            fname = f"{int(fidx):05d}.png"
        candidates = _load_frame_candidates(filt_dir, frame_meta, mask_scale_ratio=mask_scale_ratio)
        yield FrameCandidateBatch(
            local_index=local_idx,
            frame_index=fidx,
            frame_name=str(fname),
            candidates=candidates,
        )


def load_filtered_frame_by_index(
    level_root: str,
    frames_meta: List[Dict[str, Any]],
    *,
    local_index: int,
    mask_scale_ratio: float,
) -> Optional[List[Dict[str, Any]]]:
    if local_index < 0 or local_index >= len(frames_meta):
        return None
    filt_dir = os.path.join(level_root, 'filtered')
    return _load_frame_candidates(
        filt_dir,
        frames_meta[local_index],
        mask_scale_ratio=mask_scale_ratio,
    )


DEFAULT_PREVIEW_MAX_FRAMES = 12


def _select_preview_indices(
    total_frames: int,
    *,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> List[int]:
    if total_frames <= 0:
        return []

    stride_val: Optional[int] = None
    if stride is not None:
        try:
            stride_val = max(1, int(stride))
        except (TypeError, ValueError):
            stride_val = None

    if max_samples is None:
        target: Optional[int] = DEFAULT_PREVIEW_MAX_FRAMES
    else:
        try:
            parsed_target = int(max_samples)
        except (TypeError, ValueError):
            parsed_target = DEFAULT_PREVIEW_MAX_FRAMES
        target = parsed_target if parsed_target > 0 else None

    indices: List[int]
    if stride_val:
        indices = list(range(0, total_frames, stride_val))
    else:
        if target is None or target >= total_frames:
            indices = list(range(total_frames))
        else:
            positions = np.linspace(0, total_frames - 1, num=target, dtype=int)
            indices = [int(pos) for pos in positions]

    if not indices:
        indices = [0]

    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)

    indices = sorted(set(idx for idx in indices if 0 <= idx < total_frames))
    if target is not None and len(indices) > target:
        positions = np.linspace(0, len(indices) - 1, num=target, dtype=int)
        reduced = [indices[int(pos)] for pos in positions]
        reduced[0] = 0
        reduced[-1] = total_frames - 1
        indices = sorted(set(reduced))
    return indices


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


def prepare_tracking_context(
    *,
    candidates_root: str,
    level_list: List[int],
    sam2_max_propagate: Optional[int],
) -> TrackingContext:
    """Load manifest metadata and normalize values for downstream helpers."""

    manifest_path = os.path.join(candidates_root, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    selected = manifest.get('selected_frames', []) or []
    selected_indices_raw = manifest.get('selected_indices')
    if selected_indices_raw is None:
        selected_indices = list(range(len(selected)))
    else:
        selected_indices = [int(x) for x in selected_indices_raw]

    ssam_frames = manifest.get('ssam_frames', selected) or []
    ssam_abs_raw = manifest.get('ssam_absolute_indices', selected_indices_raw)
    if ssam_abs_raw is None:
        ssam_absolute_indices = list(selected_indices)
    else:
        ssam_absolute_indices = [int(x) for x in ssam_abs_raw]

    subset_dir = manifest.get('subset_dir')
    subset_map = manifest.get('subset_map', {})
    if not isinstance(subset_map, dict):
        subset_map = {}

    try:
        ssam_freq = int(manifest.get('ssam_freq', 1))
    except (TypeError, ValueError):
        ssam_freq = 1

    manifest_max_propagate = manifest.get('sam2_max_propagate')
    resolved_propagate = sam2_max_propagate
    if resolved_propagate is None:
        resolved_propagate = manifest_max_propagate
    if resolved_propagate is not None:
        raw_value = resolved_propagate
        try:
            resolved_propagate = max(0, int(resolved_propagate))
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid sam2_max_propagate=%r; defaulting to unlimited",
                raw_value,
            )
            resolved_propagate = None

    return TrackingContext(
        manifest=manifest,
        manifest_path=manifest_path,
        level_list=list(level_list),
        selected_frames=list(selected),
        selected_indices=list(selected_indices),
        ssam_frames=list(ssam_frames),
        ssam_absolute_indices=list(ssam_absolute_indices),
        ssam_freq=int(ssam_freq),
        subset_dir=subset_dir,
        subset_map=dict(subset_map),
        sam2_max_propagate=resolved_propagate,
    )


def resolve_long_tail_area_threshold(
    *,
    manifest: Dict[str, Any],
    long_tail_box_prompt: bool,
    all_box_prompt: bool,
) -> Optional[int]:
    """Compute the SAM2 box prompt threshold for long-tail objects."""

    if not long_tail_box_prompt or all_box_prompt:
        return None

    long_tail_area_threshold: Optional[int] = None
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
    return long_tail_area_threshold


def ensure_subset_video(
    context: TrackingContext,
    *,
    data_path: str,
    out_root: str,
) -> Tuple[str, Dict[int, Any]]:
    """Ensure the subset video referenced in the manifest exists and is populated."""

    subset_dir = context.subset_dir
    subset_map = context.subset_map or {}
    rebuild_subset = False
    subset_map_int: Dict[int, Any] = {}
    if subset_dir and os.path.isdir(subset_dir):
        try:
            subset_map_int = {int(k): v for k, v in subset_map.items()}
        except Exception:
            subset_map_int = {}
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
        subset_dir, subset_map_int = build_subset_video(
            frames_dir=data_path,
            selected=context.ssam_frames,
            selected_indices=context.ssam_absolute_indices,
            out_root=out_root,
        )
        context.manifest['subset_dir'] = subset_dir
        context.manifest['subset_map'] = subset_map_int
        context.subset_dir = subset_dir
        context.subset_map = dict(subset_map_int)
    else:
        context.subset_dir = subset_dir
        context.subset_map = dict(subset_map)

    return subset_dir, subset_map_int


def persist_level_outputs(
    *,
    level: int,
    tracking_output: TrackingArtifacts,
    frame_store: FrameResultStore,
    track_dir: str,
    mask_scale_ratio: float,
    level_timer: TimingAggregator,
) -> Dict[str, Optional[str]]:
    """Persist per-level video/object artifacts and clean up temporary storage."""

    level_video_path: Optional[str] = None
    level_object_path: Optional[str] = None
    try:
        with level_timer.track('persist.video_segments'):
            level_video_path, _video_manifest = build_video_segments_archive(
                frame_store.iter_frames(),
                os.path.join(track_dir, 'video_segments.npz'),
                mask_scale_ratio=mask_scale_ratio,
                metadata={'level': level},
            )

        with level_timer.track('persist.object_manifest'):
            level_object_path = build_object_segments_archive(
                tracking_output.object_refs,
                os.path.join(track_dir, 'object_segments.npz'),
                mask_scale_ratio=mask_scale_ratio,
                metadata={
                    'level': level,
                    'linked_video': os.path.basename(level_video_path) if level_video_path else None,
                },
            )
    finally:
        frame_store.cleanup()

    return {
        'video_segments': level_video_path,
        'object_segments': level_object_path,
    }


def run_level_tracking(
    *,
    level: int,
    candidates_root: str,
    data_path: str,
    subset_dir: str,
    subset_map: Dict[int, Any],
    predictor,
    frame_index_lookup: Dict[int, str],
    sam2_max_propagate: Optional[int],
    iou_threshold: float,
    long_tail_box_prompt: bool,
    all_box_prompt: bool,
    long_tail_area_threshold: Optional[int],
    mask_scale_ratio: float,
    comparison_sample_stride: Optional[int],
    comparison_max_samples: Optional[int],
    render_viz: bool,
    out_root: str,
) -> LevelRunResult:
    """Execute SAM2 tracking for a single level and persist results."""

    level_start = time.perf_counter()
    level_root = os.path.join(candidates_root, f'level_{level}')
    track_dir = ensure_dir(os.path.join(out_root, f'level_{level}', 'tracking'))

    _, frames_meta = _load_filtered_manifest(level_root)
    frame_numbers = [int(fm['frame_idx']) for fm in frames_meta]
    frame_name_lookup: Dict[int, str] = {}
    for fm in frames_meta:
        fidx = int(fm['frame_idx'])
        fname = fm.get('frame_name')
        if fname is None:
            fname = f"{fidx:05d}.png"
        frame_name_lookup[fidx] = str(fname)
    for idx, name in frame_index_lookup.items():
        frame_name_lookup.setdefault(int(idx), str(name))

    LOGGER.info(
        "Level %d: Processing %d frames with SAM2 tracking...",
        level,
        len(frame_numbers),
    )

    level_timer = TimingAggregator()
    preview_local_indices = _select_preview_indices(
        len(frame_numbers),
        stride=comparison_sample_stride,
        max_samples=comparison_max_samples,
    )
    preview_targets = {
        frame_numbers[idx] for idx in preview_local_indices if 0 <= idx < len(frame_numbers)
    }

    candidate_iter = iter_filtered_candidate_batches(
        level_root,
        frames_meta,
        mask_scale_ratio=mask_scale_ratio,
    )

    dedup_store = DedupStore()
    frame_store = FrameResultStore(prefix=f"sam2_frames_L{level}_")

    with level_timer.track('track.sam2'):
        tracking_output = sam2_tracking(
            subset_dir,
            predictor,
            candidate_iter,
            frame_numbers=frame_numbers,
            frame_name_lookup=frame_name_lookup,
            iou_threshold=iou_threshold,
            max_propagate=sam2_max_propagate,
            use_box_for_small=(long_tail_box_prompt and not all_box_prompt),
            use_box_for_all=all_box_prompt,
            small_object_area_threshold=long_tail_area_threshold,
            mask_scale_ratio=mask_scale_ratio,
            preview_targets=preview_targets,
            dedup_store=dedup_store,
            result_store=frame_store,
        )

    artifacts = persist_level_outputs(
        level=level,
        tracking_output=tracking_output,
        frame_store=frame_store,
        track_dir=track_dir,
        mask_scale_ratio=mask_scale_ratio,
        level_timer=level_timer,
    )

    level_video_path = artifacts.get('video_segments')
    level_object_path = artifacts.get('object_segments')
    LOGGER.info(
        "Level %d artifacts saved (video=%s, object=%s)",
        level,
        os.path.basename(level_video_path) if level_video_path else 'n/a',
        os.path.basename(level_object_path) if level_object_path else 'n/a',
    )

    filtered_preview: List[Optional[List[Dict[str, Any]]]] = [None] * len(frame_numbers)
    for local_idx in preview_local_indices:
        candidates_for_viz = load_filtered_frame_by_index(
            level_root,
            frames_meta,
            local_index=local_idx,
            mask_scale_ratio=mask_scale_ratio,
        )
        if candidates_for_viz is not None:
            filtered_preview[local_idx] = candidates_for_viz

    comparison_result: Optional[Dict[str, Any]] = None
    tracker_warnings: List[Dict[str, Any]] = []
    if render_viz:
        viz_dir = os.path.join(out_root, f'level_{level}', 'viz')
        with level_timer.track('viz.comparison'):
            frames_to_save = [
                frame_numbers[idx]
                for idx in preview_local_indices
                if 0 <= idx < len(frame_numbers)
            ]
            comparison_result = save_comparison_proposals(
                viz_dir=viz_dir,
                base_frames_dir=data_path,
                filtered_per_frame=filtered_preview,
                video_segments=tracking_output.preview_segments,
                level=level,
                frame_numbers=frame_numbers,
                frame_name_lookup=frame_name_lookup,
                subset_dir=subset_dir,
                subset_map=subset_map,
                frames_to_save=frames_to_save,
                sample_stride=comparison_sample_stride,
                max_samples=comparison_max_samples,
            )
        if comparison_result:
            summary_path = comparison_result.get('summary_path')
            if summary_path:
                artifacts['comparison_summary'] = os.path.relpath(summary_path, out_root)
            rel_list = comparison_result.get('rendered_images_rel') or []
            if rel_list:
                viz_rel_dir = os.path.join(out_root, f'level_{level}', 'viz')
                images_rel = [
                    os.path.relpath(os.path.join(viz_rel_dir, rel), out_root)
                    for rel in rel_list
                ]
                if images_rel:
                    artifacts['comparison_images'] = images_rel
            fallback_path = comparison_result.get('fallback_path')
            if fallback_path:
                artifacts['comparison_fallback'] = os.path.relpath(fallback_path, out_root)
            warning_payload = comparison_result.get('warning')
            if warning_payload:
                warning_entry = dict(warning_payload)
                warning_entry.setdefault('stage', 'tracker')
                warning_entry.setdefault('level', int(level))
                if summary_path:
                    warning_entry['summary_relpath'] = os.path.relpath(summary_path, out_root)
                tracker_warnings.append(warning_entry)

    level_total = time.perf_counter() - level_start

    track_time = level_timer.total('track.sam2')
    persist_time = level_timer.total_prefix('persist.')
    viz_time = level_timer.total_prefix('viz.')
    render_time = viz_time
    objects_count = len(tracking_output.objects_seen)
    frames_count = len(tracking_output.frames_with_predictions)

    LOGGER.info(
        "Level %d finished (%d objects / %d frames) → %s",
        level,
        objects_count,
        frames_count,
        level_timer.format_breakdown(),
    )

    stats = (
        level,
        objects_count,
        frames_count,
        track_time,
        persist_time,
        viz_time,
        render_time,
        level_total,
    )

    return LevelRunResult(
        level=level,
        artifacts=artifacts,
        comparison=comparison_result,
        warnings=tracker_warnings,
        stats=stats,
        timer=level_timer,
        duration=level_total,
    )


def update_manifest(
    context: TrackingContext,
    *,
    out_root: str,
    level_results: List[LevelRunResult],
    mask_scale_ratio: float,
    render_viz: bool,
) -> None:
    """Update manifest metadata with tracking outputs and persist to disk."""

    manifest = context.manifest
    manifest['mask_scale_ratio'] = float(mask_scale_ratio)
    manifest['render_viz'] = bool(render_viz)

    tracking_artifacts: Dict[str, Dict[str, Optional[str]]] = {}
    comparison_manifest: Dict[str, Dict[str, Any]] = {}
    tracker_warnings: List[Dict[str, Any]] = []

    for result in level_results:
        level_key = f"level_{result.level}"
        paths = result.artifacts or {}

        video_path = paths.get('video_segments')
        object_path = paths.get('object_segments')
        rel_video = os.path.relpath(video_path, out_root) if video_path else None
        rel_object = os.path.relpath(object_path, out_root) if object_path else None

        entry: Dict[str, Optional[str]] = {}
        if rel_video:
            entry['video_segments'] = rel_video
        if rel_object:
            entry['object_segments'] = rel_object

        summary_path = paths.get('comparison_summary')
        if summary_path:
            entry['comparison_summary'] = summary_path
        images_rel = paths.get('comparison_images')
        if images_rel:
            entry['comparison_images'] = images_rel
        fallback_rel = paths.get('comparison_fallback')
        if fallback_rel:
            entry['comparison_fallback'] = fallback_rel

        tracking_artifacts[level_key] = entry

        comparison_data = result.comparison
        if comparison_data:
            viz_dir = os.path.join(out_root, f'level_{result.level}', 'viz')
            rendered_images_rel = comparison_data.get('rendered_images_rel') or []
            comparison_manifest[level_key] = {
                'generated_at': comparison_data.get('generated_at'),
                'rendered_count': comparison_data.get('rendered_count'),
                'rendered_frames': comparison_data.get('rendered_frames'),
                'frames_attempted': comparison_data.get('frames_attempted'),
                'requested_frames': comparison_data.get('requested_frames'),
                'rendered_images': [
                    os.path.relpath(os.path.join(viz_dir, rel), out_root)
                    for rel in rendered_images_rel
                ],
                'summary': paths.get('comparison_summary'),
                'fallback': paths.get('comparison_fallback'),
                'warning': comparison_data.get('warning'),
                'issues': comparison_data.get('issues'),
            }
        elif render_viz:
            comparison_manifest[level_key] = {
                'generated_at': None,
                'rendered_count': 0,
            }

        tracker_warnings.extend(result.warnings)

    manifest['tracking_artifacts'] = tracking_artifacts
    if comparison_manifest:
        manifest['comparison_summary'] = comparison_manifest
    elif 'comparison_summary' in manifest:
        manifest.pop('comparison_summary')

    existing_warnings = manifest.get('warnings')
    if isinstance(existing_warnings, list):
        manifest['warnings'] = [
            w
            for w in existing_warnings
            if not (isinstance(w, dict) and w.get('stage') == 'tracker')
        ]
    else:
        manifest['warnings'] = []
    if tracker_warnings:
        manifest['warnings'].extend(tracker_warnings)
    if not manifest['warnings']:
        manifest.pop('warnings', None)

    try:
        with open(context.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        LOGGER.warning('Failed to update manifest at %s', context.manifest_path, exc_info=True)


def sam2_tracking(
    frames_dir: str,
    predictor,
    candidate_batches: Iterable[FrameCandidateBatch],
    *,
    frame_numbers: List[int],
    frame_name_lookup: Dict[int, str],
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
        progress = ProgressPrinter(len(frame_numbers))

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
    comparison_sample_stride: Optional[int] = None,
    comparison_max_samples: Optional[int] = None,
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

    context = prepare_tracking_context(
        candidates_root=candidates_root,
        level_list=level_list,
        sam2_max_propagate=sam2_max_propagate,
    )
    manifest = context.manifest
    level_list = context.level_list
    ssam_frames = context.ssam_frames
    ssam_absolute_indices = context.ssam_absolute_indices
    ssam_freq = context.ssam_freq
    sam2_max_propagate = context.sam2_max_propagate

    LOGGER.info(
        "Configuration: ssam_freq=%d, sam2_max_propagate=%s, ssam_frames=%d, iou_threshold=%.2f, mask_scale_ratio=%.3f, preview_stride=%s, preview_max=%s, render_viz=%s",
        ssam_freq,
        sam2_max_propagate,
        len(ssam_frames),
        iou_threshold,
        mask_scale_ratio,
        comparison_sample_stride,
        comparison_max_samples,
        render_viz,
    )

    long_tail_area_threshold = resolve_long_tail_area_threshold(
        manifest=manifest,
        long_tail_box_prompt=long_tail_box_prompt,
        all_box_prompt=all_box_prompt,
    )

    try:
        os.chdir(DEFAULT_SAM2_ROOT)
    except Exception:
        pass
    sam2_cfg_resolved = resolve_sam2_config_path(sam2_cfg)
    predictor = build_sam2_video_predictor(sam2_cfg_resolved, sam2_ckpt)

    out_root = ensure_dir(output)
    subset_dir, subset_map = ensure_subset_video(
        context,
        data_path=data_path,
        out_root=out_root,
    )
    LOGGER.info("Selected frames available at %s", subset_dir)

    frame_index_to_name = {
        int(idx): str(name)
        for idx, name in zip(ssam_absolute_indices, ssam_frames)
    }

    level_results: List[LevelRunResult] = []
    overall_timer = TimingAggregator()

    for level in level_list:
        result = run_level_tracking(
            level=level,
            candidates_root=candidates_root,
            data_path=data_path,
            subset_dir=subset_dir,
            subset_map=subset_map,
            predictor=predictor,
            frame_index_lookup=frame_index_to_name,
            sam2_max_propagate=sam2_max_propagate,
            iou_threshold=iou_threshold,
            long_tail_box_prompt=long_tail_box_prompt,
            all_box_prompt=all_box_prompt,
            long_tail_area_threshold=long_tail_area_threshold,
            mask_scale_ratio=mask_scale_ratio,
            comparison_sample_stride=comparison_sample_stride,
            comparison_max_samples=comparison_max_samples,
            render_viz=render_viz,
            out_root=out_root,
        )
        level_results.append(result)
        overall_timer.merge(result.timer)

    if level_results:
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
            ) in (result.stats for result in level_results)
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

    update_manifest(
        context,
        out_root=out_root,
        level_results=level_results,
        mask_scale_ratio=mask_scale_ratio,
        render_viz=render_viz,
    )

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
