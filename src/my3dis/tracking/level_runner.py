"""Per-level execution helpers for SAM2 tracking runs."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from my3dis.common_utils import ensure_dir
from my3dis.tracking import (
    TimingAggregator,
    build_object_segments_archive,
    build_video_segments_archive,
    save_comparison_proposals,
)
from my3dis.tracking.candidate_loader import (
    iter_candidate_batches,
    load_filtered_frame_by_index,
    load_filtered_manifest,
    select_preview_indices,
)
from my3dis.tracking.pipeline_context import LevelRunResult
from my3dis.tracking.sam2_runner import TrackingArtifacts, sam2_tracking
from my3dis.tracking.stores import DedupStore, FrameResultStore

LOGGER = logging.getLogger("my3dis.track_from_candidates")

__all__ = [
    'persist_level_outputs',
    'run_level_tracking',
]


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
    selected_indices: Sequence[int],
    ssam_local_indices: Sequence[int],
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

    filtered_manifest = load_filtered_manifest(level_root)
    frames_meta_raw = filtered_manifest.get('frames', []) or []
    frames_meta: List[Dict[str, Any]] = []
    for entry in frames_meta_raw:
        if isinstance(entry, dict):
            frames_meta.append(dict(entry))

    selected_indices_seq = [int(idx) for idx in selected_indices]
    frame_numbers: List[int] = list(selected_indices_seq)

    ssam_locals = [int(idx) for idx in ssam_local_indices]
    for meta_idx, fm in enumerate(frames_meta):
        if meta_idx < len(ssam_locals):
            local_idx = int(ssam_locals[meta_idx])
        else:
            local_idx = meta_idx
        fm['local_index'] = local_idx

    frame_name_lookup: Dict[int, str] = {}
    for abs_idx, name in frame_index_lookup.items():
        try:
            frame_name_lookup[int(abs_idx)] = str(name)
        except (TypeError, ValueError):
            continue
    for abs_idx in frame_numbers:
        abs_idx_int = int(abs_idx)
        if abs_idx_int not in frame_name_lookup:
            frame_name_lookup[abs_idx_int] = f"{abs_idx_int:05d}.png"
    for fm in frames_meta:
        fidx = int(fm.get('frame_idx', 0))
        if fidx not in frame_name_lookup:
            fname = fm.get('frame_name')
            if fname is None:
                fname = f"{fidx:05d}.png"
            frame_name_lookup[fidx] = str(fname)

    LOGGER.info(
        "Level %d: Processing %d frames with SAM2 tracking...",
        level,
        len(frame_numbers),
    )

    level_timer = TimingAggregator()
    ssam_pairs: List[Tuple[int, int, int]] = []
    for meta_idx, fm in enumerate(frames_meta):
        local_idx = int(fm.get('local_index', meta_idx))
        if not (0 <= local_idx < len(frame_numbers)):
            LOGGER.warning(
                "Level %d: SSAM local index %d out of bounds (total selected=%d)",
                level,
                local_idx,
                len(frame_numbers),
            )
            continue
        abs_idx = int(frame_numbers[local_idx])
        ssam_pairs.append((meta_idx, local_idx, abs_idx))

    ssam_count = len(ssam_pairs)
    effective_max_samples = (
        comparison_max_samples if comparison_max_samples is not None else (ssam_count if ssam_count > 0 else None)
    )
    if ssam_count > 0:
        preview_positions = select_preview_indices(
            ssam_count,
            stride=comparison_sample_stride,
            max_samples=effective_max_samples,
        )
    else:
        preview_positions = []

    preview_meta_indices: List[int] = []
    preview_local_indices: List[int] = []
    preview_abs_indices: List[int] = []
    for pos in preview_positions:
        if 0 <= pos < ssam_count:
            meta_idx, local_idx, abs_idx = ssam_pairs[pos]
            preview_meta_indices.append(meta_idx)
            preview_local_indices.append(local_idx)
            preview_abs_indices.append(abs_idx)

    if preview_abs_indices:
        # Preserve ordering while removing duplicates.
        preview_abs_indices = list(dict.fromkeys(preview_abs_indices))
    preview_targets = set(preview_abs_indices)

    candidate_iter = iter_candidate_batches(
        level_root,
        frames_meta,
        mask_scale_ratio=mask_scale_ratio,
        local_indices=ssam_locals,
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
            ssam_local_indices=ssam_locals,
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
    LOGGER.info(
        "Level %d artifacts saved (video=%s, object=%s)",
        level,
        os.path.basename(artifacts['video_segments']) if artifacts.get('video_segments') else 'n/a',
        os.path.basename(artifacts['object_segments']) if artifacts.get('object_segments') else 'n/a',
    )

    filtered_preview: List[Optional[List[Dict[str, Any]]]] = [None] * len(frame_numbers)
    for meta_idx, local_idx in zip(preview_meta_indices, preview_local_indices):
        if not (0 <= meta_idx < len(frames_meta)):
            continue
        if not (0 <= local_idx < len(filtered_preview)):
            continue
        candidates_for_viz = load_filtered_frame_by_index(
            level_root,
            frames_meta,
            local_index=meta_idx,
            mask_scale_ratio=mask_scale_ratio,
        )
        if candidates_for_viz is not None:
            filtered_preview[local_idx] = candidates_for_viz

    comparison_result: Optional[Dict[str, Any]] = None
    tracker_warnings: List[Dict[str, Any]] = []
    if render_viz and preview_abs_indices:
        viz_dir = os.path.join(out_root, f'level_{level}', 'viz')
        with level_timer.track('viz.comparison'):
            frames_to_save = list(preview_abs_indices)
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
                max_samples=effective_max_samples,
                video_segments_archive=artifacts.get('video_segments'),
            )
    if comparison_result:
        summary_path = comparison_result.get('summary_path')
        if summary_path:
            artifacts['comparison_summary'] = os.path.relpath(summary_path, out_root)
        images_rel: List[str] = []
        rel_list = comparison_result.get('rendered_images_rel') or []
        if rel_list:
            viz_dir = os.path.join(out_root, f'level_{level}', 'viz')
            images_rel = [
                os.path.relpath(os.path.join(viz_dir, rel), out_root)
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
