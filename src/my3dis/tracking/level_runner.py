"""Per-level execution helpers for SAM2 tracking runs."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

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
    frames_meta = filtered_manifest.get('frames', [])
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
    preview_local_indices = select_preview_indices(
        len(frame_numbers),
        stride=comparison_sample_stride,
        max_samples=comparison_max_samples,
    )
    preview_targets = {
        frame_numbers[idx] for idx in preview_local_indices if 0 <= idx < len(frame_numbers)
    }

    candidate_iter = iter_candidate_batches(
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
    LOGGER.info(
        "Level %d artifacts saved (video=%s, object=%s)",
        level,
        os.path.basename(artifacts['video_segments']) if artifacts.get('video_segments') else 'n/a',
        os.path.basename(artifacts['object_segments']) if artifacts.get('object_segments') else 'n/a',
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
