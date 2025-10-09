"""Context helpers and manifest utilities for SAM2 tracking runs."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from my3dis.common_utils import build_subset_video

LOGGER = logging.getLogger("my3dis.track_from_candidates")


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
    timer: "TimingAggregator"
    duration: float


def prepare_tracking_context(
    *,
    candidates_root: str,
    level_list: List[int],
    sam2_max_propagate: Optional[int],
) -> TrackingContext:
    """Load manifest metadata and normalise values for downstream helpers."""

    manifest_path = os.path.join(candidates_root, 'manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
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
                "Environment override: MY3DIS_LONG_TAIL_AREA=%d",
                long_tail_area_threshold,
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
            if os.path.splitext(f)[1] in {".jpg", ".jpeg", ".JPG", ".JPEG"}
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
        with open(context.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        LOGGER.warning('Failed to update manifest at %s', context.manifest_path, exc_info=True)


__all__ = [
    'LevelRunResult',
    'TrackingContext',
    'ensure_subset_video',
    'prepare_tracking_context',
    'resolve_long_tail_area_threshold',
    'update_manifest',
]
