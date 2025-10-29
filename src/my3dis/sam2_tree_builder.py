"""
SAM2-based tree building using containment analysis.

Builds parent-child hierarchy from SAM2 tracked objects by analyzing mask
containment relationships in temporally overlapping frames.

Key principles:
- Only analyze masks that appear in the same frames (temporal overlap)
- Compute containment only on co-occurring frames
- Use average containment across all common frames for robustness

Author: Rich Kung
Created: 2025-10-22
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from my3dis.common_utils import ensure_dir

LOGGER = logging.getLogger(__name__)

__all__ = [
    'build_sam2_containment_tree',
    'compute_containment',
]


def compute_containment(child: np.ndarray, parent: np.ndarray) -> float:
    """
    Compute what fraction of child is contained in parent.

    Args:
        child: Binary mask array (H, W)
        parent: Binary mask array (H, W)

    Returns:
        Containment ratio: intersection(child, parent) / area(child)
        Returns 0.0 if child is empty.
    """
    child_area = child.sum()
    if child_area == 0:
        return 0.0

    intersection = np.logical_and(child, parent).sum()
    return float(intersection / child_area)


def load_object_masks_from_index(
    level_dir: Path,
) -> Tuple[Dict[int, Set[int]], Dict[str, Any]]:
    """
    Load object→frames mapping from index.json.

    Args:
        level_dir: Path to level_X directory

    Returns:
        (object_frames, index_data)
        - object_frames: {obj_id: set of frame indices}
        - index_data: Full index.json content
    """
    index_path = level_dir / 'index.json'
    if not index_path.exists():
        LOGGER.warning("Missing index.json at %s", index_path)
        return {}, {}

    with index_path.open('r', encoding='utf-8') as f:
        index_data = json.load(f)

    objects = index_data.get('objects', {})
    object_frames: Dict[int, Set[int]] = {}

    for obj_id_str, obj_data in objects.items():
        obj_id = int(obj_id_str)
        frames = set(obj_data.get('frames', []))
        object_frames[obj_id] = frames

    LOGGER.debug("Loaded %d objects from %s", len(object_frames), index_path)
    return object_frames, index_data


def load_masks_from_video_segments(
    level_dir: Path,
    object_ids: Optional[Set[int]] = None,
    index_data: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load object masks from video_segments_L{level}.npz.

    Args:
        level_dir: Path to level_X directory
        object_ids: Optional set of object IDs to load (all if None)

    Returns:
        {object_id: {frame_idx: mask_array}}
    """
    import zipfile
    from my3dis.common_utils import unpack_binary_mask

    # Try to find video_segments file
    explicit_candidates: List[Path] = []
    if index_data:
        sources = index_data.get('meta', {}).get('sources', {})
        video_rel = sources.get('video_segments')
        if isinstance(video_rel, str) and video_rel:
            candidate_path = Path(video_rel)
            if not candidate_path.is_absolute():
                candidate_path = level_dir / candidate_path
            explicit_candidates.append(candidate_path)

    level = None
    if 'level_' in level_dir.name:
        try:
            level = int(level_dir.name.split('_')[1])
        except (ValueError, IndexError):
            pass

    # Search patterns
    patterns: List[str] = []
    if level is not None:
        patterns.append(f'video_segments_L{level:02d}.npz')
    patterns.append('video_segments_scale*.npz')
    patterns.append('video_segments*.npz')

    search_dirs: List[Path] = [level_dir]
    tracking_dir = level_dir / 'tracking'
    if tracking_dir.exists():
        search_dirs.append(tracking_dir)

    attempted: List[str] = []
    video_seg_path: Optional[Path] = None

    # Prefer explicit sources recorded in index.json
    for candidate in explicit_candidates:
        attempted.append(str(candidate))
        if candidate.exists():
            video_seg_path = candidate
            LOGGER.debug("Found video segments via index source: %s", video_seg_path)
            break

    if video_seg_path is None:
        for base_dir in search_dirs:
            for pattern in patterns:
                attempted.append(f"{base_dir}/{pattern}")
                candidates = list(base_dir.glob(pattern))
                if candidates:
                    video_seg_path = candidates[0]
                    LOGGER.debug("Found video segments: %s", video_seg_path)
                    break
            if video_seg_path is not None:
                break

    if video_seg_path is None:
        LOGGER.warning(
            "No video_segments found in %s (checked: %s)",
            level_dir,
            ', '.join(attempted) or 'n/a'
        )
        return {}


    object_masks: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)

    try:
        with zipfile.ZipFile(video_seg_path, 'r') as zf:
            manifest_data = zf.read('manifest.json')
            manifest = json.loads(manifest_data.decode('utf-8'))

            frames = manifest.get('frames', [])
            for frame_entry in tqdm(frames, desc=f"Loading masks L{level}", leave=False):
                frame_idx = int(frame_entry['frame_index'])
                entry_name = frame_entry['entry']

                frame_data = json.loads(zf.read(entry_name).decode('utf-8'))
                objects = frame_data.get('objects', {})

                for obj_id_str, packed_mask in objects.items():
                    obj_id = int(obj_id_str)

                    # Skip if not in requested set
                    if object_ids is not None and obj_id not in object_ids:
                        continue

                    mask = unpack_binary_mask(packed_mask)
                    if mask is not None:
                        object_masks[obj_id][frame_idx] = np.asarray(mask, dtype=bool)

    except Exception as exc:
        LOGGER.error("Failed to load masks from %s: %s", video_seg_path, exc)
        return {}

    LOGGER.info("Loaded masks for %d objects from %s", len(object_masks), video_seg_path)
    return object_masks


def find_best_parent(
    child_id: int,
    child_frames: Set[int],
    child_masks: Dict[int, np.ndarray],
    parent_candidates: Dict[int, Dict[str, Any]],
    containment_threshold: float,
    temporal_overlap_threshold: float,
) -> Optional[Tuple[int, float]]:
    """
    Find the best parent for a child object based on containment analysis.

    Args:
        child_id: Object ID of the child
        child_frames: Set of frame indices where child appears
        child_masks: {frame_idx: mask_array} for child
        parent_candidates: {parent_id: {'frames': set, 'masks': dict}}
        containment_threshold: Minimum avg containment to consider parent-child
        temporal_overlap_threshold: Minimum ratio of overlapping frames

    Returns:
        (best_parent_id, avg_containment) or None if no valid parent found
    """
    if not child_frames:
        return None

    best_parent = None
    best_score = 0.0

    for parent_id, parent_data in parent_candidates.items():
        parent_frames = parent_data['frames']
        parent_masks = parent_data['masks']

        # Check temporal overlap
        common_frames = child_frames & parent_frames
        if not common_frames:
            continue

        # Require sufficient temporal overlap
        overlap_ratio = len(common_frames) / len(child_frames)
        if overlap_ratio < temporal_overlap_threshold:
            LOGGER.debug(
                "Child %d → Parent %d: insufficient temporal overlap (%.2f < %.2f)",
                child_id, parent_id, overlap_ratio, temporal_overlap_threshold
            )
            continue

        # Compute containment in common frames
        containments = []
        for frame_idx in common_frames:
            if frame_idx not in child_masks or frame_idx not in parent_masks:
                continue

            child_mask = child_masks[frame_idx]
            parent_mask = parent_masks[frame_idx]

            containment = compute_containment(child_mask, parent_mask)
            containments.append(containment)

        if not containments:
            LOGGER.debug(
                "Child %d → Parent %d: no valid mask pairs in common frames",
                child_id, parent_id
            )
            continue

        avg_containment = sum(containments) / len(containments)

        LOGGER.debug(
            "Child %d → Parent %d: avg_containment=%.3f (frames=%d, overlap=%.2f)",
            child_id, parent_id, avg_containment, len(common_frames), overlap_ratio
        )

        # Accept if above threshold and better than current best
        if avg_containment >= containment_threshold and avg_containment > best_score:
            best_parent = parent_id
            best_score = avg_containment

    if best_parent is not None:
        LOGGER.info(
            "Child %d → Parent %d selected (containment=%.3f)",
            child_id, best_parent, best_score
        )
        return (best_parent, best_score)

    return None


def build_sam2_containment_tree(
    run_dir: str | Path,
    levels: List[int],
    containment_threshold: float = 0.95,
    temporal_overlap_threshold: float = 0.3,
    output_filename: str = 'sam2_tree.json',
) -> Path:
    """
    Build tree from SAM2 tracked objects using containment analysis.

    Args:
        run_dir: Scene run directory (e.g., outputs/.../scene_00065_00)
        levels: List of levels to process (e.g., [2, 4, 6])
        containment_threshold: Minimum avg containment for parent-child (0-1)
        temporal_overlap_threshold: Minimum temporal overlap ratio (0-1)
        output_filename: Output JSON filename (default: sam2_tree.json)

    Returns:
        Path to generated tree JSON file

    Algorithm:
        1. Load all tracked objects from SAM2 outputs (index.json)
        2. Load masks from video_segments_*.npz
        3. For each level pair (coarse → fine):
            a. Find temporal overlap between objects
            b. Compute containment only in overlapping frames
            c. Assign parent if avg_containment >= threshold
        4. Build cross-level hierarchy and paths
        5. Save as relations.json v3.0 format
    """
    run_dir = Path(run_dir)
    LOGGER.info("Building SAM2-based tree for %s", run_dir)
    LOGGER.info("Levels: %s", levels)
    LOGGER.info("Containment threshold: %.2f", containment_threshold)
    LOGGER.info("Temporal overlap threshold: %.2f", temporal_overlap_threshold)

    hierarchy: Dict[int, Dict[int, Dict[str, Any]]] = {}

    # Step 1: Load object frames for all levels
    LOGGER.info("Loading object metadata...")
    level_data: Dict[int, Dict[str, Any]] = {}
    for level in levels:
        level_dir = run_dir / f'level_{level}'
        object_frames, index_data = load_object_masks_from_index(level_dir)

        level_data[level] = {
            'dir': level_dir,
            'object_frames': object_frames,
            'index': index_data,
        }

        LOGGER.info("Level %d: %d objects", level, len(object_frames))

    # Step 2: Process each level to build initial hierarchy
    LOGGER.info("Building initial hierarchy...")
    for level in levels:
        hierarchy[level] = {}
        object_frames = level_data[level]['object_frames']

        for obj_id, frames in object_frames.items():
            hierarchy[level][obj_id] = {
                'parent': None,
                'children': [],
                'frames': frames,
            }

    # Step 3: Find parent-child relationships across levels
    LOGGER.info("Finding parent-child relationships...")
    for i, (coarse_level, fine_level) in enumerate(zip(levels[:-1], levels[1:])):
        LOGGER.info("Analyzing Level %d → Level %d...", coarse_level, fine_level)

        # Load masks for both levels
        coarse_dir = level_data[coarse_level]['dir']
        fine_dir = level_data[fine_level]['dir']

        LOGGER.info("Loading coarse level (%d) masks...", coarse_level)
        coarse_masks = load_masks_from_video_segments(
            coarse_dir,
            index_data=level_data[coarse_level]['index'],
        )

        LOGGER.info("Loading fine level (%d) masks...", fine_level)
        fine_masks = load_masks_from_video_segments(
            fine_dir,
            index_data=level_data[fine_level]['index'],
        )

        # Prepare parent candidates
        parent_candidates = {}
        for parent_id in hierarchy[coarse_level]:
            parent_candidates[parent_id] = {
                'frames': hierarchy[coarse_level][parent_id]['frames'],
                'masks': coarse_masks.get(parent_id, {}),
            }

        # Find parent for each fine-level object
        fine_objects = hierarchy[fine_level]
        for child_id in tqdm(fine_objects, desc=f"L{coarse_level}→L{fine_level}"):
            child_frames = fine_objects[child_id]['frames']
            child_masks_dict = fine_masks.get(child_id, {})

            result = find_best_parent(
                child_id=child_id,
                child_frames=child_frames,
                child_masks=child_masks_dict,
                parent_candidates=parent_candidates,
                containment_threshold=containment_threshold,
                temporal_overlap_threshold=temporal_overlap_threshold,
            )

            if result is not None:
                parent_id, containment = result
                fine_objects[child_id]['parent'] = parent_id
                fine_objects[child_id]['parent_containment'] = containment

                # Add to parent's children list
                hierarchy[coarse_level][parent_id]['children'].append(child_id)

    # Step 4: Build ancestor paths
    LOGGER.info("Building ancestor paths...")
    paths: Dict[int, List[int]] = {}

    def build_path(obj_id: int, level: int) -> List[int]:
        """Recursively build path from root to object."""
        if level not in hierarchy or obj_id not in hierarchy[level]:
            return [obj_id]

        parent_id = hierarchy[level][obj_id].get('parent')
        if parent_id is None:
            return [obj_id]

        # Find parent's level
        parent_level = None
        for lv in sorted(levels):
            if lv >= level:
                break
            if lv in hierarchy and parent_id in hierarchy[lv]:
                parent_level = lv
                break

        if parent_level is None:
            return [obj_id]

        parent_path = build_path(parent_id, parent_level)
        return parent_path + [obj_id]

    for level in levels:
        for obj_id in hierarchy[level]:
            paths[obj_id] = build_path(obj_id, level)

    # Step 5: Compute statistics
    LOGGER.info("Computing statistics...")
    total_objects = len(paths)
    roots = sum(
        1 for level_objs in hierarchy.values()
        for obj in level_objs.values()
        if obj.get('parent') is None
    )
    families = sum(
        1 for level_objs in hierarchy.values()
        for obj in level_objs.values()
        if obj.get('children')
    )
    max_depth = max((len(path) for path in paths.values()), default=0)

    LOGGER.info("Total objects: %d", total_objects)
    LOGGER.info("Root objects: %d", roots)
    LOGGER.info("Family objects (with children): %d", families)
    LOGGER.info("Max tree depth: %d", max_depth)

    # Step 6: Save to JSON
    output_dir = run_dir / 'relations'
    ensure_dir(output_dir)
    output_path = output_dir / output_filename

    # Clean up hierarchy for serialization (remove frames/masks, convert sets)
    clean_hierarchy = {}
    for level, level_objs in hierarchy.items():
        clean_hierarchy[str(level)] = {}
        for obj_id, obj_data in level_objs.items():
            clean_hierarchy[str(level)][str(obj_id)] = {
                'parent': obj_data['parent'],
                'children': obj_data['children'],
                'descendant_count': len([
                    oid for oid in paths
                    if obj_id in paths[oid] and oid != obj_id
                ])
            }
            # Include parent_containment if available
            if 'parent_containment' in obj_data:
                clean_hierarchy[str(level)][str(obj_id)]['parent_containment'] = (
                    round(obj_data['parent_containment'], 4)
                )

    payload = {
        'schema_version': '3.0',
        'meta': {
            'scene_id': run_dir.parent.name,
            'experiment': run_dir.parent.name,
            'run_name': run_dir.name,
            'levels': sorted(levels),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'generator': 'my3dis.sam2_tree_builder.build_sam2_containment_tree',
            'parameters': {
                'containment_threshold': containment_threshold,
                'temporal_overlap_threshold': temporal_overlap_threshold,
            },
            'note': 'Tree built from SAM2 tracked objects using containment analysis',
        },
        'hierarchy': clean_hierarchy,
        'paths': {str(oid): path for oid, path in sorted(paths.items())},
        'statistics': {
            'total_objects': total_objects,
            'roots': roots,
            'families': families,
            'max_depth': max_depth,
        }
    }

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    LOGGER.info("✅ SAM2 tree saved → %s", output_path)
    LOGGER.info(
        "Tree summary: %d objects, %d roots, %d families, depth %d",
        total_objects, roots, families, max_depth
    )

    return output_path
