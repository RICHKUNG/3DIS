"""
Legacy relation recovery tool.

Reconstructs parent-child hierarchies from old experiments that lack tree.json
by analyzing mask containment relationships across levels.

Usage:
    python -m my3dis.recover_relations \
        --experiment-dir /path/to/v2_experiment \
        --levels 2 4 6 \
        --containment-threshold 0.95

Author: Rich Kung
Created: 2025-10-21
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from my3dis.common_utils import unpack_binary_mask
from my3dis.relation_index import (
    RelationIndexWriter,
    build_cross_level_relations,
)

LOGGER = logging.getLogger(__name__)

__all__ = ['recover_legacy_relations', 'main']


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_containment(child: np.ndarray, parent: np.ndarray) -> float:
    """
    Compute what fraction of child is contained in parent.

    Returns:
        containment ratio: intersection(child, parent) / area(child)
    """
    child_area = child.sum()
    if child_area == 0:
        return 0.0
    intersection = np.logical_and(child, parent).sum()
    return float(intersection / child_area)


def load_object_masks_from_tracking(
    level_dir: Path,
    mask_scale_ratio: float = 1.0,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load all object masks from tracking outputs.

    Supports two layouts:
    - New: level_X/tracking/video_segments*.npz
    - Old (v2): level_X/video_segments*.npz

    Returns:
        {object_id: {frame_idx: mask_array}}
    """
    # Try new layout first
    tracking_dir = level_dir / 'tracking'
    search_dirs = [tracking_dir, level_dir]

    video_seg_path = None
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        candidates = list(search_dir.glob('video_segments*.npz'))
        if candidates:
            video_seg_path = candidates[0]
            break

    if video_seg_path is None:
        LOGGER.warning("No video_segments found in %s or %s/tracking", level_dir, level_dir)
        return {}

    LOGGER.info("Loading masks from %s", video_seg_path)

    object_masks: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)

    try:
        with zipfile.ZipFile(video_seg_path, 'r') as zf:
            manifest_data = zf.read('manifest.json')
            manifest = json.loads(manifest_data.decode('utf-8'))

            frames = manifest.get('frames', [])
            for frame_entry in tqdm(frames, desc="Loading masks", leave=False):
                frame_idx = int(frame_entry['frame_index'])
                entry_name = frame_entry['entry']

                frame_data = json.loads(zf.read(entry_name).decode('utf-8'))
                objects = frame_data.get('objects', {})

                for obj_id_str, packed_mask in objects.items():
                    obj_id = int(obj_id_str)
                    mask = unpack_binary_mask(packed_mask)
                    if mask is not None:
                        object_masks[obj_id][frame_idx] = np.asarray(mask, dtype=bool)

    except Exception as exc:
        LOGGER.error("Failed to load masks from %s: %s", video_seg_path, exc)
        return {}

    LOGGER.info("Loaded %d objects", len(object_masks))
    return object_masks


def find_parent_candidates(
    child_id: int,
    child_masks: Dict[int, np.ndarray],
    parent_level_objects: Dict[int, Dict[int, np.ndarray]],
    containment_threshold: float = 0.95,
) -> List[Tuple[int, float]]:
    """
    Find parent candidates by checking containment in overlapping frames.

    Returns:
        List of (parent_id, avg_containment) sorted by containment desc.
    """
    # Find frames where child appears
    child_frames = set(child_masks.keys())

    parent_scores: Dict[int, List[float]] = defaultdict(list)

    for parent_id, parent_masks in parent_level_objects.items():
        parent_frames = set(parent_masks.keys())
        common_frames = child_frames & parent_frames

        if not common_frames:
            continue

        for frame_idx in common_frames:
            child_mask = child_masks[frame_idx]
            parent_mask = parent_masks[frame_idx]

            containment = compute_containment(child_mask, parent_mask)
            if containment >= containment_threshold:
                parent_scores[parent_id].append(containment)

    # Compute average containment per parent
    parent_candidates: List[Tuple[int, float]] = []
    for parent_id, scores in parent_scores.items():
        if not scores:
            continue
        avg_containment = sum(scores) / len(scores)
        parent_candidates.append((parent_id, avg_containment))

    # Sort by containment descending
    parent_candidates.sort(key=lambda x: x[1], reverse=True)
    return parent_candidates


def recover_level_hierarchy(
    level_dir: Path,
    prev_level_objects: Optional[Dict[int, Dict[int, np.ndarray]]],
    containment_threshold: float = 0.95,
    mask_scale_ratio: float = 1.0,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], List[Dict[str, Any]]]:
    """
    Recover parent-child relationships for a single level.

    Returns:
        (level_objects, tree_nodes)
    """
    # Load this level's objects (handles both old v2 and new layouts)
    level_objects = load_object_masks_from_tracking(level_dir, mask_scale_ratio)

    if not level_objects:
        LOGGER.warning("No objects loaded for %s", level_dir)
        return {}, []

    tree_nodes: List[Dict[str, Any]] = []

    for obj_id in sorted(level_objects.keys()):
        obj_masks = level_objects[obj_id]

        parent_id: Optional[int] = None
        children: List[int] = []

        # Find parent from previous (coarser) level
        if prev_level_objects:
            candidates = find_parent_candidates(
                obj_id,
                obj_masks,
                prev_level_objects,
                containment_threshold,
            )
            if candidates:
                parent_id, containment = candidates[0]
                LOGGER.debug(
                    "Object %d → parent %d (containment=%.3f)",
                    obj_id,
                    parent_id,
                    containment,
                )

        node = {
            'id': int(obj_id),
            'parent': int(parent_id) if parent_id is not None else None,
            'children': children,
        }
        tree_nodes.append(node)

    # Build children lists (will be populated by next level)
    return level_objects, tree_nodes


def recover_legacy_relations(
    experiment_dir: str | Path,
    levels: List[int],
    containment_threshold: float = 0.95,
    mask_scale_ratio: float = 1.0,
) -> Path:
    """
    Recover parent-child relations for a legacy experiment.

    Args:
        experiment_dir: Path to experiment root (e.g., outputs/experiments/v2_*/scene_*/run_name)
        levels: List of levels to process (e.g., [2, 4, 6])
        containment_threshold: Minimum containment ratio to consider parent-child (default 0.95)
        mask_scale_ratio: Mask downscale ratio used in tracking (default 1.0)

    Returns:
        Path to generated relations.json
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    LOGGER.info("Recovering relations for %s", experiment_dir)
    LOGGER.info("Levels: %s", levels)
    LOGGER.info("Containment threshold: %.2f", containment_threshold)

    hierarchy: Dict[int, List[Dict[str, Any]]] = {}
    prev_level_objects: Optional[Dict[int, Dict[int, np.ndarray]]] = None

    for level in sorted(levels):
        level_dir = experiment_dir / f'level_{level}'
        if not level_dir.exists():
            LOGGER.warning("Level %d directory not found, skipping", level)
            continue

        LOGGER.info("Processing level %d...", level)
        level_objects, tree_nodes = recover_level_hierarchy(
            level_dir,
            prev_level_objects,
            containment_threshold,
            mask_scale_ratio,
        )

        if not tree_nodes:
            LOGGER.warning("No objects found for level %d", level)
            continue

        hierarchy[level] = tree_nodes

        # Update children lists for previous level
        if prev_level_objects:
            prev_level = levels[levels.index(level) - 1]
            prev_tree = hierarchy.get(prev_level, [])
            id_to_node = {node['id']: node for node in prev_tree}

            for node in tree_nodes:
                parent_id = node['parent']
                if parent_id is not None and parent_id in id_to_node:
                    if node['id'] not in id_to_node[parent_id]['children']:
                        id_to_node[parent_id]['children'].append(node['id'])

        # Save per-level tree.json
        relations_dir = level_dir / 'relations'
        relations_dir.mkdir(exist_ok=True)

        tree_path = relations_dir / 'tree.json'
        with tree_path.open('w', encoding='utf-8') as f:
            json.dump(tree_nodes, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved tree for level %d: %d nodes → %s", level, len(tree_nodes), tree_path)

        # Build level index
        LOGGER.info("Building index for level %d...", level)
        from my3dis.relation_index import build_level_index

        # Search for NPZ files in both new and old layouts
        video_seg_path = None
        object_seg_path = None

        for search_dir in [level_dir / 'tracking', level_dir]:
            if not search_dir.exists():
                continue
            video_segs = list(search_dir.glob('video_segments*.npz'))
            object_segs = list(search_dir.glob('object_segments*.npz'))
            if video_segs and video_seg_path is None:
                video_seg_path = video_segs[0]
            if object_segs and object_seg_path is None:
                object_seg_path = object_segs[0]
            if video_seg_path and object_seg_path:
                break

        index_path = build_level_index(
            level=level,
            level_root=level_dir,
            video_segments_path=video_seg_path,
            object_segments_path=object_seg_path,
            mask_scale_ratio=mask_scale_ratio,
        )
        LOGGER.info("Saved index for level %d → %s", level, index_path)

        prev_level_objects = level_objects

    # Build cross-level relations.json
    LOGGER.info("Building cross-level relations...")
    relations_path = build_cross_level_relations(experiment_dir, levels)
    LOGGER.info("✅ Recovery complete → %s", relations_path)

    return relations_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover parent-child relations from legacy experiments"
    )
    parser.add_argument(
        '--experiment-dir',
        required=True,
        help='Path to experiment directory (e.g., outputs/experiments/v2_*/scene_*/run_name)',
    )
    parser.add_argument(
        '--levels',
        type=str,
        default='2,4,6',
        help='Comma-separated list of levels (default: 2,4,6)',
    )
    parser.add_argument(
        '--containment-threshold',
        type=float,
        default=0.95,
        help='Minimum containment ratio for parent-child (default: 0.95)',
    )
    parser.add_argument(
        '--mask-scale-ratio',
        type=float,
        default=0.3,
        help='Mask downscale ratio used in tracking (default: 0.3)',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable debug logging',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    levels = [int(x.strip()) for x in args.levels.split(',')]

    recover_legacy_relations(
        experiment_dir=args.experiment_dir,
        levels=levels,
        containment_threshold=args.containment_threshold,
        mask_scale_ratio=args.mask_scale_ratio,
    )


if __name__ == '__main__':
    main()
