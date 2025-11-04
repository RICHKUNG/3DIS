"""
Unified Family Tree Builder - Creates per-scene family hierarchies.

This module aggregates provenance trees from all levels into a single unified
JSON structure that contains:
- Complete parent-child relationships across all levels
- Frame appearance information for each object
- Mask archive paths for querying
- Family groupings and statistics
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np


def load_provenance_tree(tree_path: str) -> Dict[str, Any]:
    """Load a per-level provenance tree JSON file."""
    with open(tree_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_video_segments_frames(npz_path: str) -> Dict[int, List[int]]:
    """
    Load frame indices for each object from video_segments NPZ archive.

    Supports multiple NPZ formats:
    - ZIP archive with "frames/frame_XXXXXX.json" entries (new format)
    - Direct numpy arrays with "frame_XXXXXX" keys (legacy format)

    Returns:
        Dict mapping object_id -> list of frame indices where it appears
    """
    if not os.path.exists(npz_path):
        return {}

    try:
        archive = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load NPZ archive {npz_path}: {e}")
        return {}

    object_frames: Dict[int, List[int]] = defaultdict(list)

    # Detect format by examining first key
    if not archive.files:
        return {}

    first_key = archive.files[0]

    if first_key.startswith("frames/frame_"):
        # New ZIP format with JSON entries
        for key in archive.files:
            if not key.startswith("frames/frame_"):
                continue

            try:
                # Load JSON data from ZIP entry
                data_bytes = bytes(archive[key])
                frame_data = json.loads(data_bytes.decode('utf-8'))

                frame_idx = frame_data['frame_index']
                objects_dict = frame_data.get('objects', {})

                for obj_id_str in objects_dict.keys():
                    try:
                        obj_id = int(obj_id_str)
                        object_frames[obj_id].append(frame_idx)
                    except (ValueError, TypeError):
                        continue
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                continue

    elif first_key.startswith("frame_"):
        # Legacy numpy array format
        for key in archive.files:
            if not key.startswith("frame_"):
                continue

            try:
                frame_idx = int(key.split("_")[1])
                frame_data = archive[key].item()

                if isinstance(frame_data, dict):
                    for obj_id_str in frame_data.keys():
                        try:
                            obj_id = int(obj_id_str)
                            object_frames[obj_id].append(frame_idx)
                        except (ValueError, TypeError):
                            continue
            except (ValueError, IndexError, AttributeError):
                continue

    # Sort frame lists for each object
    for obj_id in object_frames:
        object_frames[obj_id].sort()

    return dict(object_frames)


def build_family_tree(
    run_dir: str,
    levels: List[int],
    mask_scale_ratio: float = 0.3,
) -> Dict[str, Any]:
    """
    Build unified family tree from per-level provenance trees.

    Args:
        run_dir: Path to scene run directory (e.g., outputs/experiments/scene_00005_00/run_name)
        levels: List of levels to include (e.g., [2, 4, 6])
        mask_scale_ratio: Scale ratio used for mask archives (default: 0.3)

    Returns:
        Unified family tree dictionary
    """
    run_path = Path(run_dir)
    scene_name = run_path.parent.name if run_path.parent.name.startswith('scene_') else 'unknown'

    # Load all provenance trees
    level_trees: Dict[int, Dict[str, Any]] = {}
    for level in levels:
        tree_path = run_path / 'relations' / f'provenance_tree_L{level}.json'
        if tree_path.exists():
            level_trees[level] = load_provenance_tree(str(tree_path))

    if not level_trees:
        raise FileNotFoundError(f"No provenance trees found in {run_dir}/relations/")

    # Build unified object registry
    objects: Dict[int, Dict[str, Any]] = {}
    all_parent_map: Dict[int, Optional[int]] = {}
    all_virtual_children: Dict[int, List[str]] = {}

    for level, tree_data in level_trees.items():
        parent_map = tree_data.get('parent_map', {})

        # Convert string keys to int
        for obj_id_str, parent_id in parent_map.items():
            obj_id = int(obj_id_str)
            all_parent_map[obj_id] = int(parent_id) if parent_id is not None else None

        # Load virtual children from provenance tree
        virtual_children = tree_data.get('virtual_children', {})
        for parent_id_str, children_list in virtual_children.items():
            parent_id = int(parent_id_str)
            if parent_id not in all_virtual_children:
                all_virtual_children[parent_id] = []
            all_virtual_children[parent_id].extend(children_list)

        # Load frame information from video segments
        # Try multiple path formats (backward compatibility + new formats)
        # Priority: New format first, then old formats
        video_segments_candidates = [
            # New format (L-prefixed, directly in level dir)
            run_path / f'level_{level}' / f'video_segments_L{level:02d}.npz',
            # New format variant (in tracking subdir)
            run_path / f'level_{level}' / 'tracking' / f'video_segments_L{level:02d}.npz',
            # Old format (scale-based naming)
            run_path / f'level_{level}' / 'tracking' / f'video_segments_scale{mask_scale_ratio}x.npz',
            run_path / f'level_{level}' / f'video_segments_scale{mask_scale_ratio}x.npz',
        ]

        video_segments_path = None
        for candidate in video_segments_candidates:
            if candidate.exists():
                video_segments_path = candidate
                break

        if video_segments_path is None:
            # Provide detailed error message
            import warnings
            warnings.warn(
                f"No video segments NPZ found for level {level}. Tried:\n" +
                "\n".join(f"  - {p.relative_to(run_path)}" for p in video_segments_candidates)
            )
            object_frames = {}
        else:
            object_frames = load_video_segments_frames(str(video_segments_path))

        # Register objects
        for obj_id in parent_map.keys():
            obj_id_int = int(obj_id)
            frames = object_frames.get(obj_id_int, [])

            # BUGFIX: Only register if not already present (prevent overwriting with wrong level)
            if obj_id_int not in objects:
                # Store relative path from run_dir for portability
                mask_archive_rel = str(video_segments_path.relative_to(run_path)) if video_segments_path else None

                objects[obj_id_int] = {
                    'level': level,
                    'parent_id': all_parent_map.get(obj_id_int),
                    'children': [],  # Will be filled below
                    'virtual_children': [],  # Will be filled below
                    'frames': frames,
                    'mask_archive': mask_archive_rel,
                    'first_frame': min(frames) if frames else None,
                    'last_frame': max(frames) if frames else None,
                    'total_frames': len(frames),
                }

    # Build children lists
    for obj_id, parent_id in all_parent_map.items():
        if parent_id is not None and parent_id in objects:
            objects[parent_id]['children'].append(obj_id)

    # Sort children lists
    for obj_data in objects.values():
        obj_data['children'].sort()

    # Build virtual_children lists
    for parent_id, virtual_child_ids in all_virtual_children.items():
        if parent_id in objects:
            # Deduplicate and sort
            unique_virtual = sorted(set(virtual_child_ids))
            objects[parent_id]['virtual_children'] = unique_virtual

    # Identify families (connected components in parent-child graph)
    families = build_families(objects, all_parent_map)

    # Collect orphans
    orphans = [
        obj_id
        for obj_id, parent_id in all_parent_map.items()
        if parent_id is not None and parent_id not in objects
    ]
    orphans.sort()

    # Statistics
    objects_per_level = defaultdict(int)
    total_virtual_children = 0
    parents_with_virtual = 0
    for obj_data in objects.values():
        objects_per_level[f"L{obj_data['level']}"] += 1
        virtual_children_count = len(obj_data.get('virtual_children', []))
        if virtual_children_count > 0:
            total_virtual_children += virtual_children_count
            parents_with_virtual += 1

    return {
        'scene': scene_name,
        'run_dir': str(run_path.name),
        'levels': sorted(levels),
        'mask_scale_ratio': mask_scale_ratio,
        'objects': {str(k): v for k, v in sorted(objects.items())},
        'families': families,
        'statistics': {
            'total_objects': len(objects),
            'total_families': len(families),
            'objects_per_level': dict(objects_per_level),
            'orphan_count': len(orphans),
            'orphans': orphans,
            'virtual_children_count': total_virtual_children,
            'parents_with_virtual_children': parents_with_virtual,
        },
    }


def build_families(
    objects: Dict[int, Dict[str, Any]],
    parent_map: Dict[int, Optional[int]],
) -> List[Dict[str, Any]]:
    """
    Identify family groups (connected components) in the object hierarchy.

    Returns:
        List of family dictionaries with root_id, members, and level groupings
    """
    # Find all root objects (no parent or parent not in objects)
    roots: Set[int] = set()
    for obj_id, parent_id in parent_map.items():
        if parent_id is None or parent_id not in objects:
            roots.add(obj_id)

    families: List[Dict[str, Any]] = []

    for root_id in sorted(roots):
        # Collect all descendants
        members = collect_descendants(root_id, objects)
        members_sorted = sorted(members)

        # Group by level
        level_groups: Dict[str, List[int]] = defaultdict(list)
        for obj_id in members:
            if obj_id in objects:
                level = objects[obj_id]['level']
                level_groups[f"L{level}"].append(obj_id)

        # Sort each level group
        for level_key in level_groups:
            level_groups[level_key].sort()

        families.append({
            'root_id': root_id,
            'members': members_sorted,
            'levels': dict(level_groups),
            'total_members': len(members),
        })

    return families


def collect_descendants(root_id: int, objects: Dict[int, Dict[str, Any]]) -> Set[int]:
    """Recursively collect all descendants of a root object."""
    descendants = {root_id}

    if root_id not in objects:
        return descendants

    children = objects[root_id].get('children', [])
    for child_id in children:
        descendants.update(collect_descendants(child_id, objects))

    return descendants


def save_family_tree(tree: Dict[str, Any], output_path: str) -> None:
    """Save family tree to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point for building family trees."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build unified family tree from per-level provenance trees'
    )
    parser.add_argument(
        '--run-dir',
        required=True,
        help='Path to scene run directory',
    )
    parser.add_argument(
        '--levels',
        type=int,
        nargs='+',
        default=[2, 4, 6],
        help='Levels to include (default: 2 4 6)',
    )
    parser.add_argument(
        '--mask-scale-ratio',
        type=float,
        default=0.3,
        help='Mask scale ratio used in tracking (default: 0.3)',
    )
    parser.add_argument(
        '--output',
        help='Output JSON path (default: <run_dir>/relations/family_tree.json)',
    )

    args = parser.parse_args()

    tree = build_family_tree(
        args.run_dir,
        args.levels,
        args.mask_scale_ratio,
    )

    output_path = args.output or os.path.join(args.run_dir, 'relations', 'family_tree.json')
    save_family_tree(tree, output_path)

    print(f"✓ Family tree saved: {output_path}")
    print(f"  - Total objects: {tree['statistics']['total_objects']}")
    print(f"  - Total families: {tree['statistics']['total_families']}")
    print(f"  - Orphans: {tree['statistics']['orphan_count']}")
    print(f"  - Virtual children: {tree['statistics']['virtual_children_count']}")
    print(f"  - Parents with virtual children: {tree['statistics']['parents_with_virtual_children']}")
    print(f"  - Per level: {tree['statistics']['objects_per_level']}")


if __name__ == '__main__':
    main()
