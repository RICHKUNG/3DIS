#!/usr/bin/env python3
"""
Rebuild provenance trees from index.json files.

This script fixes the issue where provenance_tree_L{level}.json is missing
objects that exist in the level's index.json file.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def rebuild_provenance_tree(run_dir: str, level: int) -> None:
    """
    Rebuild provenance_tree_L{level}.json from index.json.

    Args:
        run_dir: Path to scene run directory
        level: Level number (e.g., 2, 4, 6)
    """
    run_path = Path(run_dir)

    # Read index.json
    index_path = run_path / f'level_{level}' / 'index.json'
    if not index_path.exists():
        print(f"Error: index.json not found at {index_path}")
        return

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    objects = index_data.get('objects', {})
    print(f"Level {level}: Found {len(objects)} objects in index.json")

    # Load existing provenance tree (if exists) to preserve parent relationships
    prov_tree_path = run_path / 'relations' / f'provenance_tree_L{level}.json'
    existing_parent_map = {}

    if prov_tree_path.exists():
        with open(prov_tree_path, 'r') as f:
            old_tree = json.load(f)
            # Preserve existing parent relationships
            existing_parent_map = {
                int(k): v for k, v in old_tree.get('parent_map', {}).items()
            }

    # Build parent_map from index.json, but preserve existing parent relationships
    parent_map = {}
    for obj_id_str, obj_info in objects.items():
        obj_id = int(obj_id_str)
        # Use existing parent if available (to preserve cross-level relationships)
        # Otherwise use parent from index.json (same-level relationships)
        if obj_id in existing_parent_map:
            parent_map[obj_id] = existing_parent_map[obj_id]
        else:
            parent_id = obj_info.get('parent')
            parent_map[obj_id] = parent_id

    # Count orphans (objects with non-null parent that doesn't exist)
    all_obj_ids = set(parent_map.keys())
    orphans = []
    for obj_id, parent_id in parent_map.items():
        if parent_id is not None and parent_id not in all_obj_ids:
            orphans.append(obj_id)

    orphans.sort()

    # Read existing provenance tree (if exists) to preserve statistics
    relations_dir = run_path / 'relations'
    relations_dir.mkdir(exist_ok=True)

    prov_tree_path = relations_dir / f'provenance_tree_L{level}.json'

    statistics = {
        'total_ssam_masks': len(parent_map),
        'accepted_prompts': len(parent_map),
        'rejected_prompts': 0,
        'rejection_rate': 0.0,
        'unresolved_rejections': 0,
        'virtual_children_count': 0,
        'parents_with_virtual_children': 0,
    }

    if prov_tree_path.exists():
        with open(prov_tree_path, 'r') as f:
            old_tree = json.load(f)
            # Preserve statistics if available
            if 'statistics' in old_tree:
                old_stats = old_tree['statistics']
                statistics['rejected_prompts'] = old_stats.get('rejected_prompts', 0)
                statistics['rejection_rate'] = old_stats.get('rejection_rate', 0.0)
                statistics['unresolved_rejections'] = old_stats.get('unresolved_rejections', 0)
                statistics['virtual_children_count'] = old_stats.get('virtual_children_count', 0)
                statistics['parents_with_virtual_children'] = old_stats.get('parents_with_virtual_children', 0)
                statistics['total_ssam_masks'] = statistics['accepted_prompts'] + statistics['rejected_prompts']

    # Build new provenance tree
    new_tree = {
        'level': level,
        'parent_map': {str(k): v for k, v in sorted(parent_map.items())},
        'orphan_count': len(orphans),
        'orphan_ids': orphans,
        'statistics': statistics,
    }

    # Save
    with open(prov_tree_path, 'w', encoding='utf-8') as f:
        json.dump(new_tree, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved: {prov_tree_path}")
    print(f"  - Total objects: {len(parent_map)}")
    print(f"  - Orphans: {len(orphans)}")
    if orphans:
        print(f"  - Orphan IDs: {orphans[:10]}{'...' if len(orphans) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild provenance trees from index.json files'
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
        help='Levels to rebuild (default: 2 4 6)',
    )

    args = parser.parse_args()

    print(f"Rebuilding provenance trees for: {args.run_dir}")
    print(f"Levels: {args.levels}")
    print()

    for level in args.levels:
        rebuild_provenance_tree(args.run_dir, level)
        print()


if __name__ == '__main__':
    main()
