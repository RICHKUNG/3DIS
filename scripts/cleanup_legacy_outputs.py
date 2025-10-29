#!/usr/bin/env python3
"""
Clean up legacy output files from My3DIS experiments.

This script removes:
1. SSAM tree files (tree_L*.json, tree.json) - replaced by SAM2 tree
2. Legacy relations.json (generated from SSAM trees)
3. Orphaned temporary directories

Usage:
    python scripts/cleanup_legacy_outputs.py <experiment_dir> [--dry-run]

Example:
    python scripts/cleanup_legacy_outputs.py outputs/experiments/test_1022_4
    python scripts/cleanup_legacy_outputs.py outputs/experiments/test_1022_4 --dry-run
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def find_ssam_tree_files(scene_dir: Path) -> List[Path]:
    """Find SSAM tree files in a scene directory."""
    relations_dir = scene_dir / 'relations'
    if not relations_dir.exists():
        return []

    files = []
    # tree_L*.json files
    files.extend(relations_dir.glob('tree_L*.json'))
    # tree.json (merged tree)
    tree_json = relations_dir / 'tree.json'
    if tree_json.exists():
        files.append(tree_json)

    return files


def find_legacy_relations_json(scene_dir: Path) -> List[Path]:
    """Find legacy relations.json generated from SSAM trees."""
    relations_json = scene_dir / 'relations.json'
    if not relations_json.exists():
        return []

    # Check if it's a SSAM-based relations.json
    try:
        with open(relations_json) as f:
            data = json.load(f)

        generator = data.get('meta', {}).get('generator', '')
        if 'build_cross_level_relations' in generator:
            return [relations_json]
    except Exception:
        pass

    return []


def find_orphan_temp_dirs(scene_dir: Path) -> List[Path]:
    """Find orphaned temporary directories."""
    return [d for d in scene_dir.glob('tmp*') if d.is_dir()]


def cleanup_scene(scene_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Clean up legacy files in a scene directory.

    Returns:
        (files_removed, dirs_removed)
    """
    files_removed = 0
    dirs_removed = 0

    # Find all legacy files
    ssam_trees = find_ssam_tree_files(scene_dir)
    legacy_relations = find_legacy_relations_json(scene_dir)
    temp_dirs = find_orphan_temp_dirs(scene_dir)

    # Remove SSAM tree files
    for file_path in ssam_trees:
        rel_path = file_path.relative_to(scene_dir)
        if dry_run:
            print(f"  [DRY RUN] Would remove: {rel_path}")
        else:
            print(f"  Removing: {rel_path}")
            file_path.unlink()
        files_removed += 1

    # Remove legacy relations.json
    for file_path in legacy_relations:
        rel_path = file_path.relative_to(scene_dir)
        if dry_run:
            print(f"  [DRY RUN] Would remove: {rel_path}")
        else:
            print(f"  Removing: {rel_path}")
            file_path.unlink()
        files_removed += 1

    # Remove orphan temp directories
    for dir_path in temp_dirs:
        rel_path = dir_path.relative_to(scene_dir)
        # Check if directory is effectively empty (no files)
        has_files = any(dir_path.rglob('*.*'))

        if not has_files:
            if dry_run:
                print(f"  [DRY RUN] Would remove empty temp dir: {rel_path}")
            else:
                print(f"  Removing empty temp dir: {rel_path}")
                shutil.rmtree(dir_path)
            dirs_removed += 1
        else:
            print(f"  ⚠ Skipping temp dir with files: {rel_path}")

    return files_removed, dirs_removed


def main():
    parser = argparse.ArgumentParser(
        description='Clean up legacy output files from My3DIS experiments'
    )
    parser.add_argument(
        'experiment_dir',
        type=Path,
        help='Path to experiment directory (e.g., outputs/experiments/test_1022_4)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing'
    )

    args = parser.parse_args()

    if not args.experiment_dir.exists():
        print(f"Error: Directory not found: {args.experiment_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning experiment directory: {args.experiment_dir}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be removed\n")

    # Find all scene directories
    scene_dirs = [d for d in args.experiment_dir.iterdir()
                  if d.is_dir() and d.name.startswith('scene_')]

    if not scene_dirs:
        print("No scene directories found")
        sys.exit(0)

    total_files = 0
    total_dirs = 0

    for scene_dir in sorted(scene_dirs):
        print(f"\nProcessing: {scene_dir.name}")
        files_removed, dirs_removed = cleanup_scene(scene_dir, dry_run=args.dry_run)

        if files_removed == 0 and dirs_removed == 0:
            print("  ✓ No legacy files found")

        total_files += files_removed
        total_dirs += dirs_removed

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Files removed: {total_files}")
    print(f"  Directories removed: {total_dirs}")

    if args.dry_run:
        print("\nRun without --dry-run to actually remove these files")


if __name__ == '__main__':
    main()
