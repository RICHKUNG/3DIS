#!/usr/bin/env python3
"""
Analyze SAM2 tree structure and diagnose why families are not being created.

Usage:
    python scripts/analyze_sam2_tree.py <scene_dir>

Example:
    python scripts/analyze_sam2_tree.py outputs/experiments/test_1022_4/scene_00005_00
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_sam2_tree(scene_dir: Path) -> Dict:
    """Load SAM2 tree JSON."""
    tree_path = scene_dir / 'relations' / 'sam2_tree.json'
    if not tree_path.exists():
        print(f"Error: sam2_tree.json not found at {tree_path}")
        sys.exit(1)

    with open(tree_path) as f:
        return json.load(f)


def load_level_indices(scene_dir: Path, levels: List[int]) -> Dict:
    """Load index.json for each level."""
    indices = {}
    for level in levels:
        index_path = scene_dir / f'level_{level}' / 'index.json'
        if not index_path.exists():
            print(f"Warning: index.json not found for level {level}")
            continue

        with open(index_path) as f:
            indices[level] = json.load(f)

    return indices


def analyze_tree(tree_data: Dict) -> None:
    """Analyze tree structure and statistics."""
    print("=== SAM2 Tree Analysis ===\n")

    # Basic statistics
    stats = tree_data.get('statistics', {})
    print("Statistics:")
    print(f"  Total objects: {stats.get('total_objects', 0)}")
    print(f"  Root objects: {stats.get('roots', 0)}")
    print(f"  Family objects (with children): {stats.get('families', 0)}")
    print(f"  Max tree depth: {stats.get('max_depth', 0)}")

    # Parameters
    params = tree_data.get('meta', {}).get('parameters', {})
    print(f"\nParameters:")
    print(f"  Containment threshold: {params.get('containment_threshold', 'N/A')}")
    print(f"  Temporal overlap threshold: {params.get('temporal_overlap_threshold', 'N/A')}")

    # Hierarchy analysis
    hierarchy = tree_data.get('hierarchy', {})
    print(f"\nHierarchy by level:")
    for level_str, objects in sorted(hierarchy.items()):
        level = int(level_str)
        obj_count = len(objects)
        with_children = sum(1 for obj in objects.values() if obj.get('children'))
        with_parent = sum(1 for obj in objects.values() if obj.get('parent') is not None)

        print(f"  Level {level}:")
        print(f"    Total objects: {obj_count}")
        print(f"    With children: {with_children}")
        print(f"    With parent: {with_parent}")
        print(f"    Orphans: {obj_count - with_parent}")


def analyze_object_distributions(scene_dir: Path, levels: List[int]) -> None:
    """Analyze object frame distributions across levels."""
    print("\n=== Object Distribution Analysis ===\n")

    indices = load_level_indices(scene_dir, levels)

    for level in sorted(levels):
        if level not in indices:
            continue

        index = indices[level]
        objects = index.get('objects', {})

        if not objects:
            print(f"Level {level}: No objects")
            continue

        frame_counts = [len(obj['frames']) for obj in objects.values()]
        avg_frames = sum(frame_counts) / len(frame_counts)
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)

        print(f"Level {level}:")
        print(f"  Total objects: {len(objects)}")
        print(f"  Avg frames/object: {avg_frames:.1f}")
        print(f"  Min frames: {min_frames}")
        print(f"  Max frames: {max_frames}")

        # Frame count distribution
        bins = [0, 5, 10, 20, 50, float('inf')]
        bin_labels = ['1-4', '5-9', '10-19', '20-49', '50+']
        distribution = {label: 0 for label in bin_labels}

        for count in frame_counts:
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                if low < count <= high:
                    distribution[bin_labels[i]] += 1
                    break

        print(f"  Frame distribution:")
        for label, count in distribution.items():
            percentage = (count / len(frame_counts)) * 100
            print(f"    {label} frames: {count} ({percentage:.1f}%)")


def suggest_improvements(tree_data: Dict) -> None:
    """Suggest improvements based on analysis."""
    print("\n=== Suggestions ===\n")

    stats = tree_data.get('statistics', {})
    params = tree_data.get('meta', {}).get('parameters', {})

    total_objects = stats.get('total_objects', 0)
    families = stats.get('families', 0)
    containment_threshold = params.get('containment_threshold', 0.95)
    temporal_threshold = params.get('temporal_overlap_threshold', 0.3)

    if families == 0:
        print("⚠️  No family relationships detected!")
        print("\nPossible causes:")
        print("1. Containment threshold too high")
        print(f"   Current: {containment_threshold}")
        print(f"   Suggestion: Try 0.70-0.85 range")
        print()
        print("2. Temporal overlap threshold too high")
        print(f"   Current: {temporal_threshold}")
        print(f"   Suggestion: Try 0.1-0.25 range")
        print()
        print("3. Objects in different levels have similar scales")
        print("   Check object distribution analysis above")
        print()
        print("Recommended config change:")
        print("```yaml")
        print("stages:")
        print("  tracker:")
        print("    sam2_tree:")
        print("      containment_threshold: 0.80      # Lowered from {:.2f}".format(containment_threshold))
        print("      temporal_overlap_threshold: 0.2  # Lowered from {:.2f}".format(temporal_threshold))
        print("```")
    elif families < total_objects * 0.1:
        print(f"ℹ️  Few family relationships detected ({families}/{total_objects})")
        print("\nConsider slightly lowering thresholds:")
        print(f"  containment_threshold: {max(0.70, containment_threshold - 0.05):.2f}")
        print(f"  temporal_overlap_threshold: {max(0.1, temporal_threshold - 0.05):.2f}")
    else:
        print(f"✓ Reasonable family structure detected ({families}/{total_objects})")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SAM2 tree structure and diagnose issues'
    )
    parser.add_argument(
        'scene_dir',
        type=Path,
        help='Path to scene directory (e.g., outputs/experiments/test_1022_4/scene_00005_00)'
    )

    args = parser.parse_args()

    if not args.scene_dir.exists():
        print(f"Error: Directory not found: {args.scene_dir}", file=sys.stderr)
        sys.exit(1)

    # Load SAM2 tree
    tree_data = load_sam2_tree(args.scene_dir)
    levels = tree_data.get('meta', {}).get('levels', [2, 4, 6])

    # Analyze tree structure
    analyze_tree(tree_data)

    # Analyze object distributions
    analyze_object_distributions(args.scene_dir, levels)

    # Suggest improvements
    suggest_improvements(tree_data)


if __name__ == '__main__':
    main()
