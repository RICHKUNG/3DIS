#!/usr/bin/env python3
"""
Compare provenance-based and containment-based tree building results.

This script analyzes the differences between two tree building approaches:
1. Provenance-based: Uses SSAM parent IDs tracked through SAM2 deduplication
2. Containment-based: Post-hoc geometric analysis of mask containment

Usage:
    python scripts/compare_trees.py --run-dir outputs/experiments/scene_00065_00/run_name
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_provenance_tree(tree_path: str) -> dict:
    """Analyze provenance-based tree structure."""
    tree = load_json(tree_path)
    parent_map = tree.get('parent_map', {})
    orphans = tree.get('orphan_ids', [])
    stats = tree.get('statistics', {})

    # Count families (objects with parents)
    families = sum(1 for parent_id in parent_map.values() if parent_id is not None)
    total_objects = len(parent_map)
    roots = sum(1 for parent_id in parent_map.values() if parent_id is None)

    return {
        'total_objects': total_objects,
        'families': families,
        'orphans': len(orphans),
        'roots': roots,
        'orphan_rate': len(orphans) / total_objects if total_objects > 0 else 0,
        'family_rate': families / total_objects if total_objects > 0 else 0,
        'ssam_stats': stats,
    }


def analyze_containment_tree(tree_path: str) -> dict:
    """Analyze containment-based tree structure (sam2_tree.json format)."""
    tree = load_json(tree_path)
    objects = tree.get('objects', {})

    total_objects = len(objects)
    families = 0
    orphans = 0
    roots = 0

    for obj_id, obj_data in objects.items():
        parent_id = obj_data.get('parent')
        level = obj_data.get('level')

        if parent_id is not None:
            families += 1
        elif level and level != 2:  # Not a root level (L2)
            orphans += 1
        else:
            roots += 1

    return {
        'total_objects': total_objects,
        'families': families,
        'orphans': orphans,
        'roots': roots,
        'orphan_rate': orphans / total_objects if total_objects > 0 else 0,
        'family_rate': families / total_objects if total_objects > 0 else 0,
    }


def compare_trees(run_dir: str, level: Optional[int] = None) -> None:
    """Compare provenance and containment trees for a single level or all levels."""
    relations_dir = os.path.join(run_dir, 'relations')

    if level is not None:
        levels = [level]
    else:
        # Find all provenance tree files
        levels = []
        for fname in os.listdir(relations_dir):
            if fname.startswith('provenance_tree_L') and fname.endswith('.json'):
                level_str = fname.replace('provenance_tree_L', '').replace('.json', '')
                try:
                    levels.append(int(level_str))
                except ValueError:
                    continue
        levels.sort()

    if not levels:
        print(f"❌ No provenance tree files found in {relations_dir}")
        return

    print("=" * 80)
    print("TREE COMPARISON REPORT")
    print("=" * 80)
    print(f"Run Directory: {run_dir}")
    print(f"Levels: {levels}")
    print()

    total_improvement = []

    for lvl in levels:
        print(f"\n{'─' * 80}")
        print(f"LEVEL {lvl}")
        print(f"{'─' * 80}")

        provenance_path = os.path.join(relations_dir, f'provenance_tree_L{lvl}.json')
        containment_path = os.path.join(relations_dir, 'sam2_tree.json')

        if not os.path.exists(provenance_path):
            print(f"⚠️  Provenance tree not found: {provenance_path}")
            continue

        provenance_stats = analyze_provenance_tree(provenance_path)

        # Containment tree might not exist yet
        if os.path.exists(containment_path):
            containment_stats = analyze_containment_tree(containment_path)
            has_containment = True
        else:
            containment_stats = None
            has_containment = False

        # Print provenance stats
        print(f"\n📊 Provenance-Based Tree (SSAM → SAM2 mapping):")
        print(f"   Total Objects:  {provenance_stats['total_objects']}")
        print(f"   Families:       {provenance_stats['families']} ({provenance_stats['family_rate']:.1%})")
        print(f"   Orphans:        {provenance_stats['orphans']} ({provenance_stats['orphan_rate']:.1%})")
        print(f"   Root Objects:   {provenance_stats['roots']}")

        if 'ssam_stats' in provenance_stats and provenance_stats['ssam_stats']:
            stats = provenance_stats['ssam_stats']
            print(f"\n   SSAM Mapping Statistics:")
            print(f"      Total SSAM masks:    {stats.get('total_ssam_masks', 0)}")
            print(f"      Accepted prompts:    {stats.get('accepted_prompts', 0)}")
            print(f"      Rejected prompts:    {stats.get('rejected_prompts', 0)}")
            print(f"      Rejection rate:      {stats.get('rejection_rate', 0):.1%}")

        if has_containment:
            print(f"\n📊 Containment-Based Tree (Post-hoc geometric analysis):")
            print(f"   Total Objects:  {containment_stats['total_objects']}")
            print(f"   Families:       {containment_stats['families']} ({containment_stats['family_rate']:.1%})")
            print(f"   Orphans:        {containment_stats['orphans']} ({containment_stats['orphan_rate']:.1%})")
            print(f"   Root Objects:   {containment_stats['roots']}")

            # Calculate improvement
            orphan_reduction = containment_stats['orphans'] - provenance_stats['orphans']
            orphan_reduction_pct = (orphan_reduction / containment_stats['orphans'] * 100
                                    if containment_stats['orphans'] > 0 else 0)

            family_increase = provenance_stats['families'] - containment_stats['families']
            family_increase_pct = (family_increase / containment_stats['families'] * 100
                                   if containment_stats['families'] > 0 else 0)

            print(f"\n✨ Improvement:")
            print(f"   Orphan Reduction:    {orphan_reduction:+d} ({orphan_reduction_pct:+.1f}%)")
            print(f"   Family Increase:     {family_increase:+d} ({family_increase_pct:+.1f}%)")

            if orphan_reduction > 0:
                print(f"   ✅ Provenance tree reduces orphans by {orphan_reduction}")
                total_improvement.append(orphan_reduction_pct)
            elif orphan_reduction < 0:
                print(f"   ⚠️  Provenance tree has MORE orphans ({-orphan_reduction})")
            else:
                print(f"   ➖ No change in orphan count")
        else:
            print(f"\n⚠️  Containment tree not found: {containment_path}")
            print(f"   (Run sam2_tree_builder.py to generate containment-based tree for comparison)")

    # Overall summary
    if total_improvement:
        avg_improvement = sum(total_improvement) / len(total_improvement)
        print(f"\n{'=' * 80}")
        print(f"OVERALL SUMMARY")
        print(f"{'=' * 80}")
        print(f"Average Orphan Reduction: {avg_improvement:.1f}%")
        print(f"Levels Analyzed: {len(total_improvement)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare provenance-based and containment-based tree building results"
    )
    parser.add_argument(
        '--run-dir',
        required=True,
        help="Path to run directory (e.g., outputs/experiments/scene_00065_00/run_name)"
    )
    parser.add_argument(
        '--level',
        type=int,
        help="Specific level to analyze (default: all levels)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"❌ Run directory not found: {args.run_dir}")
        sys.exit(1)

    compare_trees(args.run_dir, args.level)


if __name__ == '__main__':
    main()
