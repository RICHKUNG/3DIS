#!/usr/bin/env python3
"""
Validation script to check provenance tree completeness against index.json.

This script helps diagnose cases where objects appear in index.json but are
missing from provenance_tree_L{level}.json, indicating incomplete object
registration during SAM2 tracking.

Usage:
    python scripts/validate_provenance_trees.py --run-dir <path> --levels 2 4 6

Author: Rich Kung
Created: 2025-11-05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict if not found."""
    if not file_path.exists():
        return {}
    with file_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_index_objects(run_dir: Path, level: int) -> Set[int]:
    """Extract object IDs from index.json."""
    index_path = run_dir / f'level_{level}' / 'index.json'
    index_data = load_json_file(index_path)

    objects = index_data.get('objects', {})
    return {int(obj_id) for obj_id in objects.keys()}


def get_provenance_objects(run_dir: Path, level: int) -> Set[int]:
    """Extract object IDs from provenance_tree_L{level}.json."""
    provenance_path = run_dir / 'relations' / f'provenance_tree_L{level}.json'
    provenance_data = load_json_file(provenance_path)

    parent_map = provenance_data.get('parent_map', {})
    return {int(obj_id) for obj_id in parent_map.keys()}


def validate_level(run_dir: Path, level: int) -> Dict[str, Any]:
    """
    Validate provenance tree completeness for a single level.

    Returns:
        Dictionary with:
        - index_count: Number of objects in index.json
        - provenance_count: Number of objects in provenance_tree
        - missing_ids: Objects in index but not in provenance
        - extra_ids: Objects in provenance but not in index
        - missing_count: Number of missing objects
        - extra_count: Number of extra objects
        - is_complete: True if provenance matches index exactly
    """
    index_ids = get_index_objects(run_dir, level)
    provenance_ids = get_provenance_objects(run_dir, level)

    missing_ids = sorted(index_ids - provenance_ids)
    extra_ids = sorted(provenance_ids - index_ids)

    return {
        'index_count': len(index_ids),
        'provenance_count': len(provenance_ids),
        'missing_ids': missing_ids,
        'extra_ids': extra_ids,
        'missing_count': len(missing_ids),
        'extra_count': len(extra_ids),
        'is_complete': len(missing_ids) == 0 and len(extra_ids) == 0,
    }


def validate_provenance_completeness(run_dir: Path, levels: List[int]) -> Dict[str, Any]:
    """
    Validate provenance tree completeness across all levels.

    Returns:
        Dictionary with per-level validation results and overall summary.
    """
    results = {}
    total_missing = 0
    total_extra = 0
    all_complete = True

    for level in sorted(levels):
        level_result = validate_level(run_dir, level)
        results[f'level_{level}'] = level_result

        total_missing += level_result['missing_count']
        total_extra += level_result['extra_count']
        all_complete = all_complete and level_result['is_complete']

    results['summary'] = {
        'total_missing': total_missing,
        'total_extra': total_extra,
        'all_complete': all_complete,
    }

    return results


def print_validation_report(results: Dict[str, Any], verbose: bool = False) -> None:
    """Print validation results in human-readable format."""
    print("=" * 80)
    print("PROVENANCE TREE VALIDATION REPORT")
    print("=" * 80)

    for key in sorted(results.keys()):
        if key == 'summary':
            continue

        level_result = results[key]
        level_num = key.split('_')[1]

        print(f"\n{'─' * 80}")
        print(f"Level {level_num}")
        print(f"{'─' * 80}")
        print(f"  Index objects:      {level_result['index_count']:4d}")
        print(f"  Provenance objects: {level_result['provenance_count']:4d}")
        print(f"  Missing in provenance: {level_result['missing_count']:4d}")
        print(f"  Extra in provenance:   {level_result['extra_count']:4d}")

        if level_result['is_complete']:
            print(f"  Status: ✓ COMPLETE")
        else:
            print(f"  Status: ✗ INCOMPLETE")

        if verbose and level_result['missing_ids']:
            print(f"\n  Missing object IDs (first 20):")
            for obj_id in level_result['missing_ids'][:20]:
                print(f"    - {obj_id}")
            if len(level_result['missing_ids']) > 20:
                print(f"    ... and {len(level_result['missing_ids']) - 20} more")

        if verbose and level_result['extra_ids']:
            print(f"\n  Extra object IDs (first 20):")
            for obj_id in level_result['extra_ids'][:20]:
                print(f"    - {obj_id}")
            if len(level_result['extra_ids']) > 20:
                print(f"    ... and {len(level_result['extra_ids']) - 20} more")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    summary = results['summary']
    print(f"  Total missing objects: {summary['total_missing']}")
    print(f"  Total extra objects:   {summary['total_extra']}")

    if summary['all_complete']:
        print(f"\n  Overall status: ✓ ALL LEVELS COMPLETE")
        return_code = 0
    else:
        print(f"\n  Overall status: ✗ INCOMPLETE - requires investigation")
        return_code = 1

    print("=" * 80)

    return return_code


def main():
    parser = argparse.ArgumentParser(
        description='Validate provenance tree completeness against index.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single run
  python scripts/validate_provenance_trees.py \\
    --run-dir outputs/experiments/scene_00005_00/test_1105 \\
    --levels 2 4 6

  # Verbose output with object ID lists
  python scripts/validate_provenance_trees.py \\
    --run-dir outputs/experiments/scene_00005_00/test_1105 \\
    --levels 2 4 6 \\
    --verbose

  # Save results to JSON
  python scripts/validate_provenance_trees.py \\
    --run-dir outputs/experiments/scene_00005_00/test_1105 \\
    --levels 2 4 6 \\
    --output validation_results.json
        """
    )

    parser.add_argument(
        '--run-dir',
        required=True,
        type=Path,
        help='Path to run directory (e.g., outputs/experiments/scene_00005_00/test_1105)'
    )
    parser.add_argument(
        '--levels',
        nargs='+',
        type=int,
        default=[2, 4, 6],
        help='Levels to validate (default: 2 4 6)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save validation results to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed object ID lists'
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating run directory: {args.run_dir}")
    print(f"Levels: {args.levels}\n")

    results = validate_provenance_completeness(args.run_dir, args.levels)

    if args.output:
        with args.output.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    return_code = print_validation_report(results, verbose=args.verbose)

    sys.exit(return_code)


if __name__ == '__main__':
    main()
