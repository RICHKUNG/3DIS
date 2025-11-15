#!/usr/bin/env python3
"""
Check mesh file availability for MultiScan scenes.

Helps diagnose mesh file issues by showing what PLY files are available
and which ones would be selected by the fallback mechanism.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def find_mesh_files(scene_path: Path) -> List[Tuple[str, Path, int]]:
    """
    Find all PLY files in a scene directory.

    Args:
        scene_path: Path to scene directory

    Returns:
        List of (priority, path, size_mb) tuples sorted by priority
    """
    if not scene_path.exists():
        return []

    results = []
    scene_name = scene_path.name

    # Priority order (lower number = higher priority)
    priority_patterns = {
        'mesh_aligned_0.05.ply': 1,
        f'{scene_name}_converted.ply': 2,
        f'{scene_name}.ply': 3,
    }

    for ply_file in scene_path.glob('*.ply'):
        priority = priority_patterns.get(ply_file.name, 4)
        size_mb = ply_file.stat().st_size / (1024 * 1024)
        results.append((priority, ply_file, size_mb))

    # Sort by priority
    results.sort(key=lambda x: x[0])
    return results


def check_scene(scene_path: Path, verbose: bool = False) -> dict:
    """Check mesh file availability for a scene."""
    mesh_files = find_mesh_files(scene_path)

    result = {
        'scene': scene_path.name,
        'path': str(scene_path),
        'exists': scene_path.exists(),
        'mesh_count': len(mesh_files),
        'mesh_files': mesh_files,
        'selected': mesh_files[0] if mesh_files else None,
    }

    return result


def print_scene_report(result: dict) -> None:
    """Print a formatted report for a scene."""
    scene = result['scene']
    print(f"\n{'=' * 70}")
    print(f"Scene: {scene}")
    print(f"{'=' * 70}")

    if not result['exists']:
        print("❌ Scene directory does not exist")
        print(f"   Path: {result['path']}")
        return

    if result['mesh_count'] == 0:
        print("⚠️  No PLY mesh files found")
        print(f"   Path: {result['path']}")
        return

    print(f"✓ Found {result['mesh_count']} PLY file(s)")
    print()

    # Show selected file
    if result['selected']:
        priority, path, size_mb = result['selected']
        print("Selected mesh (highest priority):")
        print(f"  ✓ {path.name}")
        print(f"    Size: {size_mb:.2f} MB")
        print(f"    Path: {path}")
        print()

    # Show all files
    if result['mesh_count'] > 1:
        print("All available meshes (in fallback priority order):")
        for i, (priority, path, size_mb) in enumerate(result['mesh_files'], 1):
            marker = "→" if i == 1 else " "
            print(f"  {marker} {i}. {path.name}")
            print(f"       Size: {size_mb:.2f} MB")
            print(f"       Priority: {priority}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Check mesh file availability for MultiScan scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single scene
  python check_scene_meshes.py /path/to/multiscan/scene_00005_00

  # Check multiple scenes
  python check_scene_meshes.py /path/to/multiscan/scene_*

  # Check specific scenes with wildcard
  python check_scene_meshes.py /path/to/multiscan/scene_0000[5-9]_*

  # Show summary for dataset
  python check_scene_meshes.py --dataset /path/to/multiscan --summary
        """
    )

    parser.add_argument(
        'scenes',
        nargs='*',
        help='Scene directories to check'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        help='MultiScan dataset root (checks all scenes)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary statistics only'
    )

    args = parser.parse_args()

    # Collect scenes to check
    scene_paths = []

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            return 1
        scene_paths = sorted([p for p in dataset_path.iterdir() if p.is_dir()])
        print(f"Checking {len(scene_paths)} scenes in {dataset_path}")

    if args.scenes:
        for scene_arg in args.scenes:
            path = Path(scene_arg)
            if path.exists():
                scene_paths.append(path)
            else:
                print(f"Warning: Scene path does not exist: {path}")

    if not scene_paths:
        parser.print_help()
        return 1

    # Check each scene
    results = []
    for scene_path in scene_paths:
        result = check_scene(scene_path)
        results.append(result)

        if not args.summary:
            print_scene_report(result)

    # Summary statistics
    if args.summary or len(results) > 5:
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")

        total = len(results)
        with_meshes = sum(1 for r in results if r['mesh_count'] > 0)
        without_meshes = total - with_meshes
        no_default = sum(
            1 for r in results
            if r['mesh_count'] > 0 and r['selected'][1].name != 'mesh_aligned_0.05.ply'
        )

        print(f"Total scenes checked: {total}")
        print(f"  ✓ With mesh files: {with_meshes} ({with_meshes/total*100:.1f}%)")
        print(f"  ⚠️  Without mesh files: {without_meshes}")
        if no_default > 0:
            print(f"  ℹ️  Using fallback mesh: {no_default} ({no_default/total*100:.1f}%)")

        if without_meshes > 0:
            print(f"\nScenes missing mesh files:")
            for r in results:
                if r['mesh_count'] == 0:
                    print(f"  - {r['scene']}")

        if no_default > 0:
            print(f"\nScenes using fallback mesh (not mesh_aligned_0.05.ply):")
            for r in results:
                if r['mesh_count'] > 0 and r['selected'][1].name != 'mesh_aligned_0.05.ply':
                    print(f"  - {r['scene']}: {r['selected'][1].name}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
