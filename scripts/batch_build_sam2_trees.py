#!/usr/bin/env python3
"""
Batch build SAM2-based trees for all existing experiments.

This script scans an experiment root directory and builds SAM2 containment trees
for all scenes that have completed tracking (without re-running tracking).

Usage:
    python scripts/batch_build_sam2_trees.py \
        --experiment-root outputs/experiments/v4_246_ssam2_filter500_fill5k_prop30_iou06_ds03 \
        --levels 2,4,6 \
        --containment-threshold 0.95 \
        --temporal-overlap-threshold 0.3

Author: Rich Kung
Date: 2025-10-22
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for direct execution
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

from my3dis.sam2_tree_builder import build_sam2_containment_tree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
LOGGER = logging.getLogger(__name__)


def check_has_tracking_outputs(scene_dir: Path, levels: List[int]) -> bool:
    """Check if a scene directory has tracking outputs for all levels."""
    for level in levels:
        level_dir = scene_dir / f'level_{level}'
        index_path = level_dir / 'index.json'

        # Check if index.json exists and has objects
        if not index_path.exists():
            return False

        try:
            with index_path.open('r') as f:
                index_data = json.load(f)

            objects = index_data.get('objects', {})
            if not objects:
                LOGGER.debug("Scene %s level %d: index.json has no objects", scene_dir.name, level)
                return False
        except Exception as e:
            LOGGER.debug("Scene %s level %d: failed to read index.json - %s", scene_dir.name, level, e)
            return False

    return True


def check_already_has_sam2_tree(scene_dir: Path) -> bool:
    """Check if SAM2 tree already exists."""
    sam2_tree_path = scene_dir / 'relations' / 'sam2_tree.json'
    return sam2_tree_path.exists()


def find_scene_directories(experiment_root: Path) -> List[Path]:
    """Find all scene directories in experiment root."""
    scene_dirs = []

    # Pattern: scene_XXXXX_YY
    for path in sorted(experiment_root.glob('scene_*_*')):
        if path.is_dir():
            scene_dirs.append(path)

    return scene_dirs


def build_tree_for_scene(
    scene_dir: Path,
    levels: List[int],
    containment_threshold: float,
    temporal_overlap_threshold: float,
    force: bool = False,
) -> bool:
    """
    Build SAM2 tree for a single scene.

    Returns:
        True if successful, False otherwise
    """
    scene_name = scene_dir.name

    # Check if tracking outputs exist
    if not check_has_tracking_outputs(scene_dir, levels):
        LOGGER.warning("Scene %s: missing tracking outputs, skipping", scene_name)
        return False

    # Check if tree already exists
    if check_already_has_sam2_tree(scene_dir) and not force:
        LOGGER.info("Scene %s: SAM2 tree already exists, skipping (use --force to rebuild)", scene_name)
        return True

    # Build tree
    try:
        LOGGER.info("Scene %s: building SAM2 tree...", scene_name)
        tree_path = build_sam2_containment_tree(
            run_dir=scene_dir,
            levels=levels,
            containment_threshold=containment_threshold,
            temporal_overlap_threshold=temporal_overlap_threshold,
            output_filename='sam2_tree.json',
        )

        # Read and log statistics
        with tree_path.open('r') as f:
            tree_data = json.load(f)

        stats = tree_data.get('statistics', {})
        LOGGER.info(
            "Scene %s: ✅ SAM2 tree built - %d objects, %d roots, %d families",
            scene_name,
            stats.get('total_objects', 0),
            stats.get('roots', 0),
            stats.get('families', 0),
        )
        return True

    except Exception as exc:
        LOGGER.error("Scene %s: ❌ Failed to build tree - %s", scene_name, exc, exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch build SAM2 trees for all scenes in an experiment"
    )
    parser.add_argument(
        '--experiment-root',
        required=True,
        type=Path,
        help='Path to experiment root directory (e.g., outputs/experiments/v4_*)',
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
        help='Minimum avg containment for parent-child (default: 0.95)',
    )
    parser.add_argument(
        '--temporal-overlap-threshold',
        type=float,
        default=0.3,
        help='Minimum temporal overlap ratio (default: 0.3)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Rebuild trees even if they already exist',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List scenes without building trees',
    )
    parser.add_argument(
        '--scene-filter',
        type=str,
        help='Only process scenes matching this pattern (e.g., "scene_00065_*")',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable debug logging',
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate experiment root
    if not args.experiment_root.exists():
        LOGGER.error("Experiment root not found: %s", args.experiment_root)
        return 1

    # Parse levels
    levels = [int(x.strip()) for x in args.levels.split(',')]

    # Find scene directories
    scene_dirs = find_scene_directories(args.experiment_root)

    # Apply scene filter if specified
    if args.scene_filter:
        import fnmatch
        scene_dirs = [
            d for d in scene_dirs
            if fnmatch.fnmatch(d.name, args.scene_filter)
        ]

    if not scene_dirs:
        LOGGER.warning("No scene directories found in %s", args.experiment_root)
        return 0

    LOGGER.info("Found %d scene directories", len(scene_dirs))

    # Dry run mode
    if args.dry_run:
        LOGGER.info("DRY RUN MODE - No trees will be built")
        for scene_dir in scene_dirs:
            has_tracking = check_has_tracking_outputs(scene_dir, levels)
            has_tree = check_already_has_sam2_tree(scene_dir)
            status = "✅ ready" if has_tracking else "⏭️  no tracking"
            if has_tree and not args.force:
                status = "⏭️  has tree"
            LOGGER.info("  %s: %s", scene_dir.name, status)
        return 0

    # Build trees
    LOGGER.info(
        "Building SAM2 trees (containment=%.2f, temporal_overlap=%.2f)...",
        args.containment_threshold,
        args.temporal_overlap_threshold,
    )

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, scene_dir in enumerate(scene_dirs, 1):
        LOGGER.info("[%d/%d] Processing %s...", i, len(scene_dirs), scene_dir.name)

        # Check if should skip
        if not check_has_tracking_outputs(scene_dir, levels):
            skip_count += 1
            continue

        if check_already_has_sam2_tree(scene_dir) and not args.force:
            skip_count += 1
            continue

        # Build tree
        success = build_tree_for_scene(
            scene_dir=scene_dir,
            levels=levels,
            containment_threshold=args.containment_threshold,
            temporal_overlap_threshold=args.temporal_overlap_threshold,
            force=args.force,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    LOGGER.info("")
    LOGGER.info("======================================")
    LOGGER.info("Batch tree building completed")
    LOGGER.info("======================================")
    LOGGER.info("Total scenes: %d", len(scene_dirs))
    LOGGER.info("✅ Successfully built: %d", success_count)
    LOGGER.info("⏭️  Skipped: %d", skip_count)
    LOGGER.info("❌ Failed: %d", fail_count)

    if fail_count > 0:
        LOGGER.warning("Some scenes failed - check logs above for details")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
