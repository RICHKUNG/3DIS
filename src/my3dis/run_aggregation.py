"""
Standalone CLI for running 3D mask aggregation stage.

This module can be invoked directly or through the workflow orchestrator.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure src/ is in Python path for direct execution
if __name__ == '__main__':
    _src_root = Path(__file__).resolve().parent.parent
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))

    # Configure temp directory to use /media partition (avoid root partition space issues)
    import os
    _project_root = Path(__file__).resolve().parents[2]
    _tmp_dir = _project_root / 'tmp'
    _tmp_dir.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(_tmp_dir)
    os.environ['TEMP'] = str(_tmp_dir)
    os.environ['TMP'] = str(_tmp_dir)

from my3dis.aggregation import AggregationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

logger = logging.getLogger(__name__)


def run_aggregation(
    family_tree_path: str,
    pointcloud_path: str,
    depth_maps_dir: str,
    color_frames_dir: str,
    output_dir: str,
    levels: Optional[List[int]] = None,
    camera_params_path: Optional[str] = None,
    iou_threshold: float = 0.6,
    min_volume: float = 0.001,
    vis_depth_threshold: float = 0.05,
    device: str = 'cuda:0',
) -> Dict[str, str]:
    """
    Run 3D mask aggregation pipeline.

    Args:
        family_tree_path: Path to family_tree.json from SAM2 tracking
        pointcloud_path: Path to scene point cloud
        depth_maps_dir: Directory containing depth maps
        color_frames_dir: Directory containing color frames
        output_dir: Output directory for proposals
        levels: Levels to process (default: [2, 4, 6])
        camera_params_path: Path to camera parameters JSON (optional)
        iou_threshold: IoU threshold for deduplication
        min_volume: Minimum volume threshold (m^3)
        vis_depth_threshold: Visibility depth threshold (meters)
        device: Torch device

    Returns:
        Dictionary mapping level -> output NPZ path
    """
    logger.info("=" * 80)
    logger.info("3D Mask Aggregation Pipeline")
    logger.info("=" * 80)

    # Initialize pipeline
    pipeline = AggregationPipeline(
        family_tree_path=family_tree_path,
        pointcloud_path=pointcloud_path,
        depth_maps_dir=depth_maps_dir,
        color_frames_dir=color_frames_dir,
        camera_params_path=camera_params_path,
        iou_threshold=iou_threshold,
        min_volume=min_volume,
        device=device,
    )

    # Run pipeline
    results = pipeline.run(output_dir=output_dir, levels=levels)

    logger.info("=" * 80)
    logger.info("Aggregation Complete")
    logger.info("=" * 80)

    for level, path in results.items():
        logger.info(f"  Level {level}: {path}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Aggregate 2D masks to 3D proposals using family tree',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        '--family-tree',
        required=True,
        help='Path to family_tree.json from SAM2 tracking stage',
    )
    parser.add_argument(
        '--pointcloud',
        required=True,
        help='Path to scene point cloud (.ply or .pth)',
    )
    parser.add_argument(
        '--depth-maps-dir',
        required=True,
        help='Directory containing depth maps',
    )
    parser.add_argument(
        '--color-frames-dir',
        required=True,
        help='Directory containing color frames',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for 3D proposals',
    )

    # Optional arguments
    parser.add_argument(
        '--levels',
        nargs='+',
        type=int,
        default=[2, 4, 6],
        help='Levels to process',
    )
    parser.add_argument(
        '--camera-params',
        help='Path to camera parameters JSON (optional)',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.6,
        help='IoU threshold for 3D proposal deduplication',
    )
    parser.add_argument(
        '--min-volume',
        type=float,
        default=0.001,
        help='Minimum volume threshold (m^3)',
    )
    parser.add_argument(
        '--vis-depth-threshold',
        type=float,
        default=0.05,
        help='Visibility depth threshold (meters)',
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Torch device',
    )

    args = parser.parse_args()

    try:
        results = run_aggregation(
            family_tree_path=args.family_tree,
            pointcloud_path=args.pointcloud,
            depth_maps_dir=args.depth_maps_dir,
            color_frames_dir=args.color_frames_dir,
            output_dir=args.output_dir,
            levels=args.levels,
            camera_params_path=args.camera_params,
            iou_threshold=args.iou_threshold,
            min_volume=args.min_volume,
            vis_depth_threshold=args.vis_depth_threshold,
            device=args.device,
        )

        print("\nAggregation complete. Output files:")
        for level, path in results.items():
            print(f"  Level {level}: {path}")

        return 0

    except Exception as e:
        logger.exception(f"Aggregation failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
