"""
Standalone CLI for running SigLIP feature assignment stage.

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

from my3dis.siglip_assignment import SigLIPAssignmentPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

logger = logging.getLogger(__name__)


def run_siglip_assignment(
    proposal_dir: str,
    color_frames_dir: str,
    depth_maps_dir: str,
    output_dir: str,
    levels: Optional[List[int]] = None,
    model_name: str = 'google/siglip-so400m-patch14-384',
    object_labels: Optional[List[str]] = None,
    batch_size: int = 32,
    topk_frames: int = 5,
    device: str = 'cuda:0',
) -> Dict[str, str]:
    """
    Run SigLIP feature assignment pipeline.

    Args:
        proposal_dir: Directory containing proposal NPZ files from aggregation
        color_frames_dir: Directory containing color frames
        depth_maps_dir: Directory containing depth maps
        output_dir: Output directory for feature assignments
        levels: Levels to process (default: [2, 4, 6])
        model_name: SigLIP model name from HuggingFace
        object_labels: List of object labels for inference (optional)
        batch_size: Batch size for feature extraction
        topk_frames: Number of best frames per proposal
        device: Torch device

    Returns:
        Dictionary mapping level -> output NPZ path
    """
    logger.info("=" * 80)
    logger.info("SigLIP Feature Assignment Pipeline")
    logger.info("=" * 80)

    # Initialize pipeline
    pipeline = SigLIPAssignmentPipeline(
        proposal_dir=proposal_dir,
        color_frames_dir=color_frames_dir,
        depth_maps_dir=depth_maps_dir,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        topk_frames=topk_frames,
    )

    # Run pipeline
    results = pipeline.run(
        output_dir=output_dir,
        levels=levels,
        object_labels=object_labels,
    )

    logger.info("=" * 80)
    logger.info("SigLIP Assignment Complete")
    logger.info("=" * 80)

    for level, path in results.items():
        logger.info(f"  Level {level}: {path}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Assign SigLIP features to 3D proposals',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        '--proposal-dir',
        required=True,
        help='Directory containing proposal NPZ files from aggregation stage',
    )
    parser.add_argument(
        '--color-frames-dir',
        required=True,
        help='Directory containing color frames',
    )
    parser.add_argument(
        '--depth-maps-dir',
        required=True,
        help='Directory containing depth maps',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for feature assignments',
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
        '--model',
        default='google/siglip-so400m-patch14-384',
        help='SigLIP model name from HuggingFace',
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Object labels for inference (space-separated)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for feature extraction',
    )
    parser.add_argument(
        '--topk-frames',
        type=int,
        default=5,
        help='Number of best frames per proposal',
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Torch device',
    )

    args = parser.parse_args()

    try:
        results = run_siglip_assignment(
            proposal_dir=args.proposal_dir,
            color_frames_dir=args.color_frames_dir,
            depth_maps_dir=args.depth_maps_dir,
            output_dir=args.output_dir,
            levels=args.levels,
            model_name=args.model,
            object_labels=args.labels,
            batch_size=args.batch_size,
            topk_frames=args.topk_frames,
            device=args.device,
        )

        print("\nSigLIP assignment complete. Output files:")
        for level, path in results.items():
            print(f"  Level {level}: {path}")

        return 0

    except Exception as e:
        logger.exception(f"SigLIP assignment failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
