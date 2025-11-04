#!/usr/bin/env python3
"""
Example script for running inference on MultiScan scenes.

This demonstrates how to:
1. Load proposals from My3DIS tracking output
2. Load SigLIP features for queries
3. Run inference with different strategies
4. Format output for Search3D evaluation
5. Save predictions

Usage:
    python scripts/run_inference_example.py \
        --run-dir outputs/experiments/scene_00065_00/run_name \
        --config configs/inference/base.yaml \
        --queries queries.json \
        --output predictions.npz
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
import logging
from typing import Dict, List, Tuple

from my3dis.inference import InferencePipeline
from my3dis.inference.config import InferenceConfig
from my3dis.inference.retrieval import Proposal
from my3dis.inference.formatter import Search3DFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_proposals_from_my3dis(
    run_dir: str,
    levels: List[str] = ['L2', 'L4', 'L6']
) -> Tuple[Dict[str, List[Proposal]], int]:
    """
    Load proposals from My3DIS tracking output.

    Args:
        run_dir: Path to experiment run directory
        levels: Levels to load

    Returns:
        (proposals_by_level, num_points)
    """
    run_dir = Path(run_dir)
    proposals_by_level = {}
    num_points = None

    for level in levels:
        level_dir = run_dir / f"level_{level}"
        tracking_dir = level_dir / "tracking"

        if not tracking_dir.exists():
            logger.warning(f"Tracking directory not found: {tracking_dir}")
            continue

        # Load object_segments NPZ
        object_files = list(tracking_dir.glob("object_segments_*.npz"))

        if len(object_files) == 0:
            logger.warning(f"No object_segments files in {tracking_dir}")
            continue

        # Use first file (or you can select by scale)
        npz_file = object_files[0]
        logger.info(f"Loading {level} from {npz_file.name}")

        data = np.load(npz_file)

        # Extract object IDs and masks
        # NOTE: Adjust this based on your actual output format
        # This is a placeholder implementation

        if 'object_ids' in data and 'masks' in data:
            object_ids = data['object_ids']
            masks = data['masks']  # Shape: (N_objects, N_points) or similar

            if num_points is None:
                num_points = masks.shape[-1]

            proposals = []
            for i, obj_id in enumerate(object_ids):
                # Placeholder: you need to load actual SigLIP features
                # For now, use random features
                feature = np.random.rand(768)  # Replace with actual feature loading

                prop = Proposal(
                    level=level,
                    proposal_id=i,
                    mask=masks[i],
                    feature=feature,
                    object_id=int(obj_id)
                )
                proposals.append(prop)

            proposals_by_level[level] = proposals
            logger.info(f"Loaded {len(proposals)} proposals from {level}")

        else:
            logger.error(f"Unexpected NPZ format in {npz_file}")

    return proposals_by_level, num_points


def load_queries(queries_file: str) -> List[Tuple[str, str, int]]:
    """
    Load queries from JSON file.

    Expected format:
    [
        {
            "object": "cabinet",
            "part": "door",
            "label_id": 121003
        },
        ...
    ]

    Args:
        queries_file: Path to queries JSON

    Returns:
        List of (object_text, part_text, label_id) tuples
    """
    with open(queries_file, 'r') as f:
        queries_data = json.load(f)

    queries = [
        (q['object'], q['part'], q['label_id'])
        for q in queries_data
    ]

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def get_query_embedding(text: str) -> np.ndarray:
    """
    Get SigLIP embedding for query text.

    NOTE: This is a placeholder. You need to implement actual SigLIP encoding.

    Args:
        text: Query text

    Returns:
        Embedding vector
    """
    # TODO: Replace with actual SigLIP encoder
    # Example:
    # from siglip import SigLIPEncoder
    # encoder = SigLIPEncoder()
    # embedding = encoder.encode(text)

    # Placeholder: random embedding
    return np.random.rand(768)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a scene")
    parser.add_argument('--run-dir', required=True,
                       help='Path to experiment run directory')
    parser.add_argument('--config', default='configs/inference/base.yaml',
                       help='Path to inference config YAML')
    parser.add_argument('--queries', required=True,
                       help='Path to queries JSON file')
    parser.add_argument('--output', required=True,
                       help='Output path for predictions (.npz)')
    parser.add_argument('--levels', nargs='+', default=['L2', 'L4', 'L6'],
                       help='Levels to load')

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = InferenceConfig.from_yaml(args.config)

    # Load proposals
    logger.info(f"Loading proposals from {args.run_dir}")
    proposals_by_level, num_points = load_proposals_from_my3dis(
        args.run_dir,
        levels=args.levels
    )

    if not proposals_by_level:
        logger.error("No proposals loaded!")
        return

    # Load family tree
    family_tree_path = Path(args.run_dir) / "relations" / "family_tree.json"
    if not family_tree_path.exists():
        logger.warning(f"Family tree not found: {family_tree_path}")
        family_tree_path = None

    # Initialize pipeline
    logger.info(f"Initializing pipeline with strategy: {config.strategy}")
    pipeline = InferencePipeline(
        proposals_by_level=proposals_by_level,
        family_tree_path=str(family_tree_path) if family_tree_path else None,
        strategy=config.strategy,
        num_points=num_points,
        **config.get_strategy_kwargs()
    )

    # Load queries
    logger.info(f"Loading queries from {args.queries}")
    queries = load_queries(args.queries)

    # Run inference for all queries
    all_predictions = []

    for i, (obj_text, part_text, label_id) in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: "
                   f"{obj_text} + {part_text} (label={label_id})")

        # Get embeddings
        obj_embedding = get_query_embedding(obj_text)
        part_embedding = get_query_embedding(part_text)

        # Run inference
        predictions = pipeline.infer_single_query(
            object_query_feat=obj_embedding,
            part_query_feat=part_embedding,
            label_id=label_id,
            nms_iou_threshold=config.nms.iou_threshold,
            use_soft_nms=config.nms.use_soft_nms,
            keep_top_k=config.nms.keep_top_k
        )

        logger.info(f"  -> {len(predictions)} predictions")
        all_predictions.extend(predictions)

    logger.info(f"\nTotal predictions: {len(all_predictions)}")

    # Format for Search3D
    logger.info("Formatting predictions for Search3D evaluation")
    formatted = pipeline.format_predictions(all_predictions)

    # Save
    logger.info(f"Saving predictions to {args.output}")
    Search3DFormatter.save_predictions(
        {'scene': formatted},  # Use actual scene name
        args.output
    )

    logger.info("Done!")


if __name__ == '__main__':
    main()
