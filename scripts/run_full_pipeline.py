#!/usr/bin/env python3
"""
Full pipeline runner for My3DIS experiments (Aggregation + Inference).

This script orchestrates the complete workflow from My3DIS tracking outputs
to open-vocabulary instance segmentation predictions:

1. Aggregation: Load multi-level tracking outputs and aggregate to 3D proposals
2. Feature Extraction: Extract SigLIP features for each proposal
3. Inference: Run OV-3DIS inference with multi-instance handling
4. (Optional) Evaluation: Compute mAP using Search3D protocol

Usage:
    python scripts/run_full_pipeline.py --config configs/inference/full_pipeline.yaml

    # Override specific parameters
    python scripts/run_full_pipeline.py \
        --config configs/inference/full_pipeline.yaml \
        --exp-dir /path/to/experiment \
        --skip-aggregation  # Skip if aggregation already done

Features:
- Unified YAML configuration for all pipeline stages
- Automatic path management (output dirs default to exp_dir subdirectories)
- Stage toggles (enable/disable aggregation, inference, evaluation)
- Detailed logging and progress tracking
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "3DprojToSiglip"))

import argparse
import json
import logging
import time
import yaml
from typing import Dict, Optional, List

import torch

# My3DIS imports
from my3dis.aggregation import AggregationPipeline
from my3dis.inference import (
    InferencePipeline,
    SigLIPFeatureExtractor,
)
from my3dis.inference.config import InferenceConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_paths(config: Dict) -> Dict:
    """
    Setup paths for aggregation and inference outputs.

    Returns updated config with resolved paths.
    """
    exp_dir = Path(config['experiment']['exp_dir'])

    # Resolve aggregation output directory
    if config['experiment']['aggregation_output_dir'] is None:
        config['experiment']['aggregation_output_dir'] = str(exp_dir / 'aggregation_output')

    # Resolve inference output directory
    if config['experiment']['inference_output_dir'] is None:
        config['experiment']['inference_output_dir'] = str(exp_dir / 'inference_output')

    return config


def _dict_to_inference_config(inf_dict: Dict) -> InferenceConfig:
    """
    Convert inference config dictionary to InferenceConfig object.

    This is a workaround since InferenceConfig.from_yaml expects a file path,
    but we're working with an already-loaded dict.
    """
    from my3dis.inference.config import (
        RetrievalConfig,
        PairingConfig,
        NMSConfig,
        ScoringConfig,
        FormatterConfig,
        IndependentStrategyConfig,
        HierarchicalStrategyConfig,
    )

    config = InferenceConfig()

    if 'strategy' in inf_dict:
        config.strategy = inf_dict['strategy']

    if 'retrieval' in inf_dict:
        config.retrieval = RetrievalConfig(**inf_dict['retrieval'])

    if 'pairing' in inf_dict:
        config.pairing = PairingConfig(**inf_dict['pairing'])

    if 'nms' in inf_dict:
        config.nms = NMSConfig(**inf_dict['nms'])

    if 'scoring' in inf_dict:
        config.scoring = ScoringConfig(**inf_dict['scoring'])

    if 'formatter' in inf_dict:
        config.formatter = FormatterConfig(**inf_dict['formatter'])

    if 'independent' in inf_dict:
        config.independent = IndependentStrategyConfig(**inf_dict['independent'])

    if 'hierarchical' in inf_dict:
        config.hierarchical = HierarchicalStrategyConfig(**inf_dict['hierarchical'])

    return config


def run_aggregation_stage(config: Dict) -> Dict[str, str]:
    """
    Run aggregation stage: 2D masks → 3D proposals + features.

    Returns:
        Dictionary mapping level -> NPZ file path
    """
    logger.info("="*80)
    logger.info("STAGE 1: AGGREGATION (2D → 3D)")
    logger.info("="*80)

    exp_config = config['experiment']
    agg_config = config['aggregation']

    # Initialize aggregation pipeline
    pipeline = AggregationPipeline(
        exp_dir=exp_config['exp_dir'],
        scene_path=exp_config['scene_path'],
        iou_threshold=agg_config['iou_threshold'],
        area_fraction=agg_config['area_fraction'],
        device=exp_config['device'],
        config={
            "uni3dis": {
                "vis_depth_threshold": agg_config['projection']['vis_depth_threshold'],
                "frequency": agg_config['projection']['frequency']
            },
            "network2d": {
                "masklet": {
                    "resize_ratio": agg_config['projection']['resize_ratio']
                }
            }
        },
        pooling_mode=agg_config['pooling_mode'],
    )

    # Load SigLIP model if feature extraction is enabled
    siglip_model = None
    siglip_processor = None

    if agg_config['extract_features']:
        logger.info(f"Loading SigLIP model: {agg_config['siglip_model']}")
        from transformers import AutoModel, AutoProcessor

        siglip_processor = AutoProcessor.from_pretrained(
            agg_config['siglip_model'], use_fast=False
        )
        siglip_model = AutoModel.from_pretrained(
            agg_config['siglip_model']
        ).to(exp_config['device']).eval()

    # Run aggregation
    start_time = time.time()
    results = pipeline.run(
        output_dir=exp_config['aggregation_output_dir'],
        levels=exp_config['levels'],
        extract_features=agg_config['extract_features'],
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
    )
    elapsed = time.time() - start_time

    logger.info(f"✓ Aggregation completed in {elapsed:.1f}s")
    logger.info(f"  Output: {exp_config['aggregation_output_dir']}")

    return results


def run_inference_stage(config: Dict, proposal_data_paths: Dict[str, str]) -> str:
    """
    Run inference stage: 3D proposals + features → OV-3DIS predictions.

    Args:
        config: Full configuration dictionary
        proposal_data_paths: Dictionary mapping level -> NPZ file path

    Returns:
        Path to predictions JSON file
    """
    logger.info("="*80)
    logger.info("STAGE 2: INFERENCE (OV-3DIS)")
    logger.info("="*80)

    exp_config = config['experiment']
    inf_config_dict = config['inference']

    # Convert inference config dict to InferenceConfig object (from dict, not file)
    inference_config = _dict_to_inference_config(inf_config_dict)

    # Load proposals and features
    logger.info("Loading proposals and features from aggregation output...")
    proposals_by_level = {}

    for level_str, npz_path in proposal_data_paths.items():
        level = int(level_str)
        level_key = f"L{level}"

        import numpy as np
        data = np.load(npz_path, allow_pickle=False)

        proposal_masks = torch.from_numpy(data['proposal_masks'])
        proposal_features = torch.from_numpy(data['proposal_features'])
        proposal_ids = data['proposal_ids']

        logger.info(f"  Level {level}: {len(proposal_ids)} proposals")

        # Import Proposal class
        from my3dis.inference.retrieval import Proposal

        # Create Proposal objects
        proposals = []
        for i, prop_id in enumerate(proposal_ids):
            proposals.append(Proposal(
                proposal_id=int(prop_id),
                level=level_key,
                mask=proposal_masks[:, i].numpy(),
                feature=proposal_features[i].numpy(),
                object_id=int(prop_id),  # Same as proposal_id for tracking outputs
            ))

        proposals_by_level[level_key] = proposals

    # Load family tree for relations
    logger.info("Loading family tree for relations...")
    family_tree_path = Path(exp_config['exp_dir']) / 'relations' / 'family_tree.json'

    if not family_tree_path.exists():
        logger.warning(f"Family tree not found: {family_tree_path}")
        family_tree = None
    else:
        with open(family_tree_path, 'r') as f:
            family_tree = json.load(f)

    # Initialize inference pipeline
    logger.info(f"Initializing inference pipeline (strategy: {inference_config.strategy})")
    pipeline = InferencePipeline(
        config=inference_config,
        family_tree=family_tree,
        device=exp_config['device']
    )

    # Load SigLIP model for text encoding
    logger.info(f"Loading SigLIP model for text encoding...")
    feature_extractor = SigLIPFeatureExtractor(
        config['aggregation']['siglip_model'],
        exp_config['device']
    )

    # Run inference on example queries (for demonstration)
    # In practice, you would load queries from a file
    queries = [
        ("chair leg", "part", 1),
        ("table top", "part", 1),
        ("sofa cushion", "part", 1),
    ]

    logger.info(f"Running inference on {len(queries)} queries...")
    output_dir = Path(exp_config['inference_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    start_time = time.time()

    for text_query, query_type, num_instances in queries:
        logger.info(f"  Query: '{text_query}' (type={query_type}, instances={num_instances})")

        # Encode text query
        text_features = feature_extractor.encode_text([text_query])
        text_feature = text_features[0]

        # Run inference
        results = pipeline.predict(
            text_query=text_feature,
            proposals_by_level=proposals_by_level,
            num_instances=num_instances,
        )

        # Store predictions
        for result in results:
            predictions.append({
                'query': text_query,
                'query_type': query_type,
                'num_instances': num_instances,
                'score': float(result['score']),
                'object_level': result['object_level'],
                'part_level': result['part_level'],
                'object_id': result['object_id'],
                'part_id': result['part_id'],
                'mask_points': int(result['mask'].sum()),
            })

        logger.info(f"    Found {len(results)} predictions")

    elapsed = time.time() - start_time

    # Save predictions
    predictions_path = output_dir / 'predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"✓ Inference completed in {elapsed:.1f}s")
    logger.info(f"  Output: {predictions_path}")
    logger.info(f"  Total predictions: {len(predictions)}")

    return str(predictions_path)


def run_evaluation_stage(config: Dict, predictions_path: str):
    """
    Run evaluation stage: Compute mAP using Search3D protocol.

    Args:
        config: Full configuration dictionary
        predictions_path: Path to predictions JSON file
    """
    logger.info("="*80)
    logger.info("STAGE 3: EVALUATION (Search3D)")
    logger.info("="*80)

    eval_config = config['evaluation']

    if eval_config['gt_path'] is None:
        logger.warning("Ground truth path not specified, skipping evaluation")
        return

    # TODO: Implement evaluation using Search3D protocol
    logger.info("Evaluation stage not yet implemented")
    logger.info(f"  GT path: {eval_config['gt_path']}")
    logger.info(f"  Predictions: {predictions_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run full My3DIS pipeline (Aggregation + Inference + Evaluation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config file
  python scripts/run_full_pipeline.py --config configs/inference/full_pipeline.yaml

  # Override experiment directory
  python scripts/run_full_pipeline.py \\
      --config configs/inference/full_pipeline.yaml \\
      --exp-dir /path/to/experiment

  # Skip aggregation if already done
  python scripts/run_full_pipeline.py \\
      --config configs/inference/full_pipeline.yaml \\
      --skip-aggregation
        """
    )

    parser.add_argument(
        '--config', required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--exp-dir',
        help='Override experiment directory path from config'
    )
    parser.add_argument(
        '--scene-path',
        help='Override scene path from config'
    )
    parser.add_argument(
        '--skip-aggregation', action='store_true',
        help='Skip aggregation stage (use existing aggregation output)'
    )
    parser.add_argument(
        '--skip-inference', action='store_true',
        help='Skip inference stage'
    )
    parser.add_argument(
        '--skip-evaluation', action='store_true',
        help='Skip evaluation stage'
    )
    parser.add_argument(
        '--gt-path',
        help='Override ground truth path from config (for evaluation)'
    )
    parser.add_argument(
        '--pooling-mode',
        choices=['average', 'voting'],
        help='Override feature pooling mode from config'
    )
    parser.add_argument(
        '--device',
        help='Override device (cuda, cuda:0, cuda:1, cpu)'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Apply command-line overrides
    if args.exp_dir:
        config['experiment']['exp_dir'] = args.exp_dir
    if args.scene_path:
        config['experiment']['scene_path'] = args.scene_path
    if args.skip_aggregation:
        config['aggregation']['enabled'] = False
    if args.skip_inference:
        config['inference']['enabled'] = False
    if args.skip_evaluation:
        config['evaluation']['enabled'] = False
    if args.gt_path:
        config['evaluation']['gt_path'] = args.gt_path
    if args.pooling_mode:
        config['aggregation']['pooling_mode'] = args.pooling_mode
    if args.device:
        config['experiment']['device'] = args.device

    # Setup paths
    config = setup_paths(config)

    # Print configuration summary
    logger.info("="*80)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Experiment dir:   {config['experiment']['exp_dir']}")
    logger.info(f"Scene path:       {config['experiment']['scene_path']}")
    logger.info(f"Aggregation out:  {config['experiment']['aggregation_output_dir']}")
    logger.info(f"Inference out:    {config['experiment']['inference_output_dir']}")
    logger.info(f"Levels:           {config['experiment']['levels']}")
    logger.info(f"Device:           {config['experiment']['device']}")
    logger.info("")
    logger.info(f"Aggregation settings:")
    logger.info(f"  IoU threshold:   {config['aggregation']['iou_threshold']}")
    logger.info(f"  Area fraction:   {config['aggregation']['area_fraction']}")
    logger.info(f"  Pooling mode:    {config['aggregation']['pooling_mode']}")
    logger.info(f"  Extract features: {config['aggregation']['extract_features']}")
    logger.info(f"  SigLIP model:    {config['aggregation']['siglip_model']}")
    logger.info("")
    logger.info(f"Inference settings:")
    logger.info(f"  Strategy:        {config['inference']['strategy']}")
    logger.info(f"  NMS IoU:         {config['inference']['nms']['iou_threshold']}")
    logger.info(f"  Keep top-k:      {config['inference']['nms']['keep_top_k']}")
    logger.info("")
    logger.info(f"Evaluation settings:")
    logger.info(f"  GT path:         {config['evaluation']['gt_path']}")
    logger.info(f"  Query file:      {config['evaluation']['query_file']}")
    logger.info("")
    logger.info(f"Stages enabled:")
    logger.info(f"  Aggregation:    {config['aggregation']['enabled']}")
    logger.info(f"  Inference:      {config['inference']['enabled']}")
    logger.info(f"  Evaluation:     {config['evaluation']['enabled']}")
    logger.info("="*80)
    logger.info("")

    # Run pipeline stages
    proposal_data_paths = None
    predictions_path = None

    try:
        # Stage 1: Aggregation
        if config['aggregation']['enabled']:
            proposal_data_paths = run_aggregation_stage(config)
        else:
            logger.info("Skipping aggregation stage (loading existing outputs)")
            # Load existing aggregation outputs
            agg_output_dir = Path(config['experiment']['aggregation_output_dir'])
            proposal_data_paths = {}
            for level in config['experiment']['levels']:
                npz_path = agg_output_dir / f"proposal_data_level{level}.npz"
                if npz_path.exists():
                    proposal_data_paths[str(level)] = str(npz_path)
                else:
                    logger.warning(f"Aggregation output not found: {npz_path}")

        # Stage 2: Inference
        if config['inference']['enabled']:
            if not proposal_data_paths:
                logger.error("No aggregation outputs available for inference")
                return 1
            predictions_path = run_inference_stage(config, proposal_data_paths)

        # Stage 3: Evaluation
        if config['evaluation']['enabled']:
            if not predictions_path:
                logger.error("No predictions available for evaluation")
                return 1
            run_evaluation_stage(config, predictions_path)

        logger.info("")
        logger.info("="*80)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
