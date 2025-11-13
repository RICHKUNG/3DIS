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
from collections import Counter

import torch

# My3DIS imports
from my3dis.aggregation import AggregationPipeline
from my3dis.inference import (
    InferencePipeline,
    SigLIPFeatureExtractor,
)
from my3dis.inference.config import InferenceConfig
from my3dis.utils import load_scene_pointcloud, save_mask_as_ply

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


def load_scene_label_distribution(gt_root: Optional[str], scene_name: str) -> Optional[Counter]:
    """
    Load per-scene label distribution from Search3D GT annotations.

    The GT files store per-point joint semantic ids encoded as
    label_id * 1000 + instance_id. We only care about the label ids.
    """
    if not gt_root:
        return None

    gt_file = Path(gt_root) / f"{scene_name}_obj_part_inst.txt"
    if not gt_file.exists():
        logger.warning(f"GT file not found for scene {scene_name}: {gt_file}")
        return None

    label_counts: Counter = Counter()
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = int(line)
            except ValueError:
                continue
            if value <= 0:
                continue
            label_id = value // 1000
            label_counts[label_id] += 1

    if not label_counts:
        logger.warning(f"No positive labels found in GT for scene {scene_name} ({gt_file})")
        return None

    return label_counts

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

    # Parse per-level area fractions if provided
    area_fraction_per_level = agg_config.get('area_fraction_per_level', None)
    if area_fraction_per_level:
        # Convert string keys to int keys
        area_fraction_per_level = {int(k): v for k, v in area_fraction_per_level.items()}

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
        area_fraction_per_level=area_fraction_per_level,
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

    # Auto-detect or use specified levels
    levels_to_use = exp_config.get('levels', None)
    if levels_to_use is None:
        from my3dis.aggregation.aggregation_pipeline import detect_available_levels
        exp_dir = Path(exp_config['exp_dir'])
        detected_levels = detect_available_levels(exp_dir)
        if detected_levels:
            levels_to_use = detected_levels
            logger.info(f"Auto-detected levels from experiment directory: {levels_to_use}")
        else:
            logger.warning("No levels detected and none specified, pipeline will use defaults")

    # Run aggregation
    start_time = time.time()
    results = pipeline.run(
        output_dir=exp_config['aggregation_output_dir'],
        levels=levels_to_use,
        extract_features=agg_config['extract_features'],
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
    )
    elapsed = time.time() - start_time

    logger.info(f"✓ Aggregation completed in {elapsed:.1f}s")
    logger.info(f"  Output: {exp_config['aggregation_output_dir']}")

    return results


def run_inference_stage(
    config: Dict,
    proposal_data_paths: Dict[str, str],
    strategy: Optional[str] = None,
    output_subdir: Optional[str] = None
) -> str:
    """
    Run inference stage: 3D proposals + features → OV-3DIS predictions.

    Args:
        config: Full configuration dictionary
        proposal_data_paths: Dictionary mapping level -> NPZ file path
        strategy: Strategy to use (overrides config if specified)
        output_subdir: Output subdirectory name (for multi-strategy testing)

    Returns:
        Path to predictions JSON file
    """
    logger.info("="*80)
    logger.info(f"STAGE 2: INFERENCE (OV-3DIS{f' - {strategy}' if strategy else ''})")
    logger.info("="*80)

    exp_config = config['experiment']
    scene_name = Path(exp_config['exp_dir']).name
    inf_config_dict = config['inference'].copy()

    # Override strategy if specified
    if strategy is not None:
        inf_config_dict['strategy'] = strategy
        logger.info(f"Using strategy: {strategy}")

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

    # Check for family tree
    logger.info("Checking for family tree...")
    family_tree_path = Path(exp_config['exp_dir']) / 'relations' / 'family_tree.json'

    if not family_tree_path.exists():
        logger.warning(f"Family tree not found: {family_tree_path}")
        family_tree_path_str = None
    else:
        family_tree_path_str = str(family_tree_path)
        logger.info(f"Found family tree: {family_tree_path_str}")

    # Initialize inference pipeline
    logger.info(f"Initializing inference pipeline (strategy: {inference_config.strategy})")

    # Extract strategy-specific kwargs
    strategy_kwargs = {}
    if inference_config.strategy == 'independent':
        strategy_kwargs = {
            'object_levels': inference_config.independent.object_levels,
            'part_levels': inference_config.independent.part_levels,
            'top_k_per_level': inference_config.independent.top_k_per_level,
            'retrieval_threshold': inference_config.independent.retrieval_threshold,
            'min_retrieval_threshold': inference_config.independent.min_retrieval_threshold,
            'threshold_backoff_step': inference_config.independent.threshold_backoff_step,
            'fallback_to_topk': inference_config.independent.fallback_to_topk,
        }
    elif inference_config.strategy == 'hierarchical':
        strategy_kwargs = {
            'coarse_top_k': inference_config.hierarchical.coarse_top_k,
            'object_top_k': inference_config.hierarchical.object_top_k,
            'part_top_k': inference_config.hierarchical.part_top_k,
            'coarse_threshold': inference_config.hierarchical.coarse_threshold,
            'refinement_threshold': inference_config.hierarchical.refinement_threshold,
            'refinement_margin': inference_config.hierarchical.refinement_margin,
        }

    pipeline = InferencePipeline(
        proposals_by_level=proposals_by_level,
        family_tree_path=family_tree_path_str,  # Pass path instead of dict
        strategy=inference_config.strategy,
        **strategy_kwargs
    )

    # Load SigLIP model for text encoding
    logger.info(f"Loading SigLIP model for text encoding...")
    feature_extractor = SigLIPFeatureExtractor(
        config['aggregation']['siglip_model'],
        exp_config['device']
    )

    # Load Search3D queries
    logger.info("Loading Search3D queries...")
    try:
        # Add eval directory to path for imports
        eval_dir = Path(__file__).parent.parent / 'eval' / 'search3d'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))

        from data.multiscan_search3d_constants import (
            VALID_JOINT_TUPLE_NAMES,
            VALID_JOINT_SEMANTIC_IDS
        )

        # Create queries from Search3D constants
        # Each query: (object_text, part_text, label_id, num_instances)
        queries = []
        for (obj_text, part_text), label_id in zip(VALID_JOINT_TUPLE_NAMES, VALID_JOINT_SEMANTIC_IDS):
            # For now, assume single instance per class
            # TODO: Load actual instance counts from GT if needed
            queries.append((obj_text, part_text, label_id, 1))

        logger.info(f"Loaded {len(queries)} Search3D queries")
        logger.info(f"Sample queries: {queries[:5]}")

        # Optional: filter queries to labels present in GT for this scene
        eval_cfg = config.get('evaluation', {})
        scene_label_counts = load_scene_label_distribution(
            eval_cfg.get('gt_path'),
            scene_name
        )
        if scene_label_counts:
            allowed_labels = set(scene_label_counts.keys())
            filtered_queries = [q for q in queries if q[2] in allowed_labels]
            if filtered_queries:
                logger.info(
                    "Scene %s: restricting queries to %d/%d GT label_ids",
                    scene_name, len(filtered_queries), len(queries)
                )
                sample_labels = sorted(allowed_labels)[:10]
                logger.info("  GT label ids (sample): %s", sample_labels)
                queries = filtered_queries
            else:
                logger.warning(
                    "GT label filtering removed all queries for scene %s; keeping original list",
                    scene_name
                )

    except Exception as e:
        logger.warning(f"Failed to load Search3D queries: {e}")
        logger.warning("Falling back to demo queries...")
        # Fallback to valid Search3D queries (not demo values!)
        queries = [
            ("door", "frame", 4008, 1),         # door + frame
            ("cabinet", "door", 11002, 1),      # cabinet + door
            ("toilet", "lid", 32005, 1),        # toilet + lid
            ("toilet", "tank_cover", 32007, 1), # toilet + tank_cover
            ("chair", "base", 8010, 1),         # chair + base
        ]

    logger.info(f"Running inference on {len(queries)} queries...")

    # Setup output directory (with optional subdirectory for multi-strategy testing)
    base_output_dir = Path(exp_config['inference_output_dir'])
    if output_subdir:
        output_dir = base_output_dir / output_subdir
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    predictions = []
    prediction_masks = []  # Store masks for evaluation
    prediction_results = []  # Store full Prediction objects for Search3D export
    start_time = time.time()

    for object_text, part_text, label_id, num_instances in queries:
        logger.info(f"  Query: object='{object_text}', part='{part_text}' (label={label_id}, instances={num_instances})")

        # Encode text queries
        object_features = feature_extractor.extract_text_features([object_text])
        part_features = feature_extractor.extract_text_features([part_text])

        object_feat = object_features[0]
        part_feat = part_features[0]

        # Run inference
        results = pipeline.infer_single_query(
            object_query_feat=object_feat,
            part_query_feat=part_feat,
            label_id=label_id,
        )

        # Store predictions
        for i, result in enumerate(results):
            pred_dict = {
                'prediction_id': len(predictions),  # Unique ID for this prediction
                'object_query': object_text,
                'part_query': part_text,
                'label_id': label_id,
                'num_instances': num_instances,
                'score': float(result.score),
                'mask_points': int(result.mask.sum()),
            }

            # Add metadata if available
            if result.metadata:
                pred_dict.update({
                    'object_level': result.metadata.get('object_level'),
                    'part_level': result.metadata.get('part_level'),
                    'object_id': result.metadata.get('object_id'),
                    'part_id': result.metadata.get('part_id'),
                })

            predictions.append(pred_dict)
            prediction_masks.append(result.mask)  # Store full mask
            prediction_results.append(result)  # Store full result for metadata access

        logger.info(f"    Found {len(results)} predictions")

    elapsed = time.time() - start_time

    # Save predictions JSON
    predictions_path = output_dir / 'predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    # Save masks NPZ for evaluation
    if len(prediction_masks) > 0:
        import numpy as np
        masks_array = np.column_stack([m for m in prediction_masks])  # Shape: (N_points, N_predictions)
        masks_bool = masks_array.astype(bool)
        masks_path = output_dir / 'prediction_masks.npz'
        np.savez_compressed(
            masks_path,
            masks=masks_bool,
            num_predictions=len(predictions)
        )
        logger.info(f"  Saved prediction masks: {masks_path}")

        if config['inference'].get('export_predictions_ply', False):
            try:
                points, colors = load_scene_pointcloud(exp_config['scene_path'])
                export_dir = output_dir / 'ply_exports'
                max_exports = int(config['inference'].get('ply_export_max_predictions', 20))
                palette_cfg = config['inference'].get('ply_export_colors', None)
                include_full_scene = bool(
                    config['inference'].get('ply_export_include_full_scene', True)
                )
                background_cfg = config['inference'].get('ply_export_background_color', [80, 80, 80])
                if background_cfg is None:
                    background_color = None
                elif (
                    isinstance(background_cfg, (list, tuple))
                    and len(background_cfg) == 3
                ):
                    background_color = tuple(int(c) for c in background_cfg)
                else:
                    logger.warning(
                        "Invalid ply_export_background_color=%s; falling back to gray", background_cfg
                    )
                    background_color = (80, 80, 80)
                default_palette = [
                    (255, 99, 71),
                    (102, 205, 170),
                    (65, 105, 225),
                    (218, 112, 214),
                    (255, 215, 0),
                    (0, 191, 255),
                    (144, 238, 144),
                    (255, 140, 0),
                ]
                if palette_cfg:
                    palette = [
                        tuple(int(c) for c in color)
                        for color in palette_cfg
                        if isinstance(color, (list, tuple)) and len(color) == 3
                    ]
                    if not palette:
                        palette = default_palette
                else:
                    palette = default_palette

                exported = 0
                limit = min(len(predictions), max_exports)
                for idx in range(limit):
                    mask = masks_bool[:, idx]
                    pred = predictions[idx]
                    color = palette[idx % len(palette)]
                    ply_path = export_dir / f"pred_{idx:04d}_label_{pred['label_id']}.ply"
                    if save_mask_as_ply(
                        points,
                        colors,
                        mask,
                        ply_path,
                        highlight_color=color,
                        include_full_scene=include_full_scene,
                        background_color=background_color,
                    ):
                        exported += 1

                logger.info("  Exported %d/%d prediction PLY files to %s", exported, limit, export_dir)
            except Exception as e:
                logger.warning("Failed to export predictions as PLY: %s", e, exc_info=True)
    else:
        masks_path = None

    # Export Search3D format (obj_part only)
    search3d_export_stats = None
    if len(prediction_masks) > 0 and config['evaluation'].get('enabled', False):
        try:
            from my3dis.aggregation.search3d_formatter import (
                export_predictions_search3d,
                load_point_cloud_for_export
            )

            logger.info("Exporting predictions in Search3D format...")

            # Load point cloud to get total number of points
            scene_path = Path(exp_config['scene_path'])
            try:
                points = load_point_cloud_for_export(scene_path)
                num_points = len(points)
                logger.info(f"  Scene has {num_points} points")
            except Exception as e:
                logger.warning(f"Failed to load point cloud: {e}")
                logger.warning("Using mask dimensions to infer number of points")
                num_points = len(prediction_masks[0])

            # Create Prediction-like objects for formatter
            class SimplePrediction:
                def __init__(self, mask, label_id, score, metadata=None):
                    self.mask = mask
                    self.label_id = label_id
                    self.score = score
                    self.metadata = metadata or {}

            search3d_export_stats = {}

            logger.info("Exporting part-level predictions...")

            # Create predictions with part masks
            part_pred_list = []
            for i, (result, pred_dict) in enumerate(zip(prediction_results, predictions)):
                # Use part_mask from metadata if available
                if result.metadata and 'part_mask' in result.metadata:
                    part_mask = result.metadata['part_mask']
                else:
                    # Fallback to main mask
                    part_mask = result.mask

                part_pred_list.append(SimplePrediction(
                    mask=part_mask,
                    label_id=pred_dict['label_id'],  # Use composite ID
                    score=pred_dict['score'],
                    metadata=pred_dict
                ))

            part_stats = export_predictions_search3d(
                predictions=part_pred_list,
                num_points=num_points,
                output_path=output_dir / f"{scene_name}_obj_part_inst.txt",
                format_type='obj_part',
            )
            search3d_export_stats['obj_part'] = part_stats
            logger.info(f"  ✓ Part-level: {part_stats['output_path']}")

            logger.info("✓ Search3D format export completed")

        except Exception as e:
            logger.warning(f"Failed to export Search3D format: {e}", exc_info=True)
            search3d_export_stats = None

    logger.info(f"✓ Inference completed in {elapsed:.1f}s")
    logger.info(f"  Output: {predictions_path}")
    logger.info(f"  Total predictions: {len(predictions)}")

    # Return statistics for reporting
    return {
        'predictions_path': str(predictions_path),
        'masks_path': str(masks_path) if masks_path else None,
        'num_predictions': len(predictions),
        'elapsed_time': elapsed,
        'strategy': inf_config_dict['strategy'],
        'output_dir': str(output_dir),
        'search3d_export': search3d_export_stats,
    }


def generate_strategy_report(
    strategy_name: str,
    inference_stats: Dict,
    config: Dict,
    output_path: str,
    eval_results: Optional[Dict] = None
):
    """
    Generate evaluation report for a single strategy.

    Args:
        strategy_name: Name of the strategy (e.g., "independent", "hierarchical")
        inference_stats: Statistics from run_inference_stage
        config: Full configuration dictionary
        output_path: Path to save the report
        eval_results: Optional evaluation results from run_evaluation_stage
    """
    logger.info(f"Generating report for strategy: {strategy_name}")

    report_lines = [
        f"# Inference Strategy Report: {strategy_name.upper()}",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration Summary",
        "",
        f"- **Strategy**: {strategy_name}",
        f"- **Experiment Dir**: {config['experiment']['exp_dir']}",
        f"- **Scene Path**: {config['experiment']['scene_path']}",
        f"- **Device**: {config['experiment']['device']}",
        "",
        "## Inference Settings",
        "",
    ]

    # Add strategy-specific settings
    inf_config = config['inference']
    if strategy_name == 'independent':
        ind_cfg = inf_config.get('independent', {})
        report_lines.extend([
            f"- **Object Levels**: {ind_cfg.get('object_levels', [])}",
            f"- **Part Levels**: {ind_cfg.get('part_levels', [])}",
            f"- **Top-K per Level**: {ind_cfg.get('top_k_per_level', 0)}",
            f"- **Retrieval Threshold**: {ind_cfg.get('retrieval_threshold', 0)}",
            f"- **Min Retrieval Threshold**: {ind_cfg.get('min_retrieval_threshold', 0)}",
        ])
    elif strategy_name == 'hierarchical':
        hier_cfg = inf_config.get('hierarchical', {})
        report_lines.extend([
            f"- **Coarse Top-K**: {hier_cfg.get('coarse_top_k', 0)}",
            f"- **Object Top-K**: {hier_cfg.get('object_top_k', 0)}",
            f"- **Part Top-K**: {hier_cfg.get('part_top_k', 0)}",
            f"- **Coarse Threshold**: {hier_cfg.get('coarse_threshold', 0)}",
            f"- **Refinement Threshold**: {hier_cfg.get('refinement_threshold', 0)}",
        ])

    # Add common settings
    report_lines.extend([
        "",
        "### Retrieval Settings",
        f"- **Temperature**: {inf_config['retrieval']['temperature']}",
        f"- **Min Similarity**: {inf_config['retrieval']['min_similarity']}",
        "",
        "### NMS Settings",
        f"- **IoU Threshold**: {inf_config['nms']['iou_threshold']}",
        f"- **Keep Top-K**: {inf_config['nms']['keep_top_k']}",
        f"- **Use Soft NMS**: {inf_config['nms']['use_soft_nms']}",
        "",
        "### Scoring Settings",
        f"- **Method**: {inf_config['scoring']['method']}",
        f"- **Weights**: object={inf_config['scoring']['weights']['object']}, part={inf_config['scoring']['weights']['part']}, geometric={inf_config['scoring']['weights']['geometric']}",
        "",
        "## Results",
        "",
        f"- **Total Predictions**: {inference_stats['num_predictions']}",
        f"- **Inference Time**: {inference_stats['elapsed_time']:.2f}s",
        f"- **Output Directory**: {inference_stats['output_dir']}",
        f"- **Predictions File**: {inference_stats['predictions_path']}",
        "",
        "## Evaluation Metrics",
        "",
    ])

    # Add evaluation results if available
    if eval_results and eval_results.get('status') == 'success':
        report_lines.extend([
            f"- **mAP**: {eval_results['mAP']:.3f}",
            f"- **AP@50**: {eval_results['AP50']:.3f}",
            f"- **AP@25**: {eval_results['AP25']:.3f}",
            "",
            "### Per-Class Results",
            "",
            "| Class | AP | AP@50 | AP@25 |",
            "|-------|-----|-------|-------|",
        ])

        # Add per-class results if available
        if 'metrics' in eval_results and 'classes' in eval_results['metrics']:
            for class_name, class_metrics in eval_results['metrics']['classes'].items():
                ap = class_metrics.get('ap', 0)
                ap50 = class_metrics.get('ap50%', 0)
                ap25 = class_metrics.get('ap25%', 0)
                report_lines.append(
                    f"| {class_name} | {ap:.3f} | {ap50:.3f} | {ap25:.3f} |"
                )
    else:
        if eval_results and eval_results.get('status') == 'failed':
            report_lines.append(f"*Evaluation failed: {eval_results.get('error', 'unknown error')}*")
        else:
            report_lines.append("*Evaluation metrics not available or evaluation was skipped.*")

    report_lines.extend([
        "",
        "---",
        "",
        f"Report generated by My3DIS full pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"  Report saved: {output_path}")


def generate_comparison_report(
    strategy_results: Dict[str, Dict],
    config: Dict,
    output_path: str
):
    """
    Generate comparison report for all tested strategies.

    Args:
        strategy_results: Dictionary mapping strategy_name -> inference_stats
        config: Full configuration dictionary
        output_path: Path to save the comparison report
    """
    logger.info("Generating strategy comparison report...")

    report_lines = [
        "# Multi-Strategy Inference Comparison Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment Overview",
        "",
        f"- **Experiment Dir**: {config['experiment']['exp_dir']}",
        f"- **Scene Path**: {config['experiment']['scene_path']}",
        f"- **Device**: {config['experiment']['device']}",
        f"- **Levels**: {config['experiment']['levels']}",
        "",
        "## Strategy Results Summary",
        "",
        "| Strategy | Predictions | Time (s) | Output Dir |",
        "|----------|-------------|----------|------------|",
    ]

    # Add rows for each strategy
    for strategy_name in sorted(strategy_results.keys()):
        stats = strategy_results[strategy_name]
        report_lines.append(
            f"| **{strategy_name}** | {stats['num_predictions']} | "
            f"{stats['elapsed_time']:.2f} | `{Path(stats['output_dir']).name}` |"
        )

    report_lines.extend([
        "",
        "## Detailed Comparison",
        "",
    ])

    # Add detailed section for each strategy
    for strategy_name in sorted(strategy_results.keys()):
        stats = strategy_results[strategy_name]
        report_lines.extend([
            f"### {strategy_name.upper()} Strategy",
            "",
            f"- **Predictions**: {stats['num_predictions']}",
            f"- **Inference Time**: {stats['elapsed_time']:.2f}s",
            f"- **Predictions per Second**: {stats['num_predictions'] / stats['elapsed_time']:.2f}",
            f"- **Output Directory**: `{stats['output_dir']}`",
            f"- **Predictions File**: `{stats['predictions_path']}`",
            f"- **Individual Report**: [{strategy_name}_report.md](./{Path(stats['output_dir']).name}/{strategy_name}_report.md)",
            "",
        ])

    # Add performance comparison
    if len(strategy_results) > 1:
        fastest = min(strategy_results.items(), key=lambda x: x[1]['elapsed_time'])
        most_predictions = max(strategy_results.items(), key=lambda x: x[1]['num_predictions'])

        report_lines.extend([
            "## Performance Highlights",
            "",
            f"- **Fastest Strategy**: {fastest[0]} ({fastest[1]['elapsed_time']:.2f}s)",
            f"- **Most Predictions**: {most_predictions[0]} ({most_predictions[1]['num_predictions']} predictions)",
            "",
        ])

    report_lines.extend([
        "## Evaluation Metrics Comparison",
        "",
        "*Evaluation metrics comparison will be added when evaluation stage is implemented.*",
        "",
        "---",
        "",
        f"Report generated by My3DIS full pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"  Comparison report saved: {output_path}")


def convert_predictions_to_eval_format(
    predictions_path: str,
    masks_path: str,
    num_points: int
) -> Dict:
    """
    Convert predictions to Search3D evaluation format.

    Args:
        predictions_path: Path to predictions JSON file
        masks_path: Path to prediction_masks.npz file
        num_points: Number of points in the scene

    Returns:
        Dictionary with pred_classes, pred_scores, pred_masks
    """
    import numpy as np

    # Load predictions JSON
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        logger.warning(f"No predictions found in {predictions_path}")
        return {
            'pred_classes': np.array([], dtype=np.int32),
            'pred_scores': np.array([], dtype=np.float32),
            'pred_masks': np.zeros((num_points, 0), dtype=bool)
        }

    # Load masks NPZ
    masks_data = np.load(masks_path)
    pred_masks = masks_data['masks']  # Shape: (N_points, N_predictions)

    # Extract classes and scores
    pred_classes = []
    pred_scores = []

    for pred in predictions:
        label_id = pred.get('label_id', -1)
        score = pred.get('score', 0.0)
        pred_classes.append(label_id)
        pred_scores.append(score)

    return {
        'pred_classes': np.array(pred_classes, dtype=np.int32),
        'pred_scores': np.array(pred_scores, dtype=np.float32),
        'pred_masks': pred_masks
    }


def run_evaluation_stage(
    config: Dict,
    inference_output_dir: str,
) -> Dict:
    """
    Run evaluation stage: Compute mAP for OV part-level predictions using the Search3D protocol.

    Args:
        config: Full configuration dictionary
        inference_output_dir: Directory containing inference outputs
            Expected files:
            - predictions.json
            - prediction_masks.npz (old format, for backward compat)
            - scene_*_obj_part_inst.txt (Search3D part-level format)

    Returns:
        Dictionary with evaluation metrics:
        {
            'obj_part': {...},    # Part-level metrics
            'summary': {...}      # Combined summary
        }
    """
    logger.info("="*80)
    logger.info("STAGE 3: EVALUATION (Search3D OV-Part)")
    logger.info("="*80)

    eval_config = config['evaluation']
    exp_dir = Path(config['experiment']['exp_dir'])
    scene_name = exp_dir.name  # e.g., scene_00005_00
    inference_output_dir = Path(inference_output_dir)

    # Get evaluation modes from config (object evaluation removed)
    configured_modes = eval_config.get('eval_modes', ['obj_part'])
    if not isinstance(configured_modes, list):
        configured_modes = [configured_modes]

    unsupported_modes = [mode for mode in configured_modes if mode != 'obj_part']
    if unsupported_modes:
        logger.warning(
            "Skipping unsupported evaluation modes (object-level evaluation removed): %s",
            unsupported_modes
        )

    eval_modes = ['obj_part']

    logger.info(f"Scene: {scene_name}")
    logger.info(f"Evaluation modes: {eval_modes}")

    # Check for backward compatibility with old config format
    gt_paths = eval_config.get('gt_paths', {})
    if not gt_paths and eval_config.get('gt_path'):
        # Fall back to old single GT path (treat as obj_part)
        logger.warning("Using deprecated 'gt_path' config. Please migrate to 'gt_paths'.")
        gt_paths = {'obj_part': eval_config['gt_path']}

    if not gt_paths:
        logger.warning("No ground truth paths specified, skipping evaluation")
        return {'status': 'skipped', 'reason': 'no_gt_paths'}

    # Results for all modes
    results = {}

    # Evaluate each mode
    for mode in eval_modes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {mode.upper()}")
        logger.info(f"{'='*80}")

        # Check if GT path exists for this mode
        if mode not in gt_paths:
            logger.warning(f"GT path not configured for mode '{mode}', skipping")
            results[mode] = {'status': 'skipped', 'reason': 'no_gt_path'}
            continue

        gt_dir = Path(gt_paths[mode])
        if not gt_dir.exists():
            logger.warning(f"GT directory not found: {gt_dir}")
            results[mode] = {'status': 'skipped', 'reason': 'gt_dir_not_found'}
            continue

        # Determine GT filename based on mode (obj_part only)
        if mode == 'obj_part':
            gt_filename = f"{scene_name}_obj_part_inst.txt"
            pred_filename = f"{scene_name}_obj_part_inst.txt"
        else:
            logger.warning(f"Unknown evaluation mode: {mode}")
            results[mode] = {'status': 'skipped', 'reason': 'unknown_mode'}
            continue

        gt_file = gt_dir / gt_filename
        pred_file = inference_output_dir / pred_filename

        # Check files exist
        if not gt_file.exists():
            logger.warning(f"GT file not found: {gt_file}")
            results[mode] = {'status': 'skipped', 'reason': 'gt_file_not_found'}
            continue

        if not pred_file.exists():
            logger.warning(f"Prediction file not found: {pred_file}")
            results[mode] = {'status': 'skipped', 'reason': 'pred_file_not_found'}
            continue

        logger.info(f"  GT file: {gt_file}")
        logger.info(f"  Pred file: {pred_file}")

        try:
            # Get evaluation script path
            eval_scripts = eval_config.get('eval_scripts', {})
            if mode not in eval_scripts:
                logger.warning(f"Evaluation script not configured for mode '{mode}'")
                results[mode] = {'status': 'skipped', 'reason': 'no_eval_script'}
                continue

            eval_script_rel = eval_scripts[mode]
            project_root = Path(__file__).parent.parent
            eval_script = project_root / eval_script_rel

            if not eval_script.exists():
                logger.warning(f"Evaluation script not found: {eval_script}")
                results[mode] = {'status': 'skipped', 'reason': 'eval_script_not_found'}
                continue

            # Add eval directory to sys.path
            eval_dir = eval_script.parent
            if str(eval_dir) not in sys.path:
                sys.path.insert(0, str(eval_dir))

            # Import evaluation functions (obj_part only)
            if mode == 'obj_part':
                from eval_semantic_instance_parts_OV import (
                    evaluate_scan_from_files,
                    compute_averages
                )

            # Run evaluation
            logger.info(f"Running evaluation for {mode}...")
            matches = evaluate_scan_from_files(
                pred_file=str(pred_file),
                gt_file=str(gt_file),
                scene_name=scene_name
            )

            # Compute metrics
            from eval_semantic_instance_parts_OV import evaluate_matches
            ap = evaluate_matches({scene_name: matches})
            avgs = compute_averages(ap)

            logger.info(f"✓ {mode.upper()} evaluation completed")
            logger.info(f"  mAP: {avgs['all_ap']:.3f}")
            logger.info(f"  AP@50: {avgs['all_ap_50%']:.3f}")
            logger.info(f"  AP@25: {avgs['all_ap_25%']:.3f}")

            results[mode] = {
                'status': 'success',
                'metrics': avgs,
                'scene': scene_name,
                'mAP': avgs['all_ap'],
                'AP50': avgs['all_ap_50%'],
                'AP25': avgs['all_ap_25%'],
            }

        except Exception as e:
            logger.error(f"{mode.upper()} evaluation failed: {e}", exc_info=True)
            results[mode] = {
                'status': 'failed',
                'error': str(e)
            }

    # Generate summary
    summary = {
        'total_modes': len(eval_modes),
        'successful': sum(1 for r in results.values() if r.get('status') == 'success'),
        'failed': sum(1 for r in results.values() if r.get('status') == 'failed'),
        'skipped': sum(1 for r in results.values() if r.get('status') == 'skipped'),
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  Total modes: {summary['total_modes']}")
    logger.info(f"  Successful: {summary['successful']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Skipped: {summary['skipped']}")

    # Print detailed results
    for mode, result in results.items():
        if result.get('status') == 'success':
            logger.info(f"\n  {mode.upper()}:")
            logger.info(f"    mAP: {result['mAP']:.3f}")
            logger.info(f"    AP@50: {result['AP50']:.3f}")
            logger.info(f"    AP@25: {result['AP25']:.3f}")

    results['summary'] = summary
    return results


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
    logger.info(f"  Multi-strategy testing: {config['inference'].get('test_all_strategies', False)}")
    if config['inference'].get('test_all_strategies', False):
        logger.info(f"  Strategies to test:     {config['inference'].get('strategies_to_test', [])}")
    else:
        logger.info(f"  Single strategy:        {config['inference']['strategy']}")
    logger.info(f"  NMS IoU:         {config['inference']['nms']['iou_threshold']}")
    logger.info(f"  Keep top-k:      {config['inference']['nms']['keep_top_k']}")
    logger.info("")
    logger.info("Evaluation settings:")
    logger.info(f"  GT path:         {config['evaluation']['gt_path']}")
    logger.info(f"  Query file:      {config['evaluation']['query_file']}")
    logger.info("")
    logger.info("Stages enabled:")
    logger.info(f"  Aggregation:    {config['aggregation']['enabled']}")
    logger.info(f"  Inference:      {config['inference']['enabled']}")
    logger.info(f"  Evaluation:     {config['evaluation']['enabled']}")
    logger.info("="*80)
    logger.info("")

    # Run pipeline stages
    proposal_data_paths = None
    strategy_results = {}

    try:
        # Stage 1: Aggregation
        if config['aggregation']['enabled']:
            proposal_data_paths = run_aggregation_stage(config)
        else:
            logger.info("Skipping aggregation stage (loading existing outputs)")
            # Load existing aggregation outputs
            agg_output_dir = Path(config['experiment']['aggregation_output_dir'])
            proposal_data_paths = {}

            # Auto-detect or use specified levels
            levels_to_check = config['experiment'].get('levels', None)
            if levels_to_check is None:
                from my3dis.aggregation.aggregation_pipeline import detect_available_levels
                exp_dir = Path(config['experiment']['exp_dir'])
                detected_levels = detect_available_levels(exp_dir)
                if detected_levels:
                    levels_to_check = detected_levels
                    logger.info(f"Auto-detected levels: {levels_to_check}")
                else:
                    levels_to_check = [2, 4, 6]
                    logger.warning(f"No levels detected, using default: {levels_to_check}")

            for level in levels_to_check:
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

            # Check if multi-strategy testing is enabled
            test_all_strategies = config['inference'].get('test_all_strategies', False)

            if test_all_strategies:
                # Multi-strategy testing mode
                strategies_to_test = config['inference'].get('strategies_to_test', ['independent', 'hierarchical'])
                logger.info("")
                logger.info("="*80)
                logger.info(f"MULTI-STRATEGY TESTING MODE: Testing {len(strategies_to_test)} strategies")
                logger.info("="*80)

                for strategy in strategies_to_test:
                    logger.info("")
                    logger.info(f"--- Testing strategy: {strategy} ---")

                    # Run inference with this strategy
                    inference_stats = run_inference_stage(
                        config,
                        proposal_data_paths,
                        strategy=strategy,
                        output_subdir=strategy
                    )
                    strategy_results[strategy] = inference_stats

                # Generate comparison report
                logger.info("")
                logger.info("="*80)
                logger.info("GENERATING STRATEGY COMPARISON REPORT")
                logger.info("="*80)

                comparison_report_path = Path(config['experiment']['inference_output_dir']) / "strategy_comparison.md"
                generate_comparison_report(
                    strategy_results=strategy_results,
                    config=config,
                    output_path=str(comparison_report_path)
                )

            else:
                # Single strategy mode
                logger.info("")
                logger.info("Single strategy mode")
                inference_stats = run_inference_stage(config, proposal_data_paths)
                strategy_name = config['inference']['strategy']
                strategy_results[strategy_name] = inference_stats

        # Stage 3: Evaluation
        if config['evaluation']['enabled']:
            if not strategy_results:
                logger.error("No predictions available for evaluation")
                return 1

            # Infer num_points from first NPZ file
            first_level = config['experiment']['levels'][0]
            agg_output_dir = Path(config['experiment']['aggregation_output_dir'])
            first_npz = agg_output_dir / f"proposal_data_level{first_level}.npz"
            if first_npz.exists():
                import numpy as np
                data = np.load(first_npz)
                num_points = data['proposal_masks'].shape[0]
                logger.info(f"Detected num_points={num_points} from aggregation output")
            else:
                num_points = None
                logger.warning("Could not infer num_points, evaluation may fail")

            # Run evaluation for each strategy
            eval_results = {}
            for strategy_name, stats in strategy_results.items():
                logger.info(f"\nEvaluating strategy: {strategy_name}")
                eval_result = run_evaluation_stage(
                    config=config,
                    inference_output_dir=stats['output_dir'],
                )
                eval_results[strategy_name] = eval_result

                # Log evaluation results (updated for multi-mode)
                if eval_result.get('summary', {}).get('successful', 0) > 0:
                    logger.info(f"  ✓ {strategy_name} evaluation completed:")
                    for mode, result in eval_result.items():
                        if mode != 'summary' and result.get('status') == 'success':
                            logger.info(f"    {mode}: mAP={result['mAP']:.3f}, AP50={result['AP50']:.3f}, AP25={result['AP25']:.3f}")
                else:
                    logger.warning(f"  ✗ {strategy_name}: No successful evaluations")

                # Generate strategy report with evaluation results
                strategy_report_path = Path(stats['output_dir']) / f"{strategy_name}_report.md"
                generate_strategy_report(
                    strategy_name=strategy_name,
                    inference_stats=stats,
                    config=config,
                    output_path=str(strategy_report_path),
                    eval_results=eval_result
                )

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
