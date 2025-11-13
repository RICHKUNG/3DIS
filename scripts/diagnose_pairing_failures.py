#!/usr/bin/env python3
"""
Diagnose why certain queries produce 0 valid pairs.

Analyzes geometric constraints and provides detailed breakdown of rejection reasons.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def compute_containment_ratio(part_mask, object_mask):
    """Compute how much of the part is contained in the object."""
    part_bool = part_mask.astype(bool)
    object_bool = object_mask.astype(bool)

    part_size = np.count_nonzero(part_bool)
    if part_size == 0:
        return 0.0

    intersection = np.count_nonzero(part_bool & object_bool)
    return intersection / part_size


def compute_scale_ratio(mask1, mask2):
    """Compute size ratio between two masks."""
    size1 = np.count_nonzero(mask1)
    size2 = np.count_nonzero(mask2)

    if size2 == 0:
        return float('inf')

    return size1 / size2


def compute_iou(mask1, mask2):
    """Compute IoU between two masks."""
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.count_nonzero(mask1_bool & mask2_bool)
    union = np.count_nonzero(mask1_bool | mask2_bool)

    if union == 0:
        return 0.0

    return intersection / union


def diagnose_pair(part_mask, obj_mask, config):
    """Diagnose why a specific pair might be rejected."""

    # Get pairing config
    pairing_cfg = config.get('inference', {}).get('pairing', {})
    containment_threshold = pairing_cfg.get('containment_threshold', 0.5)
    scale_range = tuple(pairing_cfg.get('scale_range', [0.01, 0.9]))

    # Compute metrics
    containment = compute_containment_ratio(part_mask, obj_mask)
    scale = compute_scale_ratio(part_mask, obj_mask)
    iou = compute_iou(part_mask, obj_mask)

    # Check validity
    is_valid = True
    if containment < containment_threshold:
        is_valid = False
    if not (scale_range[0] <= scale <= scale_range[1]):
        is_valid = False

    # Build diagnostic info
    diagnosis = {
        'is_valid': is_valid,
        'metrics': {
            'containment_ratio': containment,
            'scale_ratio': scale,
            'iou': iou,
        },
        'thresholds': {
            'containment_threshold': containment_threshold,
            'scale_range': scale_range,
        },
        'violations': []
    }

    # Identify violations
    if containment < containment_threshold:
        diagnosis['violations'].append(
            f"Containment too low: {containment:.3f} < {containment_threshold}"
        )

    if not (scale_range[0] <= scale <= scale_range[1]):
        diagnosis['violations'].append(
            f"Scale out of range: {scale:.3f} not in [{scale_range[0]}, {scale_range[1]}]"
        )

    return diagnosis


def diagnose_query_failures(
    aggregation_output_dir: Path,
    config_path: Path,
    query: str = None,
    max_samples: int = 10
):
    """
    Diagnose pairing failures for specific queries.

    Args:
        aggregation_output_dir: Path to aggregation output
        config_path: Path to pipeline config
        query: Optional query to focus on (e.g., "door-frame")
        max_samples: Max number of failed pairs to sample per query
    """

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load aggregated proposals from NPZ files
    logger.info(f"Loading proposals from {aggregation_output_dir}")
    proposals = {}

    for level in config['experiment']['levels']:
        npz_path = aggregation_output_dir / f"proposal_data_level{level}.npz"
        if not npz_path.exists():
            logger.warning(f"NPZ file not found: {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=False)
        # proposal_masks shape: (num_points, num_proposals)
        # We need to split by columns to get individual proposal masks
        masks_array = data['proposal_masks']
        num_proposals = masks_array.shape[1]

        proposals[level] = {
            'masks': [masks_array[:, i] for i in range(num_proposals)],
            'ids': data.get('proposal_ids', None)
        }
        logger.info(f"  Level {level}: {num_proposals} proposals")

    # Get pairing config
    pairing_cfg = config.get('inference', {}).get('pairing', {})
    containment_threshold = pairing_cfg.get('containment_threshold', 0.5)
    scale_range = tuple(pairing_cfg.get('scale_range', [0.01, 0.9]))

    logger.info("\n" + "="*80)
    logger.info("PAIRING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Containment threshold: {containment_threshold}")
    logger.info(f"Scale range: {scale_range}")
    logger.info("")

    # Sample proposals for analysis
    # Get object and part levels
    obj_levels = config['inference'].get('independent', {}).get('object_levels', ['L2', 'L4'])
    part_levels = config['inference'].get('independent', {}).get('part_levels', ['L4', 'L6'])

    # Sample a few object and part proposals
    obj_masks = []
    for level in obj_levels:
        level_int = int(level[1])
        if level_int in proposals:
            obj_masks.extend(proposals[level_int]['masks'][:5])  # Sample first 5

    part_masks = []
    for level in part_levels:
        level_int = int(level[1])
        if level_int in proposals:
            part_masks.extend(proposals[level_int]['masks'][:5])  # Sample first 5

    if not obj_masks or not part_masks:
        logger.error("No proposals found to analyze")
        return

    logger.info("="*80)
    logger.info("SAMPLE PAIR ANALYSIS")
    logger.info("="*80)
    logger.info(f"Analyzing {len(obj_masks)} objects × {len(part_masks)} parts")
    logger.info("")

    # Analyze pairs
    valid_count = 0
    violation_stats = {
        'containment': 0,
        'scale': 0,
        'both': 0
    }

    sample_diagnoses = []

    for obj_idx, obj_mask in enumerate(obj_masks):
        for part_idx, part_mask in enumerate(part_masks):
            # Diagnose
            diagnosis = diagnose_pair(part_mask, obj_mask, config)

            if diagnosis['is_valid']:
                valid_count += 1
            else:
                # Count violations
                if len(diagnosis['violations']) == 2:
                    violation_stats['both'] += 1
                elif 'Containment' in diagnosis['violations'][0]:
                    violation_stats['containment'] += 1
                else:
                    violation_stats['scale'] += 1

                # Store sample for detailed output
                if len(sample_diagnoses) < max_samples:
                    sample_diagnoses.append({
                        'obj_idx': obj_idx,
                        'part_idx': part_idx,
                        'diagnosis': diagnosis
                    })

    total_pairs = len(obj_masks) * len(part_masks)
    invalid_count = total_pairs - valid_count

    logger.info("="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    logger.info(f"Total pairs analyzed: {total_pairs}")
    logger.info(f"Valid pairs: {valid_count} ({100*valid_count/total_pairs:.1f}%)")
    logger.info(f"Invalid pairs: {invalid_count} ({100*invalid_count/total_pairs:.1f}%)")
    logger.info("")
    logger.info("Violation breakdown:")
    logger.info(f"  Containment only: {violation_stats['containment']} ({100*violation_stats['containment']/invalid_count:.1f}%)")
    logger.info(f"  Scale only: {violation_stats['scale']} ({100*violation_stats['scale']/invalid_count:.1f}%)")
    logger.info(f"  Both: {violation_stats['both']} ({100*violation_stats['both']/invalid_count:.1f}%)")
    logger.info("")

    # Show detailed examples
    logger.info("="*80)
    logger.info(f"DETAILED FAILURE EXAMPLES (showing up to {max_samples})")
    logger.info("="*80)

    for i, sample in enumerate(sample_diagnoses, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Object idx: {sample['obj_idx']}")
        logger.info(f"  Part idx: {sample['part_idx']}")
        logger.info(f"  Metrics:")
        for metric, value in sample['diagnosis']['metrics'].items():
            logger.info(f"    {metric}: {value:.4f}")
        logger.info(f"  Violations:")
        for violation in sample['diagnosis']['violations']:
            logger.info(f"    - {violation}")

    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)

    if violation_stats['containment'] > violation_stats['scale']:
        logger.info("⚠ Most failures are due to LOW CONTAINMENT")
        logger.info(f"  Current threshold: {containment_threshold}")
        logger.info("  Suggestions:")
        logger.info(f"    1. Lower containment_threshold to 0.05 or even 0.01")
        logger.info("    2. Consider swapping object/part retrieval for some queries")
        logger.info("    3. Enable bidirectional pairing (try both part→obj and obj→part)")
    elif violation_stats['scale'] > violation_stats['containment']:
        logger.info("⚠ Most failures are due to SCALE MISMATCH")
        logger.info(f"  Current scale range: {scale_range}")
        logger.info("  Suggestions:")
        logger.info(f"    1. Expand scale range to [0.0001, 0.999]")
        logger.info("    2. Check if part candidates are actually larger than objects")
    else:
        logger.info("⚠ Failures are roughly split between containment and scale")
        logger.info("  Suggestions:")
        logger.info(f"    1. Relax both constraints further")
        logger.info(f"    2. Consider alternative pairing strategies")


def main():
    parser = argparse.ArgumentParser(description='Diagnose pairing failures')
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to pipeline config YAML')
    parser.add_argument('--aggregation-output', type=Path, required=True,
                       help='Path to aggregation output directory')
    parser.add_argument('--query', type=str, default=None,
                       help='Specific query to analyze (e.g., "door-frame")')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Max failure examples to show')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    diagnose_query_failures(
        args.aggregation_output,
        args.config,
        query=args.query,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
