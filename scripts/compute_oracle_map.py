#!/usr/bin/env python3
"""
Compute Oracle mAP: best possible mAP with current 3D proposals,
ignoring semantic matching (geometry-only upper bound).

This diagnostic test determines whether the problem is in:
- 3D proposal quality (aggregation pipeline) → Oracle mAP < 0.1
- Semantic matching (SigLIP/retrieval) → Oracle mAP > 0.3
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def load_proposals(exp_dir: Path) -> Tuple[np.ndarray, List[int]]:
    """
    Load all proposals from aggregation output.

    Returns:
        masks: (num_points, num_proposals) binary masks
        proposal_ids: List of proposal IDs
    """
    agg_dir = exp_dir / "aggregation_output"

    all_masks = []
    all_ids = []

    for level in [2, 4, 6]:
        npz_path = agg_dir / f"proposal_data_level{level}.npz"
        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping level {level}")
            continue

        data = np.load(npz_path, allow_pickle=True)

        # proposal_masks: (num_points, num_proposals_at_level)
        level_masks = data['proposal_masks']
        level_ids = data['proposal_ids']

        all_masks.append(level_masks)
        all_ids.extend(level_ids.tolist())

        print(f"Level {level}: {len(level_ids)} proposals")

    # Concatenate all levels: (num_points, total_proposals)
    combined_masks = np.concatenate(all_masks, axis=1)

    print(f"\nTotal proposals across all levels: {len(all_ids)}")
    print(f"Combined mask shape: {combined_masks.shape}")

    return combined_masks, all_ids


def load_ground_truth(gt_path: Path) -> Tuple[np.ndarray, List[int]]:
    """
    Load ground truth annotations in Search3D format.

    Returns:
        gt_masks: (num_points, num_gt_instances) binary masks
        gt_labels: List of class labels for each instance
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    # Load per-point instance labels
    labels = []
    with open(gt_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                continue
            labels.append(int(stripped))

    labels = np.array(labels, dtype=np.int64)
    num_points = len(labels)

    # Extract unique instance IDs (exclude background=0)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]

    if unique_ids.size == 0:
        raise ValueError("No GT instances found (all background)")

    # Convert to binary masks: (num_points, num_instances)
    gt_masks = labels[:, None] == unique_ids[None, :]

    # Extract class labels (ID // 1000)
    gt_labels = (unique_ids // 1000).tolist()

    print(f"\nGround Truth:")
    print(f"  Points: {num_points}")
    print(f"  Instances: {len(unique_ids)}")
    print(f"  Background coverage: {(labels == 0).sum() / num_points * 100:.1f}%")
    print(f"  Labeled coverage: {(labels > 0).sum() / num_points * 100:.1f}%")

    return gt_masks, gt_labels


def compute_oracle_map(
    proposal_masks: np.ndarray,
    gt_masks: np.ndarray,
    iou_thresholds: List[float] = [0.25, 0.5, 0.75]
) -> Dict[str, float]:
    """
    Compute Oracle mAP: best possible mAP assuming perfect semantic matching.

    For each GT instance, find the proposal with highest IoU (ignoring semantics).
    """
    num_gt = gt_masks.shape[1]
    num_proposals = proposal_masks.shape[1]

    print(f"\nComputing Oracle mAP...")
    print(f"  GT instances: {num_gt}")
    print(f"  Proposals: {num_proposals}")

    # For each GT, find best matching proposal
    best_ious = []

    for gt_idx in range(num_gt):
        gt_mask = gt_masks[:, gt_idx]

        best_iou = 0.0
        for prop_idx in range(num_proposals):
            prop_mask = proposal_masks[:, prop_idx]
            iou = compute_iou(gt_mask, prop_mask)
            best_iou = max(best_iou, iou)

        best_ious.append(best_iou)

    best_ious = np.array(best_ious)

    # Compute metrics
    results = {
        'mean_iou': float(np.mean(best_ious)),
        'median_iou': float(np.median(best_ious)),
        'max_iou': float(np.max(best_ious)),
        'min_iou': float(np.min(best_ious)),
    }

    # Compute AP at different thresholds
    for thresh in iou_thresholds:
        recall = (best_ious >= thresh).mean()
        results[f'oracle_AP@{int(thresh*100)}'] = float(recall)

    # Oracle mAP (average over thresholds 0.25-0.95, step 0.05)
    all_thresholds = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
    recalls = [(best_ious >= t).mean() for t in all_thresholds]
    results['oracle_mAP'] = float(np.mean(recalls))

    return results, best_ious


def main():
    parser = argparse.ArgumentParser(
        description="Compute Oracle mAP to diagnose proposal quality"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Experiment directory containing aggregation_output/"
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        required=True,
        help="Path to ground truth annotation file (scene_XXXXX_XX_obj_part_inst.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    gt_path = Path(args.gt_path)

    print("=" * 80)
    print("Oracle mAP Diagnostic")
    print("=" * 80)
    print(f"Experiment: {exp_dir}")
    print(f"Ground Truth: {gt_path}")
    print()

    # Load data
    print("Loading proposals...")
    proposal_masks, proposal_ids = load_proposals(exp_dir)

    print("\nLoading ground truth...")
    gt_masks, gt_labels = load_ground_truth(gt_path)

    # Sanity check
    if proposal_masks.shape[0] != gt_masks.shape[0]:
        raise ValueError(
            f"Point count mismatch: proposals={proposal_masks.shape[0]}, "
            f"GT={gt_masks.shape[0]}"
        )

    # Compute Oracle mAP
    results, best_ious = compute_oracle_map(proposal_masks, gt_masks)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Oracle mAP: {results['oracle_mAP']:.4f}")
    print(f"Oracle AP@25: {results['oracle_AP@25']:.4f}")
    print(f"Oracle AP@50: {results['oracle_AP@50']:.4f}")
    print(f"Oracle AP@75: {results['oracle_AP@75']:.4f}")
    print()
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Median IoU: {results['median_iou']:.4f}")
    print(f"Max IoU: {results['max_iou']:.4f}")
    print(f"Min IoU: {results['min_iou']:.4f}")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    oracle_ap25 = results['oracle_AP@25']

    if oracle_ap25 < 0.10:
        print("❌ CRITICAL: Oracle AP@25 < 0.10")
        print("   → 3D aggregation pipeline is BROKEN")
        print("   → Proposals do not match GT geometry")
        print("   → STOP retrieval optimization, fix proposals first")
        print()
        print("   Action: Audit aggregation_pipeline.py parameters")
        print("           Check 2D SAM2 → 3D projection logic")
    elif oracle_ap25 < 0.30:
        print("⚠️  WARNING: Oracle AP@25 = 0.10-0.30")
        print("   → Proposals have poor geometry")
        print("   → Both aggregation AND semantic matching need work")
        print("   → Fix aggregation first, then retrieval")
    elif oracle_ap25 < 0.50:
        print("✅ MODERATE: Oracle AP@25 = 0.30-0.50")
        print("   → Proposals are reasonable")
        print("   → Problem is likely in semantic matching")
        print("   → Continue with retrieval optimization")
        print()
        print("   Action: Run Feature Discriminability test (D2)")
    else:
        print("✅ GOOD: Oracle AP@25 > 0.50")
        print("   → Proposals are high quality")
        print("   → Problem is DEFINITELY in semantic matching")
        print("   → Focus on SigLIP features / retrieval logic")
        print()
        print("   Action: Run Feature Discriminability test (D2)")

    print("=" * 80)

    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add histogram of IoUs
        results['iou_histogram'] = {
            '0.00-0.10': int((best_ious < 0.10).sum()),
            '0.10-0.25': int(((best_ious >= 0.10) & (best_ious < 0.25)).sum()),
            '0.25-0.50': int(((best_ious >= 0.25) & (best_ious < 0.50)).sum()),
            '0.50-0.75': int(((best_ious >= 0.50) & (best_ious < 0.75)).sum()),
            '0.75+': int((best_ious >= 0.75).sum()),
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
