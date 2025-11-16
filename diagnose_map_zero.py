#!/usr/bin/env python3
"""
Debug script to analyze why mAP is 0.000
Checks:
1. GT vs Prediction label overlap
2. Actual IoU between predicted and GT masks
3. Evaluation matching logic
"""
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Paths
GT_FILE = Path("/media/Pluto/richkung/My3DIS/data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations/scene_00093_01_obj_part_inst.txt")
PRED_FILE = Path("/media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/inference_output/independent/scene_00093_01_obj_part_inst.txt")

def load_labels(file_path):
    """Load labels from file."""
    with open(file_path, 'r') as f:
        labels = np.array([int(line.strip()) for line in f if line.strip()])
    return labels

def analyze_labels(labels, name="Labels"):
    """Analyze label distribution."""
    unique_labels = np.unique(labels[labels > 0])
    composite_ids = unique_labels // 1000
    unique_composites = np.unique(composite_ids)

    print(f"\n{'='*60}")
    print(f"{name} Analysis:")
    print(f"{'='*60}")
    print(f"Total points: {len(labels)}")
    print(f"Labeled points: {np.sum(labels > 0)} ({100*np.sum(labels > 0)/len(labels):.1f}%)")
    print(f"Background points: {np.sum(labels == 0)} ({100*np.sum(labels == 0)/len(labels):.1f}%)")
    print(f"Unique instance IDs: {len(unique_labels)}")
    print(f"Unique composite IDs: {len(unique_composites)}")

    # Group by composite ID
    composite_instances = defaultdict(list)
    for uid in unique_labels:
        composite_id = uid // 1000
        instance_num = uid % 1000
        composite_instances[composite_id].append(instance_num)

    print(f"\nComposite ID breakdown:")
    for cid in sorted(composite_instances.keys()):
        instances = sorted(composite_instances[cid])
        print(f"  {cid}: {len(instances)} instances -> {instances}")

    return labels, unique_labels, composite_instances

def compute_iou_matrix(gt_labels, pred_labels):
    """Compute IoU between all GT and prediction instances."""
    gt_unique = np.unique(gt_labels[gt_labels > 0])
    pred_unique = np.unique(pred_labels[pred_labels > 0])

    print(f"\n{'='*60}")
    print(f"IoU Matrix Computation:")
    print(f"{'='*60}")
    print(f"GT instances: {len(gt_unique)}")
    print(f"Pred instances: {len(pred_unique)}")

    if len(gt_unique) == 0 or len(pred_unique) == 0:
        print("ERROR: No instances to compare!")
        return None, None, None

    # Compute IoU for each pair
    ious = []
    for gt_id in gt_unique:
        gt_mask = (gt_labels == gt_id)
        gt_composite = gt_id // 1000

        for pred_id in pred_unique:
            pred_mask = (pred_labels == pred_id)
            pred_composite = pred_id // 1000

            # Only compute IoU if composite IDs match
            if gt_composite != pred_composite:
                continue

            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)

            if union > 0:
                iou = intersection / union
                if iou > 0.01:  # Only record non-trivial overlaps
                    ious.append({
                        'gt_id': gt_id,
                        'pred_id': pred_id,
                        'composite': gt_composite,
                        'iou': iou,
                        'intersection': intersection,
                        'gt_size': np.sum(gt_mask),
                        'pred_size': np.sum(pred_mask),
                    })

    if not ious:
        print("❌ No overlapping instances found!")
        return None, None, None

    # Sort by IoU descending
    ious.sort(key=lambda x: x['iou'], reverse=True)

    print(f"\nFound {len(ious)} instance pairs with IoU > 0.01")
    print(f"\nTop 20 matches (sorted by IoU):")
    print(f"{'GT ID':<12} {'Pred ID':<12} {'Composite':<12} {'IoU':<8} {'GT Size':<10} {'Pred Size':<10}")
    print(f"{'-'*80}")

    for match in ious[:20]:
        print(f"{match['gt_id']:<12} {match['pred_id']:<12} {match['composite']:<12} "
              f"{match['iou']:<8.4f} {match['gt_size']:<10} {match['pred_size']:<10}")

    # Statistics
    iou_values = [m['iou'] for m in ious]
    print(f"\nIoU Statistics:")
    print(f"  Max:    {max(iou_values):.4f}")
    print(f"  Mean:   {np.mean(iou_values):.4f}")
    print(f"  Median: {np.median(iou_values):.4f}")
    print(f"  Min:    {min(iou_values):.4f}")

    # Count by threshold
    thresholds = [0.25, 0.5, 0.75]
    print(f"\nMatches above thresholds:")
    for thresh in thresholds:
        count = sum(1 for iou in iou_values if iou >= thresh)
        print(f"  IoU >= {thresh:.2f}: {count}/{len(ious)} ({100*count/len(ious):.1f}%)")

    return ious, gt_unique, pred_unique

def check_evaluation_criteria(ious, gt_labels, pred_labels):
    """Check if predictions meet evaluation criteria."""
    print(f"\n{'='*60}")
    print(f"Evaluation Criteria Check:")
    print(f"{'='*60}")

    # From eval_semantic_instance_parts_OV.py
    min_region_size = 10
    overlap_thresholds = [0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    print(f"Min region size: {min_region_size} points")
    print(f"Overlap thresholds: {overlap_thresholds}")

    if ious is None:
        print("❌ No matches to evaluate!")
        return

    # Check size filtering
    print(f"\n1. Size Filtering (>= {min_region_size} points):")
    for match in ious[:10]:
        gt_ok = match['gt_size'] >= min_region_size
        pred_ok = match['pred_size'] >= min_region_size
        status = "✓" if (gt_ok and pred_ok) else "❌"
        print(f"  {status} GT {match['gt_id']}: {match['gt_size']} pts, "
              f"Pred {match['pred_id']}: {match['pred_size']} pts")

    # Check overlap thresholds
    print(f"\n2. Overlap Threshold Check:")
    for thresh in [0.25, 0.5]:
        matches = [m for m in ious if m['iou'] >= thresh]
        if matches:
            print(f"  IoU >= {thresh}: {len(matches)} matches")
            print(f"    Best: GT {matches[0]['gt_id']} ↔ Pred {matches[0]['pred_id']} "
                  f"(IoU={matches[0]['iou']:.4f})")
        else:
            print(f"  IoU >= {thresh}: ❌ NO MATCHES")

    # Composite ID matching
    print(f"\n3. Composite ID Verification:")
    gt_composites = set((gt_labels[gt_labels > 0] // 1000).tolist())
    pred_composites = set((pred_labels[pred_labels > 0] // 1000).tolist())

    overlap = gt_composites & pred_composites
    print(f"  GT composites: {sorted(gt_composites)}")
    print(f"  Pred composites: {sorted(pred_composites)}")
    print(f"  Overlap: {sorted(overlap)} ({len(overlap)}/{len(gt_composites)} = {100*len(overlap)/max(len(gt_composites),1):.1f}%)")

def main():
    print("="*60)
    print("mAP = 0 Diagnostic Tool")
    print("="*60)

    # Load labels
    print(f"\nLoading GT from: {GT_FILE}")
    print(f"Loading Predictions from: {PRED_FILE}")

    if not GT_FILE.exists():
        print(f"ERROR: GT file not found!")
        return 1

    if not PRED_FILE.exists():
        print(f"ERROR: Prediction file not found!")
        return 1

    gt_labels = load_labels(GT_FILE)
    pred_labels = load_labels(PRED_FILE)

    # Verify same length
    if len(gt_labels) != len(pred_labels):
        print(f"ERROR: Label length mismatch! GT={len(gt_labels)}, Pred={len(pred_labels)}")
        return 1

    # Analyze both
    gt_labels, gt_unique, gt_composites = analyze_labels(gt_labels, "GT")
    pred_labels, pred_unique, pred_composites = analyze_labels(pred_labels, "Predictions")

    # Compute IoU
    ious, _, _ = compute_iou_matrix(gt_labels, pred_labels)

    # Check evaluation criteria
    check_evaluation_criteria(ious, gt_labels, pred_labels)

    # Final verdict
    print(f"\n{'='*60}")
    print("DIAGNOSIS:")
    print(f"{'='*60}")

    if ious is None or len(ious) == 0:
        print("❌ CRITICAL: No overlapping instances found!")
        print("   → Predictions and GT have no spatial overlap")
        print("   → This explains mAP = 0.000")
    else:
        max_iou = max(m['iou'] for m in ious)
        if max_iou < 0.25:
            print(f"❌ CRITICAL: Max IoU ({max_iou:.4f}) < 0.25 (minimum threshold)")
            print("   → No predictions meet the minimum overlap requirement")
            print("   → This explains mAP = 0.000")
        else:
            print(f"✓ Found matches with IoU >= 0.25")
            print(f"  → mAP should NOT be 0.000!")
            print(f"  → Check evaluation script for bugs")

    return 0

if __name__ == '__main__':
    sys.exit(main())
