#!/usr/bin/env python3
"""
Analyze SigLIP similarity rankings vs. ground-truth IoU coverage.

Given an aggregation_output directory (proposal_data_level*.npz) and the
Search3D GT obj_part label file for a scene, this script:

1. Loads all proposals + SigLIP proposal features from the NPZ files
2. Computes IoU between every proposal and every GT instance
3. Encodes combined "part of object" queries with SigLIP
4. Ranks proposals by similarity for each joint (object, part) label
5. Reports how often the high-IoU proposals appear near the top of the
   similarity ranking (helps diagnose retrieval failures)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for extra in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from eval.search3d.data.multiscan_search3d_constants import (
    VALID_JOINT_SEMANTIC_IDS,
    VALID_JOINT_TUPLE_NAMES,
)
from my3dis.inference import SigLIPFeatureExtractor


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def load_proposals(agg_dir: Path, level_tokens: List[str]):
    mask_mats = []
    feat_mats = []
    level_tags: List[str] = []

    for level in level_tokens:
        token = level.strip()
        token = token[1:] if token.lower().startswith("l") else token
        npz_path = agg_dir / f"proposal_data_level{token}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ file: {npz_path}")

        data = np.load(npz_path)
        masks = data["proposal_masks"].astype(bool)
        feats = data["proposal_features"].astype(np.float32)

        mask_mats.append(masks)
        feat_mats.append(feats)
        level_tags.extend([f"L{token}"] * masks.shape[1])

    prop_masks = np.concatenate(mask_mats, axis=1)
    prop_features = np.concatenate(feat_mats, axis=0)
    level_tags = np.array(level_tags)
    prop_features = _normalize(prop_features)

    return prop_masks, prop_features, level_tags, mask_mats


def compute_instance_iou(gt_labels: np.ndarray, prop_masks: np.ndarray):
    """Return IoU matrix between each GT instance and proposal."""
    positive = gt_labels > 0
    unique_instances = np.unique(gt_labels[positive])
    unique_instances.sort()

    gt_instance_masks = gt_labels[:, None] == unique_instances[None, :]
    gt_sizes = gt_instance_masks.sum(axis=0)
    prop_sizes = prop_masks.sum(axis=0)

    inter = gt_instance_masks.T.astype(np.int64) @ prop_masks.astype(np.int64)
    union = gt_sizes[:, None] + prop_sizes[None, :] - inter
    ious = np.where(union > 0, inter / union, 0.0)

    # Map composite IDs to instance indices
    comp_to_indices: Dict[int, List[int]] = {}
    for idx, inst_id in enumerate(unique_instances):
        comp = inst_id // 1000
        comp_to_indices.setdefault(comp, []).append(idx)

    return ious, comp_to_indices


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SigLIP similarity vs. GT IoU for a scene"
    )
    parser.add_argument("--agg-dir", required=True, help="Path to aggregation_output")
    parser.add_argument("--gt-file", required=True, help="Path to *_obj_part_inst.txt")
    parser.add_argument(
        "--levels",
        default="L2,L4,L6",
        help="Comma-separated levels to include (default: L2,L4,L6)",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device for SigLIP (e.g., cuda:0 or cpu)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-K similarity entries to display"
    )
    args = parser.parse_args()

    agg_dir = Path(args.agg_dir)
    gt_file = Path(args.gt_file)
    levels = [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]

    if not agg_dir.exists():
        raise FileNotFoundError(f"Aggregation directory not found: {agg_dir}")
    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    prop_masks, prop_features, level_tags, mask_mats = load_proposals(
        agg_dir, levels
    )
    gt_labels = np.loadtxt(gt_file, dtype=np.int64)
    ious_instances, comp_to_indices = compute_instance_iou(gt_labels, prop_masks)

    composite_ids = sorted(comp_to_indices.keys())
    label_map = dict(zip(VALID_JOINT_SEMANTIC_IDS, VALID_JOINT_TUPLE_NAMES))

    texts = []
    valid_composites = []
    for comp in composite_ids:
        if comp not in label_map:
            continue
        obj, part = label_map[comp]
        texts.append(f"{part.replace('_', ' ')} of {obj.replace('_', ' ')}")
        valid_composites.append(comp)

    extractor = SigLIPFeatureExtractor(
        "google/siglip-so400m-patch14-384", device=args.device
    )
    query_feats = extractor.extract_text_features(texts).cpu().numpy()
    query_feats = _normalize(query_feats)

    similarities = query_feats @ prop_features.T

    print(f"Aggregation dir: {agg_dir}")
    level_counts = ", ".join(
        f"L{lvl.strip().lstrip('L')}: {mat.shape[1]}" for lvl, mat in zip(levels, mask_mats)
    )
    print(f"Total proposals: {prop_masks.shape[1]} ({level_counts})")
    print(
        f"GT composites present: {len(composite_ids)} "
        f"(mapped: {len(valid_composites)})"
    )
    print("")

    records = []
    for qi, comp in enumerate(valid_composites):
        obj, part = label_map[comp]
        inst_indices = comp_to_indices.get(comp, [])
        if not inst_indices:
            continue

        inst_ious = ious_instances[inst_indices, :]
        best_per_prop = inst_ious.max(axis=0)
        best_prop_idx = int(np.argmax(best_per_prop))
        best_iou = float(best_per_prop[best_prop_idx])

        sims = similarities[qi]
        order = np.argsort(-sims)
        best_rank = int(np.where(order == best_prop_idx)[0][0]) + 1 if best_iou > 0 else -1

        top_entries = []
        for rank_pos, prop_idx in enumerate(order[: args.top_k], start=1):
            top_entries.append(
                {
                    "rank": rank_pos,
                    "level": level_tags[prop_idx],
                    "sim": float(sims[prop_idx]),
                    "iou": float(best_per_prop[prop_idx]),
                }
            )

        records.append(
            {
                "composite_id": int(comp),
                "object": obj,
                "part": part,
                "best_iou": best_iou,
                "best_rank": best_rank,
                "top_entries": top_entries,
            }
        )

    with_iou = [r for r in records if r["best_iou"] >= 0.25]
    avg_rank = (
        sum(r["best_rank"] for r in with_iou) / len(with_iou)
        if with_iou
        else float("nan")
    )

    print("Summary:")
    print(
        f"  composites with IoU >= 0.25: {len(with_iou)} / {len(records)} "
        f"({(len(with_iou)/max(len(records),1))*100:.1f}%)"
    )
    print(f"  avg similarity rank of those matches: {avg_rank:.2f}")
    print("")

    for rec in records:
        print("-" * 80)
        print(f"Composite {rec['composite_id']}: {rec['part']} of {rec['object']}")
        print(f"  Best IoU {rec['best_iou']:.3f} (rank {rec['best_rank']})")
        print("  Top-{} similarity proposals:".format(args.top_k))
        for entry in rec["top_entries"]:
            print(
                f"    #{entry['rank']:>2} | {entry['level']} | "
                f"sim={entry['sim']:.4f} | IoU={entry['iou']:.3f}"
            )


if __name__ == "__main__":
    main()
