"""
Aggregation pipeline - integrates pipeline.ipynb logic for 2D to 3D projection.

This module directly uses the validated approach from pipeline.ipynb:
1. WORLD_2_CAM for camera projection and mesh projection to 2D
2. get_visible_points for visibility computation
3. build_proposal_masks_from_masklets for 2D→3D mask conversion
4. merge_drop_level* functions for proposal filtering
5. generate_grounding_features_siglip for SigLIP feature extraction
6. pool_point_to_proposal_features for feature pooling
"""

import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import torch
import torch.nn.functional as F

# Add 3DprojToSiglip to path
proj_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(proj_root / "3DprojToSiglip"))

# Import from 3DprojToSiglip
from utils_load.common_utils import unpack_binary_mask
from utils_load.compat import load_legacy_video_segments
from utils_new import WORLD_2_CAM
from utils_new.utils_2d import get_visible_points

# Import local helper functions
from .pipeline_functions import (
    build_proposal_masks_from_masklets,
    merge_drop_level2_mask,
    merge_drop_level4_within_siblings,
    merge_drop_level6_within_siblings,
    pool_point_to_proposal_features,
    split_merged_vs_small,
    update_relations_level2_inplace_v2,
)

logger = logging.getLogger(__name__)


def detect_available_levels(exp_dir: Path) -> List[int]:
    """
    Auto-detect available levels from experiment directory structure.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Sorted list of available levels (e.g., [1, 3, 5] or [2, 4, 6])
    """
    if not exp_dir.exists():
        logger.warning(f"Experiment directory does not exist: {exp_dir}")
        return []

    levels = []
    level_pattern = re.compile(r'^level_(\d+)$')

    for item in exp_dir.iterdir():
        if item.is_dir():
            match = level_pattern.match(item.name)
            if match:
                level_num = int(match.group(1))
                # Check if video_segments file exists
                video_segments_path = item / f"video_segments_L{level_num:02d}.npz"
                if video_segments_path.exists():
                    levels.append(level_num)
                else:
                    logger.debug(f"Level directory found but video_segments missing: {item}")

    levels.sort()
    return levels


def _is_family_tree_format(data: Dict) -> bool:
    """Return True if data looks like the new family_tree.json structure."""
    return isinstance(data, dict) and isinstance(data.get("objects"), dict)


def _family_tree_to_relations(tree: Dict) -> Dict:
    """Convert family_tree objects map into hierarchy-style relations."""
    hierarchy: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

    for level in tree.get("levels", []):
        hierarchy[str(level)] = {}

    objects = tree.get("objects", {})
    for obj_id_str, obj_data in objects.items():
        level = obj_data.get("level")
        if level is None:
            continue

        level_key = str(level)
        node = {
            "parent": obj_data.get("parent_id"),
            "children": [int(c) for c in obj_data.get("children", [])],
            "descendant_count": len(obj_data.get("children", [])),
        }
        hierarchy.setdefault(level_key, {})
        hierarchy[level_key][str(obj_id_str)] = node

    return {"hierarchy": hierarchy}


def _relations_to_family_tree(relations: Dict, template: Optional[Dict]) -> Dict:
    """
    Convert hierarchy relations back to the family_tree.json structure,
    preserving metadata from template when possible.
    """
    tree = deepcopy(template) if template is not None else {}
    objects_template = tree.get("objects", {}) if isinstance(tree.get("objects"), dict) else {}

    hierarchy = relations.get("hierarchy", {})
    new_objects: Dict[str, Dict] = {}

    for level_key, nodes in hierarchy.items():
        for obj_id_str, node in nodes.items():
            key = str(obj_id_str)
            obj_data = deepcopy(objects_template.get(key, {}))
            if not obj_data:
                obj_data = {
                    "level": int(level_key) if str(level_key).isdigit() else None,
                    "parent_id": None,
                    "children": [],
                    "virtual_children": [],
                    "frames": [],
                    "mask_archive": None,
                    "first_frame": None,
                    "last_frame": None,
                    "total_frames": 0,
                }

            if str(level_key).isdigit():
                obj_data["level"] = int(level_key)
            obj_data["parent_id"] = node.get("parent")
            obj_data["children"] = sorted(int(c) for c in node.get("children", []))
            new_objects[key] = obj_data

    tree["objects"] = new_objects

    if hierarchy:
        level_set = {int(k) for k in hierarchy.keys() if str(k).isdigit()}
        if level_set:
            tree["levels"] = sorted(level_set)

    return tree


def _prepare_relations_data(raw_data: Dict) -> Tuple[Dict, str, Optional[Dict]]:
    """
    Normalize relations input for aggregation.

    Returns:
        normalized_relations, format_tag, family_tree_template
    """
    if _is_family_tree_format(raw_data):
        return _family_tree_to_relations(raw_data), "family_tree", raw_data

    if "hierarchy" in raw_data:
        return raw_data, "hierarchy", None

    # Default empty structure
    return {"hierarchy": {"2": {}, "4": {}, "6": {}}}, "hierarchy", None


class AggregationPipeline:
    """
    Main pipeline for aggregating 2D masks to 3D proposals.

    Integrates the complete 2D→3D projection workflow from pipeline.ipynb.

    Workflow:
    1. Initialize WORLD_2_CAM and compute mesh projections
    2. Get visible points per frame
    3. Load 2D masks from My3DIS tracking outputs
    4. Build 3D proposal masks from 2D masks
    5. Merge and filter proposals at each level
    6. Extract SigLIP features (optional)
    7. Pool features to proposals
    8. Export NPZ files
    """

    def __init__(
        self,
        exp_dir: str,
        scene_path: str,
        iou_threshold: float = 0.8,
        area_fraction: float = 1/500,  # P / 500 in notebook
        device: str = 'cuda',
        config: Optional[Dict] = None,
        pooling_mode: str = 'average',
        area_fraction_per_level: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize aggregation pipeline.

        Args:
            exp_dir: My3DIS experiment directory (scene-specific)
            scene_path: MultiScan scene directory
            iou_threshold: IoU threshold for merging proposals
            area_fraction: Minimum area as fraction of total points (default: 1/500)
                Used as fallback if area_fraction_per_level is not specified
            device: Torch device
            config: Configuration dict (defaults from pipeline.ipynb if None)
            pooling_mode: Feature pooling mode - 'average' or 'voting' (default: 'average')
                - 'average': Average point features, then L2 normalize
                - 'voting': Average point features, NO L2 normalization (preserves magnitude)
                  Matches utils_ov_inference.py behavior (line 54-55)
            area_fraction_per_level: Per-level area fractions (e.g., {2: 0.002, 4: 0.001, 6: 0.0005})
                If None, uses area_fraction for all levels
        """
        self.exp_dir = Path(exp_dir)
        self.scene_path = Path(scene_path)
        self.device = torch.device(device)

        # Hyperparameters
        self.iou_threshold = iou_threshold
        self.area_fraction = area_fraction
        self.area_fraction_per_level = area_fraction_per_level or {}
        self.pooling_mode = pooling_mode

        # Configuration (from pipeline.ipynb)
        if config is None:
            config = {
                "uni3dis": {"vis_depth_threshold": 0.1, "frequency": 50},
                "network2d": {"masklet": {"resize_ratio": 0.3}}
            }
        self.config = config

        self.scaling_params = [1/config["network2d"]["masklet"]["resize_ratio"]] * 2
        self.vis_depth_threshold = config["uni3dis"]["vis_depth_threshold"]
        self.frequency = config["uni3dis"]["frequency"]
        self.depth_scale = 1000.0

        # Find PLY file
        ply_file = self.scene_path / f"{self.scene_path.name}.ply"
        if not ply_file.exists():
            ply_file = self.scene_path / f"{self.scene_path.name}_converted.ply"
        if not ply_file.exists():
            raise FileNotFoundError(f"PLY file not found in {self.scene_path}")
        self.ply_file = ply_file

        logger.info(f"Initialized AggregationPipeline")
        logger.info(f"  Experiment: {exp_dir}")
        logger.info(f"  Scene: {scene_path}")
        logger.info(f"  PLY: {ply_file}")
        logger.info(f"  IoU threshold: {iou_threshold}")
        logger.info(f"  Area fraction: {area_fraction}")
        logger.info(f"  Pooling mode: {pooling_mode}")

        # Will be initialized in run()
        self.world2cam = None
        self.projected_points = None
        self.inside_mask = None
        self.point_depth = None
        self.depth_maps = None
        self.pts_coords_per_frame = None
        self.pts_idx_per_frame = None
        self.P = None  # Number of 3D points

    def _init_world2cam(self):
        """Initialize WORLD_2_CAM and compute projections."""
        logger.info("Initializing WORLD_2_CAM for 3D projection...")

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.world2cam = WORLD_2_CAM(
            dataset_type="multiscan",
            path_2_scene=str(self.scene_path),
            path_2_ply=str(self.ply_file),
            path_2_sp="",  # Not used
            path_2_data=str(self.exp_dir / "temp"),
            depth_scale=self.depth_scale,
            uni3dis_config=self.config,
            device=device_str
        )

        # Get mesh projections
        logger.info("Computing mesh projections to 2D...")
        self.projected_points, self.inside_mask, self.point_depth, self.depth_maps = \
            self.world2cam.get_mesh_projections()

        F, P = self.inside_mask.shape
        self.P = P
        logger.info(f"  Frames: {F}, 3D Points: {P}")

        # Get visible points for all frames
        logger.info("Computing visible points per frame...")
        point_masks_all_true = torch.ones(P, dtype=torch.bool)

        self.pts_coords_per_frame, points_inside_mask_num, self.pts_idx_per_frame = get_visible_points(
            color_paths=self.world2cam.color_paths,
            inside_mask=self.inside_mask,
            projected_points=self.projected_points,
            point_depth=self.point_depth,
            point_masks=point_masks_all_true,
            depth_maps=self.depth_maps,
            device=torch.device('cpu'),
            scaling_params=self.scaling_params,
            vis_depth_threshold=self.vis_depth_threshold
        )

        del point_masks_all_true
        logger.info("✓ WORLD_2_CAM initialization complete")

    def run(
        self,
        output_dir: str,
        levels: Optional[List[int]] = None,
        extract_features: bool = False,
        siglip_model = None,
        siglip_processor = None,
    ) -> Dict[str, str]:
        """
        Run aggregation pipeline for all levels.

        Args:
            output_dir: Output directory for 3D proposals
            levels: Levels to process. If None, will auto-detect from exp_dir.
                   If auto-detection finds no levels, defaults to [2, 4, 6].
            extract_features: Whether to extract SigLIP features
            siglip_model: SigLIP model (required if extract_features=True)
            siglip_processor: SigLIP processor (required if extract_features=True)

        Returns:
            Dictionary mapping level -> output NPZ path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if levels is None:
            # Auto-detect levels from experiment directory
            detected_levels = detect_available_levels(self.exp_dir)
            if detected_levels:
                levels = detected_levels
                logger.info(f"Auto-detected levels from experiment directory: {levels}")
            else:
                levels = [2, 4, 6]
                logger.warning(f"No levels detected, using default: {levels}")
        else:
            logger.info(f"Using specified levels: {levels}")

        # Initialize WORLD_2_CAM
        if self.world2cam is None:
            self._init_world2cam()

        # Compute area thresholds (per-level or global)
        area_thresh_by_level = {}
        global_area_thresh = None
        if self.area_fraction_per_level:
            logger.info("Using per-level area thresholds:")
            for level in levels:
                frac = self.area_fraction_per_level.get(level, self.area_fraction)
                area_thresh_by_level[level] = int(self.P * frac)
                logger.info(f"  Level {level}: {area_thresh_by_level[level]} points ({frac:.4f} * {self.P})")
        else:
            global_area_thresh = int(self.P * self.area_fraction)
            logger.info(f"Using global area threshold: {global_area_thresh} points ({self.area_fraction:.4f} * {self.P})")
            for level in levels:
                area_thresh_by_level[level] = global_area_thresh

        # Load family tree for relations
        logger.info("Loading family tree for relations...")
        family_tree_path = self.exp_dir / "relations" / "family_tree.json"
        if not family_tree_path.exists():
            logger.warning(f"Family tree not found at {family_tree_path}, creating empty relations")
            relations = {"hierarchy": {"2": {}, "4": {}, "6": {}}}
            relations_format = "hierarchy"
            family_tree_template = None
        else:
            with open(family_tree_path, 'r') as f:
                raw_relations = json.load(f)
            relations, relations_format, family_tree_template = _prepare_relations_data(raw_relations)
            if relations_format == "family_tree":
                logger.info("  Detected family_tree structure; converted to hierarchy for aggregation")

        # Process each level to build 3D masks
        logger.info("\n" + "="*60)
        logger.info("BUILDING 3D PROPOSAL MASKS")
        logger.info("="*60)

        masks_by_level = {}
        ids_by_level = {}

        for level in levels:
            logger.info(f"\nProcessing level {level}...")

            # Load video segments
            video_segments_path = self.exp_dir / f"level_{level}" / f"video_segments_L{level:02d}.npz"
            if not video_segments_path.exists():
                logger.warning(f"Video segments not found: {video_segments_path}")
                continue

            # Build 3D proposal masks
            proposal_masks, obj_ids = build_proposal_masks_from_masklets(
                masklet_path=str(video_segments_path),
                pts_coords_per_frame=self.pts_coords_per_frame,
                pts_idx_per_frame=self.pts_idx_per_frame,
                P=self.P,
                frequency=self.frequency,
            )

            logger.info(f"  Built {proposal_masks.shape[1]} 3D proposals from 2D masks")
            masks_by_level[level] = proposal_masks
            ids_by_level[level] = obj_ids

        # Merge and filter proposals
        logger.info("\n" + "="*60)
        logger.info("MERGING AND FILTERING PROPOSALS")
        logger.info("="*60)

        results = {}

        # Level 2
        if 2 in levels and 2 in masks_by_level:
            logger.info("\nLevel 2: Merging and filtering...")
            new_mask_l2, col_ids, old2rep, dropped_ids, groups = merge_drop_level2_mask(
                masks_by_level[2],
                self.iou_threshold,
                area_thresh_by_level[2],
                proposal_ids=ids_by_level.get(2),
            )

            merged_away_ids, dropped_small_ids = split_merged_vs_small(dropped_ids, groups)
            dropped_children_l2 = update_relations_level2_inplace_v2(
                relations, groups, dropped_small_ids
            )

            logger.info(f"  {new_mask_l2.shape[1]} proposals after merge/filter")
            logger.info(f"  Dropped: {len(dropped_ids)} (merged: {len(merged_away_ids)}, small: {len(dropped_small_ids)})")

            masks_by_level[2] = new_mask_l2
            ids_by_level[2] = col_ids

        # Level 4
        if 4 in levels and 4 in masks_by_level:
            logger.info("\nLevel 4: Merging within siblings...")
            new_mask_l4, surv4, map4, drop4_small, groups4 = merge_drop_level4_within_siblings(
                masks_by_level[4],
                relations,
                self.iou_threshold,
                area_thresh_by_level[4],
                proposal_ids=ids_by_level.get(4),
            )

            logger.info(f"  {new_mask_l4.shape[1]} proposals after merge/filter")
            logger.info(f"  Dropped (small): {len(drop4_small)}")

            masks_by_level[4] = new_mask_l4
            ids_by_level[4] = surv4

        # Level 6
        if 6 in levels and 6 in masks_by_level:
            logger.info("\nLevel 6: Merging within siblings...")
            new_mask_l6, surv6, map6, drop6_small, groups6 = merge_drop_level6_within_siblings(
                masks_by_level[6],
                relations,
                self.iou_threshold,
                area_thresh_by_level[6],
                proposal_ids=ids_by_level.get(6),
            )

            logger.info(f"  {new_mask_l6.shape[1]} proposals after merge/filter")
            logger.info(f"  Dropped (small): {len(drop6_small)}")

            masks_by_level[6] = new_mask_l6
            ids_by_level[6] = surv6

        # Extract features if requested
        features_by_level = {}
        if extract_features:
            if siglip_model is None or siglip_processor is None:
                logger.warning("SigLIP model/processor not provided, skipping feature extraction")
            else:
                logger.info("\n" + "="*60)
                logger.info("EXTRACTING SIGLIP FEATURES")
                logger.info("="*60)

                # Import FIXED feature extraction function (2025-11-16: bug fixes for proper averaging)
                # See: SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md for details
                from utils_new.feature_extraction_fixed import generate_grounding_features_siglip_batched

                for level in levels:
                    if level not in masks_by_level:
                        continue

                    logger.info(f"\nLevel {level}: Extracting features...")
                    pc_features = generate_grounding_features_siglip_batched(
                        model=siglip_model,
                        processor=siglip_processor,
                        color_paths=self.world2cam.color_paths,
                        inside_mask=self.inside_mask,
                        projected_points=self.projected_points,
                        points_depth=self.point_depth,
                        depth_maps=self.depth_maps,
                        device=self.device,
                        scaling_params=self.scaling_params,
                        vis_depth_threshold=self.vis_depth_threshold,
                        proposal_masks=masks_by_level[level],
                        get_visible_points_fn=get_visible_points,
                        topk=1,
                        batch_size=32,
                        use_sparse=True,
                    )

                    # Pool to proposals
                    proposal_features = pool_point_to_proposal_features(
                        pc_features, masks_by_level[level], mode=self.pooling_mode
                    )

                    logger.info(f"  Features shape: {proposal_features.shape}")
                    logger.info(f"  Pooling mode: {self.pooling_mode}")
                    features_by_level[level] = proposal_features
        else:
            logger.info("\nSkipping feature extraction (extract_features=False)")
            # Create dummy features
            for level in levels:
                if level in masks_by_level:
                    N = masks_by_level[level].shape[1]
                    features_by_level[level] = torch.zeros((N, 1152), dtype=torch.float32)

        # Export NPZ files
        logger.info("\n" + "="*60)
        logger.info("EXPORTING NPZ FILES")
        logger.info("="*60)

        for level in levels:
            if level not in masks_by_level:
                continue

            output_path = output_dir / f"proposal_data_level{level}.npz"

            proposal_masks_np = masks_by_level[level].cpu().numpy().astype(np.bool_)
            proposal_features_np = features_by_level[level].cpu().numpy().astype(np.float32)
            proposal_ids_np = np.array(ids_by_level[level], dtype=np.int32)

            # Verify shapes match
            assert proposal_masks_np.shape[1] == len(proposal_ids_np), \
                f"Level {level}: Mask columns ({proposal_masks_np.shape[1]}) != ID count ({len(proposal_ids_np)})"

            np.savez_compressed(
                output_path,
                proposal_masks=proposal_masks_np,
                proposal_features=proposal_features_np,
                proposal_ids=proposal_ids_np
            )

            logger.info(f"  Level {level}: {output_path}")
            logger.info(f"    Masks: {proposal_masks_np.shape}")
            logger.info(f"    Features: {proposal_features_np.shape}")
            logger.info(f"    IDs: {len(proposal_ids_np)}")

            results[str(level)] = str(output_path)

        # Save updated relations
        relations_output = output_dir / "Proposal_relation.json"
        relations_to_save = relations
        if relations_format == "family_tree":
            relations_to_save = _relations_to_family_tree(relations, family_tree_template)

        with open(relations_output, 'w') as f:
            json.dump(relations_to_save, f, indent=2)
        logger.info(f"\n  Relations: {relations_output}")

        # Export metadata
        metadata_path = output_dir / "aggregation_metadata.json"
        metadata = {
            'exp_dir': str(self.exp_dir),
            'scene_path': str(self.scene_path),
            'ply_file': str(self.ply_file),
            'iou_threshold': self.iou_threshold,
            'area_fraction': self.area_fraction,
            'area_threshold': area_thresh_by_level if self.area_fraction_per_level else global_area_thresh,
            'num_points': self.P,
            'outputs': results,
            'feature_extraction': extract_features,
            'pooling_mode': self.pooling_mode,
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"  Metadata: {metadata_path}")

        logger.info("\n✓ Aggregation complete")
        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Aggregate 2D masks to 3D proposals (pipeline.ipynb logic)'
    )
    parser.add_argument('--exp-dir', required=True,
                        help='My3DIS experiment directory (scene-specific)')
    parser.add_argument('--scene-path', required=True,
                        help='MultiScan scene directory')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory')
    parser.add_argument('--levels', nargs='+', type=int, default=[2, 4, 6],
                        help='Levels to process')
    parser.add_argument('--iou-threshold', type=float, default=0.8,
                        help='IoU threshold for merging')
    parser.add_argument('--area-fraction', type=float, default=1/500,
                        help='Minimum area as fraction of total points')
    parser.add_argument('--device', default='cuda',
                        help='Device')
    parser.add_argument('--extract-features', action='store_true',
                        help='Extract SigLIP features')
    parser.add_argument('--siglip-model-dir',
                        help='Path to SigLIP model directory')
    parser.add_argument('--pooling-mode', default='average',
                        choices=['average', 'voting'],
                        help='Feature pooling mode: "average" (default) or "voting"')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AggregationPipeline(
        exp_dir=args.exp_dir,
        scene_path=args.scene_path,
        iou_threshold=args.iou_threshold,
        area_fraction=args.area_fraction,
        device=args.device,
        pooling_mode=args.pooling_mode,
    )

    # Load SigLIP if needed
    siglip_model = None
    siglip_processor = None
    if args.extract_features:
        if args.siglip_model_dir is None:
            logger.error("--siglip-model-dir required when --extract-features is set")
            sys.exit(1)

        logger.info(f"Loading SigLIP model from {args.siglip_model_dir}")
        from transformers import AutoModel, AutoProcessor
        siglip_processor = AutoProcessor.from_pretrained(
            args.siglip_model_dir, local_files_only=True, use_fast=False
        )
        siglip_model = AutoModel.from_pretrained(
            args.siglip_model_dir, local_files_only=True
        ).to(args.device).eval()

    # Run pipeline
    results = pipeline.run(
        args.output_dir,
        levels=args.levels,
        extract_features=args.extract_features,
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
    )

    print(f"\nAggregation complete. Results:")
    for level, path in results.items():
        print(f"  Level {level}: {path}")


if __name__ == '__main__':
    main()
