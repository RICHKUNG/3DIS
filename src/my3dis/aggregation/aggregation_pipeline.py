"""
Main aggregation pipeline - orchestrates 2D to 3D projection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from my3dis.family_tree_query import FamilyTreeQuery
from .mask_projection import MaskProjector
from .aggregation_utils import aggregate_masks_3d, compute_proposal_geometry, filter_proposals_by_volume

logger = logging.getLogger(__name__)


class AggregationPipeline:
    """
    Main pipeline for aggregating 2D masks to 3D proposals.

    Workflow:
    1. Load family tree from SAM2 output
    2. Load point cloud and camera parameters
    3. Project 2D masks to 3D for each object
    4. Aggregate across frames
    5. Filter by volume and IoU thresholds
    6. Export 3D proposals
    """

    def __init__(
        self,
        family_tree_path: str,
        pointcloud_path: str,
        depth_maps_dir: str,
        color_frames_dir: str,
        camera_params_path: Optional[str] = None,
        iou_threshold: float = 0.6,
        min_volume: float = 0.001,
        device: str = 'cuda:0',
    ):
        """
        Initialize aggregation pipeline.

        Args:
            family_tree_path: Path to family_tree.json
            pointcloud_path: Path to scene point cloud (.ply or .pth)
            depth_maps_dir: Directory containing depth maps
            color_frames_dir: Directory containing color frames
            camera_params_path: Path to camera parameters JSON (optional)
            iou_threshold: IoU threshold for deduplication
            min_volume: Minimum volume threshold (m^3)
            device: Torch device
        """
        self.family_tree_path = Path(family_tree_path)
        self.pointcloud_path = Path(pointcloud_path)
        self.depth_maps_dir = Path(depth_maps_dir)
        self.color_frames_dir = Path(color_frames_dir)
        self.camera_params_path = Path(camera_params_path) if camera_params_path else None

        # Hyperparameters
        self.iou_threshold = iou_threshold
        self.min_volume = min_volume
        self.device = torch.device(device)

        # Initialize components
        self.family_query = FamilyTreeQuery(str(family_tree_path))
        self.projector = MaskProjector(
            pointcloud_path=str(pointcloud_path),
            depth_maps_dir=str(depth_maps_dir),
            camera_params_path=str(camera_params_path) if camera_params_path else None,
            device=device,
        )

        logger.info(f"Initialized AggregationPipeline")
        logger.info(f"  Family tree: {family_tree_path}")
        logger.info(f"  Point cloud: {pointcloud_path}")
        logger.info(f"  IoU threshold: {iou_threshold}")
        logger.info(f"  Min volume: {min_volume} m^3")

    def run(self, output_dir: str, levels: Optional[List[int]] = None) -> Dict[str, str]:
        """
        Run aggregation pipeline for all levels.

        Args:
            output_dir: Output directory for 3D proposals
            levels: Levels to process (default: [2, 4, 6])

        Returns:
            Dictionary mapping level -> output NPZ path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if levels is None:
            levels = [2, 4, 6]

        results = {}

        for level in levels:
            logger.info(f"Processing level {level}...")

            # Get objects at this level
            obj_ids = self.family_query.search_objects_by_level(level)
            logger.info(f"  Found {len(obj_ids)} objects at level {level}")

            if len(obj_ids) == 0:
                logger.warning(f"  No objects found at level {level}, skipping")
                continue

            # Process each object
            proposals = []
            for obj_id in obj_ids:
                proposal = self._process_object(obj_id)
                if proposal is not None:
                    proposals.append(proposal)

            logger.info(f"  Generated {len(proposals)} proposals before filtering")

            # Filter by volume
            proposals = filter_proposals_by_volume(proposals, self.min_volume)
            logger.info(f"  {len(proposals)} proposals after volume filtering")

            # Deduplicate by IoU
            proposals = self._deduplicate_proposals(proposals, self.iou_threshold)
            logger.info(f"  {len(proposals)} proposals after IoU deduplication")

            # Export
            output_path = output_dir / f"proposal_data_level{level}.npz"
            self._export_proposals(proposals, output_path)
            results[str(level)] = str(output_path)

            logger.info(f"  Saved to {output_path}")

        # Export metadata
        metadata_path = output_dir / "aggregation_metadata.json"
        self._export_metadata(metadata_path, results)

        return results

    def _process_object(self, obj_id: int) -> Optional[Dict]:
        """
        Process single object: project masks and aggregate to 3D.

        Args:
            obj_id: Object ID from family tree

        Returns:
            Proposal dictionary with 3D mask and metadata
        """
        obj_info = self.family_query.get_object_info(obj_id)
        if obj_info is None:
            return None

        frames = obj_info.get('frames', [])
        if len(frames) == 0:
            return None

        # Load masks for all frames
        masks_dict = self.family_query.load_masks_batch(obj_id, frames)
        if len(masks_dict) == 0:
            logger.warning(f"  Object {obj_id}: No masks loaded")
            return None

        # Project each frame's mask to 3D
        projected_masks = []
        for frame_idx, mask_2d in masks_dict.items():
            mask_3d = self.projector.project_mask_to_3d(mask_2d, frame_idx)
            if mask_3d is not None:
                projected_masks.append(mask_3d)

        if len(projected_masks) == 0:
            logger.warning(f"  Object {obj_id}: No valid projections")
            return None

        # Aggregate across frames
        aggregated_mask = aggregate_masks_3d(projected_masks)

        # Compute geometric properties
        geometry = compute_proposal_geometry(
            aggregated_mask,
            self.projector.points,
            self.projector.colors,
        )

        proposal = {
            'obj_id': obj_id,
            'level': obj_info.get('level'),
            'parent_id': obj_info.get('parent_id'),
            'children': obj_info.get('children', []),
            'mask': aggregated_mask,
            'num_points': int(aggregated_mask.sum()),
            'num_frames': len(frames),
            'frames': frames,
            **geometry,
        }

        return proposal

    def _deduplicate_proposals(
        self, proposals: List[Dict], iou_threshold: float
    ) -> List[Dict]:
        """
        Deduplicate proposals based on 3D IoU.

        Args:
            proposals: List of proposal dictionaries
            iou_threshold: IoU threshold

        Returns:
            Deduplicated list of proposals
        """
        if len(proposals) == 0:
            return []

        # Convert masks to tensor for efficient computation
        masks = torch.stack([torch.tensor(p['mask'], dtype=torch.bool) for p in proposals])

        keep = []
        for i in range(len(proposals)):
            if i == 0:
                keep.append(i)
                continue

            # Compute IoU with all kept proposals
            should_keep = True
            for j in keep:
                intersection = (masks[i] & masks[j]).sum()
                union = (masks[i] | masks[j]).sum()
                iou = intersection.float() / union.float() if union > 0 else 0.0

                if iou >= iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep.append(i)

        return [proposals[i] for i in keep]

    def _export_proposals(self, proposals: List[Dict], output_path: Path) -> None:
        """Export proposals to NPZ format."""
        if len(proposals) == 0:
            logger.warning(f"No proposals to export")
            return

        # Stack masks
        masks = np.stack([p['mask'] for p in proposals], axis=1)  # [N_points, N_proposals]

        # Collect metadata
        obj_ids = np.array([p['obj_id'] for p in proposals])
        levels = np.array([p['level'] for p in proposals])
        parent_ids = np.array([p['parent_id'] if p['parent_id'] is not None else -1 for p in proposals])
        volumes = np.array([p['volume'] for p in proposals])
        centroids = np.stack([p['centroid'] for p in proposals])
        bboxes = np.stack([p['bbox'] for p in proposals])

        np.savez_compressed(
            output_path,
            masks=masks,
            obj_ids=obj_ids,
            levels=levels,
            parent_ids=parent_ids,
            volumes=volumes,
            centroids=centroids,
            bboxes=bboxes,
            points=self.projector.points,
            colors=self.projector.colors,
        )

    def _export_metadata(self, metadata_path: Path, results: Dict[str, str]) -> None:
        """Export metadata JSON."""
        metadata = {
            'family_tree': str(self.family_tree_path),
            'pointcloud': str(self.pointcloud_path),
            'iou_threshold': self.iou_threshold,
            'min_volume': self.min_volume,
            'outputs': results,
            'statistics': self.family_query.get_statistics(),
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate 2D masks to 3D proposals')
    parser.add_argument('--family-tree', required=True, help='Path to family_tree.json')
    parser.add_argument('--pointcloud', required=True, help='Path to point cloud file')
    parser.add_argument('--depth-maps-dir', required=True, help='Directory with depth maps')
    parser.add_argument('--color-frames-dir', required=True, help='Directory with color frames')
    parser.add_argument('--camera-params', help='Path to camera parameters JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--levels', nargs='+', type=int, default=[2, 4, 6], help='Levels to process')
    parser.add_argument('--iou-threshold', type=float, default=0.6, help='IoU threshold for deduplication')
    parser.add_argument('--min-volume', type=float, default=0.001, help='Minimum volume (m^3)')
    parser.add_argument('--device', default='cuda:0', help='Device')

    args = parser.parse_args()

    pipeline = AggregationPipeline(
        family_tree_path=args.family_tree,
        pointcloud_path=args.pointcloud,
        depth_maps_dir=args.depth_maps_dir,
        color_frames_dir=args.color_frames_dir,
        camera_params_path=args.camera_params,
        iou_threshold=args.iou_threshold,
        min_volume=args.min_volume,
        device=args.device,
    )

    results = pipeline.run(args.output_dir, levels=args.levels)

    print(f"Aggregation complete. Results:")
    for level, path in results.items():
        print(f"  Level {level}: {path}")


if __name__ == '__main__':
    main()
