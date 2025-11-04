"""
Main SigLIP feature assignment pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .feature_extraction import SigLIPFeatureExtractor
from .assignment_utils import aggregate_features, assign_labels

logger = logging.getLogger(__name__)


class SigLIPAssignmentPipeline:
    """
    Main pipeline for assigning SigLIP features to 3D proposals.

    Workflow:
    1. Load 3D proposals from aggregation output
    2. Extract SigLIP features from 2D views
    3. Aggregate features across frames
    4. Assign semantic labels via text queries
    5. Export results
    """

    def __init__(
        self,
        proposal_dir: str,
        color_frames_dir: str,
        depth_maps_dir: str,
        model_name: str = 'google/siglip-so400m-patch14-384',
        device: str = 'cuda:0',
        batch_size: int = 32,
        topk_frames: int = 5,
    ):
        """
        Initialize SigLIP assignment pipeline.

        Args:
            proposal_dir: Directory containing proposal NPZ files
            color_frames_dir: Directory containing color frames
            depth_maps_dir: Directory containing depth maps
            model_name: SigLIP model name
            device: Torch device
            batch_size: Batch size for feature extraction
            topk_frames: Number of best frames to use per proposal
        """
        self.proposal_dir = Path(proposal_dir)
        self.color_frames_dir = Path(color_frames_dir)
        self.depth_maps_dir = Path(depth_maps_dir)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.topk_frames = topk_frames

        # Initialize feature extractor
        self.feature_extractor = SigLIPFeatureExtractor(
            model_name=model_name,
            device=device,
        )

        logger.info(f"Initialized SigLIPAssignmentPipeline")
        logger.info(f"  Proposals: {proposal_dir}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Batch size: {batch_size}")

    def run(
        self,
        output_dir: str,
        levels: Optional[List[int]] = None,
        object_labels: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Run SigLIP feature assignment for all levels.

        Args:
            output_dir: Output directory
            levels: Levels to process (default: [2, 4, 6])
            object_labels: List of object labels for inference (optional)

        Returns:
            Dictionary mapping level -> output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if levels is None:
            levels = [2, 4, 6]

        results = {}

        for level in levels:
            logger.info(f"Processing level {level}...")

            # Load proposals
            proposal_path = self.proposal_dir / f"proposal_data_level{level}.npz"
            if not proposal_path.exists():
                logger.warning(f"  Proposal file not found: {proposal_path}")
                continue

            proposals = self._load_proposals(proposal_path)
            logger.info(f"  Loaded {len(proposals['obj_ids'])} proposals")

            # Extract features
            features = self._extract_features_for_proposals(
                proposals['masks'],
                proposals['points'],
                proposals['colors'],
            )

            logger.info(f"  Extracted features: {features.shape}")

            # Assign labels if provided
            if object_labels is not None:
                labels, scores = self._assign_labels(features, object_labels)
                proposals['pred_classes'] = labels
                proposals['pred_scores'] = scores

            # Export
            output_path = output_dir / f"siglip_features_level{level}.npz"
            self._export_results(proposals, features, output_path)
            results[str(level)] = str(output_path)

            logger.info(f"  Saved to {output_path}")

        # Export metadata
        metadata_path = output_dir / "siglip_metadata.json"
        self._export_metadata(metadata_path, results)

        return results

    def _load_proposals(self, proposal_path: Path) -> Dict:
        """Load proposals from NPZ file."""
        data = np.load(proposal_path, allow_pickle=True)
        return {
            'masks': data['masks'],
            'obj_ids': data['obj_ids'],
            'levels': data['levels'],
            'parent_ids': data['parent_ids'],
            'volumes': data['volumes'],
            'centroids': data['centroids'],
            'bboxes': data['bboxes'],
            'points': data['points'],
            'colors': data['colors'],
        }

    def _extract_features_for_proposals(
        self,
        proposal_masks: np.ndarray,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> np.ndarray:
        """
        Extract SigLIP features for all proposals.

        Args:
            proposal_masks: [N_points, N_proposals] binary masks
            points: [N_points, 3] point coordinates
            colors: [N_points, 3] point colors

        Returns:
            Features [N_proposals, D] where D is feature dimension
        """
        N_proposals = proposal_masks.shape[1]

        # Get color frame paths
        color_paths = sorted(list(self.color_frames_dir.glob("*.jpg")))
        if len(color_paths) == 0:
            color_paths = sorted(list(self.color_frames_dir.glob("*.png")))

        logger.info(f"  Found {len(color_paths)} color frames")

        # Extract features
        features = self.feature_extractor.extract_batch(
            color_paths=[str(p) for p in color_paths],
            proposal_masks=torch.tensor(proposal_masks, dtype=torch.bool),
            points=torch.tensor(points, dtype=torch.float32),
            batch_size=self.batch_size,
            topk=self.topk_frames,
        )

        return features.cpu().numpy()

    def _assign_labels(
        self,
        features: np.ndarray,
        object_labels: List[str],
    ) -> tuple:
        """
        Assign labels to proposals using text queries.

        Args:
            features: [N_proposals, D] proposal features
            object_labels: List of object labels

        Returns:
            Tuple of (predicted classes, scores)
        """
        # Get text features
        text_features = self.feature_extractor.get_text_features(object_labels)

        # Compute similarity
        features_t = torch.tensor(features, device=self.device)
        text_features_t = text_features.to(self.device)

        similarity = features_t @ text_features_t.T  # [N_proposals, N_labels]

        # Get best match
        scores, labels = similarity.max(dim=1)

        return labels.cpu().numpy(), scores.cpu().numpy()

    def _export_results(
        self,
        proposals: Dict,
        features: np.ndarray,
        output_path: Path,
    ) -> None:
        """Export results to NPZ file."""
        export_dict = {
            **proposals,
            'features': features,
        }

        np.savez_compressed(output_path, **export_dict)

    def _export_metadata(self, metadata_path: Path, results: Dict[str, str]) -> None:
        """Export metadata JSON."""
        metadata = {
            'proposal_dir': str(self.proposal_dir),
            'model': self.feature_extractor.model_name,
            'batch_size': self.batch_size,
            'topk_frames': self.topk_frames,
            'outputs': results,
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Assign SigLIP features to 3D proposals')
    parser.add_argument('--proposal-dir', required=True, help='Directory with proposal NPZ files')
    parser.add_argument('--color-frames-dir', required=True, help='Directory with color frames')
    parser.add_argument('--depth-maps-dir', required=True, help='Directory with depth maps')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--levels', nargs='+', type=int, default=[2, 4, 6], help='Levels to process')
    parser.add_argument('--model', default='google/siglip-so400m-patch14-384', help='SigLIP model name')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--topk-frames', type=int, default=5, help='Top-K frames per proposal')
    parser.add_argument('--labels', nargs='+', help='Object labels for inference')
    parser.add_argument('--device', default='cuda:0', help='Device')

    args = parser.parse_args()

    pipeline = SigLIPAssignmentPipeline(
        proposal_dir=args.proposal_dir,
        color_frames_dir=args.color_frames_dir,
        depth_maps_dir=args.depth_maps_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        topk_frames=args.topk_frames,
    )

    results = pipeline.run(
        output_dir=args.output_dir,
        levels=args.levels,
        object_labels=args.labels,
    )

    print(f"SigLIP assignment complete. Results:")
    for level, path in results.items():
        print(f"  Level {level}: {path}")


if __name__ == '__main__':
    main()
