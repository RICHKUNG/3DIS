"""
Main inference pipeline for Open Vocabulary 3D Instance Segmentation.

This module ties together all components:
- Data loading (proposals, features, family trees)
- Query processing
- Strategy selection and execution
- Output formatting for Search3D evaluation
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import json

from .retrieval import MultiLevelRetriever, Proposal
from .pairing import ObjectPartPairer, FamilyTree
from .nms import Prediction
from .scoring import ScoreAggregator
from .formatter import Search3DFormatter
from .strategies import (
    InferenceStrategy,
    IndependentStrategy,
    HierarchicalStrategy,
    ExhaustivePairingStrategy,
    ExhaustivePairingWithCombinedQuery
)

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline.

    Usage:
    ```python
    # Initialize
    pipeline = InferencePipeline(
        proposals_by_level={...},
        family_tree_path="relations/family_tree.json",
        strategy="independent"
    )

    # Run inference for a query
    predictions = pipeline.infer_single_query(
        object_query="cabinet",
        part_query="door",
        label_id=121003
    )

    # Format for evaluation
    formatted = pipeline.format_predictions(predictions)
    ```
    """

    def __init__(
        self,
        proposals_by_level: Dict[str, List[Proposal]],
        family_tree_path: Optional[str] = None,
        family_tree: Optional[FamilyTree] = None,
        strategy: str = "independent",
        num_points: Optional[int] = None,
        **strategy_kwargs
    ):
        """
        Initialize inference pipeline.

        Args:
            proposals_by_level: Dict mapping level names to lists of Proposals
            family_tree_path: Path to family_tree.json
            family_tree: Pre-loaded FamilyTree object (alternative to path)
            strategy: Strategy name ('independent' or 'hierarchical')
            num_points: Number of points in scene (for formatting)
            **strategy_kwargs: Additional arguments for strategy
        """
        self.proposals_by_level = proposals_by_level
        self.num_points = num_points

        # Infer num_points if not provided
        if self.num_points is None and len(proposals_by_level) > 0:
            first_level = next(iter(proposals_by_level.values()))
            if len(first_level) > 0:
                self.num_points = len(first_level[0].mask)
                logger.info(f"Inferred num_points={self.num_points} from proposals")

        # Load family tree
        if family_tree is not None:
            self.family_tree = family_tree
        elif family_tree_path is not None:
            self.family_tree = self._load_family_tree(family_tree_path)
        else:
            logger.warning("No family tree provided, tree-based strategies unavailable")
            self.family_tree = None

        # Initialize components
        self.retriever = MultiLevelRetriever(proposals_by_level)
        self.pairer = ObjectPartPairer(family_tree=self.family_tree)
        self.score_aggregator = ScoreAggregator()

        # Initialize strategy
        self.strategy_name = strategy
        self.strategy = self._create_strategy(strategy, **strategy_kwargs)

        # Initialize formatter
        if self.num_points is not None:
            self.formatter = Search3DFormatter(num_points=self.num_points)
        else:
            self.formatter = None
            logger.warning("Formatter not initialized (num_points unknown)")

    def _load_family_tree(self, path: str) -> FamilyTree:
        """Load family tree from JSON file."""
        logger.info(f"Loading family tree from {path}")

        with open(path, 'r') as f:
            tree_data = json.load(f)

        # Convert to FamilyTree format
        # Assuming tree_data has structure like:
        # {"objects": {obj_id: {"parent": ..., "children": [...], ...}}}

        if 'objects' in tree_data:
            tree_dict = {}
            for obj_id_str, obj_info in tree_data['objects'].items():
                obj_id = int(obj_id_str)
                tree_dict[obj_id] = {
                    'parent': obj_info.get('parent'),
                    'children': obj_info.get('children', [])
                }
        else:
            tree_dict = tree_data

        return FamilyTree(tree_dict)

    def _create_strategy(
        self,
        strategy_name: str,
        **kwargs
    ) -> InferenceStrategy:
        """Create inference strategy."""
        if strategy_name == "independent":
            # Log combined query mode if enabled
            use_combined_query = kwargs.get('use_combined_query', True)
            if use_combined_query:
                logger.info("IndependentStrategy: Using combined query mode (default)")
            else:
                logger.info("IndependentStrategy: Using separate query mode")

            return IndependentStrategy(
                retriever=self.retriever,
                pairer=self.pairer,
                score_aggregator=self.score_aggregator,
                family_tree=self.family_tree,
                **kwargs
            )

        elif strategy_name == "hierarchical":
            if self.family_tree is None:
                raise ValueError("Hierarchical strategy requires family tree")

            # Log combined query mode if enabled
            use_combined_query = kwargs.get('use_combined_query', True)
            if use_combined_query:
                logger.info("HierarchicalStrategy: Using combined query mode (default)")
            else:
                logger.info("HierarchicalStrategy: Using separate query mode")

            return HierarchicalStrategy(
                retriever=self.retriever,
                pairer=self.pairer,
                score_aggregator=self.score_aggregator,
                family_tree=self.family_tree,
                **kwargs
            )

        elif strategy_name == "exhaustive_pairing":
            # Check if combined query mode is enabled
            use_combined_query = kwargs.pop('use_combined_query', False)

            if use_combined_query:
                logger.info("Using ExhaustivePairingWithCombinedQuery (combined 'X of Y' mode)")
                # Extract SigLIP-related parameters for combined query strategy
                siglip_model = kwargs.pop('siglip_model', 'google/siglip-so400m-patch14-384')
                siglip_device = kwargs.pop('siglip_device', 'cuda')
                skip_model_load = kwargs.pop('skip_model_load', False)

                return ExhaustivePairingWithCombinedQuery(
                    retriever=self.retriever,
                    pairer=self.pairer,
                    score_aggregator=self.score_aggregator,
                    family_tree=self.family_tree,
                    siglip_model=siglip_model,
                    device=siglip_device,
                    skip_model_load=skip_model_load,
                    **kwargs
                )
            else:
                logger.info("Using ExhaustivePairingStrategy (separate query mode)")
                # Remove SigLIP-related parameters that base strategy doesn't use
                kwargs.pop('siglip_model', None)
                kwargs.pop('siglip_device', None)
                kwargs.pop('skip_model_load', None)

                return ExhaustivePairingStrategy(
                    retriever=self.retriever,
                    pairer=self.pairer,
                    score_aggregator=self.score_aggregator,
                    family_tree=self.family_tree,
                    **kwargs
                )

        elif strategy_name == "exhaustive_pairing_combined":
            # Legacy support: directly specify combined query strategy
            logger.warning(
                "Strategy name 'exhaustive_pairing_combined' is deprecated. "
                "Use 'exhaustive_pairing' with 'use_combined_query: true' instead."
            )
            return ExhaustivePairingWithCombinedQuery(
                retriever=self.retriever,
                pairer=self.pairer,
                score_aggregator=self.score_aggregator,
                family_tree=self.family_tree,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def infer_single_query(
        self,
        object_query_feat: np.ndarray,
        part_query_feat: np.ndarray,
        label_id: int,
        points: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Prediction]:
        """
        Run inference for a single query.

        Args:
            object_query_feat: Object query embedding
            part_query_feat: Part query embedding
            label_id: Joint semantic label ID
            points: Point cloud coordinates
            **kwargs: Additional arguments for strategy

        Returns:
            List of predictions
        """
        # Run strategy
        pairs = self.strategy.infer(
            object_query_feat=object_query_feat,
            part_query_feat=part_query_feat,
            label_id=label_id,
            points=points,
            **kwargs
        )

        # Convert to predictions
        predictions = self.strategy.pairs_to_predictions(pairs)

        return predictions

    def infer_batch_queries(
        self,
        queries: List[Tuple[np.ndarray, np.ndarray, int]],
        points: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Prediction]:
        """
        Run inference for multiple queries.

        Args:
            queries: List of (object_feat, part_feat, label_id) tuples
            points: Point cloud coordinates
            **kwargs: Additional arguments

        Returns:
            Combined list of predictions from all queries
        """
        all_predictions = []

        for i, (obj_feat, part_feat, label_id) in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}, label={label_id}")

            predictions = self.infer_single_query(
                object_query_feat=obj_feat,
                part_query_feat=part_feat,
                label_id=label_id,
                points=points,
                **kwargs
            )

            all_predictions.extend(predictions)

        logger.info(f"Total predictions from {len(queries)} queries: {len(all_predictions)}")

        return all_predictions

    def format_predictions(
        self,
        predictions: List[Prediction]
    ) -> Dict[str, np.ndarray]:
        """
        Format predictions for Search3D evaluation.

        Args:
            predictions: List of predictions

        Returns:
            Dict with 'pred_classes', 'pred_scores', 'pred_masks'
        """
        if self.formatter is None:
            raise ValueError("Formatter not initialized (num_points unknown)")

        return self.formatter.format_predictions(predictions)

    def save_predictions(
        self,
        predictions: List[Prediction],
        output_path: str
    ):
        """
        Save predictions to file.

        Args:
            predictions: List of predictions
            output_path: Output file path
        """
        formatted = self.format_predictions(predictions)

        np.savez(output_path,
                pred_classes=formatted['pred_classes'],
                pred_scores=formatted['pred_scores'],
                pred_masks=formatted['pred_masks'])

        logger.info(f"Saved predictions to {output_path}")


def load_proposals_from_tracking_output(
    tracking_output_dir: str,
    levels: List[str] = ['L2', 'L4', 'L6'],
    feature_dir: Optional[str] = None
) -> Dict[str, List[Proposal]]:
    """
    Load proposals from My3DIS tracking output.

    Args:
        tracking_output_dir: Directory containing tracking output
                            (e.g., outputs/experiments/scene_XX/run_YY)
        levels: List of levels to load
        feature_dir: Directory containing SigLIP features (if separate)

    Returns:
        Dict mapping level names to lists of Proposals
    """
    tracking_output_dir = Path(tracking_output_dir)
    proposals_by_level = {}

    for level in levels:
        level_dir = tracking_output_dir / f"level_{level}"
        tracking_dir = level_dir / "tracking"

        if not tracking_dir.exists():
            logger.warning(f"Tracking directory not found: {tracking_dir}")
            continue

        # Load object_segments NPZ
        object_npz_files = list(tracking_dir.glob("object_segments_*.npz"))

        if len(object_npz_files) == 0:
            logger.warning(f"No object_segments files found in {tracking_dir}")
            continue

        # Use the first (or most recent) file
        object_npz = object_npz_files[0]
        logger.info(f"Loading {level} proposals from {object_npz}")

        # Load data
        data = np.load(object_npz)

        # Extract object IDs and masks
        # Format depends on your tracking output structure
        # This is a placeholder - adjust based on actual format

        proposals = []
        # TODO: Implement actual loading logic based on your output format

        proposals_by_level[level] = proposals

    return proposals_by_level


# Example usage
if __name__ == "__main__":
    # This is a usage example (would be in a separate script)

    # Load proposals
    proposals = load_proposals_from_tracking_output(
        "outputs/experiments/scene_00065_00/run_name",
        levels=['L2', 'L4', 'L6']
    )

    # Initialize pipeline
    pipeline = InferencePipeline(
        proposals_by_level=proposals,
        family_tree_path="outputs/.../relations/family_tree.json",
        strategy="independent"
    )

    # Run inference
    # (would need actual query embeddings)
    predictions = pipeline.infer_single_query(
        object_query_feat=np.random.rand(768),  # Example
        part_query_feat=np.random.rand(768),
        label_id=121003
    )

    # Format and save
    pipeline.save_predictions(predictions, "predictions.npz")
