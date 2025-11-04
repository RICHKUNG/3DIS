"""
Independent Multi-Path Retrieval Strategy.

This is the baseline strategy (Strategy A from the algorithm advice):
1. Independently retrieve object candidates from multiple levels (L2, L4)
2. Independently retrieve part candidates from multiple levels (L4, L6)
3. Pair candidates using geometric constraints
4. Apply multi-instance aware NMS

This strategy is robust and doesn't rely on level assumptions.
"""

from typing import List, Optional, Dict
import numpy as np
import logging

from .base import InferenceStrategy
from ..retrieval import MultiLevelRetriever, Proposal
from ..pairing import ObjectPartPair, ObjectPartPairer
from ..nms import multi_instance_nms

logger = logging.getLogger(__name__)


class IndependentStrategy(InferenceStrategy):
    """
    Independent multi-path retrieval with geometric pairing.

    Advantages:
    - No level assumptions required
    - High recall (searches multiple levels)
    - Simple and robust

    Disadvantages:
    - Higher computational cost (O×P pairings)
    - Doesn't leverage family tree structure
    """

    def __init__(
        self,
        retriever: MultiLevelRetriever,
        pairer: ObjectPartPairer,
        object_levels: Optional[List[str]] = None,
        part_levels: Optional[List[str]] = None,
        top_k_per_level: int = 50,
        retrieval_threshold: float = 0.3,
        **kwargs
    ):
        """
        Initialize independent strategy.

        Args:
            retriever: Multi-level retriever
            pairer: Object-part pairer
            object_levels: Levels to search for objects (default: ['L2', 'L4'])
            part_levels: Levels to search for parts (default: ['L4', 'L6'])
            top_k_per_level: Max proposals to retrieve per level
            retrieval_threshold: Minimum similarity for retrieval
            **kwargs: Additional arguments for base class
        """
        super().__init__(retriever, pairer, **kwargs)

        self.object_levels = object_levels or ['L2', 'L4']
        self.part_levels = part_levels or ['L4', 'L6']
        self.top_k_per_level = top_k_per_level
        self.retrieval_threshold = retrieval_threshold

    def infer(
        self,
        object_query_feat: np.ndarray,
        part_query_feat: np.ndarray,
        label_id: int,
        points: Optional[np.ndarray] = None,
        nms_iou_threshold: float = 0.35,
        use_soft_nms: bool = False,
        keep_top_k: Optional[int] = None,
        **kwargs
    ) -> List[ObjectPartPair]:
        """
        Run independent multi-path retrieval.

        Args:
            object_query_feat: Object query embedding
            part_query_feat: Part query embedding
            label_id: Joint semantic label ID
            points: Point cloud coordinates
            nms_iou_threshold: IoU threshold for NMS
            use_soft_nms: Whether to use soft NMS
            keep_top_k: Max pairs to return
            **kwargs: Additional parameters

        Returns:
            List of object-part pairs after NMS
        """
        # Step 1: Retrieve object candidates from multiple levels
        logger.info(f"Retrieving object candidates from levels: {self.object_levels}")
        object_results = self.retriever.retrieve_multi_level(
            object_query_feat,
            levels=self.object_levels,
            top_k_per_level=self.top_k_per_level,
            threshold=self.retrieval_threshold,
            deduplicate=True,
            iou_threshold=0.95
        )

        logger.info(f"Retrieved {len(object_results.proposals)} object proposals")

        # Step 2: Retrieve part candidates from multiple levels
        logger.info(f"Retrieving part candidates from levels: {self.part_levels}")
        part_results = self.retriever.retrieve_multi_level(
            part_query_feat,
            levels=self.part_levels,
            top_k_per_level=self.top_k_per_level,
            threshold=self.retrieval_threshold,
            deduplicate=True,
            iou_threshold=0.95
        )

        logger.info(f"Retrieved {len(part_results.proposals)} part proposals")

        if len(object_results.proposals) == 0 or len(part_results.proposals) == 0:
            logger.warning("No object or part proposals found")
            return []

        # Step 3: Pair candidates using geometric constraints
        logger.info("Pairing object and part candidates...")
        pairs = self.pairer.pair_all(
            object_proposals=object_results.proposals,
            part_proposals=part_results.proposals,
            label_id=label_id,
            points=points,
            use_tree_pruning=False  # Independent strategy doesn't use tree
        )

        logger.info(f"Created {len(pairs)} valid pairs")

        if len(pairs) == 0:
            return []

        # Step 4: Apply multi-instance aware NMS
        logger.info("Applying NMS to pairs...")
        filtered_pairs = self.apply_nms_to_pairs(
            pairs,
            nms_iou_threshold=nms_iou_threshold,
            points=points,
            use_soft_nms=use_soft_nms,
            keep_top_k=keep_top_k
        )

        logger.info(f"Final result: {len(filtered_pairs)} pairs after NMS")

        return filtered_pairs

    def infer_single_level(
        self,
        object_query_feat: np.ndarray,
        part_query_feat: np.ndarray,
        label_id: int,
        object_level: str,
        part_level: str,
        points: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[ObjectPartPair]:
        """
        Run retrieval on a single level combination.

        Useful for ablation studies or when you know the correct levels.

        Args:
            object_query_feat: Object query embedding
            part_query_feat: Part query embedding
            label_id: Joint semantic label ID
            object_level: Level for object (e.g., 'L2')
            part_level: Level for part (e.g., 'L6')
            points: Point cloud coordinates
            **kwargs: Additional parameters

        Returns:
            List of object-part pairs
        """
        # Retrieve from single levels
        object_results = self.retriever.retrieve_single_level(
            object_query_feat,
            level=object_level,
            top_k=self.top_k_per_level,
            threshold=self.retrieval_threshold
        )

        part_results = self.retriever.retrieve_single_level(
            part_query_feat,
            level=part_level,
            top_k=self.top_k_per_level,
            threshold=self.retrieval_threshold
        )

        if len(object_results.proposals) == 0 or len(part_results.proposals) == 0:
            return []

        # Pair
        pairs = self.pairer.pair_all(
            object_proposals=object_results.proposals,
            part_proposals=part_results.proposals,
            label_id=label_id,
            points=points,
            use_tree_pruning=False
        )

        return pairs
