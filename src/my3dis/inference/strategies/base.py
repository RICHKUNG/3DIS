"""
Base class for inference strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np

from ..retrieval import Proposal, MultiLevelRetriever
from ..pairing import ObjectPartPair, ObjectPartPairer, FamilyTree
from ..nms import Prediction, multi_instance_nms
from ..scoring import ScoreAggregator


class InferenceStrategy(ABC):
    """
    Abstract base class for inference strategies.

    Each strategy implements a different approach to retrieving and pairing
    object-part proposals from multi-level hierarchies.
    """

    def __init__(
        self,
        retriever: MultiLevelRetriever,
        pairer: ObjectPartPairer,
        score_aggregator: Optional[ScoreAggregator] = None,
        family_tree: Optional[FamilyTree] = None
    ):
        """
        Initialize strategy.

        Args:
            retriever: Multi-level retriever
            pairer: Object-part pairer
            score_aggregator: Score aggregation strategy
            family_tree: Family tree for hierarchical relationships
        """
        self.retriever = retriever
        self.pairer = pairer
        self.score_aggregator = score_aggregator or ScoreAggregator()
        self.family_tree = family_tree

    @abstractmethod
    def infer(
        self,
        object_query_feat: np.ndarray,
        part_query_feat: np.ndarray,
        label_id: int,
        points: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[ObjectPartPair]:
        """
        Run inference to retrieve and pair object-part proposals.

        Args:
            object_query_feat: Object query embedding
            part_query_feat: Part query embedding
            label_id: Joint semantic label ID
            points: Point cloud coordinates
            **kwargs: Strategy-specific parameters

        Returns:
            List of object-part pairs
        """
        pass

    def apply_nms_to_pairs(
        self,
        pairs: List[ObjectPartPair],
        nms_iou_threshold: float = 0.35,
        points: Optional[np.ndarray] = None,
        use_soft_nms: bool = False,
        keep_top_k: Optional[int] = None
    ) -> List[ObjectPartPair]:
        """
        Apply NMS to object-part pairs.

        Args:
            pairs: List of object-part pairs
            nms_iou_threshold: IoU threshold for NMS
            points: Point cloud coordinates
            use_soft_nms: Whether to use soft NMS
            keep_top_k: Maximum number of pairs to keep

        Returns:
            Filtered pairs
        """
        if len(pairs) == 0:
            return []

        # Convert pairs to predictions (using union of object and part masks)
        predictions = []
        for pair in pairs:
            # Create joint mask (union of object and part)
            joint_mask = np.logical_or(
                pair.object_proposal.mask,
                pair.part_proposal.mask
            ).astype(np.uint8)

            pred = Prediction(
                mask=joint_mask,
                score=pair.combined_score,
                label_id=pair.label_id,
                metadata={'pair': pair}
            )
            predictions.append(pred)

        # Apply NMS
        kept_predictions = multi_instance_nms(
            predictions,
            iou_threshold=nms_iou_threshold,
            points=points,
            use_soft_nms=use_soft_nms
        )

        # Extract pairs
        kept_pairs = [pred.metadata['pair'] for pred in kept_predictions]

        # Apply top-k
        if keep_top_k is not None and len(kept_pairs) > keep_top_k:
            kept_pairs = sorted(kept_pairs, key=lambda p: p.combined_score, reverse=True)
            kept_pairs = kept_pairs[:keep_top_k]

        return kept_pairs

    def pairs_to_predictions(
        self,
        pairs: List[ObjectPartPair],
        use_part_mask: bool = True
    ) -> List[Prediction]:
        """
        Convert object-part pairs to predictions.

        Args:
            pairs: List of pairs
            use_part_mask: If True, use part mask; if False, use object mask;
                          if 'both', use union

        Returns:
            List of predictions
        """
        predictions = []

        for pair in pairs:
            if use_part_mask:
                mask = pair.part_proposal.mask
            elif use_part_mask == 'both':
                mask = np.logical_or(
                    pair.object_proposal.mask,
                    pair.part_proposal.mask
                ).astype(np.uint8)
            else:
                mask = pair.object_proposal.mask

            pred = Prediction(
                mask=mask,
                score=pair.combined_score,
                label_id=pair.label_id,
                metadata={
                    'object_id': pair.object_proposal.object_id,
                    'part_id': pair.part_proposal.object_id,
                    'object_score': pair.object_score,
                    'part_score': pair.part_score,
                    'geometric_score': pair.geometric_score
                }
            )
            predictions.append(pred)

        return predictions
