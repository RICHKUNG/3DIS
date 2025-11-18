"""
Independent Multi-Path Retrieval Strategy.

This is the baseline strategy (Strategy A from the algorithm advice):
1. Independently retrieve object candidates from multiple levels (L2, L4)
2. Independently retrieve part candidates from multiple levels (L4, L6)
3. Pair candidates using geometric constraints
4. Apply multi-instance aware NMS

This strategy is robust and doesn't rely on level assumptions.

NOW SUPPORTS COMBINED QUERY MODE (DEFAULT):
- Use "part of object" combined queries for improved semantic matching
- Example: "door of cabinet" instead of separate "door" and "cabinet"
"""

from typing import List, Optional, Dict
import numpy as np
import logging

from .base import InferenceStrategy
from .combined_query_mixin import CombinedQueryMixin
from ..retrieval import MultiLevelRetriever, Proposal, RetrievalResult
from ..pairing import ObjectPartPair, ObjectPartPairer
from ..nms import multi_instance_nms

logger = logging.getLogger(__name__)


class IndependentStrategy(CombinedQueryMixin, InferenceStrategy):
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
        min_retrieval_threshold: float = 0.05,
        threshold_backoff_step: float = 0.05,
        fallback_to_topk: bool = True,
        use_combined_query: bool = True,
        siglip_model: str = "google/siglip-so400m-patch14-384",
        siglip_device: str = "cuda",
        feature_extractor = None,
        scale_semantic_score: float = 100.0,
        apply_softmax: bool = True,
        text_template: str = "in_room",
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
            use_combined_query: Use "part of object" combined queries (DEFAULT: True)
            siglip_model: SigLIP model for combined query features
            siglip_device: Device for SigLIP (cuda:0, cuda:1, or cpu)
            feature_extractor: Pre-initialized feature extractor (optional)
            **kwargs: Additional arguments for base class
        """
        super().__init__(retriever, pairer, **kwargs)

        self.object_levels = object_levels or ['L2', 'L4']
        self.part_levels = part_levels or ['L4', 'L6']
        self.top_k_per_level = top_k_per_level
        self.retrieval_threshold = retrieval_threshold
        self.min_retrieval_threshold = max(0.0, min_retrieval_threshold)
        self.threshold_backoff_step = max(0.0, threshold_backoff_step)
        self.fallback_to_topk = fallback_to_topk

        # Combined-query settings / scaling
        self.scale_semantic_score = scale_semantic_score
        self.apply_softmax = apply_softmax

        # Setup combined query if enabled
        self.use_combined_query = use_combined_query
        if use_combined_query:
            logger.info("IndependentStrategy: Enabling combined query mode")
            self.setup_combined_query(
                siglip_model=siglip_model,
                device=siglip_device,
                feature_extractor=feature_extractor,
                scale_semantic_score=scale_semantic_score,
                apply_softmax=apply_softmax,
                text_template=text_template,
            )
        else:
            logger.info("IndependentStrategy: Using separate query mode")

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
        object_results = self._retrieve_with_backoff(
            query_feat=object_query_feat,
            levels=self.object_levels,
            target_name="object"
        )

        logger.info(f"Retrieved {len(object_results.proposals)} object proposals")

        # Step 2: Retrieve part candidates from multiple levels
        logger.info(f"Retrieving part candidates from levels: {self.part_levels}")
        part_results = self._retrieve_with_backoff(
            query_feat=part_query_feat,
            levels=self.part_levels,
            target_name="part"
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

    def _retrieve_with_backoff(
        self,
        query_feat: np.ndarray,
        levels: List[str],
        target_name: str
    ) -> RetrievalResult:
        """
        Try progressively lower thresholds (and optional top-k fallback) so we
        never exit early with zero proposals just because the similarity scores
        are low for a given query.
        """
        if not self._has_level_candidates(levels):
            logger.warning(
                f"No proposals available for {target_name} levels {levels}; retrieval will be empty"
            )

            return self.retriever.retrieve_multi_level(
                query_feat,
                levels=levels,
                top_k_per_level=self.top_k_per_level,
                threshold=self.retrieval_threshold,
                deduplicate=True,
                iou_threshold=0.95
            )

        base_threshold = self.retrieval_threshold
        last_result: Optional[RetrievalResult] = None

        for threshold in self._build_threshold_schedule():
            result = self.retriever.retrieve_multi_level(
                query_feat,
                levels=levels,
                top_k_per_level=self.top_k_per_level,
                threshold=threshold,
                deduplicate=True,
                iou_threshold=0.95
            )
            last_result = result

            if len(result.proposals) > 0:
                if base_threshold is not None and threshold < base_threshold:
                    logger.info(
                        f"{target_name.title()} retrieval fallback: lowered similarity "
                        f"threshold to {threshold:.2f} (found {len(result.proposals)} proposals)"
                    )
                return result

        if self.fallback_to_topk:
            logger.warning(
                f"{target_name.title()} retrieval fallback: no proposals above "
                f"{self.min_retrieval_threshold:.2f}; forcing top-k results"
            )
            return self.retriever.retrieve_multi_level(
                query_feat,
                levels=levels,
                top_k_per_level=self.top_k_per_level,
                threshold=-1.0,
                deduplicate=True,
                iou_threshold=0.95
            )

        return last_result if last_result is not None else RetrievalResult(
            query="",
            proposals=[],
            scores=np.array([])
        )

    def _build_threshold_schedule(self) -> List[float]:
        """Generate a descending list of thresholds for adaptive retrieval."""
        thresholds: List[float] = []

        if self.retrieval_threshold is not None:
            thresholds.append(float(self.retrieval_threshold))
            if self.threshold_backoff_step > 0 and self.min_retrieval_threshold is not None:
                current = self.retrieval_threshold - self.threshold_backoff_step
                while current > self.min_retrieval_threshold:
                    thresholds.append(float(current))
                    current -= self.threshold_backoff_step

        retriever_min = getattr(self.retriever, "min_similarity", None)
        if retriever_min is not None:
            thresholds.append(float(retriever_min))

        if self.min_retrieval_threshold is not None:
            thresholds.append(float(self.min_retrieval_threshold))

        if not thresholds:
            thresholds.append(0.0)

        # Deduplicate while preserving descending order
        thresholds = sorted(set(round(t, 6) for t in thresholds), reverse=True)
        return thresholds

    def _has_level_candidates(self, levels: List[str]) -> bool:
        """Check if any proposals exist for the requested levels."""
        for level in levels:
            if len(self.retriever.proposals_by_level.get(level, [])) > 0:
                return True
        return False

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

    def infer_with_text_queries(
        self,
        object_query_text: str,
        part_query_text: str,
        label_id: int,
        points: Optional[np.ndarray] = None,
        nms_iou_threshold: float = 0.35,
        use_soft_nms: bool = False,
        keep_top_k: Optional[int] = None,
        **kwargs
    ) -> List[ObjectPartPair]:
        """
        Run inference using text queries with combined "part of object" query.

        This is the recommended method for OV part-level inference.

        Args:
            object_query_text: Object name (e.g., "cabinet")
            part_query_text: Part name (e.g., "door")
            label_id: Joint semantic label ID
            points: Point cloud coordinates
            nms_iou_threshold: IoU threshold for NMS
            use_soft_nms: Whether to use soft NMS
            keep_top_k: Max pairs to return
            **kwargs: Additional parameters

        Returns:
            List of object-part pairs after NMS

        Example:
            pairs = strategy.infer_with_text_queries(
                object_query_text="cabinet",
                part_query_text="door",
                label_id=121003
            )
        """
        if not self.use_combined_query:
            raise RuntimeError(
                "Combined query mode not enabled. "
                "Set use_combined_query=True when initializing the strategy."
            )

        logger.info(f"Running inference with combined query: '{part_query_text} of {object_query_text}'")

        # Extract combined feature
        combined_feat = self.extract_combined_feature(object_query_text, part_query_text)

        # Retrieve object candidates with combined feature
        logger.info(f"Retrieving object candidates from levels: {self.object_levels}")
        object_results = self._retrieve_with_backoff(
            query_feat=combined_feat,
            levels=self.object_levels,
            target_name="object"
        )
        # Update scores with combined feature
        self.update_proposals_with_combined_feature(
            object_results.proposals,
            combined_feat,
            scale_semantic_score=self.scale_semantic_score,
            apply_softmax=self.apply_softmax
        )

        logger.info(f"Retrieved {len(object_results.proposals)} object proposals")

        # Retrieve part candidates with combined feature
        logger.info(f"Retrieving part candidates from levels: {self.part_levels}")
        part_results = self._retrieve_with_backoff(
            query_feat=combined_feat,
            levels=self.part_levels,
            target_name="part"
        )
        # Update scores with combined feature
        self.update_proposals_with_combined_feature(
            part_results.proposals,
            combined_feat,
            scale_semantic_score=self.scale_semantic_score,
            apply_softmax=self.apply_softmax
        )

        logger.info(f"Retrieved {len(part_results.proposals)} part proposals")

        if len(object_results.proposals) == 0 or len(part_results.proposals) == 0:
            logger.warning("No object or part proposals found")
            return []

        # Pair candidates
        logger.info("Pairing object and part candidates...")
        pairs = self.pairer.pair_all(
            object_proposals=object_results.proposals,
            part_proposals=part_results.proposals,
            label_id=label_id,
            points=points,
            use_tree_pruning=False
        )

        logger.info(f"Created {len(pairs)} valid pairs")

        if len(pairs) == 0:
            return []

        # Apply NMS
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
