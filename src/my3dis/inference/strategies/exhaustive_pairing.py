"""
Exhaustive Level Pairing Strategy.

This strategy exhaustively tries all possible level combinations (pairs) and selects
the one that produces the best scoring object-part matches.

Key differences from Independent strategy:
- Independent: Retrieves from all levels simultaneously, then pairs all combinations
- Exhaustive: Evaluates each level-pair separately, selects best performing pair

Algorithm:
1. Enumerate all valid level pairs: (L2,L4), (L2,L6), (L4,L6)
2. For each pair:
   a. Retrieve object proposals from object_level
   b. Retrieve part proposals from part_level
   c. Pair and score all combinations
3. Return predictions from the level-pair with highest average/max score
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import logging
from itertools import product

from .base import InferenceStrategy
from ..retrieval import MultiLevelRetriever, Proposal, RetrievalResult
from ..pairing import ObjectPartPair, ObjectPartPairer
from ..scoring import ScoreAggregator

logger = logging.getLogger(__name__)


class ExhaustivePairingStrategy(InferenceStrategy):
    """
    Exhaustive level-pair evaluation strategy.

    For each query, this strategy:
    1. Enumerates all valid (object_level, part_level) combinations
    2. Retrieves and pairs proposals for each combination independently
    3. Selects the level-pair that produces the best results

    Advantages:
    - Finds optimal level granularity for each query
    - More interpretable (you know which levels worked)
    - Can detect when assumptions about levels are wrong

    Disadvantages:
    - More computationally expensive (evaluates each pair separately)
    - May miss cross-level combinations in some edge cases
    """

    def __init__(
        self,
        retriever: MultiLevelRetriever,
        pairer: ObjectPartPairer,
        available_levels: Optional[List[str]] = None,
        top_k_per_level: int = 100,
        retrieval_threshold: float = 0.2,
        min_retrieval_threshold: float = 0.05,
        threshold_backoff_step: float = 0.03,
        fallback_to_topk: bool = True,
        selection_metric: str = "max",  # 'max', 'mean', or 'sum'
        use_tree_pruning: bool = False,
        return_all_pairs: bool = False,
        **kwargs
    ):
        """
        Initialize exhaustive pairing strategy.

        Args:
            retriever: Multi-level retriever
            pairer: Object-part pairer
            available_levels: Levels to consider (default: ['L2', 'L4', 'L6'])
            top_k_per_level: Max proposals per level
            retrieval_threshold: Minimum similarity for retrieval
            min_retrieval_threshold: Minimum threshold for fallback
            threshold_backoff_step: Step size for threshold backoff
            fallback_to_topk: Fall back to top-k if no proposals found
            selection_metric: How to select best level-pair
                            - 'max': Choose pair with highest single score
                            - 'mean': Choose pair with highest average score
                            - 'sum': Choose pair with highest total score
            use_tree_pruning: Use family tree for pruning pairs
            return_all_pairs: Default value for return_all_pairs in infer()
            **kwargs: Additional arguments for base class
        """
        super().__init__(retriever, pairer, **kwargs)

        self.available_levels = available_levels or ['L2', 'L4', 'L6']
        self.top_k_per_level = top_k_per_level
        self.retrieval_threshold = retrieval_threshold
        self.min_retrieval_threshold = max(0.0, min_retrieval_threshold)
        self.threshold_backoff_step = max(0.0, threshold_backoff_step)
        self.fallback_to_topk = fallback_to_topk
        self.selection_metric = selection_metric
        self.use_tree_pruning = use_tree_pruning
        self.return_all_pairs = return_all_pairs

        # Generate all valid level pairs
        self.level_pairs = self._generate_level_pairs()
        logger.info(f"Initialized ExhaustivePairingStrategy with {len(self.level_pairs)} level pairs: {self.level_pairs}")

    def _generate_level_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate all valid (object_level, part_level) pairs.

        Valid pairs are those where part_level >= object_level in hierarchy.
        Assumes available_levels are sorted from coarse to fine.
        """
        # Create level order mapping dynamically from available_levels
        level_order = {level: idx for idx, level in enumerate(sorted(self.available_levels))}
        pairs = []

        for obj_level in self.available_levels:
            for part_level in self.available_levels:
                obj_idx = level_order.get(obj_level, -1)
                part_idx = level_order.get(part_level, -1)

                # Allow pairs where object is at same or coarser level than part
                # (obj_idx <= part_idx means obj is coarser or equal)
                if obj_idx >= 0 and part_idx >= 0 and obj_idx <= part_idx:
                    pairs.append((obj_level, part_level))

        return pairs

    def infer(
        self,
        object_query_feat: np.ndarray,
        part_query_feat: np.ndarray,
        label_id: int,
        points: Optional[np.ndarray] = None,
        nms_iou_threshold: float = 0.35,
        use_soft_nms: bool = False,
        keep_top_k: Optional[int] = None,
        return_all_pairs: Optional[bool] = None,
        **kwargs
    ) -> List[ObjectPartPair]:
        """
        Run exhaustive level-pair evaluation.

        Args:
            object_query_feat: Object query embedding
            part_query_feat: Part query embedding
            label_id: Joint semantic label ID
            points: Point cloud coordinates
            nms_iou_threshold: IoU threshold for NMS
            use_soft_nms: Whether to use soft NMS
            keep_top_k: Max pairs to return
            return_all_pairs: If True, return pairs from all level combinations
                            If False, return only from best level-pair
            **kwargs: Additional parameters

        Returns:
            List of object-part pairs (from best level-pair or all pairs)
        """
        # Use instance attribute if not provided
        if return_all_pairs is None:
            return_all_pairs = self.return_all_pairs

        logger.info(f"Starting exhaustive pairing for label_id={label_id}")

        # Dictionary to store results for each level pair
        pair_results: Dict[Tuple[str, str], List[ObjectPartPair]] = {}
        pair_scores: Dict[Tuple[str, str], float] = {}

        # Evaluate each level-pair independently
        for obj_level, part_level in self.level_pairs:
            logger.info(f"Evaluating level pair: ({obj_level}, {part_level})")

            # Retrieve proposals for this specific pair
            obj_results = self._retrieve_with_backoff(
                query_feat=object_query_feat,
                level=obj_level,
                target_name=f"object@{obj_level}"
            )

            part_results = self._retrieve_with_backoff(
                query_feat=part_query_feat,
                level=part_level,
                target_name=f"part@{part_level}"
            )

            logger.info(
                f"  Retrieved {len(obj_results.proposals)} objects from {obj_level}, "
                f"{len(part_results.proposals)} parts from {part_level}"
            )

            # Skip if no proposals found
            if len(obj_results.proposals) == 0 or len(part_results.proposals) == 0:
                logger.info(f"  Skipping ({obj_level}, {part_level}): no proposals")
                continue

            # Pair proposals
            pairs = self.pairer.pair_all(
                object_proposals=obj_results.proposals,
                part_proposals=part_results.proposals,
                label_id=label_id,
                points=points,
                use_tree_pruning=self.use_tree_pruning
            )

            logger.info(f"  Created {len(pairs)} valid pairs")

            if len(pairs) == 0:
                continue

            # Store results
            pair_results[(obj_level, part_level)] = pairs

            # Compute aggregate score for this level-pair
            scores = [p.combined_score for p in pairs]
            if self.selection_metric == "max":
                pair_scores[(obj_level, part_level)] = max(scores)
            elif self.selection_metric == "mean":
                pair_scores[(obj_level, part_level)] = np.mean(scores)
            elif self.selection_metric == "sum":
                pair_scores[(obj_level, part_level)] = sum(scores)
            else:
                pair_scores[(obj_level, part_level)] = max(scores)

            logger.info(
                f"  Level pair ({obj_level}, {part_level}) score "
                f"({self.selection_metric}): {pair_scores[(obj_level, part_level)]:.3f}"
            )

        # Check if we found any valid pairs
        if len(pair_results) == 0:
            logger.warning("No valid pairs found across all level combinations")
            return []

        # Select best level-pair or combine all
        if return_all_pairs:
            logger.info("Combining pairs from all level combinations")
            all_pairs = []
            for pairs in pair_results.values():
                all_pairs.extend(pairs)
            final_pairs = all_pairs
        else:
            # Select level-pair with best score
            best_pair = max(pair_scores.items(), key=lambda x: x[1])
            best_levels, best_score = best_pair
            logger.info(
                f"Selected best level pair: {best_levels} "
                f"(score: {best_score:.3f})"
            )
            final_pairs = pair_results[best_levels]

        # Apply NMS
        logger.info(f"Applying NMS to {len(final_pairs)} pairs...")
        filtered_pairs = self.apply_nms_to_pairs(
            final_pairs,
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
        level: str,
        target_name: str
    ) -> RetrievalResult:
        """
        Retrieve from a single level with adaptive threshold backoff.

        Args:
            query_feat: Query embedding
            level: Level to retrieve from
            target_name: Name for logging

        Returns:
            RetrievalResult
        """
        # Check if level has candidates
        if not self._has_level_candidates([level]):
            logger.warning(f"No proposals available for {level}")
            return RetrievalResult(query="", proposals=[], scores=np.array([]))

        # Try progressively lower thresholds
        for threshold in self._build_threshold_schedule():
            result = self.retriever.retrieve_single_level(
                query_feat,
                level=level,
                top_k=self.top_k_per_level,
                threshold=threshold
            )

            if len(result.proposals) > 0:
                if threshold < self.retrieval_threshold:
                    logger.info(
                        f"{target_name} retrieval fallback: lowered threshold to "
                        f"{threshold:.2f} (found {len(result.proposals)} proposals)"
                    )
                return result

        # Fallback to top-k if enabled
        if self.fallback_to_topk:
            logger.warning(
                f"{target_name} retrieval fallback: forcing top-k results"
            )
            return self.retriever.retrieve_single_level(
                query_feat,
                level=level,
                top_k=self.top_k_per_level,
                threshold=-1.0
            )

        return RetrievalResult(query="", proposals=[], scores=np.array([]))

    def _build_threshold_schedule(self) -> List[float]:
        """Generate descending list of thresholds for adaptive retrieval."""
        thresholds = [self.retrieval_threshold]

        if self.threshold_backoff_step > 0:
            current = self.retrieval_threshold - self.threshold_backoff_step
            while current > self.min_retrieval_threshold:
                thresholds.append(current)
                current -= self.threshold_backoff_step

        thresholds.append(self.min_retrieval_threshold)

        # Deduplicate and sort descending
        thresholds = sorted(set(round(t, 6) for t in thresholds), reverse=True)
        return thresholds

    def _has_level_candidates(self, levels: List[str]) -> bool:
        """Check if any proposals exist for the requested levels."""
        for level in levels:
            if len(self.retriever.proposals_by_level.get(level, [])) > 0:
                return True
        return False

    def get_level_pair_statistics(
        self,
        object_query_feat: np.ndarray,
        part_query_feat: np.ndarray,
        label_id: int,
        points: Optional[np.ndarray] = None
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Get detailed statistics for all level pairs (for analysis/debugging).

        Args:
            object_query_feat: Object query embedding
            part_query_feat: Part query embedding
            label_id: Joint semantic label ID
            points: Point cloud coordinates

        Returns:
            Dict mapping (obj_level, part_level) to statistics dict
        """
        statistics = {}

        for obj_level, part_level in self.level_pairs:
            # Retrieve proposals
            obj_results = self._retrieve_with_backoff(
                query_feat=object_query_feat,
                level=obj_level,
                target_name=f"object@{obj_level}"
            )

            part_results = self._retrieve_with_backoff(
                query_feat=part_query_feat,
                level=part_level,
                target_name=f"part@{part_level}"
            )

            # Compute pairs
            pairs = []
            if len(obj_results.proposals) > 0 and len(part_results.proposals) > 0:
                pairs = self.pairer.pair_all(
                    object_proposals=obj_results.proposals,
                    part_proposals=part_results.proposals,
                    label_id=label_id,
                    points=points,
                    use_tree_pruning=self.use_tree_pruning
                )

            # Compute statistics
            stats = {
                'num_objects': len(obj_results.proposals),
                'num_parts': len(part_results.proposals),
                'num_pairs': len(pairs),
                'avg_object_score': np.mean(obj_results.scores) if len(obj_results.scores) > 0 else 0.0,
                'avg_part_score': np.mean(part_results.scores) if len(part_results.scores) > 0 else 0.0,
            }

            if len(pairs) > 0:
                pair_scores = [p.combined_score for p in pairs]
                stats['avg_pair_score'] = np.mean(pair_scores)
                stats['max_pair_score'] = max(pair_scores)
                stats['min_pair_score'] = min(pair_scores)
            else:
                stats['avg_pair_score'] = 0.0
                stats['max_pair_score'] = 0.0
                stats['min_pair_score'] = 0.0

            statistics[(obj_level, part_level)] = stats

        return statistics
