"""
Exhaustive Level Pairing Strategy with Combined Query Embeddings.

This strategy extends ExhaustivePairingStrategy by using "X of Y" combined queries:
- For each level-pair, creates a combined query: "{part} of {object}"
  Example: "door of cabinet"
- Extracts feature embedding for the combined query
- Scores proposals using the combined feature

This approach leverages the compositional nature of language models to create
semantically meaningful combined representations.

Algorithm:
1. Enumerate all valid level pairs: (L2,L4), (L2,L6), (L4,L6)
2. For each pair:
   a. Create combined query: "{part_text} of {object_text}"
   b. Extract feature for combined query
   c. Retrieve object proposals using combined feature
   d. Retrieve part proposals using combined feature
   e. Score pairs using combined feature similarity
3. Return predictions from the level-pair with highest score
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import logging
from itertools import product

from .exhaustive_pairing import ExhaustivePairingStrategy
from .combined_query_mixin import CombinedQueryMixin
from ..retrieval import MultiLevelRetriever, Proposal, RetrievalResult
from ..pairing import ObjectPartPair, ObjectPartPairer
from ..scoring import ScoreAggregator
from ..feature_extraction import SigLIPFeatureExtractor

logger = logging.getLogger(__name__)


class ExhaustivePairingWithCombinedQuery(CombinedQueryMixin, ExhaustivePairingStrategy):
    """
    Exhaustive level-pair evaluation with combined "X of Y" query embeddings.

    Key difference from base ExhaustivePairingStrategy:
    - Base: Uses separate object_query_feat and part_query_feat
    - This: Creates "{part} of {object}" combined query and extracts new feature

    Advantages:
    - Leverages compositional semantics of language models
    - Single unified feature captures the object-part relationship
    - More natural for part-level queries

    Disadvantages:
    - Requires text queries (not just features)
    - Requires feature extractor (SigLIP)
    - Slightly more expensive (feature extraction per pair)
    """

    def __init__(
        self,
        retriever: MultiLevelRetriever,
        pairer: ObjectPartPairer,
        feature_extractor: Optional[SigLIPFeatureExtractor] = None,
        siglip_model: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        scale_semantic_score: float = 100.0,
        apply_softmax: bool = True,
        text_template: str = "in_room",
        **kwargs
    ):
        """
        Initialize strategy with feature extractor for combined queries.

        Args:
            retriever: Multi-level retriever
            pairer: Object-part pairer
            feature_extractor: Pre-initialized feature extractor (optional)
            siglip_model: SigLIP model name (if feature_extractor not provided)
            device: Device for feature extraction
            scale_semantic_score: Scaling factor for semantic scores (default 100)
            apply_softmax: Whether to apply softmax to scores
            text_template: Text template for queries (default "in_room")
            **kwargs: Additional arguments for base class
        """
        super().__init__(retriever, pairer, **kwargs)

        # Setup combined query using Mixin
        logger.info("ExhaustivePairingWithCombinedQuery: Enabling combined query mode")
        self.scale_semantic_score = scale_semantic_score
        self.apply_softmax = apply_softmax

        self.setup_combined_query(
            siglip_model=siglip_model,
            device=device,
            feature_extractor=feature_extractor,
            scale_semantic_score=scale_semantic_score,
            apply_softmax=apply_softmax,
            text_template=text_template
        )

    def infer_with_text_queries(
        self,
        object_query_text: str,
        part_query_text: str,
        label_id: int,
        points: Optional[np.ndarray] = None,
        nms_iou_threshold: float = 0.35,
        use_soft_nms: bool = False,
        keep_top_k: Optional[int] = None,
        return_all_pairs: bool = False,
        **kwargs
    ) -> List[ObjectPartPair]:
        """
        Run exhaustive pairing with text queries and combined features.

        Args:
            object_query_text: Object query text (e.g., "cabinet")
            part_query_text: Part query text (e.g., "door")
            label_id: Joint semantic label ID
            points: Point cloud coordinates
            nms_iou_threshold: IoU threshold for NMS
            use_soft_nms: Whether to use soft NMS
            keep_top_k: Max pairs to return
            return_all_pairs: Return from all levels or only best
            **kwargs: Additional parameters

        Returns:
            List of object-part pairs
        """
        logger.info(
            f"Starting exhaustive pairing with combined queries: "
            f"'{part_query_text} of {object_query_text}'"
        )

        # Dictionary to store results for each level pair
        pair_results: Dict[Tuple[str, str], List[ObjectPartPair]] = {}
        pair_scores: Dict[Tuple[str, str], float] = {}

        # Evaluate each level-pair with combined query
        for obj_level, part_level in self.level_pairs:
            logger.info(f"Evaluating level pair: ({obj_level}, {part_level})")

            # Extract feature for combined query using Mixin
            combined_feat = self.extract_combined_feature(object_query_text, part_query_text)
            combined_query_text = self.create_combined_query(object_query_text, part_query_text)
            logger.info(f"  Combined query: '{combined_query_text}'")

            # Retrieve proposals using combined feature
            obj_results = self._retrieve_with_backoff(
                query_feat=combined_feat,
                level=obj_level,
                target_name=f"object@{obj_level} (combined)"
            )

            part_results = self._retrieve_with_backoff(
                query_feat=combined_feat,
                level=part_level,
                target_name=f"part@{part_level} (combined)"
            )

            logger.info(
                f"  Retrieved {len(obj_results.proposals)} objects from {obj_level}, "
                f"{len(part_results.proposals)} parts from {part_level}"
            )

            # Skip if no proposals found
            if len(obj_results.proposals) == 0 or len(part_results.proposals) == 0:
                logger.info(f"  Skipping ({obj_level}, {part_level}): no proposals")
                continue

            # Update proposal scores with combined feature similarity using Mixin
            self.update_proposals_with_combined_feature(
                obj_results.proposals,
                combined_feat,
                scale_semantic_score=self.scale_semantic_score,
                apply_softmax=self.apply_softmax
            )
            self.update_proposals_with_combined_feature(
                part_results.proposals,
                combined_feat,
                scale_semantic_score=self.scale_semantic_score,
                apply_softmax=self.apply_softmax
            )

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

            # Compute aggregate score
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

    def get_level_pair_statistics_with_text(
        self,
        object_query_text: str,
        part_query_text: str,
        label_id: int,
        points: Optional[np.ndarray] = None
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Get detailed statistics for all level pairs using text queries.

        Args:
            object_query_text: Object query text
            part_query_text: Part query text
            label_id: Joint semantic label ID
            points: Point cloud coordinates

        Returns:
            Dict mapping (obj_level, part_level) to statistics dict
        """
        statistics = {}

        for obj_level, part_level in self.level_pairs:
            # Create combined query and extract feature using Mixin
            combined_query_text = self.create_combined_query(object_query_text, part_query_text)
            combined_feat = self.extract_combined_feature(object_query_text, part_query_text)

            # Retrieve proposals
            obj_results = self._retrieve_with_backoff(
                query_feat=combined_feat,
                level=obj_level,
                target_name=f"object@{obj_level}"
            )

            part_results = self._retrieve_with_backoff(
                query_feat=combined_feat,
                level=part_level,
                target_name=f"part@{part_level}"
            )

            # Update scores using Mixin
            self.update_proposals_with_combined_feature(
                obj_results.proposals,
                combined_feat,
                scale_semantic_score=self.scale_semantic_score,
                apply_softmax=self.apply_softmax
            )
            self.update_proposals_with_combined_feature(
                part_results.proposals,
                combined_feat,
                scale_semantic_score=self.scale_semantic_score,
                apply_softmax=self.apply_softmax
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
                'combined_query': combined_query_text,
                'num_objects': len(obj_results.proposals),
                'num_parts': len(part_results.proposals),
                'num_pairs': len(pairs),
                'avg_object_score': np.mean([p.score for p in obj_results.proposals])
                    if len(obj_results.proposals) > 0 else 0.0,
                'avg_part_score': np.mean([p.score for p in part_results.proposals])
                    if len(part_results.proposals) > 0 else 0.0,
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
