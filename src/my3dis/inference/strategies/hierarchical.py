"""
Hierarchical Coarse-to-Fine Strategy.

This strategy (Strategy B from the algorithm advice) leverages the family tree
to perform efficient coarse-to-fine retrieval:

1. Coarse Object Localization (L2)
2. Multi-resolution Object Refinement (L2 → L4)
3. Part Search in Relevant Regions (descendants of selected objects)
4. Joint Scoring and Validation
5. Cross-path NMS

This strategy is more efficient and leverages hierarchical structure.

NOW SUPPORTS COMBINED QUERY MODE (DEFAULT):
- Use "part of object" combined queries for improved semantic matching
- Example: "door of cabinet" instead of separate "door" and "cabinet"
"""

from typing import List, Optional, Dict, Set
import numpy as np
import logging

from .base import InferenceStrategy
from .combined_query_mixin import CombinedQueryMixin
from ..retrieval import MultiLevelRetriever, Proposal
from ..pairing import ObjectPartPair, ObjectPartPairer, FamilyTree
from ..nms import multi_instance_nms

logger = logging.getLogger(__name__)


class HierarchicalStrategy(CombinedQueryMixin, InferenceStrategy):
    """
    Hierarchical coarse-to-fine retrieval with tree-guided search.

    Advantages:
    - Efficient (uses tree to prune search space)
    - Coarse-to-fine aligns with human intuition
    - Leverages hierarchical structure

    Disadvantages:
    - Can miss objects if coarse localization fails
    - Depends on quality of family tree
    - Requires careful tuning of K values
    """

    def __init__(
        self,
        retriever: MultiLevelRetriever,
        pairer: ObjectPartPairer,
        family_tree: FamilyTree,
        coarse_top_k: int = 100,
        object_top_k: int = 50,
        part_top_k: int = 30,
        coarse_threshold: float = 0.2,
        refinement_threshold: float = 0.3,
        refinement_margin: float = 0.1,
        use_combined_query: bool = True,
        siglip_model: str = "google/siglip-so400m-patch14-384",
        siglip_device: str = "cuda",
        skip_model_load: bool = False,
        feature_extractor = None,
        **kwargs
    ):
        """
        Initialize hierarchical strategy.

        Args:
            retriever: Multi-level retriever
            pairer: Object-part pairer
            family_tree: Family tree for hierarchical search
            coarse_top_k: Top-K for coarse localization (Stage 1)
            object_top_k: Max object candidates after refinement (Stage 2)
            part_top_k: Max part candidates per object (Stage 3)
            coarse_threshold: Minimum similarity for coarse candidates
            refinement_threshold: Minimum similarity for refined candidates
            refinement_margin: Margin for preferring child over parent
            use_combined_query: Use "part of object" combined queries (DEFAULT: True)
            siglip_model: SigLIP model for combined query features
            siglip_device: Device for SigLIP (cuda:0, cuda:1, or cpu)
            skip_model_load: Skip SigLIP loading (for testing)
            feature_extractor: Pre-initialized feature extractor (optional)
            **kwargs: Additional arguments
        """
        super().__init__(retriever, pairer, family_tree=family_tree, **kwargs)

        if family_tree is None:
            raise ValueError("HierarchicalStrategy requires a family tree")

        self.coarse_top_k = coarse_top_k
        self.object_top_k = object_top_k
        self.part_top_k = part_top_k
        self.coarse_threshold = coarse_threshold
        self.refinement_threshold = refinement_threshold
        self.refinement_margin = refinement_margin

        # Setup combined query if enabled
        self.use_combined_query = use_combined_query
        if use_combined_query:
            logger.info("HierarchicalStrategy: Enabling combined query mode")
            self.setup_combined_query(
                siglip_model=siglip_model,
                device=siglip_device,
                skip_model_load=skip_model_load,
                feature_extractor=feature_extractor
            )
        else:
            logger.info("HierarchicalStrategy: Using separate query mode")

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
        Run hierarchical coarse-to-fine retrieval.

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
        # Stage 1: Coarse Object Localization
        logger.info("Stage 1: Coarse object localization at L2")
        coarse_candidates = self._coarse_localization(object_query_feat)

        if len(coarse_candidates) == 0:
            logger.warning("No coarse candidates found")
            return []

        # Stage 2: Multi-resolution Object Refinement
        logger.info("Stage 2: Multi-resolution object refinement")
        object_candidates = self._refine_objects(object_query_feat, coarse_candidates)

        logger.info(f"Refined to {len(object_candidates)} object candidates")

        if len(object_candidates) == 0:
            return []

        # Stage 3: Part Search in Relevant Regions
        logger.info("Stage 3: Part search in relevant regions")
        all_pairs = self._search_parts_for_objects(
            part_query_feat,
            object_candidates,
            label_id,
            points
        )

        logger.info(f"Created {len(all_pairs)} candidate pairs")

        if len(all_pairs) == 0:
            return []

        # Stage 4 & 5: Cross-path NMS
        logger.info("Stage 5: Applying cross-path NMS")
        filtered_pairs = self.apply_nms_to_pairs(
            all_pairs,
            nms_iou_threshold=nms_iou_threshold,
            points=points,
            use_soft_nms=use_soft_nms,
            keep_top_k=keep_top_k
        )

        logger.info(f"Final result: {len(filtered_pairs)} pairs after NMS")

        return filtered_pairs

    def _coarse_localization(
        self,
        object_query_feat: np.ndarray
    ) -> List[Proposal]:
        """
        Stage 1: Coarse localization at L2.

        Args:
            object_query_feat: Object query embedding

        Returns:
            Top-K L2 proposals as coarse candidates
        """
        result = self.retriever.retrieve_single_level(
            object_query_feat,
            level='L2',
            top_k=self.coarse_top_k,
            threshold=self.coarse_threshold
        )

        logger.info(f"Coarse localization: {len(result.proposals)} candidates from L2")

        return result.proposals

    def _refine_objects(
        self,
        object_query_feat: np.ndarray,
        coarse_candidates: List[Proposal]
    ) -> List[Proposal]:
        """
        Stage 2: Multi-resolution refinement.

        For each L2 coarse candidate:
        1. Keep the L2 proposal itself
        2. Check its L4 children
        3. If any child scores higher (+ margin), include it

        Args:
            object_query_feat: Object query embedding
            coarse_candidates: L2 coarse candidates

        Returns:
            Refined list of object candidates (mix of L2 and L4)
        """
        object_candidates = []

        for l2_prop in coarse_candidates:
            # Keep the L2 proposal
            object_candidates.append(l2_prop)

            # Get L4 children
            if l2_prop.object_id is not None:
                children_ids = self.family_tree.get_children(
                    l2_prop.object_id,
                    target_level='L4'
                )

                if len(children_ids) > 0:
                    # Get L4 proposals
                    l4_proposals = self.retriever.proposals_by_level.get('L4', [])
                    l4_children = [
                        p for p in l4_proposals
                        if p.object_id in children_ids
                    ]

                    if len(l4_children) > 0:
                        # Compute scores for children
                        from ..retrieval import compute_cosine_similarity

                        child_features = np.stack([p.feature for p in l4_children])
                        child_scores = compute_cosine_similarity(
                            object_query_feat,
                            child_features,
                            temperature=self.retriever.temperature,
                            scale_semantic_score=self.retriever.scale_semantic_score
                        )

                        # Add children that score well
                        for child, score in zip(l4_children, child_scores):
                            # Include if score is high enough or better than parent
                            if (score > self.refinement_threshold or
                                score > l2_prop.score + self.refinement_margin):
                                child.score = float(score)
                                object_candidates.append(child)

        # Sort by score and limit
        object_candidates = sorted(
            object_candidates,
            key=lambda p: p.score,
            reverse=True
        )[:self.object_top_k]

        return object_candidates

    def _search_parts_for_objects(
        self,
        part_query_feat: np.ndarray,
        object_candidates: List[Proposal],
        label_id: int,
        points: Optional[np.ndarray]
    ) -> List[ObjectPartPair]:
        """
        Stage 3: Search for parts in regions defined by object candidates.

        For each object candidate:
        1. Get its descendants (L4, L6)
        2. Score them against part query
        3. Pair valid parts with the object

        Args:
            part_query_feat: Part query embedding
            object_candidates: Object candidates from Stage 2
            label_id: Joint semantic label
            points: Point cloud coordinates

        Returns:
            All candidate pairs
        """
        all_pairs = []

        for obj_prop in object_candidates:
            # Get potential part proposals
            part_candidates = self._get_part_candidates_for_object(
                obj_prop,
                part_query_feat
            )

            if len(part_candidates) == 0:
                continue

            # Pair with this object
            pairs = self.pairer.pair_all(
                object_proposals=[obj_prop],
                part_proposals=part_candidates,
                label_id=label_id,
                points=points,
                use_tree_pruning=True
            )

            all_pairs.extend(pairs)

        return all_pairs

    def _get_part_candidates_for_object(
        self,
        obj_prop: Proposal,
        part_query_feat: np.ndarray
    ) -> List[Proposal]:
        """
        Get part candidates for a specific object proposal.

        Args:
            obj_prop: Object proposal
            part_query_feat: Part query embedding

        Returns:
            List of part proposals (scored and filtered)
        """
        part_pool_ids = set()

        # Get descendants based on object level
        if obj_prop.object_id is not None:
            if obj_prop.level == 'L2':
                # Get L4 and L6 descendants
                descendants = self.family_tree.get_descendants(
                    obj_prop.object_id,
                    levels=['L4', 'L6']
                )
                part_pool_ids.update(descendants)

            elif obj_prop.level == 'L4':
                # Get L6 children + include the object itself (part could be same level)
                children = self.family_tree.get_children(
                    obj_prop.object_id,
                    target_level='L6'
                )
                part_pool_ids.update(children)
                part_pool_ids.add(obj_prop.object_id)  # Include itself

        # Get proposals from pool
        part_proposals = []
        for level in ['L4', 'L6']:
            level_proposals = self.retriever.proposals_by_level.get(level, [])
            part_proposals.extend([
                p for p in level_proposals
                if p.object_id in part_pool_ids
            ])

        if len(part_proposals) == 0:
            return []

        # Score parts
        from ..retrieval import compute_cosine_similarity

        part_features = np.stack([p.feature for p in part_proposals])
        part_scores = compute_cosine_similarity(
            part_query_feat,
            part_features,
            temperature=self.retriever.temperature,
            scale_semantic_score=self.retriever.scale_semantic_score
        )

        # Filter and sort
        valid_parts = []
        for prop, score in zip(part_proposals, part_scores):
            if score > self.refinement_threshold:
                prop.score = float(score)
                valid_parts.append(prop)

        valid_parts = sorted(valid_parts, key=lambda p: p.score, reverse=True)

        # Limit to top-K per object
        return valid_parts[:self.part_top_k]

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
        Run hierarchical inference using text queries with combined "part of object" query.

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

        logger.info(f"Running hierarchical inference with combined query: '{part_query_text} of {object_query_text}'")

        # Extract combined feature
        combined_feat = self.extract_combined_feature(object_query_text, part_query_text)

        # Stage 1: Coarse Object Localization with combined feature
        logger.info("Stage 1: Coarse object localization at L2")
        coarse_candidates = self._coarse_localization_with_combined_feature(combined_feat)

        if len(coarse_candidates) == 0:
            logger.warning("No coarse candidates found")
            return []

        # Stage 2: Multi-resolution Object Refinement with combined feature
        logger.info("Stage 2: Multi-resolution object refinement")
        object_candidates = self._refine_objects_with_combined_feature(combined_feat, coarse_candidates)

        logger.info(f"Refined to {len(object_candidates)} object candidates")

        if len(object_candidates) == 0:
            return []

        # Stage 3: Part Search in Relevant Regions with combined feature
        logger.info("Stage 3: Part search in relevant regions")
        all_pairs = self._search_parts_for_objects_with_combined_feature(
            combined_feat,
            object_candidates,
            label_id,
            points
        )

        logger.info(f"Created {len(all_pairs)} candidate pairs")

        if len(all_pairs) == 0:
            return []

        # Stage 4 & 5: Cross-path NMS
        logger.info("Stage 5: Applying cross-path NMS")
        filtered_pairs = self.apply_nms_to_pairs(
            all_pairs,
            nms_iou_threshold=nms_iou_threshold,
            points=points,
            use_soft_nms=use_soft_nms,
            keep_top_k=keep_top_k
        )

        logger.info(f"Final result: {len(filtered_pairs)} pairs after NMS")

        return filtered_pairs

    def _coarse_localization_with_combined_feature(
        self,
        combined_feat: np.ndarray
    ) -> List[Proposal]:
        """
        Stage 1: Coarse localization at L2 using combined feature.

        Args:
            combined_feat: Combined query feature

        Returns:
            Top-K L2 proposals as coarse candidates
        """
        result = self.retriever.retrieve_single_level(
            combined_feat,
            level='L2',
            top_k=self.coarse_top_k,
            threshold=self.coarse_threshold
        )

        # Update scores with combined feature
        self.update_proposals_with_combined_feature(result.proposals, combined_feat)

        logger.info(f"Coarse localization: {len(result.proposals)} candidates from L2")

        return result.proposals

    def _refine_objects_with_combined_feature(
        self,
        combined_feat: np.ndarray,
        coarse_candidates: List[Proposal]
    ) -> List[Proposal]:
        """
        Stage 2: Multi-resolution refinement using combined feature.

        For each L2 coarse candidate:
        1. Keep the L2 proposal itself
        2. Check its L4 children
        3. If any child scores higher (+ margin), include it

        Args:
            combined_feat: Combined query feature
            coarse_candidates: L2 coarse candidates

        Returns:
            Refined list of object candidates (mix of L2 and L4)
        """
        object_candidates = []

        for l2_prop in coarse_candidates:
            # Keep the L2 proposal
            object_candidates.append(l2_prop)

            # Get L4 children
            if l2_prop.object_id is not None:
                children_ids = self.family_tree.get_children(
                    l2_prop.object_id,
                    target_level='L4'
                )

                if len(children_ids) > 0:
                    # Get L4 proposals
                    l4_proposals = self.retriever.proposals_by_level.get('L4', [])
                    l4_children = [
                        p for p in l4_proposals
                        if p.object_id in children_ids
                    ]

                    if len(l4_children) > 0:
                        # Compute scores for children using combined feature
                        from ..retrieval import compute_cosine_similarity

                        child_features = np.stack([p.feature for p in l4_children])
                        child_scores = compute_cosine_similarity(
                            combined_feat,
                            child_features,
                            temperature=self.retriever.temperature,
                            scale_semantic_score=self.retriever.scale_semantic_score
                        )

                        # Add children that score well
                        for child, score in zip(l4_children, child_scores):
                            # Include if score is high enough or better than parent
                            if (score > self.refinement_threshold or
                                score > l2_prop.score + self.refinement_margin):
                                child.score = float(score)
                                object_candidates.append(child)

        # Sort by score and limit
        object_candidates = sorted(
            object_candidates,
            key=lambda p: p.score,
            reverse=True
        )[:self.object_top_k]

        return object_candidates

    def _search_parts_for_objects_with_combined_feature(
        self,
        combined_feat: np.ndarray,
        object_candidates: List[Proposal],
        label_id: int,
        points: Optional[np.ndarray]
    ) -> List[ObjectPartPair]:
        """
        Stage 3: Search for parts in regions defined by object candidates using combined feature.

        For each object candidate:
        1. Get its descendants (L4, L6)
        2. Score them against combined feature
        3. Pair valid parts with the object

        Args:
            combined_feat: Combined query feature
            object_candidates: Object candidates from Stage 2
            label_id: Joint semantic label
            points: Point cloud coordinates

        Returns:
            All candidate pairs
        """
        all_pairs = []

        for obj_prop in object_candidates:
            # Get potential part proposals
            part_candidates = self._get_part_candidates_for_object_with_combined_feature(
                obj_prop,
                combined_feat
            )

            if len(part_candidates) == 0:
                continue

            # Pair with this object
            pairs = self.pairer.pair_all(
                object_proposals=[obj_prop],
                part_proposals=part_candidates,
                label_id=label_id,
                points=points,
                use_tree_pruning=True
            )

            all_pairs.extend(pairs)

        return all_pairs

    def _get_part_candidates_for_object_with_combined_feature(
        self,
        obj_prop: Proposal,
        combined_feat: np.ndarray
    ) -> List[Proposal]:
        """
        Get part candidates for a specific object proposal using combined feature.

        Args:
            obj_prop: Object proposal
            combined_feat: Combined query feature

        Returns:
            List of part proposals (scored and filtered)
        """
        part_pool_ids = set()

        # Get descendants based on object level
        if obj_prop.object_id is not None:
            if obj_prop.level == 'L2':
                # Get L4 and L6 descendants
                descendants = self.family_tree.get_descendants(
                    obj_prop.object_id,
                    levels=['L4', 'L6']
                )
                part_pool_ids.update(descendants)

            elif obj_prop.level == 'L4':
                # Get L6 children + include the object itself (part could be same level)
                children = self.family_tree.get_children(
                    obj_prop.object_id,
                    target_level='L6'
                )
                part_pool_ids.update(children)
                part_pool_ids.add(obj_prop.object_id)  # Include itself

        # Get proposals from pool
        part_proposals = []
        for level in ['L4', 'L6']:
            level_proposals = self.retriever.proposals_by_level.get(level, [])
            part_proposals.extend([
                p for p in level_proposals
                if p.object_id in part_pool_ids
            ])

        if len(part_proposals) == 0:
            return []

        # Score parts using combined feature
        from ..retrieval import compute_cosine_similarity

        part_features = np.stack([p.feature for p in part_proposals])
        part_scores = compute_cosine_similarity(
            combined_feat,
            part_features,
            temperature=self.retriever.temperature,
            scale_semantic_score=self.retriever.scale_semantic_score
        )

        # Filter and sort
        valid_parts = []
        for prop, score in zip(part_proposals, part_scores):
            if score > self.refinement_threshold:
                prop.score = float(score)
                valid_parts.append(prop)

        valid_parts = sorted(valid_parts, key=lambda p: p.score, reverse=True)

        # Limit to top-K per object
        return valid_parts[:self.part_top_k]
