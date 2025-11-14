"""
Multi-level retrieval module for open vocabulary 3D instance segmentation.

Provides functionality to retrieve object and part proposals from multiple
hierarchical levels using SigLIP feature embeddings and cosine similarity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    """A single proposal at a specific level."""
    level: str  # e.g., 'L2', 'L4', 'L6'
    proposal_id: int
    mask: np.ndarray  # Binary mask, shape (N_points,)
    feature: np.ndarray  # SigLIP feature embedding
    object_id: Optional[int] = None  # From tracking output
    score: float = 0.0


@dataclass
class RetrievalResult:
    """Result of retrieval for a single query."""
    query: str
    proposals: List[Proposal]
    scores: np.ndarray  # Cosine similarity scores


def compute_cosine_similarity(
    query_feat: np.ndarray,
    proposal_feats: np.ndarray,
    temperature: float = 1.0,
    scale_semantic_score: float = 300.0
) -> np.ndarray:
    """
    Compute cosine similarity between query and proposal features.

    Following the implementation in utils_ov_inference.py:
    1. Compute cosine similarity (normalized dot product)
    2. Scale by scale_semantic_score (default 300)
    3. Apply softmax across all proposals

    Args:
        query_feat: Query embedding, shape (D,)
        proposal_feats: Proposal embeddings, shape (N, D)
        temperature: Temperature for scaling (higher = softer distribution) - DEPRECATED, use scale_semantic_score instead
        scale_semantic_score: Semantic score scaling factor (default 300, matching utils_ov_inference.py)

    Returns:
        Similarity scores after softmax, shape (N,)
    """
    # Normalize features
    query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    proposal_norms = proposal_feats / (np.linalg.norm(proposal_feats, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    similarities = proposal_norms.dot(query_norm)  # [N]

    # Scale by semantic score (matching utils_ov_inference.py line 48)
    scaled_similarities = scale_semantic_score * similarities  # [N]

    # Apply softmax across all proposals
    # exp_scores = exp(scaled_similarities)
    # softmax_scores = exp_scores / sum(exp_scores)
    exp_scores = np.exp(scaled_similarities - np.max(scaled_similarities))  # Numerical stability
    softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)

    return softmax_scores


class MultiLevelRetriever:
    """
    Retrieves proposals from multiple hierarchical levels based on query embeddings.

    Supports:
    - Per-level retrieval with configurable thresholds
    - Multi-level fusion strategies
    - Top-K filtering
    """

    def __init__(
        self,
        proposals_by_level: Dict[str, List[Proposal]],
        temperature: float = 0.2,
        min_similarity: float = 0.2,
        scale_semantic_score: float = 300.0
    ):
        """
        Initialize retriever with proposals at different levels.

        Args:
            proposals_by_level: Dict mapping level names to lists of Proposals
            temperature: Temperature for similarity scoring - DEPRECATED, use scale_semantic_score instead
            min_similarity: Minimum similarity threshold for keeping proposals
            scale_semantic_score: Semantic score scaling factor (default 300, matching utils_ov_inference.py)
        """
        self.proposals_by_level = proposals_by_level
        self.temperature = temperature  # Kept for backward compatibility
        self.min_similarity = min_similarity
        self.scale_semantic_score = scale_semantic_score

    def retrieve_single_level(
        self,
        query_feat: np.ndarray,
        level: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> RetrievalResult:
        """
        Retrieve proposals from a single level.

        Args:
            query_feat: Query embedding, shape (D,)
            level: Level name (e.g., 'L2', 'L4', 'L6')
            top_k: Maximum number of proposals to return
            threshold: Similarity threshold (overrides self.min_similarity)

        Returns:
            RetrievalResult with proposals and scores
        """
        if level not in self.proposals_by_level:
            logger.warning(f"Level {level} not found in proposals")
            return RetrievalResult(query="", proposals=[], scores=np.array([]))

        proposals = self.proposals_by_level[level]
        if len(proposals) == 0:
            return RetrievalResult(query="", proposals=[], scores=np.array([]))

        # Extract features
        features = np.stack([p.feature for p in proposals])

        # Compute similarities with scale_semantic_score (matching utils_ov_inference.py)
        scores = compute_cosine_similarity(
            query_feat,
            features,
            temperature=self.temperature,  # Kept for backward compatibility but not used
            scale_semantic_score=self.scale_semantic_score
        )

        # Apply threshold
        threshold = threshold if threshold is not None else self.min_similarity
        valid_mask = scores >= threshold

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        # Sort by score descending
        sorted_idx = np.argsort(-valid_scores)
        valid_indices = valid_indices[sorted_idx]
        valid_scores = valid_scores[sorted_idx]

        # Apply top-k
        if top_k is not None and len(valid_indices) > top_k:
            valid_indices = valid_indices[:top_k]
            valid_scores = valid_scores[:top_k]

        # Build result
        result_proposals = [proposals[i] for i in valid_indices]
        for prop, score in zip(result_proposals, valid_scores):
            prop.score = float(score)

        return RetrievalResult(
            query="",
            proposals=result_proposals,
            scores=valid_scores
        )

    def retrieve_multi_level(
        self,
        query_feat: np.ndarray,
        levels: List[str],
        top_k_per_level: Optional[int] = None,
        threshold: Optional[float] = None,
        deduplicate: bool = True,
        iou_threshold: float = 0.95
    ) -> RetrievalResult:
        """
        Retrieve proposals from multiple levels and combine results.

        Args:
            query_feat: Query embedding
            levels: List of level names to search
            top_k_per_level: Max proposals per level
            threshold: Similarity threshold
            deduplicate: Whether to remove near-duplicate proposals
            iou_threshold: IoU threshold for deduplication

        Returns:
            Combined RetrievalResult
        """
        all_proposals = []
        all_scores = []

        for level in levels:
            result = self.retrieve_single_level(
                query_feat, level, top_k=top_k_per_level, threshold=threshold
            )
            all_proposals.extend(result.proposals)
            all_scores.extend(result.scores)

        all_scores = np.array(all_scores) if all_scores else np.array([])

        # Deduplicate if requested
        if deduplicate and len(all_proposals) > 1:
            all_proposals, all_scores = self._deduplicate_proposals(
                all_proposals, all_scores, iou_threshold
            )

        return RetrievalResult(
            query="",
            proposals=all_proposals,
            scores=all_scores
        )

    def _deduplicate_proposals(
        self,
        proposals: List[Proposal],
        scores: np.ndarray,
        iou_threshold: float
    ) -> Tuple[List[Proposal], np.ndarray]:
        """Remove near-duplicate proposals based on IoU."""
        from .geometry import compute_iou

        if len(proposals) <= 1:
            return proposals, scores

        # Sort by score descending
        sorted_idx = np.argsort(-scores)

        keep_idx = []
        for i in sorted_idx:
            # Check against all kept proposals
            should_keep = True
            for j in keep_idx:
                iou = compute_iou(proposals[i].mask, proposals[j].mask)
                if iou > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep_idx.append(i)

        kept_proposals = [proposals[i] for i in keep_idx]
        kept_scores = scores[keep_idx]

        return kept_proposals, kept_scores

    def retrieve_for_object_part(
        self,
        object_query_feat: np.ndarray,
        part_query_feat: np.ndarray,
        object_levels: Optional[List[str]] = None,
        part_levels: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[RetrievalResult, RetrievalResult]:
        """
        Retrieve object and part proposals separately.

        Args:
            object_query_feat: Object query embedding
            part_query_feat: Part query embedding
            object_levels: Levels to search for objects (default: ['L2', 'L4'])
            part_levels: Levels to search for parts (default: ['L4', 'L6'])
            **kwargs: Additional arguments for retrieve_multi_level

        Returns:
            (object_results, part_results)
        """
        if object_levels is None:
            object_levels = ['L2', 'L4']
        if part_levels is None:
            part_levels = ['L4', 'L6']

        object_results = self.retrieve_multi_level(object_query_feat, object_levels, **kwargs)
        part_results = self.retrieve_multi_level(part_query_feat, part_levels, **kwargs)

        return object_results, part_results
