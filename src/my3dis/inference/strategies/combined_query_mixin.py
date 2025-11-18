"""
Combined Query Mixin for Inference Strategies.

Provides "part of object" combined query functionality that can be mixed into
any inference strategy. This is the recommended approach for OV part-level inference.

Example Usage:
    class MyStrategy(CombinedQueryMixin, InferenceStrategy):
        def __init__(self, retriever, pairer, use_combined_query=True, **kwargs):
            super().__init__(retriever, pairer, **kwargs)

            if use_combined_query:
                self.setup_combined_query(
                    siglip_model=kwargs.get('siglip_model'),
                    device=kwargs.get('siglip_device', 'cuda')
                )
"""

from typing import List, Optional
import numpy as np
import logging

from ..retrieval import Proposal

logger = logging.getLogger(__name__)


class CombinedQueryMixin:
    """
    Mixin to add combined "part of object" query functionality to strategies.

    This mixin provides:
    - SigLIP feature extraction for combined text queries
    - Proposal score updating with combined features
    - Convenience methods for text-based inference

    Key Method:
        create_combined_query(object_text, part_text) -> "part_text of object_text"
    """

    def setup_combined_query(
        self,
        siglip_model: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        feature_extractor = None,
        scale_semantic_score: float = 100.0,
        apply_softmax: bool = True,
        text_template: str = "in_room",
    ):
        """
        Initialize combined query functionality.

        Args:
            siglip_model: SigLIP model name
            device: Device for feature extraction
            feature_extractor: Pre-initialized feature extractor (optional)
            scale_semantic_score: Scaling factor for semantic scores (default 100)
            apply_softmax: Whether to apply softmax to scores
            text_template: Text template for queries (default "in_room")
                - "raw": Use class name directly
                - "in_room": "a {class} in a room" (BEST)
                - "photo_of": "a photo of a {class}"
                - "indoor": "a {class} in an indoor scene"
        """
        self.scale_semantic_score = scale_semantic_score
        self.apply_softmax = apply_softmax
        self.text_template = text_template

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            self.has_combined_query = True
            logger.info("Using provided feature extractor for combined queries")
        else:
            from ..feature_extraction import SigLIPFeatureExtractor

            logger.info(f"Initializing SigLIP for combined queries: {siglip_model}")
            self.feature_extractor = SigLIPFeatureExtractor(
                model_name=siglip_model,
                device=device
            )
            self.has_combined_query = True

        logger.info(f"Combined query config: template={text_template}, scale={scale_semantic_score}, softmax={apply_softmax}")

    def create_combined_query(self, object_text: str, part_text: str) -> str:
        """
        Create combined "part of object" query text.

        Args:
            object_text: Object name (e.g., "cabinet")
            part_text: Part name (e.g., "door")

        Returns:
            Combined query (e.g., "door of cabinet")
        """
        return f"{part_text} of {object_text}"

    def extract_combined_feature(
        self,
        object_text: str,
        part_text: str
    ) -> np.ndarray:
        """
        Extract feature for combined "part of object" query.

        Args:
            object_text: Object name
            part_text: Part name

        Returns:
            Combined feature embedding, shape (D,)
        """
        if not hasattr(self, 'has_combined_query') or not self.has_combined_query:
            raise RuntimeError(
                "Combined query not initialized. "
                "Call setup_combined_query() first."
            )

        combined_text = self.create_combined_query(object_text, part_text)
        template = getattr(self, 'text_template', 'in_room')
        logger.debug(f"Extracting feature for: '{combined_text}' with template={template}")

        # Extract features with template (returns torch.Tensor)
        feat_tensor = self.feature_extractor.extract_text_features(
            [combined_text],
            template=template
        )

        # Convert to numpy and squeeze
        feat_np = feat_tensor.cpu().numpy().squeeze()

        return feat_np

    def update_proposals_with_combined_feature(
        self,
        proposals: List[Proposal],
        combined_feat: np.ndarray,
        scale_semantic_score: Optional[float] = None,
        apply_softmax: Optional[bool] = None,
    ):
        """
        Update proposal scores using combined feature similarity.

        Following utils_ov_inference.py approach:
        1. Compute cosine similarity for all proposals
        2. Scale by scale_semantic_score (default 300)
        3. Apply softmax across all proposals

        Args:
            proposals: List of proposals to update
            combined_feat: Combined query feature embedding
            scale_semantic_score: Semantic score scaling factor (default 300)
        """
        if len(proposals) == 0:
            return

        # Collect all proposal features
        proposal_feats = np.stack([p.feature for p in proposals])

        # Use compute_cosine_similarity to get scaled + softmax scores
        from ..retrieval import compute_cosine_similarity

        scores = compute_cosine_similarity(
            combined_feat,
            proposal_feats,
            temperature=1.0,  # Not used
            scale_semantic_score=scale_semantic_score if scale_semantic_score is not None else getattr(self, 'scale_semantic_score', 300.0),
            apply_softmax=apply_softmax if apply_softmax is not None else getattr(self, 'apply_softmax', True)
        )

        # Update proposal scores
        for prop, score in zip(proposals, scores):
            prop.score = float(score)

        logger.debug(f"Updated {len(proposals)} proposals with combined feature scores (scale={scale_semantic_score})")

    def retrieve_with_combined_query(
        self,
        object_text: str,
        part_text: str,
        levels: List[str],
        top_k_per_level: Optional[int] = None,
        threshold: Optional[float] = None
    ):
        """
        Retrieve proposals using combined query feature.

        This is a convenience method that:
        1. Creates combined query text
        2. Extracts combined feature
        3. Retrieves from specified levels
        4. Updates scores with combined feature

        Args:
            object_text: Object name
            part_text: Part name
            levels: Levels to retrieve from
            top_k_per_level: Max proposals per level
            threshold: Similarity threshold

        Returns:
            RetrievalResult with proposals scored by combined feature
        """
        # Extract combined feature
        combined_feat = self.extract_combined_feature(object_text, part_text)

        # Retrieve from multiple levels
        result = self.retriever.retrieve_multi_level(
            combined_feat,
            levels=levels,
            top_k_per_level=top_k_per_level,
            threshold=threshold,
            deduplicate=True,
            iou_threshold=0.95
        )

        # Update scores with combined feature
        # (They were already scored during retrieval, but we ensure consistency)
        self.update_proposals_with_combined_feature(
            result.proposals,
            combined_feat
        )

        return result

    def check_combined_query_enabled(self) -> bool:
        """Check if combined query functionality is enabled."""
        return hasattr(self, 'has_combined_query') and self.has_combined_query
