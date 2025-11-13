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
        skip_model_load: bool = False,
        feature_extractor = None
    ):
        """
        Initialize combined query functionality.

        Args:
            siglip_model: SigLIP model name
            device: Device for feature extraction
            skip_model_load: Skip model loading (for testing)
            feature_extractor: Pre-initialized feature extractor (optional)
        """
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            self.has_combined_query = True
            logger.info("Using provided feature extractor for combined queries")
        else:
            from ..feature_extraction import SigLIPFeatureExtractor

            logger.info(f"Initializing SigLIP for combined queries: {siglip_model}")
            self.feature_extractor = SigLIPFeatureExtractor(
                model_name=siglip_model,
                device=device,
                skip_model_load=skip_model_load
            )
            self.has_combined_query = True

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
        logger.debug(f"Extracting feature for: '{combined_text}'")

        # Extract features (returns torch.Tensor)
        feat_tensor = self.feature_extractor.extract_text_features([combined_text])

        # Convert to numpy and squeeze
        feat_np = feat_tensor.cpu().numpy().squeeze()

        return feat_np

    def update_proposals_with_combined_feature(
        self,
        proposals: List[Proposal],
        combined_feat: np.ndarray
    ):
        """
        Update proposal scores using combined feature similarity.

        This computes cosine similarity between each proposal's feature
        and the combined query feature, then updates the proposal's score.

        Args:
            proposals: List of proposals to update
            combined_feat: Combined query feature embedding
        """
        if len(proposals) == 0:
            return

        # Normalize combined feature
        combined_norm = combined_feat / (np.linalg.norm(combined_feat) + 1e-8)

        for prop in proposals:
            # Normalize proposal feature
            prop_norm = prop.feature / (np.linalg.norm(prop.feature) + 1e-8)

            # Compute cosine similarity
            similarity = float(np.dot(prop_norm, combined_norm))

            # Update score
            prop.score = similarity

        logger.debug(f"Updated {len(proposals)} proposals with combined feature scores")

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
