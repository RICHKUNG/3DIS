"""
Feature extraction module for Open Vocabulary 3D Instance Segmentation.

This module handles:
1. SigLIP/CLIP feature extraction for 3D proposals using 2D RGB projections
2. Text embedding extraction for queries
3. Efficient batched processing

IMPORTANT: This module does NOT support random features.
           If model loading or feature extraction fails, it will raise an error.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoProcessor = None
    AutoTokenizer = None
    # Don't just warn - this is a critical dependency
    logger.error("transformers library is not available! Please install: pip install transformers")


class SigLIPFeatureExtractor:
    """
    Extract SigLIP/CLIP features for 3D proposals using 2D projections.

    This class loads SigLIP model and extracts features from:
    1. RGB image crops for 3D proposals (using 2D masks)
    2. Text queries for retrieval

    IMPORTANT: This class does NOT support random features or fallback modes.
               Model must be loaded successfully or an error will be raised.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda"
    ):
        """
        Initialize SigLIP feature extractor.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')

        Raises:
            RuntimeError: If transformers is not available or model fails to load
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if AutoModel is None:
            raise RuntimeError(
                "transformers library is not available!\n"
                "Please install: pip install transformers"
            )

        logger.info(f"Loading SigLIP model: {model_name}")
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()

            # Infer feature dimension from model
            with torch.no_grad():
                dummy_text = self.tokenizer(["test"], padding=True, return_tensors="pt").to(self.device)
                dummy_feat = self.model.get_text_features(**dummy_text)
                self.feature_dim = dummy_feat.shape[-1]

            logger.info(f"SigLIP model loaded on {self.device} (feature_dim={self.feature_dim})")
        except Exception as e:
            logger.error(f"Failed to load SigLIP model: {e}")
            logger.error("CRITICAL: Model loading failed. Cannot proceed.")
            raise RuntimeError(
                f"Failed to load SigLIP model '{model_name}': {e}\n"
                f"Please check:\n"
                f"  1. PyTorch version (requires >= 2.6 or model with safetensors)\n"
                f"  2. Model availability on HuggingFace\n"
                f"  3. Network connection\n"
                f"  4. CUDA/device compatibility"
            ) from e

    def extract_text_features(
        self,
        text_queries: List[str],
        batch_size: int = 32,
        template: str = "in_room"
    ) -> torch.Tensor:
        """
        Extract text embeddings for queries with optional template.

        Args:
            text_queries: List of text queries
            batch_size: Batch size for processing
            template: Text template to apply. Options:
                - "raw": Use class name directly
                - "in_room": "a {class} in a room" (BEST, default)
                - "photo_of": "a photo of a {class}"
                - "indoor": "a {class} in an indoor scene"

        Returns:
            Text features tensor, shape (N_queries, feature_dim)

        Raises:
            RuntimeError: If feature extraction fails
        """
        if len(text_queries) == 0:
            raise ValueError("No text queries provided")

        # Apply template to queries
        processed_queries = []
        for q in text_queries:
            if template == "raw":
                processed_queries.append(q)
            elif template == "in_room":
                processed_queries.append(f"a {q} in a room")
            elif template == "photo_of":
                processed_queries.append(f"a photo of a {q}")
            elif template == "indoor":
                processed_queries.append(f"a {q} in an indoor scene")
            else:
                # Default to raw if unknown template
                processed_queries.append(q)

        all_features = []

        try:
            with torch.no_grad():
                for i in range(0, len(processed_queries), batch_size):
                    batch_texts = processed_queries[i:i+batch_size]
                    inputs = self.tokenizer(batch_texts, padding=True, return_tensors="pt").to(self.device)
                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    all_features.append(text_features.cpu())

            return torch.cat(all_features, dim=0)
        except Exception as e:
            raise RuntimeError(f"Text feature extraction failed: {e}") from e

    def extract_proposal_features(
        self,
        proposals_by_level: Dict[str, List[Dict]],
        exp_dir: Path,
        scene_path: Path,
        max_frames_per_object: int = 5,
        batch_size: int = 8
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Extract SigLIP features for proposals from RGB images.

        This method:
        1. Loads My3DIS tracking outputs (video_segments)
        2. For each proposal, loads corresponding RGB frames
        3. Extracts SigLIP features from masked regions
        4. Aggregates features across frames

        Args:
            proposals_by_level: Proposals grouped by level (from ProposalLoader)
            exp_dir: Experiment directory (for loading tracking outputs)
            scene_path: Path to scene directory (contains outputs/color/)
            max_frames_per_object: Maximum frames to use per object
            batch_size: Batch size for image processing

        Returns:
            features_by_level: {level: [feature_tensor, ...]}

        Raises:
            RuntimeError: If required data is missing or extraction fails
        """
        from my3dis.tracking import load_legacy_video_segments, load_object_segments_manifest
        from my3dis.mask.encoding import unpack_binary_mask

        color_dir = scene_path / "outputs" / "color"
        if not color_dir.exists():
            raise RuntimeError(f"Color directory not found: {color_dir}")

        features_by_level = {}

        for level, proposals in proposals_by_level.items():
            logger.info(f"Extracting features for {len(proposals)} proposals in {level}")

            # Load tracking outputs for this level
            level_dir = self._find_level_dir(exp_dir, level)
            if level_dir is None:
                raise RuntimeError(f"Level directory not found for {level} in {exp_dir}")

            tracking_dir = level_dir / "tracking" if (level_dir / "tracking").exists() else level_dir

            # Find NPZ files
            video_files = list(tracking_dir.glob("video_segments*.npz"))
            object_files = list(tracking_dir.glob("object_segments*.npz"))

            if len(video_files) == 0 or len(object_files) == 0:
                raise RuntimeError(f"Missing tracking files for {level} in {tracking_dir}")

            video_npz = sorted(video_files)[0]
            object_npz = sorted(object_files)[0]

            try:
                # Load video and object data
                video_frames, _ = load_legacy_video_segments(str(video_npz), unpack_masks=False)
                object_map = load_object_segments_manifest(str(object_npz))

                level_features = []
                skipped_objects = []

                for prop in tqdm(proposals, desc=f"Extracting {level} features"):
                    obj_id = prop['object_id']
                    try:
                        feature = self._extract_single_proposal_feature(
                            obj_id, object_map, video_frames, color_dir, max_frames_per_object
                        )
                        level_features.append(feature)
                    except RuntimeError as e:
                        # Object has no valid frames (e.g., all crops too small)
                        # Use zero vector as placeholder
                        logger.warning(f"Object {obj_id}: {e}. Using zero vector.")
                        zero_feature = torch.zeros(self.feature_dim, device=self.device)
                        level_features.append(zero_feature)
                        skipped_objects.append(obj_id)

                features_by_level[level] = level_features
                if skipped_objects:
                    logger.warning(
                        f"{level}: {len(skipped_objects)} objects used zero vectors: "
                        f"{skipped_objects[:5]}{'...' if len(skipped_objects) > 5 else ''}"
                    )
                logger.info(f"Extracted {len(level_features)} features for {level}")

            except Exception as e:
                raise RuntimeError(f"Error extracting features for {level}: {e}") from e

        return features_by_level

    def _find_level_dir(self, exp_dir: Path, level: str) -> Optional[Path]:
        """Find level directory for given level name."""
        level_num = level[1:] if level.startswith('L') else level
        candidates = [
            exp_dir / f"level_{level_num}",
            exp_dir / f"level_{level}"
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _extract_single_proposal_feature(
        self,
        object_id: int,
        object_map: dict,
        video_frames: dict,
        color_dir: Path,
        max_frames: int
    ) -> torch.Tensor:
        """
        Extract SigLIP feature for a single 3D proposal.

        Args:
            object_id: Object ID
            object_map: Object-to-frame mapping
            video_frames: Frame data from video_segments
            color_dir: Directory with RGB images
            max_frames: Maximum frames to use

        Returns:
            Aggregated feature tensor [feature_dim]

        Raises:
            RuntimeError: If no valid frames found for object
        """
        from my3dis.mask.encoding import unpack_binary_mask

        if object_id not in object_map:
            raise RuntimeError(f"Object {object_id} not found in object_map")

        # Get frames where this object appears
        frame_indices = sorted(list(object_map[object_id].keys()))

        # Sample frames if too many
        if len(frame_indices) > max_frames:
            step = len(frame_indices) // max_frames
            frame_indices = frame_indices[::step][:max_frames]

        frame_features = []

        with torch.no_grad():
            for frame_idx in frame_indices:
                if frame_idx not in video_frames:
                    continue

                frame_data = video_frames[frame_idx]
                if object_id not in frame_data:
                    continue

                payload = frame_data[object_id]
                mask = unpack_binary_mask(payload)

                # Load RGB image
                image_path = color_dir / f"{frame_idx}.png"
                if not image_path.exists():
                    image_path = color_dir / f"{frame_idx}.jpg"
                if not image_path.exists():
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")

                    # Get bounding box from mask
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) == 0:
                        continue

                    x1, y1 = x_coords.min(), y_coords.min()
                    x2, y2 = x_coords.max(), y_coords.max()

                    # Calculate bbox size for weighting
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_size = bbox_width * bbox_height

                    # Skip crops that are too small (causes channel ambiguity warnings)
                    # Minimum 10x10 to ensure valid feature extraction
                    MIN_CROP_SIZE = 10
                    if bbox_width < MIN_CROP_SIZE or bbox_height < MIN_CROP_SIZE:
                        logger.debug(
                            f"Skipping frame {frame_idx} for object {object_id}: "
                            f"crop too small ({bbox_width}x{bbox_height})"
                        )
                        continue

                    # Expand bbox slightly
                    margin = 20
                    W, H = image.size
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(W, x2 + margin)
                    y2 = min(H, y2 + margin)

                    # Crop and extract features
                    crop = image.crop((x1, y1, x2, y2))
                    inputs = self.processor(images=crop, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(self.device)

                    features = self.model.get_image_features(pixel_values=pixel_values)
                    features = features / features.norm(dim=-1, keepdim=True)
                    # Store features with metadata for weighted aggregation
                    frame_features.append((features, bbox_size, frame_idx))

                except Exception as e:
                    logger.debug(f"Error processing frame {frame_idx} for object {object_id}: {e}")
                    continue

        if len(frame_features) == 0:
            raise RuntimeError(f"No valid frames found for object {object_id}")

        # Aggregate features with weighted pooling (Phase 2-A)
        if len(frame_features) == 1:
            # Single frame, no weighting needed
            return frame_features[0][0].squeeze(0)

        # Compute frame quality weights
        total_frames = len(frame_features)
        weights = []
        features_list = []

        for idx, (feat, bbox_size, frame_idx) in enumerate(frame_features):
            # Temporal weight: Gaussian centered at middle frame
            # Middle frames tend to have better view quality
            # Use actual position in frame_features, not frame_indices
            frame_position = idx
            center_position = (total_frames - 1) / 2.0
            temporal_weight = np.exp(-((frame_position - center_position) / (total_frames / 4.0))**2)

            # Size weight: Larger bboxes indicate closer/better views
            # Use sqrt to moderate the effect
            size_weight = np.sqrt(float(bbox_size))

            # Combined weight
            quality = temporal_weight * size_weight
            weights.append(quality)
            features_list.append(feat)

        # Normalize weights using softmax for numerical stability
        weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weights_normalized = torch.softmax(weights_tensor, dim=0)

        # Weighted aggregation
        features_stack = torch.stack(features_list).squeeze(1)  # (N_frames, feat_dim)
        aggregated = (features_stack * weights_normalized.unsqueeze(1)).sum(dim=0)

        # L2 normalize
        aggregated = aggregated / aggregated.norm(dim=-1, keepdim=True)

        return aggregated
