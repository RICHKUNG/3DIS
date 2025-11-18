"""
Feature extraction module for Open Vocabulary 3D Instance Segmentation.

This module handles:
1. SigLIP feature extraction for 3D proposals using 2D RGB projections
2. Text embedding extraction for queries
3. Efficient batched processing
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
    logger.warning("transformers not available - feature extraction will use random features")


class SigLIPFeatureExtractor:
    """
    Extract SigLIP features for 3D proposals using 2D projections.

    This class loads SigLIP model and extracts features from:
    1. RGB image crops for 3D proposals (using 2D masks)
    2. Text queries for retrieval
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        skip_model_load: bool = False
    ):
        """
        Initialize SigLIP feature extractor.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
            skip_model_load: Skip model loading (for testing, uses random features)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.skip_model_load = skip_model_load
        self.model_name = model_name

        if skip_model_load or AutoModel is None:
            logger.warning("Skipping SigLIP model loading (will use random features)")
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.feature_dim = 1152  # Default for siglip-so400m
        else:
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
                logger.warning("Falling back to random features")
                self.model = None
                self.processor = None
                self.tokenizer = None
                self.skip_model_load = True
                self.feature_dim = 1152

    def extract_text_features(
        self,
        text_queries: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Extract text embeddings for queries.

        Args:
            text_queries: List of text queries
            batch_size: Batch size for processing

        Returns:
            Text features tensor, shape (N_queries, feature_dim)
        """
        if self.skip_model_load or self.model is None:
            # Return random features
            features = torch.randn(len(text_queries), self.feature_dim)
            return features / features.norm(dim=-1, keepdim=True)

        all_features = []

        with torch.no_grad():
            for i in range(0, len(text_queries), batch_size):
                batch_texts = text_queries[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, return_tensors="pt").to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_features.append(text_features.cpu())

        return torch.cat(all_features, dim=0)

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
        """
        if self.skip_model_load or self.model is None:
            logger.warning("SigLIP model not loaded - using random features")
            return self._extract_random_features(proposals_by_level)

        from my3dis.tracking import load_legacy_video_segments, load_object_segments_manifest
        from my3dis.mask.encoding import unpack_binary_mask

        color_dir = scene_path / "outputs" / "color"
        if not color_dir.exists():
            logger.error(f"Color directory not found: {color_dir}")
            return self._extract_random_features(proposals_by_level)

        features_by_level = {}

        for level, proposals in proposals_by_level.items():
            logger.info(f"Extracting features for {len(proposals)} proposals in {level}")

            # Load tracking outputs for this level
            level_dir = self._find_level_dir(exp_dir, level)
            if level_dir is None:
                logger.warning(f"Level directory not found for {level}, using random features")
                features_by_level[level] = self._extract_random_features_for_level(proposals)
                continue

            tracking_dir = level_dir / "tracking" if (level_dir / "tracking").exists() else level_dir

            # Find NPZ files
            video_files = list(tracking_dir.glob("video_segments*.npz"))
            object_files = list(tracking_dir.glob("object_segments*.npz"))

            if len(video_files) == 0 or len(object_files) == 0:
                logger.warning(f"Missing tracking files for {level}, using random features")
                features_by_level[level] = self._extract_random_features_for_level(proposals)
                continue

            video_npz = sorted(video_files)[0]
            object_npz = sorted(object_files)[0]

            try:
                # Load video and object data
                video_frames, _ = load_legacy_video_segments(str(video_npz), unpack_masks=False)
                object_map = load_object_segments_manifest(str(object_npz))

                level_features = []

                for prop in tqdm(proposals, desc=f"Extracting {level} features"):
                    obj_id = prop['object_id']
                    feature = self._extract_single_proposal_feature(
                        obj_id, object_map, video_frames, color_dir, max_frames_per_object
                    )
                    level_features.append(feature)

                features_by_level[level] = level_features
                logger.info(f"Extracted {len(level_features)} features for {level}")

            except Exception as e:
                logger.error(f"Error loading tracking data for {level}: {e}")
                features_by_level[level] = self._extract_random_features_for_level(proposals)

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
        """
        from my3dis.mask.encoding import unpack_binary_mask

        if object_id not in object_map:
            # Return random feature if object not found
            feat = torch.randn(self.feature_dim)
            return feat / feat.norm()

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
                    frame_features.append(features)

                except Exception as e:
                    logger.debug(f"Error processing frame {frame_idx} for object {object_id}: {e}")
                    continue

        if len(frame_features) == 0:
            # Fallback to random if no valid frames
            feat = torch.randn(self.feature_dim)
            return feat / feat.norm()

        # Aggregate features across frames (mean pooling)
        aggregated = torch.stack(frame_features).mean(dim=0)
        aggregated = aggregated / aggregated.norm(dim=-1, keepdim=True)

        return aggregated.squeeze(0)

    def _extract_random_features(self, proposals_by_level: Dict[str, List[Dict]]) -> Dict[str, List[torch.Tensor]]:
        """Extract random features as fallback."""
        features_by_level = {}
        for level, proposals in proposals_by_level.items():
            features_by_level[level] = self._extract_random_features_for_level(proposals)
        return features_by_level

    def _extract_random_features_for_level(self, proposals: List[Dict]) -> List[torch.Tensor]:
        """Extract random features for a single level."""
        features = []
        for _ in proposals:
            feat = torch.randn(self.feature_dim)
            feat = feat / feat.norm()
            features.append(feat)
        return features
