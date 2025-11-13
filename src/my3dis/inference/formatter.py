"""
Output formatter for Search3D evaluation.

Converts internal predictions to the format required by Search3D evaluation:
- pred_classes: numpy array (M,) of label IDs
- pred_scores: numpy array (M,) of confidence scores
- pred_masks: numpy array (P, M) of binary masks

Where P = number of points, M = number of predictions.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .nms import Prediction
from .pairing import ObjectPartPair

logger = logging.getLogger(__name__)


class Search3DFormatter:
    """
    Formats predictions for Search3D evaluation.

    The Search3D evaluation expects:
    ```python
    preds = {
        "scene_name": {
            "pred_classes": np.array([label1, label2, ...], dtype=np.int32),
            "pred_scores": np.array([score1, score2, ...], dtype=float),
            "pred_masks": np.array([[mask1], [mask2], ...])  # shape (P, M)
        }
    }
    ```
    """

    def __init__(
        self,
        num_points: int,
        use_part_mask: bool = True,
        min_mask_size: int = 10
    ):
        """
        Initialize formatter.

        Args:
            num_points: Total number of points in the scene
            use_part_mask: Whether to use part mask (True) or object mask (False)
            min_mask_size: Minimum number of points for a valid mask
        """
        self.num_points = num_points
        self.use_part_mask = use_part_mask
        self.min_mask_size = min_mask_size

    def format_predictions(
        self,
        predictions: List[Prediction]
    ) -> Dict[str, np.ndarray]:
        """
        Format predictions for a single scene.

        Args:
            predictions: List of predictions

        Returns:
            Dictionary with 'pred_classes', 'pred_scores', 'pred_masks'
        """
        if len(predictions) == 0:
            logger.warning("No predictions to format")
            return {
                'pred_classes': np.array([], dtype=np.int32),
                'pred_scores': np.array([], dtype=float),
                'pred_masks': np.zeros((self.num_points, 0), dtype=np.uint8)
            }

        # Filter out small masks
        valid_predictions = [
            p for p in predictions
            if np.count_nonzero(p.mask) >= self.min_mask_size
        ]

        if len(valid_predictions) < len(predictions):
            logger.info(f"Filtered {len(predictions) - len(valid_predictions)} "
                       f"predictions with mask size < {self.min_mask_size}")

        if len(valid_predictions) == 0:
            logger.warning("All predictions filtered out due to small mask size")
            return {
                'pred_classes': np.array([], dtype=np.int32),
                'pred_scores': np.array([], dtype=float),
                'pred_masks': np.zeros((self.num_points, 0), dtype=np.uint8)
            }

        # Extract arrays
        pred_classes = np.array([p.label_id for p in valid_predictions], dtype=np.int32)
        pred_scores = np.array([p.score for p in valid_predictions], dtype=float)

        # Stack masks: shape (P, M)
        masks = []
        for pred in valid_predictions:
            mask = pred.mask
            # Ensure mask has correct length
            if len(mask) != self.num_points:
                logger.warning(f"Mask length {len(mask)} != num_points {self.num_points}, padding/truncating")
                if len(mask) < self.num_points:
                    # Pad
                    mask = np.pad(mask, (0, self.num_points - len(mask)))
                else:
                    # Truncate
                    mask = mask[:self.num_points]

            masks.append(mask)

        pred_masks = np.stack(masks, axis=1)  # Shape: (P, M)

        logger.info(f"Formatted {len(valid_predictions)} predictions "
                   f"(classes: {pred_classes.shape}, scores: {pred_scores.shape}, "
                   f"masks: {pred_masks.shape})")

        return {
            'pred_classes': pred_classes,
            'pred_scores': pred_scores,
            'pred_masks': pred_masks
        }

    def format_pairs(
        self,
        pairs: List[ObjectPartPair]
    ) -> Dict[str, np.ndarray]:
        """
        Format object-part pairs for evaluation.

        Args:
            pairs: List of object-part pairs

        Returns:
            Dictionary with 'pred_classes', 'pred_scores', 'pred_masks'
        """
        # Convert pairs to predictions
        predictions = []

        for pair in pairs:
            # Choose mask for primary output
            if self.use_part_mask:
                mask = pair.part_proposal.mask
            else:
                # Use union of object and part
                mask = np.logical_or(
                    pair.object_proposal.mask,
                    pair.part_proposal.mask
                ).astype(np.uint8)

            # Store both object and part masks in metadata for Search3D export
            metadata = {
                'object_mask': pair.object_proposal.mask,
                'part_mask': pair.part_proposal.mask,
                'object_id': pair.object_proposal.object_id,
                'part_id': pair.part_proposal.object_id,
                'object_level': pair.object_proposal.level,
                'part_level': pair.part_proposal.level,
                'object_score': pair.object_score,
                'part_score': pair.part_score,
            }

            pred = Prediction(
                mask=mask,
                score=pair.combined_score,
                label_id=pair.label_id,
                metadata=metadata
            )
            predictions.append(pred)

        return self.format_predictions(predictions)

    def format_multi_scene(
        self,
        predictions_by_scene: Dict[str, List[Prediction]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Format predictions for multiple scenes.

        Args:
            predictions_by_scene: Dict mapping scene names to prediction lists

        Returns:
            Dict mapping scene names to prediction dicts
        """
        formatted = {}

        for scene_name, predictions in predictions_by_scene.items():
            logger.info(f"Formatting scene: {scene_name}")
            formatted[scene_name] = self.format_predictions(predictions)

        return formatted

    @staticmethod
    def save_predictions(
        preds: Dict[str, Dict[str, np.ndarray]],
        output_path: str
    ):
        """
        Save predictions to file.

        Args:
            preds: Predictions dict (multi-scene)
            output_path: Output file path (.npz)
        """
        np.savez(output_path, **{
            scene: pred_dict for scene, pred_dict in preds.items()
        })
        logger.info(f"Saved predictions to {output_path}")

    @staticmethod
    def load_predictions(input_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load predictions from file.

        Args:
            input_path: Input file path (.npz)

        Returns:
            Predictions dict
        """
        data = np.load(input_path, allow_pickle=True)

        preds = {}
        for scene in data.files:
            scene_data = data[scene].item()
            preds[scene] = scene_data

        logger.info(f"Loaded predictions from {input_path} for {len(preds)} scenes")
        return preds


def validate_prediction_format(pred: Dict[str, np.ndarray]) -> bool:
    """
    Validate that a prediction dict has the correct format.

    Args:
        pred: Prediction dict

    Returns:
        True if valid
    """
    required_keys = {'pred_classes', 'pred_scores', 'pred_masks'}

    if not all(key in pred for key in required_keys):
        logger.error(f"Missing required keys. Expected {required_keys}, got {set(pred.keys())}")
        return False

    classes = pred['pred_classes']
    scores = pred['pred_scores']
    masks = pred['pred_masks']

    # Check types
    if not isinstance(classes, np.ndarray) or classes.dtype != np.int32:
        logger.error(f"pred_classes must be np.int32 array, got {type(classes)}")
        return False

    if not isinstance(scores, np.ndarray):
        logger.error(f"pred_scores must be numpy array, got {type(scores)}")
        return False

    if not isinstance(masks, np.ndarray):
        logger.error(f"pred_masks must be numpy array, got {type(masks)}")
        return False

    # Check shapes
    M = len(classes)
    if len(scores) != M:
        logger.error(f"pred_scores length {len(scores)} != pred_classes length {M}")
        return False

    if masks.ndim != 2:
        logger.error(f"pred_masks must be 2D, got shape {masks.shape}")
        return False

    if masks.shape[1] != M:
        logger.error(f"pred_masks shape[1] {masks.shape[1]} != M {M}")
        return False

    logger.info(f"Prediction format validated: M={M}, P={masks.shape[0]}")
    return True
