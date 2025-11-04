"""
Multi-instance aware Non-Maximum Suppression (NMS).

This module provides NMS strategies specifically designed for handling
multiple instances of the same class in open vocabulary 3D segmentation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from .geometry import compute_iou, compute_centroid_distance

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prediction with mask, score, and label."""
    mask: np.ndarray  # Binary mask, shape (N_points,)
    score: float
    label_id: int
    metadata: Optional[dict] = None  # Additional info (e.g., object_id, part_id)


def standard_nms(
    predictions: List[Prediction],
    iou_threshold: float = 0.5,
    points: Optional[np.ndarray] = None
) -> List[Prediction]:
    """
    Standard Non-Maximum Suppression.

    Args:
        predictions: List of predictions
        iou_threshold: IoU threshold for suppression
        points: Point cloud coordinates for centroid calculation

    Returns:
        Filtered list of predictions
    """
    if len(predictions) == 0:
        return []

    # Sort by score descending
    sorted_preds = sorted(predictions, key=lambda p: p.score, reverse=True)

    keep = []
    suppressed = set()

    for i, pred in enumerate(sorted_preds):
        if i in suppressed:
            continue

        keep.append(pred)

        # Suppress overlapping predictions
        for j in range(i + 1, len(sorted_preds)):
            if j in suppressed:
                continue

            # Only suppress if same label
            if sorted_preds[j].label_id != pred.label_id:
                continue

            iou = compute_iou(pred.mask, sorted_preds[j].mask)
            if iou > iou_threshold:
                suppressed.add(j)

    logger.info(f"Standard NMS: {len(predictions)} -> {len(keep)} predictions")
    return keep


def multi_instance_nms(
    predictions: List[Prediction],
    iou_threshold: float = 0.35,
    points: Optional[np.ndarray] = None,
    centroid_distance_ratio: float = 0.2,
    use_soft_nms: bool = False,
    soft_nms_sigma: float = 0.5
) -> List[Prediction]:
    """
    Multi-instance aware NMS that preserves spatially separated instances.

    This NMS variant adds a spatial separation check: two predictions with
    high IoU are only suppressed if their centroids are also close together.
    This prevents merging of adjacent instances (e.g., multiple cabinet doors).

    Args:
        predictions: List of predictions
        iou_threshold: IoU threshold for suppression consideration
        points: Point cloud coordinates, shape (N, 3)
        centroid_distance_ratio: Maximum centroid distance as ratio of avg mask size
        use_soft_nms: Whether to use soft NMS (score decay instead of removal)
        soft_nms_sigma: Sigma parameter for soft NMS Gaussian decay

    Returns:
        Filtered list of predictions
    """
    if len(predictions) == 0:
        return []

    # Sort by score descending
    sorted_preds = sorted(predictions, key=lambda p: p.score, reverse=True)

    keep = []
    scores = [p.score for p in sorted_preds]

    for i, pred in enumerate(sorted_preds):
        # Skip if score has been decayed too much (for soft NMS)
        if scores[i] < 0.01:
            continue

        keep.append(pred)

        # Check for suppression/decay of subsequent predictions
        for j in range(i + 1, len(sorted_preds)):
            # Only suppress if same label
            if sorted_preds[j].label_id != pred.label_id:
                continue

            iou = compute_iou(pred.mask, sorted_preds[j].mask)

            if iou > iou_threshold:
                # Additional spatial separation check
                if points is not None and centroid_distance_ratio > 0:
                    centroid_dist = compute_centroid_distance(
                        pred.mask, sorted_preds[j].mask, points
                    )

                    # Compute average mask size in 3D space
                    pred_size = np.sqrt(np.count_nonzero(pred.mask))
                    other_size = np.sqrt(np.count_nonzero(sorted_preds[j].mask))
                    avg_size = (pred_size + other_size) / 2

                    # If centroids are far apart, they are separate instances
                    if centroid_dist > centroid_distance_ratio * avg_size:
                        continue  # Don't suppress

                # Suppress or decay score
                if use_soft_nms:
                    # Soft NMS: decay score using Gaussian
                    decay = np.exp(-(iou ** 2) / soft_nms_sigma)
                    scores[j] *= decay
                    sorted_preds[j].score = scores[j]
                else:
                    # Hard NMS: set score to 0 (will be skipped)
                    scores[j] = 0.0

    # Filter out heavily decayed predictions
    if use_soft_nms:
        keep = [p for p in keep if p.score >= 0.01]

    logger.info(f"Multi-instance NMS: {len(predictions)} -> {len(keep)} predictions")
    return keep


def class_agnostic_nms(
    predictions: List[Prediction],
    iou_threshold: float = 0.5,
    points: Optional[np.ndarray] = None
) -> List[Prediction]:
    """
    Class-agnostic NMS: suppress overlapping predictions regardless of label.

    Useful for removing overlapping proposals when labels might be uncertain.

    Args:
        predictions: List of predictions
        iou_threshold: IoU threshold
        points: Point cloud coordinates

    Returns:
        Filtered predictions
    """
    if len(predictions) == 0:
        return []

    sorted_preds = sorted(predictions, key=lambda p: p.score, reverse=True)
    keep = []
    suppressed = set()

    for i, pred in enumerate(sorted_preds):
        if i in suppressed:
            continue

        keep.append(pred)

        for j in range(i + 1, len(sorted_preds)):
            if j in suppressed:
                continue

            iou = compute_iou(pred.mask, sorted_preds[j].mask)
            if iou > iou_threshold:
                suppressed.add(j)

    logger.info(f"Class-agnostic NMS: {len(predictions)} -> {len(keep)} predictions")
    return keep


def nms_with_instance_center_check(
    predictions: List[Prediction],
    iou_threshold: float = 0.35,
    points: Optional[np.ndarray] = None,
    min_centroid_distance: float = 0.1
) -> List[Prediction]:
    """
    NMS with explicit instance center distance check.

    Args:
        predictions: List of predictions
        iou_threshold: IoU threshold
        points: Point cloud coordinates in meters
        min_centroid_distance: Minimum centroid distance in meters

    Returns:
        Filtered predictions
    """
    if len(predictions) == 0:
        return []

    if points is None:
        logger.warning("Points not provided, falling back to standard NMS")
        return standard_nms(predictions, iou_threshold)

    sorted_preds = sorted(predictions, key=lambda p: p.score, reverse=True)
    keep = []
    suppressed = set()

    for i, pred in enumerate(sorted_preds):
        if i in suppressed:
            continue

        keep.append(pred)

        for j in range(i + 1, len(sorted_preds)):
            if j in suppressed:
                continue

            # Only suppress same label
            if sorted_preds[j].label_id != pred.label_id:
                continue

            iou = compute_iou(pred.mask, sorted_preds[j].mask)

            if iou > iou_threshold:
                # Check centroid distance
                centroid_dist = compute_centroid_distance(
                    pred.mask, sorted_preds[j].mask, points
                )

                if centroid_dist < min_centroid_distance:
                    # Close together and high IoU -> suppress
                    suppressed.add(j)
                # else: spatially separated -> keep both

    logger.info(f"Instance-center NMS: {len(predictions)} -> {len(keep)} predictions")
    return keep
