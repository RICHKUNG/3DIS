"""
Geometric constraint checking for Object-Part pairing.

Provides functions to validate spatial relationships between proposals,
including containment, proximity, and scale consistency checks.
"""

import numpy as np
from typing import Tuple, Optional


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.

    Args:
        mask1: Binary mask array, shape (N,)
        mask2: Binary mask array, shape (N,)

    Returns:
        IoU value in [0, 1]
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.count_nonzero(mask1_bool & mask2_bool)
    union = np.count_nonzero(mask1_bool | mask2_bool)

    if union == 0:
        return 0.0

    return intersection / union


def compute_containment_ratio(part_mask: np.ndarray, object_mask: np.ndarray) -> float:
    """
    Compute containment ratio: how much of the part is contained in the object.

    Args:
        part_mask: Binary mask of the part proposal
        object_mask: Binary mask of the object proposal

    Returns:
        Ratio in [0, 1], where 1.0 means part is fully contained in object
    """
    part_bool = part_mask.astype(bool)
    object_bool = object_mask.astype(bool)

    part_size = np.count_nonzero(part_bool)
    if part_size == 0:
        return 0.0

    intersection = np.count_nonzero(part_bool & object_bool)
    return intersection / part_size


def compute_centroid(mask: np.ndarray, points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute centroid of a mask.

    Args:
        mask: Binary mask array, shape (N,)
        points: Point cloud coordinates, shape (N, 3). If None, uses indices.

    Returns:
        Centroid coordinates, shape (3,) or (1,) if points not provided
    """
    mask_bool = mask.astype(bool)

    if points is not None:
        masked_points = points[mask_bool]
        if len(masked_points) == 0:
            return np.zeros(3)
        return np.mean(masked_points, axis=0)
    else:
        # Use indices as 1D coordinates
        indices = np.where(mask_bool)[0]
        if len(indices) == 0:
            return np.array([0.0])
        return np.array([np.mean(indices)])


def compute_centroid_distance(mask1: np.ndarray, mask2: np.ndarray,
                              points: Optional[np.ndarray] = None) -> float:
    """
    Compute distance between centroids of two masks.

    Args:
        mask1: Binary mask array
        mask2: Binary mask array
        points: Point cloud coordinates, shape (N, 3)

    Returns:
        Euclidean distance between centroids
    """
    c1 = compute_centroid(mask1, points)
    c2 = compute_centroid(mask2, points)
    return float(np.linalg.norm(c1 - c2))


def compute_scale_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute size ratio between two masks.

    Args:
        mask1: Binary mask array
        mask2: Binary mask array

    Returns:
        Ratio of mask1 size to mask2 size
    """
    size1 = np.count_nonzero(mask1)
    size2 = np.count_nonzero(mask2)

    if size2 == 0:
        return float('inf')

    return size1 / size2


def check_geometric_validity(
    part_mask: np.ndarray,
    object_mask: np.ndarray,
    points: Optional[np.ndarray] = None,
    containment_threshold: float = 0.5,
    scale_range: Tuple[float, float] = (0.01, 0.9),
    max_centroid_distance: Optional[float] = None
) -> Tuple[bool, dict]:
    """
    Check if a part-object pair satisfies geometric constraints.

    Args:
        part_mask: Binary mask of part proposal
        object_mask: Binary mask of object proposal
        points: Point cloud coordinates for centroid calculation
        containment_threshold: Minimum containment ratio required
        scale_range: (min, max) acceptable size ratio (part/object)
        max_centroid_distance: Maximum centroid distance (in same units as points)

    Returns:
        (is_valid, metrics) where metrics contains:
            - containment_ratio
            - scale_ratio
            - centroid_distance
            - iou
    """
    metrics = {}

    # Compute metrics
    metrics['containment_ratio'] = compute_containment_ratio(part_mask, object_mask)
    metrics['scale_ratio'] = compute_scale_ratio(part_mask, object_mask)
    metrics['iou'] = compute_iou(part_mask, object_mask)
    metrics['centroid_distance'] = compute_centroid_distance(part_mask, object_mask, points)

    # Check constraints
    is_valid = True

    # Containment check
    if metrics['containment_ratio'] < containment_threshold:
        is_valid = False

    # Scale check
    if not (scale_range[0] <= metrics['scale_ratio'] <= scale_range[1]):
        is_valid = False

    # Centroid distance check
    if max_centroid_distance is not None:
        if metrics['centroid_distance'] > max_centroid_distance:
            is_valid = False

    return is_valid, metrics


def compute_geometric_score(
    part_mask: np.ndarray,
    object_mask: np.ndarray,
    points: Optional[np.ndarray] = None,
    weights: Optional[dict] = None
) -> float:
    """
    Compute a geometric compatibility score for a part-object pair.

    Args:
        part_mask: Binary mask of part proposal
        object_mask: Binary mask of object proposal
        points: Point cloud coordinates
        weights: Dictionary with keys 'containment', 'iou', 'compactness'

    Returns:
        Geometric score in [0, 1]
    """
    if weights is None:
        weights = {'containment': 0.6, 'iou': 0.3, 'compactness': 0.1}

    # Containment score
    containment = compute_containment_ratio(part_mask, object_mask)

    # IoU score
    iou = compute_iou(part_mask, object_mask)

    # Compactness score (penalize very small parts relative to object)
    scale = compute_scale_ratio(part_mask, object_mask)
    compactness = 1.0 / (1.0 + np.exp(-10 * (scale - 0.05)))  # Sigmoid centered at 5% size

    # Weighted combination
    score = (
        weights.get('containment', 0.6) * containment +
        weights.get('iou', 0.3) * iou +
        weights.get('compactness', 0.1) * compactness
    )

    return float(np.clip(score, 0.0, 1.0))
