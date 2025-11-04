"""
Utility functions for 3D mask aggregation.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def aggregate_masks_3d(masks: List[np.ndarray], method: str = 'union') -> np.ndarray:
    """
    Aggregate multiple 3D masks into single mask.

    Args:
        masks: List of binary 3D masks [N_points]
        method: Aggregation method ('union', 'intersection', 'majority')

    Returns:
        Aggregated binary mask [N_points]
    """
    if len(masks) == 0:
        raise ValueError("No masks to aggregate")

    if len(masks) == 1:
        return masks[0]

    masks_stack = np.stack(masks, axis=0)  # [N_masks, N_points]

    if method == 'union':
        return masks_stack.any(axis=0)
    elif method == 'intersection':
        return masks_stack.all(axis=0)
    elif method == 'majority':
        votes = masks_stack.sum(axis=0)
        return votes >= (len(masks) / 2)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_proposal_geometry(
    mask: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
) -> Dict:
    """
    Compute geometric properties of a 3D proposal.

    Args:
        mask: Binary mask [N_points]
        points: Point cloud coordinates [N_points, 3]
        colors: Point cloud colors [N_points, 3]

    Returns:
        Dictionary with geometric properties:
        - centroid: [3] center of mass
        - bbox: [6] axis-aligned bounding box (min_x, min_y, min_z, max_x, max_y, max_z)
        - volume: Approximate volume (m^3)
        - surface_area: Approximate surface area (m^2)
    """
    if mask.sum() == 0:
        return {
            'centroid': np.zeros(3),
            'bbox': np.zeros(6),
            'volume': 0.0,
            'surface_area': 0.0,
        }

    masked_points = points[mask]

    # Centroid
    centroid = masked_points.mean(axis=0)

    # Bounding box
    min_coords = masked_points.min(axis=0)
    max_coords = masked_points.max(axis=0)
    bbox = np.concatenate([min_coords, max_coords])

    # Volume (approximate using convex hull or bounding box)
    dims = max_coords - min_coords
    volume = dims.prod()  # Simple bbox volume

    # Surface area (approximate)
    surface_area = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])

    return {
        'centroid': centroid,
        'bbox': bbox,
        'volume': float(volume),
        'surface_area': float(surface_area),
    }


def filter_proposals_by_volume(
    proposals: List[Dict],
    min_volume: float,
) -> List[Dict]:
    """
    Filter proposals by minimum volume threshold.

    Args:
        proposals: List of proposal dictionaries
        min_volume: Minimum volume (m^3)

    Returns:
        Filtered list of proposals
    """
    filtered = [p for p in proposals if p.get('volume', 0.0) >= min_volume]

    logger.info(f"Volume filtering: {len(proposals)} -> {len(filtered)} proposals")

    return filtered


def compute_iou_3d(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute 3D IoU between two masks.

    Args:
        mask1: Binary mask [N_points]
        mask2: Binary mask [N_points]

    Returns:
        IoU score (0 to 1)
    """
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)
