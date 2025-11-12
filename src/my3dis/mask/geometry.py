"""Geometric operations on masks.

This module provides utilities for extracting bounding boxes and other
geometric properties from binary masks.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def bbox_from_mask_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Extract bounding box from a binary mask in xyxy format.

    Args:
        mask: Boolean mask array

    Returns:
        Tuple of (x_min, y_min, x_max, y_max). Returns (0, 0, 0, 0) for empty masks.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_xyxy_to_xywh(bounds: Sequence[int]) -> List[int]:
    """Convert bounding box from xyxy to xywh format.

    Args:
        bounds: Bounding box in (x_min, y_min, x_max, y_max) format

    Returns:
        Bounding box in (x, y, width, height) format
    """
    x1, y1, x2, y2 = bounds
    return [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]
