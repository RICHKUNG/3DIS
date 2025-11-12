"""Mask encoding, decoding, and geometry utilities for My3DIS."""

from .encoding import (
    PACKED_MASK_KEY,
    PACKED_MASK_B64_KEY,
    PACKED_SHAPE_KEY,
    PACKED_ORIG_SHAPE_KEY,
    encode_mask,
    pack_binary_mask,
    unpack_binary_mask,
    is_packed_mask,
    downscale_binary_mask,
    normalize_shape_tuple,
)
from .geometry import bbox_from_mask_xyxy, bbox_xyxy_to_xywh

__all__ = [
    # Encoding constants
    'PACKED_MASK_KEY',
    'PACKED_MASK_B64_KEY',
    'PACKED_SHAPE_KEY',
    'PACKED_ORIG_SHAPE_KEY',
    # Encoding functions
    'encode_mask',
    'pack_binary_mask',
    'unpack_binary_mask',
    'is_packed_mask',
    'downscale_binary_mask',
    'normalize_shape_tuple',
    # Geometry functions
    'bbox_from_mask_xyxy',
    'bbox_xyxy_to_xywh',
]
