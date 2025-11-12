"""Mask encoding and decoding utilities.

This module provides functions for packing/unpacking binary masks into
compact representations suitable for JSON serialization and NPZ storage.
"""

from __future__ import annotations

import base64
from typing import Optional, Sequence, Tuple, Union

import numpy as np


# Packed mask keys
PACKED_MASK_KEY = "packed_bits"
PACKED_MASK_B64_KEY = "packed_bits_b64"
PACKED_SHAPE_KEY = "shape"
PACKED_ORIG_SHAPE_KEY = "full_resolution_shape"


def normalize_shape_tuple(shape: Union[np.ndarray, list, tuple, int]) -> Tuple[int, ...]:
    """
    Normalize various shape representations to a tuple of integers.

    This utility handles shape coercion from:
    - NumPy arrays (from JSON/NPZ loading)
    - Python lists
    - Tuples (ensures int conversion)
    - Scalar integers (returns (n,))

    Args:
        shape: Shape in various formats

    Returns:
        Tuple of integers representing the shape

    Examples:
        >>> normalize_shape_tuple(np.array([1080, 1920]))
        (1080, 1920)
        >>> normalize_shape_tuple([512, 512])
        (512, 512)
        >>> normalize_shape_tuple((100, 200))
        (100, 200)
        >>> normalize_shape_tuple(1000)
        (1000,)
    """
    if isinstance(shape, np.ndarray):
        return tuple(int(v) for v in shape.flat)
    elif isinstance(shape, (list, tuple)):
        return tuple(int(v) for v in shape)
    else:
        # Scalar case
        return (int(shape),)


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    """Serialize boolean mask into JSON-friendly packed bits payload."""
    bool_mask = np.asarray(mask, dtype=np.bool_, order="C")
    packed = np.packbits(bool_mask.reshape(-1))
    return {
        "shape": [int(dim) for dim in bool_mask.shape],
        "packed_bits_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def pack_binary_mask(
    mask: np.ndarray,
    *,
    full_resolution_shape: Optional[Sequence[int]] = None,
) -> dict[str, object]:
    """Pack a boolean mask into a compact dict.

    Args:
        mask: Boolean mask array
        full_resolution_shape: Original shape before downscaling (optional)

    Returns:
        Dictionary with packed mask data
    """
    bool_mask = np.asarray(mask, dtype=np.bool_, order="C")
    packed = np.packbits(bool_mask.reshape(-1))
    payload: dict[str, object] = {
        PACKED_MASK_KEY: packed,
        PACKED_SHAPE_KEY: tuple(int(dim) for dim in bool_mask.shape),
    }

    if full_resolution_shape is not None:
        shape_arr = np.atleast_1d(full_resolution_shape)
        if shape_arr.size:
            payload[PACKED_ORIG_SHAPE_KEY] = tuple(int(v) for v in shape_arr.tolist())

    return payload


def is_packed_mask(entry: object) -> bool:
    """Check if an object is a packed mask dictionary."""
    if not isinstance(entry, dict):
        return False
    if PACKED_SHAPE_KEY not in entry:
        return False
    return PACKED_MASK_KEY in entry or PACKED_MASK_B64_KEY in entry


def unpack_binary_mask(entry: object) -> np.ndarray:
    """Unpack a packed mask dictionary into a boolean array.

    Args:
        entry: Packed mask dictionary or raw array

    Returns:
        Boolean NumPy array
    """
    if not is_packed_mask(entry):
        array = np.asarray(entry)
        if array.dtype != np.bool_:
            array = array.astype(np.bool_)
        return array

    payload = dict(entry)
    shape = normalize_shape_tuple(payload[PACKED_SHAPE_KEY])

    total = int(np.prod(shape))

    if PACKED_MASK_KEY in payload:
        packed_arr = np.asarray(payload[PACKED_MASK_KEY], dtype=np.uint8)
    else:
        packed_bytes = base64.b64decode(payload[PACKED_MASK_B64_KEY])
        packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8)

    unpacked = np.unpackbits(packed_arr, count=total)
    return unpacked.reshape(shape).astype(np.bool_)


def downscale_binary_mask(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Downscale a boolean mask by ``ratio`` using a box filter + threshold.

    Args:
        mask: Boolean mask array
        ratio: Scaling ratio (0 < ratio <= 1)

    Returns:
        Downscaled boolean mask

    Raises:
        ValueError: If ratio is not in valid range or mask has invalid shape
    """
    try:
        ratio_val = float(ratio)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        ratio_val = 1.0

    if ratio_val >= 1.0:
        return np.asarray(mask, dtype=np.bool_)
    if ratio_val <= 0.0:
        raise ValueError("downscale ratio must be within (0, 1]")

    arr = np.asarray(mask, dtype=np.bool_)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported mask dimensionality: {arr.shape}")

    h, w = arr.shape
    new_h = max(1, int(round(h * ratio_val)))
    new_w = max(1, int(round(w * ratio_val)))
    if new_h == h and new_w == w:
        return arr

    from PIL import Image

    img = Image.fromarray(arr.astype(np.uint8) * 255)
    resample = getattr(Image, 'BOX', Image.BILINEAR)
    resized = img.resize((new_w, new_h), resample=resample)
    arr_float = np.asarray(resized, dtype=np.float32) / 255.0
    return arr_float >= 0.5
