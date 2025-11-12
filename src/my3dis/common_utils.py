"""Shared utility helpers for the My3DIS pipeline.

DEPRECATED: This module is kept for backward compatibility only.
New code should import from specific modules instead:

- my3dis.mask.encoding for mask encoding/decoding operations
- my3dis.mask.geometry for bounding box and geometric operations
- my3dis.io.fs_utils for filesystem operations
- my3dis.utils.time_utils for time formatting
- my3dis.utils.parsing for parsing utilities
- my3dis.utils.sorting for sorting utilities
- my3dis.utils.logging for logging configuration

This compatibility layer will be maintained until v2.2 (approximately 9 months).
"""

from __future__ import annotations

# Re-export constants
from my3dis.mask.encoding import (
    PACKED_MASK_KEY,
    PACKED_MASK_B64_KEY,
    PACKED_SHAPE_KEY,
    PACKED_ORIG_SHAPE_KEY,
)

# Re-export mask encoding functions
from my3dis.mask.encoding import (
    normalize_shape_tuple,
    encode_mask,
    pack_binary_mask,
    is_packed_mask,
    unpack_binary_mask,
    downscale_binary_mask,
)

# Re-export geometry functions
from my3dis.mask.geometry import (
    bbox_from_mask_xyxy,
    bbox_xyxy_to_xywh,
)

# Re-export filesystem functions
from my3dis.io.fs_utils import (
    ensure_dir,
    build_subset_video,
)

# Re-export time utilities
from my3dis.utils.time_utils import format_duration

# Re-export parsing utilities
from my3dis.utils.parsing import (
    parse_levels,
    parse_range,
    list_to_csv,
)

# Re-export sorting utilities
from my3dis.utils.sorting import numeric_frame_sort_key

# Re-export logging utilities
from my3dis.utils.logging import (
    setup_logging,
    configure_entry_log_format,
    ENTRY_LOG_FORMAT,
)

# Legacy constants (keep for backward compatibility)
RAW_DIR_NAME = "raw"
RAW_META_TEMPLATE = "frame_{frame_idx:05d}.json"
RAW_MASK_TEMPLATE = "frame_{frame_idx:05d}.npz"


__all__ = [
    # Constants
    'RAW_DIR_NAME',
    'RAW_META_TEMPLATE',
    'RAW_MASK_TEMPLATE',
    'PACKED_MASK_KEY',
    'PACKED_MASK_B64_KEY',
    'PACKED_SHAPE_KEY',
    'PACKED_ORIG_SHAPE_KEY',
    'ENTRY_LOG_FORMAT',
    # Mask encoding
    'normalize_shape_tuple',
    'encode_mask',
    'pack_binary_mask',
    'is_packed_mask',
    'unpack_binary_mask',
    'downscale_binary_mask',
    # Geometry
    'bbox_from_mask_xyxy',
    'bbox_xyxy_to_xywh',
    # Filesystem
    'ensure_dir',
    'build_subset_video',
    # Time utilities
    'format_duration',
    # Parsing
    'parse_levels',
    'parse_range',
    'list_to_csv',
    # Sorting
    'numeric_frame_sort_key',
    # Logging
    'setup_logging',
    'configure_entry_log_format',
]
