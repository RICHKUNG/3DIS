"""
Backward-compatible wrapper around the refactored semantic refinement core.

The previous codebase imported ``progressive_refinement_core`` directly, so we
keep this module as a thin alias that emits a clear deprecation warning.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "progressive_refinement_core is deprecated. "
    "Import from my3dis.semantic_refinement instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .semantic_refinement import (  # noqa: F401
    DEFAULT_SEMANTIC_SAM_ROOT,
    SemanticSamAutomaticMaskGenerator,
    bbox_from_mask,
    console,
    create_experiment_folder,
    create_masked_image,
    get_experiment_timestamp,
    instance_map_to_anns,
    instance_map_to_color_image,
    prepare_image,
    prepare_image_from_pil,
    progressive_refinement_masks,
    setup_output_directories,
    save_original_image_info,
    timer_decorator,
)

__all__ = [
    "DEFAULT_SEMANTIC_SAM_ROOT",
    "SemanticSamAutomaticMaskGenerator",
    "bbox_from_mask",
    "console",
    "create_experiment_folder",
    "create_masked_image",
    "get_experiment_timestamp",
    "instance_map_to_anns",
    "instance_map_to_color_image",
    "prepare_image",
    "prepare_image_from_pil",
    "progressive_refinement_masks",
    "setup_output_directories",
    "save_original_image_info",
    "timer_decorator",
]
