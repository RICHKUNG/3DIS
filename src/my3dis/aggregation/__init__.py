"""
3D Mask Aggregation Module - Projects 2D masks to 3D point clouds.

This module handles the first stage of 3D projection pipeline:
1. Loads family tree from SAM2 tracking output
2. Projects 2D masks to 3D point cloud
3. Aggregates masks across frames
4. Generates 3D proposals with geometric properties

Input: family_tree.json from SAM2 stage
Output: 3D proposals (NPZ format) ready for SigLIP feature assignment

Components:
- AggregationPipeline: Full 2D→3D projection pipeline (requires depth maps)
- ProposalLoader: Lightweight loader for 2D masks from tracking outputs
- MaskProjector: 2D to 3D projection utilities
"""

from .aggregation_pipeline import AggregationPipeline, detect_available_levels
from .mask_projection import MaskProjector
from .aggregation_utils import aggregate_masks_3d, compute_proposal_geometry
from .proposal_loader import ProposalLoader, load_proposals_from_experiment
from .search3d_formatter import (
    export_predictions_search3d,
    export_predictions_both_formats,
    load_point_cloud_for_export,
    validate_search3d_format,
)

__all__ = [
    'AggregationPipeline',
    'detect_available_levels',
    'MaskProjector',
    'aggregate_masks_3d',
    'compute_proposal_geometry',
    'ProposalLoader',
    'load_proposals_from_experiment',
    'export_predictions_search3d',
    'export_predictions_both_formats',
    'load_point_cloud_for_export',
    'validate_search3d_format',
]
