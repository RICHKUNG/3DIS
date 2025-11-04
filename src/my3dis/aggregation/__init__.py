"""
3D Mask Aggregation Module - Projects 2D masks to 3D point clouds.

This module handles the first stage of 3D projection pipeline:
1. Loads family tree from SAM2 tracking output
2. Projects 2D masks to 3D point cloud
3. Aggregates masks across frames
4. Generates 3D proposals with geometric properties

Input: family_tree.json from SAM2 stage
Output: 3D proposals (NPZ format) ready for SigLIP feature assignment
"""

from .aggregation_pipeline import AggregationPipeline
from .mask_projection import MaskProjector
from .aggregation_utils import aggregate_masks_3d, compute_proposal_geometry

__all__ = [
    'AggregationPipeline',
    'MaskProjector',
    'aggregate_masks_3d',
    'compute_proposal_geometry',
]
