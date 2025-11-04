"""
SigLIP Feature Assignment Module - Assigns vision-language features to 3D proposals.

This module handles the second stage of 3D projection pipeline:
1. Loads 3D proposals from aggregation stage
2. Extracts SigLIP features from 2D views
3. Aggregates features across frames
4. Assigns semantic labels via open-vocabulary inference

Input: 3D proposals (NPZ) from aggregation stage
Output: Proposals with SigLIP features ready for open-vocabulary query
"""

from .siglip_pipeline import SigLIPAssignmentPipeline
from .feature_extraction import SigLIPFeatureExtractor
from .assignment_utils import aggregate_features, assign_labels

__all__ = [
    'SigLIPAssignmentPipeline',
    'SigLIPFeatureExtractor',
    'aggregate_features',
    'assign_labels',
]
