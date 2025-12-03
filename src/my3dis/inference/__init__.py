"""
Open Vocabulary 3D Instance Segmentation Inference Module

This module provides inference capabilities for multi-level object-part retrieval
in 3D scenes, designed for evaluation on MultiScan dataset with Search3D benchmark.

Main components:
- retrieval: Multi-level feature retrieval with SigLIP embeddings
- pairing: Object-Part pairing with geometric constraints
- nms: Multi-instance aware Non-Maximum Suppression
- strategies: Different inference strategies (independent, hierarchical, exhaustive)
- formatter: Output formatting for Search3D evaluation
- feature_extraction: SigLIP feature extraction for proposals and text queries
"""

from .retrieval import MultiLevelRetriever
from .pairing import ObjectPartPairer
from .nms import multi_instance_nms
from .scoring import ScoreAggregator
from .formatter import Search3DFormatter
from .inference_pipeline import InferencePipeline
from .feature_extraction import SigLIPFeatureExtractor
from .feature_extraction_optimized import (
    OptimizedSigLIPFeatureExtractor,
    ImageCache
)

__all__ = [
    'MultiLevelRetriever',
    'ObjectPartPairer',
    'multi_instance_nms',
    'ScoreAggregator',
    'Search3DFormatter',
    'InferencePipeline',
    'SigLIPFeatureExtractor',
    'OptimizedSigLIPFeatureExtractor',
    'ImageCache',
]
