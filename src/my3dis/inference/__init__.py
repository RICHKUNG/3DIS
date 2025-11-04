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
"""

from .retrieval import MultiLevelRetriever
from .pairing import ObjectPartPairer
from .nms import multi_instance_nms
from .scoring import ScoreAggregator
from .formatter import Search3DFormatter
from .inference_pipeline import InferencePipeline

__all__ = [
    'MultiLevelRetriever',
    'ObjectPartPairer',
    'multi_instance_nms',
    'ScoreAggregator',
    'Search3DFormatter',
    'InferencePipeline',
]
