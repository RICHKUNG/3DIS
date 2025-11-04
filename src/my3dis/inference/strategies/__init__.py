"""
Inference strategies for open vocabulary 3D instance segmentation.

Provides different strategies for retrieving and pairing object-part proposals:
- IndependentStrategy: Baseline independent retrieval + geometric pairing
- HierarchicalStrategy: Coarse-to-fine with tree-guided search
- ExhaustiveStrategy: Exhaustive scoring with smart pruning
"""

from .base import InferenceStrategy
from .independent import IndependentStrategy
from .hierarchical import HierarchicalStrategy

__all__ = [
    'InferenceStrategy',
    'IndependentStrategy',
    'HierarchicalStrategy',
]
