"""
Inference strategies for open vocabulary 3D instance segmentation.

Provides different strategies for retrieving and pairing object-part proposals:
- IndependentStrategy: Baseline independent retrieval + geometric pairing
- HierarchicalStrategy: Coarse-to-fine with tree-guided search
- ExhaustivePairingStrategy: Exhaustive level-pair evaluation with smart selection
- ExhaustivePairingWithCombinedQuery: Exhaustive pairing with "X of Y" combined queries
"""

from .base import InferenceStrategy
from .independent import IndependentStrategy
from .hierarchical import HierarchicalStrategy
from .exhaustive_pairing import ExhaustivePairingStrategy
from .exhaustive_pairing_with_combined_query import ExhaustivePairingWithCombinedQuery

__all__ = [
    'InferenceStrategy',
    'IndependentStrategy',
    'HierarchicalStrategy',
    'ExhaustivePairingStrategy',
    'ExhaustivePairingWithCombinedQuery',
]
