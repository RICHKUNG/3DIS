"""
Configuration for inference pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class RetrievalConfig:
    """Configuration for multi-level retrieval."""
    temperature: float = 0.2
    min_similarity: float = 0.2
    top_k_per_level: int = 50
    object_levels: List[str] = field(default_factory=lambda: ['L1', 'L3'])
    part_levels: List[str] = field(default_factory=lambda: ['L3', 'L5'])


@dataclass
class PairingConfig:
    """Configuration for object-part pairing."""
    containment_threshold: float = 0.5
    scale_range: tuple = (0.01, 0.9)
    use_geometric_score: bool = True
    use_tree_pruning: bool = True


@dataclass
class NMSConfig:
    """Configuration for NMS."""
    iou_threshold: float = 0.35
    use_soft_nms: bool = False
    soft_nms_sigma: float = 0.5
    centroid_distance_ratio: float = 0.2
    keep_top_k: Optional[int] = None


@dataclass
class ScoringConfig:
    """Configuration for score aggregation."""
    method: str = "weighted_sum"  # arithmetic, geometric, weighted_sum
    weights: Dict[str, float] = field(default_factory=lambda: {
        'object': 0.4,
        'part': 0.4,
        'geometric': 0.2
    })
    temperature: float = 1.0
    hierarchy_bonus: float = 0.05


@dataclass
class IndependentStrategyConfig:
    """Configuration for independent retrieval strategy."""
    object_levels: List[str] = field(default_factory=lambda: ['L1', 'L3'])
    part_levels: List[str] = field(default_factory=lambda: ['L3', 'L5'])
    top_k_per_level: int = 50
    retrieval_threshold: float = 0.3
    min_retrieval_threshold: float = 0.05
    threshold_backoff_step: float = 0.05
    fallback_to_topk: bool = True

    # Combined query settings
    use_combined_query: bool = True
    siglip_model: str = "google/siglip-so400m-patch14-384"
    siglip_device: str = "cuda:0"


@dataclass
class HierarchicalStrategyConfig:
    """Configuration for hierarchical strategy."""
    coarse_top_k: int = 100
    object_top_k: int = 50
    part_top_k: int = 30
    coarse_threshold: float = 0.2
    refinement_threshold: float = 0.3
    refinement_margin: float = 0.1

    # Combined query settings
    use_combined_query: bool = True
    siglip_model: str = "google/siglip-so400m-patch14-384"
    siglip_device: str = "cuda:0"


@dataclass
class ExhaustivePairingConfig:
    """Configuration for exhaustive pairing strategy."""
    available_levels: List[str] = field(default_factory=lambda: ['L1', 'L3', 'L5'])
    top_k_per_level: int = 100
    retrieval_threshold: float = 0.2
    min_retrieval_threshold: float = 0.05
    threshold_backoff_step: float = 0.03
    fallback_to_topk: bool = True
    selection_metric: str = 'max'  # max, mean, or sum
    use_tree_pruning: bool = False
    return_all_pairs: bool = False

    # Combined query settings
    use_combined_query: bool = True
    siglip_model: str = "google/siglip-so400m-patch14-384"
    siglip_device: str = "cuda:0"


@dataclass
class FormatterConfig:
    """Configuration for output formatting."""
    use_part_mask: bool = True
    min_mask_size: int = 10


@dataclass
class InferenceConfig:
    """Complete inference configuration."""
    strategy: str = "independent"  # independent, hierarchical, exhaustive_pairing
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    nms: NMSConfig = field(default_factory=NMSConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    formatter: FormatterConfig = field(default_factory=FormatterConfig)

    # Strategy-specific configs
    independent: IndependentStrategyConfig = field(default_factory=IndependentStrategyConfig)
    hierarchical: HierarchicalStrategyConfig = field(default_factory=HierarchicalStrategyConfig)
    exhaustive_pairing: ExhaustivePairingConfig = field(default_factory=ExhaustivePairingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'InferenceConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Build nested configs
        config = cls()

        if 'strategy' in data:
            config.strategy = data['strategy']

        if 'retrieval' in data:
            config.retrieval = RetrievalConfig(**data['retrieval'])

        if 'pairing' in data:
            config.pairing = PairingConfig(**data['pairing'])

        if 'nms' in data:
            config.nms = NMSConfig(**data['nms'])

        if 'scoring' in data:
            config.scoring = ScoringConfig(**data['scoring'])

        if 'formatter' in data:
            config.formatter = FormatterConfig(**data['formatter'])

        if 'independent' in data:
            config.independent = IndependentStrategyConfig(**data['independent'])

        if 'hierarchical' in data:
            config.hierarchical = HierarchicalStrategyConfig(**data['hierarchical'])

        if 'exhaustive_pairing' in data:
            config.exhaustive_pairing = ExhaustivePairingConfig(**data['exhaustive_pairing'])

        return config

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = self.to_dict()

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a serializable dictionary."""
        return {
            'strategy': self.strategy,
            'retrieval': {
                'temperature': self.retrieval.temperature,
                'min_similarity': self.retrieval.min_similarity,
                'top_k_per_level': self.retrieval.top_k_per_level,
                'object_levels': self.retrieval.object_levels,
                'part_levels': self.retrieval.part_levels,
            },
            'pairing': {
                'containment_threshold': self.pairing.containment_threshold,
                'scale_range': list(self.pairing.scale_range),
                'use_geometric_score': self.pairing.use_geometric_score,
                'use_tree_pruning': self.pairing.use_tree_pruning,
            },
            'nms': {
                'iou_threshold': self.nms.iou_threshold,
                'use_soft_nms': self.nms.use_soft_nms,
                'soft_nms_sigma': self.nms.soft_nms_sigma,
                'centroid_distance_ratio': self.nms.centroid_distance_ratio,
                'keep_top_k': self.nms.keep_top_k,
            },
            'scoring': {
                'method': self.scoring.method,
                'weights': self.scoring.weights,
                'temperature': self.scoring.temperature,
                'hierarchy_bonus': self.scoring.hierarchy_bonus,
            },
            'formatter': {
                'use_part_mask': self.formatter.use_part_mask,
                'min_mask_size': self.formatter.min_mask_size,
            },
            'independent': {
                'object_levels': self.independent.object_levels,
                'part_levels': self.independent.part_levels,
                'top_k_per_level': self.independent.top_k_per_level,
                'retrieval_threshold': self.independent.retrieval_threshold,
                'min_retrieval_threshold': self.independent.min_retrieval_threshold,
                'threshold_backoff_step': self.independent.threshold_backoff_step,
                'fallback_to_topk': self.independent.fallback_to_topk,
                'use_combined_query': self.independent.use_combined_query,
                'siglip_model': self.independent.siglip_model,
                'siglip_device': self.independent.siglip_device,
            },
            'hierarchical': {
                'coarse_top_k': self.hierarchical.coarse_top_k,
                'object_top_k': self.hierarchical.object_top_k,
                'part_top_k': self.hierarchical.part_top_k,
                'coarse_threshold': self.hierarchical.coarse_threshold,
                'refinement_threshold': self.hierarchical.refinement_threshold,
                'refinement_margin': self.hierarchical.refinement_margin,
                'use_combined_query': self.hierarchical.use_combined_query,
                'siglip_model': self.hierarchical.siglip_model,
                'siglip_device': self.hierarchical.siglip_device,
            },
            'exhaustive_pairing': {
                'available_levels': self.exhaustive_pairing.available_levels,
                'top_k_per_level': self.exhaustive_pairing.top_k_per_level,
                'retrieval_threshold': self.exhaustive_pairing.retrieval_threshold,
                'min_retrieval_threshold': self.exhaustive_pairing.min_retrieval_threshold,
                'threshold_backoff_step': self.exhaustive_pairing.threshold_backoff_step,
                'fallback_to_topk': self.exhaustive_pairing.fallback_to_topk,
                'selection_metric': self.exhaustive_pairing.selection_metric,
                'use_tree_pruning': self.exhaustive_pairing.use_tree_pruning,
                'return_all_pairs': self.exhaustive_pairing.return_all_pairs,
                'use_combined_query': self.exhaustive_pairing.use_combined_query,
                'siglip_model': self.exhaustive_pairing.siglip_model,
                'siglip_device': self.exhaustive_pairing.siglip_device,
            },
        }

    def get_strategy_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for strategy initialization based on selected strategy."""
        if self.strategy == "independent":
            return {
                'object_levels': self.independent.object_levels,
                'part_levels': self.independent.part_levels,
                'top_k_per_level': self.independent.top_k_per_level,
                'retrieval_threshold': self.independent.retrieval_threshold,
                'min_retrieval_threshold': self.independent.min_retrieval_threshold,
                'threshold_backoff_step': self.independent.threshold_backoff_step,
                'fallback_to_topk': self.independent.fallback_to_topk,
                'use_combined_query': self.independent.use_combined_query,
                'siglip_model': self.independent.siglip_model,
                'siglip_device': self.independent.siglip_device,
            }
        elif self.strategy == "hierarchical":
            return {
                'coarse_top_k': self.hierarchical.coarse_top_k,
                'object_top_k': self.hierarchical.object_top_k,
                'part_top_k': self.hierarchical.part_top_k,
                'coarse_threshold': self.hierarchical.coarse_threshold,
                'refinement_threshold': self.hierarchical.refinement_threshold,
                'refinement_margin': self.hierarchical.refinement_margin,
                'use_combined_query': self.hierarchical.use_combined_query,
                'siglip_model': self.hierarchical.siglip_model,
                'siglip_device': self.hierarchical.siglip_device,
            }
        elif self.strategy == "exhaustive_pairing":
            return {
                'available_levels': self.exhaustive_pairing.available_levels,
                'top_k_per_level': self.exhaustive_pairing.top_k_per_level,
                'retrieval_threshold': self.exhaustive_pairing.retrieval_threshold,
                'min_retrieval_threshold': self.exhaustive_pairing.min_retrieval_threshold,
                'threshold_backoff_step': self.exhaustive_pairing.threshold_backoff_step,
                'fallback_to_topk': self.exhaustive_pairing.fallback_to_topk,
                'selection_metric': self.exhaustive_pairing.selection_metric,
                'use_tree_pruning': self.exhaustive_pairing.use_tree_pruning,
                'return_all_pairs': self.exhaustive_pairing.return_all_pairs,
                'use_combined_query': self.exhaustive_pairing.use_combined_query,
                'siglip_model': self.exhaustive_pairing.siglip_model,
                'siglip_device': self.exhaustive_pairing.siglip_device,
            }
        else:
            return {}
