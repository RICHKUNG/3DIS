"""
Score aggregation strategies for combining multiple signals.

Provides methods to combine object/part similarity scores with geometric scores
to produce final confidence values for predictions.
"""

import numpy as np
from typing import Dict, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScoreAggregationMethod(Enum):
    """Methods for aggregating multiple scores."""
    ARITHMETIC_MEAN = "arithmetic"
    GEOMETRIC_MEAN = "geometric"
    WEIGHTED_SUM = "weighted_sum"
    MAX = "max"
    MIN = "min"


class ScoreAggregator:
    """
    Aggregates multiple scores (object similarity, part similarity, geometric)
    into a final confidence score.
    """

    def __init__(
        self,
        method: ScoreAggregationMethod = ScoreAggregationMethod.WEIGHTED_SUM,
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0
    ):
        """
        Initialize score aggregator.

        Args:
            method: Aggregation method
            weights: Weight for each score component (if using weighted_sum)
            temperature: Temperature for softening scores
        """
        self.method = method
        self.weights = weights or {'object': 0.4, 'part': 0.4, 'geometric': 0.2}
        self.temperature = temperature

    def aggregate(
        self,
        object_score: float,
        part_score: float,
        geometric_score: Optional[float] = None
    ) -> float:
        """
        Aggregate scores into a single confidence value.

        Args:
            object_score: Object similarity score [0, 1]
            part_score: Part similarity score [0, 1]
            geometric_score: Geometric compatibility score [0, 1]

        Returns:
            Aggregated score in [0, 1]
        """
        scores = [object_score, part_score]
        score_names = ['object', 'part']

        if geometric_score is not None:
            scores.append(geometric_score)
            score_names.append('geometric')

        if self.method == ScoreAggregationMethod.ARITHMETIC_MEAN:
            result = np.mean(scores)

        elif self.method == ScoreAggregationMethod.GEOMETRIC_MEAN:
            result = np.prod(scores) ** (1.0 / len(scores))

        elif self.method == ScoreAggregationMethod.WEIGHTED_SUM:
            # Normalize weights
            total_weight = sum(self.weights.get(name, 0.0) for name in score_names)
            if total_weight == 0:
                logger.warning("Total weight is 0, falling back to arithmetic mean")
                result = np.mean(scores)
            else:
                result = sum(
                    score * self.weights.get(name, 0.0) / total_weight
                    for score, name in zip(scores, score_names)
                )

        elif self.method == ScoreAggregationMethod.MAX:
            result = np.max(scores)

        elif self.method == ScoreAggregationMethod.MIN:
            result = np.min(scores)

        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        # Apply temperature
        if self.temperature != 1.0:
            result = result ** (1.0 / self.temperature)

        return float(np.clip(result, 0.0, 1.0))

    def aggregate_with_hierarchy_bonus(
        self,
        object_score: float,
        part_score: float,
        geometric_score: Optional[float] = None,
        object_level: Optional[str] = None,
        part_level: Optional[str] = None,
        bonus: float = 0.05
    ) -> float:
        """
        Aggregate scores with bonus for expected hierarchy relationships.

        Args:
            object_score: Object similarity
            part_score: Part similarity
            geometric_score: Geometric score
            object_level: Level of object proposal (e.g., 'L2', 'L4')
            part_level: Level of part proposal
            bonus: Bonus to add for expected hierarchy

        Returns:
            Aggregated score with hierarchy bonus
        """
        base_score = self.aggregate(object_score, part_score, geometric_score)

        # Add bonus if hierarchy is as expected (object at coarser level than part)
        if object_level is not None and part_level is not None:
            # Extract level numbers dynamically (L1→1, L2→2, etc.)
            import re
            obj_match = re.match(r'^L(\d+)$', object_level)
            part_match = re.match(r'^L(\d+)$', part_level)

            if obj_match and part_match:
                obj_idx = int(obj_match.group(1))
                part_idx = int(part_match.group(1))

                # Coarser level (smaller number) should be object, finer level (larger) should be part
                if obj_idx < part_idx:
                    base_score = min(1.0, base_score + bonus)

        return base_score


def apply_temperature_scaling(scores: np.ndarray, temperature: float = 0.2) -> np.ndarray:
    """
    Apply temperature scaling to similarity scores.

    Args:
        scores: Array of scores
        temperature: Temperature parameter (lower = sharper distribution)

    Returns:
        Scaled scores
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    # Apply exponential scaling
    scaled = np.exp(scores / temperature)

    # Normalize to [0, 1] range
    min_val = np.exp(0.0 / temperature)  # Assuming min possible score is 0
    max_val = np.exp(1.0 / temperature)  # Assuming max possible score is 1

    normalized = (scaled - min_val) / (max_val - min_val + 1e-8)

    return np.clip(normalized, 0.0, 1.0)


def compute_confidence_distribution(
    scores: np.ndarray,
    method: str = 'softmax',
    temperature: float = 1.0
) -> np.ndarray:
    """
    Convert scores to a confidence distribution.

    Args:
        scores: Array of scores
        method: 'softmax', 'normalize', or 'raw'
        temperature: Temperature for softmax

    Returns:
        Confidence distribution
    """
    if len(scores) == 0:
        return np.array([])

    if method == 'softmax':
        exp_scores = np.exp(scores / temperature)
        return exp_scores / (np.sum(exp_scores) + 1e-8)

    elif method == 'normalize':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-8:
            return np.ones_like(scores) / len(scores)
        return (scores - min_score) / (max_score - min_score)

    elif method == 'raw':
        return scores

    else:
        raise ValueError(f"Unknown method: {method}")


class AdaptiveScoreAggregator(ScoreAggregator):
    """
    Score aggregator that adapts weights based on score reliability.

    For example, if geometric score is very high, increase its weight.
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        adaptation_strength: float = 0.3
    ):
        """
        Initialize adaptive aggregator.

        Args:
            base_weights: Base weights for each component
            adaptation_strength: How much to adapt weights (0 = no adaptation, 1 = full)
        """
        super().__init__(
            method=ScoreAggregationMethod.WEIGHTED_SUM,
            weights=base_weights
        )
        self.adaptation_strength = adaptation_strength
        self.base_weights = self.weights.copy()

    def aggregate(
        self,
        object_score: float,
        part_score: float,
        geometric_score: Optional[float] = None
    ) -> float:
        """Aggregate with adaptive weights."""
        scores = {'object': object_score, 'part': part_score}
        if geometric_score is not None:
            scores['geometric'] = geometric_score

        # Adapt weights based on score magnitudes
        adapted_weights = {}
        for name, score in scores.items():
            base_weight = self.base_weights.get(name, 0.0)
            # Increase weight if score is high, decrease if low
            adaptation = (score - 0.5) * self.adaptation_strength
            adapted_weights[name] = base_weight + adaptation

        # Normalize
        total = sum(adapted_weights.values())
        if total > 0:
            adapted_weights = {k: v / total for k, v in adapted_weights.items()}

        # Compute weighted sum
        result = sum(scores[name] * adapted_weights.get(name, 0.0) for name in scores)

        return float(np.clip(result, 0.0, 1.0))
