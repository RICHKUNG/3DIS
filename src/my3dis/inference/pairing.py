"""
Object-Part pairing module.

Pairs object and part proposals based on geometric constraints and
hierarchical relationships from the family tree.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import logging
import re

from .retrieval import Proposal
from .geometry import (
    check_geometric_validity,
    compute_geometric_score,
    compute_iou
)
from .scoring import ScoreAggregator

logger = logging.getLogger(__name__)


def parse_level_to_prefix(level: str) -> int:
    """
    Extract level number from level string and compute ID prefix.

    Args:
        level: Level string (e.g., 'L1', 'L2', ..., 'L6')

    Returns:
        ID prefix for this level (e.g., L1→1000, L2→2000, L6→6000)

    Raises:
        ValueError: If level format is invalid
    """
    match = re.match(r'^L(\d+)$', level)
    if not match:
        raise ValueError(
            f"Invalid level format: '{level}'. Expected format: L<number> (e.g., 'L1', 'L2', 'L6')"
        )

    level_num = int(match.group(1))
    if level_num < 1 or level_num > 9:
        raise ValueError(
            f"Invalid level number: {level_num}. Must be between 1 and 9"
        )

    return level_num * 1000


@dataclass
class ObjectPartPair:
    """A paired object-part prediction."""
    object_proposal: Proposal
    part_proposal: Proposal
    object_score: float
    part_score: float
    geometric_score: float
    combined_score: float
    label_id: int  # Joint semantic label (e.g., 121003 for "cabinet door")


class FamilyTree:
    """
    Simplified family tree interface for hierarchical relationships.

    This should be replaced with actual family_tree_query.FamilyTreeQuery
    in production, but provides a simple interface for the pairing logic.
    """

    def __init__(self, tree_data: Optional[dict] = None):
        """
        Initialize family tree.

        Args:
            tree_data: Dictionary with parent-child relationships
                      Format: {obj_id: {'parent': parent_id, 'children': [child_ids]}}
        """
        self.tree_data = tree_data or {}

    def get_children(self, obj_id: int, target_level: Optional[str] = None) -> List[int]:
        """Get children of an object, optionally filtered by level."""
        if obj_id not in self.tree_data:
            return []

        children = self.tree_data[obj_id].get('children', [])

        if target_level is not None:
            # Try level-based ID format first (L1→1000+, L2→2000+, etc.)
            try:
                prefix = parse_level_to_prefix(target_level)
                filtered = [c for c in children if prefix <= c < prefix + 1000]

                # If no matches found, try using level field from tree_data
                if len(filtered) == 0 and len(children) > 0:
                    # Extract target level number (e.g., 'L3' -> 3)
                    target_level_num = int(target_level[1:])
                    filtered = [
                        c for c in children
                        if c in self.tree_data and self.tree_data[c].get('level') == target_level_num
                    ]

                children = filtered
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid target_level '{target_level}': {e}")
                return []

        return children

    def get_descendants(self, obj_id: int, levels: Optional[List[str]] = None) -> List[int]:
        """Get all descendants of an object, optionally filtered by levels."""
        descendants = []
        queue = [obj_id]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            children = self.get_children(current)
            descendants.extend(children)
            queue.extend(children)

        if levels is not None:
            # Try level-based ID format first
            try:
                level_prefixes = [parse_level_to_prefix(l) for l in levels]
                filtered = [
                    d for d in descendants
                    if any(prefix <= d < prefix + 1000 for prefix in level_prefixes)
                ]

                # If no matches found, try using level field from tree_data
                if len(filtered) == 0 and len(descendants) > 0:
                    # Extract target level numbers (e.g., ['L3', 'L5'] -> [3, 5])
                    target_level_nums = [int(level[1:]) for level in levels]
                    filtered = [
                        d for d in descendants
                        if d in self.tree_data and self.tree_data[d].get('level') in target_level_nums
                    ]

                descendants = filtered
            except (ValueError, KeyError) as e:
                logger.error(f"Invalid level in levels list: {e}")
                # Return all descendants if level parsing fails
                pass

        return descendants

    def is_ancestor(self, ancestor_id: int, descendant_id: int) -> bool:
        """Check if ancestor_id is an ancestor of descendant_id."""
        descendants = self.get_descendants(ancestor_id)
        return descendant_id in descendants


class ObjectPartPairer:
    """
    Pairs object and part proposals with geometric and hierarchical constraints.
    """

    def __init__(
        self,
        score_aggregator: Optional[ScoreAggregator] = None,
        family_tree: Optional[FamilyTree] = None,
        containment_threshold: float = 0.5,
        scale_range: Tuple[float, float] = (0.01, 0.9),
        use_geometric_score: bool = True
    ):
        """
        Initialize pairer.

        Args:
            score_aggregator: Aggregator for combining scores
            family_tree: Family tree for hierarchical constraints
            containment_threshold: Minimum containment ratio for valid pairs
            scale_range: (min, max) acceptable size ratio (part/object)
            use_geometric_score: Whether to include geometric score
        """
        self.score_aggregator = score_aggregator or ScoreAggregator()
        self.family_tree = family_tree
        self.containment_threshold = containment_threshold
        self.scale_range = scale_range
        self.use_geometric_score = use_geometric_score

    def pair_all(
        self,
        object_proposals: List[Proposal],
        part_proposals: List[Proposal],
        label_id: int,
        points: Optional[np.ndarray] = None,
        use_tree_pruning: bool = True
    ) -> List[ObjectPartPair]:
        """
        Pair all object and part proposals.

        Args:
            object_proposals: List of object proposals
            part_proposals: List of part proposals
            label_id: Joint semantic label for this query
            points: Point cloud coordinates
            use_tree_pruning: Whether to use family tree for pruning

        Returns:
            List of valid object-part pairs
        """
        pairs = []

        for obj_prop in object_proposals:
            for part_prop in part_proposals:
                # Skip self-pairing
                if (obj_prop.object_id is not None and
                    part_prop.object_id is not None and
                    obj_prop.object_id == part_prop.object_id):
                    continue

                # Tree-based pruning
                if use_tree_pruning and self.family_tree is not None:
                    if not self._is_feasible_pair(obj_prop, part_prop):
                        continue

                # Geometric validation
                is_valid, metrics = check_geometric_validity(
                    part_prop.mask,
                    obj_prop.mask,
                    points=points,
                    containment_threshold=self.containment_threshold,
                    scale_range=self.scale_range
                )

                if not is_valid:
                    continue

                # Compute scores
                geo_score = 0.0
                if self.use_geometric_score:
                    geo_score = compute_geometric_score(
                        part_prop.mask,
                        obj_prop.mask,
                        points=points
                    )

                # Combine scores
                combined = self.score_aggregator.aggregate(
                    obj_prop.score,
                    part_prop.score,
                    geo_score if self.use_geometric_score else None
                )

                # Create pair
                pair = ObjectPartPair(
                    object_proposal=obj_prop,
                    part_proposal=part_prop,
                    object_score=obj_prop.score,
                    part_score=part_prop.score,
                    geometric_score=geo_score,
                    combined_score=combined,
                    label_id=label_id
                )

                pairs.append(pair)

        logger.info(f"Created {len(pairs)} valid pairs from "
                   f"{len(object_proposals)} objects × {len(part_proposals)} parts")

        return pairs

    def _is_feasible_pair(
        self,
        obj_prop: Proposal,
        part_prop: Proposal
    ) -> bool:
        """
        Check if an object-part pair is feasible based on family tree.

        Args:
            obj_prop: Object proposal
            part_prop: Part proposal

        Returns:
            True if pair is feasible
        """
        if self.family_tree is None:
            return True

        if obj_prop.object_id is None or part_prop.object_id is None:
            # No tree info available, allow pairing
            return True

        # Part should be a descendant of object
        if self.family_tree.is_ancestor(obj_prop.object_id, part_prop.object_id):
            return True

        # Allow same-level pairing if they are close (will be checked by geometry)
        if obj_prop.level == part_prop.level:
            return True

        # Allow sibling's children (relaxed constraint)
        # This would require parent lookup - skip for now

        return False

    def pair_with_tree_guidance(
        self,
        object_proposals: List[Proposal],
        part_proposals: List[Proposal],
        label_id: int,
        points: Optional[np.ndarray] = None
    ) -> List[ObjectPartPair]:
        """
        Pair proposals using family tree to guide the search.

        This is more efficient than pair_all as it only checks feasible pairs.

        Args:
            object_proposals: Object proposals
            part_proposals: Part proposals
            label_id: Joint semantic label
            points: Point cloud coordinates

        Returns:
            List of valid pairs
        """
        if self.family_tree is None:
            logger.warning("No family tree provided, falling back to pair_all")
            return self.pair_all(
                object_proposals, part_proposals, label_id, points,
                use_tree_pruning=False
            )

        pairs = []

        # Build part proposal lookup by object_id
        part_by_id = {p.object_id: p for p in part_proposals if p.object_id is not None}

        for obj_prop in object_proposals:
            if obj_prop.object_id is None:
                continue

            # Get descendant IDs
            descendant_ids = self.family_tree.get_descendants(obj_prop.object_id)

            # Find part proposals that are descendants
            candidate_parts = [
                part_by_id[did] for did in descendant_ids
                if did in part_by_id
            ]

            # Also consider same-level parts (will be filtered by geometry)
            same_level_parts = [
                p for p in part_proposals
                if p.level == obj_prop.level and p.object_id != obj_prop.object_id
            ]

            candidate_parts.extend(same_level_parts)

            # Pair with candidates
            for part_prop in candidate_parts:
                is_valid, metrics = check_geometric_validity(
                    part_prop.mask,
                    obj_prop.mask,
                    points=points,
                    containment_threshold=self.containment_threshold,
                    scale_range=self.scale_range
                )

                if not is_valid:
                    continue

                geo_score = 0.0
                if self.use_geometric_score:
                    geo_score = compute_geometric_score(
                        part_prop.mask, obj_prop.mask, points
                    )

                combined = self.score_aggregator.aggregate(
                    obj_prop.score, part_prop.score,
                    geo_score if self.use_geometric_score else None
                )

                pair = ObjectPartPair(
                    object_proposal=obj_prop,
                    part_proposal=part_prop,
                    object_score=obj_prop.score,
                    part_score=part_prop.score,
                    geometric_score=geo_score,
                    combined_score=combined,
                    label_id=label_id
                )

                pairs.append(pair)

        logger.info(f"Tree-guided pairing: {len(pairs)} valid pairs")
        return pairs
