"""
Cascade Filtering - Parent-Aware SSAM Mask Filtering.

This module implements filtering logic that respects parent-child relationships:
- If a parent mask is filtered out, all its descendants are also filtered (cascade)
- This prevents orphans caused by filtered parents with accepted children
- Maintains tree structure integrity in the final output

Key Concept:
    Traditional filtering: Apply thresholds independently to each mask
    Cascade filtering: Apply thresholds with tree-aware cascading
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class CascadeFilter:
    """
    Filter SSAM masks with parent-child awareness.

    Filtering process:
    1. Build parent-child graph from unique_ids
    2. Apply thresholds to individual masks
    3. Identify filtered parents
    4. Cascade filter all descendants of filtered parents
    5. Return filtered masks + rejection metadata
    """

    def __init__(
        self,
        min_area: Optional[float] = None,
        stability_threshold: Optional[float] = None,
    ):
        """
        Initialize cascade filter with thresholds.

        Args:
            min_area: Minimum mask area (None = no area filter)
            stability_threshold: Minimum stability score (None = no stability filter)
        """
        self.min_area = min_area
        self.stability_threshold = stability_threshold

    def filter_masks(
        self,
        masks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Filter masks with cascade logic.

        Args:
            masks: List of mask dicts with keys:
                - unique_id: SSAM composite ID
                - parent_unique_id: Parent's SSAM ID (None for roots)
                - area: Mask area
                - stability: Stability score
                - ... (other mask metadata)

        Returns:
            Tuple of:
            - filtered_masks: Masks that passed filtering
            - rejection_reasons: Dict[unique_id -> reason] for rejected masks
        """
        if not masks:
            return [], {}

        # Step 1: Build parent-child graph
        children_map: Dict[Optional[str], List[str]] = defaultdict(list)
        mask_by_id: Dict[str, Dict[str, Any]] = {}

        for mask in masks:
            unique_id = mask.get("unique_id")
            parent_id = mask.get("parent_unique_id")

            if unique_id:
                mask_by_id[unique_id] = mask
                children_map[parent_id].append(unique_id)

        # Step 2: Apply individual thresholds
        individually_rejected: Dict[str, str] = {}

        for mask in masks:
            unique_id = mask.get("unique_id")
            if not unique_id:
                continue

            # Area threshold
            if self.min_area is not None:
                area = mask.get("area", 0)
                if area < self.min_area:
                    individually_rejected[unique_id] = f"area_too_small ({area:.1f} < {self.min_area})"
                    continue

            # Stability threshold
            if self.stability_threshold is not None:
                stability = mask.get("stability")
                if stability is None:
                    stability = mask.get("stability_score")
                if stability is None:
                    stability = mask.get("stabilityScore")
                if stability is None:
                    stability = mask.get("stability-score")
                if stability is None:
                    stability = 0
                try:
                    stability = float(stability)
                except (TypeError, ValueError):
                    stability = 0
                if stability < self.stability_threshold:
                    individually_rejected[unique_id] = f"low_stability ({stability:.3f} < {self.stability_threshold})"
                    continue

        # Step 3: Cascade filtering - remove descendants of filtered parents
        cascade_rejected: Dict[str, str] = {}

        def cascade_reject(parent_id: str, reason_prefix: str):
            """Recursively reject all descendants of a filtered parent."""
            for child_id in children_map.get(parent_id, []):
                if child_id not in individually_rejected and child_id not in cascade_rejected:
                    # This child passed individual thresholds but parent was filtered
                    original_reason = individually_rejected.get(parent_id, "unknown")
                    cascade_rejected[child_id] = f"parent_filtered (parent: {parent_id}, reason: {original_reason})"

                # Recurse to grandchildren
                cascade_reject(child_id, reason_prefix)

        # Cascade from all individually rejected parents
        for rejected_id in individually_rejected.keys():
            cascade_reject(rejected_id, individually_rejected[rejected_id])

        # Step 4: Collect all rejections
        all_rejections = {**individually_rejected, **cascade_rejected}

        # Step 5: Build filtered output (masks NOT rejected)
        # BUGFIX (2025-11-13): Also reject masks without unique_id
        # Previously, masks with None/missing unique_id would bypass all filters
        filtered_masks = []
        for mask in masks:
            unique_id = mask.get("unique_id")

            # Reject masks without unique_id
            if not unique_id:
                # Don't add to rejections dict (can't use None as key), but don't include in output
                continue

            # Reject masks that failed filtering
            if unique_id in all_rejections:
                continue

            # Mask passed all checks
            filtered_masks.append(mask)

        return filtered_masks, all_rejections

    def get_statistics(
        self, total_masks: int, rejection_reasons: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Compute filtering statistics.

        Args:
            total_masks: Total number of input masks
            rejection_reasons: Dict of unique_id -> reason

        Returns:
            Statistics dict with counts by rejection type
        """
        stats: Dict[str, int] = defaultdict(int)

        for reason in rejection_reasons.values():
            if reason.startswith("area_too_small"):
                stats["area_rejected"] += 1
            elif reason.startswith("low_stability"):
                stats["stability_rejected"] += 1
            elif reason.startswith("parent_filtered"):
                stats["cascade_rejected"] += 1
            else:
                stats["other_rejected"] += 1

        passed = total_masks - len(rejection_reasons)

        return {
            "total_input": total_masks,
            "passed": passed,
            "rejected": len(rejection_reasons),
            "rejection_rate": len(rejection_reasons) / total_masks if total_masks > 0 else 0.0,
            "breakdown": dict(stats),
        }


def cascade_filter_masks(
    masks: List[Dict[str, Any]],
    min_area: Optional[float] = None,
    stability_threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Convenience function for cascade filtering.

    Args:
        masks: List of mask dicts
        min_area: Minimum mask area (None = no area filter)
        stability_threshold: Minimum stability score (None = no stability filter)

    Returns:
        Tuple of (filtered_masks, rejection_reasons)
    """
    filter_obj = CascadeFilter(min_area=min_area, stability_threshold=stability_threshold)
    return filter_obj.filter_masks(masks)


def validate_parent_child_consistency(masks: List[Dict[str, Any]]) -> List[str]:
    """
    Validate that all parent_unique_id references are valid.

    This helps detect cases where filtering would create orphans.

    Args:
        masks: List of mask dicts

    Returns:
        List of error messages (empty if consistent)
    """
    errors = []
    all_ids = {mask.get("unique_id") for mask in masks if mask.get("unique_id")}

    for mask in masks:
        unique_id = mask.get("unique_id")
        parent_id = mask.get("parent_unique_id")

        if parent_id is not None and parent_id not in all_ids:
            errors.append(f"Mask {unique_id} references non-existent parent {parent_id}")

    return errors
