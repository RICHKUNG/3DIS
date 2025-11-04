"""
Provenance Tracker - Maps SSAM candidates to SAM2 tracked objects.

This module implements a unified mapping approach where both accepted and rejected
SSAM prompts are tracked to their corresponding SAM2 object IDs. This enables
accurate parent-child relationship reconstruction without relying on post-hoc
geometric containment analysis.

Key Design:
- Unified mapping: Single dict ssam_to_sam2 for both accepted and rejected prompts
- O(1) parent lookup: No need for multiple dictionary lookups
- Orphan detection: Identify objects that should have parents but don't
"""

from typing import Dict, Optional, Any, List


class ProvenanceTracker:
    """
    Track SSAM candidate → SAM2 object mappings using unified mapping approach.

    Core Concept:
    - Each SSAM unique_id maps to exactly one SAM2 object ID
    - Whether that SSAM mask was accepted (new tracking) or rejected (dedup match)
    - Single unified mapping table, making parent lookup O(1)

    Core Functionality:
    1. Record SSAM → SAM2 mappings (accepted and rejected unified)
    2. Record SAM2 objects' complete provenance metadata
    3. Query parent relationships: ssam_parent_id → sam2_parent_obj_id (O(1))
    """

    def __init__(self):
        # Unified mapping: SSAM ID → SAM2 object ID
        # Includes both accepted prompts and rejected prompts (dedup matches)
        self.ssam_to_sam2: Dict[str, int] = {}

        # SAM2 object ID → complete provenance metadata
        # Note: Only accepted prompts have provenance (rejected aren't new objects)
        self.sam2_provenance: Dict[int, Dict[str, Any]] = {}

        # Track unresolved rejections (for debugging orphan issues)
        # List of (ssam_unique_id, matched_mask_idx, frame_idx) tuples
        self.unresolved_rejections: List[tuple] = []

        # Virtual children: SAM2 parent ID → list of child SSAM IDs that were deduped
        # These children were rejected (matched existing parent) but conceptually belong
        # to the parent's family tree
        self.virtual_children: Dict[int, List[str]] = {}

    def register_accepted_prompt(
        self,
        sam2_obj_id: int,
        ssam_unique_id: str,
        ssam_parent_id: Optional[str],
        ssam_frame_idx: int,
        level: int,
        lineage: Optional[List[str]] = None,
    ):
        """
        Record when SSAM candidate is accepted as new SAM2 tracking prompt.

        This SSAM mask starts being tracked by SAM2, producing a new tracked object.

        Args:
            sam2_obj_id: SAM2-assigned object ID
            ssam_unique_id: SSAM composite ID (e.g., "0050_2_0001")
            ssam_parent_id: Parent's SSAM ID (None for root level)
            ssam_frame_idx: Frame index where SSAM generated this mask
            level: SSAM level (2, 4, 6, etc.)
            lineage: Complete ancestor chain (optional)
        """
        self.ssam_to_sam2[ssam_unique_id] = sam2_obj_id
        self.sam2_provenance[sam2_obj_id] = {
            "ssam_unique_id": ssam_unique_id,
            "ssam_parent_id": ssam_parent_id,
            "ssam_frame_idx": ssam_frame_idx,
            "level": level,
            "lineage": lineage or [],
        }

    def register_rejected_prompt(
        self,
        ssam_unique_id: str,
        matched_sam2_obj_id: int,
        ssam_parent_id: Optional[str] = None,
    ):
        """
        Record when SSAM candidate is rejected due to dedup IoU match.

        This SSAM mask doesn't start new tracking; it's identified as existing object.
        Record this mapping so subsequent parent lookups can find the SAM2 ID.

        If ssam_parent_id is provided, also track this as a virtual child relationship:
        - Find the parent's SAM2 ID from ssam_parent_id
        - Record ssam_unique_id as virtual child of that parent
        - Virtual children represent "family merging" - children that were deduped
          into their parent's mask but conceptually belong to the family tree

        Args:
            ssam_unique_id: SSAM composite ID that was rejected
            matched_sam2_obj_id: Existing SAM2 object it matched
            ssam_parent_id: Optional SSAM parent ID (for virtual children tracking)

        Note: If ssam_unique_id already recorded, keep first encounter mapping.
        """
        if ssam_unique_id not in self.ssam_to_sam2:
            self.ssam_to_sam2[ssam_unique_id] = matched_sam2_obj_id

        # Track virtual children if parent info provided
        if ssam_parent_id is not None:
            # Find parent's SAM2 ID
            parent_sam2_id = self.ssam_to_sam2.get(ssam_parent_id)
            if parent_sam2_id is not None:
                # Record this SSAM mask as virtual child of parent
                if parent_sam2_id not in self.virtual_children:
                    self.virtual_children[parent_sam2_id] = []
                if ssam_unique_id not in self.virtual_children[parent_sam2_id]:
                    self.virtual_children[parent_sam2_id].append(ssam_unique_id)

    def register_unresolved_rejection(
        self, ssam_unique_id: str, matched_mask_idx: int, frame_idx: int
    ):
        """
        Record when a rejection cannot be resolved (mapping not found).

        This helps diagnose orphan issues caused by timing or mapping problems.

        Args:
            ssam_unique_id: SSAM composite ID that was rejected
            matched_mask_idx: Index of matched mask in dedup store
            frame_idx: Frame index where rejection occurred
        """
        self.unresolved_rejections.append((ssam_unique_id, matched_mask_idx, frame_idx))

    def find_parent(self, sam2_obj_id: int) -> Optional[int]:
        """
        Find parent SAM2 object ID for given object.

        Process:
        1. Get ssam_parent_id from sam2_provenance
        2. Lookup ssam_to_sam2 mapping to find parent's SAM2 ID
        3. Not found → None (orphan)

        Complexity: O(1) single dictionary lookup

        Args:
            sam2_obj_id: SAM2 object to find parent for

        Returns:
            Parent's SAM2 object ID, or None if orphan/root
        """
        prov = self.sam2_provenance.get(sam2_obj_id)
        if prov is None:
            return None

        ssam_parent_id = prov.get("ssam_parent_id")
        if ssam_parent_id is None:
            return None  # Root level object (e.g., L2)

        # Single lookup! Works for both accepted and rejected parents
        return self.ssam_to_sam2.get(ssam_parent_id)

    def get_virtual_children(self, sam2_obj_id: int) -> List[str]:
        """
        Get list of virtual children for a SAM2 object.

        Virtual children are SSAM masks that were rejected (deduped) but conceptually
        belong to this object's family tree. They represent "family merging" where
        child-level masks were identified as the same object as the parent.

        Args:
            sam2_obj_id: SAM2 object ID to query

        Returns:
            List of SSAM unique IDs that are virtual children of this object
        """
        return self.virtual_children.get(sam2_obj_id, [])

    def has_virtual_children(self, sam2_obj_id: int) -> bool:
        """
        Check if a SAM2 object has any virtual children.

        Args:
            sam2_obj_id: SAM2 object ID to check

        Returns:
            True if object has virtual children, False otherwise
        """
        return sam2_obj_id in self.virtual_children and len(self.virtual_children[sam2_obj_id]) > 0

    def get_family_hierarchy(self) -> Dict[str, Any]:
        """
        Export complete parent map: {child_sam2_id: parent_sam2_id}.

        Also compute orphan statistics (objects with ssam_parent_id but no SAM2 parent).

        Returns:
            Dictionary with:
            - parent_map: {child_id: parent_id} for all tracked objects
            - orphan_count: Number of orphans
            - orphan_ids: List of orphan SAM2 IDs
        """
        parent_map = {}
        orphans = []

        for sam2_obj_id in self.sam2_provenance.keys():
            parent_id = self.find_parent(sam2_obj_id)
            parent_map[sam2_obj_id] = parent_id

            # Count orphans (child level but no parent)
            prov = self.sam2_provenance[sam2_obj_id]
            if prov.get("ssam_parent_id") is not None and parent_id is None:
                orphans.append(sam2_obj_id)

        return {
            "parent_map": parent_map,
            "orphan_count": len(orphans),
            "orphan_ids": orphans,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Export statistics about SSAM → SAM2 mapping.

        Note:
        - accepted_prompts = len(sam2_provenance) (only accepted have provenance)
        - rejected_prompts = len(ssam_to_sam2) - len(sam2_provenance)

        Returns:
            Dictionary with:
            - total_ssam_masks: Total SSAM masks encountered
            - accepted_prompts: How many became new SAM2 objects
            - rejected_prompts: How many were dedup-rejected
            - rejection_rate: Proportion rejected
            - unresolved_rejections: How many rejections couldn't be mapped
            - virtual_children_count: Total virtual children across all parents
            - parents_with_virtual_children: Number of parents with virtual children
        """
        accepted_count = len(self.sam2_provenance)
        total_count = len(self.ssam_to_sam2)
        rejected_count = total_count - accepted_count
        unresolved_count = len(self.unresolved_rejections)

        # Virtual children statistics
        total_virtual_children = sum(len(children) for children in self.virtual_children.values())
        parents_with_virtual = len(self.virtual_children)

        return {
            "total_ssam_masks": total_count,
            "accepted_prompts": accepted_count,
            "rejected_prompts": rejected_count,
            "rejection_rate": rejected_count / total_count if total_count > 0 else 0.0,
            "unresolved_rejections": unresolved_count,
            "virtual_children_count": total_virtual_children,
            "parents_with_virtual_children": parents_with_virtual,
        }

    def export_full_metadata(self) -> Dict[str, Any]:
        """
        Export complete provenance data for serialization.

        Returns:
            Dictionary with all mappings and metadata for saving to JSON.
        """
        return {
            "ssam_to_sam2": self.ssam_to_sam2,
            "sam2_provenance": self.sam2_provenance,
            "virtual_children": {
                str(parent_id): children
                for parent_id, children in self.virtual_children.items()
            },
            "unresolved_rejections": [
                {
                    "ssam_unique_id": ssam_id,
                    "matched_mask_idx": mask_idx,
                    "frame_idx": frame_idx,
                }
                for ssam_id, mask_idx, frame_idx in self.unresolved_rejections
            ],
        }
