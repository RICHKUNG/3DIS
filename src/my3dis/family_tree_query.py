"""
Family Tree Query Functions - Query frames and masks from unified family trees.

Provides convenient functions to:
- Query which frames an object appears in
- Load binary masks for specific object+frame combinations
- Find related objects (parents, children, siblings)
- Get family members
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class FamilyTreeQuery:
    """Query interface for unified family trees."""

    def __init__(self, family_tree_path: str):
        """
        Initialize query interface from family tree JSON.

        Args:
            family_tree_path: Path to family_tree.json file
        """
        with open(family_tree_path, 'r', encoding='utf-8') as f:
            self.tree = json.load(f)

        self.run_dir = Path(family_tree_path).parent.parent
        self.objects = {int(k): v for k, v in self.tree['objects'].items()}
        self._archive_cache: Dict[str, Any] = {}

    def get_object_info(self, obj_id: int) -> Optional[Dict[str, Any]]:
        """Get full information about an object."""
        return self.objects.get(obj_id)

    def get_object_frames(self, obj_id: int) -> List[int]:
        """Get list of frame indices where object appears."""
        obj_info = self.objects.get(obj_id)
        if obj_info is None:
            return []
        return obj_info.get('frames', [])

    def get_parent(self, obj_id: int) -> Optional[int]:
        """Get parent object ID."""
        obj_info = self.objects.get(obj_id)
        if obj_info is None:
            return None
        return obj_info.get('parent_id')

    def get_children(self, obj_id: int) -> List[int]:
        """Get list of child object IDs."""
        obj_info = self.objects.get(obj_id)
        if obj_info is None:
            return []
        return obj_info.get('children', [])

    def get_virtual_children(self, obj_id: int) -> List[str]:
        """
        Get list of virtual children (SSAM IDs) for this object.

        Virtual children are SSAM masks that were rejected (deduped) but conceptually
        belong to this object's family tree. They represent "family merging" where
        child-level masks were identified as the same object as the parent.

        Args:
            obj_id: Object ID to query

        Returns:
            List of SSAM unique IDs that are virtual children of this object
        """
        obj_info = self.objects.get(obj_id)
        if obj_info is None:
            return []
        return obj_info.get('virtual_children', [])

    def has_virtual_children(self, obj_id: int) -> bool:
        """
        Check if an object has any virtual children.

        Args:
            obj_id: Object ID to check

        Returns:
            True if object has virtual children, False otherwise
        """
        virtual_children = self.get_virtual_children(obj_id)
        return len(virtual_children) > 0

    def get_all_children(self, obj_id: int, include_virtual: bool = False) -> Dict[str, List]:
        """
        Get both regular children and virtual children.

        Args:
            obj_id: Object ID to query
            include_virtual: If True, include virtual children in results

        Returns:
            Dictionary with 'children' (object IDs) and optionally 'virtual_children' (SSAM IDs)
        """
        result = {
            'children': self.get_children(obj_id),
        }
        if include_virtual:
            result['virtual_children'] = self.get_virtual_children(obj_id)
        return result

    def get_siblings(self, obj_id: int) -> List[int]:
        """Get list of sibling object IDs (objects with same parent)."""
        parent_id = self.get_parent(obj_id)
        if parent_id is None:
            return []

        siblings = self.get_children(parent_id)
        return [s for s in siblings if s != obj_id]

    def get_family_root(self, obj_id: int) -> Optional[int]:
        """Get root object ID of the family this object belongs to."""
        for family in self.tree['families']:
            if obj_id in family['members']:
                return family['root_id']
        return None

    def get_family_members(self, obj_id: int) -> List[int]:
        """Get all members of the family this object belongs to."""
        for family in self.tree['families']:
            if obj_id in family['members']:
                return family['members']
        return [obj_id]  # Orphan - only itself

    def get_common_frames(self, obj_ids: List[int]) -> List[int]:
        """Get frames where all specified objects appear together."""
        if not obj_ids:
            return []

        frame_sets = [set(self.get_object_frames(obj_id)) for obj_id in obj_ids]
        common = set.intersection(*frame_sets) if frame_sets else set()
        return sorted(common)

    def load_mask(self, obj_id: int, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load binary mask for object at specific frame.

        Args:
            obj_id: Object ID
            frame_idx: Frame index

        Returns:
            Binary mask as (H, W) numpy array, or None if not found
        """
        obj_info = self.objects.get(obj_id)
        if obj_info is None:
            import warnings
            warnings.warn(f"Object {obj_id} not found in family tree")
            return None

        if frame_idx not in obj_info.get('frames', []):
            import warnings
            warnings.warn(f"Object {obj_id} does not appear in frame {frame_idx}")
            return None

        # Load from video segments archive
        archive_rel_path = obj_info.get('mask_archive')
        if archive_rel_path is None:
            import warnings
            warnings.warn(f"Object {obj_id} has no mask_archive path")
            return None

        archive_path = self.run_dir / archive_rel_path

        # If path doesn't exist, try fallback paths (for backward compatibility)
        if not archive_path.exists():
            level = obj_info.get('level')
            if level is not None:
                fallback_candidates = [
                    # New format (L-prefixed, directly in level dir) - PRIORITY 1
                    self.run_dir / f'level_{level}' / f'video_segments_L{level:02d}.npz',
                    # New format variant (in tracking subdir) - PRIORITY 2
                    self.run_dir / f'level_{level}' / 'tracking' / f'video_segments_L{level:02d}.npz',
                    # Old format (scale-based naming) - PRIORITY 3
                    self.run_dir / f'level_{level}' / 'tracking' / f'video_segments_scale0.3x.npz',
                    self.run_dir / f'level_{level}' / f'video_segments_scale0.3x.npz',
                ]
                for candidate in fallback_candidates:
                    if candidate.exists():
                        archive_path = candidate
                        break

        # Check cache first
        archive_key = str(archive_path)
        if archive_key not in self._archive_cache:
            if not archive_path.exists():
                import warnings
                warnings.warn(f"Mask archive not found: {archive_path}")
                return None
            self._archive_cache[archive_key] = np.load(archive_path, allow_pickle=True)

        archive = self._archive_cache[archive_key]

        # Detect format and load frame data
        frame_data = None

        # Try new ZIP format first (frames/frame_XXXXXX.json)
        frame_key_new = f"frames/frame_{frame_idx:06d}.json"
        if frame_key_new in archive.files:
            try:
                data_bytes = bytes(archive[frame_key_new])
                import json
                frame_dict = json.loads(data_bytes.decode('utf-8'))
                frame_data = frame_dict.get('objects', {})
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                import warnings
                warnings.warn(f"Failed to parse new format frame data: {e}")
                return None
        else:
            # Try legacy format (frame_XXXXXX)
            frame_key_old = f"frame_{frame_idx:06d}"
            if frame_key_old in archive.files:
                try:
                    frame_data = archive[frame_key_old].item()
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to load legacy format frame data: {e}")
                    return None
            else:
                import warnings
                warnings.warn(f"Frame {frame_idx} not found in archive (tried: {frame_key_new}, {frame_key_old})")
                return None

        if not isinstance(frame_data, dict):
            import warnings
            warnings.warn(f"Invalid frame data type: {type(frame_data)}")
            return None

        # Extract object mask
        obj_data = frame_data.get(str(obj_id))
        if obj_data is None:
            import warnings
            warnings.warn(f"Object {obj_id} not found in frame {frame_idx} data")
            return None

        # Decode packed binary mask
        from my3dis.common_utils import unpack_binary_mask

        return unpack_binary_mask(obj_data)

    def load_masks_batch(
        self, obj_id: int, frame_indices: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Load masks for multiple frames (more efficient than loading one by one).

        Args:
            obj_id: Object ID
            frame_indices: Frame indices to load (None = all frames)

        Returns:
            Dict mapping frame_idx -> binary mask
        """
        obj_info = self.objects.get(obj_id)
        if obj_info is None:
            return {}

        if frame_indices is None:
            frame_indices = obj_info.get('frames', [])
        else:
            # Filter to only frames where object actually appears
            available_frames = set(obj_info.get('frames', []))
            frame_indices = [f for f in frame_indices if f in available_frames]

        masks = {}
        for frame_idx in frame_indices:
            mask = self.load_mask(obj_id, frame_idx)
            if mask is not None:
                masks[frame_idx] = mask

        return masks

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from the tree."""
        return self.tree.get('statistics', {})

    def search_objects_by_level(self, level: int) -> List[int]:
        """Find all objects at a specific level."""
        return [
            obj_id
            for obj_id, obj_info in self.objects.items()
            if obj_info.get('level') == level
        ]

    def search_objects_by_frame(self, frame_idx: int) -> List[int]:
        """Find all objects that appear in a specific frame."""
        return [
            obj_id
            for obj_id, obj_info in self.objects.items()
            if frame_idx in obj_info.get('frames', [])
        ]

    def export_family_subgraph(self, obj_id: int, output_path: str) -> None:
        """
        Export the family subgraph containing this object to a separate JSON.

        Useful for visualizing or analyzing a single family in isolation.
        """
        family_members = self.get_family_members(obj_id)

        subgraph_objects = {
            str(member_id): self.objects[member_id]
            for member_id in family_members
            if member_id in self.objects
        }

        # Find the family entry
        family_entry = None
        for family in self.tree['families']:
            if obj_id in family['members']:
                family_entry = family
                break

        subgraph = {
            'scene': self.tree['scene'],
            'run_dir': self.tree['run_dir'],
            'levels': self.tree['levels'],
            'mask_scale_ratio': self.tree['mask_scale_ratio'],
            'objects': subgraph_objects,
            'family': family_entry,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subgraph, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point for querying family trees."""
    import argparse

    parser = argparse.ArgumentParser(description='Query family tree')
    parser.add_argument('--tree', required=True, help='Path to family_tree.json')
    parser.add_argument('--obj-id', type=int, help='Object ID to query')
    parser.add_argument('--list-families', action='store_true', help='List all families')
    parser.add_argument('--stats', action='store_true', help='Show statistics')

    args = parser.parse_args()

    query = FamilyTreeQuery(args.tree)

    if args.stats:
        stats = query.get_statistics()
        print("=== Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    if args.list_families:
        print("\n=== Families ===")
        for i, family in enumerate(query.tree['families'], 1):
            print(f"Family {i}:")
            print(f"  Root: {family['root_id']}")
            print(f"  Members: {family['total_members']}")
            print(f"  Levels: {family['levels']}")

    if args.obj_id is not None:
        print(f"\n=== Object {args.obj_id} ===")
        info = query.get_object_info(args.obj_id)
        if info is None:
            print("  Not found")
        else:
            print(f"  Level: {info['level']}")
            print(f"  Parent: {info['parent_id']}")
            print(f"  Children: {info['children']}")

            virtual_children = query.get_virtual_children(args.obj_id)
            if virtual_children:
                print(f"  Virtual children: {len(virtual_children)} SSAM masks")
                # Show first few for brevity
                if len(virtual_children) <= 5:
                    print(f"    {virtual_children}")
                else:
                    print(f"    {virtual_children[:5]} ... and {len(virtual_children) - 5} more")

            print(f"  Frames: {len(info['frames'])} total")
            print(f"    First: {info['first_frame']}, Last: {info['last_frame']}")

            family_members = query.get_family_members(args.obj_id)
            print(f"  Family size: {len(family_members)} members")


if __name__ == '__main__':
    main()
