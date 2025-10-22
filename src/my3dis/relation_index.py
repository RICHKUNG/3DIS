"""
Relation indexing system for cross-level object hierarchy tracking.

Provides lightweight JSON indices that link object IDs, frames, and parent-child
relationships without duplicating binary mask data.

Author: Rich Kung
Created: 2025-10-21
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from my3dis.common_utils import ensure_dir

LOGGER = logging.getLogger(__name__)

__all__ = [
    'RelationIndexWriter',
    'RelationIndexReader',
    'build_level_index',
    'build_cross_level_relations',
    'save_ssam_relations',
]


class RelationIndexWriter:
    """Build and write relation index files for a single level."""

    def __init__(
        self,
        level: int,
        level_root: str | Path,
        mask_scale_ratio: float = 1.0,
    ) -> None:
        self.level = int(level)
        self.level_root = Path(level_root)
        self.mask_scale_ratio = float(mask_scale_ratio)

        self._objects: Dict[int, Dict[str, Any]] = {}
        self._frames: Dict[int, Dict[str, Any]] = {}
        self._sources: Dict[str, str] = {}

    def add_object(
        self,
        obj_id: int,
        parent_id: Optional[int] = None,
        frames: Optional[List[int]] = None,
    ) -> None:
        """Register an object with its frame appearances."""
        if frames is None:
            frames = []

        if obj_id not in self._objects:
            self._objects[obj_id] = {
                'parent_id': parent_id,
                'frames': sorted(set(frames)),
                'frame_count': len(frames),
            }
        else:
            # Merge frames
            existing = set(self._objects[obj_id]['frames'])
            existing.update(frames)
            self._objects[obj_id]['frames'] = sorted(existing)
            self._objects[obj_id]['frame_count'] = len(existing)

            if parent_id is not None:
                self._objects[obj_id]['parent_id'] = parent_id

    def add_frame(
        self,
        frame_idx: int,
        frame_name: str,
        objects: List[int],
    ) -> None:
        """Register a frame with its visible objects."""
        self._frames[frame_idx] = {
            'frame_name': frame_name,
            'objects': sorted(set(objects)),
        }

    def set_source(self, key: str, path: str) -> None:
        """Record a data source path (relative to level_root)."""
        self._sources[key] = path

    def write(self, output_path: Optional[str | Path] = None) -> Path:
        """Write index.json to disk."""
        if output_path is None:
            output_path = self.level_root / 'index.json'
        else:
            output_path = Path(output_path)

        ensure_dir(output_path.parent)

        # Compute first/last frame for each object
        for obj_id, obj_data in self._objects.items():
            frames = obj_data['frames']
            if frames:
                obj_data['first_frame'] = frames[0]
                obj_data['last_frame'] = frames[-1]

        payload = {
            'meta': {
                'level': self.level,
                'mask_scale_ratio': self.mask_scale_ratio,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'sources': self._sources,
            },
            'objects': {str(k): v for k, v in sorted(self._objects.items())},
            'frames': {str(k): v for k, v in sorted(self._frames.items())},
        }

        with output_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        LOGGER.info(
            "Wrote level %d index: %d objects, %d frames → %s",
            self.level,
            len(self._objects),
            len(self._frames),
            output_path,
        )
        return output_path


class RelationIndexReader:
    """Read and query relation index files."""

    def __init__(self, level_root: str | Path) -> None:
        self.level_root = Path(level_root)
        self.index_path = self.level_root / 'index.json'

        self._data: Optional[Dict[str, Any]] = None
        self._objects: Dict[int, Dict[str, Any]] = {}
        self._frames: Dict[int, Dict[str, Any]] = {}

        if self.index_path.exists():
            self._load()

    def _load(self) -> None:
        with self.index_path.open('r', encoding='utf-8') as f:
            self._data = json.load(f)

        self._objects = {
            int(k): v for k, v in self._data.get('objects', {}).items()
        }
        self._frames = {
            int(k): v for k, v in self._data.get('frames', {}).items()
        }

    @property
    def level(self) -> Optional[int]:
        if self._data is None:
            return None
        return self._data.get('meta', {}).get('level')

    @property
    def objects(self) -> Dict[int, Dict[str, Any]]:
        return self._objects

    @property
    def frames(self) -> Dict[int, Dict[str, Any]]:
        return self._frames

    def get_object(self, obj_id: int) -> Optional[Dict[str, Any]]:
        return self._objects.get(obj_id)

    def get_frame(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        return self._frames.get(frame_idx)

    def get_source_path(self, key: str) -> Optional[str]:
        if self._data is None:
            return None
        sources = self._data.get('meta', {}).get('sources', {})
        return sources.get(key)


def build_level_index(
    level: int,
    level_root: str | Path,
    video_segments_path: Optional[str | Path] = None,
    object_segments_path: Optional[str | Path] = None,
    candidates_manifest: Optional[str | Path] = None,
    mask_scale_ratio: float = 1.0,
) -> Path:
    """
    Build index.json for a single level from tracking outputs.

    Scans video_segments and object_segments to extract object-frame mappings.
    """
    level_root = Path(level_root)
    writer = RelationIndexWriter(level, level_root, mask_scale_ratio)

    # Register source paths (relative to level_root)
    if video_segments_path:
        rel_path = os.path.relpath(video_segments_path, level_root)
        writer.set_source('video_segments', rel_path)

    if object_segments_path:
        rel_path = os.path.relpath(object_segments_path, level_root)
        writer.set_source('object_segments', rel_path)

    if candidates_manifest:
        rel_path = os.path.relpath(candidates_manifest, level_root)
        writer.set_source('filtered_candidates', rel_path)

    # Extract object-frame mappings from object_segments
    if object_segments_path and Path(object_segments_path).exists():
        _extract_from_object_segments(writer, object_segments_path)

    # Extract frame-object mappings from video_segments
    if video_segments_path and Path(video_segments_path).exists():
        _extract_from_video_segments(writer, video_segments_path)

    return writer.write()


def _extract_from_object_segments(
    writer: RelationIndexWriter,
    path: str | Path,
) -> None:
    """Read object_segments NPZ and populate object→frames mapping."""
    import zipfile

    try:
        with zipfile.ZipFile(path, 'r') as zf:
            manifest_data = zf.read('manifest.json')
            manifest = json.loads(manifest_data.decode('utf-8'))

            objects = manifest.get('objects', {})
            for obj_id_str, entries in objects.items():
                obj_id = int(obj_id_str)
                frames = [int(e['frame_index']) for e in entries]
                writer.add_object(obj_id, frames=frames)
    except Exception as exc:
        LOGGER.warning("Failed to read object_segments at %s: %s", path, exc)


def _extract_from_video_segments(
    writer: RelationIndexWriter,
    path: str | Path,
) -> None:
    """Read video_segments NPZ and populate frame→objects mapping."""
    import zipfile

    try:
        with zipfile.ZipFile(path, 'r') as zf:
            manifest_data = zf.read('manifest.json')
            manifest = json.loads(manifest_data.decode('utf-8'))

            frames = manifest.get('frames', [])
            for frame_entry in frames:
                frame_idx = int(frame_entry['frame_index'])
                frame_name = frame_entry.get('frame_name', '')
                objects = [int(oid) for oid in frame_entry.get('objects', [])]
                writer.add_frame(frame_idx, frame_name, objects)
    except Exception as exc:
        LOGGER.warning("Failed to read video_segments at %s: %s", path, exc)


def build_cross_level_relations(
    run_dir: str | Path,
    levels: List[int],
) -> Path:
    """
    Build relations.json by merging parent-child info from all levels.

    Reads relations/tree.json from each level and constructs the cross-level hierarchy.
    """
    run_dir = Path(run_dir)
    hierarchy: Dict[int, Dict[int, Dict[str, Any]]] = {}
    paths: Dict[int, List[int]] = {}

    # Collect tree.json from each level
    for level in sorted(levels):
        level_dir = run_dir / f'level_{level}'
        tree_path = level_dir / 'relations' / 'tree.json'

        if not tree_path.exists():
            LOGGER.warning("Missing tree.json for level %d at %s", level, tree_path)
            continue

        with tree_path.open('r', encoding='utf-8') as f:
            tree_nodes = json.load(f)

        level_hierarchy: Dict[int, Dict[str, Any]] = {}
        for node in tree_nodes:
            obj_id = int(node['id'])
            parent_id = node.get('parent')
            if parent_id is not None:
                parent_id = int(parent_id)
            children = [int(c) for c in node.get('children', [])]

            level_hierarchy[obj_id] = {
                'parent': parent_id,
                'children': children,
            }

        hierarchy[level] = level_hierarchy

    # Build cross-level children relationships
    # (The children from tree.json are intra-level, we need to add cross-level children)
    for level in sorted(levels, reverse=True):  # Process from higher to lower levels
        if level not in hierarchy:
            continue
        # Find all objects in this level
        for obj_id, node in hierarchy[level].items():
            parent_id = node.get('parent')
            if parent_id is None:
                continue

            # Find parent's level (should be a lower level number)
            for parent_level in sorted(levels):
                if parent_level >= level:
                    break
                if parent_level in hierarchy and parent_id in hierarchy[parent_level]:
                    # Add this object as a child of its parent
                    parent_node = hierarchy[parent_level][parent_id]
                    if obj_id not in parent_node['children']:
                        parent_node['children'].append(obj_id)
                    break

    # Build ancestor paths
    def build_path(obj_id: int, level: int) -> List[int]:
        if level not in hierarchy or obj_id not in hierarchy[level]:
            return [obj_id]

        parent_id = hierarchy[level][obj_id].get('parent')
        if parent_id is None:
            return [obj_id]

        # Find parent's level
        parent_level = None
        for lv in sorted(levels):
            if lv >= level:
                break
            if lv in hierarchy and parent_id in hierarchy[lv]:
                parent_level = lv

        if parent_level is None:
            return [obj_id]

        parent_path = build_path(parent_id, parent_level)
        return parent_path + [obj_id]

    for level in levels:
        if level not in hierarchy:
            continue
        for obj_id in hierarchy[level]:
            paths[obj_id] = build_path(obj_id, level)

    # Add descendant counts
    for level in levels:
        if level not in hierarchy:
            continue
        for obj_id, node in hierarchy[level].items():
            descendants = set()
            def collect_descendants(oid: int, lv: int) -> None:
                if lv not in hierarchy or oid not in hierarchy[lv]:
                    return
                for child in hierarchy[lv][oid].get('children', []):
                    descendants.add(child)
                    # Find child's level
                    for next_lv in sorted(levels):
                        if next_lv <= lv:
                            continue
                        if next_lv in hierarchy and child in hierarchy[next_lv]:
                            collect_descendants(child, next_lv)
                            break

            collect_descendants(obj_id, level)
            node['descendant_count'] = len(descendants)

    relations_path = run_dir / 'relations.json'
    payload = {
        'meta': {
            'scene': run_dir.parent.name,
            'run_name': run_dir.name,
            'levels': sorted(levels),
            'generated_at': datetime.now(timezone.utc).isoformat(),
        },
        'hierarchy': {
            str(lv): {str(oid): node for oid, node in nodes.items()}
            for lv, nodes in sorted(hierarchy.items())
        },
        'paths': {str(oid): path for oid, path in sorted(paths.items())},
    }

    with relations_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Wrote cross-level relations: %d levels, %d objects → %s",
        len(levels),
        len(paths),
        relations_path,
    )
    return relations_path


def save_ssam_relations(
    progressive_results: Dict[str, Any],
    save_root: str | Path,
    levels: List[int],
) -> None:
    """
    Save SSAM progressive refinement tree.json to relations/ subdirectory.

    Called from ssam_progressive_adapter after progressive_refinement_masks().
    """
    save_root = Path(save_root)
    relations_dir = save_root / 'relations'
    ensure_dir(relations_dir)

    tree_relations: Dict[int, List[Dict[str, Any]]] = {}

    # Extract parent-child info from progressive_results
    for level in levels:
        level_data = progressive_results.get('levels', {}).get(level, {})
        masks = level_data.get('masks', [])

        tree_nodes: List[Dict[str, Any]] = []
        for mask in masks:
            unique_id = mask.get('unique_id')
            if unique_id is None:
                continue

            parent_id = mask.get('parent_unique_id')
            node = {
                'id': int(unique_id),
                'parent': int(parent_id) if parent_id is not None else None,
                'children': [],
            }
            tree_nodes.append(node)

        tree_relations[level] = tree_nodes

    # Build children lists
    id_to_node: Dict[int, Dict[str, Any]] = {}
    for level_nodes in tree_relations.values():
        for node in level_nodes:
            id_to_node[node['id']] = node

    for level_nodes in tree_relations.values():
        for node in level_nodes:
            parent_id = node['parent']
            if parent_id is not None and parent_id in id_to_node:
                if node['id'] not in id_to_node[parent_id]['children']:
                    id_to_node[parent_id]['children'].append(node['id'])

    # Write per-level tree.json
    for level, nodes in tree_relations.items():
        level_tree_path = relations_dir / f'tree_L{level}.json'
        with level_tree_path.open('w', encoding='utf-8') as f:
            json.dump(nodes, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved SSAM tree for level %d: %d nodes → %s", level, len(nodes), level_tree_path)

    # Write merged tree.json
    merged_tree_path = relations_dir / 'tree.json'
    merged_payload = {str(lv): nodes for lv, nodes in sorted(tree_relations.items())}
    with merged_tree_path.open('w', encoding='utf-8') as f:
        json.dump(merged_payload, f, indent=2, ensure_ascii=False)

    LOGGER.info("Saved merged SSAM tree: %d levels → %s", len(tree_relations), merged_tree_path)
