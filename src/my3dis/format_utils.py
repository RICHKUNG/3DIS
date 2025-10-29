"""
Format detection and compatibility utilities for My3DIS data formats.

Supports multiple versions of relations.json, index.json, and NPZ manifests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

FormatVersion = Literal["1.0", "2.0", "2.5", "3.0", "unknown"]


def detect_relations_format(data: Dict[str, Any]) -> FormatVersion:
    """
    Detect relations.json format version.

    Args:
        data: Loaded JSON data

    Returns:
        Format version string

    Examples:
        >>> data = {"schema_version": "3.0", ...}
        >>> detect_relations_format(data)
        '3.0'

        >>> data = {"hierarchy": {...}, "paths": {...}}  # No version field
        >>> detect_relations_format(data)
        '2.5'

        >>> data = {"levels": [2, 4, 6], "objects": [...], "tree": {...}}
        >>> detect_relations_format(data)
        '2.0'
    """
    # Check for explicit version field (v3.0+)
    if 'schema_version' in data:
        return data['schema_version']

    # Detect v2.5 (current format without version field)
    if 'hierarchy' in data and 'paths' in data:
        if 'meta' in data and isinstance(data.get('meta'), dict):
            return "2.5"

    # Detect v2.0 (old format with objects array)
    if 'objects' in data and isinstance(data['objects'], list):
        # Check if objects have string IDs like "L02_00001"
        if data['objects'] and 'id' in data['objects'][0]:
            first_id = data['objects'][0]['id']
            if isinstance(first_id, str) and '_' in first_id:
                return "2.0"

    # Detect v1.0 (very old format)
    if 'levels' in data and 'tree' in data and isinstance(data.get('tree'), dict):
        return "1.0"

    return "unknown"


def detect_index_format(data: Dict[str, Any]) -> FormatVersion:
    """
    Detect index.json format version.

    Args:
        data: Loaded JSON data

    Returns:
        Format version string
    """
    if 'schema_version' in data:
        return data['schema_version']

    # v2.5/3.0 format: has meta, objects (dict), frames (dict)
    if 'meta' in data and 'objects' in data and 'frames' in data:
        if isinstance(data['objects'], dict) and isinstance(data['frames'], dict):
            return "2.5"

    # Old format: objects is list
    if 'objects' in data and isinstance(data['objects'], list):
        return "2.0"

    return "unknown"


def detect_npz_manifest_format(data: Dict[str, Any]) -> FormatVersion:
    """
    Detect NPZ manifest.json format version.

    Args:
        data: Loaded manifest JSON from NPZ

    Returns:
        Format version string
    """
    if 'schema_version' in data:
        return data['schema_version']

    # Check for v2.5/3.0 features
    if 'meta' in data:
        meta = data['meta']
        # New format has linked_video with new naming
        if 'linked_video' in meta:
            linked = meta['linked_video']
            if isinstance(linked, str) and '_L' in linked:
                return "2.5"  # Transitional
            elif isinstance(linked, str) and 'scale' in linked:
                return "2.0"  # Old naming

    # Very old format
    if 'objects' in data and isinstance(data['objects'], dict):
        return "2.0"

    return "unknown"


def normalize_relations_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize relations.json to v3.0 format regardless of input version.

    Args:
        data: Relations data in any supported format

    Returns:
        Normalized v3.0 format dict

    Raises:
        ValueError: If format is unsupported or data is invalid
    """
    version = detect_relations_format(data)

    if version == "3.0":
        # Already in v3.0 format
        return data

    elif version == "2.5":
        # Add schema_version and statistics
        result = {
            "schema_version": "3.0",
            "meta": data.get('meta', {}),
            "hierarchy": data['hierarchy'],
            "paths": data['paths']
        }

        # Compute statistics
        total_objects = len(result['paths'])
        roots = sum(1 for level_objs in result['hierarchy'].values()
                    for obj in level_objs.values()
                    if obj.get('parent') is None)
        families = sum(1 for level_objs in result['hierarchy'].values()
                       for obj in level_objs.values()
                       if obj.get('children'))
        max_depth = max((len(path) for path in result['paths'].values()), default=0)

        result['statistics'] = {
            "total_objects": total_objects,
            "roots": roots,
            "families": families,
            "max_depth": max_depth
        }

        return result

    elif version == "2.0":
        # Convert old list format to v3.0 hierarchy
        hierarchy: Dict[str, Dict[str, Dict[str, Any]]] = {}
        paths: Dict[str, list] = {}

        for obj in data['objects']:
            # Parse "L02_00001" → level=2, id=1
            obj_id_str = obj['id']
            parts = obj_id_str.split('_')
            if len(parts) != 2:
                continue

            level = int(parts[0][1:])  # "L02" → 2
            obj_id = int(parts[1])      # "00001" → 1

            level_str = str(level)
            id_str = str(obj_id)

            if level_str not in hierarchy:
                hierarchy[level_str] = {}

            hierarchy[level_str][id_str] = {
                "parent": obj.get('parent'),
                "children": obj.get('children', []),
                "descendant_count": obj.get('descendant_count', 0)
            }

            # Build path (would need full tree traversal for accuracy)
            paths[id_str] = [obj_id]  # Simplified

        result = {
            "schema_version": "3.0",
            "meta": {
                "levels": data.get('levels', []),
                "migrated_from": "2.0"
            },
            "hierarchy": hierarchy,
            "paths": paths,
            "statistics": {
                "total_objects": len(paths),
                "roots": 0,  # Would need computation
                "families": 0,  # Would need computation
                "max_depth": 1
            }
        }

        return result

    else:
        raise ValueError(f"Unsupported relations format: {version}")


def get_object_count(relations_data: Dict[str, Any]) -> int:
    """
    Get total object count from relations.json in any format.

    Args:
        relations_data: Loaded relations JSON

    Returns:
        Total number of objects

    Examples:
        >>> data_v3 = {"schema_version": "3.0", "paths": {"1": [1], "2": [2]}}
        >>> get_object_count(data_v3)
        2

        >>> data_v2 = {"objects": [{"id": "L02_00001"}, {"id": "L02_00002"}]}
        >>> get_object_count(data_v2)
        2
    """
    version = detect_relations_format(relations_data)

    if version in ("3.0", "2.5"):
        # Use paths count
        return len(relations_data.get('paths', {}))

    elif version == "2.0":
        # Use objects array length
        return len(relations_data.get('objects', []))

    elif version == "1.0":
        # Sum all objects from tree
        tree = relations_data.get('tree', {})
        return sum(len(level_data) for level_data in tree.values() if isinstance(level_data, dict))

    else:
        return 0


def validate_format_v3(data: Dict[str, Any], data_type: str = "relations") -> bool:
    """
    Validate data conforms to v3.0 format standards.

    Args:
        data: Loaded JSON data
        data_type: Type of data ("relations", "index", "npz_manifest")

    Returns:
        True if valid, False otherwise

    Note:
        For production use, consider using jsonschema for stricter validation.
    """
    try:
        if data_type == "relations":
            required_keys = {'schema_version', 'meta', 'hierarchy', 'paths'}
            if not required_keys.issubset(data.keys()):
                return False

            if data['schema_version'] != "3.0":
                return False

            # Check hierarchy structure
            if not isinstance(data['hierarchy'], dict):
                return False

            for level_str, level_objs in data['hierarchy'].items():
                if not level_str.isdigit():
                    return False
                if not isinstance(level_objs, dict):
                    return False

            return True

        elif data_type == "index":
            required_keys = {'schema_version', 'meta', 'objects', 'frames'}
            if not required_keys.issubset(data.keys()):
                return False

            if data['schema_version'] != "3.0":
                return False

            return True

        elif data_type == "npz_manifest":
            required_keys = {'schema_version', 'meta'}
            if not required_keys.issubset(data.keys()):
                return False

            if data['schema_version'] != "3.0":
                return False

            return True

        else:
            return False

    except Exception:
        return False


def load_relations_compat(path: Path) -> Dict[str, Any]:
    """
    Load relations.json with automatic format detection and normalization.

    Args:
        path: Path to relations.json

    Returns:
        Relations data normalized to v3.0 format

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported
    """
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize to v3.0
    return normalize_relations_v3(data)


# Example usage
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python format_utils.py <relations.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    with path.open('r') as f:
        data = json.load(f)

    version = detect_relations_format(data)
    print(f"Detected format: {version}")

    obj_count = get_object_count(data)
    print(f"Total objects: {obj_count}")

    if version != "3.0":
        print(f"\nNormalizing to v3.0...")
        normalized = normalize_relations_v3(data)
        print(f"Normalized format: {normalized['schema_version']}")
        print(f"Statistics: {normalized.get('statistics', {})}")
