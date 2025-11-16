"""
Search3D Format Exporter

This module provides utilities to export My3DIS predictions to Search3D evaluation format.

Search3D Format:
- One text file per scene: scene_XXXXX_XX_[obj/obj_part]_inst.txt
- One line per point in the scene point cloud
- Each line contains an integer label:
  - 0: background/unlabeled
  - 1000+: instance ID for object-level
  - composite ID (object_id * 1000 + part_id) for part-level

Example:
    >>> predictions = [...]  # List of Prediction objects
    >>> point_cloud = np.load('scene.ply')  # (P, 3)
    >>> export_predictions_search3d(
    ...     predictions=predictions,
    ...     num_points=len(point_cloud),
    ...     output_path='scene_00005_00_obj_part_inst.txt',
    ...     format_type='obj_part'
    ... )
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Union
import numpy as np

logger = logging.getLogger(__name__)


def export_predictions_search3d(
    predictions: List,
    num_points: int,
    output_path: Union[str, Path],
    format_type: str = 'obj_part',
    instance_id_start: int = 1000,
) -> Dict[str, int]:
    """
    Export predictions to Search3D evaluation format.

    Args:
        predictions: List of Prediction objects with attributes:
            - mask: (P,) boolean array indicating which points belong to this prediction
            - label_id: Integer label ID from query (for obj_part format)
            - score: Confidence score
            - metadata: Dict with 'object_id', 'part_id', etc.
        num_points: Total number of points in the scene point cloud
        output_path: Output .txt file path
        format_type: Output format type
            - 'obj': Object-level format (instance IDs 1000+)
            - 'obj_part': Object-part format (composite IDs: object_id * 1000 + part_id)
        instance_id_start: Starting instance ID for 'obj' format (default: 1000)

    Returns:
        Dict with statistics:
            - 'num_predictions': Number of predictions processed
            - 'num_labeled_points': Number of points with non-zero labels
            - 'num_background_points': Number of background points (label=0)
            - 'label_distribution': Dict of label_id -> count

    Output Format:
        Text file with one line per point in the point cloud.
        - obj format: instance ID (1000, 1001, 1002, ...)
        - obj_part format: composite ID (e.g., 4008 for door frame)

    Notes:
        - Overlapping predictions: Later predictions overwrite earlier ones
        - Background points (no prediction): label = 0
        - For 'obj' format: Predictions are assigned sequential instance IDs
        - For 'obj_part' format: Uses label_id from query
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize labels array (all points start as background)
    labels = np.zeros(num_points, dtype=np.int32)

    # Track statistics
    label_distribution = {}

    if format_type == 'obj':
        # Object-level format: instance_id = class_id * 1000 + instance_number
        # The label_id in pred should be the object class ID (e.g., 4 for door)
        # extracted from the composite ID (e.g., 4008 → 4) in calling code

        logger.info(f"Exporting {len(predictions)} predictions in object-level format")

        # Group predictions by class ID
        predictions_by_class = {}
        for pred in predictions:
            # Get object class ID from prediction
            if hasattr(pred, 'label_id') and pred.label_id is not None:
                class_id = pred.label_id  # Should be pure object class ID (4, 11, 32)
            else:
                logger.warning(f"Prediction has no label_id, skipping")
                continue

            if class_id not in predictions_by_class:
                predictions_by_class[class_id] = []
            predictions_by_class[class_id].append(pred)

        # Export each class
        for class_id, class_preds in predictions_by_class.items():
            logger.info(f"  Class {class_id}: {len(class_preds)} instances")

            for instance_num, pred in enumerate(class_preds, start=1):
                # Create instance ID: class_id * 1000 + instance_number
                instance_id = class_id * 1000 + instance_num
                mask = pred.mask  # (P,) boolean mask (should be object_mask!)

                if not isinstance(mask, np.ndarray):
                    logger.warning(f"Class {class_id} instance {instance_num}: mask is not numpy array, skipping")
                    continue

                if len(mask) != num_points:
                    logger.warning(
                        f"Class {class_id} instance {instance_num}: mask length ({len(mask)}) != num_points ({num_points}), skipping"
                    )
                    continue

                # Assign instance ID to masked points
                labels[mask] = instance_id
                label_distribution[instance_id] = int(mask.sum())

        logger.info(f"  Total classes: {len(predictions_by_class)}")
        logger.info(f"  Total instances: {len(label_distribution)}")

    elif format_type == 'obj_part':
        # Object-part format: composite_id * 1000 + instance_number
        # GT format: 11001001 = composite_id=11001, instance=1
        logger.info(f"Exporting {len(predictions)} predictions in object-part format")

        # Group predictions by label_id for instance numbering
        predictions_by_label = {}
        for idx, pred in enumerate(predictions):
            # Get label_id from prediction
            if hasattr(pred, 'label_id') and pred.label_id is not None:
                label_id = pred.label_id
            elif hasattr(pred, 'metadata') and pred.metadata and 'label_id' in pred.metadata:
                label_id = pred.metadata['label_id']
            else:
                logger.warning(f"Prediction {idx}: No label_id found, skipping")
                continue

            mask = pred.mask  # (P,) boolean mask

            if not isinstance(mask, np.ndarray):
                logger.warning(f"Prediction {idx}: mask is not numpy array, skipping")
                continue

            if len(mask) != num_points:
                logger.warning(
                    f"Prediction {idx}: mask length ({len(mask)}) != num_points ({num_points}), skipping"
                )
                continue

            # Store prediction with its mask
            if label_id not in predictions_by_label:
                predictions_by_label[label_id] = []
            predictions_by_label[label_id].append((pred, mask))

        # Assign instance IDs: composite_id * 1000 + instance_number
        for label_id, label_preds in predictions_by_label.items():
            logger.info(f"  Label {label_id}: {len(label_preds)} instances")

            for instance_num, (pred, mask) in enumerate(label_preds, start=1):
                # Create 8-digit instance ID: composite_id * 1000 + instance_number
                instance_id = label_id * 1000 + instance_num

                # Assign instance ID to masked points
                labels[mask] = instance_id
                label_distribution[instance_id] = int(mask.sum())

        logger.info(f"  Total unique labels: {len(predictions_by_label)}")
        logger.info(f"  Total instances: {sum(len(preds) for preds in predictions_by_label.values())}")

    else:
        raise ValueError(f"Unknown format_type: {format_type}. Must be 'obj' or 'obj_part'")

    # Write labels to file (one per line)
    logger.info(f"Writing labels to {output_path}")
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    # Compute statistics
    num_labeled = int((labels > 0).sum())
    num_background = int((labels == 0).sum())

    stats = {
        'num_predictions': len(predictions),
        'num_labeled_points': num_labeled,
        'num_background_points': num_background,
        'label_distribution': label_distribution,
        'output_path': str(output_path),
    }

    logger.info(f"Export complete:")
    logger.info(f"  - Labeled points: {num_labeled}/{num_points} ({100*num_labeled/num_points:.1f}%)")
    logger.info(f"  - Background points: {num_background}/{num_points} ({100*num_background/num_points:.1f}%)")
    logger.info(f"  - Unique labels: {len(label_distribution)}")

    return stats


def export_predictions_both_formats(
    predictions: List,
    num_points: int,
    output_dir: Union[str, Path],
    scene_name: str,
    instance_id_start: int = 1000,
) -> Dict[str, Dict]:
    """
    Export predictions in both object-level and object-part formats.

    Args:
        predictions: List of Prediction objects
        num_points: Total number of points in the scene
        output_dir: Output directory for both files
        scene_name: Scene identifier (e.g., 'scene_00005_00')
        instance_id_start: Starting instance ID for object-level format

    Returns:
        Dict with stats for both formats:
            {
                'obj': {...},
                'obj_part': {...}
            }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Export object-level format
    obj_path = output_dir / f"{scene_name}_obj_inst.txt"
    logger.info(f"Exporting object-level format to {obj_path}")
    results['obj'] = export_predictions_search3d(
        predictions=predictions,
        num_points=num_points,
        output_path=obj_path,
        format_type='obj',
        instance_id_start=instance_id_start,
    )

    # Export object-part format
    obj_part_path = output_dir / f"{scene_name}_obj_part_inst.txt"
    logger.info(f"Exporting object-part format to {obj_part_path}")
    results['obj_part'] = export_predictions_search3d(
        predictions=predictions,
        num_points=num_points,
        output_path=obj_part_path,
        format_type='obj_part',
        instance_id_start=instance_id_start,
    )

    return results


def load_point_cloud_for_export(
    scene_path: Union[str, Path],
    mesh_name: str = 'mesh_aligned_0.05.ply'
) -> np.ndarray:
    """
    Load point cloud from scene directory for Search3D export.

    Tries multiple fallback mesh files if the primary mesh is not found:
    1. mesh_aligned_0.05.ply (default)
    2. <scene_name>_converted.ply
    3. <scene_name>.ply
    4. Any other .ply file in the directory

    Args:
        scene_path: Path to MultiScan scene directory
        mesh_name: Name of the PLY mesh file (tried first)

    Returns:
        (P, 3) numpy array of point coordinates

    Raises:
        FileNotFoundError: If no mesh file found
    """
    scene_path = Path(scene_path)
    mesh_path = scene_path / mesh_name

    # Build list of candidate mesh files to try
    candidates = [mesh_path]

    # Add common fallback patterns
    scene_name = scene_path.name
    candidates.append(scene_path / f"{scene_name}_converted.ply")
    candidates.append(scene_path / f"{scene_name}.ply")

    # Try to find any other .ply files as last resort
    if scene_path.exists():
        other_plys = list(scene_path.glob("*.ply"))
        for ply in other_plys:
            if ply not in candidates:
                candidates.append(ply)

    # Try each candidate
    for candidate in candidates:
        if candidate.exists():
            mesh_path = candidate
            break
    else:
        # No mesh found
        tried_paths = '\n  '.join(str(c) for c in candidates[:3])
        raise FileNotFoundError(
            f"Mesh file not found. Tried:\n  {tried_paths}\n"
            f"Please ensure a PLY mesh file exists in {scene_path}"
        )

    # Import here to avoid circular dependency
    try:
        from plyfile import PlyData
    except ImportError:
        logger.error("plyfile not installed. Install with: pip install plyfile")
        raise

    logger.info(f"Loading point cloud from {mesh_path}")
    plydata = PlyData.read(str(mesh_path))
    vertices = plydata['vertex']

    points = np.stack([
        vertices['x'],
        vertices['y'],
        vertices['z']
    ], axis=1)

    logger.info(f"Loaded {len(points)} points from {mesh_path.name}")
    return points


def validate_search3d_format(
    prediction_file: Union[str, Path],
    expected_num_points: Optional[int] = None,
) -> Dict[str, Union[bool, int, str]]:
    """
    Validate a Search3D format prediction file.

    Args:
        prediction_file: Path to prediction .txt file
        expected_num_points: Expected number of points (optional)

    Returns:
        Dict with validation results:
            {
                'valid': bool,
                'num_lines': int,
                'num_labeled': int,
                'unique_labels': int,
                'errors': List[str]
            }
    """
    prediction_file = Path(prediction_file)
    errors = []

    if not prediction_file.exists():
        return {
            'valid': False,
            'errors': [f"File not found: {prediction_file}"]
        }

    try:
        with open(prediction_file, 'r') as f:
            lines = f.readlines()

        num_lines = len(lines)
        labels = []

        for i, line in enumerate(lines):
            try:
                label = int(line.strip())
                labels.append(label)
            except ValueError:
                errors.append(f"Line {i+1}: Invalid integer value: {line.strip()}")

        if expected_num_points is not None and num_lines != expected_num_points:
            errors.append(
                f"Number of lines ({num_lines}) != expected ({expected_num_points})"
            )

        labels = np.array(labels)
        num_labeled = int((labels > 0).sum())
        unique_labels = len(np.unique(labels[labels > 0]))

        return {
            'valid': len(errors) == 0,
            'num_lines': num_lines,
            'num_labeled': num_labeled,
            'unique_labels': unique_labels,
            'errors': errors
        }

    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Error reading file: {str(e)}"]
        }
