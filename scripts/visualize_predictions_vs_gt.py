#!/usr/bin/env python3
"""
Visualize Predictions vs Ground Truth in 3D

Creates a PLY file that can be opened in CloudCompare or MeshLab to:
- See where predictions are located vs GT
- Check if predictions are too large/small
- Identify spatial misalignments

Color coding:
- Gray: Background (unlabeled)
- Green: Ground truth only
- Red: Predictions only
- Yellow: Overlap (both GT and predictions)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple


def load_point_cloud(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from PLY file.

    Returns:
        points: (N, 3) xyz coordinates
        colors: (N, 3) RGB colors (if available, else None)
    """
    try:
        import plyfile
    except ImportError:
        raise ImportError("plyfile not installed. Run: pip install plyfile")

    ply_data = plyfile.PlyData.read(str(ply_path))
    vertex = ply_data['vertex']

    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    # Try to load colors
    colors = None
    if 'red' in vertex.dtype.names:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T

    print(f"Loaded point cloud: {points.shape[0]} points")
    if colors is not None:
        print(f"  RGB colors: available")

    return points, colors


def load_labels(label_path: Path) -> np.ndarray:
    """
    Load per-point instance labels.

    Returns:
        labels: (N,) array of instance IDs
    """
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                continue
            labels.append(int(stripped))

    return np.array(labels, dtype=np.int64)


def create_visualization_colors(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray
) -> np.ndarray:
    """
    Create color array for visualization.

    Color scheme:
    - Gray (128, 128, 128): Background
    - Green (0, 255, 0): GT only
    - Red (255, 0, 0): Prediction only
    - Yellow (255, 255, 0): Overlap (both GT and pred)
    """
    num_points = len(gt_labels)
    colors = np.zeros((num_points, 3), dtype=np.uint8)

    # Background: gray
    bg_mask = (gt_labels == 0) & (pred_labels == 0)
    colors[bg_mask] = [128, 128, 128]

    # GT only: green
    gt_only_mask = (gt_labels > 0) & (pred_labels == 0)
    colors[gt_only_mask] = [0, 255, 0]

    # Prediction only: red
    pred_only_mask = (gt_labels == 0) & (pred_labels > 0)
    colors[pred_only_mask] = [255, 0, 0]

    # Overlap: yellow
    overlap_mask = (gt_labels > 0) & (pred_labels > 0)
    colors[overlap_mask] = [255, 255, 0]

    # Statistics
    print("\nVisualization Statistics:")
    print(f"  Background (gray): {bg_mask.sum():,} points ({bg_mask.sum()/num_points*100:.1f}%)")
    print(f"  GT only (green): {gt_only_mask.sum():,} points ({gt_only_mask.sum()/num_points*100:.1f}%)")
    print(f"  Pred only (red): {pred_only_mask.sum():,} points ({pred_only_mask.sum()/num_points*100:.1f}%)")
    print(f"  Overlap (yellow): {overlap_mask.sum():,} points ({overlap_mask.sum()/num_points*100:.1f}%)")

    return colors


def save_ply(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path
):
    """Save point cloud with colors to PLY file."""
    try:
        import plyfile
    except ImportError:
        raise ImportError("plyfile not installed. Run: pip install plyfile")

    # Create structured array
    vertex = np.zeros(
        len(points),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )

    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]

    # Create PLY element
    vertex_element = plyfile.PlyElement.describe(vertex, 'vertex')

    # Write to file
    plyfile.PlyData([vertex_element]).write(str(output_path))

    print(f"\nSaved visualization to: {output_path}")
    print(f"  Open with: CloudCompare {output_path}")
    print(f"  Or: meshlab {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predictions vs ground truth in 3D"
    )
    parser.add_argument(
        "--point-cloud",
        type=str,
        required=True,
        help="Path to scene PLY file (e.g., scene_00093_01.ply)"
    )
    parser.add_argument(
        "--gt-labels",
        type=str,
        required=True,
        help="Path to ground truth labels (scene_XXXXX_XX_obj_part_inst.txt)"
    )
    parser.add_argument(
        "--pred-labels",
        type=str,
        required=True,
        help="Path to prediction labels (scene_XXXXX_XX_obj_part_inst.txt from inference_output)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PLY file path"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("3D Visualization: Predictions vs Ground Truth")
    print("=" * 80)
    print(f"Point cloud: {args.point_cloud}")
    print(f"GT labels: {args.gt_labels}")
    print(f"Pred labels: {args.pred_labels}")
    print(f"Output: {args.output}")
    print()

    # Load data
    print("Loading point cloud...")
    points, _ = load_point_cloud(Path(args.point_cloud))

    print("\nLoading ground truth labels...")
    gt_labels = load_labels(Path(args.gt_labels))

    print("\nLoading prediction labels...")
    pred_labels = load_labels(Path(args.pred_labels))

    # Sanity check
    if len(points) != len(gt_labels):
        raise ValueError(
            f"Point count mismatch: cloud={len(points)}, GT labels={len(gt_labels)}"
        )
    if len(points) != len(pred_labels):
        raise ValueError(
            f"Point count mismatch: cloud={len(points)}, pred labels={len(pred_labels)}"
        )

    # Create visualization colors
    print("\nCreating visualization...")
    colors = create_visualization_colors(gt_labels, pred_labels)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(points, colors, output_path)

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("Color meanings:")
    print("  Gray:   Background (no GT, no prediction)")
    print("  Green:  Ground truth only (missed by predictions)")
    print("  Red:    Predictions only (false positives)")
    print("  Yellow: Overlap (correct predictions)")
    print()
    print("What to look for:")
    print("  1. If mostly RED: Predictions in wrong locations")
    print("  2. If mostly GREEN: Predictions missing GT instances")
    print("  3. If RED blobs are larger than GREEN: Predictions too large")
    print("  4. If RED blobs are smaller than GREEN: Predictions too small")
    print("  5. If YELLOW is minimal: Poor overlap (mAP=0 makes sense)")
    print("=" * 80)


if __name__ == "__main__":
    main()
