"""
Family Visualization Tool - Visualize L2/L4/L6 instances side-by-side.

Generates comparison visualizations for families showing:
- Original RGB frame
- L2 instance masks
- L4 instance masks
- L6 instance masks

Only selects frames where multiple levels appear together.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Add src to path for imports
_src_path = str(Path(__file__).parent.parent / 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from my3dis.family_tree_query import FamilyTreeQuery


def generate_color_map(num_colors: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization."""
    random.seed(seed)
    np.random.seed(seed)

    colors = []
    for i in range(num_colors):
        hue = (i * 137.508) % 360  # Golden angle for good distribution
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
        value = 0.8 + (i % 2) * 0.15  # Vary brightness

        # Convert HSV to RGB
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors.append((
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255),
        ))

    return colors


def overlay_masks(
    base_image: np.ndarray,
    masks: Dict[int, np.ndarray],
    obj_ids: List[int],
    colors: List[Tuple[int, int, int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay multiple colored masks on base image.

    Args:
        base_image: RGB image (H, W, 3)
        masks: Dict mapping obj_id -> binary mask
        obj_ids: List of object IDs to overlay (in order)
        colors: List of colors (one per object)
        alpha: Transparency (0=transparent, 1=opaque)

    Returns:
        RGB image with overlaid masks
    """
    result = base_image.copy()
    h, w = result.shape[:2]

    for obj_id, color in zip(obj_ids, colors):
        mask = masks.get(obj_id)
        if mask is None:
            continue

        # Resize mask to match image if needed
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        # Create colored overlay
        overlay = np.zeros_like(result)
        overlay[mask] = color

        # Blend
        result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)

    return result


def load_frame_image(
    data_path: str,
    frame_idx: int,
    max_dim: int = 1280,
) -> Optional[np.ndarray]:
    """Load RGB frame image and resize if needed."""
    # Try multiple filename formats
    frame_path_candidates = [
        Path(data_path) / f"{frame_idx}.png",        # No padding (e.g., "0.png", "100.png")
        Path(data_path) / f"{frame_idx:05d}.png",    # 5-digit padding (e.g., "00000.png")
        Path(data_path) / f"{frame_idx:06d}.png",    # 6-digit padding
        Path(data_path) / f"{frame_idx}.jpg",
        Path(data_path) / f"{frame_idx:05d}.jpg",
    ]

    frame_path = None
    for candidate in frame_path_candidates:
        if candidate.exists():
            frame_path = candidate
            break

    if frame_path is None:
        return None

    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def visualize_family(
    query: FamilyTreeQuery,
    family_members: List[int],
    data_path: str,
    output_dir: str,
    family_idx: int,
    max_frames: int = 3,
) -> List[str]:
    """
    Visualize a single family across multiple frames.

    Returns:
        List of generated image paths
    """
    # Group members by level
    members_by_level: Dict[int, List[int]] = {}
    for obj_id in family_members:
        obj_info = query.get_object_info(obj_id)
        if obj_info is not None:
            level = obj_info['level']
            members_by_level.setdefault(level, []).append(obj_id)

    # Find common frames where multiple levels appear
    level_frame_sets = {}
    for level, obj_ids in members_by_level.items():
        frames_union = set()
        for obj_id in obj_ids:
            frames_union.update(query.get_object_frames(obj_id))
        level_frame_sets[level] = frames_union

    # Find frames where at least 2 levels appear
    if len(level_frame_sets) < 2:
        print(f"  Family {family_idx}: Only 1 level present, skipping")
        return []

    common_frames = set.intersection(*level_frame_sets.values())
    if not common_frames:
        # Relax: find frames where at least 2 levels appear
        common_frames = set()
        level_list = list(level_frame_sets.values())
        for i in range(len(level_list)):
            for j in range(i + 1, len(level_list)):
                common_frames.update(level_list[i] & level_list[j])

    if not common_frames:
        print(f"  Family {family_idx}: No common frames across levels, skipping")
        return []

    # Sample frames
    selected_frames = sorted(common_frames)
    if len(selected_frames) > max_frames:
        # Evenly spaced sampling
        step = len(selected_frames) / max_frames
        selected_frames = [selected_frames[int(i * step)] for i in range(max_frames)]

    print(f"  Family {family_idx}: Visualizing {len(selected_frames)} frames")

    # Generate colors for each object
    all_obj_ids = sorted(family_members)
    colors = generate_color_map(len(all_obj_ids))
    obj_color_map = {obj_id: colors[i] for i, obj_id in enumerate(all_obj_ids)}

    output_paths = []

    for frame_idx in selected_frames:
        # Load base image
        base_img = load_frame_image(data_path, frame_idx)
        if base_img is None:
            print(f"    Warning: Frame {frame_idx} not found")
            continue

        h, w = base_img.shape[:2]

        # Create 4-panel visualization: [Original, L2, L4, L6]
        panels = []

        # Panel 1: Original
        panels.append(base_img.copy())

        # Panels 2-4: L2, L4, L6
        for level in [2, 4, 6]:
            level_obj_ids = members_by_level.get(level, [])

            if not level_obj_ids:
                # Empty panel with label
                panel = np.full_like(base_img, 255)
                cv2.putText(
                    panel,
                    f"L{level}: No objects",
                    (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (128, 128, 128),
                    2,
                )
                panels.append(panel)
                continue

            # Load masks for this level
            masks = {}
            for obj_id in level_obj_ids:
                mask = query.load_mask(obj_id, frame_idx)
                if mask is not None:
                    masks[obj_id] = mask

            if not masks:
                # No masks at this frame
                panel = base_img.copy()
                cv2.putText(
                    panel,
                    f"L{level}: Not visible",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                panels.append(panel)
                continue

            # Overlay masks
            panel = overlay_masks(
                base_img,
                masks,
                sorted(masks.keys()),
                [obj_color_map[obj_id] for obj_id in sorted(masks.keys())],
                alpha=0.5,
            )

            # Add label
            cv2.putText(
                panel,
                f"L{level}: {len(masks)} objects",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            panels.append(panel)

        # Concatenate panels horizontally
        combined = np.hstack(panels)

        # Add title bar
        title_height = 40
        title_bar = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            title_bar,
            f"Family {family_idx} | Frame {frame_idx} | Root: {family_members[0]}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        combined_with_title = np.vstack([title_bar, combined])

        # Save
        output_path = os.path.join(
            output_dir,
            f"family_{family_idx:03d}_frame_{frame_idx:05d}.png",
        )
        cv2.imwrite(output_path, cv2.cvtColor(combined_with_title, cv2.COLOR_RGB2BGR))
        output_paths.append(output_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description='Visualize families with L2/L4/L6 instances side-by-side'
    )
    parser.add_argument(
        '--tree',
        required=True,
        help='Path to family_tree.json',
    )
    parser.add_argument(
        '--data-path',
        required=True,
        help='Path to RGB frames directory',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for visualizations',
    )
    parser.add_argument(
        '--num-families',
        type=int,
        default=5,
        help='Number of families to visualize (default: 5)',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=3,
        help='Max frames per family (default: 3)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for family selection (default: 42)',
    )

    args = parser.parse_args()

    # Load query interface
    print(f"Loading family tree: {args.tree}")
    query = FamilyTreeQuery(args.tree)

    # Get families
    families = query.tree['families']
    print(f"Found {len(families)} families")

    # Filter families that have multiple levels
    valid_families = []
    for family in families:
        levels = family.get('levels', {})
        if len(levels) >= 2:  # At least 2 levels
            valid_families.append(family)

    print(f"  {len(valid_families)} families have multiple levels")

    if not valid_families:
        print("No families with multiple levels found!")
        return

    # Sample families
    random.seed(args.seed)
    num_to_sample = min(args.num_families, len(valid_families))
    selected_families = random.sample(valid_families, num_to_sample)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Visualize each family
    print(f"\nGenerating visualizations...")
    total_images = 0

    for i, family in enumerate(selected_families, 1):
        family_members = family['members']
        output_paths = visualize_family(
            query,
            family_members,
            args.data_path,
            args.output_dir,
            i,
            args.max_frames,
        )
        total_images += len(output_paths)

    print(f"\n✓ Generated {total_images} visualizations in {args.output_dir}")


if __name__ == '__main__':
    main()
