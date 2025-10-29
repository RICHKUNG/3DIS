#!/usr/bin/env python3
"""
Visualize family trees from SAM2 tracking results.

For each family (nodes with children), show the frame with highest total containment
across all levels (L2, L4, L6) side-by-side with original image and mask overlays.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_sam2_tree(tree_path: Path) -> Dict[str, Any]:
    """Load sam2_tree.json."""
    with open(tree_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_families(tree_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find all families (nodes with children) and resolve child metadata."""
    families = []
    hierarchy = tree_data.get('hierarchy', {})

    # Build a lookup: obj_id -> (level, metadata)
    obj_lookup = {}
    for level, objects in hierarchy.items():
        for obj_id, node in objects.items():
            obj_lookup[int(obj_id)] = {
                'level': int(level),
                'parent': node.get('parent'),
                'children': node.get('children', []),
                'descendant_count': node.get('descendant_count', 0),
            }

    # Find families and enrich child info
    for level, objects in hierarchy.items():
        for obj_id, node in objects.items():
            child_ids = node.get('children', [])
            if child_ids and len(child_ids) > 0:
                # Resolve children metadata
                children_meta = []
                for child_id in child_ids:
                    child_info = obj_lookup.get(int(child_id))
                    if child_info:
                        children_meta.append({
                            'obj_id': int(child_id),
                            'level': child_info['level'],
                        })
                    else:
                        # Child not found in hierarchy (shouldn't happen)
                        print(f"⚠ Child {child_id} not found in hierarchy")

                families.append({
                    'level': int(level),
                    'obj_id': int(obj_id),
                    'parent_id': node.get('parent'),
                    'children': children_meta,
                    'descendant_count': node.get('descendant_count', 0),
                })

    return families


def load_npz_manifest(npz_path: Path) -> Tuple[Dict[int, List[int]], str]:
    """
    Load NPZ file and extract object_id -> frame_indices mapping.

    Returns:
        Tuple of (object_frames dict, linked_video_path)
    """
    with np.load(npz_path, allow_pickle=True) as data:
        # Try different manifest key formats
        manifest_raw = None
        if 'manifest.json' in data:
            manifest_raw = data['manifest.json']
        elif 'manifest' in data:
            manifest_raw = data['manifest']
        else:
            raise ValueError(f"No manifest found in {npz_path}, keys: {list(data.keys())}")

        # Parse manifest (could be bytes, string, or dict)
        if isinstance(manifest_raw, bytes):
            manifest_dict = json.loads(manifest_raw.decode('utf-8'))
        elif isinstance(manifest_raw, str):
            manifest_dict = json.loads(manifest_raw)
        elif hasattr(manifest_raw, 'item'):
            manifest_dict = manifest_raw.item()
        else:
            manifest_dict = manifest_raw

        # Extract linked video path from meta
        meta = manifest_dict.get('meta', {})
        linked_video = meta.get('linked_video', '')

        # Extract objects (new format: objects is a dict of object_id -> list of frame entries)
        objects = manifest_dict.get('objects', {})
        object_frames = {}

        for obj_id_str, frame_list in objects.items():
            obj_id = int(obj_id_str)
            if isinstance(frame_list, list):
                # Each entry is {'frame_index': int, 'frame_entry': str}
                frame_indices = sorted([entry['frame_index'] for entry in frame_list])
            else:
                frame_indices = []
            object_frames[obj_id] = frame_indices

        return object_frames, linked_video


def load_mask_from_video_npz(video_npz_path: Path, obj_id: int, frame_idx: int) -> Optional[np.ndarray]:
    """Load a specific mask from video_segments NPZ file."""
    if not video_npz_path.exists():
        return None

    with np.load(video_npz_path, allow_pickle=True) as data:
        # Format 1: Direct JSON files in NPZ (current format)
        frame_entry_name = f'frames/frame_{frame_idx:06d}.json'
        if frame_entry_name in data:
            frame_json_bytes = data[frame_entry_name]
            if isinstance(frame_json_bytes, bytes):
                frame_json = json.loads(frame_json_bytes.decode('utf-8'))
            else:
                frame_json = json.loads(str(frame_json_bytes))

            objects = frame_json.get('objects', {})
            if str(obj_id) in objects:
                mask_packed = objects[str(obj_id)]
                return unpack_mask(mask_packed)

        # Format 2: Nested ZIP archive (alternative format)
        elif 'video_segments.npz' in data:
            video_seg_bytes = data['video_segments.npz']
            import io
            import zipfile

            with zipfile.ZipFile(io.BytesIO(video_seg_bytes)) as zf:
                if frame_entry_name in zf.namelist():
                    frame_json = json.loads(zf.read(frame_entry_name).decode('utf-8'))
                    objects = frame_json.get('objects', {})
                    if str(obj_id) in objects:
                        mask_packed = objects[str(obj_id)]
                        return unpack_mask(mask_packed)

        # Format 3: Dict-based format (legacy)
        elif 'frames' in data:
            frames_dict = data['frames'].item()
            frame_data = frames_dict.get(frame_idx, {})
            objects = frame_data.get('objects', {})
            if obj_id in objects or str(obj_id) in objects:
                mask_packed = objects.get(obj_id) or objects.get(str(obj_id))
                return unpack_mask(mask_packed)

    return None


def unpack_mask(packed: Dict[str, Any]) -> np.ndarray:
    """Unpack binary mask from packed format."""
    import base64

    shape = tuple(packed['shape'])
    total_pixels = int(np.prod(shape))

    # Handle both packed_bits and packed_bits_b64
    if 'packed_bits' in packed:
        packed_bits = np.asarray(packed['packed_bits'], dtype=np.uint8)
    elif 'packed_bits_b64' in packed:
        packed_bytes = base64.b64decode(packed['packed_bits_b64'])
        packed_bits = np.frombuffer(packed_bytes, dtype=np.uint8)
    else:
        raise ValueError(f"No packed_bits or packed_bits_b64 in mask data: {list(packed.keys())}")

    # Unpack bits
    unpacked = np.unpackbits(packed_bits, count=total_pixels)
    return unpacked.reshape(shape).astype(np.bool_)


def get_frame_list(data_path: Path) -> List[Tuple[int, str]]:
    """Get sorted list of (frame_index, filename) from color frames directory."""
    if not data_path.exists():
        return []

    frames = []
    # Support both .jpg and .png
    for pattern in ['*.jpg', '*.png']:
        for img_file in data_path.glob(pattern):
            stem = img_file.stem
            try:
                # Try various naming patterns
                if stem.startswith('frame_'):
                    idx = int(stem.split('_')[1])
                else:
                    # Direct numeric filename (e.g., "900.png" -> 900)
                    idx = int(stem)
                frames.append((idx, img_file.name))
            except (ValueError, IndexError):
                continue

    return sorted(frames)


def find_best_frame_for_family(
    family: Dict[str, Any],
    level_data: Dict[int, Dict[int, List[int]]],
    level_video_npz: Optional[Dict[int, Path]] = None,
) -> Optional[int]:
    """
    Find the frame index with highest "family presence" score.

    Score = number of family members (parent + children) present in that frame.
    If level_video_npz is provided, also verify that masks are non-empty.
    """
    parent_level = family['level']
    parent_id = family['obj_id']
    children = family['children']

    # Get frame indices for parent
    parent_frames = set(level_data[parent_level].get(parent_id, []))
    if not parent_frames:
        return None

    # Score each frame by how many family members are present
    frame_scores = {}
    for frame_idx in parent_frames:
        # If we have video NPZ, check if parent mask is non-empty
        if level_video_npz and parent_level in level_video_npz:
            parent_mask = load_mask_from_video_npz(
                level_video_npz[parent_level], parent_id, frame_idx
            )
            if parent_mask is None or parent_mask.sum() == 0:
                # Skip frames with empty parent masks
                continue

        score = 1  # Parent is present

        # Check each child
        for child_info in children:
            child_level = child_info['level']
            child_id = child_info['obj_id']
            child_frames = level_data.get(child_level, {}).get(child_id, [])
            if frame_idx in child_frames:
                # Optionally verify child mask is non-empty
                if level_video_npz and child_level in level_video_npz:
                    child_mask = load_mask_from_video_npz(
                        level_video_npz[child_level], child_id, frame_idx
                    )
                    if child_mask is not None and child_mask.sum() > 0:
                        score += 1
                else:
                    score += 1

        if score > 0:  # Only include frames with at least parent present
            frame_scores[frame_idx] = score

    if not frame_scores:
        return None

    # Return frame with highest score
    best_frame = max(frame_scores.items(), key=lambda x: x[1])[0]
    return best_frame


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate mask without wrap-around artifacts."""
    if radius <= 0:
        return mask

    h, w = mask.shape
    padded = np.pad(mask, radius, mode='constant', constant_values=False)
    dilated = np.zeros((h, w), dtype=bool)

    for dy in range(-radius, radius + 1):
        y_start = radius + dy
        y_end = y_start + h
        for dx in range(-radius, radius + 1):
            x_start = radius + dx
            x_end = x_start + w
            dilated |= padded[y_start:y_end, x_start:x_end]

    return dilated


def create_colored_overlay(
    image: Image.Image,
    masks: List[Tuple[np.ndarray, str, str]],
    alpha: float = 0.5,
    show_segments: bool = False,
) -> Image.Image:
    """
    Create image with colored mask overlays or segmented cutouts.

    Args:
        image: Original image
        masks: List of (mask_array, color_hex, label)
        alpha: Transparency of overlay (only used if show_segments=False)
        show_segments: If True, show only the masked regions with colored borders
    """
    if not masks:
        return image.copy()

    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]

    if show_segments:
        # Create output with black background
        output = np.zeros((h, w, 3), dtype=np.uint8)

        for mask, color_hex, label in masks:
            if mask is None:
                continue

            # Resize mask to image size (handle PIL version compatibility)
            try:
                resample_mode = Image.Resampling.NEAREST
            except AttributeError:
                resample_mode = Image.NEAREST

            mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (w, h), resample_mode
            )) > 128

            # Convert hex to RGB
            color_rgb = np.array([int(color_hex[i:i+2], 16) for i in (1, 3, 5)])

            # Extract segmented region
            output[mask_resized] = img_array[mask_resized]

            # Build colored border without wrap-around artifacts
            border_width = 3
            dilated = dilate_mask(mask_resized, border_width)
            border = dilated & ~mask_resized
            output[border] = color_rgb.astype(np.uint8)

        return Image.fromarray(output)
    else:
        # Original overlay mode
        img_rgba = img_rgb.convert('RGBA')

        for mask, color_hex, label in masks:
            if mask is None:
                continue

            # Create colored overlay
            overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))

            # Convert hex to RGB
            color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            fill_color = (*color_rgb, int(255 * alpha))

            # Resize mask to image size (handle PIL version compatibility)
            try:
                resample_mode = Image.Resampling.NEAREST
            except AttributeError:
                resample_mode = Image.NEAREST

            mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                img_rgba.size, resample_mode
            )) > 128

            # Create colored mask
            colored_mask = np.zeros((img_rgba.size[1], img_rgba.size[0], 4), dtype=np.uint8)
            colored_mask[mask_resized] = fill_color

            overlay = Image.fromarray(colored_mask, 'RGBA')
            img_rgba = Image.alpha_composite(img_rgba, overlay)

        return img_rgba.convert('RGB')


def generate_family_colors(num_members: int) -> List[str]:
    """Generate distinct colors for family members."""
    # Predefined color palette
    colors = [
        '#FF0000',  # Red (parent)
        '#00FF00',  # Green
        '#0000FF',  # Blue
        '#FFFF00',  # Yellow
        '#FF00FF',  # Magenta
        '#00FFFF',  # Cyan
        '#FFA500',  # Orange
        '#800080',  # Purple
        '#FFC0CB',  # Pink
        '#A52A2A',  # Brown
    ]

    if num_members <= len(colors):
        return colors[:num_members]

    # Generate more colors if needed
    import colorsys
    extra = []
    for i in range(num_members - len(colors)):
        hue = i / (num_members - len(colors))
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        extra.append(hex_color)

    return colors + extra


def visualize_family(
    family: Dict[str, Any],
    frame_idx: int,
    scene_root: Path,
    data_path: Path,
    output_path: Path,
    level_video_npz: Dict[int, Path],
    levels: List[int] = [2, 4, 6],
) -> None:
    """
    Create visualization for one family at the best frame.

    Args:
        family: Family metadata dict
        frame_idx: Frame index to visualize
        scene_root: Scene output root path
        data_path: Path to color frames
        output_path: Output image path
        level_video_npz: Dict mapping level -> video_segments NPZ path
        levels: List of levels to show

    Layout: [Original] [L2 overlay] [L4 overlay] [L6 overlay]
    """
    # Load original frame
    frame_list = get_frame_list(data_path)
    frame_name = None
    for idx, name in frame_list:
        if idx == frame_idx:
            frame_name = name
            break

    if frame_name is None:
        print(f"⚠ Frame {frame_idx} not found in {data_path}")
        return

    frame_path = data_path / frame_name
    original_img = Image.open(frame_path).convert('RGB')

    # Prepare colors for family members
    parent_id = family['obj_id']
    parent_level = family['level']
    children = family['children']
    num_members = 1 + len(children)
    colors = generate_family_colors(num_members)

    # Create panels for each level
    panels = [original_img.copy()]
    panel_labels = ['Original']

    for level in levels:
        video_npz_path = level_video_npz.get(level)
        if video_npz_path is None or not video_npz_path.exists():
            print(f"⚠ Video NPZ not found for L{level}")
            panels.append(original_img.copy())
            panel_labels.append(f'L{level} (missing)')
            continue

        masks_to_draw = []

        # Parent mask (if this level matches parent level)
        if level == parent_level:
            parent_mask = load_mask_from_video_npz(video_npz_path, parent_id, frame_idx)
            if parent_mask is not None:
                masks_to_draw.append((parent_mask, colors[0], f'P:{parent_id}'))

        # Children masks
        for i, child_info in enumerate(children):
            child_level = child_info['level']
            child_id = child_info['obj_id']

            if level == child_level:
                child_mask = load_mask_from_video_npz(video_npz_path, child_id, frame_idx)
                if child_mask is not None:
                    masks_to_draw.append((child_mask, colors[i + 1], f'C:{child_id}'))

        # Create segmented cutout (show_segments=True) with colored borders
        if masks_to_draw:
            segmented_img = create_colored_overlay(
                original_img, masks_to_draw, alpha=0.5, show_segments=True
            )
            panels.append(segmented_img)
        else:
            # No masks for this level, show original
            panels.append(original_img.copy())
        panel_labels.append(f'L{level}')

    # Combine panels side-by-side
    panel_width = panels[0].width
    panel_height = panels[0].height

    # Add text header height
    header_height = 40
    total_width = panel_width * len(panels)
    total_height = panel_height + header_height

    combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Draw headers and panels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(combined)

    for i, (panel, label) in enumerate(zip(panels, panel_labels)):
        x_offset = i * panel_width

        # Draw label (use textsize for older PIL versions)
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, _ = draw.textsize(label, font=font)

        text_x = x_offset + (panel_width - text_width) // 2
        draw.text((text_x, 10), label, fill=(0, 0, 0), font=font)

        # Paste panel
        combined.paste(panel, (x_offset, header_height))

    # Add family info at the bottom
    info_text = (
        f"Family: L{parent_level} Obj#{parent_id} → "
        f"{len(children)} children, {family['descendant_count']} descendants | "
        f"Frame: {frame_idx}"
    )
    draw.text((10, total_height - 30), info_text, fill=(0, 0, 0), font=font)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=95)
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize family trees from SAM2 tracking results'
    )
    parser.add_argument(
        'scene_root',
        type=Path,
        help='Scene output root (e.g., outputs/experiments/test_1024_1/scene_00005_01)',
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        help='Path to color frames directory (default: inferred from scene_root)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for visualizations (default: scene_root/family_viz)',
    )
    parser.add_argument(
        '--min-children',
        type=int,
        default=1,
        help='Minimum number of children to visualize (default: 1)',
    )
    parser.add_argument(
        '--max-families',
        type=int,
        default=None,
        help='Maximum number of families to visualize (default: all)',
    )

    args = parser.parse_args()

    scene_root = args.scene_root.resolve()
    tree_path = scene_root / 'relations' / 'sam2_tree.json'

    if not tree_path.exists():
        print(f"❌ Tree file not found: {tree_path}", file=sys.stderr)
        return 1

    # Infer data_path if not provided
    if args.data_path:
        data_path = args.data_path
    else:
        # Try to infer from scene name
        scene_name = scene_root.name
        dataset_root = Path('/media/public_dataset2/multiscan')
        data_path = dataset_root / scene_name / 'outputs' / 'color'

        if not data_path.exists():
            print(f"❌ Could not infer data_path. Please specify --data-path", file=sys.stderr)
            return 1

    output_dir = args.output_dir or (scene_root / 'family_viz')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Scene root: {scene_root}")
    print(f"📂 Data path: {data_path}")
    print(f"📂 Output dir: {output_dir}")

    # Load tree
    print("\n🌲 Loading SAM2 tree...")
    tree_data = load_sam2_tree(tree_path)
    levels = tree_data['meta']['levels']

    # Find families
    families = find_families(tree_data)
    print(f"Found {len(families)} families with children")

    # Filter by min_children
    families = [f for f in families if len(f['children']) >= args.min_children]
    print(f"After filtering (min_children={args.min_children}): {len(families)} families")

    if args.max_families:
        families = families[:args.max_families]
        print(f"Limiting to first {args.max_families} families")

    if not families:
        print("⚠ No families to visualize")
        return 0

    # Load NPZ manifests for all levels
    print("\n📊 Loading object-frame mappings...")
    level_data = {}
    level_video_npz = {}

    for level in levels:
        obj_npz_path = scene_root / f"level_{level}" / f"object_segments_L{level:02d}.npz"
        if obj_npz_path.exists():
            obj_frames, linked_video = load_npz_manifest(obj_npz_path)
            level_data[level] = obj_frames
            print(f"  L{level}: {len(obj_frames)} objects, linked video: {linked_video}")

            # Get video NPZ path (try manifest's linked_video first, then fallback)
            if linked_video:
                video_npz_path = scene_root / f"level_{level}" / linked_video
                if not video_npz_path.exists():
                    # Try alternative naming: video_segments_LXX.npz
                    video_npz_path = scene_root / f"level_{level}" / f"video_segments_L{level:02d}.npz"

                if video_npz_path.exists():
                    level_video_npz[level] = video_npz_path
                else:
                    print(f"    ⚠ Video segments not found for L{level}")
        else:
            print(f"  L{level}: ⚠ Object segments NPZ not found")
            level_data[level] = {}

    # Visualize each family
    print(f"\n🎨 Generating visualizations for {len(families)} families...\n")

    for i, family in enumerate(families, 1):
        # Find best frame (with mask validation)
        best_frame = find_best_frame_for_family(family, level_data, level_video_npz)

        if best_frame is None:
            print(f"[{i}/{len(families)}] ⚠ Family L{family['level']} Obj#{family['obj_id']}: No frames found")
            continue

        # Generate visualization
        output_name = f"family_L{family['level']}_obj{family['obj_id']:04d}_frame{best_frame:06d}.jpg"
        output_path = output_dir / output_name

        print(f"[{i}/{len(families)}] Visualizing Family L{family['level']} Obj#{family['obj_id']} "
              f"({len(family['children'])} children) at frame {best_frame}...")

        try:
            visualize_family(
                family,
                best_frame,
                scene_root,
                data_path,
                output_path,
                level_video_npz,
                levels=levels,
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Done! Visualizations saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
