#!/usr/bin/env python3
"""
Diagnose SigLIP feature extraction by visualizing crops and intermediate results.

This helps understand why GT similarity is low:
- Visualize the bounding boxes being extracted
- Check if crops contain the actual object
- Compute text-to-text similarity as baseline
- Compare different text prompt formats
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from my3dis.aggregation.aggregation_pipeline import AggregationPipeline


def test_text_prompts(
    model,
    tokenizer,
    device: torch.device,
) -> None:
    """Test different text prompt formats."""
    print("\n" + "="*60)
    print("Text-to-Text Similarity Tests")
    print("="*60)

    # Test different prompt formats
    prompt_variants = {
        "door frame": [
            "door frame",
            "frame of door",
            "wooden door frame",
            "door frame in bathroom",
            "white door frame",
        ],
        "toilet lid": [
            "toilet lid",
            "lid of toilet",
            "white toilet lid",
            "toilet seat lid",
            "bathroom toilet lid",
        ],
        "cabinet door": [
            "cabinet door",
            "door of cabinet",
            "wooden cabinet door",
            "bathroom cabinet door",
            "white cabinet door",
        ],
    }

    for base_label, variants in prompt_variants.items():
        print(f"\n--- Testing variants of '{base_label}' ---")

        # Get features for all variants
        inputs = tokenizer(variants, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # Compute pairwise similarities
        sim_matrix = feats @ feats.T

        for i, text_i in enumerate(variants):
            for j, text_j in enumerate(variants):
                if i < j:  # Only upper triangle
                    sim = sim_matrix[i, j].item()
                    print(f"  '{text_i:30s}' <-> '{text_j:30s}': {sim:.4f}")


def visualize_proposal_projection(
    pipeline: AggregationPipeline,
    proposal_mask: torch.Tensor,
    output_dir: Path,
    label_text: str,
    max_frames: int = 10,
) -> None:
    """
    Visualize how a proposal projects onto frames.

    Shows:
    - Which frames the proposal is visible in
    - The bounding boxes being used
    - Number of visible points per frame
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    prop_points = torch.where(proposal_mask)[0]
    print(f"\nProposal '{label_text}': {len(prop_points)} points")

    # Find visible frames
    visible_frames = []
    for frame_idx in range(pipeline.inside_mask.shape[0]):
        inside = pipeline.inside_mask[frame_idx, prop_points]
        if inside.sum() > 0:
            # Check depth visibility
            proj_coords = pipeline.projected_points[frame_idx, prop_points[inside]]
            depths = pipeline.point_depth[frame_idx, prop_points[inside]]

            H, W = pipeline.depth_maps[frame_idx].shape
            x_coords = proj_coords[:, 0].long().clamp(0, H - 1)
            y_coords = proj_coords[:, 1].long().clamp(0, W - 1)

            depth_at_pixels = pipeline.depth_maps[frame_idx][x_coords, y_coords]
            depth_diff = torch.abs(depths - depth_at_pixels)
            visible = depth_diff < pipeline.vis_depth_threshold

            num_visible = visible.sum().item()
            if num_visible > 20:  # Same threshold as in feature extraction
                visible_frames.append((frame_idx, num_visible))

    visible_frames = sorted(visible_frames, key=lambda x: -x[1])[:max_frames]
    print(f"Visible in {len(visible_frames)} frames (showing top {max_frames})")

    # Visualize top frames
    for rank, (frame_idx, num_visible) in enumerate(visible_frames):
        # Load image
        img = np.array(Image.open(pipeline.world2cam.color_paths[frame_idx]))

        # Get visible points in this frame
        inside = pipeline.inside_mask[frame_idx, prop_points]
        proj_coords = pipeline.projected_points[frame_idx, prop_points[inside]]
        depths = pipeline.point_depth[frame_idx, prop_points[inside]]

        H, W = pipeline.depth_maps[frame_idx].shape
        x_coords = proj_coords[:, 0].long().clamp(0, H - 1)
        y_coords = proj_coords[:, 1].long().clamp(0, W - 1)

        depth_at_pixels = pipeline.depth_maps[frame_idx][x_coords, y_coords]
        depth_diff = torch.abs(depths - depth_at_pixels)
        visible = depth_diff < pipeline.vis_depth_threshold

        visible_coords = proj_coords[visible]

        if len(visible_coords) == 0:
            continue

        # Apply scaling
        scale_x, scale_y = pipeline.scaling_params
        coords_scaled = visible_coords.clone().float()
        coords_scaled[:, 0] *= scale_x
        coords_scaled[:, 1] *= scale_y

        # Compute bounding box
        mi = torch.min(coords_scaled, dim=0).values
        ma = torch.max(coords_scaled, dim=0).values

        x1, y1 = int(mi[0].item()), int(mi[1].item())
        x2, y2 = int(ma[0].item()), int(ma[1].item())

        # Draw bounding boxes for 3 scales
        img_draw = Image.fromarray(img)
        draw = ImageDraw.Draw(img_draw)

        colors = ['red', 'orange', 'yellow']
        kexp = 0.2

        for scale_idx in range(3):
            x1_clip = max(0, x1)
            y1_clip = max(0, y1)
            x2_clip = min(img.shape[0], x2)
            y2_clip = min(img.shape[1], y2)

            draw.rectangle(
                [y1_clip, x1_clip, y2_clip, x2_clip],
                outline=colors[scale_idx],
                width=3,
            )

            # Draw scale label
            draw.text(
                (y1_clip + 5, x1_clip + 5),
                f"Scale {scale_idx+1}",
                fill=colors[scale_idx],
            )

            # Grow bbox
            dx = int((x2 - x1) * kexp)
            dy = int((y2 - y1) * kexp)
            x1 -= dx
            y1 -= dy
            x2 += dx
            y2 += dy

        # Draw visible points
        for coord in visible_coords:
            x, y = int(coord[0] * scale_x), int(coord[1] * scale_y)
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                draw.ellipse([y-2, x-2, y+2, x+2], fill='cyan')

        # Save
        output_path = output_dir / f"frame_{frame_idx:04d}_rank{rank+1}_vis{num_visible}.jpg"
        img_draw.save(output_path)
        print(f"  Frame {frame_idx}: {num_visible} visible points → {output_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose SigLIP feature extraction")
    parser.add_argument("--scene-path", required=True, type=Path)
    parser.add_argument("--annotation-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/siglip_diagnosis"))
    parser.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--max-frames-per-proposal", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    print(f"Loading SigLIP model: {args.siglip_model}")
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(args.siglip_model).to(device)
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    tokenizer = AutoTokenizer.from_pretrained(args.siglip_model)
    model.eval()

    # Test text prompts
    test_text_prompts(model, tokenizer, device)

    # Initialize pipeline
    print(f"\nInitializing WORLD_2_CAM for scene {args.scene_path.name}")
    exp_dir = args.output_dir / args.scene_path.name / "temp"
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "uni3dis": {"vis_depth_threshold": 0.1, "frequency": 50},
        "network2d": {"masklet": {"resize_ratio": 0.3}},
    }

    pipeline = AggregationPipeline(
        exp_dir=str(exp_dir),
        scene_path=str(args.scene_path),
        device=str(device),
        config=config,
        iou_threshold=0.8,
        area_fraction=1 / 500,
        pooling_mode="average",
    )
    pipeline._init_world2cam()

    # Load GT masks
    print(f"\nLoading GT annotations from {args.annotation_file}")
    labels = np.loadtxt(args.annotation_file, dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Sort by size
    sorted_indices = np.argsort(-counts)

    # Visualize top proposals
    print(f"\nVisualizing top {args.max_proposals} proposals:")
    for idx in sorted_indices[:args.max_proposals]:
        raw_label = int(unique_labels[idx])
        if raw_label <= 0:
            continue

        point_count = int(counts[idx])
        if point_count < 150:
            continue

        label_id = raw_label // 1000
        mask = torch.from_numpy((labels == raw_label))

        output_subdir = args.output_dir / args.scene_path.name / f"proposal_{raw_label}"

        visualize_proposal_projection(
            pipeline=pipeline,
            proposal_mask=mask,
            output_dir=output_subdir,
            label_text=f"label_{label_id}",
            max_frames=args.max_frames_per_proposal,
        )

    print(f"\n✓ Diagnosis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
