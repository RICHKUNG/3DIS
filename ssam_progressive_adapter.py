"""
Adapter that reuses Semantic-SAM's progressive_refinement.py to generate
per-level mask candidates for a list of frames, returning data in the
format expected by our pipeline (bbox XYWH, segmentation, area, level, etc.).

Runs under the Semantic-SAM environment.
"""

import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np

DEFAULT_SEMANTIC_SAM_ROOT = "/media/Pluto/richkung/Semantic-SAM"

if DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.append(DEFAULT_SEMANTIC_SAM_ROOT)

# Import Semantic-SAM modules
from semantic_sam import build_semantic_sam, prepare_image  # noqa: E402
from progressive_refinement import (
    progressive_refinement_masks,  # noqa: E402
    setup_output_directories,      # noqa: E402
)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def bbox_from_mask_xyxy(seg: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]


def generate_with_progressive(
    frames_dir: str,
    selected_frames: List[str],
    sam_ckpt_path: str,
    levels: List[int],
    min_area: int = 300,
    save_root: str = None,
) -> Dict[int, List[List[Dict[str, Any]]]]:
    """Generate per-level candidates using progressive_refinement.

    Returns: dict level -> list over frames -> list of candidates dicts
    Each candidate: {'frame_idx', 'frame_name', 'bbox'(XYWH), 'area', 'stability_score'(1.0), 'level', 'segmentation'}
    """
    # Workdir for Semantic-SAM so relative configs resolve
    try:
        os.chdir(DEFAULT_SEMANTIC_SAM_ROOT)
    except Exception:
        pass

    semantic_sam = build_semantic_sam(model_type="L", ckpt=sam_ckpt_path)

    # Prepare a lightweight output_dirs for progressive_refinement (disable viz to minimize IO)
    if save_root is None:
        save_root = ensure_dir(os.path.join(frames_dir, "_tmp_progressive"))

    per_level: Dict[int, List[List[Dict[str, Any]]]] = {L: [] for L in levels}

    for f_idx, fname in enumerate(selected_frames):
        image_path = os.path.join(frames_dir, fname)
        # Create isolated output dirs for this image to avoid clashes
        image_out = ensure_dir(os.path.join(save_root, f"pr_{os.path.splitext(fname)[0]}"))
        out_dirs = setup_output_directories(image_out)

        # Run progressive refinement on this single image
        results = progressive_refinement_masks(
            semantic_sam,
            image_path,
            level_sequence=levels,
            output_dirs=out_dirs,
            min_area=min_area,
            max_masks_per_level=2000,
            save_viz=False,
        )

        for L in levels:
            masks = results['levels'].get(L, {}).get('masks', [])
            # Build candidate list for this frame
            frame_cands: List[Dict[str, Any]] = []
            for m in masks:
                seg = m.get('segmentation')
                if seg is None:
                    continue
                area = int(m.get('area', int(np.sum(seg))))
                x1, y1, x2, y2 = bbox_from_mask_xyxy(seg)
                bbox = xyxy_to_xywh((x1, y1, x2, y2))
                stability = float(m.get('stability_score', 1.0))
                frame_cands.append({
                    'frame_idx': f_idx,
                    'frame_name': fname,
                    'bbox': bbox,
                    'area': area,
                    'stability_score': stability,
                    'level': L,
                    'segmentation': seg,
                })
            per_level[L].append(frame_cands)

    return per_level
