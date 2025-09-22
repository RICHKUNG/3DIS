"""
Adapter that reuses Semantic-SAM's progressive_refinement.py to generate
per-level mask candidates for a list of frames, returning data in the
format expected by our pipeline (bbox XYWH, segmentation, area, level, etc.).

Runs under the Semantic-SAM environment.
"""

import os
import sys
import contextlib
import io
import logging
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


LOGGER = logging.getLogger("my3dis.ssam_progressive")


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


def _extract_gap_components(segs: List[np.ndarray], min_area: int) -> List[np.ndarray]:
    """Return boolean masks for uncovered connected components above min_area."""
    valid = [np.asarray(seg, dtype=bool) for seg in segs if seg is not None]
    if not valid:
        return []

    ref_shape = valid[0].shape
    coverage = np.zeros(ref_shape, dtype=bool)
    for seg in valid:
        if seg.shape != ref_shape:
            LOGGER.debug(
                "Skipping gap computation for mask with mismatched shape %s != %s",
                seg.shape,
                ref_shape,
            )
            continue
        coverage |= seg

    gap = np.logical_not(coverage)
    if not gap.any():
        return []

    h, w = gap.shape
    visited = np.zeros_like(gap, dtype=bool)
    gap_components: List[np.ndarray] = []

    def neighbors(y: int, x: int):
        if y > 0:
            yield y - 1, x
        if y + 1 < h:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x + 1 < w:
            yield y, x + 1

    for y in range(h):
        for x in range(w):
            if not gap[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: List[Tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in neighbors(cy, cx):
                    if gap[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(coords) < min_area:
                continue

            comp_mask = np.zeros_like(gap, dtype=bool)
            ys, xs = zip(*coords)
            comp_mask[ys, xs] = True
            gap_components.append(comp_mask)

    return gap_components


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

    verbose = os.environ.get("MY3DIS_VERBOSE_PROGRESSIVE") == "1"

    base_level = min(levels) if levels else None

    for f_idx, fname in enumerate(selected_frames):
        image_path = os.path.join(frames_dir, fname)
        # Create isolated output dirs for this image to avoid clashes
        image_out = ensure_dir(os.path.join(save_root, f"pr_{os.path.splitext(fname)[0]}"))
        out_dirs = setup_output_directories(image_out)

        # Run progressive refinement on this single image
        if verbose:
            results = progressive_refinement_masks(
                semantic_sam,
                image_path,
                level_sequence=levels,
                output_dirs=out_dirs,
                min_area=min_area,
                max_masks_per_level=2000,
                save_viz=False,
            )
        else:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    results = progressive_refinement_masks(
                        semantic_sam,
                        image_path,
                        level_sequence=levels,
                        output_dirs=out_dirs,
                        min_area=min_area,
                        max_masks_per_level=2000,
                        save_viz=False,
                    )
            except Exception:
                # Preserve captured logs to aid debugging before re-raising
                print(buf_out.getvalue(), file=sys.stderr, end="")
                print(buf_err.getvalue(), file=sys.stderr, end="")
                raise

        additional_gap_masks: List[Dict[str, Any]] = []
        if base_level is not None:
            base_masks = results['levels'].get(base_level, {}).get('masks', [])
            base_segs = [m.get('segmentation') for m in base_masks if m.get('segmentation') is not None]
            gap_components = _extract_gap_components(base_segs, min_area)
            if gap_components:
                for comp in gap_components:
                    additional_gap_masks.append({
                        'segmentation': comp,
                        'stability_score': 1.0,
                        'area': int(comp.sum()),
                        'level': base_level,
                        'source': 'gap_fill',
                    })
                LOGGER.info(
                    "Frame %s: added %d gap-fill masks at level %s",
                    fname,
                    len(additional_gap_masks),
                    base_level,
                )

        for L in levels:
            masks = list(results['levels'].get(L, {}).get('masks', []))
            if L == base_level and additional_gap_masks:
                masks.extend(additional_gap_masks)
            # Build candidate list for this frame
            frame_cands: List[Dict[str, Any]] = []
            for m in masks:
                seg = m.get('segmentation')
                if seg is None:
                    continue
                seg_bool = np.asarray(seg, dtype=bool)
                area = int(m.get('area', int(np.sum(seg_bool))))
                if area == 0:
                    continue
                x1, y1, x2, y2 = bbox_from_mask_xyxy(seg_bool)
                bbox = xyxy_to_xywh((x1, y1, x2, y2))
                stability = float(m.get('stability_score', 1.0))
                frame_cands.append({
                    'frame_idx': f_idx,
                    'frame_name': fname,
                    'bbox': bbox,
                    'area': area,
                    'stability_score': stability,
                    'level': L,
                    'segmentation': seg_bool,
                })
            per_level[L].append(frame_cands)

    return per_level
