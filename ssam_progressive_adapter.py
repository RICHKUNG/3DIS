"""
Adapter that reuses Semantic-SAM's progressive_refinement.py to generate
per-level mask candidates for a list of frames, returning data in the
format expected by our pipeline (bbox XYWH, segmentation, area, level, etc.).

Runs under the Semantic-SAM environment.
"""

import contextlib
import io
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common_utils import (
    bbox_from_mask_xyxy,
    bbox_xyxy_to_xywh,
    ensure_dir,
)
from pipeline_defaults import (
    DEFAULT_SEMANTIC_SAM_ROOT as _DEFAULT_SEMANTIC_SAM_ROOT,
    expand_default,
)

_SEM_ROOT_STR = expand_default(_DEFAULT_SEMANTIC_SAM_ROOT)
DEFAULT_SEMANTIC_SAM_ROOT = _SEM_ROOT_STR

if DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.append(DEFAULT_SEMANTIC_SAM_ROOT)

# Import Semantic-SAM modules
from semantic_sam import build_semantic_sam, prepare_image  # noqa: E402
from progressive_refinement import (
    progressive_refinement_masks,  # noqa: E402
    setup_output_directories,      # noqa: E402
)


LOGGER = logging.getLogger("my3dis.ssam_progressive")


def _extract_gap_components(segs: List[np.ndarray], fill_area: int) -> List[np.ndarray]:
    """Return boolean masks for uncovered regions above the fill_area threshold."""
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

            if len(coords) < fill_area:
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
    fill_area: Optional[int] = None,
    save_root: str = None,
    persist_outputs: bool = False,
    enable_gap_fill: bool = True,
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

    if fill_area is None:
        fill_area = min_area
    fill_area = int(max(0, fill_area))

    # When persist_outputs is False we rely on temporary directories so no artifacts remain on disk.
    if save_root is not None and persist_outputs:
        save_root = ensure_dir(save_root)

    per_level: Dict[int, List[List[Dict[str, Any]]]] = {L: [] for L in levels}

    verbose = os.environ.get("MY3DIS_VERBOSE_PROGRESSIVE") == "1"

    base_level = min(levels) if levels else None

    for f_idx, fname in enumerate(selected_frames):
        image_path = os.path.join(frames_dir, fname)

        def run_progressive(output_dirs: Dict[str, str]):
            if verbose:
                return progressive_refinement_masks(
                    semantic_sam,
                    image_path,
                    level_sequence=levels,
                    output_dirs=output_dirs,
                    min_area=min_area,
                    max_masks_per_level=2000,
                    save_viz=False,
                )
            buf_out, buf_err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    return progressive_refinement_masks(
                        semantic_sam,
                        image_path,
                        level_sequence=levels,
                        output_dirs=output_dirs,
                        min_area=min_area,
                        max_masks_per_level=2000,
                        save_viz=False,
                    )
            except Exception:
                # Preserve captured logs to aid debugging before re-raising
                print(buf_out.getvalue(), file=sys.stderr, end="")
                print(buf_err.getvalue(), file=sys.stderr, end="")
                raise

        if persist_outputs:
            image_out = ensure_dir(os.path.join(save_root, f"pr_{os.path.splitext(fname)[0]}"))
            out_dirs = setup_output_directories(image_out)
            results = run_progressive(out_dirs)
        else:
            tmp_kwargs = {}
            if save_root is not None:
                ensure_dir(save_root)
                tmp_kwargs["dir"] = save_root
            with tempfile.TemporaryDirectory(**tmp_kwargs) as tmp_root:
                out_dirs = setup_output_directories(tmp_root)
                results = run_progressive(out_dirs)

        additional_gap_masks: List[Dict[str, Any]] = []
        if enable_gap_fill and base_level is not None:
            base_masks = results['levels'].get(base_level, {}).get('masks', [])
            base_segs = [m.get('segmentation') for m in base_masks if m.get('segmentation') is not None]
            gap_components = _extract_gap_components(base_segs, fill_area)
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
                bbox = bbox_xyxy_to_xywh((x1, y1, x2, y2))
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
