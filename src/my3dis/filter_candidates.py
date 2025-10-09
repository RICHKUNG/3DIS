#!/usr/bin/env python3
"""Re-apply filtering on stored Semantic-SAM raw candidates."""
from __future__ import annotations

if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))




import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from my3dis.common_utils import (
    PACKED_MASK_KEY,
    PACKED_SHAPE_KEY,
    encode_mask,
    ensure_dir,
    unpack_binary_mask,
)
from my3dis.raw_archive import RawCandidateArchiveReader


@dataclass
class FilterStats:
    frames: int = 0
    kept: int = 0
    dropped: int = 0

    def add_frame(self, kept_count: int, dropped_count: int) -> None:
        self.frames += 1
        self.kept += kept_count
        self.dropped += dropped_count

    def to_dict(self) -> Dict[str, int]:
        return {
            'frames': self.frames,
            'kept': self.kept,
            'dropped': self.dropped,
        }


def bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]




def filter_level(
    *,
    level_root: str,
    min_area: int,
    stability_threshold: float,
    verbose: bool = True,
) -> FilterStats:
    stats = FilterStats()
    archive = RawCandidateArchiveReader(level_root)
    raw_frames = archive.frame_indices()
    if not raw_frames:
        if verbose:
            print(f"No raw frames found in {level_root} — skipping")
        return stats

    filtered_dir = ensure_dir(os.path.join(level_root, 'filtered'))
    frames_meta: List[Dict[str, object]] = []

    for frame_idx in raw_frames:
        payload = archive.load_frame(frame_idx)
        if payload is None:
            continue
        meta = payload['meta']
        mask_stack = payload.get('mask_stack')
        packed_masks = payload.get('packed_masks')
        mask_shape = payload.get('mask_shape')
        has_mask = payload.get('has_mask')

        candidates: List[Dict[str, object]] = meta.get('candidates', [])  # type: ignore[assignment]
        kept_items: List[Dict[str, object]] = []
        dropped_count = 0
        local_id = 0

        for cand in candidates:
            stability = float(cand.get('stability_score', 1.0))
            area_meta = cand.get('area')
            raw_index = cand.get('raw_index')
            if raw_index is None:
                dropped_count += 1
                continue
            try:
                ri = int(raw_index)
            except (TypeError, ValueError):
                dropped_count += 1
                continue

            mask_arr = None
            if mask_stack is not None and 0 <= ri < len(mask_stack):
                if has_mask is None or bool(has_mask[ri]):
                    mask_arr = np.asarray(mask_stack[ri], dtype=bool)
            elif (
                packed_masks is not None
                and mask_shape is not None
                and 0 <= ri < len(packed_masks)
            ):
                if has_mask is None or bool(has_mask[ri]):
                    mask_payload = {
                        PACKED_MASK_KEY: packed_masks[ri],
                        PACKED_SHAPE_KEY: mask_shape,
                    }
                    mask_arr = unpack_binary_mask(mask_payload)

            if mask_arr is None:
                dropped_count += 1
                continue

            area = int(mask_arr.sum())
            if area == 0 and area_meta is not None:
                area = int(area_meta)

            if area < min_area or stability < stability_threshold:
                dropped_count += 1
                continue

            bbox = bbox_from_mask(mask_arr)
            if bbox is None:
                dropped_count += 1
                continue

            item = {
                k: v
                for k, v in cand.items()
                if k not in {'segmentation', 'raw_index'}
            }
            item['id'] = local_id
            item['area'] = area
            item['bbox_xyxy'] = bbox
            item['mask'] = encode_mask(mask_arr)
            kept_items.append(item)
            local_id += 1

        stats.add_frame(len(kept_items), dropped_count)
        frames_meta.append(
            {
                'frame_idx': int(meta.get('frame_idx', frame_idx)),
                'frame_name': meta.get('frame_name'),
                'count': len(kept_items),
                'items': kept_items,
            }
        )

    with open(os.path.join(filtered_dir, 'filtered.json'), 'w') as f:
        json.dump({'frames': frames_meta}, f, indent=2)

    return stats


def run_filtering(
    *,
    root: str,
    levels: Optional[List[int]] = None,
    min_area: int,
    stability_threshold: float,
    update_manifest: bool,
    quiet: bool = False,
) -> Dict[int, Dict[str, int]]:
    manifest_path = os.path.join(root, 'manifest.json')

    if levels is not None:
        levels = [int(x) for x in levels]
    elif os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        manifest_levels = manifest.get('levels')
        if isinstance(manifest_levels, list):
            try:
                levels = [int(x) for x in manifest_levels]
            except ValueError:
                levels = None
    if not levels:
        raise ValueError('No levels specified and manifest missing levels')

    stats_summary: Dict[int, Dict[str, int]] = {}

    for level in levels:
        level_root = os.path.join(root, f'level_{level}')
        if not os.path.isdir(level_root):
            print(f"Level {level} directory missing at {level_root}", file=sys.stderr)
            continue
        if not quiet:
            print(f"Filtering level {level} (min_area={min_area}, stability>={stability_threshold})")
        stats = filter_level(
            level_root=level_root,
            min_area=min_area,
            stability_threshold=stability_threshold,
            verbose=not quiet,
        )
        stats_summary[level] = stats.to_dict()
        if not quiet:
            print(
                f"  Level {level}: kept {stats.kept} masks over {stats.frames} frames; dropped {stats.dropped}"
            )

    if update_manifest and os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            manifest = {}
        manifest.setdefault('filtering', {})
        manifest['filtering'].update(
            {
                'applied': True,
                'min_area': int(min_area),
                'stability_threshold': float(stability_threshold),
                'ts_epoch': int(time.time()),
                'stats': stats_summary,
            }
        )
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    return stats_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Filter stored Semantic-SAM candidates')
    parser.add_argument('--candidates-root', required=True, help='Run directory that contains level_* folders')
    parser.add_argument('--levels', default=None, help='Comma separated levels; defaults to manifest levels')
    parser.add_argument('--min-area', type=int, default=300)
    parser.add_argument('--stability-threshold', type=float, default=0.9)
    parser.add_argument('--update-manifest', action='store_true', help='Write filtering config back to manifest.json')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-level logs')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.candidates_root
    start_time = time.perf_counter()
    levels = None
    if args.levels:
        try:
            levels = [int(x) for x in args.levels.split(',') if x.strip()]
        except ValueError:
            print(f"Invalid --levels value: {args.levels}", file=sys.stderr)
            return 1

    try:
        run_filtering(
            root=root,
            levels=levels,
            min_area=args.min_area,
            stability_threshold=args.stability_threshold,
            update_manifest=args.update_manifest,
            quiet=args.quiet,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_time = time.perf_counter() - start_time
    if not args.quiet:
        print(f"Filtering finished in {total_time:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
