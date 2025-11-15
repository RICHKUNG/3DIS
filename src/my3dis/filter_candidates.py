#!/usr/bin/env python3
"""Re-apply filtering on stored Semantic-SAM raw candidates."""
from __future__ import annotations

# Ensure src/ is in path for direct execution (inline to avoid circular import)
if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Configure temp directory to use /media partition (avoid root partition space issues)
    tmp_dir = project_root / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    import os
    os.environ['TMPDIR'] = str(tmp_dir)
    os.environ['TEMP'] = str(tmp_dir)
    os.environ['TMP'] = str(tmp_dir)

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from my3dis.cascade_filter import CascadeFilter
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
    area_rejected: int = 0
    stability_rejected: int = 0
    cascade_rejected: int = 0

    def add_frame(self, kept_count: int, dropped_count: int,
                  area_rej: int = 0, stab_rej: int = 0, casc_rej: int = 0) -> None:
        self.frames += 1
        self.kept += kept_count
        self.dropped += dropped_count
        self.area_rejected += area_rej
        self.stability_rejected += stab_rej
        self.cascade_rejected += casc_rej

    def to_dict(self) -> Dict[str, int]:
        result = {
            'frames': self.frames,
            'kept': self.kept,
            'dropped': self.dropped,
        }
        if self.area_rejected > 0 or self.stability_rejected > 0 or self.cascade_rejected > 0:
            result['breakdown'] = {
                'area_rejected': self.area_rejected,
                'stability_rejected': self.stability_rejected,
                'cascade_rejected': self.cascade_rejected,
            }
        return result


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
    use_cascade: bool = False,
    verbose: bool = True,
) -> FilterStats:
    stats = FilterStats()
    archive = RawCandidateArchiveReader(level_root)
    raw_frames = archive.frame_indices()
    if not raw_frames:
        if verbose:
            print(f"No raw frames found in {level_root} — skipping")
        return stats

    # Check if data supports cascade filtering
    supports_cascade = False
    cascade_warned = False
    if use_cascade:
        # Check first frame for unique_id presence
        first_payload = archive.load_frame(raw_frames[0])
        if first_payload and first_payload.get('meta'):
            first_candidates = first_payload['meta'].get('candidates', [])
            if first_candidates and any('unique_id' in c for c in first_candidates):
                supports_cascade = True
            else:
                if verbose:
                    print(f"  WARNING: Cascade filtering requested but data lacks unique_id/parent_unique_id")
                    print(f"           Falling back to traditional filtering")
                    print(f"           Re-run SSAM stage with cascade_filtering=true to enable")
                cascade_warned = True

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

        # Prepare candidates with area and stability
        enriched_candidates = []
        for cand in candidates:
            raw_index = cand.get('raw_index')
            if raw_index is None:
                continue
            try:
                ri = int(raw_index)
            except (TypeError, ValueError):
                continue

            # Load mask
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
                continue

            # Calculate area
            area = int(mask_arr.sum())
            if area == 0 and cand.get('area') is not None:
                area = int(cand.get('area'))

            # Enrich candidate with computed values
            enriched = dict(cand)
            enriched['computed_area'] = area
            enriched['mask_array'] = mask_arr
            enriched['stability'] = float(cand.get('stability_score', 1.0))
            enriched_candidates.append(enriched)

        # Apply filtering (cascade or traditional)
        if supports_cascade and use_cascade:
            # Prepare masks for cascade filter
            cascade_masks = []
            for ec in enriched_candidates:
                cascade_masks.append({
                    'unique_id': ec.get('unique_id'),
                    'parent_unique_id': ec.get('parent_unique_id'),
                    'area': ec['computed_area'],
                    'stability': ec['stability'],
                })

            # Apply cascade filter
            cascade_filter = CascadeFilter(
                min_area=float(min_area),
                stability_threshold=float(stability_threshold),
            )
            _, rejection_reasons = cascade_filter.filter_masks(cascade_masks)

            # Count rejection types
            area_rej_count = sum(1 for r in rejection_reasons.values() if r.startswith('area_too_small'))
            stab_rej_count = sum(1 for r in rejection_reasons.values() if r.startswith('low_stability'))
            casc_rej_count = sum(1 for r in rejection_reasons.values() if r.startswith('parent_filtered'))

            # Filter out rejected candidates
            kept_items = []
            local_id = 0
            for ec in enriched_candidates:
                unique_id = ec.get('unique_id')
                # BUGFIX (2025-11-13): Also reject candidates without unique_id
                # Cascade filter rejects these but doesn't add to rejection_reasons (can't use None as dict key)
                if not unique_id:
                    continue  # No unique_id - reject
                if unique_id in rejection_reasons:
                    continue  # Rejected by cascade filter

                # Build output item
                bbox = bbox_from_mask(ec['mask_array'])
                if bbox is None:
                    continue

                item = {
                    k: v
                    for k, v in ec.items()
                    if k not in {'segmentation', 'raw_index', 'mask_array', 'computed_area'}
                }
                item['id'] = local_id
                item['area'] = ec['computed_area']
                item['bbox_xyxy'] = bbox
                item['mask'] = encode_mask(ec['mask_array'])
                kept_items.append(item)
                local_id += 1

            dropped_count = len(enriched_candidates) - len(kept_items)
            stats.add_frame(len(kept_items), dropped_count, area_rej_count, stab_rej_count, casc_rej_count)
        else:
            # Traditional filtering (no cascade)
            kept_items = []
            local_id = 0
            area_rej_count = 0
            stab_rej_count = 0

            for ec in enriched_candidates:
                area = ec['computed_area']
                stability = ec['stability']

                # Apply thresholds
                if area < min_area:
                    area_rej_count += 1
                    continue
                if stability < stability_threshold:
                    stab_rej_count += 1
                    continue

                bbox = bbox_from_mask(ec['mask_array'])
                if bbox is None:
                    continue

                item = {
                    k: v
                    for k, v in ec.items()
                    if k not in {'segmentation', 'raw_index', 'mask_array', 'computed_area'}
                }
                item['id'] = local_id
                item['area'] = area
                item['bbox_xyxy'] = bbox
                item['mask'] = encode_mask(ec['mask_array'])
                kept_items.append(item)
                local_id += 1

            dropped_count = len(enriched_candidates) - len(kept_items)
            stats.add_frame(len(kept_items), dropped_count, area_rej_count, stab_rej_count, 0)

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
    use_cascade: bool = False,
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
            cascade_str = " (cascade)" if use_cascade else ""
            print(f"Filtering level {level}{cascade_str} (min_area={min_area}, stability>={stability_threshold})")
        stats = filter_level(
            level_root=level_root,
            min_area=min_area,
            stability_threshold=stability_threshold,
            use_cascade=use_cascade,
            verbose=not quiet,
        )
        stats_summary[level] = stats.to_dict()
        if not quiet:
            msg = f"  Level {level}: kept {stats.kept} masks over {stats.frames} frames; dropped {stats.dropped}"
            if stats.cascade_rejected > 0:
                msg += f" (cascade: {stats.cascade_rejected})"
            print(msg)

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
                'use_cascade': use_cascade,
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
    parser.add_argument('--cascade', action='store_true',
                        help='Enable cascade filtering (requires unique_id/parent_unique_id in data)')
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
            use_cascade=args.cascade,
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
