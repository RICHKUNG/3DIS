#!/usr/bin/env python3
"""Re-apply filtering on stored Semantic-SAM raw candidates."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


RAW_DIR_NAME = 'raw'
RAW_META_TEMPLATE = 'frame_{frame_idx:05d}.json'
RAW_MASK_TEMPLATE = 'frame_{frame_idx:05d}.npz'


def encode_mask(mask: np.ndarray) -> Dict[str, object]:
    bool_mask = np.asarray(mask, dtype=np.bool_, order='C')
    packed = np.packbits(bool_mask.reshape(-1))
    return {
        'shape': [int(dim) for dim in bool_mask.shape],
        'packed_bits_b64': base64.b64encode(packed.tobytes()).decode('ascii'),
    }


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


def iter_level_raw_frames(level_root: str) -> Iterable[int]:
    raw_dir = os.path.join(level_root, RAW_DIR_NAME)
    if not os.path.isdir(raw_dir):
        return []
    frame_indices = []
    for fname in os.listdir(raw_dir):
        if not fname.endswith('.json'):
            continue
        stem = fname[:-5]
        if not stem.startswith('frame_'):
            continue
        try:
            frame_idx = int(stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        frame_indices.append(frame_idx)
    return sorted(frame_indices)


def load_raw_frame(level_root: str, frame_idx: int) -> Optional[Dict[str, object]]:
    raw_dir = os.path.join(level_root, RAW_DIR_NAME)
    meta_path = os.path.join(raw_dir, RAW_META_TEMPLATE.format(frame_idx=frame_idx))
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    mask_path = os.path.join(raw_dir, RAW_MASK_TEMPLATE.format(frame_idx=frame_idx))
    mask_stack = None
    has_mask = None
    if os.path.exists(mask_path):
        npz = np.load(mask_path)
        mask_stack = npz['masks']
        has_mask = npz['has_mask'].astype(bool)
    return {
        'meta': meta,
        'mask_stack': mask_stack,
        'has_mask': has_mask,
    }


def filter_level(
    *,
    level_root: str,
    min_area: int,
    stability_threshold: float,
    verbose: bool = True,
) -> FilterStats:
    stats = FilterStats()
    raw_frames = list(iter_level_raw_frames(level_root))
    if not raw_frames:
        if verbose:
            print(f"No raw frames found in {level_root} — skipping")
        return stats

    filtered_dir = ensure_dir(os.path.join(level_root, 'filtered'))
    frames_meta: List[Dict[str, object]] = []

    for frame_idx in raw_frames:
        payload = load_raw_frame(level_root, frame_idx)
        if payload is None:
            continue
        meta = payload['meta']
        mask_stack = payload['mask_stack']
        has_mask = payload['has_mask']

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
