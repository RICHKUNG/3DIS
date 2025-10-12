"""Compatibility helpers for consuming tracking outputs with legacy-style APIs."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from my3dis.common_utils import unpack_binary_mask

from .outputs import decode_packed_mask_from_json

__all__ = [
    'LegacyFrame',
    'iter_legacy_frames',
    'load_legacy_frame_dict',
    'load_legacy_video_segments',
    'load_object_segments_manifest',
]


@dataclass(frozen=True)
class LegacyFrame:
    """Frame-major payload mirroring the pre-refactor tracking output."""

    frame_index: int
    frame_name: Optional[str]
    objects: Dict[int, Dict[str, Any]]


def _read_json_from_zip(zf: zipfile.ZipFile, entry: str) -> Dict[str, Any]:
    try:
        data = zf.read(entry)
    except KeyError as exc:
        raise FileNotFoundError(f'Entry {entry!r} not found in archive') from exc
    return json.loads(data.decode('utf-8'))


def _decode_object_payload(
    payload: Dict[str, Any],
    *,
    unpack_masks: bool,
) -> Dict[str, Any]:
    decoded = decode_packed_mask_from_json(payload)
    if unpack_masks:
        mask_arr = unpack_binary_mask(decoded)
        decoded = dict(decoded)
        decoded.setdefault('mask', mask_arr)
    return decoded


def iter_legacy_frames(
    archive_path: str,
    *,
    unpack_masks: bool = False,
    frame_filter: Optional[Iterable[int]] = None,
) -> Iterator[LegacyFrame]:
    """Yield frame payloads in a shape close to the legacy np.savez output."""

    target = Path(archive_path)
    if not target.exists():
        raise FileNotFoundError(f'Archive not found: {archive_path}')

    allowed: Optional[set[int]] = None
    if frame_filter is not None:
        allowed = {int(idx) for idx in frame_filter}

    with zipfile.ZipFile(target, mode='r') as zf:
        manifest = _read_json_from_zip(zf, 'manifest.json')
        frames_meta = manifest.get('frames') or []
        for meta in frames_meta:
            if not isinstance(meta, dict):
                continue
            try:
                frame_idx = int(meta.get('frame_index'))
            except (TypeError, ValueError):
                continue
            if allowed is not None and frame_idx not in allowed:
                continue
            entry_name = meta.get('entry')
            if not isinstance(entry_name, str):
                continue
            frame_json = _read_json_from_zip(zf, entry_name)
            frame_name = frame_json.get('frame_name')
            objects_raw = frame_json.get('objects', {})
            converted: Dict[int, Dict[str, Any]] = {}
            if isinstance(objects_raw, dict):
                for obj_id_str, payload in objects_raw.items():
                    try:
                        obj_id = int(obj_id_str)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    converted[obj_id] = _decode_object_payload(
                        payload,
                        unpack_masks=unpack_masks,
                    )
            yield LegacyFrame(frame_index=frame_idx, frame_name=frame_name, objects=converted)


def load_legacy_frame_dict(
    archive_path: str,
    *,
    unpack_masks: bool = False,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Return frame-major mapping {frame_idx: {obj_id: mask_payload}}."""

    frame_map: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for frame in iter_legacy_frames(archive_path, unpack_masks=unpack_masks):
        frame_map[frame.frame_index] = frame.objects
    return frame_map


def load_legacy_video_segments(
    archive_path: str,
    *,
    unpack_masks: bool = False,
) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]], Dict[str, Any]]:
    """Mimic the legacy np.savez payload, returning (frames_dict, manifest)."""

    frames = load_legacy_frame_dict(archive_path, unpack_masks=unpack_masks)
    target = Path(archive_path)
    with zipfile.ZipFile(target, mode='r') as zf:
        manifest = _read_json_from_zip(zf, 'manifest.json')
    return frames, manifest


def load_object_segments_manifest(object_archive_path: str) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Load the object-major manifest and map obj_id → frame_index → packed entry."""

    target = Path(object_archive_path)
    if not target.exists():
        raise FileNotFoundError(f'Archive not found: {object_archive_path}')

    results: Dict[int, Dict[int, Dict[str, Any]]] = {}
    with zipfile.ZipFile(target, mode='r') as zf:
        manifest = _read_json_from_zip(zf, 'manifest.json')
        objects = manifest.get('objects') or {}
        if not isinstance(objects, dict):
            return results
        for obj_id_str, entries in objects.items():
            try:
                obj_id = int(obj_id_str)
            except (TypeError, ValueError):
                continue
            per_frame: Dict[int, Dict[str, Any]] = {}
            if isinstance(entries, list):
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    try:
                        frame_idx = int(item.get('frame_index'))
                    except (TypeError, ValueError):
                        continue
                    reference = item.get('frame_entry')
                    if isinstance(reference, str):
                        per_frame[frame_idx] = {'frame_entry': reference}
            results[obj_id] = per_frame
    return results
