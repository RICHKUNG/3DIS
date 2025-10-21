"""追蹤結果輸出與視覺化輔助。"""

from __future__ import annotations

import base64
import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from my3dis.common_utils import (
    PACKED_MASK_B64_KEY,
    PACKED_MASK_KEY,
    PACKED_ORIG_SHAPE_KEY,
    PACKED_SHAPE_KEY,
    ensure_dir,
    unpack_binary_mask,
)

from .helpers import format_scale_suffix, resize_mask_to_shape, scaled_npz_path

__all__ = [
    'encode_packed_mask_for_json',
    'decode_packed_mask_from_json',
    'build_video_segments_archive',
    'build_object_segments_archive',
    'save_comparison_proposals',
]

LOGGER = logging.getLogger(__name__)


def encode_packed_mask_for_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serialisable copy of a packed mask payload."""

    encoded: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == PACKED_MASK_KEY:
            arr = np.asarray(value, dtype=np.uint8)
            encoded[PACKED_MASK_B64_KEY] = base64.b64encode(arr.tobytes()).decode('ascii')
        elif isinstance(value, np.ndarray):
            encoded[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            encoded[key] = value.item()
        else:
            encoded[key] = value
    if PACKED_MASK_KEY in encoded:
        encoded.pop(PACKED_MASK_KEY, None)
    return encoded


def decode_packed_mask_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON-friendly packed mask back to numpy-compatible form."""

    decoded: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == PACKED_MASK_B64_KEY:
            decoded[PACKED_MASK_KEY] = np.frombuffer(
                base64.b64decode(value.encode('ascii')),
                dtype=np.uint8,
            )
        elif key in (PACKED_SHAPE_KEY, PACKED_ORIG_SHAPE_KEY) and isinstance(value, list):
            decoded[key] = tuple(int(v) for v in value)
        else:
            decoded[key] = value
    decoded.pop(PACKED_MASK_B64_KEY, None)
    return decoded


def _ensure_frames_dir(path: str) -> str:
    root_path = Path(path)
    root_path.parent.mkdir(parents=True, exist_ok=True)
    return str(root_path)


def _frame_entry_name(frame_idx: int) -> str:
    return f"frames/frame_{int(frame_idx):06d}.json"

DEFAULT_COMPARISON_MAX_FRAMES = 12


def _normalize_stride(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        stride = int(value)
    except (TypeError, ValueError):
        return None
    return stride if stride > 0 else None


def _normalize_max_samples(value: Optional[int]) -> Optional[int]:
    if value is None:
        return DEFAULT_COMPARISON_MAX_FRAMES
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_COMPARISON_MAX_FRAMES
    if parsed <= 0:
        return None
    return parsed


def _downsample_evenly(values: Sequence[int], target: int) -> List[int]:
    if target >= len(values):
        return list(values)
    if target <= 1:
        return [values[0]]
    positions = np.linspace(0, len(values) - 1, num=target, dtype=int)
    selected = [int(pos) for pos in positions]
    selected[0] = 0
    selected[-1] = len(values) - 1
    result: List[int] = []
    for idx in selected:
        value = values[idx]
        if value in result:
            continue
        result.append(value)
    if result[0] != values[0]:
        result.insert(0, values[0])
    if result[-1] != values[-1]:
        result.append(values[-1])
    return result[:target] if target > 0 else result


def _apply_sampling_to_frames(
    frames: List[int],
    sample_stride: Optional[int],
    max_samples: Optional[int],
) -> List[int]:
    if not frames:
        return []
    stride = _normalize_stride(sample_stride)
    target = _normalize_max_samples(max_samples)

    sampled = list(frames)
    if stride:
        sampled = sampled[::stride]
        if sampled[-1] != frames[-1]:
            sampled.append(frames[-1])

    sampled = list(dict.fromkeys(sampled))
    if sampled[0] != frames[0]:
        sampled.insert(0, frames[0])
    if sampled[-1] != frames[-1]:
        sampled.append(frames[-1])

    if target is not None and len(sampled) > target:
        sampled = _downsample_evenly(sampled, target)
    return sorted(set(sampled))

def build_video_segments_archive(
    frames: Iterable[Dict[str, Any]],
    path: str,
    *,
    mask_scale_ratio: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Persist frame-major SAM2 results into a manifest-backed archive."""

    actual_path = scaled_npz_path(path, mask_scale_ratio)
    _ensure_frames_dir(actual_path)

    manifest_frames: List[Dict[str, Any]] = []
    meta = dict(metadata or {})
    meta['mask_scale_ratio'] = float(mask_scale_ratio)

    with zipfile.ZipFile(actual_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for frame in frames:
            frame_idx = int(frame['frame_index'])
            entry_name = _frame_entry_name(frame_idx)
            if 'objects' not in frame:
                frame['objects'] = {}
            manifest_frames.append(
                {
                    'frame_index': frame_idx,
                    'frame_name': frame.get('frame_name'),
                    'entry': entry_name,
                    'objects': sorted(frame['objects'].keys()),
                }
            )
            zf.writestr(entry_name, json.dumps(frame, ensure_ascii=False).encode('utf-8'))

        manifest = {
            'meta': meta,
            'frames': manifest_frames,
        }
        zf.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False).encode('utf-8'))

    return actual_path, manifest


def build_object_segments_archive(
    object_manifest: Dict[int, Dict[int, str]],
    path: str,
    *,
    mask_scale_ratio: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist an object→frame reference manifest mirroring the video archive."""

    actual_path = scaled_npz_path(path, mask_scale_ratio)
    _ensure_frames_dir(actual_path)

    meta = dict(metadata or {})
    meta['mask_scale_ratio'] = float(mask_scale_ratio)

    serialisable_objects: Dict[str, List[Dict[str, Any]]] = {}
    for obj_id, frame_map in object_manifest.items():
        entries: List[Dict[str, Any]] = []
        for frame_idx, entry_name in sorted(frame_map.items()):
            entries.append(
                {
                    'frame_index': int(frame_idx),
                    'frame_entry': entry_name,
                }
            )
        serialisable_objects[str(obj_id)] = entries

    payload = {
        'meta': meta,
        'objects': serialisable_objects,
    }

    with zipfile.ZipFile(actual_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('manifest.json', json.dumps(payload, ensure_ascii=False).encode('utf-8'))

    return actual_path


# NOTE: keep in sync with workflow/summary consumers when updating payload schema.
def save_comparison_proposals(
    viz_dir: str,
    base_frames_dir: str,
    filtered_per_frame: List[Optional[List[Dict[str, Any]]]],
    video_segments: Optional[Dict[int, Dict[int, Any]]],
    level: int,
    frame_numbers: Optional[List[int]] = None,
    frames_to_save: Optional[List[int]] = None,
    frame_name_lookup: Optional[Dict[int, str]] = None,
    subset_dir: Optional[str] = None,
    subset_map: Optional[Dict[int, str]] = None,
    sample_stride: Optional[int] = None,
    max_samples: Optional[int] = None,
    video_segments_archive: Optional[str] = None,
) -> Dict[str, Any]:
    """輸出 SAM2 與 SSAM 的遮罩對照圖，並回傳摘要描述。"""
    from PIL import Image, ImageDraw

    viz_path = Path(ensure_dir(viz_dir))
    compare_path = Path(ensure_dir(os.path.join(str(viz_path), 'compare')))

    archive_loader: Optional[Any] = None

    def _frame_has_mask(payload: Dict[int, Any]) -> bool:
        if not payload:
            return False
        for obj_payload in payload.values():
            arr = obj_payload
            if isinstance(arr, dict):
                arr = unpack_binary_mask(arr)
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.size and np.any(arr_np):
                return True
        return False

    def _close_archive_loader() -> None:
        nonlocal archive_loader
        if archive_loader is not None:
            try:
                archive_loader.close()  # type: ignore[attr-defined]
            except AttributeError:
                pass
            archive_loader = None

    def _load_archive_frame(frame_idx: int) -> Dict[int, Dict[str, Any]]:
        if not video_segments_archive:
            return {}
        nonlocal archive_loader
        if archive_loader is None:
            try:
                archive_loader = np.load(video_segments_archive, allow_pickle=True)
            except OSError:
                archive_loader = None
        if archive_loader is None:
            return {}
        entry_name = _frame_entry_name(frame_idx)
        try:
            raw_payload = archive_loader[entry_name]
        except KeyError:
            return {}
        if isinstance(raw_payload, bytes):
            json_bytes = raw_payload
        else:
            try:
                json_bytes = raw_payload.tobytes()
            except AttributeError:
                json_bytes = bytes(raw_payload)
        try:
            payload = json.loads(json_bytes.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        objects = payload.get('objects', {})
        result: Dict[int, Dict[str, Any]] = {}
        for obj_id_str, obj_payload in objects.items():
            if not isinstance(obj_payload, dict):
                continue
            decoded = decode_packed_mask_from_json(obj_payload)
            shape_val = decoded.get(PACKED_SHAPE_KEY)
            if isinstance(shape_val, list):
                decoded[PACKED_SHAPE_KEY] = tuple(int(v) for v in shape_val)
            full_shape_val = decoded.get(PACKED_ORIG_SHAPE_KEY)
            if isinstance(full_shape_val, list):
                decoded[PACKED_ORIG_SHAPE_KEY] = tuple(int(v) for v in full_shape_val)
            result[int(obj_id_str)] = decoded
        return result

    if frame_numbers is None:
        frame_numbers = [idx for idx in range(len(filtered_per_frame))]
    frame_numbers = [int(fn) for fn in frame_numbers]
    frame_number_to_local = {fn: idx for idx, fn in enumerate(frame_numbers)}

    requested_frames = sorted(frame_numbers)
    frames_to_render = list(requested_frames)
    if frames_to_save is not None:
        frames_to_render = [f for f in frames_to_save if f in frame_number_to_local]

    frames_to_render = _apply_sampling_to_frames(frames_to_render, sample_stride, max_samples)
    if not frames_to_render and requested_frames:
        frames_to_render = [requested_frames[0]]
    candidate_targets = list(frames_to_render)
    selected_frames = list(frames_to_render)

    rng = np.random.default_rng(0)
    sam_color_map: Dict[int, Tuple[int, int, int]] = {}

    subset_lookup: Dict[int, str] = {}
    if subset_map:
        try:
            subset_lookup = {int(k): str(v) for k, v in subset_map.items()}
        except Exception:
            subset_lookup = {}

    def resolve_frame_path(frame_idx: int) -> Optional[str]:
        """Find the best effort frame path across naming conventions."""
        candidates: List[str] = []
        seen: set[str] = set()

        if frame_name_lookup:
            name = frame_name_lookup.get(frame_idx)
            if name:
                name = str(name)
                candidates.append(os.path.join(base_frames_dir, name))
                if subset_dir:
                    candidates.append(os.path.join(subset_dir, name))

        subset_name = subset_lookup.get(frame_idx)
        if subset_name and subset_dir:
            candidates.append(os.path.join(subset_dir, subset_name))

        base_names = [
            f"{frame_idx:05d}.png",
            f"{frame_idx}.png",
            f"{frame_idx:05d}.jpg",
            f"{frame_idx}.jpg",
        ]
        for name in base_names:
            candidates.append(os.path.join(base_frames_dir, name))
            if subset_dir:
                candidates.append(os.path.join(subset_dir, name))

        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.exists(path):
                return path
        return None

    def build_instance_map_img(target_size: Tuple[int, int], masks: List[np.ndarray]) -> Image.Image:
        H, W = target_size
        inst_map = np.zeros((H, W), dtype=np.int32)
        label = 0
        for seg in masks:
            if seg is None:
                continue
            if seg.shape[:2] != (H, W):
                seg_img = Image.fromarray((seg.astype(np.uint8) * 255))
                seg_img = seg_img.resize((W, H), resample=Image.NEAREST)
                seg = np.array(seg_img) > 127
            label += 1
            inst_map[(seg) & (inst_map == 0)] = label
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for idx in range(1, inst_map.max() + 1):
            color = tuple(rng.integers(50, 255, size=3).tolist())
            rgb[inst_map == idx] = color
        return Image.fromarray(rgb, 'RGB')

    def build_sam_instance_map_img(target_size: Tuple[int, int], masks_by_id: Dict[int, np.ndarray]) -> Image.Image:
        """Render SAM instance map keeping colors aligned with global object ids."""
        H, W = target_size
        inst_map = np.zeros((H, W), dtype=np.int32)
        for obj_id in sorted(masks_by_id.keys()):
            seg = masks_by_id[obj_id]
            if seg is None:
                continue
            seg_arr = np.asarray(seg)
            if seg_arr.ndim > 2:
                seg_arr = np.squeeze(seg_arr)
            seg_arr = seg_arr > 0
            if seg_arr.shape != (H, W):
                seg_img = Image.fromarray((seg_arr.astype(np.uint8) * 255))
                seg_img = seg_img.resize((W, H), resample=Image.NEAREST)
                seg_arr = np.array(seg_img) > 127
            inst_map[(seg_arr) & (inst_map == 0)] = obj_id
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        unique_obj_ids = np.unique(inst_map)
        for obj_id in unique_obj_ids:
            if obj_id == 0:
                continue
            if obj_id not in sam_color_map:
                sam_color_map[obj_id] = tuple(rng.integers(50, 255, size=3).tolist())
            rgb[inst_map == obj_id] = sam_color_map[obj_id]
        return Image.fromarray(rgb, 'RGB')

    rendered_frames: List[int] = []
    rendered_images_rel: List[str] = []
    missing_frame_sources: List[int] = []
    missing_mask_shapes: List[int] = []
    skipped_without_candidates: List[int] = []

    try:
        for f_idx in frames_to_render:
            local_idx = frame_number_to_local.get(f_idx)
            if local_idx is None or local_idx >= len(filtered_per_frame):
                skipped_without_candidates.append(int(f_idx))
                continue

            filtered_candidates = filtered_per_frame[local_idx] or []

            H = W = None
            if filtered_candidates:
                seg0 = filtered_candidates[0].get('segmentation')
                if isinstance(seg0, np.ndarray):
                    H, W = seg0.shape[:2]

            preview_frame: Dict[int, Any] = {}
            if isinstance(video_segments, dict):
                preview_frame = video_segments.get(f_idx) or {}

            archive_frame = _load_archive_frame(int(f_idx))

            sam_frame: Dict[int, Any] = {}
            if archive_frame:
                sam_frame.update(archive_frame)

            if preview_frame:
                # Preview segments may be incomplete; merge while normalising keys.
                for obj_id, payload in preview_frame.items():
                    sam_frame[int(obj_id)] = payload

            if not _frame_has_mask(sam_frame):
                if _frame_has_mask(preview_frame):
                    sam_frame = {int(obj_id): payload for obj_id, payload in preview_frame.items()}
                else:
                    sam_frame = archive_frame or {}
            if H is None or W is None:
                if sam_frame:
                    first_mask = next(iter(sam_frame.values()))
                    if isinstance(first_mask, np.ndarray):
                        H, W = first_mask.shape[:2]
                    elif isinstance(first_mask, dict):
                        from my3dis.common_utils import normalize_shape_tuple
                        shape_hint = first_mask.get(PACKED_SHAPE_KEY) or first_mask.get('shape')
                        if shape_hint is not None:
                            shape_seq = normalize_shape_tuple(shape_hint)
                            if len(shape_seq) >= 2:
                                H, W = shape_seq[-2], shape_seq[-1]

            frame_path = resolve_frame_path(f_idx)
            if frame_path is None:
                missing_frame_sources.append(int(f_idx))
                continue
            if H is None or W is None:
                missing_mask_shapes.append(int(f_idx))
                continue

            frame_img = Image.open(frame_path).convert('RGB')
            frame_img = frame_img.resize((W, H), resample=Image.BILINEAR)

            filtered_masks = [
                resize_mask_to_shape(item.get('segmentation'), (H, W))
                for item in filtered_candidates
                if item.get('segmentation') is not None
            ]
            filtered_masks = [m for m in filtered_masks if m is not None]
            filtered_img = build_instance_map_img((H, W), filtered_masks)

            sam_masks: Dict[int, np.ndarray] = {}
            for obj_id, payload in sam_frame.items():
                mask = payload
                if isinstance(payload, dict):
                    mask = unpack_binary_mask(payload)
                arr = resize_mask_to_shape(mask, (H, W))
                if arr is None:
                    continue
                sam_masks[int(obj_id)] = np.asarray(arr, dtype=np.bool_)

            sam_img = build_sam_instance_map_img((H, W), sam_masks)

            canvas_w = frame_img.width * 3
            canvas_h = frame_img.height
            canvas = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
            canvas.paste(frame_img, (0, 0))
            canvas.paste(filtered_img, (frame_img.width, 0))
            canvas.paste(sam_img, (frame_img.width * 2, 0))

            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), f"Frame {f_idx}", fill=(255, 255, 255))
            draw.text((frame_img.width + 10, 10), "SSAM filtered", fill=(255, 255, 255))
            draw.text((frame_img.width * 2 + 10, 10), "SAM2 propagated", fill=(255, 255, 255))

            out_path = compare_path / f"L{int(level)}_{int(f_idx):05d}.png"
            canvas.save(out_path)
            rendered_frames.append(int(f_idx))
            rendered_images_rel.append(os.path.relpath(out_path, viz_path))
    finally:
        _close_archive_loader()

    rendered_count = len(rendered_frames)
    generated_at = datetime.now(timezone.utc).isoformat()

    issues: Dict[str, List[int]] = {}
    if missing_frame_sources:
        issues['missing_frame_sources'] = sorted(set(missing_frame_sources))
    if missing_mask_shapes:
        issues['missing_mask_shapes'] = sorted(set(missing_mask_shapes))
    if skipped_without_candidates:
        issues['skipped_without_candidates'] = sorted(set(skipped_without_candidates))

    warning_payload: Optional[Dict[str, Any]] = None
    fallback_path: Optional[Path] = None

    if rendered_count == 0:
        if not candidate_targets:
            warning_code = 'comparison.no_targets'
            warning_msg = 'No frames were selected for comparison rendering.'
        elif len(missing_frame_sources) == len(selected_frames) and selected_frames:
            warning_code = 'comparison.missing_sources'
            warning_msg = 'Source frames required for comparison previews were not found.'
        elif len(missing_mask_shapes) == len(selected_frames) and selected_frames:
            warning_code = 'comparison.missing_masks'
            warning_msg = 'Mask metadata was unavailable for comparison previews.'
        else:
            warning_code = 'comparison.render_empty'
            warning_msg = 'Comparison previews could not be generated.'

        warning_payload = {
            'code': warning_code,
            'message': warning_msg,
            'level': int(level),
            'details': {
                'requested_frames': requested_frames,
                'frames_attempted': selected_frames,
                **({'issues': issues} if issues else {}),
            },
        }
        fallback_path = compare_path / f"L{int(level):02d}_no_comparisons.json"
        fallback_payload = {
            'level': int(level),
            'generated_at': generated_at,
            'warning': warning_payload,
            'issues': issues,
            'requested_frames': requested_frames,
            'frames_attempted': selected_frames,
        }
        try:
            with fallback_path.open('w', encoding='utf-8') as handle:
                json.dump(fallback_payload, handle, indent=2, ensure_ascii=False)
        except OSError:
            LOGGER.warning("Failed to write comparison fallback artifact at %s", fallback_path, exc_info=True)

    summary_path = compare_path / f"L{int(level):02d}_comparison_summary.json"
    summary_payload = {
        'level': int(level),
        'generated_at': generated_at,
        'requested_frames': requested_frames,
        'frames_attempted': selected_frames,
        'rendered_frames': rendered_frames,
        'rendered_count': rendered_count,
        'rendered_images': rendered_images_rel,
    }
    if issues:
        summary_payload['issues'] = issues
    if warning_payload:
        summary_payload['warning'] = warning_payload
    if fallback_path is not None:
        summary_payload['fallback_path'] = os.path.relpath(fallback_path, viz_path)
    try:
        with summary_path.open('w', encoding='utf-8') as handle:
            json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
    except OSError:
        LOGGER.warning("Failed to write comparison summary at %s", summary_path, exc_info=True)

    result: Dict[str, Any] = {
        'level': int(level),
        'generated_at': generated_at,
        'rendered_count': rendered_count,
        'requested_frames': requested_frames,
        'frames_attempted': selected_frames,
        'rendered_frames': rendered_frames,
        'rendered_images_rel': rendered_images_rel,
        'summary_path': str(summary_path),
        'compare_dir': str(compare_path),
    }
    if issues:
        result['issues'] = issues
    if warning_payload:
        result['warning'] = warning_payload
    if fallback_path is not None:
        result['fallback_path'] = str(fallback_path)

    if warning_payload:
        LOGGER.warning(
            "[Level %s] No comparison previews generated (code=%s)",
            level,
            warning_payload['code'],
        )

    return result
