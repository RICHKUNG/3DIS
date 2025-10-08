"""Shared utility helpers for the My3DIS pipeline."""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


RAW_DIR_NAME = "raw"
RAW_META_TEMPLATE = "frame_{frame_idx:05d}.json"
RAW_MASK_TEMPLATE = "frame_{frame_idx:05d}.npz"


PACKED_MASK_KEY = "packed_bits"
PACKED_MASK_B64_KEY = "packed_bits_b64"
PACKED_SHAPE_KEY = "shape"
PACKED_ORIG_SHAPE_KEY = "full_resolution_shape"


def ensure_dir(path: str | Path) -> str:
    """Create directory if missing and return the absolute string path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def format_duration(seconds: float) -> str:
    """Render duration as mm:ss or HH:MM:SS when hours present."""
    total_seconds = max(0.0, float(seconds))
    rounded = int(round(total_seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_seconds(seconds: float) -> str:
    """Render duration using hours when needed, otherwise mm:ss."""
    return format_duration(seconds)


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    """Serialize boolean mask into JSON-friendly packed bits payload."""
    bool_mask = np.asarray(mask, dtype=np.bool_, order="C")
    packed = np.packbits(bool_mask.reshape(-1))
    return {
        "shape": [int(dim) for dim in bool_mask.shape],
        "packed_bits_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def downscale_binary_mask(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Downscale a boolean mask by ``ratio`` using a box filter + threshold."""

    try:
        ratio_val = float(ratio)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        ratio_val = 1.0

    if ratio_val >= 1.0:
        return np.asarray(mask, dtype=np.bool_)
    if ratio_val <= 0.0:
        raise ValueError("downscale ratio must be within (0, 1]")

    arr = np.asarray(mask, dtype=np.bool_)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported mask dimensionality: {arr.shape}")

    h, w = arr.shape
    new_h = max(1, int(round(h * ratio_val)))
    new_w = max(1, int(round(w * ratio_val)))
    if new_h == h and new_w == w:
        return arr

    from PIL import Image

    img = Image.fromarray(arr.astype(np.uint8) * 255)
    resample = getattr(Image, 'BOX', Image.BILINEAR)
    resized = img.resize((new_w, new_h), resample=resample)
    arr_float = np.asarray(resized, dtype=np.float32) / 255.0
    return arr_float >= 0.5


def pack_binary_mask(
    mask: np.ndarray,
    *,
    full_resolution_shape: Optional[Sequence[int]] = None,
) -> dict[str, object]:
    """Pack a boolean mask into a compact dict."""

    bool_mask = np.asarray(mask, dtype=np.bool_, order="C")
    packed = np.packbits(bool_mask.reshape(-1))
    payload: dict[str, object] = {
        PACKED_MASK_KEY: packed,
        PACKED_SHAPE_KEY: tuple(int(dim) for dim in bool_mask.shape),
    }

    if full_resolution_shape is not None:
        shape_arr = np.atleast_1d(full_resolution_shape)
        if shape_arr.size:
            payload[PACKED_ORIG_SHAPE_KEY] = tuple(int(v) for v in shape_arr.tolist())

    return payload


def is_packed_mask(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    if PACKED_SHAPE_KEY not in entry:
        return False
    return PACKED_MASK_KEY in entry or PACKED_MASK_B64_KEY in entry


def unpack_binary_mask(entry: object) -> np.ndarray:
    if not is_packed_mask(entry):
        array = np.asarray(entry)
        if array.dtype != np.bool_:
            array = array.astype(np.bool_)
        return array

    payload = dict(entry)
    shape = payload[PACKED_SHAPE_KEY]
    if isinstance(shape, np.ndarray):
        shape = tuple(int(v) for v in shape.tolist())
    elif isinstance(shape, list):
        shape = tuple(int(v) for v in shape)
    elif isinstance(shape, tuple):
        shape = tuple(int(v) for v in shape)
    else:
        shape = (int(shape),)

    total = int(np.prod(shape))

    if PACKED_MASK_KEY in payload:
        packed_arr = np.asarray(payload[PACKED_MASK_KEY], dtype=np.uint8)
    else:
        packed_bytes = base64.b64decode(payload[PACKED_MASK_B64_KEY])
        packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8)

    unpacked = np.unpackbits(packed_arr, count=total)
    return unpacked.reshape(shape).astype(np.bool_)


def numeric_frame_sort_key(fname: str) -> Tuple[float, str]:
    """Ensure filenames sort by embedded integer before lexical fallback."""
    stem = Path(fname).stem
    match = re.search(r"\d+", stem)
    if match:
        try:
            return float(int(match.group())), fname
        except ValueError:
            pass
    return float("inf"), fname


def list_to_csv(values: Iterable[object] | None) -> str:
    if not values:
        return ""
    return ",".join(str(v) for v in values)


def parse_levels(levels: Union[str, Sequence[int], None]) -> List[int]:
    """Parse level input (string or iterable) into a list of ints."""
    if levels is None:
        return []
    if isinstance(levels, (list, tuple, set)):
        return [int(v) for v in levels]
    return [int(x) for x in str(levels).split(',') if str(x).strip()]


def parse_range(range_str: Union[str, Sequence[int]]) -> Tuple[int, int, int]:
    """Parse ``start:end:step`` strings or 3-element iterables."""
    if isinstance(range_str, (list, tuple)):
        if len(range_str) != 3:
            raise ValueError(f"Range iterable must have 3 elements, got {range_str!r}")
        start, end, step = range_str
    else:
        parts = str(range_str).split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid range spec: {range_str!r}")
        start, end, step = parts
    start = int(start) if start != '' else 0
    end = int(end) if end != '' else -1
    step = int(step) if step != '' else 1
    if step <= 0:
        raise ValueError('step must be positive')
    return start, end, step


def bbox_from_mask_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_xyxy_to_xywh(bounds: Sequence[int]) -> List[int]:
    x1, y1, x2, y2 = bounds
    return [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]


def bbox_xywh_to_xyxy(bounds: Sequence[int]) -> List[int]:
    x, y, w, h = bounds
    return [int(x), int(y), int(x + w), int(y + h)]


def build_subset_video(
    frames_dir: str,
    selected: Sequence[str],
    selected_indices: Sequence[int],
    out_root: str,
    folder_name: str = "selected_frames",
) -> Tuple[str, dict[int, str]]:
    """Symlink/copy selected frames into a compact folder for later stages."""

    subset_dir = ensure_dir(Path(out_root) / folder_name)
    index_to_subset: dict[int, str] = {}
    for local_idx, (abs_idx, fname) in enumerate(zip(selected_indices, selected)):
        src = Path(frames_dir) / fname
        dst_name = f"{local_idx:06d}.jpg"
        dst = Path(subset_dir) / dst_name
        index_to_subset[int(abs_idx)] = dst_name
        if dst.exists():
            # Preserve existing symlink/copy when it already matches the source.
            try:
                if dst.is_symlink() and dst.resolve() == src.resolve():
                    continue
            except FileNotFoundError:
                pass
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            from shutil import copy2

            copy2(src, dst)
    return subset_dir, index_to_subset


def setup_logging(
    *,
    explicit_level: Optional[int] = None,
    env_var: str = "MY3DIS_LOG_LEVEL",
    logger_names_to_quiet: Optional[Sequence[str]] = None,
) -> int:
    """Configure root logging once and return the effective level."""

    if explicit_level is None:
        level_name = os.environ.get(env_var, "INFO").upper()
        explicit_level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=explicit_level, format="%(message)s")
    root_logger.setLevel(explicit_level)

    if logger_names_to_quiet:
        for name in logger_names_to_quiet:
            logging.getLogger(name).setLevel(logging.WARNING)

    return explicit_level
