"""
Tracking-only stage: read filtered candidates saved by generate_candidates.py
and run SAM2 masklet propagation. Designed to run in a SAM2-capable env.
"""

import os
import sys
import json
import argparse
import time
import logging
import base64
import re
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np

DEFAULT_SAM2_ROOT = "/media/Pluto/richkung/SAM2"
if DEFAULT_SAM2_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor

import torch

LOGGER = logging.getLogger("my3dis.track_from_candidates")


# 簡單的終端進度列，避免依賴 tqdm 以保持可控
class _ProgressPrinter:
    def __init__(self, total: int) -> None:
        self.total = max(0, int(total))
        self._last_len = 0
        self._closed = False

    def update(self, index: int, abs_frame: Optional[int] = None) -> None:
        if self._closed or self.total == 0:
            return
        current = min(index + 1, self.total)
        pct = (current / self.total) * 100.0
        bars = int((current / self.total) * 20)
        bar = f"[{'#' * bars}{'.' * (20 - bars)}]"
        frame_info = f" → frame {abs_frame:05d}" if abs_frame is not None else ""
        msg = f"SAM2 tracking {bar} {current}/{self.total} SSAM frames ({pct:5.1f}%)" + frame_info
        pad = ' ' * max(0, self._last_len - len(msg))
        sys.stdout.write('\r' + msg + pad)
        sys.stdout.flush()
        self._last_len = len(msg)

    def close(self) -> None:
        if self._closed:
            return
        if self._last_len:
            sys.stdout.write('\n')
            sys.stdout.flush()
        self._closed = True


# Masks are stored packed-bytes to keep peak RAM usage manageable while
# still allowing downstream code to transparently request dense arrays.
PACKED_MASK_KEY = "packed_bits"
PACKED_MASK_B64_KEY = "packed_bits_b64"
PACKED_SHAPE_KEY = "shape"


def pack_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    bool_mask = np.asarray(mask, dtype=np.bool_, order="C")
    packed = np.packbits(bool_mask.ravel())
    return {
        PACKED_MASK_KEY: packed,
        PACKED_SHAPE_KEY: tuple(int(dim) for dim in bool_mask.shape),
    }


def is_packed_mask(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if PACKED_SHAPE_KEY not in entry:
        return False
    return PACKED_MASK_KEY in entry or PACKED_MASK_B64_KEY in entry


def unpack_binary_mask(entry: Any) -> np.ndarray:
    if is_packed_mask(entry):
        shape = entry[PACKED_SHAPE_KEY]
        if isinstance(shape, np.ndarray):
            shape = tuple(int(v) for v in shape.tolist())
        elif isinstance(shape, list):
            shape = tuple(int(v) for v in shape)
        total = int(np.prod(shape))
        if PACKED_MASK_B64_KEY in entry:
            packed_bytes = base64.b64decode(entry[PACKED_MASK_B64_KEY])
            packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8)
        else:
            packed_arr = np.asarray(entry[PACKED_MASK_KEY], dtype=np.uint8)
        unpacked = np.unpackbits(packed_arr, count=total)
        return unpacked.reshape(shape).astype(np.bool_)
    array = np.asarray(entry)
    if array.dtype != np.bool_:
        array = array.astype(np.bool_)
    return array


def numeric_frame_sort_key(fname: str) -> Tuple[float, str]:
    """Ensure frame iteration respects numeric order in mixed-padded names."""
    stem, _ = os.path.splitext(fname)
    match = re.search(r'\d+', stem)
    if match:
        try:
            return float(int(match.group())), fname
        except ValueError:
            pass
    return float('inf'), fname


DEFAULT_SAM2_CFG = os.path.join(
    DEFAULT_SAM2_ROOT,
    "sam2",
    "configs",
    "sam2.1",
    "sam2.1_hiera_l.yaml",
)
DEFAULT_SAM2_CKPT = os.path.join(
    DEFAULT_SAM2_ROOT,
    "checkpoints",
    "sam2.1_hiera_large.pt",
)

def resolve_sam2_config_path(config_arg: str) -> str:
    cfg_path = os.path.expanduser(config_arg)
    if os.path.isfile(cfg_path):
        base = os.path.join(DEFAULT_SAM2_ROOT, 'sam2')
        rel = os.path.relpath(cfg_path, base)
        rel = rel.replace(os.sep, '/')
        if rel.endswith('.yaml'):
            rel = rel[:-5]
        return rel
    return config_arg


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration_precise(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1e-3:
        return "0m"
    minutes = seconds / 60.0
    if seconds < 60.0:
        return f"{minutes:.2f}m"
    hours = minutes / 60.0
    if seconds < 3600.0:
        return f"{minutes:.1f}m"
    return f"{hours:.1f}h"


class _TimingSection:
    def __init__(self, aggregator: "TimingAggregator", stage: str) -> None:
        self._aggregator = aggregator
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        duration = time.perf_counter() - self._start
        self._aggregator.add(self._stage, duration)
        return False


class TimingAggregator:
    """Collects and aggregates named timing spans while preserving insertion order."""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}
        self._order: List[str] = []

    def add(self, stage: str, duration: float) -> None:
        stage = str(stage)
        duration = float(duration)
        if stage not in self._totals:
            self._totals[stage] = 0.0
            self._order.append(stage)
        self._totals[stage] += max(0.0, duration)

    def track(self, stage: str) -> _TimingSection:
        return _TimingSection(self, stage)

    def total(self, stage: str) -> float:
        return self._totals.get(stage, 0.0)

    def total_prefix(self, prefix: str) -> float:
        return sum(self._totals[name] for name in self._totals if name.startswith(prefix))

    def total_all(self) -> float:
        return sum(self._totals.values())

    def items(self) -> List[Tuple[str, float]]:
        return [(stage, self._totals[stage]) for stage in self._order]

    def merge(self, other: "TimingAggregator") -> None:
        for stage, duration in other.items():
            self.add(stage, duration)

    def format_breakdown(self) -> str:
        if not self._order:
            return "n/a"
        parts = [f"{name}={format_duration_precise(duration)}" for name, duration in self.items()]
        return ", ".join(parts)


def bbox_transform_xywh_to_xyxy(bboxes: List[List[int]]) -> List[List[int]]:
    for bbox in bboxes:
        x, y, w, h = bbox
        bbox[0] = x
        bbox[1] = y
        bbox[2] = x + w
        bbox[3] = y + h
    return bboxes


def bbox_scalar_fit(bboxes: List[List[int]], scalar_x: float, scalar_y: float) -> List[List[int]]:
    for bbox in bboxes:
        bbox[0] = int(bbox[0] * scalar_x)
        bbox[1] = int(bbox[1] * scalar_y)
        bbox[2] = int(bbox[2] * scalar_x)
        bbox[3] = int(bbox[3] * scalar_y)
    return bboxes


def compute_iou(mask1: Any, mask2: Any) -> float:
    arr1 = unpack_binary_mask(mask1)
    arr2 = unpack_binary_mask(mask2)
    if arr1.shape == arr2.shape:
        inter = np.logical_and(arr1, arr2).sum()
        union = np.logical_or(arr1, arr2).sum()
        return float(inter) / float(union) if union else 0.0
    m1 = torch.from_numpy(arr1.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2 = torch.from_numpy(arr2.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m2r = torch.nn.functional.interpolate(m2, size=m1.shape[-2:], mode='bilinear', align_corners=False)
    m2r = (m2r > 0.5).squeeze()
    m1 = m1.squeeze()
    inter = torch.logical_and(m1, m2r).sum().item()
    union = torch.logical_or(m1, m2r).sum().item()
    return float(inter) / float(union) if union else 0.0


def build_subset_video(
    frames_dir: str,
    selected: List[str],
    selected_indices: List[int],
    out_root: str,
    folder_name: str = "selected_frames",
) -> Tuple[str, Dict[int, str]]:
    subset_dir = os.path.join(out_root, folder_name)
    os.makedirs(subset_dir, exist_ok=True)
    index_to_subset: Dict[int, str] = {}
    for i, (abs_idx, fname) in enumerate(zip(selected_indices, selected)):
        src = os.path.join(frames_dir, fname)
        dst_name = f"{i:06d}.jpg"
        dst = os.path.join(subset_dir, dst_name)
        index_to_subset[abs_idx] = dst_name
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                from shutil import copy2
                copy2(src, dst)
    return subset_dir, index_to_subset


def configure_logging(explicit_level: Optional[int] = None) -> int:
    if explicit_level is None:
        log_level_name = os.environ.get("MY3DIS_LOG_LEVEL", "INFO").upper()
        explicit_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=explicit_level, format="%(message)s")
    root_logger.setLevel(explicit_level)
    LOGGER.setLevel(explicit_level)
    return explicit_level


def load_filtered_candidates(level_root: str) -> Tuple[List[List[Dict[str, Any]]], List[int]]:
    """加載篩選後的候選項，返回候選項列表和對應的幀索引"""
    filt_dir = os.path.join(level_root, 'filtered')
    with open(os.path.join(filt_dir, 'filtered.json'), 'r') as f:
        meta = json.load(f)
    frames_meta = meta.get('frames', [])
    per_frame = []
    frame_indices = []
    
    for fm in frames_meta:
        fidx = fm['frame_idx']
        items = fm['items']
        seg_path = os.path.join(filt_dir, f'seg_frame_{fidx:05d}.npy')
        seg_stack = None
        if os.path.exists(seg_path):
            seg_stack = np.load(seg_path)
        lst = []
        for j, it in enumerate(items):
            d = dict(it)
            mask_payload = d.pop('mask', None)
            seg = None
            if mask_payload is not None:
                seg = unpack_binary_mask(mask_payload)
            elif seg_stack is not None and j < seg_stack.shape[0]:
                seg = seg_stack[j]
            if seg is not None:
                seg = np.asarray(seg, dtype=np.bool_)
            d['segmentation'] = seg
            lst.append(d)
        per_frame.append(lst)
        frame_indices.append(fidx)
    
    return per_frame, frame_indices


def sam2_tracking(
    frames_dir: str,
    predictor,
    mask_candidates: List[List[Dict[str, Any]]],
    frame_numbers: List[int],
    iou_threshold: float = 0.6,
    max_propagate: Optional[int] = None,
    use_box_for_small: bool = False,
    use_box_for_all: bool = False,
    small_object_area_threshold: Optional[int] = None,
):
    # 禁用 tqdm 進度條
    import os
    os.environ['TQDM_DISABLE'] = '1'
    
    # 也可以嘗試直接禁用 tqdm
    try:
        import tqdm
        tqdm.tqdm.disable = True
    except ImportError:
        pass
    
    with torch.inference_mode(), torch.autocast("cuda"):
        obj_count = 1
        final_video_segments: Dict[int, Dict[int, Any]] = {}

        # init state once and reuse it across iterations
        state = predictor.init_state(video_path=frames_dir)
        # get scaling
        h0 = mask_candidates[0][0]['segmentation'].shape[0]
        w0 = mask_candidates[0][0]['segmentation'].shape[1]
        sx = state['video_width'] / w0
        sy = state['video_height'] / h0

        local_to_abs = {i: frame_numbers[i] for i in range(len(frame_numbers))}
        total_frames = len(frame_numbers)
        progress = _ProgressPrinter(total_frames)

        if max_propagate is not None:
            try:
                max_propagate = max(0, int(max_propagate))
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Invalid max_propagate=%r supplied; disabling propagation limit",
                    max_propagate,
                )
                max_propagate = None

        try:
            for frame_idx, frame_masks in enumerate(mask_candidates):
                # clear prompts from the previous loop without reloading video frames
                predictor.reset_state(state)
                abs_idx = local_to_abs.get(frame_idx)
                progress.update(frame_idx, abs_idx)
                if abs_idx is None:
                    continue

                # choose objects to add
                to_add = []
                prev = final_video_segments.get(abs_idx, {})
                prev_masks = list(prev.values())
                if prev_masks:
                    for m in frame_masks:
                        seg = m.get('segmentation')
                        if seg is None:
                            to_add.append(m)
                            continue
                        tracked = any(compute_iou(pm, seg) > iou_threshold for pm in prev_masks)
                        if not tracked:
                            to_add.append(m)
                else:
                    to_add = list(frame_masks)

                if len(to_add) == 0:
                    continue

                # Prefer mask prompts for higher fidelity on the annotated frame
                for m in to_add:
                    seg = m.get('segmentation')
                    area_val = m.get('area')
                    if area_val is None:
                        bbox_dims = m.get('bbox')
                        if bbox_dims is not None and len(bbox_dims) == 4:
                            area_val = int(bbox_dims[2]) * int(bbox_dims[3])
                    use_box_prompt = use_box_for_all or seg is None
                    if not use_box_prompt and use_box_for_small and small_object_area_threshold is not None:
                        try:
                            area_int = int(area_val)
                        except (TypeError, ValueError):
                            area_int = None
                        if area_int is not None and area_int <= small_object_area_threshold:
                            use_box_prompt = True

                    if not use_box_prompt and seg is not None:
                        predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=frame_idx,
                            obj_id=obj_count,
                            mask=np.asarray(seg, dtype=bool),
                        )
                    else:
                        bbox = m.get('bbox')
                        if bbox is None:
                            continue
                        xyxy = bbox_transform_xywh_to_xyxy([list(bbox)])[0]
                        xyxy = bbox_scalar_fit([xyxy], sx, sy)[0]
                        _ , _out_ids, _out_logits = predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=frame_idx, obj_id=obj_count, box=xyxy
                        )
                    obj_count += 1

                segs = {}

                def collect(iterator):
                    for out_fidx, out_obj_ids, out_mask_logits in iterator:
                        abs_out_idx = local_to_abs.get(out_fidx)
                        if abs_out_idx is None:
                            continue
                        frame_data = {}
                        for i, out_obj_id in enumerate(out_obj_ids):
                            mask_arr = (out_mask_logits[i] > 0.0).cpu().numpy()
                            frame_data[int(out_obj_id)] = pack_binary_mask(mask_arr)
                        if abs_out_idx not in segs:
                            segs[abs_out_idx] = {}
                        segs[abs_out_idx].update(frame_data)

                # Determine propagation budget in both temporal directions.
                forward_budget = max(0, total_frames - frame_idx - 1)
                backward_budget = max(0, frame_idx)
                if max_propagate is not None:
                    forward_budget = min(forward_budget, max_propagate)
                    backward_budget = min(backward_budget, max_propagate)

                # 使用上下文管理器來禁用 tqdm
                import contextlib
                import io

                # 捕獲並丟棄 tqdm 輸出
                with contextlib.redirect_stderr(io.StringIO()):
                    if forward_budget > 0:
                        forward_kwargs = {
                            'start_frame_idx': frame_idx,
                            'reverse': False,
                        }
                        if max_propagate is not None:
                            forward_kwargs['max_frame_num_to_track'] = forward_budget
                        collect(predictor.propagate_in_video(state, **forward_kwargs))

                    if backward_budget > 0:
                        backward_kwargs = {
                            'start_frame_idx': frame_idx,
                            'reverse': True,
                        }
                        if max_propagate is not None:
                            backward_kwargs['max_frame_num_to_track'] = backward_budget
                        collect(predictor.propagate_in_video(state, **backward_kwargs))

                for abs_out_idx, frame_data in segs.items():
                    if abs_out_idx not in final_video_segments:
                        final_video_segments[abs_out_idx] = {}
                    final_video_segments[abs_out_idx].update(frame_data)
        finally:
            progress.close()

        return final_video_segments


def save_video_segments_npz(segments: Dict[int, Dict[int, Any]], path: str) -> None:
    packed = {
        str(frame_idx): {str(obj_id): mask for obj_id, mask in frame_data.items()}
        for frame_idx, frame_data in segments.items()
    }
    np.savez_compressed(path, data=packed)


def reorganize_segments_by_object(segments: Dict[int, Dict[int, Any]]) -> Dict[int, Dict[int, Any]]:
    """Re-index frame-major predictions into an object-major structure."""
    per_object: Dict[int, Dict[int, Any]] = {}
    for frame_idx, frame_data in segments.items():
        for obj_id_raw, mask in frame_data.items():
            obj_id = int(obj_id_raw)
            frame_id = int(frame_idx)
            if obj_id not in per_object:
                per_object[obj_id] = {}
            per_object[obj_id][frame_id] = mask
    return per_object


def save_object_segments_npz(segments: Dict[int, Dict[int, Any]], path: str) -> None:
    """Persist object-major mask stacks mirroring the frame-major archive."""
    packed = {
        str(obj_id): {str(frame_idx): mask for frame_idx, mask in frames.items()}
        for obj_id, frames in segments.items()
    }
    np.savez_compressed(path, data=packed)


def save_viz_frames(
    viz_dir: str,
    frames_dir: str,
    video_segments: Dict[int, Dict[int, Any]],
    level: int,
    frame_lookup: Dict[int, str] = None,
):
    """Save per-frame overlays of all masks with colored fills and labels."""
    ensure_dir(viz_dir)
    from PIL import Image, ImageDraw
    if frame_lookup is not None:
        frame_items = [(idx, frame_lookup[idx]) for idx in sorted(frame_lookup.keys())]
    else:
        frame_filenames = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))],
            key=numeric_frame_sort_key,
        )
        frame_items = list(enumerate(frame_filenames))
    rng = np.random.default_rng(0)
    color_map: Dict[int, Tuple[int,int,int]] = {}
    for frame_idx, frame_name in frame_items:
        img_path = os.path.join(frames_dir, frame_name)
        base = Image.open(img_path).convert('RGB')
        overlay = Image.new('RGBA', base.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                if int(obj_id) not in color_map:
                    color_map[int(obj_id)] = tuple(rng.integers(50, 255, size=3).tolist())
                color = color_map[int(obj_id)]
                m = np.squeeze(unpack_binary_mask(mask))
                # create colored alpha mask
                alpha = (m.astype(np.uint8)*120)
                rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
                rgba[...,0]=color[0]; rgba[...,1]=color[1]; rgba[...,2]=color[2]; rgba[...,3]=alpha
                overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, 'RGBA'))
                # label
                draw = ImageDraw.Draw(overlay)
                draw.text((8,8), f"L{level}", fill=(255,255,255,255))
        comp = Image.alpha_composite(base.convert('RGBA'), overlay)
        out_path = os.path.join(viz_dir, f"{os.path.splitext(frame_name)[0]}_L{level}.png")
        comp.convert('RGB').save(out_path)


def save_instance_maps(
    viz_dir: str,
    video_segments: Dict[int, Dict[int, Any]],
    level: int,
    frame_lookup: Dict[int, str] = None,
):
    """Save colored instance maps with consistent colors for an object across frames."""
    inst_dir = ensure_dir(os.path.join(viz_dir, 'instance_map'))
    from PIL import Image

    if frame_lookup:
        target_frames = [int(idx) for idx in sorted(frame_lookup.keys())]
    else:
        target_frames = [int(idx) for idx in sorted(video_segments.keys())]

    if not target_frames:
        return

    all_obj_ids = sorted(
        {
            int(obj_id)
            for frame_idx in target_frames
            for obj_id in video_segments.get(frame_idx, {}).keys()
        }
    )
    if not all_obj_ids:
        return

    rng = np.random.default_rng(0)
    color_map = {oid: tuple(rng.integers(50, 255, size=3).tolist()) for oid in all_obj_ids}

    for frame_idx in target_frames:
        objs = video_segments.get(frame_idx)
        if not objs:
            continue
        first_mask = unpack_binary_mask(next(iter(objs.values())))
        H, W = first_mask.shape[-2], first_mask.shape[-1]
        inst_map = np.zeros((H, W), dtype=np.int32)
        for obj_id_key in sorted(objs.keys(), key=lambda x: int(x)):
            obj_id = int(obj_id_key)
            m = np.squeeze(unpack_binary_mask(objs[obj_id_key]))
            inst_map[(m) & (inst_map == 0)] = obj_id
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for obj_id, col in color_map.items():
            rgb[inst_map == obj_id] = col
        Image.fromarray(rgb, 'RGB').save(os.path.join(inst_dir, f"frame_{frame_idx:05d}_L{level}.png"))
        np.save(os.path.join(inst_dir, f"frame_{frame_idx:05d}_L{level}.npy"), inst_map)


def save_comparison_proposals(
    viz_dir: str,
    base_frames_dir: str,
    filtered_per_frame: List[List[Dict[str, Any]]],
    video_segments: Dict[int, Dict[int, Any]],
    level: int,
    frame_numbers: List[int] = None,
    frames_to_save: List[int] = None,
):
    """Save side-by-side comparisons using instance maps (no base image).
    Only render frames that have SSAM processing.

    Left: SemanticSAM instance map
    Right: SAM2 instance map
    """
    from PIL import Image, ImageDraw

    out_dir = ensure_dir(os.path.join(viz_dir, 'compare'))

    if frame_numbers is None:
        frame_numbers = list(range(len(filtered_per_frame)))
    frame_number_to_local = {fn: idx for idx, fn in enumerate(frame_numbers)}

    # 只渲染有 SSAM 處理的幀（即 frame_numbers 中的幀）
    frames_to_render = sorted(frame_numbers)
    if frames_to_save is not None:
        frames_to_render = [f for f in frames_to_save if f in set(frame_numbers)]

    # 只輸出每 10 個 SSAM 幀的比較圖，避免產生過多檔案
    if frames_to_render:
        frames_to_render = [f for idx, f in enumerate(frames_to_render) if idx % 10 == 0]

    rng = np.random.default_rng(0)
    sam_color_map: Dict[int, Tuple[int, int, int]] = {}

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

    rendered_count = 0
    for f_idx in frames_to_render:
        # 確保這一幀有 SSAM 處理
        local_idx = frame_number_to_local.get(f_idx)
        if local_idx is None or local_idx >= len(filtered_per_frame):
            continue

        # 從有 SSAM 分割的幀獲取尺寸
        H = W = None
        if filtered_per_frame[local_idx]:
            seg0 = filtered_per_frame[local_idx][0].get('segmentation')
            if isinstance(seg0, np.ndarray):
                H, W = seg0.shape[:2]
        
        # 如果無法從 SSAM 獲取尺寸，嘗試從 SAM2 結果獲取
        if H is None or W is None:
            if f_idx in video_segments and len(video_segments[f_idx]) > 0:
                first_mask = unpack_binary_mask(next(iter(video_segments[f_idx].values())))
                H, W = first_mask.shape[-2], first_mask.shape[-1]
        
        if H is None or W is None:
            continue

        # 收集 SSAM masks
        sem_masks = []
        for m in filtered_per_frame[local_idx]:
            seg = m.get('segmentation')
            if isinstance(seg, np.ndarray):
                sem_masks.append(seg.astype(bool))

        # 收集 SAM2 masks
        sam2_masks: Dict[int, np.ndarray] = {}
        if f_idx in video_segments:
            for obj_key, mask in video_segments[f_idx].items():
                obj_id = int(obj_key)
                sam2_masks[obj_id] = np.squeeze(unpack_binary_mask(mask))

        sem_img = build_instance_map_img((H, W), sem_masks)
        if sam2_masks:
            sam2_img = build_sam_instance_map_img((H, W), sam2_masks)
        else:
            sam2_img = build_instance_map_img((H, W), [])

        pad = 10
        canvas = Image.new('RGB', (W * 2 + pad, H), (0, 0, 0))
        canvas.paste(sem_img, (0, 0))
        canvas.paste(sam2_img, (W + pad, 0))

        draw = ImageDraw.Draw(canvas)
        def draw_label(x, y, text):
            tw = 8 * len(text) + 10
            draw.rectangle([x, y, x + tw, y + 22], fill=(0, 0, 0))
            draw.text((x + 5, y + 5), text, fill=(255, 255, 255))
        draw_label(5, 5, f"SemanticSAM Instances L{level}")
        draw_label(W + pad + 5, 5, f"SAM2 Instances L{level}")

        out_path = os.path.join(out_dir, f"frame_{f_idx:05d}_L{level}.png")
        canvas.save(out_path)
        rendered_count += 1

    # 創建代表性圖片
    if rendered_count > 0:
        first_frame = frames_to_render[0]
        rep_src = os.path.join(out_dir, f"frame_{first_frame:05d}_L{level}.png")
        rep_dst = os.path.join(viz_dir, f"compare_L{level}.png")
        if os.path.exists(rep_src):
            from shutil import copy2
            copy2(rep_src, rep_dst)


def run_tracking(
    *,
    data_path: str,
    candidates_root: str,
    output: str,
    levels: Union[str, List[int]] = "2,4,6",
    sam2_cfg: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CFG,
    sam2_ckpt: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CKPT,
    sam2_max_propagate: Optional[int] = None,
    log_level: Optional[int] = None,
    iou_threshold: float = 0.6,
    long_tail_box_prompt: bool = False,
    all_box_prompt: bool = False,
) -> str:
    if not sam2_cfg:
        sam2_cfg = DEFAULT_SAM2_CFG
    if not sam2_ckpt:
        sam2_ckpt = DEFAULT_SAM2_CKPT
    sam2_cfg = os.fspath(sam2_cfg) if isinstance(sam2_cfg, os.PathLike) else sam2_cfg
    sam2_ckpt = os.fspath(sam2_ckpt) if isinstance(sam2_ckpt, os.PathLike) else sam2_ckpt

    configure_logging(log_level)

    overall_start = time.perf_counter()
    if isinstance(levels, str):
        level_list = [int(x) for x in levels.split(',') if x.strip()]
    else:
        level_list = [int(x) for x in levels]

    LOGGER.info("SAM2 tracking started (levels=%s)", ",".join(str(x) for x in level_list))

    with open(os.path.join(candidates_root, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    
    # 讀取相關參數
    selected = manifest.get('selected_frames', [])
    selected_indices = manifest.get('selected_indices')
    ssam_frames = manifest.get('ssam_frames', selected)  # 有 SSAM 分割的幀
    ssam_absolute_indices = manifest.get('ssam_absolute_indices', selected_indices)
    ssam_freq = manifest.get('ssam_freq', 1)
    manifest_max_propagate = manifest.get('sam2_max_propagate')
    
    if selected_indices is None:
        selected_indices = list(range(len(selected)))
    else:
        selected_indices = [int(x) for x in selected_indices]
    
    if ssam_absolute_indices is None:
        ssam_absolute_indices = selected_indices
    else:
        ssam_absolute_indices = [int(x) for x in ssam_absolute_indices]
        
    subset_dir_manifest = manifest.get('subset_dir')

    if sam2_max_propagate is None:
        sam2_max_propagate = manifest_max_propagate
    if sam2_max_propagate is not None:
        try:
            sam2_max_propagate = max(0, int(sam2_max_propagate))
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid sam2_max_propagate=%r; defaulting to unlimited",
                sam2_max_propagate,
            )
            sam2_max_propagate = None

    LOGGER.info(
        "Configuration: ssam_freq=%d, sam2_max_propagate=%s, ssam_frames=%d, iou_threshold=%.2f",
        ssam_freq,
        sam2_max_propagate if sam2_max_propagate is not None else "unlimited",
        len(ssam_frames),
        float(iou_threshold),
    )

    long_tail_area_threshold: Optional[int] = None
    if all_box_prompt:
        LOGGER.info("All mask candidates will be converted to SAM2 box prompts")
        if long_tail_box_prompt:
            LOGGER.info("Long-tail box prompt flag ignored because all-box prompt is active")
    elif long_tail_box_prompt:
        env_area = os.environ.get("MY3DIS_LONG_TAIL_AREA")
        if env_area:
            try:
                long_tail_area_threshold = max(1, int(env_area))
                LOGGER.info(
                    "Environment override: MY3DIS_LONG_TAIL_AREA=%d", long_tail_area_threshold
                )
            except (TypeError, ValueError):
                LOGGER.warning("Invalid MY3DIS_LONG_TAIL_AREA=%r; ignoring", env_area)
        if long_tail_area_threshold is None:
            manifest_min_area = manifest.get('min_area')
            try:
                manifest_min_area = int(manifest_min_area) if manifest_min_area is not None else None
            except (TypeError, ValueError):
                manifest_min_area = None
            if manifest_min_area is not None and manifest_min_area > 0:
                long_tail_area_threshold = max(manifest_min_area * 3, manifest_min_area + 1)
        if long_tail_area_threshold is None:
            long_tail_area_threshold = 1500
        LOGGER.info(
            "Long-tail box prompt enabled: masks with area ≤ %d px will use SAM2 box prompts",
            long_tail_area_threshold,
        )

    try:
        os.chdir(DEFAULT_SAM2_ROOT)
    except Exception:
        pass
    sam2_cfg_resolved = resolve_sam2_config_path(sam2_cfg)
    predictor = build_sam2_video_predictor(sam2_cfg_resolved, sam2_ckpt)

    out_root = ensure_dir(output)
    rebuild_subset = False
    if subset_dir_manifest and os.path.isdir(subset_dir_manifest):
        subset_dir = subset_dir_manifest
        subset_map = manifest.get('subset_map', {})
        subset_map = {int(k): v for k, v in subset_map.items()}
        valid_imgs = [
            f
            for f in os.listdir(subset_dir)
            if os.path.splitext(f)[1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        if len(valid_imgs) == 0:
            rebuild_subset = True
    else:
        rebuild_subset = True

    if rebuild_subset:
        # 使用 SSAM 幀來重建 subset
        subset_dir, subset_map = build_subset_video(
            frames_dir=data_path,
            selected=ssam_frames,
            selected_indices=ssam_absolute_indices,
            out_root=out_root,
        )
        manifest['subset_dir'] = subset_dir
        manifest['subset_map'] = subset_map
        with open(os.path.join(candidates_root, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
    LOGGER.info("Selected frames available at %s", subset_dir)
    
    # 建立幀索引對應關係（針對有 SSAM 分割的幀）
    frame_index_to_name = {idx: name for idx, name in zip(ssam_absolute_indices, ssam_frames)}

    level_stats = []
    overall_timer = TimingAggregator()
    for level in level_list:
        level_start = time.perf_counter()
        level_root = os.path.join(candidates_root, f'level_{level}')
        track_dir = ensure_dir(os.path.join(out_root, f'level_{level}', 'tracking'))
        
        # 加載候選項和對應的幀索引
        per_frame, frame_indices = load_filtered_candidates(level_root)
        
        LOGGER.info(f"Level {level}: Processing {len(per_frame)} frames with SAM2 tracking...")
        level_timer = TimingAggregator()

        with level_timer.track('track.sam2'):
            segs = sam2_tracking(
                subset_dir,
                predictor,
                per_frame,
                frame_numbers=frame_indices,  # 使用實際的幀索引
                iou_threshold=iou_threshold,
                max_propagate=sam2_max_propagate,
                use_box_for_small=(long_tail_box_prompt and not all_box_prompt),
                use_box_for_all=all_box_prompt,
                small_object_area_threshold=long_tail_area_threshold,
            )

        with level_timer.track('persist.video_segments'):
            save_video_segments_npz(segs, os.path.join(track_dir, 'video_segments.npz'))

        with level_timer.track('persist.object_npz'):
            obj_segments = reorganize_segments_by_object(segs)
            save_object_segments_npz(obj_segments, os.path.join(track_dir, 'object_segments.npz'))

        viz_dir = os.path.join(out_root, f'level_{level}', 'viz')
        with level_timer.track('viz.comparison'):
            save_comparison_proposals(
                viz_dir=viz_dir,
                base_frames_dir=data_path,
                filtered_per_frame=per_frame,
                video_segments=segs,
                level=level,
                frame_numbers=frame_indices,  # 使用實際的幀索引
                frames_to_save=None,
            )

        level_total = time.perf_counter() - level_start

        track_time = level_timer.total('track.sam2')
        persist_time = level_timer.total_prefix('persist.')
        viz_time = level_timer.total_prefix('viz.')
        render_time = viz_time

        LOGGER.info(
            "Level %d finished (%d objects / %d frames) → %s",
            level,
            len(obj_segments),
            len(segs),
            level_timer.format_breakdown(),
        )

        level_stats.append(
            (
                level,
                len(obj_segments),
                len(segs),
                track_time,
                persist_time,
                viz_time,
                render_time,
                level_total,
            )
        )

        overall_timer.merge(level_timer)

    if level_stats:
        summary = "; ".join(
            f"L{lvl}: {objs} objects / {frames} frames "
            f"(track={format_duration_precise(track)}, persist={format_duration_precise(persist)}, "
            f"viz={format_duration_precise(viz)}, render={format_duration_precise(render)}, "
            f"total={format_duration_precise(total)})"
            for (
                lvl,
                objs,
                frames,
                track,
                persist,
                viz,
                render,
                total,
            ) in level_stats
        )
        LOGGER.info("Tracking summary → %s", summary)

    if overall_timer.items():
        category_summary = []
        for label, prefix in [
            ("track", 'track.'),
            ("persist", 'persist.'),
            ("viz", 'viz.'),
        ]:
            total = overall_timer.total_prefix(prefix)
            if total > 0:
                category_summary.append(f"{label}={format_duration_precise(total)}")
        if category_summary:
            LOGGER.info("Aggregate timing by stage → %s", ", ".join(category_summary))
        LOGGER.debug("Aggregate timing breakdown → %s", overall_timer.format_breakdown())
    LOGGER.info("Tracking results saved at %s", out_root)
    LOGGER.info(
        "Tracking completed in %s",
        format_duration_precise(time.perf_counter() - overall_start),
    )

    return out_root


def main():
    ap = argparse.ArgumentParser(description="SAM2 tracking from pre-generated candidates")
    ap.add_argument('--data-path', required=True, help='Original frames dir')
    ap.add_argument('--candidates-root', required=True, help='Root containing level_*/filtered')
    ap.add_argument('--sam2-cfg', default=DEFAULT_SAM2_CFG,
                    help='SAM2 config YAML or Hydra path (default: sam2.1_hiera_l)')
    ap.add_argument('--sam2-ckpt', default=DEFAULT_SAM2_CKPT,
                    help='SAM2 checkpoint path (default: sam2.1_hiera_large.pt)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--levels', default='2,4,6')
    ap.add_argument('--sam2-max-propagate', type=int, default=None,
                    help='Limit SAM2 propagation to N frames per direction (default: unlimited)')
    ap.add_argument('--iou-threshold', type=float, default=0.6,
                    help='IoU threshold for deduplicating SAM2 prompts (default: 0.6)')
    ap.add_argument('--long-tail-box-prompt', action='store_true',
                    help='Convert long-tail small objects to SAM2 box prompts')
    ap.add_argument('--all-box-prompt', action='store_true',
                    help='Convert all mask prompts to SAM2 box prompts')
    args = ap.parse_args()

    run_tracking(
        data_path=args.data_path,
        candidates_root=args.candidates_root,
        output=args.output,
        levels=args.levels,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        sam2_max_propagate=args.sam2_max_propagate,
        iou_threshold=args.iou_threshold,
        long_tail_box_prompt=args.long_tail_box_prompt,
        all_box_prompt=args.all_box_prompt,
    )


if __name__ == '__main__':
    main()
