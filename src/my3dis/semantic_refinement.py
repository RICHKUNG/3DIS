"""
Semantic-SAM progressive refinement core logic.

This module hosts the reusable algorithmic pieces that used to live inside
``progressive_refinement.py`` so other components (CLI, adapters, trackers)
can import a single, dependency-light module.

Author: Rich Kung
Updated: 2025-01-XX
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from .common_utils import parse_levels, parse_range

# ---------------------------------------------------------------------------
# Semantic-SAM integration helpers
# ---------------------------------------------------------------------------

DEFAULT_SEMANTIC_SAM_ROOT = os.environ.get(
    "SEMANTIC_SAM_ROOT",
    os.path.expanduser("~/repos/Semantic-SAM"),
)

if DEFAULT_SEMANTIC_SAM_ROOT and DEFAULT_SEMANTIC_SAM_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_SEMANTIC_SAM_ROOT)

_semantic_sam_import_error: Optional[BaseException] = None
try:  # pragma: no cover - heavy dependency, optional for tests
    from semantic_sam import (  # type: ignore
        build_semantic_sam,
        prepare_image,
        plot_results,
        SemanticSamAutomaticMaskGenerator,
    )
except Exception as exc:  # pragma: no cover - fallback for unit tests
    _semantic_sam_import_error = exc

if _semantic_sam_import_error is not None:
    _missing_dependency_error = _semantic_sam_import_error

    def _missing_dependency(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(
            "Semantic-SAM dependencies not available. "
            "Ensure SEMANTIC_SAM_ROOT is configured correctly."
        ) from _missing_dependency_error

    build_semantic_sam = _missing_dependency  # type: ignore
    prepare_image = _missing_dependency  # type: ignore
    plot_results = _missing_dependency  # type: ignore

    class SemanticSamAutomaticMaskGenerator:  # type: ignore
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _missing_dependency()

    def instance_map_to_anns(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore
        _missing_dependency()
else:
    try:
        from auto_generation_inference import instance_map_to_anns  # type: ignore
    except Exception as exc:  # pragma: no cover - tolerate import-time side effects
        _auto_generation_import_error = exc

        def instance_map_to_anns(instance_map: np.ndarray) -> List[Dict[str, Any]]:  # type: ignore
            """Fallback converter when Semantic-SAM helper cannot be imported."""
            unique_ids = np.unique(instance_map)
            unique_ids = unique_ids[unique_ids != 0]
            annotations: List[Dict[str, Any]] = []
            for instance_id in unique_ids:
                mask = instance_map == instance_id
                annotations.append(
                    {
                        'segmentation': mask,
                        'area': int(mask.sum()),
                    }
                )
            if os.environ.get("MY3DIS_PROGRESSIVE_VERBOSE", "").strip() == "1":
                print(
                    "⚠️ 使用內建 instance_map_to_anns 回退，"
                    f"原因：auto_generation_inference 匯入失敗 ({_auto_generation_import_error})",
                    file=sys.stderr,
                )
            return annotations


VERBOSE = os.environ.get("MY3DIS_PROGRESSIVE_VERBOSE", "").strip() == "1"


# ---------------------------------------------------------------------------
# Utility helpers (console, timers, filesystem)
# ---------------------------------------------------------------------------

def console(message: str, *, important: bool = False) -> None:
    """Gate stdout messages behind a verbosity flag."""
    if important or VERBOSE:
        print(message)


def timer_decorator(func):
    """Decorator that prints elapsed time for long-running helpers."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        func_name = func.__name__.replace('_', ' ').title()
        if elapsed < 60:
            print(f"⏱️  {func_name} 完成，耗時: {elapsed:.2f} 秒")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"⏱️  {func_name} 完成，耗時: {minutes}分{seconds:.1f}秒")
        return result

    return wrapper


def log_step(step_name: str, start_time: Optional[float] = None) -> None:
    """Emit user-friendly progress messages."""
    if start_time is None:
        print(f"🔄 開始: {step_name}")
        return

    duration = time.time() - start_time
    if duration < 60:
        print(f"✅ {step_name} - 耗時: {duration:.2f}秒")
    else:
        minutes = int(duration // 60)
        seconds = duration % 60
        print(f"✅ {step_name} - 耗時: {minutes}分{seconds:.1f}秒")


def get_experiment_timestamp() -> str:
    """Generate a filesystem-friendly timestamp string."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_folder(
    base_path: str | os.PathLike[str],
    experiment_name: str,
    timestamp: Optional[str] = None,
) -> Tuple[str, str]:
    """Create an experiment directory, optionally appending a timestamp."""
    if timestamp is None:
        timestamp = get_experiment_timestamp()

    experiment_folder = f"{experiment_name}_{timestamp}"
    full_path = os.path.join(base_path, experiment_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path, timestamp


def setup_output_directories(experiment_path: str) -> Dict[str, str]:
    """
    Create the nested directory layout used by the progressive refinement run.

    experiment_path/
      original/
      levels/
      relations/
      summary/
      viz/
    """
    dirs = [
        "original",
        "levels",
        "relations",
        "summary",
        "viz",
    ]
    created_dirs: Dict[str, str] = {}
    for folder in dirs:
        folder_path = os.path.join(experiment_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        created_dirs[folder] = folder_path
    return created_dirs


@timer_decorator
def save_original_image_info(
    image_path: str,
    output_dirs: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """Copy the source image into the `original` subdirectory."""
    try:
        original_dir = output_dirs["original"]
        import shutil

        original_filename = os.path.basename(image_path)
        copied_path = os.path.join(original_dir, f"original_{original_filename}")
        shutil.copy2(image_path, copied_path)
        return {"original_path": image_path, "copied_path": copied_path}
    except Exception as exc:  # pragma: no cover - defensive
        console(f"❌ 儲存原始圖片時發生錯誤: {exc}", important=True)
        return None


def prepare_image_from_pil(pil_img: Image.Image) -> Tuple[np.ndarray, Any]:
    """
    Mirror ``semantic_sam.prepare_image`` but accept in-memory PIL objects.

    Returns:
        (numpy_image, torch_tensor_on_cuda)
    """
    try:
        import torch
        from torchvision import transforms
    except Exception:
        torch = None
        transforms = None
    interpolation_mode = None
    if transforms is not None:
        interpolation_mode = getattr(transforms, "InterpolationMode", None)

    if transforms is None or torch is None:
        # Fallback: persist to disk then delegate to Semantic-SAM util.
        tmp_path = os.path.join("/tmp", f"masked_{int(time.time() * 1000)}.png")
        try:
            pil_img.save(tmp_path)
            return prepare_image(image_pth=tmp_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    if interpolation_mode is not None:
        resize_interpolation = interpolation_mode.BICUBIC
    else:
        resize_interpolation = Image.BICUBIC
    transform1 = transforms.Resize(640, interpolation=resize_interpolation)
    image_resized = transform1(pil_img)
    image_ori = np.asarray(image_resized)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    return image_ori, images


def create_masked_image(
    original_image: Image.Image,
    mask_data: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Generate a masked PIL image where the uncovered region is filled."""
    try:
        if not isinstance(original_image, Image.Image):
            raise ValueError("輸入必須是 PIL Image")

        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")

        img_array = np.array(original_image)
        result_array = img_array.copy()

        if isinstance(mask_data, np.ndarray):
            if mask_data.dtype == bool:
                mask_binary = mask_data
            elif mask_data.dtype == np.uint8:
                mask_binary = mask_data > 0
            else:
                mask_binary = mask_data > 0.5

            if len(mask_binary.shape) > 2:
                mask_binary = mask_binary.squeeze()
                if len(mask_binary.shape) > 2:
                    mask_binary = mask_binary[:, :, 0]

            if mask_binary.shape != img_array.shape[:2]:
                mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8), "L")
                mask_pil = mask_pil.resize(original_image.size, Image.NEAREST)
                mask_binary = np.array(mask_pil) > 127

            mask_inverse = ~mask_binary
            for channel in range(3):
                result_array[mask_inverse, channel] = background_color[channel]

            return Image.fromarray(result_array, "RGB")

        if VERBOSE:
            console(f"警告：mask 數據格式不正確: {type(mask_data)}")
        return original_image

    except Exception as exc:  # pragma: no cover - defensive
        if VERBOSE:
            console(f"創建 masked 圖片時發生錯誤: {exc}")
        return original_image


def bbox_from_mask(seg: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return XYXY bounds for a boolean segmentation array."""
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _normalize_mask_shape(mask: np.ndarray, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Ensure a mask is boolean and matches the requested shape.

    Args:
        mask: Raw segmentation array from Semantic-SAM.
        target_shape: Desired (H, W); if None the original shape is kept.
    """
    mask_array = np.asarray(mask)
    if mask_array.dtype != np.bool_:
        mask_array = mask_array > 0

    if target_shape is None or mask_array.shape == target_shape:
        return mask_array

    mask_img = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")
    mask_img = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return np.asarray(mask_img) > 127


def instance_map_to_color_image(instance_map: np.ndarray) -> Image.Image:
    """Assign deterministic colours to each instance ID for quick inspection."""
    h, w = instance_map.shape
    coloured = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(instance_map)
    rng = np.random.default_rng(0)
    colour_map = {0: (0, 0, 0)}
    for uid in ids:
        if uid == 0:
            continue
        colour_map[int(uid)] = tuple(rng.integers(0, 255, size=3).tolist())
    for uid, colour in colour_map.items():
        coloured[instance_map == uid] = colour
    return Image.fromarray(coloured, "RGB")


# ---------------------------------------------------------------------------
# Progressive refinement core algorithm
# ---------------------------------------------------------------------------

def progressive_refinement_masks(
    semantic_sam,
    image_path: str,
    level_sequence: List[int],
    output_dirs: Dict[str, str],
    *,
    min_area: int = 50,
    max_masks_per_level: int = 200,
    save_viz: bool = False,
    fill_area: Optional[int] = None,
    gap_fill_enabled: bool = True,
    similarity_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Run the multi-level refinement pipeline and persist structured outputs.
    """
    _ = gap_fill_enabled  # kept for signature compatibility, not used here
    _ = similarity_threshold
    if fill_area is None:
        fill_area = min_area

    main_start = time.time()
    log_step(f"漸進式細化處理 (Levels: {level_sequence})")

    if not level_sequence or len(level_sequence) < 2:
        raise ValueError("level_sequence 必須包含至少 2 個 level")
    if level_sequence != sorted(level_sequence):
        raise ValueError("level_sequence 必須是遞增序列")

    original_image_pil, input_image = prepare_image(image_pth=image_path)

    if not isinstance(original_image_pil, Image.Image):
        if isinstance(original_image_pil, np.ndarray):
            arr = np.asarray(original_image_pil)
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                pil_mode = "RGBA" if arr.shape[2] == 4 else "RGB"
                original_image_pil = Image.fromarray(arr, pil_mode).convert("RGB")
            elif arr.ndim == 2:
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                original_image_pil = Image.fromarray(arr, "L").convert("RGB")
            else:
                raise TypeError(f"不支援的圖片格式: {type(original_image_pil)}")
        else:
            raise TypeError(f"不支援的圖片格式: {type(original_image_pil)}")

    if original_image_pil.mode != "RGB":
        original_image_pil = original_image_pil.convert("RGB")

    console(
        f"原始圖片格式確認: {type(original_image_pil)}, 模式: {original_image_pil.mode}, 尺寸: {original_image_pil.size}"
    )

    sequence_str = "_".join(map(str, level_sequence))
    parent_child_dir = os.path.join(output_dirs["viz"], f"sequence_{sequence_str}")
    levels_root = os.path.join(output_dirs["levels"], f"sequence_{sequence_str}")
    os.makedirs(levels_root, exist_ok=True)
    if save_viz:
        os.makedirs(parent_child_dir, exist_ok=True)

    refinement_results: Dict[str, Any] = {
        "level_sequence": level_sequence,
        "levels": {},
    }
    tree_relations: Dict[int, List[Dict[str, Any]]] = {}
    id_to_node: Dict[int, Dict[str, Any]] = {}
    mask_id_counter = 1

    first_level = level_sequence[0]
    console(f"\n🎯 Level {first_level} (原圖)")

    mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[first_level])
    first_level_masks = mask_generator.generate(input_image)

    height, width = original_image_pil.size[1], original_image_pil.size[0]
    instance_map = np.zeros((height, width), dtype=np.int32)
    for idx, mask in enumerate(first_level_masks):
        if "segmentation" in mask:
            seg = mask["segmentation"]
            instance_map[(seg) & (instance_map == 0)] = idx + 1
    first_level_masks = instance_map_to_anns(instance_map)

    tree_relations[first_level] = []
    for mask in first_level_masks:
        node = {"id": mask_id_counter, "parent": None, "children": []}
        tree_relations[first_level].append(node)
        id_to_node[mask_id_counter] = node
        mask["unique_id"] = mask_id_counter
        mask_id_counter += 1

    first_level_masks = [m for m in first_level_masks if m.get("area", 0) >= min_area]
    if len(first_level_masks) > max_masks_per_level:
        first_level_masks = first_level_masks[:max_masks_per_level]

    console(f"✨ Level {first_level}: {len(first_level_masks)} 個有效 mask")

    level_dir = os.path.join(levels_root, f"level_{first_level}")
    os.makedirs(level_dir, exist_ok=True)
    instance_map_unique = np.zeros((height, width), dtype=np.int32)
    for m in first_level_masks:
        if "segmentation" in m:
            seg = m["segmentation"]
            instance_map_unique[(seg) & (instance_map_unique == 0)] = m["unique_id"]
    np.save(os.path.join(level_dir, "instance_map.npy"), instance_map_unique)
    try:
        instance_img = instance_map_to_color_image(instance_map_unique)
        instance_img.save(os.path.join(level_dir, "colored_instances.png"))
    except Exception:
        pass

    try:
        plot_results(first_level_masks, original_image_pil, save_path=level_dir)
    except AttributeError as err:
        console(
            f"⚠️ plot_results 失敗，略過可視化輸出：{err}",
            important=VERBOSE,
        )
    except Exception as err:  # pragma: no cover - best effort visualisation
        console(
            f"⚠️ plot_results 發生未預期錯誤，略過可視化輸出：{err}",
            important=VERBOSE,
        )

    refinement_results["levels"][first_level] = {
        "masks": first_level_masks,
        "mask_count": len(first_level_masks),
    }

    current_masks = first_level_masks
    current_level = first_level

    level_pbar = tqdm(level_sequence[1:], desc="漸進細化", bar_format="{l_bar}{bar:20}{r_bar}")

    for next_level in level_pbar:
        level_start = time.time()
        level_pbar.set_description(f"Level {current_level}→{next_level}")

        next_level_all_masks: List[Dict[str, Any]] = []
        next_level_parent_info: List[Dict[str, Any]] = []

        parent_child_level_dir = os.path.join(parent_child_dir, f"L{current_level}_to_L{next_level}")
        if save_viz:
            os.makedirs(parent_child_level_dir, exist_ok=True)

        parent_pbar = tqdm(
            enumerate(current_masks),
            total=len(current_masks),
            desc="處理 parent masks",
            leave=False,
            bar_format="{l_bar}{bar:15}{r_bar}{postfix}",
        )

        successful_parents = 0

        for parent_idx, parent_mask in parent_pbar:
            try:
                if "segmentation" not in parent_mask:
                    next_level_parent_info.append(
                        {"parent_id": parent_idx, "child_count": 0, "error": "no_segmentation"}
                    )
                    continue

                parent_seg = parent_mask["segmentation"]
                parent_seg_bool = _normalize_mask_shape(parent_seg, (height, width))

                masked_image = create_masked_image(
                    original_image_pil, parent_seg, background_color=(0, 0, 0)
                )

                if not isinstance(masked_image, Image.Image):
                    next_level_parent_info.append(
                        {"parent_id": parent_idx, "child_count": 0, "error": "masked_image_format_error"}
                    )
                    continue

                if save_viz:
                    try:
                        parent_uid = current_masks[parent_idx]["unique_id"]
                        parent_dir = os.path.join(parent_child_level_dir, f"P{parent_uid}")
                        os.makedirs(parent_dir, exist_ok=True)
                        masked_img_path = os.path.join(
                            parent_dir, f"parent_L{current_level}_P{parent_uid}.png"
                        )
                        masked_image.save(masked_img_path)
                    except Exception as save_error:
                        if VERBOSE:
                            console(f"儲存 masked 圖片失敗: {save_error}")

                masked_np, masked_tensor = prepare_image_from_pil(masked_image)
                mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam, level=[next_level])
                child_masks = mask_generator.generate(masked_tensor)

                instance_map_child = np.zeros((masked_np.shape[0], masked_np.shape[1]), dtype=np.int32)
                for child_idx, child_mask in enumerate(child_masks):
                    if child_mask.get("segmentation") is not None:
                        seg = child_mask["segmentation"]
                        instance_map_child[(seg) & (instance_map_child == 0)] = child_idx + 1
                child_masks = instance_map_to_anns(instance_map_child)

                valid_children: List[Dict[str, Any]] = []
                for child in child_masks:
                    seg = child.get("segmentation")
                    if seg is None:
                        continue
                    # Clamp the child mask to its parent so progressive refinement never grows mask extent.
                    child_seg_bool = _normalize_mask_shape(seg, parent_seg_bool.shape)
                    intersect_seg = np.logical_and(child_seg_bool, parent_seg_bool)

                    if not intersect_seg.any():
                        continue

                    # Drop children that collapse back to the full parent region.
                    if np.array_equal(intersect_seg, parent_seg_bool):
                        continue

                    trimmed_area = int(intersect_seg.sum())
                    if trimmed_area < min_area:
                        continue

                    child["parent_unique_id"] = current_masks[parent_idx]["unique_id"]
                    child["unique_id"] = mask_id_counter
                    mask_id_counter += 1
                    child["segmentation"] = intersect_seg
                    child["area"] = trimmed_area

                    bbox = bbox_from_mask(intersect_seg)
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        child["bbox"] = [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]
                        child["bbox_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]

                    valid_children.append(child)

                if len(valid_children) > max_masks_per_level:
                    valid_children = valid_children[:max_masks_per_level]

                if save_viz and valid_children:
                    parent_uid = current_masks[parent_idx]["unique_id"]
                    parent_dir = os.path.join(parent_child_level_dir, f"P{parent_uid}")
                    os.makedirs(parent_dir, exist_ok=True)
                    parent_instance_map = np.zeros((height, width), dtype=np.int32)
                    for c in valid_children:
                        if c.get("segmentation") is not None:
                            seg = c["segmentation"]
                            parent_instance_map[(seg) & (parent_instance_map == 0)] = c["unique_id"]
                    np.save(
                        os.path.join(parent_dir, f"children_instance_map.npy"),
                        parent_instance_map,
                    )
                    try:
                        instance_img = instance_map_to_color_image(parent_instance_map)
                        instance_img.save(os.path.join(parent_dir, "children_colored.png"))
                    except Exception:
                        pass

                next_level_all_masks.extend(valid_children)
                next_level_parent_info.append(
                    {"parent_id": parent_idx, "child_count": len(valid_children)}
                )
                if valid_children:
                    successful_parents += 1

            except Exception as err:  # pragma: no cover - defensive
                next_level_parent_info.append(
                    {"parent_id": parent_idx, "child_count": 0, "error": str(err)}
                )

        parent_pbar.close()

        next_level_all_masks = [
            m for m in next_level_all_masks if m.get("area", 0) >= min_area
        ]
        if len(next_level_all_masks) > max_masks_per_level:
            next_level_all_masks = next_level_all_masks[:max_masks_per_level]

        next_level_dir = os.path.join(levels_root, f"level_{next_level}")
        os.makedirs(next_level_dir, exist_ok=True)
        instance_map_unique = np.zeros((height, width), dtype=np.int32)
        for mask in next_level_all_masks:
            if mask.get("segmentation") is not None:
                seg = mask["segmentation"]
                instance_map_unique[(seg) & (instance_map_unique == 0)] = mask["unique_id"]
        np.save(os.path.join(next_level_dir, "instance_map.npy"), instance_map_unique)
        try:
            instance_img = instance_map_to_color_image(instance_map_unique)
            instance_img.save(os.path.join(next_level_dir, "colored_instances.png"))
        except Exception:
            pass

        tree_relations[next_level] = []
        for mask in next_level_all_masks:
            node = {
                "id": mask["unique_id"],
                "parent": mask.get("parent_unique_id"),
                "children": [],
            }
            tree_relations[next_level].append(node)
            id_to_node[node["id"]] = node

        for node in tree_relations[next_level]:
            pid = node["parent"]
            if pid is not None and pid in id_to_node:
                if node["id"] not in id_to_node[pid]["children"]:
                    id_to_node[pid]["children"].append(node["id"])

        with open(os.path.join(next_level_dir, "tree.json"), "w", encoding="utf-8") as f:
            json.dump(tree_relations[next_level], f, ensure_ascii=False, indent=2)

        refinement_results["levels"][next_level] = {
            "masks": next_level_all_masks,
            "mask_count": len(next_level_all_masks),
        }

        current_masks = next_level_all_masks
        current_level = next_level

        level_duration = time.time() - level_start
        console(
            f"✨ Level {next_level}: {len(next_level_all_masks)} 個 mask "
            f"(來自 {successful_parents}/{len(next_level_parent_info)} 個 parent，"
            f"耗時 {level_duration:.1f}秒)",
            important=True,
        )

    level_pbar.close()

    first_level_dir = os.path.join(levels_root, f"level_{first_level}")
    with open(os.path.join(first_level_dir, "tree.json"), "w", encoding="utf-8") as f:
        json.dump(tree_relations[first_level], f, ensure_ascii=False, indent=2)

    relations_dir = output_dirs["relations"]
    with open(os.path.join(relations_dir, "tree.json"), "w", encoding="utf-8") as f:
        json.dump(tree_relations, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(relations_dir, "relations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["id", "parent", "level", "area", "bbox"])
        for level, nodes in tree_relations.items():
            masks = refinement_results["levels"].get(level, {}).get("masks", [])
            id2mask = {m.get("unique_id"): m for m in masks}
            for node in nodes:
                mask = id2mask.get(node["id"])
                area = int(mask.get("area", 0)) if mask else 0
                bbox = None
                if mask and mask.get("segmentation") is not None:
                    bbox = bbox_from_mask(mask["segmentation"])
                writer.writerow(
                    [
                        node["id"],
                        node["parent"],
                        level,
                        area,
                        ("%d,%d,%d,%d" % bbox) if bbox else "",
                    ]
                )

    log_step("漸進細化處理完成", main_start)
    return refinement_results


__all__ = [
    "DEFAULT_SEMANTIC_SAM_ROOT",
    "build_semantic_sam",
    "prepare_image",
    "plot_results",
    "SemanticSamAutomaticMaskGenerator",
    "instance_map_to_anns",
    "console",
    "timer_decorator",
    "log_step",
    "get_experiment_timestamp",
    "create_experiment_folder",
    "setup_output_directories",
    "save_original_image_info",
    "prepare_image_from_pil",
    "create_masked_image",
    "bbox_from_mask",
    "instance_map_to_color_image",
    "progressive_refinement_masks",
    "parse_levels",
    "parse_range",
]
