"""
Command-line interface for the Semantic-SAM progressive refinement pipeline.

This is the spiritual successor to ``progressive_refinement.py``.  It keeps the
original ergonomics (scene-based batches or single-image execution) while
delegating heavy lifting to :mod:`my3dis.semantic_refinement`.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence

from .common_utils import list_to_csv, parse_levels, parse_range, setup_logging
from .semantic_refinement import (
    DEFAULT_SEMANTIC_SAM_ROOT,
    build_semantic_sam,
    console,
    create_experiment_folder,
    get_experiment_timestamp,
    progressive_refinement_masks,
    save_original_image_info,
    setup_output_directories,
)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "SCENE_NAME": "scene_00065_00",
    "DATA_ROOT": "/media/public_dataset2/multiscan/",
    "LEVELS": [2, 4, 6],
    "RANGE_STR": "1400:1500:20",
    "MIN_AREA": 200,
    "MAX_MASKS": 2000,
    "MODEL_TYPE": "L",
    "OUTPUT_ROOT": "./exp_outputs/semantic_refinement",
}

DEFAULT_CHECKPOINT = Path(DEFAULT_SEMANTIC_SAM_ROOT or ".") / "checkpoints" / "swinl_only_sam_many2many.pth"


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic-SAM Progressive Refinement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--scene", default=DEFAULT_CONFIG["SCENE_NAME"], help="Scene name")
    parser.add_argument("--data-root", default=DEFAULT_CONFIG["DATA_ROOT"], help="Dataset root directory")
    parser.add_argument("--image-path", help="Single image path (overrides scene-based processing)")

    parser.add_argument(
        "--levels",
        default=",".join(map(str, DEFAULT_CONFIG["LEVELS"])),
        help="Processing levels (comma-separated)",
    )
    parser.add_argument("--frames", default=DEFAULT_CONFIG["RANGE_STR"], help="Frame range (start:end:step)")
    parser.add_argument("--min-area", type=int, default=DEFAULT_CONFIG["MIN_AREA"], help="Minimum area threshold")
    parser.add_argument("--max-masks", type=int, default=DEFAULT_CONFIG["MAX_MASKS"], help="Maximum masks per level")
    parser.add_argument("--fill-area", type=int, help="Gap fill area threshold (defaults to min-area)")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for filtering duplicate masks",
    )

    parser.add_argument("--output", default=DEFAULT_CONFIG["OUTPUT_ROOT"], help="Output directory")
    parser.add_argument("--experiment-name", default="experiment", help="Experiment name prefix")
    parser.add_argument("--no-timestamp", action="store_true", help="Don't add timestamp to output directory")
    parser.add_argument("--save-viz", action="store_true", help="Save visualization outputs")

    parser.add_argument("--model-type", default=DEFAULT_CONFIG["MODEL_TYPE"], help="Semantic-SAM model type")
    parser.add_argument("--checkpoint", help="Custom checkpoint path")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without executing")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _resolve_image_paths(args: argparse.Namespace) -> List[Path]:
    if args.image_path:
        path = Path(args.image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"指定的圖片路徑不存在: {path}")
        return [path]

    frames_root = Path(args.data_root).expanduser() / args.scene / "outputs" / "color"
    if not frames_root.exists():
        raise FileNotFoundError(f"找不到場景影像資料夾: {frames_root}")

    available = sorted(frames_root.iterdir())
    start, end, step = parse_range(args.frames)
    if end < 0 or end > len(available):
        end = len(available)
    indices = range(start, min(end, len(available)), step)
    selected: List[Path] = []
    for idx in indices:
        if 0 <= idx < len(available):
            selected.append(available[idx])
    if not selected:
        raise ValueError("選出的圖片集合為空，請調整 --frames 或資料集設定")
    return selected


def _load_model(args: argparse.Namespace):
    ckpt_path = Path(args.checkpoint or DEFAULT_CHECKPOINT).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型權重檔案: {ckpt_path}")
    console(f"🔧 正在載入 Semantic-SAM 模型 ({args.model_type})...", important=True)
    model = build_semantic_sam(model_type=args.model_type, ckpt=str(ckpt_path))
    console("✅ 模型載入完成", important=True)
    return model


def _write_manifest(experiment_path: Path, metadata: dict) -> None:
    summary_dir = experiment_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = summary_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _update_index(
    index_csv: Path,
    rows: Iterable[Sequence[object]],
) -> None:
    exists = index_csv.exists()
    with index_csv.open("a", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        if not exists:
            writer.writerow(["image", "image_path", "total_masks", "levels", "output_dir"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(explicit_level=log_level)

    levels = parse_levels(args.levels)
    if len(levels) < 2:
        raise ValueError("progressive refinement 需要至少兩個層級，請調整 --levels 設定")

    if args.dry_run:
        print("=== Dry Run Configuration ===")
        print(f"Scene: {args.scene}")
        print(f"Levels: {levels}")
        print(f"Frames: {args.frames}")
        print(f"Min area: {args.min_area}")
        print(f"Output: {args.output}")
        return

    images = _resolve_image_paths(args)
    console(f"🚀 啟動 Semantic-SAM 影像分割管線，共 {len(images)} 張圖片", important=True)

    model = _load_model(args)

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.no_timestamp:
        experiment_path = output_root / args.experiment_name
        timestamp = get_experiment_timestamp()
        experiment_path.mkdir(parents=True, exist_ok=True)
    else:
        experiment_str, timestamp = create_experiment_folder(
            str(output_root),
            args.experiment_name,
            timestamp=get_experiment_timestamp(),
        )
        experiment_path = Path(experiment_str)

    console(f"📂 實驗資料夾: {experiment_path}")
    setup_output_directories(str(experiment_path))

    manifest = {
        "timestamp": timestamp,
        "scene": args.scene,
        "levels": levels,
        "range": args.frames,
        "min_area": args.min_area,
        "max_masks": args.max_masks,
        "model_type": args.model_type,
        "ckpt": str(Path(args.checkpoint or DEFAULT_CHECKPOINT).expanduser()),
        "save_viz": bool(args.save_viz),
        "total_images": len(images),
        "image_source": args.image_path or str(
            Path(args.data_root).expanduser() / args.scene / "outputs" / "color"
        ),
    }
    _write_manifest(experiment_path, manifest)

    index_csv = experiment_path / "summary" / "index.csv"
    processed_rows = []

    for image_path in images:
        image_stem = image_path.stem
        console(f"\n=== 處理 {image_path} ===", important=True)

        image_output_root = experiment_path / image_stem
        image_output_root.mkdir(parents=True, exist_ok=True)
        image_output_dirs = setup_output_directories(str(image_output_root))

        save_original_image_info(str(image_path), image_output_dirs)

        refinement_results = progressive_refinement_masks(
            model,
            str(image_path),
            level_sequence=levels,
            output_dirs=image_output_dirs,
            min_area=args.min_area,
            max_masks_per_level=args.max_masks,
            save_viz=args.save_viz,
            fill_area=args.fill_area,
            similarity_threshold=args.similarity_threshold,
        )

        total_masks = 0
        metrics = {"levels": {}, "total_masks": 0}
        for level, data in refinement_results["levels"].items():
            mask_count = data.get("mask_count", 0)
            metrics["levels"][str(level)] = {"mask_count": mask_count}
            total_masks += mask_count
            console(f"Level {level}: {mask_count:4d} 個 mask")
        metrics["total_masks"] = total_masks

        metrics_path = image_output_root / "summary" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        processed_rows.append(
            [image_stem, str(image_path), total_masks, list_to_csv(levels), str(image_output_root)]
        )
        console(f"🎉 {image_stem} 總計生成 {total_masks} 個 mask", important=True)

    if processed_rows:
        _update_index(index_csv, processed_rows)
        console("\n✅ 實驗完成！所有結果已寫入 summary/index.csv", important=True)


if __name__ == "__main__":
    main()
