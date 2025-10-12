```python
"""
候選過濾器
重新命名自 filter_candidates.py，提供更清晰的模組名稱

Author: Rich Kung
Updated: 2025-01-XX
"""

import argparse
import os
import json
from pathlib import Path

from my3dis.progressive_refinement import PipelineConfig, load_json, save_json


def filter_level(level_root, min_area, stability_threshold, verbose):
    """讀取 raw archive、依條件重新過濾、輸出 filtered.json"""
    raw_archive = level_root / "raw"
    filtered_path = level_root / "filtered" / "filtered.json"

    if not raw_archive.exists():
        print(f"跳過 {level_root.name}：找不到原始資料夾")  # type: ignore
        return

    # 讀取原始檔案列表
    frame_files = sorted(raw_archive.glob("frame_*.json"))

    # 儲存過濾後的結果
    filtered_results = []

    for frame_file in frame_files:
        # 讀取每個 frame 的原始資料
        frame_data = load_json(frame_file)

        # 濾除小面積區域
        if frame_data["area"] < min_area:
            if verbose:
                print(f"移除 {frame_file.name}：面積 {frame_data['area']} 小於閾值 {min_area}")
            continue

        # 濾除不穩定區域
        if frame_data["stability"] < stability_threshold:
            if verbose:
                print(f"移除 {frame_file.name}：穩定度 {frame_data['stability']} 小於閾值 {stability_threshold}")
            continue

        # 通過過濾，加入結果
        filtered_results.append(frame_data)

    # 將過濾後的結果寫入 filtered.json
    save_json(filtered_path, filtered_results)

    print(f"層級 {level_root.name} 過濾完成，保留 {len(filtered_results)} 幀")


def run_filtering(root, levels, min_area, stability_threshold, update_manifest, quiet):
    """遍歷各層級執行 filter_level，並視需求更新 manifest"""
    for level in levels:
        level_root = root / f"level_{level}"

        if not level_root.exists():
            print(f"跳過層級 {level}：找不到資料夾")
            continue

        filter_level(level_root, min_area, stability_threshold, not quiet)

        # 更新 manifest 檔案
        if update_manifest:
            manifest_path = level_root / "manifest.json"
            if manifest_path.exists():
                os.remove(manifest_path)
                print(f"更新 manifest：{manifest_path}")

    print("所有層級過濾完成")


def main():
    parser = argparse.ArgumentParser(description="重新過濾候選區域")
    parser.add_argument(
        "--candidates-root",
        type=Path,
        required=True,
        help="候選區域根資料夾",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="2,4,6",
        help="要過濾的層級（逗號分隔）",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=400,
        help="最小面積閾值",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.85,
        help="穩定度閾值",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="過濾後更新 manifest",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安靜模式，僅顯示錯誤",
    )

    args = parser.parse_args()

    levels = [int(level) for level in args.levels.split(",")]
    run_filtering(
        args.candidates_root,
        levels,
        args.min_area,
        args.stability_threshold,
        args.update_manifest,
        args.quiet,
    )


if __name__ == "__main__":
    main()
```