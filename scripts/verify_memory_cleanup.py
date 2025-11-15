#!/usr/bin/env python3
"""
驗證記憶體清理補丁不會造成數據丟失
檢查 family tree 和其他關鍵數據是否正確保存
"""

import json
import os
import sys
from pathlib import Path

def verify_run_output(run_dir: str) -> bool:
    """驗證一個 run 目錄的輸出完整性"""
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return False

    print(f"🔍 驗證輸出: {run_dir}")
    print("=" * 80)

    all_good = True

    # 檢查 relations 目錄
    relations_dir = run_path / "relations"
    if not relations_dir.exists():
        print("⚠️  relations/ 目錄不存在")
        all_good = False
    else:
        print("✅ relations/ 目錄存在")

        # 檢查 provenance trees
        for level in [2, 4, 6]:
            tree_file = relations_dir / f"provenance_tree_L{level}.json"
            if tree_file.exists():
                try:
                    with open(tree_file, 'r') as f:
                        data = json.load(f)

                    required_keys = ['level', 'parent_map', 'orphan_count', 'orphan_ids', 'statistics']
                    missing_keys = [k for k in required_keys if k not in data]

                    if missing_keys:
                        print(f"❌ Level {level} provenance tree 缺少欄位: {missing_keys}")
                        all_good = False
                    else:
                        stats = data['statistics']
                        print(f"✅ Level {level} provenance tree 完整:")
                        print(f"   - Total SSAM masks: {stats.get('total_ssam_masks', 0)}")
                        print(f"   - Accepted: {stats.get('accepted_count', 0)}")
                        print(f"   - Rejected: {stats.get('rejected_count', 0)}")
                        print(f"   - Orphans: {data['orphan_count']}")

                        # 檢查 parent_map 不是空的
                        if not data['parent_map']:
                            print(f"⚠️  Level {level} parent_map 是空的")
                        else:
                            print(f"   - Parent relationships: {len(data['parent_map'])}")

                except json.JSONDecodeError as e:
                    print(f"❌ Level {level} provenance tree JSON 格式錯誤: {e}")
                    all_good = False
                except Exception as e:
                    print(f"❌ Level {level} provenance tree 讀取失敗: {e}")
                    all_good = False
            else:
                print(f"⚠️  Level {level} provenance tree 不存在: {tree_file}")

        # 檢查 family_tree.json
        family_tree = relations_dir / "family_tree.json"
        if family_tree.exists():
            try:
                with open(family_tree, 'r') as f:
                    data = json.load(f)
                print(f"✅ family_tree.json 存在")
                print(f"   - Total objects: {len(data.get('objects', {}))}")
                print(f"   - Cross-level families: {data.get('statistics', {}).get('family_count', 0)}")
            except Exception as e:
                print(f"❌ family_tree.json 讀取失敗: {e}")
                all_good = False
        else:
            print(f"⚠️  family_tree.json 不存在")

    # 檢查每個 level 的輸出
    for level in [2, 4, 6]:
        level_dir = run_path / f"level_{level}"
        if not level_dir.exists():
            print(f"⚠️  level_{level}/ 目錄不存在")
            continue

        tracking_dir = level_dir / "tracking"
        if tracking_dir.exists():
            # 檢查 NPZ 檔案
            video_segments = list(tracking_dir.glob("video_segments*.npz"))
            object_segments = list(tracking_dir.glob("object_segments*.npz"))

            if video_segments:
                print(f"✅ Level {level} video_segments: {video_segments[0].name}")
            else:
                print(f"❌ Level {level} video_segments 不存在")
                all_good = False

            if object_segments:
                print(f"✅ Level {level} object_segments: {object_segments[0].name}")
            else:
                print(f"❌ Level {level} object_segments 不存在")
                all_good = False
        else:
            print(f"❌ Level {level} tracking/ 目錄不存在")
            all_good = False

    print("=" * 80)
    if all_good:
        print("✅ 所有檢查通過！記憶體清理補丁不會造成數據丟失。")
    else:
        print("⚠️  部分檢查失敗，請檢查上述警告。")

    return all_good


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_memory_cleanup.py <run_directory>")
        print("範例: python verify_memory_cleanup.py outputs/experiments/v7_*/scene_00005_00")
        sys.exit(1)

    run_dir = sys.argv[1]

    # 支援 glob pattern
    if '*' in run_dir:
        from glob import glob
        matches = glob(run_dir)
        if not matches:
            print(f"❌ 找不到符合的目錄: {run_dir}")
            sys.exit(1)

        print(f"找到 {len(matches)} 個符合的目錄\n")
        all_pass = True
        for match in matches:
            if not verify_run_output(match):
                all_pass = False
            print()

        sys.exit(0 if all_pass else 1)
    else:
        success = verify_run_output(run_dir)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
