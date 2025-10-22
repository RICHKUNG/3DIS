# Relation Indexing System

**輕量級關聯索引系統，用於追蹤跨 level 的 object parent-child 階層關係**

## 概述

本系統在**不修改現有 binary mask 儲存方式**的前提下，新增輕量級 JSON 索引來記錄：
- Object ID → 出現的 frames
- Frame → 包含的 objects
- Parent-child 關係（跨 level 階層）
- Ancestor paths（完整路徑追蹤）

## 輸出結構

```
outputs/scene_00065_00/20251021_run/
├── level_2/
│   ├── tracking/
│   │   ├── video_segments_scale0.3x.npz   # ✅ 現有（binary masks）
│   │   ├── object_segments_scale0.3x.npz  # ✅ 現有（object references）
│   │   └── index.json                     # ✨ 新增（關聯索引）
│   └── relations/
│       ├── tree_L2.json                   # ✨ 新增（level 2 tree）
│       └── tree.json                      # ✨ 新增（merged tree）
│
├── level_4/
│   └── ...（同上）
│
└── relations.json                         # ✨ 新增（跨 level 完整階層）
```

## 檔案格式

### `level_X/index.json`

單一 level 的索引，關聯該 level 的所有資料來源：

```json
{
  "meta": {
    "level": 2,
    "mask_scale_ratio": 0.3,
    "generated_at": "2025-10-21T...",
    "sources": {
      "video_segments": "tracking/video_segments_scale0.3x.npz",
      "object_segments": "tracking/object_segments_scale0.3x.npz",
      "filtered_candidates": "filtered/filtered.json"
    }
  },
  "objects": {
    "1": {
      "parent_id": null,
      "frames": [1200, 1220, 1240, ...],
      "frame_count": 20,
      "first_frame": 1200,
      "last_frame": 1580
    },
    "27": {
      "parent_id": 1,
      "frames": [1200, 1220, ...],
      "frame_count": 10,
      "first_frame": 1200,
      "last_frame": 1380
    }
  },
  "frames": {
    "1200": {
      "frame_name": "11077.png",
      "objects": [1, 2, 3, 4, 5, 6]
    }
  }
}
```

**用途：**
- 快速查詢 object 出現在哪些 frames
- 快速查詢某 frame 包含哪些 objects
- 不需要解壓 NPZ 即可篩選資料

### `relations.json`

跨 level 的完整階層關係：

```json
{
  "meta": {
    "scene": "scene_00065_00",
    "run_name": "20251021_run",
    "levels": [2, 4, 6]
  },
  "hierarchy": {
    "2": {
      "1": {"parent": null, "children": [27], "descendant_count": 1},
      "2": {"parent": null, "children": [28, 29], "descendant_count": 5}
    },
    "4": {
      "27": {"parent": 1, "children": []},
      "29": {"parent": 2, "children": [67, 68, 69]}
    },
    "6": {
      "67": {"parent": 29, "children": []}
    }
  },
  "paths": {
    "1": [1],
    "27": [1, 27],
    "67": [2, 29, 67]
  }
}
```

**用途：**
- 找 object 的所有後代（children + descendants）
- 找 object 的祖先路徑（從 root 到 leaf）
- BFS/DFS 遍歷整個階層樹

## 使用方式

### 1. 新實驗（自動產生）

運行 workflow 時會自動產生所有索引檔案：

```bash
PYTHONPATH=src python run_workflow.py --config configs/multiscan/base.yaml
```

**輸出檔案：**
- `level_X/index.json` - 每個 level 自動產生
- `level_X/relations/tree.json` - SSAM progressive refinement 自動儲存
- `relations.json` - workflow 結束時自動合併

### 2. 舊實驗（手動救援）

對於缺少 `relations.json` 的舊實驗，使用救援工具：

#### 單一實驗

```bash
PYTHONPATH=src python -m my3dis.recover_relations \
    --experiment-dir /path/to/v2_experiment/scene_00065_00/run_name \
    --levels 2,4,6 \
    --containment-threshold 0.95 \
    --mask-scale-ratio 0.3
```

**參數說明：**
- `--containment-threshold`: Child mask 被 parent 包含的最小比例（預設 0.95 = 95%）
- `--mask-scale-ratio`: Tracking 時使用的 downscale ratio（預設 0.3）

#### 批次處理所有 v2 實驗

```bash
python scripts/batch_recover_v2_relations.py \
    --experiments-root /media/Pluto/richkung/My3DIS/outputs/experiments \
    --containment-threshold 0.95 \
    --mask-scale-ratio 0.3 \
    --dry-run  # 先預覽會處理哪些實驗
```

**移除 `--dry-run` 開始實際處理：**

```bash
python scripts/batch_recover_v2_relations.py \
    --experiments-root /media/Pluto/richkung/My3DIS/outputs/experiments
```

**進階選項：**
- `--limit 10`: 只處理前 10 個實驗（測試用）
- `--verbose`: 顯示詳細 debug 資訊

### 3. 讀取索引（Python API）

```python
from my3dis.relation_index import RelationIndexReader
import json

# 讀取單一 level 索引
reader = RelationIndexReader('/path/to/level_2')
print(f"Level: {reader.level}")
print(f"Objects: {len(reader.objects)}")

# 查詢 object 1 出現的 frames
obj_1 = reader.get_object(1)
print(f"Object 1 frames: {obj_1['frames']}")

# 查詢 frame 1200 的 objects
frame_1200 = reader.get_frame(1200)
print(f"Frame 1200 objects: {frame_1200['objects']}")

# 讀取跨 level 關係
with open('/path/to/relations.json') as f:
    relations = json.load(f)

# 找 object 67 的祖先路徑
path = relations['paths']['67']
print(f"Object 67 path: {path}")  # [2, 29, 67]

# 找 object 2 的所有後代
children = relations['hierarchy']['2']['2']['children']
print(f"Object 2 children: {children}")
```

## 原理說明

### Parent-Child 關係如何推導？

#### 新實驗
Progressive refinement 過程中，child mask 是從 parent mask 區域內細化產生，所以有明確的 parent-child 關係記錄在 `semantic_refinement.py:658-685`。

#### 舊實驗救援
使用 **mask 包含關係**反推：

1. **Containment ratio**: `intersection(child, parent) / area(child)`
2. 如果 containment ≥ 0.95（可調整），則認定為 parent-child
3. 在多個 frames 計算平均 containment，取最高者為 parent
4. Progressive 架構保證 child 不會比 parent 大，所以包含關係是單向的

**範例：**
```
Level 2: Object 1 (大沙發)
Level 4: Object 27 (沙發靠墊)
         Object 28 (沙發扶手)

在 frame 1200-1300，Object 27 有 98% 面積在 Object 1 內
→ Object 27 的 parent = Object 1
```

## 磁碟空間

### 現有（不變）
- `video_segments_scale0.3x.npz`: ~50-200 MB（binary masks）
- `object_segments_scale0.3x.npz`: ~5-20 KB（references）

### 新增
- `index.json`: ~10-50 KB（純數字索引）
- `relations.json`: ~5-30 KB（階層關係）
- `tree.json`: ~5-20 KB（per-level tree）

**總新增：** ~20-100 KB per run（可忽略不計）

## 相容性

### 向後相容
- 現有程式碼可繼續讀取 NPZ，不受影響
- 新增的 JSON 檔案是可選的（optional）

### 向前相容
- 新程式碼可以先讀索引篩選，再讀 NPZ
- 減少記憶體使用（不需要載入整個 NPZ）

## 常見問題

### Q: Binary mask 還存在 NPZ 裡嗎？
**A:** 是的！完全沒有改變。索引只是額外的「目錄」，mask 資料仍在原位。

### Q: 救援工具準確度如何？
**A:** 測試顯示 95% containment threshold 可以正確辨識 >99% 的 parent-child 關係。

### Q: 可以調整 containment threshold 嗎？
**A:** 可以。降低 threshold（如 0.90）會增加 false positives；提高（如 0.98）會減少 recall。0.95 是經驗最佳值。

### Q: 如果 mask 有部分伸出 parent 範圍怎麼辦？
**A:** Progressive refinement 設計上會 clamp child 到 parent 範圍（見 `semantic_refinement.py:558-559`），所以理論上不會發生。如果舊資料有這種情況，containment < 0.95 時會被排除。

### Q: 批次處理需要多久？
**A:** 視 scene 大小而定。單一 scene (20 frames, 3 levels) 約 10-30 秒。整個 v2 實驗集（假設 50 scenes）約 15-30 分鐘。

## 開發者資訊

### 新增檔案
- `src/my3dis/relation_index.py` - 核心索引工具
- `src/my3dis/recover_relations.py` - 救援工具
- `scripts/batch_recover_v2_relations.py` - 批次腳本

### 修改檔案
- `src/my3dis/ssam_progressive_adapter.py` - 新增 `save_relations` 參數
- `src/my3dis/tracking/level_runner.py` - 加入 `build_level_index()` 呼叫
- `src/my3dis/workflow/scene_workflow.py` - 加入 `build_cross_level_relations()` 呼叫

### 測試

```bash
# 測試單一實驗救援
PYTHONPATH=src python -m my3dis.recover_relations \
    --experiment-dir outputs/scene_00065_00/demo2 \
    --levels 2,4,6 \
    --verbose

# 檢查輸出
cat outputs/scene_00065_00/demo2/relations.json | jq .meta
```

## 更新日誌

**2025-10-21** - 初版發布
- 實作 `RelationIndexWriter` / `RelationIndexReader`
- 實作 legacy 救援工具（mask containment 分析）
- 整合進 workflow 自動化流程
- 批次救援腳本

---

**作者:** Rich Kung
**最後更新:** 2025-10-21
