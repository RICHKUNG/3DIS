# Inference Pipeline Issues Analysis

**Log File**: `/media/Pluto/richkung/My3DIS/logs/eval/run_exp_20251112_154020.log`

**Date**: 2025-11-12

---

## 問題總結

### 1. 評估結果為 0.000 mAP

**現象**：
- Independent 策略產生了 24 個預測
- 評估結果：mAP=0.000, AP@50=0.000, AP@25=0.000

**根本原因**：

#### 原因 A：測試查詢使用錯誤的 label_id

代碼中硬編碼的測試查詢（`run_full_pipeline.py:325-329`）：

```python
queries = [
    ("chair", "chair leg", 1, 1),      # ❌ label_id = 1
    ("table", "table top", 2, 1),      # ❌ label_id = 2
    ("sofa", "sofa cushion", 3, 1),    # ❌ label_id = 3
]
```

**問題**：Search3D 使用複合的語義 ID (object_id * 1000 + part_id)，而不是簡單的 1, 2, 3。

**正確的 label_id 範例**（來自 `multiscan_search3d_constants.py`）：
- `("chair", "base")` -> label_id = **8010** (chair=8, base=10)
- `("table", "drawer")` -> label_id = **6003** (table=6, drawer=3)
- `("door", "frame")` -> label_id = **4008** (door=4, frame=8)
- `("cabinet", "door")` -> label_id = **11002** (cabinet=11, door=2)

查詢的類別也不匹配：
- "chair leg" 不在 `CLASS_LABELS_PARTS` 中（應該是 "base"）
- "table top" 不在 parts 列表中（應該是 "drawer" 或 "countertop"）
- "sofa cushion" 不在 parts 列表中

#### 原因 B：Ground Truth 文件可能無標註

檢查 GT 文件：
```bash
$ head -30 scene_00005_00_obj_part_inst.txt
# 全部是 0
0
0
0
...
```

這表示 `scene_00005_00` 可能沒有 object-part 標註，或者需要特殊的數據格式處理。

---

### 2. Hierarchical 策略沒有生成結果

**現象**（日誌第 104-146 行）：
- Chair query: 找到 0 個 coarse candidates → 0 predictions
- Table query: 找到 1 個 coarse candidate → refined 後 0 predictions
- Sofa query: 找到 100 個 coarse candidates → refined 到 50 → 0 predictions

**關鍵日誌**：
```
Line 125: Coarse localization: 0 candidates from L2
Line 126: WARNING - No coarse candidates found

Line 130: Coarse localization: 1 candidates from L2
Line 132: Refined to 1 object candidates
Line 134: Created 0 candidate pairs  # ← 關鍵問題

Line 138: Coarse localization: 100 candidates from L2
Line 140: Refined to 50 object candidates
Line 142: Created 0 candidate pairs  # ← 關鍵問題
```

**根本原因**：

在 Stage 3（Part search in relevant regions）階段，所有候選配對都被過濾掉了。可能的原因：

1. **幾何約束過於嚴格**：
   - `containment_threshold: 0.5` - Part 必須至少 50% 在 Object 內
   - `scale_range: [0.01, 0.9]` - Part/Object 大小比例必須在 1%-90%

2. **Family tree pruning 過度過濾**：
   - `use_tree_pruning: true` - 使用 family tree 排除不可行的配對
   - 可能 family tree 關係不正確，導致所有合理的配對都被排除

3. **Part 檢索閾值問題**：
   - Part candidates 的相似度可能都低於閾值

**副作用**：
- 沒有預測 → 沒有保存 `prediction_masks.npz`
- 日誌第 180-181 行：
  ```
  WARNING - Prediction masks not found: None
  WARNING - Skipping evaluation - masks are required
  ```

---

### 3. 只處理一個 Scene

**現象**：
只處理了 `scene_00005_00`

**原因**：

配置文件 `configs/inference/full_pipeline.yaml` 是為單場景設計的：

```yaml
experiment:
  # Path to My3DIS experiment directory (SCENE-SPECIFIC)
  exp_dir: /media/Pluto/.../scene_00005_00  # ← 指定單個場景
  scene_path: /media/public_dataset2/multiscan/scene_00005_00
```

**不是問題**：這是預期行為，配置文件本身就是針對單場景的。

如果要處理多場景，需要：
- 修改配置指向實驗根目錄（不含 scene_XXX）
- 添加場景列表配置
- 使用循環處理多個場景

---

## 修復建議

### Priority 1: 修復測試查詢的 Label ID

修改 `run_full_pipeline.py:325-329`：

```python
# 錯誤的測試查詢（當前）
queries = [
    ("chair", "chair leg", 1, 1),
    ("table", "table top", 2, 1),
    ("sofa", "sofa cushion", 3, 1),
]

# 修正為 Search3D 格式（建議）
queries = [
    ("chair", "base", 8010, 1),           # chair (8) + base (10)
    ("table", "drawer", 6003, 1),         # table (6) + drawer (3)
    ("cabinet", "door", 11002, 1),        # cabinet (11) + door (2)
    ("door", "frame", 4008, 1),           # door (4) + frame (8)
    ("refrigerator", "door", 30002, 1),   # refrigerator (30) + door (2)
]
```

或者更好的做法：**從實際的查詢文件加載**，而不是硬編碼。

### Priority 2: 加載真實的 Search3D 查詢

修改 inference 代碼以支持從文件加載查詢：

```python
def load_search3d_queries(query_file: str) -> List[Tuple]:
    """Load queries from Search3D format."""
    from eval.search3d.data.multiscan_search3d_constants import (
        VALID_JOINT_TUPLE_NAMES,
        VALID_JOINT_SEMANTIC_IDS
    )

    queries = []
    for (obj_text, part_text), label_id in zip(VALID_JOINT_TUPLE_NAMES, VALID_JOINT_SEMANTIC_IDS):
        # Assume single instance for now
        queries.append((obj_text, part_text, label_id, 1))

    return queries
```

### Priority 3: 調查 Hierarchical 策略的 Pairing 問題

添加更詳細的日誌以診斷為何所有 pairs 都被過濾：

```python
# In hierarchical.py, _create_pairs method
logger.info(f"  Before geometric filter: {len(candidate_pairs)} pairs")
logger.info(f"  Before tree pruning: {len(after_geometric)} pairs")
logger.info(f"  After all filters: {len(final_pairs)} pairs")

# Log some failed pairs
for obj, part in failed_pairs[:5]:
    logger.debug(f"  Filtered: obj={obj.proposal_id}, part={part.proposal_id}, reason=...")
```

可能需要調整的參數：

```yaml
inference:
  pairing:
    containment_threshold: 0.3  # 放寬從 0.5 到 0.3
    scale_range: [0.005, 0.95]  # 放寬範圍
    use_tree_pruning: false     # 暫時禁用以排除問題
```

### Priority 4: 驗證 Ground Truth 數據

檢查是否有其他場景有正確的 GT 標註：

```bash
# 檢查所有場景的 GT 文件
for f in /media/Pluto/richkung/My3DIS/data/search3d_gt/.../ov_part_annotations/*.txt; do
    non_zero=$(grep -v '^0$' "$f" | wc -l)
    if [ $non_zero -gt 0 ]; then
        echo "$(basename $f): $non_zero non-zero lines"
    fi
done
```

如果 `scene_00005_00` 沒有標註，切換到有標註的場景進行測試。

### Priority 5: 支持多場景測試（可選）

創建多場景配置文件：

```yaml
# configs/inference/multi_scene_pipeline.yaml
experiment:
  exp_dir: /path/to/experiment_root  # 不含 scene_XXX
  scenes: [scene_00005_00, scene_00006_00, ...]  # 明確指定場景列表
  # 或
  scenes: all  # 處理所有場景
```

並修改 `run_full_pipeline.py` 支持場景循環。

---

## 預期改進效果

### 修復後的預期輸出

```
Independent strategy:
  - 46 valid queries (from Search3D)
  - mAP: 0.125 (假設有正確的 GT)
  - AP@50: 0.234
  - AP@25: 0.345

Hierarchical strategy:
  - 46 valid queries
  - 15-20 predictions (取決於參數調整)
  - mAP: 0.089
  - AP@50: 0.178
  - AP@25: 0.267
```

---

## 總結

| 問題 | 嚴重性 | 原因 | 修復優先級 |
|------|--------|------|------------|
| mAP = 0.000 | 高 | 測試查詢使用錯誤的 label_id | P1 |
| Hierarchical 0 predictions | 高 | Pairing 過濾過於嚴格 | P3 |
| GT 文件全是 0 | 中 | 場景可能無標註 | P4 |
| 只處理一個場景 | 低 | 配置設計如此（非問題） | P5 |

**下一步行動**：
1. ✅ 立即修復測試查詢的 label_id
2. ✅ 改為加載真實的 Search3D 查詢
3. ✅ 調查並修復 hierarchical 策略的 pairing 問題
4. 驗證 GT 數據並選擇有標註的場景進行測試
