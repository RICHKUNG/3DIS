# My3DIS 問題分析與修復報告

## 問題總結

### 1. 為何 total_families > L2 objects？

**現象**：scene_00006_00 有 178 個 L2 物件，但有 191 個 families

**根本原因**：
- 有 13 個 L4 orphan 物件（parent_id 指向不存在的 L2 物件）
- 這些 orphan 各自成為一個 family 的 root
- **計算**：191 families = 178 L2 roots + 13 L4 orphan roots

**驗證**：
```python
# Orphan 分析結果
Orphan ID 4007: parent_id=2004 (L2 物件 2004 不存在)
Orphan ID 4018: parent_id=2016 (L2 物件 2016 不存在)
Orphan ID 4111: parent_id=2075 (L2 物件 2075 不存在)
...

# Families by root level
L2 roots: 178
L4 roots: 13  ← orphans
Total: 191 ✓
```

---

### 2. 為何會有 orphans？

**根本原因**：L4 物件的 parent SSAM masks 沒有被 SAM2 接受追蹤

**可能原因**：
1. **Filter 階段被過濾**：Parent mask 面積太小或穩定度不夠
2. **Dedup reject**：Parent mask 與其他 L2 物件 IoU 重疊過高被拒絕
3. **追蹤失敗**：SAM2 無法成功追蹤該 mask

**追蹤路徑**：
```
SSAM L2 candidate (frame X, mask 2004)
  → Filter stage (pass/fail?)
  → SAM2 tracking (accept/reject?)
  → Provenance tree
       ↓
    若被 reject → 不會出現在 provenance_tree_L2.json
       ↓
    L4 child (4007) 的 parent_id=2004
       ↓
    但 2004 不在 parent_map 中
       ↓
    4007 成為 orphan ✗
```

**建議調查**：
- 檢查 `level_2/raw/manifest.json` - SSAM 是否生成 2004
- 檢查 `level_2/filtered/filtered.json` - 2004 是否通過 filter
- 檢查 SAM2 dedup logs - 2004 是否被 reject

---

### 3. 為何 L4 物件沒有 L2 parents？

**答案**：與問題 2 相同 - parent SSAM masks 沒有成功進入 provenance tree

---

### 4. 現在會有一個 mask 有兩個 parents 的現象嗎？

**答案**：**不會**

**證據**：
- `ProvenanceTracker` 中每個物件只有一個 `ssam_parent_id`
- `family_tree.json` 中每個 object 只有一個 `parent_id` field
- 資料結構設計上不支援多 parents

```python
# src/my3dis/tracking/provenance_tracker.py
self.sam2_provenance[sam2_obj_id] = {
    "ssam_parent_id": ssam_parent_id,  # 單一 parent
    ...
}
```

---

### 5. Family tree visualization 失敗（只顯示原圖）

**現象**：所有 visualization 影像的 L2/L4/L6 panels 都沒有 mask overlay，只有原始 RGB

**根本原因 1：mask_archive 路徑錯誤**

`family_tree_builder.py` 儲存的路徑：
```python
'mask_archive': 'level_2/tracking/video_segments_scale0.3x.npz'  # 舊格式路徑
```

實際檔案位置：
```
level_2/video_segments_L02.npz  # 新格式路徑
```

**根本原因 2：NPZ format 不相容**

新 NPZ 格式（ZIP structure）：
```python
archive.files = [
    'frames/frame_000000.json',  # ← 新格式有 'frames/' 前綴
    'frames/frame_000050.json',
    ...
]
```

`family_tree_query.py:130` 使用的 key：
```python
frame_key = f"frame_{frame_idx:06d}"  # ✗ 沒有 'frames/' 前綴
```

**結果**：
- `load_mask()` 找不到檔案 → 回傳 `None`
- `visualize_families.py` 收到 `None` → 沒有 mask overlay
- 只顯示原始 RGB frame

---

### 6. scene_00006_00 沒有出現在 check.md

**根本原因**：`reporting.py` 尋找舊格式 `relations.json`

```python
# src/my3dis/workflow/reporting.py:53
relations_path = scene_dir / "relations.json"  # ✗ 舊格式
if not relations_path.is_file():
    return None  # ← scene_00006_00 沒有 relations.json，直接回傳 None
```

**實際情況**：
- scene_00006_00 有 `workflow_summary.json` ✓
- scene_00006_00 有 `family_tree.json` ✓
- 但沒有 `relations.json` ✗（新版本已廢棄）

---

## 修復方案

### 修復 1：family_tree_builder.py - 正確尋找 video_segments 路徑

**問題**：Hard-coded 路徑順序不包含新格式

**修復**：優先尋找新格式 `video_segments_L{level:02d}.npz`

### 修復 2：family_tree_query.py - 支援新 ZIP 格式

**問題**：只處理舊格式 `frame_XXXXXX` keys

**修復**：檢測格式並使用正確的 key prefix

### 修復 3：reporting.py - 使用 family_tree.json 讀取統計

**問題**：依賴已廢棄的 `relations.json`

**修復**：
- 選項 A：直接讀取 `family_tree.json` (推薦)
- 選項 B：從 `workflow_summary.json` 讀取統計

---

## 影響範圍

- **現有 experiments**：需要重新生成 family_tree.json（使用修復後的程式碼）
- **現有 visualizations**：需要重新執行 visualization（修復後才能正常顯示 masks）
- **check.md reports**：需要重新生成（修復後才能讀取統計資料）

---

## 後續建議

1. **增強 orphan 追蹤**：
   - 在 workflow 中記錄 SSAM → Filter → Dedup 各階段的 reject 原因
   - 輸出 orphan analysis report，說明每個 orphan 的 parent 為何不存在

2. **統一格式檢測**：
   - 所有讀取 NPZ 的地方都應該有格式檢測邏輯
   - 考慮提供 migration script 轉換舊格式 → 新格式

3. **改進 error messages**：
   - `load_mask()` 失敗時應該 warn，而不是靜默回傳 None
   - Visualization 應該顯示錯誤原因（如 "Mask not found"）

4. **重構 relations 系統**：
   - 完全移除 `relations.json` (舊格式)
   - 統一使用 `family_tree.json` 作為唯一 source of truth
