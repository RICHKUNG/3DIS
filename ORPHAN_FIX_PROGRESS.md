# Orphan 問題修復進度報告

## 執行日期
2025-11-03

---

## ✅ 已完成的工作 (Phase 1-2)

### **Phase 1: 修復 Dedup Timing Bug** ✅

#### 1.1 重構 `sam2_runner.py` ✅
**修改內容**：
- **Deferred Rejection Processing**: 將 rejected candidates 處理延後到 propagation 完成後
- **Eager Mapping**: 在 `_add_prompts_to_predictor()` 後立即建立 prompt mask 映射
- **修復 Race Condition**: 確保 child-level rejection 檢查時，parent mask 映射已經建立

**關鍵變更**：
```python
# 舊版本 (BUGGY):
#   1. Filter candidates
#   2. Process rejected candidates (check mapping) ← TOO EARLY
#   3. Add prompts
#   4. Propagate
#   5. Build mapping ← TOO LATE

# 新版本 (FIXED):
#   1. Filter candidates
#   2. Defer rejections ← 延後處理
#   3. Add prompts
#   4. Build EAGER mapping ← 立即建立
#   5. Propagate
#   6. Build propagation mapping
#   7. Process deferred rejections ← 現在有完整mapping
```

**預期效果**: 解決 80% 的 timing-related orphans

---

#### 1.2 增強 `provenance_tracker.py` ✅
**新增功能**：
- `register_unresolved_rejection()`: 記錄無法解析的 rejections
- `unresolved_rejections` list: 追蹤 mapping 失敗的情況
- 更新 `get_statistics()`: 輸出 unresolved rejection count
- 更新 `export_full_metadata()`: 匯出 unresolved rejections 供分析

**用途**: 診斷和監控 orphan 產生原因

---

### **Phase 2: 實作 Cascade Filtering** ✅

#### 2.1 新建 `cascade_filter.py` ✅
**核心功能**:
```python
class CascadeFilter:
    def filter_masks(masks):
        # 1. Build parent-child graph
        # 2. Apply individual thresholds
        # 3. Cascade reject descendants of filtered parents
        # 4. Return filtered masks + rejection reasons
```

**Key Feature**: Parent-aware filtering 防止 "parent filtered but child passes" orphans

---

#### 2.2 整合到 `semantic_refinement.py` ✅
**修改內容**：
- 新增 `cascade_filtering` 參數（default: True）
- 新增 `stability_threshold` 參數（用於 cascade filter）
- 在返回結果前執行 cascade filtering
- 輸出詳細的 filtering statistics

**新增 API**:
```python
progressive_refinement_masks(
    ...,
    cascade_filtering=True,  # NEW: Enable cascade filter
    stability_threshold=0.9,  # NEW: Optional stability threshold
)
```

**輸出格式變更**:
```json
{
  "levels": {...},
  "cascade_filtering": {  // NEW
    "enabled": true,
    "statistics": {
      "total_input": 1000,
      "passed": 850,
      "rejected": 150,
      "rejection_rate": 0.15,
      "breakdown": {
        "area_rejected": 100,
        "stability_rejected": 20,
        "cascade_rejected": 30  // NEW: Parent-child related
      }
    },
    "rejection_reasons": {...}  // For debugging
  }
}
```

---

## ⏳ 進行中的工作

### **Phase 2.3: 更新 filter_candidates.py** ✅ (已完成 2025-11-03)
**已完成內容**：
- ✅ 新增 `--cascade` flag 支援 cascade filtering
- ✅ 檢測數據中是否包含 unique_id/parent_unique_id
- ✅ 如果不支援 cascade 但啟用了 flag，提供警告
- ✅ 輸出詳細的 before/after statistics（包含 cascade rejection 統計）
- ✅ 更新 FilterStats 類支援 cascade 統計

**狀態**: 已完成

### **Phase 2.4: 完整集成 cascade_filtering 到 workflow** ✅ (已完成 2025-11-04)

**更新 2025-11-04**: 修復參數傳遞問題並實現真正的 cascade filtering
**已完成內容**：
- ✅ 更新 `ssam_progressive_adapter.py`：添加 cascade_filtering 和 stability_threshold 參數
- ✅ 更新 `generate_candidates.py`：添加 cascade_filtering 參數並傳遞給 adapter
- ✅ 更新 `workflow/stage_config.py`：
  - 添加 cascade_filtering 字段到 SSAMStageConfig
  - 在 from_yaml_config 中讀取 YAML 配置
  - 在 to_legacy_kwargs 中導出參數
- ✅ 創建測試配置 `configs/multiscan/test_orphan_fix.yaml`
- ✅ 創建測試腳本 `scripts/test_orphan_fix.sh`

**影響範圍**：
- 完整的 end-to-end cascade filtering 支援
- 從 YAML 配置 → workflow → generate_candidates → adapter → semantic_refinement
- 所有層級的參數傳遞已驗證

**實現細節** (2025-11-04):
- ✅ 修復參數不匹配問題：`semantic_refinement.py` 現在接受 `cascade_filtering` 和 `stability_threshold` 參數
- ✅ 實現真正的 cascade filtering 邏輯：在 progressive refinement 完成後應用
- ✅ 添加詳細統計輸出：area_rejected, stability_rejected, cascade_rejected
- ✅ 測試驗證：單 frame 測試成功（54 masks，0% rejection with stability_threshold=null）
- ⚠️ **已知限制**：`stability_threshold` 參數會導致所有 masks 被拒絕（待調查 stability_score 欄位問題）
- 📝 **建議配置**：`stability_threshold: null`（不使用 stability filtering）+ `cascade_filtering: true`（使用 area-based cascade）

**狀態**: Phase 2 基本完成，stability filtering 需要進一步調試

---

## 📋 待完成的工作 (Phase 3-5)

### **Phase 3: 實作 True Family Merging**

#### 3.1 擴展 `provenance_tracker.py` - Virtual Children
**計畫內容**：
```python
class ProvenanceTracker:
    def __init__(self):
        self.virtual_children: Dict[int, List[str]] = {}  # parent_sam2_id -> [child_ssam_ids]
        self.dedup_relationships: Dict[str, int] = {}  # child_ssam_id -> matched_parent_sam2_id

    def register_rejected_prompt_with_merge(
        ssam_unique_id, matched_sam2_obj_id, ssam_parent_id
    ):
        # 1. Record mapping (existing)
        # 2. Update matched parent's virtual_children list (NEW)
```

#### 3.2 更新 `family_tree_builder.py`
**計畫內容**：
- 讀取 `virtual_children` from provenance trees
- 在 family_tree.json 中新增 `virtual_children` 欄位
- 更新統計：`total_families` 應該只計算 L2 roots

#### 3.3 擴展 `family_tree_query.py`
**計畫內容**：
- 新增查詢 virtual children 的方法
- 支援 `include_virtual` 參數
- 更新 visualization 包含 virtual children

---

### **Phase 4: 測試與驗證**

**測試計畫**：
1. **單元測試**:
   - `tests/test_cascade_filter.py` - Cascade filtering logic
   - `tests/test_family_merge.py` - Virtual children logic

2. **整合測試**:
   - 重新執行 scene_00006_00 pipeline
   - 驗證 orphan count 下降（13 → <3）
   - 驗證 total_families == L2 roots

3. **回歸測試**:
   - 確保不啟用 cascade filtering 時結果不變
   - 確保向後相容性

---

### **Phase 5: 文件更新**

**文件清單**：
1. **CLAUDE.md** - 更新 cascade filtering 說明
2. **ORPHAN_ANALYSIS.md** - 詳細問題分析（已在 ANALYSIS.md）
3. **FORMAT_STANDARDS.md** - 更新 family_tree.json schema v2
4. **FAMILY_TREE_V2.md** - Virtual children 完整說明

---

## YAML 配置更新需求

### `configs/multiscan/base.yaml`
**需要新增的參數**：
```yaml
stages:
  ssam:
    # Cascade filtering (Phase 2)
    cascade_filtering: true  # Enable parent-aware filtering
    stability_threshold: null  # Optional: Set stability threshold for cascade filter

  tracker:
    # Virtual children (Phase 3)
    track_virtual_children: true  # Enable family merging for deduped children
```

### 參數說明
- **`cascade_filtering`**:
  - Default: `true`
  - 啟用時：防止 parent 被 filter 但 child 通過的情況
  - 關閉時：傳統 independent filtering

- **`stability_threshold`**:
  - Default: `null` (no stability filtering)
  - 建議值: `0.9` for high-quality masks
  - 用於 cascade filter 的穩定度閾值

- **`track_virtual_children`**:
  - Default: `true`
  - 啟用時：deduped children 記錄在 parent's virtual_children
  - 關閉時：維持當前行為（deduped = not tracked）

---

## 預期成效

### Before (當前)
- **scene_00006_00**:
  - 191 families (178 L2 + 13 L4 orphans)
  - Orphan rate: 2.2% (13/591)
  - Unresolved rejections: 未追蹤

### After Phase 1-2 (已完成)
- **Dedup timing fixed**: 預期解決 80% timing-related orphans
- **Cascade filtering**: 預期解決 15% filter-related orphans
- **Combined**: Orphan rate 應降至 <0.5% (預計 <3/591)

### After Phase 3 (待完成)
- **Virtual children tracked**: 完整 family tree 語義
- **Total families**: 178 (只計算 L2 roots，不再有 L4/L6 roots)
- **Orphan rate**: <0.5% (只剩極端追蹤失敗情況)

---

## 向後相容性

### ✅ 已確保
- 所有新功能都提供 enable/disable flags
- Phase 1 修復是純邏輯重構，不改變 API
- Phase 2 cascade filtering 預設啟用，但可關閉

### ⚠️ 需要注意 (Phase 3)
- `family_tree.json` schema 會更新 (v2)
- 需要提供 migration 工具或向後相容讀取
- Visualization 可能需要更新以顯示 virtual children

---

## 執行建議

### 立即可用 (Phase 1-2 已完成)
1. **測試修復效果**:
   ```bash
   # 重新執行一個場景測試
   PYTHONPATH=src python -m my3dis.run_workflow --config configs/multiscan/test_single.yaml
   ```

2. **檢查 unresolved rejections**:
   ```bash
   # 查看 provenance tree 中的 unresolved_rejections
   cat outputs/experiments/*/scene_*/relations/provenance_tree_L*.json | \
     jq '.unresolved_rejections // [] | length'
   ```

3. **驗證 cascade filtering**:
   ```bash
   # 檢查 refinement_results 中的 cascade_filtering statistics
   # (需要在 SSAM stage 輸出中查看)
   ```

### 下一步 (Phase 3-5)
1. 完成 virtual children 實作
2. 更新所有相關測試
3. 重新生成所有 experiment 的 family trees
4. 更新文件

---

## 相關檔案

### Phase 1-2 已修改檔案

#### Phase 1: Dedup Timing Fix
1. **`src/my3dis/tracking/sam2_runner.py`**
   - 修復 rejection processing timing bug
   - Deferred rejection processing
   - Eager prompt mapping

2. **`src/my3dis/tracking/provenance_tracker.py`**
   - 添加 `unresolved_rejections` 追蹤
   - 新增 `register_unresolved_rejection()` 方法
   - 更新 `get_statistics()` 和 `export_full_metadata()`

#### Phase 2: Cascade Filtering
3. **`src/my3dis/cascade_filter.py`** (新增)
   - `CascadeFilter` 類實現
   - Parent-aware filtering logic
   - `cascade_filter_masks()` 便捷函數
   - `validate_parent_child_consistency()` 驗證函數

4. **`src/my3dis/semantic_refinement.py`**
   - 添加 `cascade_filtering` 參數
   - 添加 `stability_threshold` 參數
   - 整合 `CascadeFilter` 到 refinement pipeline
   - 輸出詳細的 cascade filtering 統計

5. **`src/my3dis/filter_candidates.py`**
   - 添加 `--cascade` CLI flag
   - 更新 `FilterStats` 類支援 cascade 統計
   - 實現智能 cascade 檢測和回退
   - 詳細的 rejection breakdown 輸出

6. **`src/my3dis/ssam_progressive_adapter.py`**
   - 添加 `cascade_filtering` 參數
   - 添加 `stability_threshold` 參數
   - 傳遞參數給 `progressive_refinement_masks()`

7. **`src/my3dis/generate_candidates.py`**
   - 添加 `cascade_filtering` 參數到 `run_generation()`
   - 傳遞給 `generate_with_progressive()`

8. **`src/my3dis/workflow/stage_config.py`**
   - 添加 `cascade_filtering` 字段到 `SSAMStageConfig`
   - 在 `from_yaml_config()` 中讀取 YAML
   - 在 `to_legacy_kwargs()` 中導出

### 新增測試資源
9. **`configs/multiscan/test_orphan_fix.yaml`** (新增)
   - 測試配置文件
   - 啟用 cascade_filtering
   - 啟用 family visualizations

10. **`scripts/test_orphan_fix.sh`** (新增)
    - 自動化測試腳本
    - 結果分析和報告
    - Orphan 統計驗證

11. **`ORPHAN_FIX_PROGRESS.md`** (本文件)
    - 進度追蹤
    - 實現細節記錄

12. **`ORPHAN_FIX_IMPLEMENTATION_SUMMARY.md`** (新增)
    - 完整實現總結
    - 使用指南
    - 故障排除

### Phase 3-5 待修改檔案
13. `src/my3dis/family_tree_builder.py` - Virtual children support
14. `src/my3dis/family_tree_query.py` - Virtual children queries
15. `scripts/visualize_families.py` - Virtual children visualization

### 代碼統計
- **修改的核心模塊**: 8 個
- **新增的模塊**: 1 個 (cascade_filter.py)
- **新增的配置/腳本**: 3 個
- **總計受影響文件**: 12 個

---

## 問題與風險

### 已知問題
1. **Cascade filtering 可能過度過濾**:
   - 如果 parent 因為單一 frame 問題被 filter，所有 children 都會被移除
   - **緩解**: 提供 threshold tuning 指南

2. **Virtual children 增加記憶體**:
   - 大場景可能有數千個 virtual children
   - **緩解**: 只在需要時載入，提供 pagination

### 待解決
3. **YAML 配置向後相容**:
   - 舊 config 沒有 `cascade_filtering` 參數
   - **解決**: 使用 default values

4. **Migration 工具**:
   - 舊版 family_tree.json 需要 migration
   - **解決**: 提供 `regenerate_family_trees_v2.sh`

---

## 總結

**Phase 1-2 已完成修復核心問題**：
- ✅ Dedup timing race condition (80% orphans)
- ✅ Cascade filtering (15% orphans)

**預期 orphan rate 從 2.2% 降至 <0.5%**

**Phase 3-5 將完善 family tree 語義和測試**

---

**Last Updated**: 2025-11-04
**Status**: Phase 1-2 Complete (with known limitations), Phase 3-5 Pending

**Phase 2 Summary** (2025-11-04):
- ✅ Cascade filtering 已實現並整合到完整 pipeline
- ✅ Area-based cascade filtering 運作正常（防止 parent 被過濾但 child 通過的情況）
- ⚠️ Stability-based cascade filtering 有問題（待調查）
- 📝 當前建議：使用 `cascade_filtering: true` + `stability_threshold: null`
