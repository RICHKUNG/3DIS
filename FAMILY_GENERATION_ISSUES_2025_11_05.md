# Family Generation 機制缺陷分析與修復方案

**日期**: 2025-11-05
**實驗**: test_1105_ssam4_filter300_fill500/scene_00005_00
**日誌**: /media/Pluto/richkung/My3DIS/logs/test/run_exp_20251105_161145.log

## 問題總結

### 1. Orphan 誤報問題 (已識別根本原因)

**現象**: 4077 和 4078 在 family_tree.json 中同時有 `parent_id: 2043` 但被列為 orphans

**根本原因**:
- **provenance_tracker 遺漏物件**: 原始的 `provenance_tree_L2.json` 只包含 139 個物件，但 `index.json` 有 209 個物件（**遺漏 70 個物件，包括 2043**）
- provenance_tracker 在 SAM2 tracking 階段只記錄了部分物件，導致 parent 物件丟失
- family_tree_builder 依賴 provenance_tree，當 parent 不在 provenance_tree 中時，children 被錯誤標記為 orphan

**影響範圍**:
```
L2: 139/209 objects recorded (66.5%) - 70 missing
L4: 494/355 objects recorded - 待確認
L6: 待確認
```

### 2. 可視化黑色問題 (已修復)

**現象**: L2/L4 顯示黑色但標註有 "X objects"

**根本原因**:
- `family_tree_builder.py` 的路徑查找邏輯無法找到新格式的 video_segments 文件
- 舊路徑格式: `level_X/tracking/video_segments_scale0.3x.npz`
- 新路徑格式: `level_X/video_segments_L0X.npz`

**修復方案**: ✅ 已修復
1. 更新 `family_tree_builder.py:161-170` 的路徑候選列表，優先查找新格式
2. 更新 `family_tree_query.py:176-192` 添加 fallback 路徑查找邏輯

**驗證結果**:
```bash
# 修復前
Mask archive not found: level_2/tracking/video_segments_scale0.3x.npz

# 修復後
✓ Mask loaded successfully! Shape: (432, 576), Non-zero pixels: 12565
```

### 3. 包含關係異常 (已識別，SSAM 階段問題)

**現象**:
- Family 46 中 L6 的 mask 比 L2 還要大
- Family 42 中 L2 和 L4 幾乎一樣

**可能原因**:
1. SSAM 階段的包含關係檢查未正確執行
2. SAM2 tracking 階段的 mask 傳播導致尺寸變化
3. Gap filling 機制可能引入了不符合層級包含關係的 mask

**待調查**: 需檢查 `semantic_refinement.py` 和 `progressive_refinement` 的包含關係邏輯

### 4. Cross-Level Parent 關係丟失 (核心問題)

**現象**: 重建後所有物件都變成獨立 family (747 families for 747 objects)

**根本原因**:
- `rebuild_provenance_trees.py` 從 `index.json` 重建，但 index.json **不包含 cross-level parent 關係**
- index.json 只記錄同一 level 內的關係
- 原始的 provenance_tree 由 provenance_tracker 在 tracking 階段動態生成，包含 cross-level 關係
- 重建破壞了這些關係

## 根本問題：ProvenanceTracker 的 Bug

**核心缺陷**: `tracking/provenance_tracker.py` 的 `get_family_hierarchy()` 方法只遍歷 `self.sam2_provenance.keys()`，這導致：

```python
# provenance_tracker.py:207
for sam2_obj_id in self.sam2_provenance.keys():  # ← 只包含部分物件
    parent_id = self.find_parent(sam2_obj_id)
    parent_map[sam2_obj_id] = parent_id
```

**問題**:
1. `sam2_provenance` 只包含在 tracking 過程中**被記錄**的物件
2. 某些物件（如 2043）雖然被 tracking 並存在於 video_segments 和 index.json 中，但**未被記錄到 sam2_provenance**
3. 導致這些物件不出現在 provenance_tree 的 parent_map 中
4. 最終導致 orphan 誤報

## 解決方案

### 短期方案 (臨時修復)

1. **使用 rebuild_provenance_trees.py** - ⚠️ 會失去 cross-level 關係
   ```bash
   python scripts/rebuild_provenance_trees.py --run-dir <run_dir> --levels 2 4 6
   ```
   - 從 index.json 補充缺失的物件到 provenance_tree
   - 但會失去所有 cross-level parent 關係
   - 適合用於檢查單個 level 內的物件

2. **重新運行實驗** - ✅ 推薦
   ```bash
   # 使用修復後的代碼重新運行完整 pipeline
   ./run_experiment.sh --config configs/multiscan/test.yaml
   ```
   - 確保 provenance_tracker 正確記錄所有物件
   - 保留完整的 cross-level 關係

### 長期方案 (根本修復)

#### 1. 修復 ProvenanceTracker

**文件**: `src/my3dis/tracking/provenance_tracker.py`

**修改方案**:
```python
def get_family_hierarchy(self) -> Dict[str, Any]:
    """
    Export complete parent map: {child_sam2_id: parent_sam2_id}.

    BUGFIX: Should include ALL objects that were tracked,
    not just those in sam2_provenance.
    """
    parent_map = {}
    orphans = []

    # ===== BUGFIX START =====
    # Get all tracked objects from multiple sources
    all_sam2_ids = set(self.sam2_provenance.keys())

    # Also include objects from ssam_to_sam2 mapping that were accepted
    for ssam_id, sam2_id in self.ssam_to_sam2.items():
        if sam2_id is not None:  # Accepted objects
            all_sam2_ids.add(sam2_id)

    # Build parent map for ALL tracked objects
    for sam2_obj_id in all_sam2_ids:
        parent_id = self.find_parent(sam2_obj_id)
        parent_map[sam2_obj_id] = parent_id

        # Count orphans (child level but no parent)
        prov = self.sam2_provenance.get(sam2_obj_id, {})
        if prov.get("ssam_parent_id") is not None and parent_id is None:
            orphans.append(sam2_obj_id)
    # ===== BUGFIX END =====

    return {
        "parent_map": parent_map,
        "orphan_count": len(orphans),
        "orphan_ids": orphans,
    }
```

**關鍵改進**:
1. 從多個來源收集所有被 tracking 的物件
2. 確保 parent_map 包含所有物件，不僅僅是 sam2_provenance 中的
3. 需要調查為什麼某些物件不在 sam2_provenance 中

#### 2. 增強 ProvenanceTracker 的物件記錄

**調查重點**:
- sam2_runner.py 中何時調用 `provenance_tracker.register_object()`
- 是否有物件被 tracking 但沒有被 register
- gap filling 或其他機制是否繞過了 provenance_tracker

**可能的修復位置**:
- `tracking/sam2_runner.py`: 確保所有創建的物件都被記錄
- `tracking/level_runner.py`: 在保存 provenance_tree 前驗證完整性

#### 3. 添加驗證機制

**新增工具**: `scripts/validate_provenance_trees.py`

```python
def validate_provenance_completeness(run_dir: str, levels: List[int]) -> Dict[str, Any]:
    """
    驗證 provenance_tree 與 index.json 的一致性

    Returns:
        {
            'level_X': {
                'index_count': int,
                'provenance_count': int,
                'missing_ids': List[int],
                'extra_ids': List[int],
            }
        }
    """
    pass
```

## 已修復的代碼

### 1. family_tree_builder.py

**修改**: 路徑查找邏輯 (line 161-170)

```python
video_segments_candidates = [
    # New format (L-prefixed, directly in level dir) - PRIORITY 1
    run_path / f'level_{level}' / f'video_segments_L{level:02d}.npz',
    # New format variant (in tracking subdir) - PRIORITY 2
    run_path / f'level_{level}' / 'tracking' / f'video_segments_L{level:02d}.npz',
    # Old format (scale-based naming, in tracking subdir) - PRIORITY 3
    run_path / f'level_{level}' / 'tracking' / f'video_segments_scale{mask_scale_ratio}x.npz',
    # Old format variant (directly in level dir) - PRIORITY 4
    run_path / f'level_{level}' / f'video_segments_scale{mask_scale_ratio}x.npz',
]
```

### 2. family_tree_query.py

**修改**: Fallback 路徑查找 (line 176-192)

```python
# If path doesn't exist, try fallback paths (for backward compatibility)
if not archive_path.exists():
    level = obj_info.get('level')
    if level is not None:
        fallback_candidates = [
            # New format (L-prefixed, directly in level dir) - PRIORITY 1
            self.run_dir / f'level_{level}' / f'video_segments_L{level:02d}.npz',
            # New format variant (in tracking subdir) - PRIORITY 2
            self.run_dir / f'level_{level}' / 'tracking' / f'video_segments_L{level:02d}.npz',
            # Old format (scale-based naming) - PRIORITY 3
            self.run_dir / f'level_{level}' / 'tracking' / f'video_segments_scale0.3x.npz',
            self.run_dir / f'level_{level}' / f'video_segments_scale0.3x.npz',
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                archive_path = candidate
                break
```

### 3. rebuild_provenance_trees.py (新增工具)

**文件**: `scripts/rebuild_provenance_trees.py`

**功能**: 從 index.json 補充缺失的物件到 provenance_tree

**限制**: ⚠️ 會失去 cross-level parent 關係

## 測試計劃

### 1. 單元測試
- [ ] 測試 ProvenanceTracker.get_family_hierarchy() 是否包含所有物件
- [ ] 測試 family_tree_builder 的路徑查找邏輯
- [ ] 測試 family_tree_query 的 fallback 機制

### 2. 集成測試
- [ ] 運行完整 pipeline 並驗證 provenance_tree 完整性
- [ ] 檢查 family_tree 中 orphan 數量是否為 0
- [ ] 驗證 cross-level parent 關係是否正確

### 3. 可視化驗證
- [ ] 檢查是否有多 level 的 families
- [ ] 驗證 L2/L4/L6 的包含關係
- [ ] 確認沒有黑色可視化問題

## 後續步驟

1. **立即**: 修復 ProvenanceTracker 的 bug
2. **短期**: 重新運行 test_1105 實驗，驗證修復效果
3. **中期**: 添加自動化驗證工具，在每次運行後檢查 provenance_tree 完整性
4. **長期**: 調查 SSAM 階段的包含關係問題

## 相關文件

- `src/my3dis/tracking/provenance_tracker.py` - 核心 bug 位置
- `src/my3dis/tracking/level_runner.py` - provenance_tree 生成
- `src/my3dis/family_tree_builder.py` - family_tree 構建
- `src/my3dis/family_tree_query.py` - mask 加載
- `scripts/rebuild_provenance_trees.py` - 臨時修復工具
- `scripts/visualize_families.py` - 可視化工具

## 參考資料

- [docs/PROVENANCE_TRACKING.md](docs/PROVENANCE_TRACKING.md) - Provenance tracking 系統說明
- [ORPHAN_FIX_PROGRESS.md](ORPHAN_FIX_PROGRESS.md) - 之前的 orphan 修復記錄
- [CLAUDE.md](CLAUDE.md) - 項目整體文檔
