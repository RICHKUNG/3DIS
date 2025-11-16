# mAP 表現診斷報告 (2025-11-16)

## 問題總結
實驗 `test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01` 的 mAP 為 **0.000**。

## 已修復的問題

### 1. Search3D 格式錯誤 ✅ **已修復**
**問題**: 預測文件使用 5 位數字格式（例如 `96002`），而 GT 使用 8 位數字格式（例如 `96002001`）

**修復**: `src/my3dis/aggregation/search3d_formatter.py:130-177`
- 實現了實例編號生成
- 格式現在為 `composite_id * 1000 + instance_number`
- 驗證通過：預測現在包含 `11003002`, `86002003` 等 8 位 ID

### 2. 查詢過濾未啟用 ✅ **已修復**
**問題**: `evaluation.gt_path` 設為 null，導致查詢沒有根據 GT labels 過濾

**修復**: `scripts/run_full_pipeline.py:540-549`
- 添加了對 `evaluation.gt_paths.obj_part` 的支持
- 查詢從 47 個減少到 10 個（僅包含場景中的 GT labels）
- 驗證通過：日誌顯示 "restricting queries to 10/47 GT label_ids"

### 3. Label 匹配驗證 ✅ **已確認**
**預測 labels** (來自 independent strategy):
```
11003002, 15002002, 86002003, 86002005, 86003003, 86003005, 86009002, 86009005
```

**GT labels** (composite IDs):
```
11001, 11002, 11003, 11014, 11024, 15002, 19002, 30002, 86002, 86003, 86009
```

**匹配情況**: ✅ 8 個預測 labels 都對應 GT 中的合法 composite IDs

## 當前問題診斷

### mAP 仍為 0.000 的可能原因

#### 原因 1: GT 中的無效 ID (11001)
```python
# 從 multiscan_search3d_constants.py:
VALID_JOINT_SEMANTIC_IDS = [4008, 134020, 11024, 25002, ..., 11003, ...]  # 47 個

# GT 中出現但不在 VALID_JOINT_SEMANTIC_IDS 中:
11001  # ❌ 無效！
```

**影響**: 評估器可能跳過包含無效 ID 的 GT 實例

#### 原因 2: Mask 質量問題
**統計** (來自 pipeline_results.json):
- **Independent**: 22 predictions, 5,680/76,675 labeled points (7.4%)
- **Hierarchical**: 124 predictions, 29,910/76,675 labeled points (39.0%)
- **Exhaustive**: 41 predictions, 26,698/76,675 labeled points (34.8%)

**問題**: Independent strategy 的覆蓋率極低 (7.4%)，可能導致與 GT 的 IoU 不足

#### 原因 3: 評估閾值過嚴
從 `eval_semantic_instance_parts_OV.py:33`:
```python
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)  # 0.25, 0.50, ..., 0.90
opt["min_region_sizes"] = np.array([10])
```

**可能問題**: 如果預測 mask 與 GT mask 的 IoU < 0.25，則不計入任何 AP

#### 原因 4: 實例匹配問題
GT 包含每個 composite ID 的多個實例：
```
11001001, 11002001, 11002002, 11002003, 11002004, ...
```

但從預測的 label 分佈來看，某些 composite IDs 缺少實例 1:
```
預測: 11003002  (缺少 11003001)
預測: 15002002  (缺少 15002001)
預測: 86002003, 86002005  (缺少 86002001, 86002002, 86002004)
```

## 下一步建議

### 立即行動
1. **檢查 GT 的 11001 問題**
   - 驗證這是否是標註錯誤
   - 查看是否影響評估

2. **分析 mask IoU**
   - 計算預測 mask 與 GT mask 的實際 IoU
   - 檢查是否有任何匹配 > 0.25

3. **可視化對比**
   - 渲染幾個預測和 GT 的 masks
   - 確認空間重疊情況

### 深入調查
4. **檢查評估器邏輯**
   - 閱讀 `eval_semantic_instance_parts_OV.py` 的匹配算法
   - 確認實例編號是否影響匹配

5. **對比其他場景**
   - 運行 scene_00005_01 來對比結果
   - 查看是否是場景特定問題

6. **調整推理參數**
   - 降低 `min_similarity` 閾值
   - 增加 `top_k_per_level`
   - 調整 pairing 的 `containment_threshold`

## 代碼修改總結

### 文件 1: `src/my3dis/aggregation/search3d_formatter.py`
**行數**: 130-177
**更改**: 添加實例編號生成，從 `label_id` 格式改為 `label_id * 1000 + instance_num`
**影響**: 修復了預測格式，使其與 GT 格式匹配

### 文件 2: `scripts/run_full_pipeline.py`
**行數**: 540-549
**更改**: 添加對 `evaluation.gt_paths.obj_part` 的支持以啟用查詢過濾
**影響**: 查詢從 47 個減少到 10 個，僅包含場景實際的 GT labels

## 參考資料
- Search3D 常量: `data/search3d_gt/multiscan_data_search3d/multiscan_search3d_constants.py`
- 評估器: `eval/search3d/eval_semantic_instance_parts_OV.py`
- 配置: `configs/inference/full_pipeline.yaml`
- 日誌: `logs/eval/run_exp_20251116_172850.log`
