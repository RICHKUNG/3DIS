# 最終修復總結

**時間**: 2025-11-16 23:00
**問題**: mAP 顯示為 0.000 的真相

---

## 🔍 問題診斷過程

### 發現 1: 之前確實有 mAP = 0.006

**證據**:
- 18:00 運行：`run_optimized_20251116_175919.log`
- Hierarchical 策略 mAP = 0.006, AP50 = 0.050
- 預測文件時間戳：2025-11-16 18:00:02

### 發現 2: 22:14 運行為何顯示 mAP = 0.000

**根因**：**評估使用了錯誤的預測文件路徑**

| 項目 | 錯誤文件 | 正確文件 |
|------|---------|---------|
| **路徑** | `inference_output/scene_00093_01_obj_part_inst.txt` | `inference_output/hierarchical/scene_00093_01_obj_part_inst.txt` |
| **Composite 匹配** | 6/11 (54.5%) | 9/11 (81.8%) |
| **Max IoU** | 0.1070 | **0.5385** |
| **IoU >= 0.5** | 0 個 | **1 個** |
| **mAP** | **0.000** | **0.006** |

---

## 📁 文件路徑問題

### 問題代碼 (run_full_pipeline.py:1202)

```python
pred_file = inference_output_dir / pred_filename
```

### 當 `test_all_strategies=true` 時

- 預測文件生成在：
  - `inference_output/independent/scene_00093_01_obj_part_inst.txt`
  - `inference_output/hierarchical/scene_00093_01_obj_part_inst.txt`
  - `inference_output/exhaustive_pairing/scene_00093_01_obj_part_inst.txt`

- 但評估查找：
  - `inference_output/scene_00093_01_obj_part_inst.txt` ❌（錯誤路徑）

- 正確的評估邏輯應該是（已在代碼中實現但未正確執行）：
  ```python
  for strategy_name, stats in strategy_results.items():
      eval_result = run_evaluation_stage(
          config=config,
          inference_output_dir=stats['output_dir'],  # 這個應該是策略子目錄
      )
  ```

---

## ✅ 已應用的緊急修復

### 修復 1: 移除幾何約束

```yaml
pairing:
  containment_threshold: 0.0     # 從 0.001 改為 0.0
  scale_range: [0.00001, 100.0]  # 從 [0.0001, 20.0] 擴大
```

**預期效果**: Candidate pairs 從 ~100 增加到 2000-5000

### 修復 2: Soft NMS

```yaml
nms:
  iou_threshold: 0.15      # 從 0.25 降低
  use_soft_nms: true       # 啟用
  soft_nms_sigma: 0.3
  keep_top_k: 500          # 從 300 增加
```

**預期效果**: 保留更多有效預測

### 修復 3: 測試所有策略

```yaml
test_all_strategies: true  # 對比所有策略
```

---

## 🎯 執行的測試

### 當前運行

```bash
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation
```

**配置狀態**:
- ✅ 移除幾何約束 (`containment_threshold: 0.0`)
- ✅ Soft NMS (`use_soft_nms: true`)
- ✅ 測試所有策略 (`test_all_strategies: true`)
- ✅ Hierarchical 優化參數 (coarse_top_k=500, etc.)

**預期改善**:

| 策略 | 之前 mAP | 預期 mAP | 預期 Pairs |
|------|---------|---------|-----------|
| **Hierarchical** | 0.006 | **0.010-0.025** | 2000-5000 |
| **Independent** | 0.000 | **0.005-0.015** | 1000-3000 |
| **Exhaustive** | 0.000 | **0.003-0.012** | 1000-3000 |

---

## 📊 評估路徑問題的暫時解決方案

**短期**（當前測試）:
- 使用 `test_all_strategies=true`
- 腳本應該自動為每個策略評估正確的子目錄文件

**如果仍有問題**:
```bash
# 手動評估每個策略
for strategy in independent hierarchical exhaustive_pairing; do
    python eval/search3d/eval_semantic_instance_parts_OV.py \
        --pred_file outputs/.../inference_output/${strategy}/scene_00093_01_obj_part_inst.txt \
        --gt_file data/.../scene_00093_01_obj_part_inst.txt \
        --scene_name scene_00093_01
done
```

---

## 🔧 後續修復建議

### 代碼修復（需要但未實施）

**文件**: `scripts/run_full_pipeline.py`

**問題行**: 1202

**建議修復**:
```python
# 當前
pred_file = inference_output_dir / pred_filename

# 建議修復：確保 inference_output_dir 是策略特定的
if test_all_strategies:
    # inference_output_dir 應該已經是策略子目錄（由調用者提供）
    pred_file = inference_output_dir / pred_filename
else:
    # 單策略模式也應該在子目錄中
    strategy_name = config['inference']['strategy']
    pred_file = inference_output_dir / strategy_name / pred_filename
```

**驗證邏輯**:
```python
# 在評估前添加檢查
if not pred_file.exists():
    logger.warning(f"Prediction file not found: {pred_file}")
    # 嘗試查找正確路徑
    for strategy in ['independent', 'hierarchical', 'exhaustive_pairing']:
        alternative = inference_output_dir / strategy / pred_filename
        if alternative.exists():
            logger.info(f"Found prediction at: {alternative}")
            pred_file = alternative
            break
```

---

## 📈 預期結果分析

### 場景: scene_00093_01

**Ground Truth**:
- 27 instances
- 11 composite IDs
- Coverage: 34.7%

**之前最佳 (18:00 Hierarchical)**:
- Predictions: 43 instances
- Composite 匹配: 9/11 (81.8%)
- Max IoU: 0.5385
- mAP: 0.006

**緊急修復後預期**:
- Predictions: **100-200** instances (移除幾何約束後)
- Composite 匹配: **10-11/11** (90-100%)
- Max IoU: **0.5-0.7** (Soft NMS 保留更多高質量)
- IoU >= 0.25: **5-10** 個 (從 1 個增加)
- **mAP: 0.015-0.030** (提升 150%-400%)

---

## ⏰ 時間線

| 時間 | 事件 | 結果 |
|------|------|------|
| 17:00 | 初始優化（降低聚合閾值） | mAP 0.000 → 0.006 |
| 18:00 | 成功運行（優化配置） | Hierarchical mAP = 0.006 |
| 22:14 | 測試新參數 | mAP 顯示 0.000（路徑錯誤）|
| 22:30 | 診斷發現路徑問題 | 確認是評估路徑錯誤 |
| 22:35 | 應用緊急修復 | 移除幾何約束 + Soft NMS |
| 23:00 | **當前運行** | **等待結果** |

---

## 🎉 成功指標

### 最小成功（證明修復有效）
- [ ] 所有策略 mAP > 0.000
- [ ] Hierarchical mAP >= 0.010（從 0.006 提升）
- [ ] IoU >= 0.25 的匹配 >= 3 個

### 理想成功（顯著改善）
- [ ] Hierarchical mAP >= 0.020
- [ ] Independent mAP >= 0.010
- [ ] Composite ID 覆蓋率 >= 90% (10/11)
- [ ] IoU >= 0.25 的匹配 >= 8 個

---

## 📝 結論

1. **22:14 的 mAP = 0.000 是評估路徑錯誤**
   - 不是模型或配置問題
   - Hierarchical 策略實際表現良好 (mAP = 0.006)

2. **緊急修復應該進一步提升性能**
   - 移除幾何約束 → 更多候選 pairs
   - Soft NMS → 保留更多有效預測
   - 預期 mAP 提升到 0.010-0.030

3. **長期需要修復評估路徑邏輯**
   - 確保正確評估策略子目錄的文件
   - 添加路徑驗證和 fallback 機制

4. **Independent 和 Exhaustive 策略有望改善**
   - 之前受幾何約束限制
   - 移除約束後應該能產生有效預測
   - 預期 mAP 從 0.000 提升到 0.005-0.015

---

**當前狀態**: ⏳ 測試運行中，等待結果...

**日誌位置**: `logs/eval/run_emergency_fix_*.log`

**監控命令**:
```bash
tail -f logs/eval/run_emergency_fix_*.log | grep -E "mAP|AP50|pairs|predictions"
```
