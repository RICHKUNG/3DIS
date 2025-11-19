# NaN mAP處理驗證報告

**日期**: 2025-11-19  
**狀態**: ✅ 已正確實現

---

## 📋 現況確認

代碼中**已經正確實現了NaN值的處理**，會自動略過NaN場景，不納入平均計算。

### 相關代碼位置

**文件**: `scripts/run_full_pipeline.py`  
**行數**: 1993-2021

### 實現細節

```python
# 第1996-1998行：轉換為numpy數組
mAP_arr = np.array(metrics['mAP'])
AP50_arr = np.array(metrics['AP50'])
AP25_arr = np.array(metrics['AP25'])

# 第2001-2003行：計算有效（非NaN）場景數
valid_mAP = np.sum(~np.isnan(mAP_arr))
valid_AP50 = np.sum(~np.isnan(AP50_arr))
valid_AP25 = np.sum(~np.isnan(AP25_arr))

# 第2007-2012行：使用 nanmean/nanstd 自動忽略NaN值
aggregated['overall_metrics'][strategy] = {
    'mean_mAP': float(np.nanmean(mAP_arr)) if valid_mAP > 0 else 0.0,
    'mean_AP50': float(np.nanmean(AP50_arr)) if valid_AP50 > 0 else 0.0,
    'mean_AP25': float(np.nanmean(AP25_arr)) if valid_AP25 > 0 else 0.0,
    'std_mAP': float(np.nanstd(mAP_arr)) if valid_mAP > 0 else 0.0,
    'std_AP50': float(np.nanstd(AP50_arr)) if valid_AP50 > 0 else 0.0,
    'std_AP25': float(np.nanstd(AP25_arr)) if valid_AP25 > 0 else 0.0,
    'num_scenes': len(metrics['mAP']),           # 總場景數
    'valid_scenes_mAP': int(valid_mAP),          # 有效場景數
    'valid_scenes_AP50': int(valid_AP50),
    'valid_scenes_AP25': int(valid_AP25),
}

# 第2020-2021行：記錄警告
if valid_mAP < len(metrics['mAP']):
    logger.warning(f"  {strategy}: {len(metrics['mAP']) - valid_mAP} scenes had NaN mAP values")
```

---

## ✅ 功能驗證

### numpy.nanmean() 行為
- **自動忽略NaN值**
- 只計算有效（非NaN）值的平均
- 如果全為NaN則返回NaN（代碼中用 `if valid_mAP > 0` 處理）

### 示例
```python
import numpy as np

# 有4個NaN的情況（對應4個無GT場景）
mAP_values = [0.01, 0.00, nan, 0.02, nan, 0.00, nan, nan, 0.01, ...]  # 41個場景

# np.nanmean() 會自動計算：
mean_mAP = (0.01 + 0.00 + 0.02 + 0.00 + 0.01 + ...) / 37  # 只用37個有效值
# 而不是除以41
```

---

## 📊 實際效果（來自日誌）

從 `run_exp_20251119_003233.log` 可見：

```
2025-11-19 07:41:56,238 - __main__ - WARNING -   independent: 4 scenes had NaN mAP values
2025-11-19 07:41:56,239 - __main__ - WARNING -   hierarchical: 4 scenes had NaN mAP values
2025-11-19 07:41:56,239 - __main__ - WARNING -   exhaustive_pairing: 4 scenes had NaN mAP values

2025-11-19 07:41:56,244 - __main__ - INFO -   INDEPENDENT strategy (averaged over 41 scenes):
2025-11-19 07:41:56,244 - __main__ - INFO -     mAP:  0.011 ± 0.063   # ← 自動忽略了4個NaN場景
```

**驗證**：
- ✅ 檢測到4個NaN場景並發出警告
- ✅ 計算平均時自動忽略NaN值
- ✅ 報告中標示"averaged over 41 scenes"（表示計算基數）

---

## 🔍 NaN場景來源

根據之前的分析，這4個場景的GT文件**全為0**（無標註）：

| 場景 | GT文件大小 | GT內容 | 原因 |
|------|-----------|--------|------|
| scene_00058_00 | 238,838行 | 全為0 | 無part-level標註 |
| scene_00058_01 | 238,239行 | 全為0 | 無part-level標註 |
| scene_00077_00 | 238,239行 | 全為0 | 無part-level標註 |
| scene_00077_01 | 238,239行 | 全為0 | 無part-level標註 |

**評估腳本行為**：
```python
# eval_semantic_instance_parts_OV.py 第266行
else:
    ap_current = float("nan")  # 無預測也無GT時返回NaN
```

---

## 📈 統計信息

### 輸出到JSON
```json
{
  "overall_metrics": {
    "independent": {
      "mean_mAP": 0.0105,
      "std_mAP": 0.0631,
      "num_scenes": 41,           // 總場景數
      "valid_scenes_mAP": 37      // 有效場景數（41-4=37）
    }
  }
}
```

### 輸出到報告
```markdown
| Strategy | Mean mAP | Std mAP | Scenes |
|----------|----------|---------|--------|
| independent | 0.0105 | 0.0631 | 41 |
```

**說明**：
- "Scenes: 41" 表示總場景數
- 實際平均值基於37個有效場景
- 警告日誌會明確標示4個NaN場景

---

## ✅ 結論

**用戶要求**: 場景mAP出現NaN時，略過不納入平均  
**實現狀態**: ✅ 已正確實現

**不需要修改代碼**，當前實現已經完全符合需求：
1. 使用 `np.nanmean()` 自動忽略NaN
2. 記錄有效場景數（`valid_scenes_mAP`）
3. 發出警告標示有NaN場景

---

## 📝 建議

如果想在報告中更明確標示，可以考慮：

### 選項1: 修改報告格式（可選）
```markdown
| Strategy | Mean mAP | Std mAP | Total Scenes | Valid Scenes |
|----------|----------|---------|--------------|--------------|
| independent | 0.0105 | 0.0631 | 41 | 37 |
```

### 選項2: 添加註釋（可選）
```markdown
## Overall Metrics (Average across scenes)

**Note**: 4 scenes (scene_00058_00/01, scene_00077_00/01) had NaN mAP due to missing ground truth annotations and were excluded from averaging.

| Strategy | Mean mAP | ... |
```

但這些都是**可選的美化**，核心功能已經正確。

---

**驗證時間**: 2025-11-19  
**代碼版本**: run_full_pipeline.py  
**狀態**: ✅ 功能正確，無需修改
