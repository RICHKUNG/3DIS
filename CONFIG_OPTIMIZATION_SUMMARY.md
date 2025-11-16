# 配置優化總結

**日期**: 2025-11-16 18:30
**文件**: `/media/Pluto/richkung/My3DIS/configs/inference/full_pipeline.yaml`

---

## 🎯 核心變更

### 1. 策略選擇（最重要）

```yaml
# 之前：測試所有策略
test_all_strategies: true
strategy: independent

# 現在：只使用最佳策略
test_all_strategies: false  # ✅ 禁用多策略測試
strategy: hierarchical      # ✅ 使用 Hierarchical（唯一有效的策略）
```

**理由**:
- Independent 和 Exhaustive_Pairing 在當前參數下表現差（mAP = 0.000）
- Hierarchical 是唯一達到 mAP > 0 的策略（mAP = 0.006）
- 節省計算時間（不再測試無效策略）

---

## 📊 參數優化對比

### Hierarchical 策略（主要優化）

| 參數 | 優化前 | 優化後 | 變化 | 預期效果 |
|------|--------|--------|------|----------|
| `coarse_top_k` | 200 | **500** | +150% | 更多粗定位候選 |
| `object_top_k` | 200 | **500** | +150% | 更大 object 池 |
| `part_top_k` | 100 | **200** | +100% | 每個 object 更多 part 候選 |
| `coarse_threshold` | 0.01 | **0.005** | -50% | 更寬鬆的篩選 |
| `refinement_threshold` | 0.01 | **0.005** | -50% | 更寬鬆的篩選 |

**預期改善**:
- 預測數：110 → **150-200**
- mAP: 0.006 → **0.010-0.020** (提升 67%-233%)
- IoU >= 0.25 的匹配：1 個 → **3-5 個**

---

### Independent 策略（備用優化）

| 參數 | 優化前 | 優化後 | 變化 | 預期效果 |
|------|--------|--------|------|----------|
| `top_k_per_level` | 200 | **500** | +150% | 與全局設置一致 |
| `retrieval_threshold` | 0.05 | **0.01** | -80% | 與 Hierarchical 一致 |
| `min_retrieval_threshold` | 0.005 | **0.001** | -80% | 更寬鬆 fallback |
| `threshold_backoff_step` | 0.01 | **0.005** | -50% | 更細緻的閾值調整 |

**預期改善**:
- 預測數：13 → **30-50**
- mAP: 0.000 → **0.001-0.003**（可能突破 0）

---

### Exhaustive_Pairing 策略（備用優化）

| 參數 | 優化前 | 優化後 | 變化 | 預期效果 |
|------|--------|--------|------|----------|
| `top_k_per_level` | 200 | **500** | +150% | 與全局設置一致 |
| `retrieval_threshold` | 0.05 | **0.01** | -80% | 與 Hierarchical 一致 |
| `min_retrieval_threshold` | 0.005 | **0.001** | -80% | 更寬鬆 fallback |
| `threshold_backoff_step` | 0.01 | **0.005** | -50% | 更細緻的閾值調整 |
| `selection_metric` | max | **sum** | 變更 | 聚合所有組合的分數 |
| `return_all_pairs` | false | **true** | 變更 | 返回所有 level 組合的結果 |

**預期改善**:
- 預測數：22 → **50-80**
- mAP: 0.000 → **0.002-0.004**（可能突破 0）

---

## 🔄 優化策略說明

### 為何提高 `top_k` 參數？

**問題**:
- 之前各策略的 `top_k` (200) 小於全局設置 (500)
- 在小場景中影響不大，但在大場景中會嚴重限制候選數

**解決**:
- 統一為 500，與全局設置一致
- 確保所有策略都能獲得足夠的候選

---

### 為何降低閾值？

**問題**:
- Independent 和 Exhaustive_Pairing 的 `retrieval_threshold` (0.05) 太高
- Hierarchical 的閾值 (0.01) 更低，這是其成功的關鍵之一

**解決**:
- 統一降低到 0.01（Hierarchical 進一步降到 0.005）
- 更容易找到候選，提高召回率

---

### 為何 Exhaustive_Pairing 改為 `return_all_pairs: true`？

**問題**:
- 之前只返回最佳 level pair 的結果
- 如果最佳 pair 恰好是不合理的（例如 L2,L2），結果會很差

**解決**:
- 返回所有 level 組合的結果，再統一 NMS
- 相當於變成"多路徑聚合"策略
- `selection_metric: sum` 確保所有組合的分數都被考慮

---

## 📁 備份文件

**原始配置已備份至**:
```
configs/inference/full_pipeline.yaml.backup_20251116_HHMMSS
```

可以隨時恢復：
```bash
cp configs/inference/full_pipeline.yaml.backup_20251116_HHMMSS \
   configs/inference/full_pipeline.yaml
```

---

## 🚀 使用方法

### 快速測試（單場景）
```bash
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

### 多場景測試
配置已設置為 `multi_scene_mode: true`，會自動處理所有場景：
```bash
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

### 如果想測試所有策略（對比實驗）
臨時修改配置：
```yaml
inference:
  test_all_strategies: true  # 改為 true
```

---

## 🎯 預期結果

### 短期（當前優化）

**Scene_00093_01 預期表現**:

| 指標 | 優化前 (Hierarchical) | 優化後 (Hierarchical) | 改善 |
|------|---------------------|---------------------|------|
| **mAP** | 0.006 | 0.010-0.020 | +67%-233% |
| **AP@50** | 0.050 | 0.080-0.150 | +60%-200% |
| **預測數** | 110 | 150-200 | +36%-82% |
| **IoU >= 0.25** | 1 個 | 3-5 個 | +200%-400% |
| **IoU >= 0.50** | 1 個 | 1-3 個 | +0%-200% |

---

### 中期（進一步優化方向）

#### 1. 聚合參數更激進（如果短期結果良好）
```yaml
aggregation:
  iou_threshold: 0.5 → 0.3      # 更激進的合併
  area_fraction: 0.001 → 0.0005 # 保留更小區域
```

**預期**: mAP 0.010-0.020 → **0.020-0.030**

#### 2. NMS 參數調整
```yaml
nms:
  iou_threshold: 0.25 → 0.20    # 保留更多預測
  keep_top_k: 300 → 500         # 增加輸出數量
```

**預期**: 提升 AP@25，增加低 IoU 匹配數量

#### 3. 特徵提取優化
```yaml
aggregation:
  projection:
    resize_ratio: 0.3 → 0.5     # 更高分辨率投影
```

**預期**: 提升 mask 質量，增加 IoU

---

## 📋 驗證清單

### 階段 1：配置驗證（5 分鐘）
- [x] 備份原始配置
- [x] 更新策略選擇為 Hierarchical
- [x] 更新 Hierarchical 參數
- [x] 更新 Independent 參數（備用）
- [x] 更新 Exhaustive_Pairing 參數（備用）
- [ ] 運行測試（等待執行）

### 階段 2：結果分析（1 小時）
- [ ] 檢查預測數量是否增加
- [ ] 檢查 mAP 是否提升
- [ ] 分析新增的高質量匹配
- [ ] 可視化成功案例

### 階段 3：參數微調（2 小時）
- [ ] 如果 mAP >= 0.015，測試更激進的聚合參數
- [ ] 如果預測數過多（>300），適當提高閾值
- [ ] 如果 IoU 仍然偏低，檢查 mask 質量

---

## 📝 配置變更日誌

| 時間 | 版本 | 主要變更 | mAP | 備註 |
|------|------|----------|-----|------|
| 2025-11-16 17:00 | v1.0 | 初始優化（降低聚合閾值） | 0.006 | 從 0.000 突破 |
| 2025-11-16 18:30 | v2.0 | 策略優化（Hierarchical + 激進參數） | 預測 0.010-0.020 | 當前版本 |
| 未來 | v3.0 | 聚合參數更激進 | 預測 0.020-0.030 | 計劃中 |

---

## 🎉 總結

### ✅ 已完成
1. ✅ 確認特徵來源（SigLIP 提取，非隨機）
2. ✅ 分析策略失敗原因（詳見 `STRATEGY_FAILURE_ANALYSIS.md`）
3. ✅ 應用短期優化（更激進參數）
4. ✅ 更新配置文件為最佳設定

### 🎯 核心優化
- **策略**: Independent → **Hierarchical**（唯一有效策略）
- **候選數**: top_k 200 → **500**（+150%）
- **閾值**: 0.01 → **0.005**（-50%，更寬鬆）
- **預期 mAP**: 0.006 → **0.010-0.020**（+67%-233%）

### 🚀 下一步
- 運行新配置測試
- 分析結果並決定是否進一步激進
- 如果效果好，推廣到多場景實驗
