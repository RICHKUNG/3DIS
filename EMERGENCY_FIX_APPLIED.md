# 緊急修復已應用

**時間**: 2025-11-16 22:35
**目標**: 提升所有策略的 mAP，特別是 Independent 和 Exhaustive_Pairing

---

## 🚨 核心問題發現

### 問題診斷
```
Hierarchical: 99 candidate pairs → NMS → 23 predictions → mAP 0.000
```

**根因**:
1. ⭐ **Pairing 幾何約束過嚴** → 配對成功率極低
2. ⭐ **NMS 過度抑制** → 丟失太多有效預測
3. Query 覆蓋不完整 (10/11 類別)

---

## ✅ 應用的修復

### 修復 1: 移除幾何約束 ⭐⭐⭐⭐⭐

```yaml
# 之前
pairing:
  containment_threshold: 0.001  # 要求 part 0.1% 在 object 內
  scale_range: [0.0001, 20.0]

# 現在
pairing:
  containment_threshold: 0.0     # ✅ 完全移除包含要求
  scale_range: [0.00001, 100.0]  # ✅ 幾乎無尺寸限制
```

**預期效果**:
- Pair 創建數: 99 → **2000-5000**
- 有效配對率: ~1% → **10-30%**

---

### 修復 2: 啟用 Soft NMS ⭐⭐⭐⭐

```yaml
# 之前
nms:
  iou_threshold: 0.25
  use_soft_nms: false
  keep_top_k: 300

# 現在
nms:
  iou_threshold: 0.15      # ✅ 降低閾值
  use_soft_nms: true       # ✅ 啟用 soft NMS
  soft_nms_sigma: 0.3      # ✅ 激進保留
  keep_top_k: 500          # ✅ 保留更多
```

**預期效果**:
- NMS 保留率: 23/99 (23%) → **100-200/2000 (5-10%)**
- 預測數: 23 → **100-200**

---

### 修復 3: 啟用所有策略測試 ⭐⭐⭐

```yaml
# 之前
test_all_strategies: false
strategy: hierarchical

# 現在
test_all_strategies: true  # ✅ 測試所有策略
strategies_to_test: [independent, hierarchical, exhaustive_pairing]
```

**預期效果**:
- 可以對比三種策略的表現
- 為每種策略找到最優配置

---

## 📊 預期改善

### Hierarchical 策略

| 指標 | 修復前 | 預期修復後 | 改善幅度 |
|------|--------|-----------|----------|
| Candidate pairs | 99 | **2000-5000** | +2000%-5000% |
| Predictions (after NMS) | 23 | **100-200** | +335%-770% |
| mAP | 0.000 | **0.008-0.020** | 突破 0 |
| IoU >= 0.25 | 1 | **5-10** | +400%-900% |

---

### Independent 策略

**之前的問題**:
- 搜索空間: 111 objects × 155 parts = 17,205 組合
- 幾何約束後: ~10-20 有效 pairs
- 預測數: ~13

**修復後預期**:
- 有效 pairs: **1000-3000** (提升 100-200 倍)
- NMS 後: **50-150 predictions**
- **mAP: 0.005-0.015** （從 0.000 突破）

---

### Exhaustive_Pairing 策略

**之前的問題**:
- 嘗試 6 個 level pairs，但大多數配對數極少
- 最終只有 ~22 predictions

**修復後預期**:
- 每個 level pair: **300-800 candidate pairs**
- 聚合所有 pairs: **1000-3000**
- NMS 後: **60-120 predictions**
- **mAP: 0.003-0.012**

---

## 🎯 成功指標

### 最小成功標準
- [ ] mAP > 0.000 (所有策略)
- [ ] Hierarchical mAP >= 0.005
- [ ] Independent mAP >= 0.003

### 理想成功標準
- [ ] Hierarchical mAP >= 0.015
- [ ] Independent mAP >= 0.010
- [ ] Exhaustive mAP >= 0.008
- [ ] IoU >= 0.25 匹配數 >= 5

---

## 🚀 測試命令

```bash
# 運行測試
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation

# 監控日誌
tail -f logs/eval/run_exp_$(date +%Y%m%d)_*.log | grep -E "mAP|pairs|predictions"
```

---

## 📝 分析腳本

測試完成後運行診斷：

```bash
# 對比所有策略
python /tmp/diagnose_all_strategies.py

# 分析配對統計
grep "Created.*pairs" logs/eval/run_exp_*.log | tail -100

# 檢查 mAP
grep -A 5 "strategy.*averaged" logs/eval/run_exp_*.log
```

---

## ⚠️ 可能的副作用

### 風險 1: 誤檢增加
**現象**: 預測數過多（>500）
**緩解**: Soft NMS 會自然降低低質量預測的分數

### 風險 2: 計算時間增加
**現象**: Pairing 階段變慢
**原因**: 配對組合數從 ~100 增加到 ~2000
**預期影響**: 每個 query 從 0.1s → 0.5-1s（可接受）

### 風險 3: 記憶體使用增加
**現象**: OOM (unlikely)
**原因**: 需要存儲更多 candidate pairs
**預期峰值**: ~500MB → ~2GB（GPU 顯存充足）

---

## 🔄 如果效果不佳

### Plan B: 漸進式放寬約束

如果完全移除約束導致太多噪音：

```yaml
# 中等寬鬆版本
pairing:
  containment_threshold: 0.0001  # 幾乎沒有（0.01%）
  scale_range: [0.0001, 50.0]

nms:
  iou_threshold: 0.20  # 稍微嚴格
  use_soft_nms: true
  soft_nms_sigma: 0.5  # 較大 sigma
```

### Plan C: 為不同策略使用不同約束

```yaml
# Hierarchical: 可以稍微嚴格（有 family tree 引導）
hierarchical_pairing:
  containment_threshold: 0.0001

# Independent: 需要極寬鬆（無 family tree）
independent_pairing:
  containment_threshold: 0.0
```

---

## 📊 備份與恢復

### 備份位置
```
configs/inference/full_pipeline.yaml.backup_phase1_20251116_HHMMSS
```

### 恢復命令
```bash
# 如果需要回滾
cp configs/inference/full_pipeline.yaml.backup_phase1_20251116_HHMMSS \
   configs/inference/full_pipeline.yaml
```

---

## 🎉 預期時間線

| 階段 | 時間 | 狀態 |
|------|------|------|
| 配置修改 | 5 分鐘 | ✅ 完成 |
| 運行測試 | 10 分鐘 | ⏳ 待執行 |
| 分析結果 | 10 分鐘 | ⏳ 待執行 |
| **總計** | **25 分鐘** | |

---

## 📋 下一步

1. ✅ 應用配置修改
2. ⏳ 運行測試
3. ⏳ 檢查 mAP 是否改善
4. ⏳ 如果成功，調優參數
5. ⏳ 如果失敗，執行 Plan B

**當前狀態**: ✅ 配置已修改，等待測試執行
