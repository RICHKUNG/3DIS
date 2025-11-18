# Phase 3 代碼審查與實驗設計總結

**日期**: 2025-11-17
**任務**: 審查代碼,復刻 SigLIP baseline (AP@25=0.062),設計實驗突破 mAP=0

---

## 📋 執行摘要

### 關鍵發現

1. ✅ **代碼審查無誤**:
   - `scale_semantic_score=300` 和 `apply_softmax=True` 在所有路徑中正確使用
   - `retrieval.py`, `hierarchical.py`, `config.py` 的參數傳遞一致
   - 沒有導致性能下降的代碼 bug

2. ✅ **當前 Baseline 確認**:
   - Phase 2-A: **AP@25=0.062** (使用 OpenAI CLIP + 加權幀聚合)
   - 這是目前的最佳成績
   - 瓶頸在 Mask Quality,不在特徵提取

3. ✅ **實驗設計完成**:
   - 創建了 3 個實驗配置 (A1, B1, B2)
   - 測試方向: 閾值調整 + scale 調整
   - 預期突破 AP@50 = 0

---

## 🔍 詳細分析

### 1. scale_semantic_score 和 apply_softmax 使用情況

**審查結果**: ✅ 所有路徑正確

**代碼位置**:
- `retrieval.py:38`: 函數默認值 `scale_semantic_score=300.0, apply_softmax=True`
- `retrieval.py:140`: `retrieve_single_level()` 正確傳遞參數
- `hierarchical.py:279, 398, 587, 712`: 四處直接調用都正確傳遞 `scale_semantic_score=self.retriever.scale_semantic_score`
- `config.py:17-18, 86-87`: 配置類默認值正確

**結論**: 用戶提到的"之前是會乘上300然後softmax"在當前代碼中**已經正確實現**,不存在參數傳遞錯誤。

---

### 2. Phase 2-A Baseline 分析

**配置**: `configs/inference/phase2_weighted_agg.yaml`

**關鍵參數**:
```yaml
retrieval:
  min_similarity: 0.005  # 非常寬鬆
  # scale_semantic_score: 300  # 使用默認值
  # apply_softmax: true        # 使用默認值

hierarchical:
  coarse_top_k: 200
  object_top_k: 200
  part_top_k: 100
  coarse_threshold: 0.01      # 非常寬鬆
  refinement_threshold: 0.01  # 非常寬鬆
  # scale_semantic_score: 300  # 使用默認值
  # apply_softmax: true        # 使用默認值
```

**結果**:
- Predictions: 211
- **AP@25: 0.062** ⭐ (目前最佳)
- AP@50: 0.000
- mAP: 0.000

**問題診斷**:
- ❌ **不在特徵**: CLIP 已改善語義匹配
- ❌ **不在聚合**: 加權聚合已提升性能
- ✅ **在 Mask IoU**: 預測的 mask 與 GT 重疊度 < 0.5

---

### 3. 改進方向

#### 方向 A: 收緊閾值 (優先級 P0)

**假設**: 當前閾值過寬鬆,大量低質量候選降低 precision

**實驗 A1**: `configs/inference/phase3_exp_a1_tighter_thresholds.yaml`
```yaml
retrieval:
  min_similarity: 0.01  # 0.005 → 0.01 (2x 嚴格)
hierarchical:
  coarse_top_k: 100        # 200 → 100
  object_top_k: 100        # 200 → 100
  part_top_k: 50           # 100 → 50
  coarse_threshold: 0.02   # 0.01 → 0.02
  refinement_threshold: 0.02  # 0.01 → 0.02
```

**預期**:
- AP@25: +0.003 to +0.013
- AP@50: 0.000 → 0.005-0.015 (突破 0!)
- Precision ↑, Recall ↓

---

#### 方向 B: 調整 Scale (優先級 P1, P2)

**假設**: scale=300 過大,導致 softmax 過度 peaked

**理論分析**:
```python
# scale=300 時:
cosine_sim = [0.3, 0.29, 0.28]
scaled = [90, 87, 84]
softmax(scaled) ≈ [0.95, 0.04, 0.01]  # 極度 peaked,幾乎只選第一個

# scale=100 時:
scaled = [30, 29, 28]
softmax(scaled) ≈ [0.58, 0.25, 0.17]  # 更平滑,考慮前三個

# scale=50 時:
scaled = [15, 14.5, 14]
softmax(scaled) ≈ [0.38, 0.33, 0.29]  # 極度平滑,幾乎均等
```

**實驗 B1**: `configs/inference/phase3_exp_b1_scale100.yaml`
- scale: 300 → 100
- 預期: AP@25 不確定 (±0.015)

**實驗 B2**: `configs/inference/phase3_exp_b2_scale50.yaml`
- scale: 300 → 50
- 預期: 可能過度平滑,風險較高

---

## 🚀 實驗執行計劃

### 準備步驟

1. **創建 symlinks** (復用 aggregation 輸出):
```bash
mkdir -p outputs/experiments/phase3_exp_a1_tighter/scene_00093_01
ln -s /media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/aggregation_output \
      outputs/experiments/phase3_exp_a1_tighter/scene_00093_01/aggregation_output

mkdir -p outputs/experiments/phase3_exp_b1_scale100/scene_00093_01
ln -s /media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/aggregation_output \
      outputs/experiments/phase3_exp_b1_scale100/scene_00093_01/aggregation_output

mkdir -p outputs/experiments/phase3_exp_b2_scale50/scene_00093_01
ln -s /media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/aggregation_output \
      outputs/experiments/phase3_exp_b2_scale50/scene_00093_01/aggregation_output
```

2. **激活環境**:
```bash
conda activate 3Dsiglip
cd /media/Pluto/richkung/My3DIS
```

---

### 運行實驗

#### 實驗 A1 (最優先)
```bash
mkdir -p logs/phase3

PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/phase3_exp_a1_tighter_thresholds.yaml \
  2>&1 | tee logs/phase3/exp_a1_tighter.log

# 快速查看結果
tail -50 logs/phase3/exp_a1_tighter.log | grep -E "mAP|AP@"
```

#### 實驗 B1
```bash
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/phase3_exp_b1_scale100.yaml \
  2>&1 | tee logs/phase3/exp_b1_scale100.log

tail -50 logs/phase3/exp_b1_scale100.log | grep -E "mAP|AP@"
```

#### 實驗 B2
```bash
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/phase3_exp_b2_scale50.yaml \
  2>&1 | tee logs/phase3/exp_b2_scale50.log

tail -50 logs/phase3/exp_b2_scale50.log | grep -E "mAP|AP@"
```

---

### 結果對比

```bash
echo "=== Baseline (Phase 2-A) ==="
echo "mAP: 0.000, AP@50: 0.000, AP@25: 0.062, Predictions: 211"

echo -e "\n=== Exp A1 ==="
tail -20 logs/phase3/exp_a1_tighter.log | grep -E "mAP|AP@|num_predictions"

echo -e "\n=== Exp B1 ==="
tail -20 logs/phase3/exp_b1_scale100.log | grep -E "mAP|AP@|num_predictions"

echo -e "\n=== Exp B2 ==="
tail -20 logs/phase3/exp_b2_scale50.log | grep -E "mAP|AP@|num_predictions"
```

---

## 📊 成功標準

### 最低標準 (繼續優化)
```
任一實驗:
  AP@25 > 0.070  (+0.008)
  或
  AP@50 > 0.000  (突破 0)
```

### 中等標準 (Phase 3 成功)
```
最佳實驗:
  AP@25 > 0.080  (+0.018)
  AP@50 > 0.010
  mAP > 0.000
```

### 理想標準 (重大突破)
```
最佳實驗:
  AP@25 > 0.100  (+0.038)
  AP@50 > 0.020
  mAP > 0.005
```

---

## 📝 記錄模板

創建 `PHASE3_RESULTS.md` 記錄結果:

```markdown
# Phase 3 實驗結果

## Baseline (Phase 2-A)
- mAP: 0.000
- AP@50: 0.000
- AP@25: 0.062
- Predictions: 211

## 實驗 A1 (Tighter Thresholds)
**日期**:
**日誌**: logs/phase3/exp_a1_tighter.log

| Metric | Value | Δ from Baseline |
|--------|-------|-----------------|
| mAP    |       |                 |
| AP@50  |       |                 |
| AP@25  |       |                 |
| Predictions |  |                 |

**分析**:
-

## 實驗 B1 (Scale=100)
...

## 實驗 B2 (Scale=50)
...

## 最佳配置
-

## 下一步
-
```

---

## 🎯 決策樹

```
運行 A1, B1, B2
    ↓
有實驗達到最低標準?
    ├─ Yes → 選最佳配置
    │         ├─ 達到理想標準? → 多場景測試
    │         └─ 未達理想標準 → 繼續 Phase 3-B (組合優化)
    └─ No  → 考慮:
              1. Phase 4: Fine-tune CLIP
              2. 嘗試其他策略 (Independent, Exhaustive)
              3. 優化 Combined Query 構建方式
```

---

## 📦 已創建的文件

1. ✅ `CODE_ANALYSIS_2025_11_17.md` - 詳細代碼分析
2. ✅ `PHASE3_EXPERIMENT_PLAN.md` - 實驗執行指南
3. ✅ `PHASE3_SUMMARY.md` - 本文檔 (總結)
4. ✅ `configs/inference/phase3_exp_a1_tighter_thresholds.yaml`
5. ✅ `configs/inference/phase3_exp_b1_scale100.yaml`
6. ✅ `configs/inference/phase3_exp_b2_scale50.yaml`

---

## 🔗 相關文檔

- `MAP_EXPERIMENT_SUMMARY.md` - 歷史實驗總結
- `PHASE1_RESULTS.md` - Phase 1 (特徵提取器測試)
- `PHASE2_P2A_RESULTS.md` - Phase 2-A (加權聚合)
- `PHASE1_DECISION.md` - Phase 1 → Phase 2 決策依據

---

**狀態**: ✅ 代碼審查完成,實驗配置就緒
**下一步**: 運行實驗 A1 (最優先)
**預計時間**: 每個實驗 10-20 分鐘,總計 30-60 分鐘
**最佳實踐**: 先運行 A1,看到結果後再決定是否運行 B1/B2
