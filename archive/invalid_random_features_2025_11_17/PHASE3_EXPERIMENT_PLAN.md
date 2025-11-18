# Phase 3 實驗計劃: 參數調優突破僵局

**日期**: 2025-11-17
**Baseline**: Phase 2-A (AP@25=0.062, AP@50=0.000, mAP=0.000)
**目標**: 突破 mAP=0,達到 AP@50 > 0

---

## 實驗設計

### 實驗 A1: 收緊閾值 (Tighter Thresholds)

**假設**: 當前閾值過於寬鬆,導致大量低質量候選降低 precision

**配置**: `configs/inference/phase3_exp_a1_tighter_thresholds.yaml`

**關鍵變更**:
```yaml
# Retrieval 層級
retrieval:
  min_similarity: 0.01  # 0.005 → 0.01 (2x 更嚴格)

# Hierarchical 層級
hierarchical:
  coarse_top_k: 100       # 200 → 100 (減半)
  object_top_k: 100       # 200 → 100 (減半)
  part_top_k: 50          # 100 → 50 (減半)
  coarse_threshold: 0.02  # 0.01 → 0.02 (2x 更嚴格)
  refinement_threshold: 0.02  # 0.01 → 0.02 (2x 更嚴格)
```

**預期**:
- 預測數量: 211 → 100-150
- AP@25: 0.062 → 0.065-0.075 (+0.003-0.013)
- AP@50: 0.000 → 0.005-0.015 (突破 0!)
- Precision 提升, Recall 可能下降

---

### 實驗 B1: 降低 Scale 到 100

**假設**: scale=300 可能過大,導致 softmax 過度 peaked,損失語義細節

**配置**: `configs/inference/phase3_exp_b1_scale100.yaml`

**關鍵變更**:
```yaml
retrieval:
  scale_semantic_score: 100.0  # 300 → 100

hierarchical:
  scale_semantic_score: 100.0  # 300 → 100
```

**理論分析**:
```
scale=300: softmax([90, 87, 84]) ≈ [0.95, 0.04, 0.01] (極度 peaked)
scale=100: softmax([30, 29, 28]) ≈ [0.58, 0.25, 0.17] (更平滑)
```

**預期**:
- 相似度分數分布更平滑
- Top-K 選擇更多樣
- AP@25: 0.062 → 0.055-0.075 (不確定)
- 可能提升也可能下降

---

### 實驗 B2: 進一步降低 Scale 到 50

**假設**: 更平滑的分數分布可能改善 recall

**配置**: `configs/inference/phase3_exp_b2_scale50.yaml`

**關鍵變更**:
```yaml
retrieval:
  scale_semantic_score: 50.0  # 300 → 50

hierarchical:
  scale_semantic_score: 50.0  # 300 → 50
```

**預期**:
- 分數分布極度平滑
- 可能過度平滑,損失語義區分度
- AP@25: 0.062 → 0.045-0.070 (風險較高)

---

## 執行流程

### 準備工作

1. **確認環境**:
```bash
conda activate 3Dsiglip
cd /media/Pluto/richkung/My3DIS
```

2. **設置 aggregation_output 路徑**:

由於實驗復用 Phase 2 的 aggregation 輸出,需要手動設置路徑:

```bash
# 方法 1: 使用 symlink (推薦)
mkdir -p outputs/experiments/phase3_exp_a1_tighter/scene_00093_01
ln -s /media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/aggregation_output \
      outputs/experiments/phase3_exp_a1_tighter/scene_00093_01/aggregation_output

mkdir -p outputs/experiments/phase3_exp_b1_scale100/scene_00093_01
ln -s /media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/aggregation_output \
      outputs/experiments/phase3_exp_b1_scale100/scene_00093_01/aggregation_output

mkdir -p outputs/experiments/phase3_exp_b2_scale50/scene_00093_01
ln -s /media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/aggregation_output \
      outputs/experiments/phase3_exp_b2_scale50/scene_00093_01/aggregation_output

# 方法 2: 修改 YAML 中的 aggregation_output_dir (備選)
# 直接編輯 YAML,設置:
# aggregation_output_dir: /media/Pluto/richkung/My3DIS/outputs/experiments/test_v7_246_ssam4_filter200_fill200_prop30_1116/scene_00093_01/aggregation_output
```

---

### 運行實驗

#### 實驗 A1 (優先級 P0)

```bash
# 創建日誌目錄
mkdir -p logs/phase3

# 運行實驗
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/phase3_exp_a1_tighter_thresholds.yaml \
  2>&1 | tee logs/phase3/exp_a1_tighter.log

# 快速檢查結果
tail -50 logs/phase3/exp_a1_tighter.log | grep -E "mAP|AP@"
```

#### 實驗 B1 (優先級 P1)

```bash
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/phase3_exp_b1_scale100.yaml \
  2>&1 | tee logs/phase3/exp_b1_scale100.log

tail -50 logs/phase3/exp_b1_scale100.log | grep -E "mAP|AP@"
```

#### 實驗 B2 (優先級 P2)

```bash
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/phase3_exp_b2_scale50.yaml \
  2>&1 | tee logs/phase3/exp_b2_scale50.log

tail -50 logs/phase3/exp_b2_scale50.log | grep -E "mAP|AP@"
```

---

### 快速對比結果

```bash
echo "=== Phase 2-A Baseline ==="
tail -20 logs/phase2/p2a_fixed.log | grep -E "mAP|AP@"

echo -e "\n=== Exp A1 (Tighter Thresholds) ==="
tail -20 logs/phase3/exp_a1_tighter.log | grep -E "mAP|AP@"

echo -e "\n=== Exp B1 (Scale=100) ==="
tail -20 logs/phase3/exp_b1_scale100.log | grep -E "mAP|AP@"

echo -e "\n=== Exp B2 (Scale=50) ==="
tail -20 logs/phase3/exp_b2_scale50.log | grep -E "mAP|AP@"
```

---

## 結果記錄模板

每個實驗完成後,記錄到 `PHASE3_RESULTS.md`:

```markdown
### 實驗 A1 結果

**日期**: YYYY-MM-DD HH:MM
**配置**: `configs/inference/phase3_exp_a1_tighter_thresholds.yaml`
**日誌**: `logs/phase3/exp_a1_tighter.log`

**指標**:
| Metric | Baseline (P2-A) | Exp A1 | 變化 |
|--------|-----------------|--------|------|
| mAP    | 0.000           | X.XXX  | +X.XXX |
| AP@50  | 0.000           | X.XXX  | +X.XXX |
| AP@25  | 0.062           | X.XXX  | +X.XXX |
| Predictions | 211        | XXX    | ±XXX |

**分析**:
- [ ] AP@25 是否提升?
- [ ] AP@50 是否突破 0?
- [ ] 預測數量變化如何?
- [ ] 是否有 trade-off?

**決策**:
- [ ] 繼續下一個實驗
- [ ] 調整參數再試
- [ ] 採用此配置
```

---

## 成功標準

### 最低標準 (繼續實驗)
```
任一實驗達到:
  AP@25 > 0.070  (+0.008 from baseline)
  或
  AP@50 > 0.000  (突破 0)
```

### 中等標準 (實驗成功)
```
最佳實驗達到:
  AP@25 > 0.080  (+0.018 from baseline)
  AP@50 > 0.010
  mAP > 0.000
```

### 理想標準 (重大突破)
```
最佳實驗達到:
  AP@25 > 0.100  (+0.038 from baseline)
  AP@50 > 0.020
  mAP > 0.005
```

---

## 決策樹

```
運行實驗 A1, B1, B2
    ↓
有任一實驗達到最低標準?
    ├─ Yes → 選擇最佳配置,進入 Phase 3-B (進一步優化)
    │         或記錄成功,準備多場景測試
    └─ No  → 所有實驗都失敗?
              ├─ Yes → 考慮:
              │         1. 實驗 C (嘗試其他策略)
              │         2. Phase 4 (fine-tuning CLIP)
              └─ No  → 調整參數,再次實驗
```

---

## 風險與備選方案

### 風險 1: 所有實驗都降低了 AP@25
**應對**:
- 檢查日誌中的錯誤
- 確認 aggregation_output 路徑正確
- 回退到 Phase 2-A 配置

### 風險 2: 實驗時間過長
**應對**:
- 優先運行 A1 (最可能成功)
- B1, B2 可並行運行
- 設置 timeout: 每個實驗最多 30 分鐘

### 風險 3: GPU OOM
**應對**:
- 降低 batch size (如果有)
- 減少 top-k 值
- 清空 CUDA cache

---

**狀態**: ✅ 實驗計劃完成,配置就緒
**下一步**: 執行實驗 A1
**預計時間**: 每個實驗 10-20 分鐘
