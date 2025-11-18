# SigLIP Baseline 代碼分析與改進方案
**日期**: 2025-11-17
**目標**: 分析當前 AP@25=0.062 的 baseline,設計實驗突破 mAP=0 的僵局

---

## 1. 代碼審查結果

### 1.1 scale_semantic_score 和 apply_softmax 使用情況

**✅ 確認**: 代碼中**所有路徑**都正確使用了這兩個參數

**關鍵發現**:
- `retrieval.py:38-72`: `compute_cosine_similarity()` 函數默認值正確
  ```python
  scale_semantic_score: float = 300.0  # ✅
  apply_softmax: bool = True            # ✅
  ```

- `config.py`: 所有策略配置都有正確的默認值
  ```python
  # RetrievalConfig (line 17-18)
  scale_semantic_score: float = 300.0
  apply_softmax: bool = True

  # HierarchicalStrategyConfig (line 86-87)
  scale_semantic_score: float = 300.0
  apply_softmax: bool = True
  ```

- `hierarchical.py`: 所有 4 處直接調用都正確傳遞參數
  - Line 279: ✅ `scale_semantic_score=self.retriever.scale_semantic_score`
  - Line 398: ✅ `scale_semantic_score=self.retriever.scale_semantic_score`
  - Line 587: ✅ `scale_semantic_score=self.retriever.scale_semantic_score`
  - Line 712: ✅ `scale_semantic_score=self.retriever.scale_semantic_score`

**結論**: **參數使用無誤**,代碼層面沒有導致性能下降的 bug。

---

## 2. 當前 Baseline 分析

### 2.1 Phase 2-A 配置 (AP@25=0.062)

**特徵提取器**: OpenAI CLIP (`openai/clip-vit-large-patch14-336`)
**策略**: Hierarchical
**關鍵改進**: 加權幀聚合 (weighted frame aggregation)

**配置參數** (`configs/inference/phase2_weighted_agg.yaml`):
```yaml
inference:
  retrieval:
    temperature: 0.2
    min_similarity: 0.005        # ⚠️ 非常寬鬆
    top_k_per_level: 200
    # scale_semantic_score: 300  # ⚠️ 未顯式設置,使用默認值
    # apply_softmax: true        # ⚠️ 未顯式設置,使用默認值

  hierarchical:
    coarse_top_k: 200
    object_top_k: 200
    part_top_k: 100
    coarse_threshold: 0.01       # ⚠️ 非常寬鬆
    refinement_threshold: 0.01   # ⚠️ 非常寬鬆
    refinement_margin: 0.01
    use_combined_query: true
    # scale_semantic_score: 300  # ⚠️ 未顯式設置
    # apply_softmax: true        # ⚠️ 未顯式設置

  pairing:
    containment_threshold: 0.001  # ⚠️ 極度寬鬆
    use_geometric_score: false    # ⚠️ 關閉幾何分數

  nms:
    iou_threshold: 0.25
    use_soft_nms: false
    keep_top_k: 300
```

**結果**:
- Predictions: 211
- mAP: 0.000
- AP@50: 0.000
- **AP@25: 0.062** ⭐

---

## 3. 問題診斷

### 3.1 為什麼 AP@25 > 0 但 mAP = 0?

**原因**: Mask IoU 分布問題
```
估計的 IoU 分布:
  0.25 ≤ IoU < 0.50: 少數預測 ✅ (產生 AP@25 = 0.062)
  IoU ≥ 0.50: 0 個預測 ❌ (導致 AP@50 = 0, mAP = 0)
```

**瓶頸定位**:
1. ❌ **不在特徵提取**: CLIP 已經改善了語義匹配 (從 SigLIP 的 0.000 到 0.037)
2. ❌ **不在聚合方式**: Phase 2-A 加權聚合進一步提升到 0.062
3. ✅ **在 Mask Quality**: 預測的 mask 與 GT 的重疊度不足

### 3.2 Mask Quality 的來源

在 My3DIS pipeline 中,最終預測的 mask 來自:
```
SAM2 tracking output → 3D aggregation → 2D projection → 與 GT 比較
```

**無法直接改進的部分**:
- SAM2 tracking 已經完成
- 3D aggregation 已經固定

**可以改進的部分**:
1. **Retrieval 階段**: 選擇更好的候選
2. **Pairing 階段**: 更準確的 object-part 配對
3. **Scoring 階段**: 更好的分數聚合
4. **NMS 階段**: 更精準的去重

---

## 4. 改進方向

### 4.1 方向 A: 調整 Hierarchical 參數

**假設**: 當前閾值過於寬鬆,導致大量低質量候選
**實驗**: 逐步收緊閾值,觀察 precision-recall 變化

**參數調整方案**:
```yaml
# 實驗 A1: 收緊 retrieval 閾值
inference:
  retrieval:
    min_similarity: 0.01  # 0.005 → 0.01
  hierarchical:
    coarse_threshold: 0.02   # 0.01 → 0.02
    refinement_threshold: 0.02  # 0.01 → 0.02

# 實驗 A2: 減少 top-k
inference:
  hierarchical:
    coarse_top_k: 100  # 200 → 100
    object_top_k: 100  # 200 → 100
    part_top_k: 50     # 100 → 50

# 實驗 A3: 啟用幾何分數
inference:
  pairing:
    use_geometric_score: true  # false → true
```

**預期**:
- 預測數量減少
- Precision 提升
- AP@25 可能提升 0.005-0.010
- 有機會產生 AP@50 > 0

---

### 4.2 方向 B: 調整 scale_semantic_score

**假設**: scale=300 可能過大,導致 softmax 過度 peaked
**理論分析**:
```python
# 當 scale=300 時:
cosine_sim = 0.3  # 例如
scaled = 300 * 0.3 = 90
softmax(90) ≈ 1.0  # 非常接近 1

# 當 scale=100 時:
scaled = 100 * 0.3 = 30
softmax(30) ≈ 0.9-1.0  # 稍微平滑

# 當 scale=50 時:
scaled = 50 * 0.3 = 15
softmax(15) ≈ 0.7-0.9  # 更平滑的分布
```

**實驗方案**:
```yaml
# 實驗 B1: 降低 scale
inference:
  retrieval:
    scale_semantic_score: 100  # 300 → 100
  hierarchical:
    scale_semantic_score: 100

# 實驗 B2: 進一步降低
inference:
  retrieval:
    scale_semantic_score: 50  # 300 → 50
  hierarchical:
    scale_semantic_score: 50

# 實驗 B3: 關閉 softmax
inference:
  retrieval:
    apply_softmax: false  # true → false
    scale_semantic_score: 1.0  # 僅使用 cosine similarity
  hierarchical:
    apply_softmax: false
    scale_semantic_score: 1.0
```

**注意**: 實驗 B3 需要同時調整所有閾值到 [-1, 1] 範圍

---

### 4.3 方向 C: 嘗試不同的聚合策略

**假設**: Hierarchical 策略可能不是最優的

**實驗方案**:
```yaml
# 實驗 C1: Independent 策略
inference:
  strategy: independent
  independent:
    use_combined_query: true
    retrieval_threshold: 0.02
    top_k_per_level: 100
    scale_semantic_score: 300
    apply_softmax: true

# 實驗 C2: 測試所有策略
inference:
  test_all_strategies: true
  strategies_to_test: ["hierarchical", "independent", "exhaustive_pairing"]
```

---

### 4.4 方向 D: 優化 Combined Query

**假設**: "part of object" 組合查詢的構建方式可以優化

**當前實現** (`combined_query_mixin.py:80-90`):
```python
# 簡單拼接
combined_text = f"{part_text} of {object_text}"
# 例如: "door of cabinet"
```

**改進實驗**:
```python
# D1: 不同的模板
combined_text = f"{part_text} from {object_text}"
combined_text = f"{part_text} in {object_text}"
combined_text = f"{part_text} on {object_text}"

# D2: 分開編碼再組合
part_feat = extract_feature(part_text)
obj_feat = extract_feature(object_text)
combined_feat = (part_feat + obj_feat) / 2  # 平均
combined_feat = 0.6 * part_feat + 0.4 * obj_feat  # 加權
```

---

## 5. 實驗計劃

### 5.1 Phase 3: 參數調優實驗

**優先級排序**:
1. **實驗 A1** (收緊閾值) - 最簡單,最有可能立即見效
2. **實驗 B3** (關閉 softmax) - 參考 PHASE1 失敗經驗,謹慎嘗試
3. **實驗 B1** (降低 scale 到 100) - 平衡實驗
4. **實驗 C1** (Independent 策略) - 策略級對比
5. **實驗 A3** (啟用幾何分數) - 可能改善 mask quality

**實驗流程**:
```bash
# 每個實驗:
1. 複製 configs/inference/phase2_weighted_agg.yaml
2. 修改參數
3. 運行推理: PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py --config <new_config>
4. 檢查結果: AP@25, AP@50, mAP
5. 記錄到 PHASE3_RESULTS.md
```

---

### 5.2 成功標準

**最低目標** (繼續調參):
```
AP@25 > 0.070  (+0.008 from baseline)
AP@50 > 0.000  (突破 0)
```

**中等目標** (策略成功):
```
AP@25 > 0.100  (+0.038 from baseline)
AP@50 > 0.010
mAP > 0.000
```

**理想目標** (重現早期成績):
```
AP@25 > 0.150
AP@50 > 0.030
mAP > 0.005
```

---

## 6. 風險評估

### 6.1 實驗 A (閾值調整)
- **風險**: 低
- **預期改善**: 小到中等 (+0.005-0.015)
- **副作用**: 可能減少預測數量

### 6.2 實驗 B (scale 調整)
- **風險**: 中
- **預期改善**: 不確定 (-0.010 to +0.020)
- **副作用**: 可能破壞現有平衡

### 6.3 實驗 C (策略切換)
- **風險**: 低 (獨立測試)
- **預期改善**: 中等 (+0.010-0.030)
- **副作用**: 無

---

## 7. 下一步行動

### 立即執行 (優先級 P0)
1. ✅ **備份當前代碼**
2. ✅ **創建實驗配置**: `configs/inference/phase3_exp_a1.yaml`
3. ⏳ **運行實驗 A1**: 收緊閾值
4. ⏳ **記錄結果**: 創建 `PHASE3_A1_RESULTS.md`

### 短期 (優先級 P1)
- 運行實驗 B1, B3
- 對比所有結果
- 選擇最佳配置

### 中期 (優先級 P2)
- 實驗 C: 測試其他策略
- 實驗 D: 優化 combined query
- 如果所有實驗無效,考慮 Phase 4 (fine-tuning)

---

**狀態**: ✅ 分析完成,準備實驗
**下一階段**: Phase 3 - 參數調優實驗
**預計時間**: 4-6 小時 (實驗 + 分析)
