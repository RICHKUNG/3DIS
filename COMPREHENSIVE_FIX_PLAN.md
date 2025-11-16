# 全面修復方案

**日期**: 2025-11-16 22:30
**問題**: mAP 仍為 0.000，Independent 和 Exhaustive_Pairing 策略無效

---

## 🔍 根本問題診斷

### 問題 1: Pairing 幾何約束過嚴 ⭐⭐⭐⭐⭐

**證據**:
```
Created 99 candidate pairs → NMS → 23 predictions (Hierarchical)
Created 59 candidate pairs → NMS → 16 predictions (Hierarchical 另一query)
```

**問題**: 配對階段創建的 pairs 太少（每個 object 平均只有 1-2 個 part 候選）

**原因**: `pairing.containment_threshold` 和 `scale_range` 過於嚴格

**當前設置**:
```yaml
pairing:
  containment_threshold: 0.001  # Part 必須有 0.1% 在 object 內
  scale_range: [0.0001, 20.0]   # Part 可以是 object 的 0.01%-2000%
```

**問題所在**:
- `containment_threshold: 0.001` 看似很小，但對於小 parts 仍然過嚴
- 當 part 和 object 重疊很小時（例如 door 只有一小部分在 cabinet 內），會被過濾

---

### 問題 2: Query 覆蓋不完整

**證據**:
- GT 有 11 個 composite IDs
- 只處理了 10 個 queries
- 缺少: 11001 (cabinet + handle)

---

### 問題 3: Independent 策略配對數量極少

**根因**: Independent 不使用 family tree 引導，完全依賴幾何約束

**預期配對數**:
- Object 候選: 假設 L2=37, L4=74（top_k=500 不是瓶頸）
- Part 候選: L4=74, L6=81
- 理論組合: (37+74) × (74+81) = 111 × 155 = 17,205 組合
- 幾何約束後: **可能只剩 < 100 個有效 pairs**（當前約束下）

---

### 問題 4: Exhaustive_Pairing 評分機制問題

**當前設置**:
```yaml
exhaustive_pairing:
  return_all_pairs: true   # 返回所有 level pairs
  selection_metric: sum
```

**問題**: 即使返回所有 pairs，如果幾何約束過嚴，仍然沒有足夠的候選

---

## 🎯 解決方案（分優先級）

### 🔥 P0: 放寬幾何約束（最關鍵）

#### 方案 A: 完全移除 containment 要求
```yaml
pairing:
  containment_threshold: 0.0    # ⭐ 移除包含要求
  scale_range: [0.00001, 100.0] # 更寬的尺寸範圍
  use_geometric_score: true     # 改為用分數而非硬約束
  geometric_score_weight: 0.1   # 低權重（主要依賴語義）
```

**理由**:
- 在 OV 任務中，語義匹配比幾何約束更重要
- 許多 object-part 對在 3D 中重疊很小（例如 door 和 cabinet）
- NMS 會自然過濾掉不合理的預測

#### 方案 B: 使用極寬鬆閾值（妥協方案）
```yaml
pairing:
  containment_threshold: 0.0001  # 幾乎沒有要求（0.01%）
  scale_range: [0.00001, 100.0]
  use_geometric_score: false
```

---

### 🔥 P1: 降低 NMS 閾值

**當前設置**:
```yaml
nms:
  iou_threshold: 0.25  # 過高
  keep_top_k: 300
```

**建議**:
```yaml
nms:
  iou_threshold: 0.15   # ⭐ 降低到 0.15
  keep_top_k: 500       # 增加保留數量
  use_soft_nms: true    # ⭐ 改用 soft NMS
  soft_nms_sigma: 0.3   # 較小的 sigma（更激進）
```

**理由**:
- 當前 NMS 可能過度抑制了有效預測
- Soft NMS 可以保留更多候選，只降低分數

---

### 🔥 P2: 為 Independent 策略添加專門優化

**問題**: Independent 不用 family tree，完全依賴幾何約束

**解決方案**: 為 Independent 使用更寬鬆的幾何約束

**實現**: 在 Independent 策略調用 pairing 時，傳遞特殊參數

**代碼修改** (`independent.py`):
```python
pairs = self.pairer.pair_all(
    object_proposals=object_results.proposals,
    part_proposals=part_results.proposals,
    label_id=label_id,
    points=points,
    use_tree_pruning=False,
    # ⭐ 添加參數覆蓋
    override_containment_threshold=0.0,  # 移除包含要求
    override_use_geometric_score=True,   # 改用分數
)
```

---

### 🔥 P3: 啟用所有策略測試模式

**目的**: 同時運行三種策略，聚合結果

```yaml
inference:
  test_all_strategies: true  # ⭐ 改回 true
  strategies_to_test: [independent, hierarchical, exhaustive_pairing]

  # 聚合模式（NEW）
  aggregate_strategies: true        # 聚合所有策略的結果
  aggregation_method: union_nms     # union_nms | voting | best_per_query
  aggregation_nms_threshold: 0.20   # 最終聚合 NMS 閾值
```

**理由**:
- 不同策略可能找到不同的有效預測
- 聚合可以提升覆蓋率

---

### 🔥 P4: 為 Exhaustive_Pairing 優化 level pairs

**當前問題**: 嘗試所有組合，包括不合理的（L2,L2）、（L6,L6）

**解決方案**: 只測試合理的 level pairs

```yaml
exhaustive_pairing:
  # ⭐ 明確指定合理的 level pairs
  valid_level_pairs: [[L2, L4], [L2, L6], [L4, L6]]  # 只測試這些組合
  # 或者使用規則
  require_part_level_finer: true  # Part level 必須 >= object level
```

---

## 📝 實施計劃

### Phase 1: 緊急修復（30 分鐘）

#### 步驟 1: 修改配置文件
```yaml
pairing:
  containment_threshold: 0.0     # ⭐ 移除包含要求
  scale_range: [0.00001, 100.0]  # 更寬範圍
  use_geometric_score: false     # 禁用幾何分數

nms:
  iou_threshold: 0.15            # ⭐ 降低 NMS 閾值
  keep_top_k: 500                # 增加保留數量
  use_soft_nms: true             # ⭐ 使用 soft NMS
  soft_nms_sigma: 0.3
```

#### 步驟 2: 重新運行測試
```bash
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation
```

#### 預期結果:
- Hierarchical: 99 pairs → **保留更多** → 預測數 50-100
- mAP: 0.000 → **0.005-0.015**

---

### Phase 2: Independent 策略專門優化（1 小時）

#### 需要代碼修改

**文件**: `src/my3dis/inference/pairing/pairer.py`

添加參數覆蓋功能：
```python
def pair_all(
    self,
    object_proposals: List[Proposal],
    part_proposals: List[Proposal],
    label_id: int,
    points: Optional[np.ndarray] = None,
    use_tree_pruning: bool = False,
    # ⭐ NEW: 允許策略級別覆蓋
    override_containment_threshold: Optional[float] = None,
    override_scale_range: Optional[Tuple[float, float]] = None,
    override_use_geometric_score: Optional[bool] = None,
) -> List[ObjectPartPair]:
    # 使用覆蓋參數（如果提供）
    containment_threshold = (
        override_containment_threshold
        if override_containment_threshold is not None
        else self.containment_threshold
    )
    # ... 其他參數同理
```

然後在 `independent.py` 中使用：
```python
pairs = self.pairer.pair_all(
    object_proposals=object_results.proposals,
    part_proposals=part_results.proposals,
    label_id=label_id,
    points=points,
    use_tree_pruning=False,
    override_containment_threshold=0.0,  # ⭐ Independent 專用
)
```

---

### Phase 3: 多策略聚合（2 小時）

#### 需要實現聚合邏輯

**文件**: `scripts/run_full_pipeline.py`

```python
if config['inference']['test_all_strategies'] and config['inference'].get('aggregate_strategies', False):
    # 收集所有策略的預測
    all_predictions = []
    for strategy_name in ['independent', 'hierarchical', 'exhaustive_pairing']:
        preds = load_predictions(strategy_name)
        all_predictions.extend(preds)

    # 聚合 NMS
    final_predictions = aggregate_predictions(
        all_predictions,
        method=config['inference']['aggregation_method'],
        nms_threshold=config['inference']['aggregation_nms_threshold']
    )
```

---

## 🎯 預期改善

### 配置優化後（Phase 1）

| 策略 | 當前 pairs | 優化後 pairs | 當前 mAP | 預期 mAP |
|------|-----------|-------------|----------|----------|
| Hierarchical | 99 → 23 | **200-300 → 80-120** | 0.000 | **0.005-0.015** |
| Independent | ~10 | **500-1000 → 100-200** | 0.000 | **0.003-0.010** |
| Exhaustive | ~20 | **300-600 → 60-100** | 0.000 | **0.002-0.008** |

### 代碼優化後（Phase 2）

| 策略 | 預期 mAP |
|------|----------|
| Hierarchical | **0.010-0.020** |
| Independent | **0.008-0.015** |
| Exhaustive | **0.005-0.012** |

### 多策略聚合後（Phase 3）

| 指標 | 預期值 |
|------|--------|
| **聚合 mAP** | **0.015-0.030** |
| **覆蓋率** | **60-70%** （vs GT 35%） |
| **高質量匹配 (IoU >= 0.25)** | **5-10 個** |

---

## ⚠️ 風險與緩解

### 風險 1: 移除幾何約束可能導致過多誤檢

**緩解**:
- 使用更嚴格的 NMS
- 依賴語義匹配分數（SigLIP 特徵相似度）
- 保留 `keep_top_k` 限制

### 風險 2: 聚合可能引入重複預測

**緩解**:
- 最終聚合時使用嚴格的 NMS（threshold 0.20）
- 使用 voting 機制選擇最佳預測

---

## 🚀 立即行動項

### ✅ 可以立即執行（無需代碼修改）

1. 修改配置文件：
   - `containment_threshold: 0.0`
   - `iou_threshold: 0.15`
   - `use_soft_nms: true`

2. 重新運行測試

3. 分析結果

### ⏳ 需要代碼修改（1-2 小時）

1. 實現 pairing 參數覆蓋
2. 為 Independent 策略添加專門配置
3. 實現多策略聚合

---

## 📊 成功指標

### 短期（配置優化）
- [x] mAP > 0.000（突破 0）
- [ ] mAP >= 0.005
- [ ] IoU >= 0.25 的匹配 >= 3 個

### 中期（代碼優化）
- [ ] mAP >= 0.010
- [ ] Independent 策略 mAP >= 0.005
- [ ] 覆蓋率 >= 50%

### 長期（聚合優化）
- [ ] 聚合 mAP >= 0.020
- [ ] IoU >= 0.25 的匹配 >= 8 個
- [ ] 覆蓋率 >= 60%
