# YAML參數優化摘要

**日期**: 2025-11-19  
**目標**: 提升hierarchical和exhaustive_pairing策略的mAP性能  
**配置文件**: `configs/inference/experiment_pipeline.yaml`

---

## 🎯 優化目標

- **當前mAP**:
  - Independent: 1.05%
  - Hierarchical: 0.15% (修復bug後預期恢復)
  - Exhaustive_pairing: 1.43%

- **優化目標**:
  - Hierarchical: 0.15% → **2-5%**
  - Exhaustive_pairing: 1.43% → **3-6%**

---

## 📝 修改詳情

### 1. Pairing配置 (第146-150行)

**目的**: 加強幾何約束，減少無效配對

| 參數 | 修改前 | 修改後 | 說明 |
|------|--------|--------|------|
| `containment_threshold` | 0.0 | **0.3** | 要求part至少30%在object內 |
| `scale_range` | [0.00001, 100.0] | **[0.005, 0.95]** | 限制合理的大小比例範圍 |
| `use_geometric_score` | false | **true** | 啟用幾何分數獎勵 |
| `use_tree_pruning` | false | **true** | 利用family tree過濾 |

**預期影響**:
- ✅ 減少不合理的object-part配對
- ✅ 利用層級關係提高配對質量
- ✅ 降低假陽性率

---

### 2. NMS配置 (第153-158行)

**目的**: 減少過度過濾，保留更多有效預測

| 參數 | 修改前 | 修改後 | 說明 |
|------|--------|--------|------|
| `iou_threshold` | 0.15 | **0.35** | 提高閾值以減少過濾 |
| `use_soft_nms` | true | **false** | 使用hard NMS更明確 |
| `keep_top_k` | 500 | **1000** | 允許更多預測通過 |

**預期影響**:
- ✅ IoU=0.15太低會過度過濾，提高到0.35
- ✅ 保留更多有效預測
- ⚠️ 可能略微增加假陽性（可接受）

---

### 3. Scoring配置 (第160-168行)

**目的**: 改善評分方法，獎勵幾何和層級關係

| 參數 | 修改前 | 修改後 | 說明 |
|------|--------|--------|------|
| `method` | arithmetic | **weighted_sum** | 更靈活的加權方法 |
| `weights.object` | 0.5 | **0.4** | 降低object權重 |
| `weights.part` | 0.5 | **0.4** | 降低part權重 |
| `weights.geometric` | 0.0 | **0.2** | 加入幾何分數權重 |
| `hierarchy_bonus` | 0.05 | **0.1** | 加倍層級一致性獎勵 |

**預期影響**:
- ✅ 幾何關係合理的配對得到獎勵
- ✅ 符合family tree的配對優先排序
- ✅ 更平衡的評分機制

---

### 4. Hierarchical策略配置 (第188-197行)

**目的**: 放寬閾值，增加候選數量

| 參數 | 修改前 | 修改後 | 說明 |
|------|--------|--------|------|
| `part_top_k` | 200 | **300** | 每個object允許更多part |
| `coarse_threshold` | 0.01 | **0.005** | 降低初始篩選門檻 |
| `refinement_threshold` | 0.01 | **0.005** | 降低精煉篩選門檻 |
| `refinement_margin` | 0.01 | **0.05** | 更傾向選擇子層級 |

**預期影響**:
- ✅ Coarse階段找到更多候選
- ✅ 允許更多part候選參與配對
- ✅ 更傾向使用細粒度層級

---

## 📊 預期改善

### Hierarchical策略
- **修復前**: mAP 0.000（bug導致）
- **修復後（無優化）**: mAP 0.15-1%
- **優化後**: mAP **2-5%**
- **提升倍數**: 10-30倍（相比修復前）

### Exhaustive_pairing策略  
- **當前**: mAP 1.43%
- **優化後**: mAP **3-6%**
- **提升倍數**: 2-4倍

### 預測數量變化
- **Hierarchical**: 
  - 當前: 0-100個/場景
  - 優化後: 100-400個/場景
- **Exhaustive_pairing**:
  - 當前: 100-500個/場景
  - 優化後: 200-800個/場景

---

## 🚀 測試建議

### 階段1: 單場景快速驗證
```bash
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/experiment_pipeline.yaml \
  --scenes scene_00018_00 \
  --strategies hierarchical,exhaustive_pairing
```

**檢查點**:
- ✓ 預測數量在100-400範圍
- ✓ 日誌無錯誤
- ✓ mAP > 0（至少有非零預測）

### 階段2: 多場景小規模測試
選擇3-5個代表性場景：
```bash
--scenes scene_00018_00,scene_00051_00,scene_00093_00
```

**驗證**:
- ✓ 平均mAP提升
- ✓ 無內存溢出
- ✓ 運行時間可接受

### 階段3: 完整評估
```bash
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/experiment_pipeline.yaml \
  --strategies hierarchical,exhaustive_pairing
```

---

## ⚠️ 注意事項

1. **預測數量監控**
   - 如果某場景預測數 > 2000，考慮調低top_k
   - 如果預測數 < 50，考慮進一步降低threshold

2. **內存使用**
   - Exhaustive_pairing可能產生大量候選
   - 必要時調整`keep_top_k`和`top_k_per_level`

3. **回滾方案**
   ```bash
   git checkout configs/inference/experiment_pipeline.yaml
   ```

4. **調試建議**
   - 啟用詳細日誌: `logging.level: DEBUG`
   - 逐個策略測試，避免混淆結果

---

## 📈 效果追蹤

### 測試記錄表

| 測試日期 | 場景數 | Hierarchical mAP | Exhaustive mAP | 備註 |
|---------|-------|------------------|----------------|------|
| 2025-11-19 (baseline) | 41 | 0.15% | 1.43% | 修復bug前 |
| _待測試_ | 1 | - | - | 單場景驗證 |
| _待測試_ | 5 | - | - | 小規模測試 |
| _待測試_ | 41 | - | - | 完整評估 |

### 效果確認清單
- [ ] Hierarchical預測數 > 0
- [ ] Hierarchical mAP > 0.15%
- [ ] Exhaustive mAP提升 > 50%
- [ ] 無內存溢出
- [ ] 無明顯運行時間增加

---

## 🔧 進一步調優

如果效果不理想，可嘗試：

1. **更激進的閾值**:
   ```yaml
   hierarchical:
     coarse_threshold: 0.001
     refinement_threshold: 0.001
   ```

2. **更寬鬆的NMS**:
   ```yaml
   nms:
     iou_threshold: 0.5
     keep_top_k: 2000
   ```

3. **完全禁用幾何約束**（測試用）:
   ```yaml
   pairing:
     containment_threshold: 0.0
     use_geometric_score: false
   ```

---

**生成時間**: 2025-11-19  
**配置版本**: v7_experiment_pipeline  
**狀態**: ✅ 已應用，待測試
