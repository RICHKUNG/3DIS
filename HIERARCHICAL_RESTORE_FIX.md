# Hierarchical 策略恢復修復

**時間**: 2025-11-16 23:50
**問題**: Emergency fix 導致 Hierarchical 從 mAP 0.006 退步到 0.000
**原因**: 一刀切的參數調整幫助了 Independent 但傷害了 Hierarchical

---

## 根本原因分析

### 為何 Emergency Fix 幫助 Independent 但傷害 Hierarchical?

**Independent Strategy 特性:**
- ✅ 不依賴層次結構 → 沒有噪音傳播問題
- ✅ 窮舉搜索所有層級 → 受益於更多候選
- ✅ 無幾何假設 → 移除約束反而有幫助

**Hierarchical Strategy 特性:**
- ❌ 依賴粗層級候選質量 → 噪音在層次中傳播
- ❌ 幾何約束是必需的 → 用於剪枝不合理分支
- ❌ 更多候選 = 更多噪音 → 在層次搜索中放大

### Emergency Fix 的副作用

| 修改 | 對 Independent 影響 | 對 Hierarchical 影響 |
|------|-------------------|---------------------|
| `containment_threshold: 0.0` | ✅ 允許更多配對 | ❌ 引入幾何上不合理的配對 |
| `coarse_threshold: 0.005` | ✅ 更多初始候選 | ❌ 粗層級噪音 2 倍增加 |
| `use_soft_nms: true` | ✅ 保留多樣性 | ❌ 可能抑制了少數好配對 |
| `scale_range: [0.00001, 100.0]` | ✅ 極度寬鬆 | ❌ 允許尺寸不合理的配對 |

---

## 已應用的修復

### 1. 恢復幾何約束 ⭐⭐⭐⭐⭐
```yaml
pairing:
  containment_threshold: 0.001   # RESTORED from 0.0
  scale_range: [0.0001, 20.0]    # RESTORED from [0.00001, 100.0]
```

**原因**: Hierarchical 需要幾何約束來過濾不可能的配對

### 2. 恢復 Hard NMS ⭐⭐⭐⭐
```yaml
nms:
  use_soft_nms: false            # RESTORED from true
  iou_threshold: 0.25            # RESTORED from 0.15
  keep_top_k: 300                # RESTORED from 500
```

**原因**: Hard NMS 對 Hierarchical 效果更好（18:00 運行驗證）

### 3. 恢復嚴格閾值 ⭐⭐⭐⭐
```yaml
hierarchical:
  coarse_threshold: 0.01         # RESTORED from 0.005
  refinement_threshold: 0.01     # RESTORED from 0.005
```

**原因**: 更嚴格的閾值減少粗層級噪音

### 4. 保留有益改進
```yaml
hierarchical:
  coarse_top_k: 500              # KEPT (was 200)
  object_top_k: 500              # KEPT (was 200)
  part_top_k: 200                # KEPT (was 100)
```

**原因**: 更大的 top-k 仍然有幫助（增加召回）

---

## 預期結果

### 18:00 運行（成功配置）:
- Coarse candidates: 11
- Refined objects: 30
- Candidate pairs: 62 → 12 NMS
- **mAP: 0.006, AP50: 0.050**

### Emergency Fix 運行（失敗）:
- Coarse candidates: 21 (2x 噪音)
- Refined objects: 47 (噪音傳播)
- Candidate pairs: 65 → 17 NMS (更多但質量低)
- **mAP: 0.000** ❌

### 本次修復預期:
- Coarse candidates: ~11-15（回到健康水平）
- Refined objects: ~30-40
- Candidate pairs: ~50-70 → ~10-15 NMS
- **預期 mAP: 0.006-0.010** ✅

---

## 對比三種策略

| 策略 | 最佳配置 | 預期 mAP | 狀態 |
|------|---------|---------|------|
| **Hierarchical** | 幾何約束 + Hard NMS | **0.006-0.010** | ⏳ 測試中 |
| **Independent** | 無約束 + Soft NMS | **0.003** | ✅ 已驗證 |
| **Exhaustive** | TBD | **0.000** | ⏳ 需優化 |

---

## 長期解決方案

**需要實現：策略特定參數覆蓋**

當前問題：全局 `pairing` 和 `nms` 配置對所有策略生效

**建議實現**:
```yaml
inference:
  pairing:  # 默認配置
    containment_threshold: 0.001
    
  nms:  # 默認配置
    iou_threshold: 0.25
    
  hierarchical:
    # 策略特定覆蓋
    pairing_override:
      containment_threshold: 0.001  # 嚴格幾何約束
    nms_override:
      use_soft_nms: false           # Hard NMS
      
  independent:
    # 策略特定覆蓋
    pairing_override:
      containment_threshold: 0.0    # 無約束
    nms_override:
      use_soft_nms: true            # Soft NMS
      iou_threshold: 0.15
```

---

## 執行

### 當前運行
```bash
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation
```

**配置狀態**:
- ✅ 單策略模式 (`test_all_strategies: false`)
- ✅ Hierarchical 策略
- ✅ 恢復幾何約束 (`containment_threshold: 0.001`)
- ✅ 恢復 Hard NMS (`use_soft_nms: false`)
- ✅ 恢復嚴格閾值 (`coarse_threshold: 0.01`)

---

## 成功指標

### 最小成功（驗證修復有效）
- [ ] mAP >= 0.006 （恢復到 18:00 水平）
- [ ] AP50 >= 0.050
- [ ] Coarse candidates ~10-15（噪音減少）

### 理想成功（超越之前）
- [ ] mAP >= 0.008 （top-k 增加的好處）
- [ ] AP50 >= 0.060
- [ ] IoU >= 0.25 匹配數 >= 2

---

## 總結

1. **Emergency fix 是一把雙刃劍**
   - ✅ 幫助 Independent 突破 0 (mAP 0.003)
   - ❌ 傷害 Hierarchical (0.006 → 0.000)

2. **不同策略需要不同參數**
   - Independent: 無約束、窮舉、Soft NMS
   - Hierarchical: 幾何約束、層次引導、Hard NMS

3. **本次修復聚焦 Hierarchical**
   - 目標: 恢復到 mAP 0.006
   - 未來可以分別優化 Independent

4. **需要長期改進**
   - 實現策略特定參數覆蓋機制
   - 允許同時優化所有策略

---

**當前狀態**: ⏳ 測試運行中

**日誌位置**: `logs/eval/run_hierarchical_restore_*.log`

**監控命令**:
```bash
tail -f logs/eval/run_hierarchical_restore_*.log | grep -E "mAP|AP50|Coarse|pairs"
```
