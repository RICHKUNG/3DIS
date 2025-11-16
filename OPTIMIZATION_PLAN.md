# mAP 優化方案 (2025-11-16)

## 根本原因已確認 ✅

**診斷結果**: Mask IoU 不足導致 mAP = 0.000

### 問題量化
- **最大 IoU**: 0.1907 (< 0.25 最小閾值)
- **預測覆蓋率**: 7.4% (GT 為 34.7%)
- **匹配實例**: 5/27 (僅 18.5%)

**結論**: 預測的 masks 太小、不完整，與 GT 的空間重疊不足。

---

## 優化方案（按優先級）

### 🔴 **優先級 1: 提高 Mask 覆蓋率**

#### 方案 1.1: 調整聚合參數
**目標**: 生成更完整的 3D masks

**修改** `configs/inference/full_pipeline.yaml`:
```yaml
aggregation:
  # 當前: iou_threshold: 0.8
  iou_threshold: 0.5  # 降低 IoU 閾值以合併更多 masks

  # 當前: area_fraction: 0.002 (1/500)
  area_fraction: 0.001  # 降低面積閾值以保留更多小區域

  # 新增: 針對不同 level 的面積閾值
  area_fraction_per_level:
    2: 0.002  # 粗粒度維持原值
    4: 0.001  # 中粒度放寬
    6: 0.0005  # 細粒度更寬鬆
```

**預期效果**: 提升覆蓋率 7.4% → 15-20%

#### 方案 1.2: 調整推理策略參數
**目標**: 選擇更大、更完整的 masks

**修改** `configs/inference/full_pipeline.yaml`:
```yaml
inference:
  retrieval:
    # 當前: top_k_per_level: 200
    top_k_per_level: 500  # 增加候選數量

    # 當前: min_similarity: 0.01
    min_similarity: 0.005  # 更寬鬆的相似度閾值

  pairing:
    # 當前: containment_threshold: 0.001
    containment_threshold: 0.0001  # 極度寬鬆的包含閾值

    # 當前: scale_range: [0.0001, 20.0]
    scale_range: [0.00001, 50.0]  # 允許更大的尺寸範圍

  nms:
    # 當前: iou_threshold: 0.35
    iou_threshold: 0.25  # 降低 NMS 閾值以保留更多候選
```

**預期效果**: 保留更多大型 masks，提升 IoU

---

### 🟡 **優先級 2: 改進推理策略選擇**

#### 方案 2.1: 使用 Hierarchical 策略
**觀察**: Hierarchical 覆蓋率 39.0% >> Independent 7.4%

**測試**:
```yaml
inference:
  test_all_strategies: false
  strategy: hierarchical  # 改用 hierarchical
```

**預期效果**: 覆蓋率提升至 30-40%

#### 方案 2.2: 使用 Exhaustive Pairing
**觀察**: Exhaustive 覆蓋率 34.8%，中等性能

**測試**:
```yaml
inference:
  test_all_strategies: false
  strategy: exhaustive_pairing
```

---

### 🟢 **優先級 3: 特徵質量優化**

#### 方案 3.1: 調整特徵聚合模式
**當前**: `pooling_mode: voting`

**測試**:
```yaml
aggregation:
  pooling_mode: average  # 改為 average，L2 正規化可能提升相似度
```

#### 方案 3.2: 降低 Temperature
**當前**: `retrieval.temperature: 0.2`

**測試**:
```yaml
inference:
  retrieval:
    temperature: 0.1  # 更銳利的相似度分數
```

---

### 🔵 **優先級 4: 數據質量檢查**

#### 方案 4.1: 修復 GT 無效 ID
**發現**: GT 中的 `11001` 不在 `VALID_JOINT_SEMANTIC_IDS` 中

**調查**:
1. 檢查 `11001` 是否是標註錯誤
2. 如果是有效 ID，更新 `VALID_JOINT_SEMANTIC_IDS`
3. 如果無效，從 GT 中移除

**文件**: `data/search3d_gt/multiscan_data_search3d/multiscan_search3d_constants.py`

#### 方案 4.2: 檢查 3D 投影質量
**可能問題**: 2D masks → 3D 投影過程丟失細節

**檢查**:
```yaml
aggregation:
  projection:
    vis_depth_threshold: 0.1  # 當前值
    resize_ratio: 0.3  # 檢查是否需要提高分辨率
```

---

## 快速測試腳本

### 測試 1: Hierarchical 策略
```bash
# 修改 configs/inference/full_pipeline.yaml:
#   strategy: hierarchical
#   test_all_strategies: false

conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation \
  2>&1 | tee logs/eval/test_hierarchical.log

# 檢查結果
tail -50 logs/eval/test_hierarchical.log | grep -A 3 "mAP:"
```

### 測試 2: 寬鬆聚合參數
```bash
# 修改 configs/inference/full_pipeline.yaml:
#   iou_threshold: 0.5
#   area_fraction: 0.001

conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  2>&1 | tee logs/eval/test_relaxed_agg.log

# 重新診斷
python3 diagnose_map_zero.py
```

### 測試 3: 組合優化
**最激進配置** (同時應用所有優化):
```yaml
aggregation:
  iou_threshold: 0.5
  area_fraction: 0.001
  area_fraction_per_level:
    2: 0.002
    4: 0.001
    6: 0.0005

inference:
  strategy: hierarchical
  retrieval:
    top_k_per_level: 500
    min_similarity: 0.005
  pairing:
    containment_threshold: 0.0001
  nms:
    iou_threshold: 0.25
```

---

## 預期改善路徑

### 階段 1: 策略切換 (立即見效)
- 從 Independent → Hierarchical
- **預期 mAP**: 0.000 → 0.01-0.05
- **覆蓋率**: 7.4% → 35-40%

### 階段 2: 參數調優 (中期)
- 寬鬆聚合 + 推理參數
- **預期 mAP**: 0.01-0.05 → 0.10-0.15
- **IoU**: 0.19 → 0.30-0.40

### 階段 3: 深度優化 (長期)
- 特徵提取改進
- 數據質量修復
- **預期 mAP**: 0.10-0.15 → 0.20-0.30

---

## 當前已修復問題總結

### ✅ 格式問題 (已修復)
1. **Search3D 格式**: 5 位 → 8 位實例 ID
2. **查詢過濾**: 使用 GT labels 過濾查詢

### ✅ 診斷工具 (已完成)
1. **diagnose_map_zero.py**: IoU 分析工具
2. **MAPerformance_DIAGNOSIS.md**: 完整診斷報告
3. **OPTIMIZATION_PLAN.md**: 本優化方案

---

## 下一步行動

### 立即執行 (今天)
1. ✅ 運行 `diagnose_map_zero.py` → **已完成**
2. 🔄 測試 Hierarchical 策略
3. 🔄 調整聚合參數並重新聚合

### 本週執行
1. 系統測試所有優化組合
2. 對比不同策略的 mAP
3. 分析 hierarchical vs independent 的差異

### 長期優化
1. 深入檢查 3D 投影質量
2. 調查 SigLIP 特徵提取
3. 可能需要調整 SAM2 追蹤參數（上游 pipeline）

---

## 參考資料

- **診斷日誌**: `/tmp/map_diagnosis.log`
- **診斷腳本**: `diagnose_map_zero.py`
- **配置文件**: `configs/inference/full_pipeline.yaml`
- **評估器**: `eval/search3d/eval_semantic_instance_parts_OV.py`
- **Search3D 常量**: `data/search3d_gt/multiscan_data_search3d/multiscan_search3d_constants.py`
