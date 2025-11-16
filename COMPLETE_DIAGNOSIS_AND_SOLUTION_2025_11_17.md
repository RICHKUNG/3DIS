# 完整診斷與解決方案

**時間**: 2025-11-17 00:00
**問題**: mAP 表現不佳的完整分析與解決建議

---

## 📋 完整時間線

| 時間 | 運行 | 配置 | 結果 | 備註 |
|------|------|------|------|------|
| 17:00-18:00 | 初始優化 | containment=0.001, threshold=0.01 | **Hierarchical mAP=0.006** | ✅ 最佳結果 |
| 22:14 | 測試運行 | containment=0.0, threshold=0.005 | mAP=0.000 (所有策略) | ❌ 評估路徑錯誤 |
| 22:47 | Emergency Fix | containment=0.0, threshold=0.005 | Independent=0.003, Hierarchical=0.000 | ⚠️ Independent 成功 |
| 23:35 | 恢復配置 | containment=0.001, threshold=0.01 | mAP=0.000 | ❌ 約束過嚴 |

---

## 🔍 核心發現

### 發現 1: 所有運行的 Max IoU 都很低

| 運行 | Max IoU | mAP | AP25 |
|------|---------|-----|------|
| 18:00 (Hierarchical) | **未知**（文件已覆寫） | 0.006 | 0.050 |
| 22:47 (Emergency Hierarchical) | 0.1070 | 0.000 | 0.000 |
| 22:47 (Emergency Independent) | **未測** | 0.003 | 0.028 |
| 23:35 (Restore) | 0.0935 | 0.000 | 0.000 |

**關鍵洞察**: 
- 即使 18:00 的 "成功" 運行，AP25 = 0.050 意味著只有 5% 的 GT 得到 IoU >= 0.25 的匹配
- mAP = 0.006 是非常低的分數，說明整體匹配質量很差
- **這是一個更深層的問題，不只是參數調整的問題**

### 發現 2: Independent 策略的突破

**22:47 Emergency Fix Independent 策略**:
- mAP = 0.003
- AP25 = 0.028
- AP50 = 0.027

雖然分數仍然很低，但這是第一次非 Hierarchical 策略獲得非零 mAP！

### 發現 3: 評估路徑問題已解決

**問題**: 22:14 運行評估使用錯誤路徑
- 評估使用: `inference_output/scene_00093_01_obj_part_inst.txt`
- 應該使用: `inference_output/{strategy}/scene_00093_01_obj_part_inst.txt`

**現狀**: 
- 單策略模式下，預測直接寫到 `inference_output/` 根目錄
- 評估正確使用該文件
- **路徑問題已不是主要障礙**

---

## 🎯 根本原因分析

### 問題不是參數調整，而是模型匹配質量

**證據**:
1. **所有配置的 Max IoU < 0.25** → 幾乎沒有高質量匹配
2. **即使最佳運行 (18:00) 的 AP25 = 0.050** → 只有 5% 召回率
3. **不同參數的差異主要體現在預測數量，而非質量**

**可能的根本原因**:

#### 1. SigLIP 特徵匹配不準確 ⭐⭐⭐⭐⭐
- Combined query "X of Y" 的語義理解可能不夠精確
- SigLIP 是為圖像級檢索設計，可能不適合 part-level 匹配
- 特徵提取的 pooling 方式可能丟失空間信息

#### 2. 3D Proposal 質量問題 ⭐⭐⭐⭐
- 聚合階段（2D SAM2 → 3D proposals）可能引入誤差
- Proposals 的邊界不準確 → IoU 計算受影響
- Level 選擇可能不合適（L2/L4/L6 的粒度）

#### 3. 幾何配對邏輯缺陷 ⭐⭐⭐
- Containment threshold 可能基於錯誤假設
- Part-object 空間關係計算可能有bug
- Family tree 關係可能不準確

#### 4. Ground Truth 覆蓋率低 ⭐⭐
- GT 只覆蓋 34.7% 的點
- 許多正確預測可能因為 GT 不完整而被計為錯誤

---

## ✅ 成功案例：Independent 策略

**22:47 Emergency Fix 配置對 Independent 有效**:

```yaml
pairing:
  containment_threshold: 0.0     # 無幾何約束
  scale_range: [0.00001, 100.0]  # 極寬鬆

nms:
  use_soft_nms: true             # Soft NMS
  iou_threshold: 0.15            # 低閾值
  keep_top_k: 500                # 大 K 值

independent:
  top_k_per_level: 500           # 大候選池
  retrieval_threshold: 0.01      # 低閾值
  min_retrieval_threshold: 0.001 # 極低
```

**為何有效**:
- Independent 窮舉搜索 → 不依賴幾何假設
- 無約束 → 允許發現非典型 part-object 關係
- Soft NMS → 保留更多候選

---

## 📊 當前最佳配置

### 推薦配置：Emergency Fix 配置

**理由**:
1. ✅ Independent 策略已驗證有效 (mAP = 0.003)
2. ✅ 移除過嚴的幾何約束
3. ✅ 使用 Soft NMS 保留多樣性
4. ⚠️ Hierarchical 仍需優化，但 Independent 是可行替代

### 當前最佳 mAP 排行

| 策略 | mAP | AP50 | AP25 | 配置 |
|------|-----|------|------|------|
| 1️⃣ **Hierarchical** | **0.006** | **0.050** | **0.050** | 18:00 配置（已覆寫）|
| 2️⃣ **Independent** | **0.003** | **0.027** | **0.028** | Emergency Fix |
| 3️⃣ Exhaustive | 0.000 | 0.000 | 0.104 | Emergency Fix |

---

## 🚀 推薦行動

### 短期行動（立即可行）

#### 方案 A: 保留 Emergency Fix 配置（推薦）✅

**優點**:
- Independent 策略已驗證
- 參數簡單（無約束）
- 可進一步調優

**執行**:
```bash
# 恢復 Emergency Fix 配置
git checkout configs/inference/full_pipeline.yaml  # 如果已提交

# 或手動設置
pairing:
  containment_threshold: 0.0
  scale_range: [0.00001, 100.0]

nms:
  use_soft_nms: true
  iou_threshold: 0.15
  keep_top_k: 500
```

**預期**: Independent mAP >= 0.003

#### 方案 B: 重新運行 18:00 配置以驗證

**目標**: 驗證是否能重現 mAP = 0.006

**執行**:
```bash
# 設置 18:00 參數
pairing:
  containment_threshold: 0.001
  scale_range: [0.0001, 20.0]

hierarchical:
  coarse_top_k: 200    # 恢復原始值
  object_top_k: 200
  part_top_k: 100
  coarse_threshold: 0.01
  refinement_threshold: 0.01

nms:
  use_soft_nms: false
  iou_threshold: 0.25
  keep_top_k: 300
```

**如果成功**: 證明 top-k 增加有害  
**如果失敗**: 證明問題更深層（見長期行動）

---

### 中期行動（1-2 天）

#### 1. 診斷 SigLIP 特徵質量

**檢查 SigLIP 是否真的能匹配 part-level 語義**:

```python
# 測試腳本
from my3dis.inference.feature_extraction import SigLIPFeatureExtractor

extractor = SigLIPFeatureExtractor("google/siglip-so400m-patch14-384")

# 測試 combined query 相似度
queries = [
    "door of cabinet",
    "handle of door", 
    "leg of chair"
]

# 與 GT 對比，看相似度是否有區分度
```

**如果 SigLIP 相似度分布不合理** → 考慮其他特徵提取方法

#### 2. 分析 3D Proposal 邊界質量

**檢查 proposals 與 GT 的空間對齊**:

```python
# 計算 proposal 與 GT 的最佳 IoU（不考慮語義）
# 如果最佳 IoU 仍然很低 → 問題在聚合階段
```

**如果 proposals 邊界質量差** → 調整聚合參數

#### 3. 實現策略特定參數覆蓋

**目標**: 允許 Independent 和 Hierarchical 使用不同參數

**代碼修改**:
```python
# src/my3dis/inference/inference_pipeline.py

# 當前
pairing_cfg = self.config.pairing

# 修改為
strategy_name = self.config.strategy
if hasattr(self.config, strategy_name) and hasattr(getattr(self.config, strategy_name), 'pairing_override'):
    pairing_cfg = {**self.config.pairing, **self.config[strategy_name].pairing_override}
else:
    pairing_cfg = self.config.pairing
```

---

### 長期行動（1-2 週）

#### 1. 評估其他特徵提取方法

**候選**:
- CLIP (原始 OpenAI)
- OpenCLIP variants
- DINOv2 + language alignment
- PointCLIP (3D-native)

#### 2. 改進 Combined Query 策略

**問題**: "X of Y" 可能過於抽象

**替代方案**:
- 分開檢索 object 和 part，後處理配對
- 使用視覺 grounding（定位 part 在 object 中的位置）
- 多模態融合（text + visual + geometric）

#### 3. 調整聚合階段參數

**目標**: 改善 3D proposal 質量

**參數**:
```yaml
aggregation:
  iou_threshold: 0.5 → 0.6 (更嚴格合併)
  area_fraction: 0.001 → 0.0005 (保留更小區域)
```

---

## 📝 結論

### 當前狀態

1. **最佳 mAP: 0.006** (Hierarchical, 18:00 配置)
2. **Independent 突破: 0.003** (Emergency Fix 配置)
3. **問題本質**: 特徵匹配質量不足，而非單純參數問題

### 立即建議

**優先級 1**: 保留 Emergency Fix 配置
- 確保 Independent 策略可用 (mAP = 0.003)
- 作為 baseline 繼續優化

**優先級 2**: 嘗試重現 18:00 成功配置
- 驗證 Hierarchical mAP = 0.006 是否可重現
- 如果成功，兩個策略都可用

**優先級 3**: 深入診斷根本原因
- SigLIP 特徵質量
- 3D proposal 邊界
- Part-level 匹配的語義理解

### 現實期望

**短期（本週內）**:
- 目標: 穩定達到 mAP >= 0.006
- 策略: 參數調優

**中期（本月內）**:
- 目標: mAP >= 0.015 (2.5x 提升)
- 策略: 特徵提取或聚合改進

**長期（下個月）**:
- 目標: mAP >= 0.050 (接近實用水平)
- 策略: 可能需要架構級改進

---

## 🎯 最終推薦

**立即執行**:

```yaml
# configs/inference/full_pipeline.yaml

inference:
  test_all_strategies: false
  strategy: independent  # 使用驗證有效的策略

pairing:
  containment_threshold: 0.0
  scale_range: [0.00001, 100.0]

nms:
  use_soft_nms: true
  iou_threshold: 0.15
  keep_top_k: 500

independent:
  top_k_per_level: 500
  retrieval_threshold: 0.01
  min_retrieval_threshold: 0.001
```

**預期結果**: Independent mAP >= 0.003 (已驗證)

**下一步**: 系統性診斷特徵匹配質量問題

---

**文檔**: 
- 技術分析: `/tmp/hierarchical_regression_analysis.md`
- Emergency Fix: `EMERGENCY_FIX_APPLIED.md`
- Hierarchical 恢復: `HIERARCHICAL_RESTORE_FIX.md`
- 完整診斷: 本文檔

**代碼狀態**: 
- 配置文件: 當前為恢復配置（約束過嚴）
- 需回滾至: Emergency Fix 配置
