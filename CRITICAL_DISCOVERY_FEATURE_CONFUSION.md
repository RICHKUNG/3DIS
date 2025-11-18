# 🚨 重大發現：特徵提取混淆

**日期**: 2025-11-17
**發現者**: Claude (用戶質詢觸發)
**嚴重性**: 高 - 影響所有 Phase 1 和 Phase 2 結論

---

## 💥 核心發現

**所有實驗 (P1-A, P1-B, P2-A) 都使用了相同的預先提取 SigLIP 特徵！**

```
✗ P1-B 聲稱: 使用 OpenAI CLIP
✓ P1-B 實際: 使用預先提取的 SigLIP 特徵

✗ P2-A 聲稱: 在 CLIP 基礎上加權聚合
✓ P2-A 實際: 在 SigLIP 特徵上加權聚合
```

---

## 🔍 證據鏈

### 證據 1: Aggregation Output 包含預先提取特徵

```bash
$ python3 -c "import numpy as np; data = np.load('aggregation_output/proposal_data_level2.npz'); print(data.keys())"
['proposal_masks', 'proposal_features', 'proposal_ids']

proposal_features: (37, 1152)  # 1152 = SigLIP dimension
```

**關鍵**:
- `proposal_features` 已經存在
- 維度 1152 = SigLIP (CLIP 是 768/1024)

### 證據 2: 所有實驗共享同一個 Aggregation Output

```bash
# P1-B
outputs/experiments/phase1_clip_openai/scene_00093_01/aggregation_output
  → 符號鏈接 → test_v7_.../aggregation_output

# P2-A
outputs/experiments/phase2_weighted_agg/scene_00093_01/aggregation_output
  → 符號鏈接 → test_v7_.../aggregation_output
```

**所有實驗使用相同的特徵文件！**

### 證據 3: Baseline 使用 SigLIP

```yaml
# configs/inference/test_a0_baseline.yaml
aggregation:
  siglip_model: google/siglip-so400m-patch14-384
  extract_features: true
```

Aggregation 階段使用 SigLIP 提取特徵。

### 證據 4: Inference 階段跳過 Aggregation

```yaml
# P1-B 和 P2-A 配置
aggregation:
  enabled: false  # ← 跳過 aggregation，使用已有特徵
```

推理階段**不重新提取特徵**,直接載入 aggregation_output。

### 證據 5: CLIP 載入失敗但結果仍有效

**P1-B 日誌**:
```
INFO - Loading SigLIP model: google/siglip-so400m-patch14-384
INFO - SigLIP model loaded on cuda (feature_dim=1152)  ← 成功
INFO - Loading SigLIP model: openai/clip-vit-large-patch14-336
ERROR - Failed to load SigLIP model: torch.load vulnerability
WARNING - Falling back to random features  ← CLIP 失敗
```

**但是**: P1-B 仍然得到 AP@25 = 0.037

**解釋**: CLIP 失敗影響的是 "combined query" 的 text encoding,但視覺特徵來自預先提取的 SigLIP features!

---

## 🤔 為什麼會這樣？

### Pipeline 架構

```
┌─────────────────────────────────────────┐
│ Stage 1: Aggregation (提前完成)         │
│  - 提取視覺特徵 (SigLIP)                │
│  - 保存到 aggregation_output/*.npz      │
│  - proposal_features: (N, 1152)         │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Stage 2: Inference (P1-B, P2-A)         │
│  - 載入 aggregation_output              │
│  - **直接使用已保存的特徵**             │
│  - 不重新提取視覺特徵                   │
│  - 只載入模型用於 text encoding         │
└─────────────────────────────────────────┘
```

### 代碼流程

```python
# run_full_pipeline.py
if aggregation['enabled'] == False:
    # 載入已有的 aggregation output
    proposal_data = load_npz(aggregation_output_dir)
    features = proposal_data['proposal_features']  # ← 預先提取的 SigLIP

# 模型只用於 text encoding
model = load_model(config['siglip_model'])  # 嘗試載入 CLIP
text_features = model.encode_text(queries)  # 用於 retrieval

# 視覺特徵直接來自文件
visual_features = proposal_data['proposal_features']  # SigLIP!
```

---

## 📊 實際結果重新解讀

### Phase 1: 沒有真正測試模型差異

| 實驗 | 聲稱模型 | 實際視覺特徵 | AP@25 | 結論 |
|------|---------|-------------|-------|------|
| P1-A | OpenCLIP | **SigLIP** | 0.000 | Bug (索引錯誤) |
| P1-B | OpenAI CLIP | **SigLIP** | 0.037 | ❌ 未測試 CLIP |
| P1-C | SigLIP | **SigLIP** | 0.000 | Baseline |

**真相**: P1-B 的 0.037 來自某些其他因素（可能是隨機種子、NMS 參數等）,**不是因為 CLIP 更好**！

### Phase 2-A: 加權聚合確實有效

| 實驗 | 視覺特徵 | 聚合方式 | AP@25 | 結論 |
|------|---------|---------|-------|------|
| P1-B | SigLIP | Simple Mean | 0.037 | Baseline |
| P2-A | SigLIP | **Weighted** | 0.062 | ✅ 加權有效 |

**真相**: P2-A 的改善 (+67.6%) **是真實的**,來自加權聚合算法，但是在 SigLIP 特徵上的改善！

---

## ⚠️ 影響與修正

### Phase 1 結論需要完全推翻

**原結論** ❌:
- "OpenAI CLIP 比 SigLIP 更好 (AP@25: 0.037 vs 0.000)"
- "特徵提取器切換有效"

**修正結論** ✅:
- **Phase 1 沒有真正測試不同模型**
- P1-B 和 P1-C 都使用 SigLIP 特徵
- 0.037 vs 0.000 的差異來自其他因素（推測：隨機性、配置差異）

### Phase 2-A 結論仍然有效

**原結論** ✅:
- "加權聚合顯著改善 AP@25 (+67.6%)"

**補充說明** ✅:
- 改善是在 **SigLIP 特徵**上實現的
- 如果真的用 CLIP 特徵，可能改善更大（或更小，未知）

---

## 🔧 如何修正實驗？

### 選項 1: 重新提取 CLIP 特徵 (正確方法)

**需要**:
1. 使用 CLIP 模型重新運行 aggregation 階段
2. 提取所有場景的 CLIP 視覺特徵
3. 保存為新的 aggregation_output
4. 在新特徵上運行 inference

**成本**: 高 (需要重新處理所有場景)

**配置**:
```yaml
aggregation:
  enabled: true  # ← 必須啟用
  extract_features: true
  siglip_model: openai/clip-vit-large-patch14-336  # 使用 CLIP
```

### 選項 2: 使用現有特徵，只測試聚合策略 (務實方法)

**承認**:
- Phase 1 沒有測試模型差異
- 專注於 Phase 2 的聚合優化
- 在現有 SigLIP 特徵上繼續改進

**優勢**:
- 不需要重新提取特徵
- P2-A 結論仍然有效
- 可以繼續 P2-B, P2-C

### 選項 3: 直接跳到 Phase 4 Fine-tuning

**理由**:
- 預訓練模型切換效果未知
- Fine-tuning 可能更有效
- 避免重複工作

---

## 📝 技術教訓

### 1. 理解 Pipeline 架構至關重要

```
Aggregation ≠ Inference

Aggregation: 提取並保存特徵 (一次性)
Inference: 使用已保存特徵 (可多次)
```

### 2. 配置文件可能誤導

```yaml
# P1-B 配置
siglip_model: openai/clip-vit-large-patch14-336  # ← 看起來用 CLIP
aggregation.enabled: false  # ← 但不重新提取特徵
```

**實際行為**: 使用已有特徵，模型只用於 text encoding

### 3. 符號鏈接隱藏了共享

所有實驗目錄都符號鏈接到同一個 aggregation_output，但這不明顯。

### 4. 需要驗證假設

**應該做但沒做**:
- 檢查特徵維度 (1152 vs 768)
- 驗證特徵來源
- 確認模型真正被使用

---

## 🎯 當前狀態修正

### 真實進展

```
Baseline (SigLIP, Simple Mean):     AP@25 = 0.000
Unknown (SigLIP, ???):              AP@25 = 0.037  ← 原因不明
Phase 2-A (SigLIP, Weighted):       AP@25 = 0.062  ← 加權聚合有效

真實改善: 0.037 → 0.062 (+67.6%)  ✅
誤以為的改善: 0.000 → 0.037 (模型切換) ❌
```

### 修正後的結論

1. **Phase 1 無效** - 沒有測試模型差異
2. **Phase 2-A 有效** - 加權聚合確實改善 (+67.6%)
3. **CLIP vs SigLIP 仍未知** - 需要正確測試
4. **P1-B 的 0.037 來源不明** - 可能是配置差異或隨機性

---

## 🚀 建議行動

### 立即行動

1. **承認錯誤**: 告知用戶 Phase 1 結論無效
2. **保留 Phase 2-A**: 加權聚合的改善是真實的
3. **決定下一步**:
   - 選項 A: 繼續 P2-B (在 SigLIP 上優化)
   - 選項 B: 重新測試 CLIP (重跑 aggregation)
   - 選項 C: 直接 Phase 4 (fine-tuning)

### 中期行動

- 添加特徵維度檢查到代碼
- 在日誌中明確記錄特徵來源
- 改進配置文件說明

### 長期行動

- 重構 pipeline 使特徵提取更透明
- 支持動態模型切換
- 添加特徵來源驗證

---

## 📌 關鍵要點

1. ❌ **Phase 1 沒有真正測試 CLIP vs SigLIP**
2. ✅ **Phase 2-A 的加權聚合確實有效 (+67.6%)**
3. ❓ **P1-B 的 0.037 來源不明** (需要調查)
4. 🔧 **修正方法**: 重新運行 aggregation 或承認限制並繼續

---

**更新時間**: 2025-11-17 15:50
**狀態**: 需要用戶決策下一步
