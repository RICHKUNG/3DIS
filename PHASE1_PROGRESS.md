# Phase 1 實驗進度報告

**實驗目標**: 測試不同特徵提取器是否能改善 mAP=0.0 問題
**日期**: 2025-11-17

---

## 實驗記錄

### P1-A: OpenCLIP laion2b ❌
**模型**: hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K (嘗試使用)
**狀態**: 失敗 (技術限制)

**結果**:
- mAP: 0.000 (實際使用 SigLIP,非預期)
- AP25: 0.000
- AP50: 0.000
- Predictions: 241

**問題**:
- 當前代碼僅支持 HuggingFace Hub 格式 (transformers 庫)
- OpenCLIP 格式 `hf-hub:...` 無法載入
- 系統回退到 SigLIP baseline

**日誌證據**:
```
2025-11-17 11:25:31,657 - ERROR - Failed to load SigLIP model:
Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden
```

**記錄**: logs/phase1/P1A_RECORD.md

---

### P1-B: OpenAI CLIP (HuggingFace) 🔄
**模型**: openai/clip-vit-large-patch14-336
**狀態**: 運行中 (PID: 4022579, CPU: 21.9%)

**進度**:
- ✓ 配置文件創建
- ✓ 符號鏈接設置 (aggregation_output, relations)
- ✓ 實驗啟動
- 🔄 模型載入中 (可能在下載中)
- ⏳ 等待推理完成...

**啟動時間**: 11:27
**預計完成**: ~11:30-11:35 (估計)

---

### P1-C: SigLIP Baseline ⏳
**模型**: google/siglip-so400m-patch14-384
**狀態**: 待執行 (作為對照組)

**目的**: 確認可重現性,與 P1-A/P1-B 對比

---

## 決策框架

**成功標準**:
```python
if map_clip > 0.01:
    decision = "✅ CLIP 有效,繼續 Phase 2 優化"
elif map_clip > 0.005:
    decision = "⚠️ 小幅改善,嘗試 Phase 2"
elif map_clip > map_siglip:
    decision = "⚠️ 微小改善,考慮 Phase 2 或直接 Phase 4"
else:
    decision = "❌ 模型切換無效,跳到 Phase 4 (fine-tuning)"
```

**下一步**:
- 等待 P1-B 完成
- 運行 P1-C baseline 對照
- 比較三組結果
- 決定是否進入 Phase 2

---

## 技術發現

### 代碼限制
- `SigLIPFeatureExtractor` 類僅支持 `transformers.AutoModel`
- 不支持 `open_clip` 庫
- 需要代碼修改才能使用 OpenCLIP 原生格式

### 解決方案
- **短期**: 使用 HuggingFace Hub 上的 CLIP/ViT 模型
- **長期**: 擴展代碼支持多種模型庫 (transformers, open_clip, etc.)

---

**更新時間**: 2025-11-17 11:29
**下次更新**: P1-B 完成後
