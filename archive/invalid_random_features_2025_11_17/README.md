已歸檔的無效實驗記錄 (使用了隨機特徵)

歸檔日期: 2025-11-17 21:44:44

## 原因
所有這些實驗都因為 PyTorch 2.5.0 無法加載 OpenAI CLIP 模型而使用了隨機特徵，
導致所有結果無效。

## 歸檔內容
- Phase 2-A 日誌
- Phase 3 所有實驗日誌 (A1, A1b, B1, B2, C1, C2)
- Phase 3 相關文檔

## 修復措施
1. 移除 feature_extraction.py 中的 fallback 機制
2. 轉換 OpenAI CLIP 到 safetensors 格式
3. 重新運行所有實驗

