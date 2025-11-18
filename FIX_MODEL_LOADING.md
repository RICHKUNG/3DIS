# 修復模型加載問題指南

**問題**: PyTorch 2.5.0 無法加載 OpenAI CLIP (pytorch_model.bin 格式)
**解決**: 轉換模型到 safetensors 格式

---

## 方案 1: 轉換 OpenAI CLIP 到 Safetensors (推薦)

### 步驟 1: 運行轉換腳本

```bash
cd /media/Pluto/richkung/My3DIS

# 啟動正確的環境
conda activate 3Dsiglip

# 轉換模型
PYTHONPATH=src python scripts/convert_clip_to_safetensors.py \
  --model openai/clip-vit-large-patch14-336 \
  --output ./models/clip-vit-large-safetensors
```

**預期輸出**:
```
Loading model from: openai/clip-vit-large-patch14-336
Saving model with safetensors to: ./models/clip-vit-large-safetensors
✅ Model converted successfully!

Saved files:
  - config.json
  - model.safetensors
  - preprocessor_config.json
  - tokenizer.json
  - tokenizer_config.json
  - ...

✅ Found 1 safetensors files
✅ Model test passed! Feature dim: 768
```

### 步驟 2: 更新配置文件

編輯你的配置文件,將模型路徑改為本地轉換後的路徑:

```yaml
# 例如: configs/inference/phase3_exp_b2_scale50.yaml

inference:
  hierarchical:
    siglip_model: /media/Pluto/richkung/My3DIS/models/clip-vit-large-safetensors  # ⭐ 使用本地路徑
    siglip_device: cuda:0
```

### 步驟 3: 測試模型加載

```bash
# 快速測試
PYTHONPATH=src python -c "
from my3dis.inference.feature_extraction import SigLIPFeatureExtractor
extractor = SigLIPFeatureExtractor(
    model_name='/media/Pluto/richkung/My3DIS/models/clip-vit-large-safetensors',
    device='cuda:0'
)
print('✅ Model loaded successfully!')
print(f'Feature dim: {extractor.feature_dim}')
"
```

**成功輸出**:
```
Loading SigLIP model: /media/Pluto/richkung/My3DIS/models/clip-vit-large-safetensors
SigLIP model loaded on cuda:0 (feature_dim=768)
✅ Model loaded successfully!
Feature dim: 768
```

**失敗輸出** (會立即終止,不會使用隨機特徵):
```
Failed to load SigLIP model: ...
❌ CRITICAL: Model loading failed. Terminating to prevent invalid results.
RuntimeError: Failed to load SigLIP model ...
```

---

## 方案 2: 使用支持 Safetensors 的替代模型

### 選項 A: LAION CLIP

```yaml
inference:
  hierarchical:
    siglip_model: laion/CLIP-ViT-L-14-laion2B-s32B-b82K
```

**優點**: 可能已經有 safetensors 格式
**缺點**: 需要驗證性能

### 選項 B: 保持使用 SigLIP

```yaml
inference:
  hierarchical:
    siglip_model: google/siglip-so400m-patch14-384  # ⭐ 這個模型有 safetensors
```

**優點**: 確定可以加載
**缺點**: 與 Phase 2-A 不一致

---

## 方案 3: 升級 PyTorch (不推薦)

**問題**: PyTorch 2.6 可能還未正式發布

```bash
# 檢查是否有 2.6
pip index versions torch

# 如果有,升級
conda activate 3Dsiglip
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**風險**:
- 可能破壞其他依賴
- 可能影響 SigLIP 加載

---

## 驗證修復

### 1. 檢查代碼修改

```bash
grep -A 5 "Failed to load SigLIP model" src/my3dis/inference/feature_extraction.py
```

**應該看到**:
```python
except Exception as e:
    logger.error(f"Failed to load SigLIP model: {e}")
    logger.error("❌ CRITICAL: Model loading failed. Terminating to prevent invalid results.")
    raise RuntimeError(...)  # ⭐ 不再 fallback
```

### 2. 運行測試實驗

```bash
# 使用轉換後的模型
PYTHONPATH=src timeout 60 conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/test_model_loading.yaml \
  2>&1 | tee logs/test_model_loading.log

# 檢查日誌
grep "SigLIP model loaded\|Failed to load\|random features" logs/test_model_loading.log
```

**成功輸出**:
```
SigLIP model loaded on cuda (feature_dim=1152)
SigLIP model loaded on cuda (feature_dim=768)
```

**失敗輸出** (會立即終止):
```
Failed to load SigLIP model: ...
❌ CRITICAL: Model loading failed. Terminating to prevent invalid results.
RuntimeError: ...
```

### 3. 確認沒有隨機特徵

```bash
# 搜索所有實驗日誌
grep -r "random features\|Falling back" logs/

# 應該沒有任何輸出 (或只有舊日誌)
```

---

## 重新運行實驗

### 步驟 1: 清理舊結果

```bash
# 標記舊結果為無效
mkdir -p logs/phase3_invalid
mv logs/phase3/*.log logs/phase3_invalid/

mkdir -p outputs/experiments_invalid
mv outputs/experiments/phase3_* outputs/experiments_invalid/
```

### 步驟 2: 使用正確模型重新運行

**最簡單的測試**: 只運行 scale=50 實驗

```yaml
# configs/inference/phase3_valid_b2_scale50.yaml
inference:
  hierarchical:
    siglip_model: /media/Pluto/richkung/My3DIS/models/clip-vit-large-safetensors  # ⭐
    scale_semantic_score: 50.0
    apply_softmax: true
```

```bash
PYTHONPATH=src conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/phase3_valid_b2_scale50.yaml \
  2>&1 | tee logs/phase3_valid/exp_b2_valid.log
```

### 步驟 3: 驗證結果有效性

```bash
# 檢查模型加載
grep "SigLIP model loaded" logs/phase3_valid/exp_b2_valid.log

# 應該看到兩行 (SigLIP + CLIP)
# SigLIP model loaded on cuda (feature_dim=1152)
# SigLIP model loaded on cuda (feature_dim=768)

# 檢查沒有隨機特徵
grep "random" logs/phase3_valid/exp_b2_valid.log
# 應該沒有輸出

# 查看結果
grep "AP@25" logs/phase3_valid/exp_b2_valid.log
```

---

## 預期結果

### 修復前 (無效)
```
AP@25: 0.075  # ❌ 隨機特徵
AP@25: 0.062  # ❌ 隨機特徵
AP@25: 0.000  # ❌ 隨機特徵
```

### 修復後 (有效)
```
AP@25: ????  # ✅ 真實 CLIP 特徵
```

**重要**: 修復後的結果可能與之前完全不同,因為之前使用的是隨機特徵！

---

## 總結

**立即執行**:
1. ✅ 代碼已修改 (移除 fallback)
2. ⏳ 運行轉換腳本
3. ⏳ 更新配置文件
4. ⏳ 測試模型加載
5. ⏳ 重新運行實驗

**檢查點**:
- [ ] 轉換腳本成功
- [ ] 測試加載成功
- [ ] 配置文件更新
- [ ] 重新運行實驗
- [ ] 驗證沒有 "random features" 日誌
- [ ] 獲得有效的實驗結果

**預計時間**: ~30分鐘
