# SigLIP Feature Extraction 流程分析 - 發現的問題

**日期**: 2025-11-16
**分析對象**: `3DprojToSiglip/notebook_pipeline.py` 和 `utils_new/feature_extraction.py`

---

## 🔴 核心問題總結

經過仔細審視代碼，我發現了 **5 個嚴重的邏輯問題**，這些問題可以完全解釋為什麼 visual features 與 text features 的相似度如此之低 (0.03-0.06 而非預期的 >0.5)。

---

## 問題 1: 不合理的特徵累加策略 ⚠️⚠️⚠️

### 🔍 問題描述

**位置**:
- `notebook_pipeline.py:153`
- `feature_extraction.py:176, 178`

**代碼**:
```python
# Original version (notebook_pipeline.py:153)
pc_features[batch_index[k]] += image_features[k]

# Batched version (feature_extraction.py:176)
pc_features_sparse[sparse_idxes_tensor] += frame_feature

# Then later (line 157/184)
pc_features = F.normalize(pc_features, dim=1)
```

**問題**:
1. 每個 3D 點的 feature 是通過 **累加** 多個 crops/views 的 features 得到的
2. 如果一個點在多個視角中可見，它的 feature 會被累加多次
3. 如果一個點在同一視角的 3 個不同 scale crops 中都出現，會被累加 3 次
4. 最後只做一次 L2 normalize

### ❌ 為什麼這是錯的

**數學上的問題**:
```
假設點 p 在 3 個視角 (v1, v2, v3) 中可見，每個視角有 3 個 scale crops:

原始 feature = feat_v1_s1 + feat_v1_s2 + feat_v1_s3 +
               feat_v2_s1 + feat_v2_s2 + feat_v2_s3 +
               feat_v3_s1 + feat_v3_s2 + feat_v3_s3

normalize(原始) ≠ average([feat_v1, feat_v2, feat_v3])
```

累加後再 normalize 會導致：
- **方向偏差**: 如果某個視角的 features 較大，會主導最終方向
- **語義丟失**: 多個不同語義的 features 累加後，可能抵消彼此的語義信息
- **不公平**: 在更多視角中可見的點會有不同的特徵分布

### ✅ 正確的做法

應該是 **先 normalize 每個 crop feature，然後平均**:

```python
# 正確的方式
normalized_features = []
for crop_feat in crop_features:
    normalized_features.append(crop_feat / crop_feat.norm())

point_feature = torch.stack(normalized_features).mean(dim=0)
# 或者再做一次 normalize (可選)
# point_feature = point_feature / point_feature.norm()
```

### 🎯 影響嚴重程度: **CRITICAL**

這個問題會導致 features 完全失去語義一致性。

---

## 問題 2: 每個視角產生 3 個不同大小的 crops，都被累加 ⚠️⚠️

### 🔍 問題描述

**位置**: `notebook_pipeline.py:124-136`

**代碼**:
```python
kexp = 0.2

for _ in range(3):  # 產生 3 個 crops
    crop = images[v][x1:x2, y1:y2, :]
    if crop.size != 0:
        crop_tensor = processor(images=crop, return_tensors="pt")["pixel_values"].to(device)
        cropped_regions.append(crop_tensor)
        batch_index.append(torch.tensor(pts_idx[v], device=device, dtype=torch.long))

    # 每次擴大 bbox 20%
    dx = (x2 - x1) * kexp
    dy = (y2 - y1) * kexp
    x1 = max(0, int(x1 - dx))
    y1 = max(0, int(y1 - dy))
    x2 = min(H, int(x2 + dx))
    y2 = min(W, int(y2 + dy))

# 然後所有 3 個 crops 的 features 都被累加到同一組點上 (line 153)
for k in trange(len(cropped_regions), leave=False):
    pc_features[batch_index[k]] += image_features[k]
```

**問題**:
1. **第 1 次 crop**: 只包含物體本身 (tight bbox)
2. **第 2 次 crop**: bbox 擴大 20%，包含一些背景
3. **第 3 次 crop**: bbox 擴大 44% (1.2² ≈ 1.44)，包含更多背景

這 3 個 crops 包含**不同程度的背景噪聲**，但它們的 features 都被累加到同一組點上！

### ❌ 為什麼這是錯的

**語義稀釋**:
```
假設:
- crop1 (tight): feature 代表 "cabinet door"  (100% 物體)
- crop2 (medium): feature 代表 "cabinet door + wall" (70% 物體, 30% 背景)
- crop3 (loose): feature 代表 "cabinet door + wall + floor" (50% 物體, 50% 背景)

累加後: feature ∝ "cabinet door" × 2.2 + "wall" × 0.8 + "floor" × 0.5
```

背景噪聲會稀釋物體的語義信息！

### ✅ 正確的做法

應該只使用 **一個最佳大小的 crop**，或者對 3 個 scales **加權平均** (tight crop 權重最高)：

```python
# 方案 A: 只用 tight crop
crop = images[v][x1:x2, y1:y2, :]
feature = extract_feature(crop)

# 方案 B: 加權平均 (tight crop 權重更高)
weights = [1.0, 0.5, 0.25]  # tight, medium, loose
weighted_feature = sum(w * f for w, f in zip(weights, crop_features))
```

### 🎯 影響嚴重程度: **HIGH**

小物體受影響更嚴重，因為它們的 tight bbox 更容易被背景稀釋。

---

## 問題 3: 硬編碼的可見點數閾值 (>20) 對小物體太嚴格 ⚠️

### 🔍 問題描述

**位置**:
- `notebook_pipeline.py:100`
- `feature_extraction.py:113`

**代碼**:
```python
visible_counts = torch.tensor([pts_count.get(f, 0) for f in range(num_frames)], device=device)
valid = torch.nonzero(visible_counts > 20).view(-1)  # 硬編碼閾值 20
if len(valid) == 0:
    continue  # 跳過這個 proposal
```

**問題**:
- 只有當 frame 中有 **超過 20 個可見點** 時，該 frame 才會被考慮
- 對於小物體 (如 cabinet door 只有 309-347 總點數)，很多 frames 可能都達不到這個閾值

### ❌ 為什麼這是錯的

**實例分析 (cabinet door)**:
```
總點數: 347
假設平均分散在 20 個 frames:
每個 frame 平均: 347 / 20 = 17.35 點

結果: 大部分 frames 都會被過濾掉！
```

這意味著：
- **可選視角很少**: cabinet door 可能只有 2-3 個 frames 超過 20 點
- **TopK 無效**: 如果 valid frames < topk，TopK 選擇沒有意義
- **特徵不穩定**: 基於極少數視角的 features 會很不穩定

### ✅ 正確的做法

**動態閾值**，根據 proposal 大小調整:

```python
min_points_threshold = max(5, int(proposal_size * 0.05))  # 至少 5% 的點
valid = torch.nonzero(visible_counts > min_points_threshold).view(-1)
```

或者 **相對閾值**:
```python
# 使用百分位數而非絕對值
threshold = torch.quantile(visible_counts[visible_counts > 0], 0.3)  # 前 70% 的 frames
valid = torch.nonzero(visible_counts > threshold).view(-1)
```

### 🎯 影響嚴重程度: **HIGH**

嚴重影響小物體的 feature extraction。

---

## 問題 4: TopK 視角選擇標準不佳 ⚠️

### 🔍 問題描述

**位置**: `notebook_pipeline.py:103`

**代碼**:
```python
top_ids = torch.topk(visible_counts[valid], k=min(topk, len(valid)), largest=True).indices
views = valid[top_ids]
```

**問題**:
TopK 選擇僅基於 **可見點數量最多** 的 frames，但：
- 點數多 ≠ 視角好
- 點數多 ≠ 語義清晰
- 點數多 ≠ 遮擋少

### ❌ 為什麼這是錯的

**反例**:
```
Frame A: 50 個可見點，但大部分被遮擋，物體只露出一小角
Frame B: 25 個可見點，但完整清晰地顯示整個物體正面

當前算法: 選擇 Frame A (50 > 25)
正確選擇: 應該選 Frame B (更清晰完整)
```

這解釋了為什麼 **topk=5 比 topk=1 更差**：
- topk=5 會選擇更多「點數多但質量差」的視角
- 這些低質量視角的 features 會稀釋高質量視角的 features
- 累加策略更放大了這個問題

### ✅ 正確的做法

**多因素評分**:

```python
def compute_view_quality(visible_count, bbox_area, center_distance, occlusion_score):
    # 綜合考慮:
    # 1. 可見點數 (但不是唯一因素)
    # 2. Bbox 面積 (更大更好，說明物體更清晰)
    # 3. 到圖像中心的距離 (中心更好)
    # 4. 遮擋程度 (基於 depth consistency)

    score = (
        0.3 * normalize(visible_count) +
        0.4 * normalize(bbox_area) +
        0.2 * (1 - normalize(center_distance)) +
        0.1 * (1 - occlusion_score)
    )
    return score

top_views = torch.topk(view_scores, k=topk).indices
```

### 🎯 影響嚴重程度: **MEDIUM-HIGH**

這個問題解釋了為什麼增加 topk 反而降低性能。

---

## 問題 5: Batched version 在處理 multi-scale crops 時的不一致 ⚠️

### 🔍 問題描述

**位置**: `feature_extraction.py:164-165`

**代碼**:
```python
# Batched version (feature_extraction.py)
with torch.no_grad():
    feats = model.get_image_features(pixel_values=crops.to(device))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    # Average across scales
    frame_feature = feats.mean(dim=0)  # [D]

# 然後累加 (line 176)
pc_features_sparse[sparse_idxes_tensor] += frame_feature
```

vs

```python
# Original version (notebook_pipeline.py)
with torch.no_grad():
    for img_batch in batches:
        feats = model.get_image_features(pixel_values=img_batch.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        image_features.append(feats)

# 直接累加所有 crops (line 153)
for k in trange(len(cropped_regions), leave=False):
    pc_features[batch_index[k]] += image_features[k]
```

**問題**:
1. **Batched version**: 先對 3 個 scales 的 features 做平均，再累加到點上
2. **Original version**: 直接累加所有 3 個 scales 的 features 到點上

這兩種方式數學上不等價，但都有問題！

### ❌ 為什麼這是錯的

**Batched version 稍好** (因為做了 scale averaging)，但仍然：
- 使用累加而非平均 (問題 1)
- 3 個 scales 權重相等，忽略了 tight crop 應該更重要

**Original version 更差**:
- 直接累加 3 個 scales
- 沒有任何 averaging

### ✅ 正確的做法

統一使用 **weighted averaging**:

```python
# 1. 對 3 個 scales 加權平均
scale_weights = torch.tensor([1.0, 0.7, 0.5], device=device)  # tight > medium > loose
frame_feature = (feats * scale_weights.view(-1, 1)).sum(dim=0) / scale_weights.sum()

# 2. 跨 frames 也應該加權平均，而非累加
# 記錄每個點被累加了多少次
point_counts[point_indices] += 1
pc_features[point_indices] += frame_feature

# 3. 最後除以累加次數 (而非直接 normalize)
pc_features = pc_features / point_counts.clamp_min(1).unsqueeze(-1)
pc_features = F.normalize(pc_features, dim=1)
```

### 🎯 影響嚴重程度: **MEDIUM**

兩個版本不一致會導致結果難以重現。

---

## 🔥 問題優先級排序

基於對 feature quality 的影響程度：

1. **🔴 CRITICAL - 問題 1**: 累加而非平均 (導致特徵語義混亂)
2. **🟠 HIGH - 問題 2**: Multi-scale crops 都被累加 (背景噪聲稀釋)
3. **🟠 HIGH - 問題 3**: 硬編碼閾值 20 (小物體被過度過濾)
4. **🟡 MEDIUM-HIGH - 問題 4**: TopK 選擇標準差 (解釋 topk=5 worse)
5. **🟡 MEDIUM - 問題 5**: 版本不一致 (影響可重現性)

---

## 📊 這些問題如何解釋觀察到的現象

### 現象 1: 原始相似度極低 (0.03-0.06 vs 預期 >0.5)

**解釋**:
- **問題 1** (累加): 導致 features 方向偏離語義中心
- **問題 2** (背景噪聲): 稀釋物體語義信息
- 結果: visual features 不再與 text features 對齊

### 現象 2: Cabinet door 完全失敗 (0.023-0.045)

**解釋**:
- **問題 3** (閾值 20): Cabinet door (347 點) 很多 frames 都 < 20 點，被過濾
- **問題 2** (背景): 小物體的 tight bbox 更容易被背景稀釋
- **問題 1** (累加): 基於極少數 frames 的累加更不穩定
- 結果: Cabinet door 的 features 幾乎是隨機的

### 現象 3: topk=5 比 topk=1 更差 (40% vs 60%)

**解釋**:
- **問題 4** (TopK 選擇): 選擇了更多「點數多但質量差」的視角
- **問題 1** (累加): 低質量視角的 features 被累加，稀釋高質量視角
- **問題 2** (multi-scale): 更多視角 × 3 scales = 更多噪聲累加
- 結果: 增加視角反而引入更多噪聲

### 現象 4: Door frame 勉強成功 (0.064)

**解釋**:
- Door frame (2149 點) 足夠大，在很多 frames 都 > 20 點
- 即使有累加和背景噪聲的問題，因為樣本多所以相對穩定
- 但仍然比預期低 (0.064 << 0.5)

---

## ✅ 建議的修復方案

### 短期方案 (Quick Fix - 1-2 天)

**目標**: 快速驗證假設，看修復後是否能達到 >0.5 的相似度

**修改位置**: `3DprojToSiglip/notebook_pipeline.py` 或創建新版本

**關鍵修改**:

```python
def generate_grounding_features_siglip_fixed(
    model, processor, color_paths, inside_mask, projected_points,
    points_depth, depth_maps, device, scaling_params,
    vis_depth_threshold, proposal_masks, topk=1,
    batch_size=32
):
    """Fixed version with proper averaging instead of accumulation."""

    model = model.to(device).eval()
    images = [np.array(Image.open(p)) for p in color_paths]

    P = inside_mask.shape[1]
    D = 1152

    # 記錄每個點的累加次數
    pc_features = torch.zeros((P, D), dtype=torch.float32, device=device)
    pc_counts = torch.zeros(P, dtype=torch.int32, device=device)

    N = proposal_masks.shape[1]

    for i in tqdm(range(N), desc="SigLIP: fixed accumulation"):
        point_mask = proposal_masks[:, i].to(device)

        # ... (visibility checking code same as before) ...

        visible_counts = torch.tensor([pts_count.get(f, 0) for f in range(num_frames)], device=device)

        # FIX 3: 動態閾值
        min_threshold = max(5, int(point_mask.sum() * 0.05))  # 至少 5% 的點
        valid = torch.nonzero(visible_counts > min_threshold).view(-1)

        if len(valid) == 0:
            continue

        # FIX 4: 改進 TopK 選擇 (先用簡化版)
        top_ids = torch.topk(visible_counts[valid], k=min(topk, len(valid)), largest=True).indices
        views = valid[top_ids]

        for v in views.tolist():
            coords = pts_coords[v]
            if coords.shape[0] == 0:
                continue

            # ... (bbox computation) ...

            # FIX 2: 只用 1 個 crop (tight bbox)，不要 3 個 scales
            crop = images[v][x1:x2, y1:y2, :]
            if crop.size == 0:
                continue

            crop_tensor = processor(images=crop, return_tensors="pt")["pixel_values"].to(device)
            point_indices = torch.tensor(pts_idx[v], device=device, dtype=torch.long)

            with torch.no_grad():
                feat = model.get_image_features(pixel_values=crop_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # Normalize before accumulating

                # FIX 1: 記錄累加次數，而非直接累加
                pc_features[point_indices] += feat.squeeze(0)
                pc_counts[point_indices] += 1

    # FIX 1: 除以累加次數得到平均
    valid_points = pc_counts > 0
    pc_features[valid_points] = pc_features[valid_points] / pc_counts[valid_points].unsqueeze(-1).float()

    # 最後再 normalize (可選)
    pc_features = F.normalize(pc_features, dim=1)

    return pc_features.cpu()
```

**測試計劃**:
1. 用修復後的版本重新跑 GT sanity check
2. 預期結果:
   - Cabinet door: 0.023 → **>0.3** (提升 10x)
   - Door frame: 0.064 → **>0.5** (提升 8x)
   - Overall rank-1: 60% → **>85%**

### 中期方案 (Complete Fix - 1 週)

1. **實現完整的 view quality scoring** (問題 4)
2. **添加 adaptive thresholding** (問題 3)
3. **統一 batched 和 original 版本** (問題 5)
4. **添加 visualization tools** 驗證修復效果

### 長期方案 (Optimization - 2-3 週)

1. **學習式的 view selection**
2. **Attention-based aggregation**
3. **Multi-scale fusion with learned weights**
4. **End-to-end fine-tuning**

---

## 🧪 驗證計劃

### Step 1: 單點測試

選擇一個明確失敗的 instance (如 cabinet door ID 11002001):

```bash
# 運行修復後的版本
python scripts/test_fixed_feature_extraction.py \
  --instance-id 11002001 \
  --expected-label "cabinet door" \
  --output /tmp/fix_test.json
```

預期:
- 修復前相似度: 0.023
- 修復後相似度: **>0.4**

### Step 2: 完整 GT 測試

```bash
# 重新跑所有 7 個 GT instances
python scripts/siglip_gt_sanity_check_fixed.py \
  --scene-path /media/public_dataset2/multiscan/scene_00005_00 \
  --annotation-file .../scene_00005_00_obj_part_inst.txt \
  --limit 7 \
  --min-points 150
```

預期:
- Rank-1 accuracy: 60% → **>85%**
- Mean similarity: 0.045 → **>0.45**

### Step 3: 對比測試

同時運行修復前和修復後的版本，生成對比報告。

---

## 📝 結論

我們發現的這些問題完全可以解釋觀察到的低相似度現象：

1. ✅ **累加而非平均** → features 語義混亂 → 相似度極低
2. ✅ **Multi-scale crops 累加** → 背景噪聲稀釋 → 小物體失敗
3. ✅ **硬編碼閾值太高** → 小物體 frames 不足 → cabinet door 失敗
4. ✅ **TopK 選擇不佳** → 低質量視角 → topk=5 worse than topk=1
5. ✅ **版本不一致** → 結果難以重現

**最關鍵的發現**: 問題不在 SigLIP 模型本身，也不在 text features，而是在 **feature aggregation 的數學邏輯錯誤**。

**預期修復效果**: 修復問題 1-3 後，相似度應該能從 0.03-0.06 提升到 **>0.4-0.6**，rank-1 accuracy 從 60% 提升到 **>85%**。

**下一步**: 實現 Quick Fix 並驗證假設。

---

**報告生成時間**: 2025-11-16
**分析者**: Claude Code
**建議優先級**: **URGENT** - 建議立即實現 Quick Fix 並測試
