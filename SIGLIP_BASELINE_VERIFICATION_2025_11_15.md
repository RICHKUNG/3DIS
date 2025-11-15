# SigLIP 模型基础验证结果

**日期**: 2025-11-15
**目标**: 验证 SigLIP 模型是否正确加载和工作

---

## 执行摘要

✅ **SigLIP 模型本身工作正常！**

通过 text-text 相似度测试确认：
- ✅ 模型加载正确
- ✅ 文本特征提取正确  
- ✅ 余弦相似度计算正确
- ✅ 自匹配相似度都是 1.0000（完美）

**但是**，文本 prompt 模板测试显示：**添加模板会降低与原始标签的相似度**。

---

## TEST 1: Real Scene Images

❌ **测试跳过** - Color 目录路径错误
- 预期路径：`/media/public_dataset2/multiscan/scene_00005_00/color`
- 实际路径应该是：`/media/public_dataset2/multiscan/scene_00005_00/outputs/color`

**待修正并重新测试**。

---

## TEST 2: Text-Text Similarity (文本对文本相似度)

✅ **完美通过！**

### Test Case 1: "door frame"

| 候选文本 | 相似度 | 说明 |
|---------|-------|------|
| ✓ **door frame** | **1.0000** | 自匹配（完美） |
| door | 0.6695 | 部分匹配 |
| window frame | 0.6463 | 结构相似 |
| cabinet | 0.4373 | 低相关性 |
| toilet | 0.3598 | 无关 |

✓ 自匹配是最高的 (1.0000)

### Test Case 2: "cabinet door"

| 候选文本 | 相似度 | 说明 |
|---------|-------|------|
| ✓ **cabinet door** | **1.0000** | 自匹配（完美） |
| drawer | 0.6124 | 相关功能 |
| cabinet | 0.6038 | 父类别 |
| door | 0.4832 | 组成部分 |
| window | 0.4014 | 低相关性 |

✓ 自匹配是最高的 (1.0000)

### Test Case 3: "toilet lid"

| 候选文本 | 相似度 | 说明 |
|---------|-------|------|
| ✓ **toilet lid** | **1.0000** | 自匹配（完美） |
| toilet | 0.7884 | 父类别（高相关） |
| lid | 0.6369 | 组成部分 |
| bathtub | 0.5862 | 相关功能 |
| sink | 0.5759 | 相关功能 |

✓ 自匹配是最高的 (1.0000)

### 结论

**SigLIP 文本编码器工作完美**：
1. 所有自匹配相似度都是 1.0000
2. 语义相关性正确排序（toilet lid → toilet > bathtub/sink）
3. 无关词汇相似度最低

---

## TEST 3: Prompt Template Comparison (提示模板影响)

🔍 **关键发现：添加模板会降低相似度！**

### 模板对比结果

| 模板格式 | door frame | cabinet door | toilet lid | **平均** |
|---------|-----------|-------------|-----------|---------|
| `{}` **(原始)** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| `a {}` | 0.9443 | 0.9550 | 0.9365 | **0.9453** |
| `indoor {}` | 0.9089 | 0.9017 | 0.9107 | **0.9071** |
| `{} in an indoor scene` | 0.8709 | 0.8125 | 0.8362 | **0.8399** |
| `a {} in a room` | 0.8105 | 0.8057 | 0.8093 | **0.8085** |
| `a photo of a {}` | 0.6747 | 0.7215 | 0.7043 | **0.7002** |

### 详细分析

**door frame**:
```
✓ 'door frame':                    1.0000
  'a door frame':                   0.9443  (-5.6%)
  'indoor door frame':              0.9089  (-9.1%)
  'door frame in an indoor scene':  0.8709  (-12.9%)
  'a door frame in a room':         0.8105  (-19.0%)
  'a photo of a door frame':        0.6747  (-32.5%)
```

**cabinet door**:
```
✓ 'cabinet door':                      1.0000
  'a cabinet door':                     0.9550  (-4.5%)
  'indoor cabinet door':                0.9017  (-9.8%)
  'cabinet door in an indoor scene':    0.8125  (-18.8%)
  'a cabinet door in a room':           0.8057  (-19.4%)
  'a photo of a cabinet door':          0.7215  (-27.9%)
```

**toilet lid**:
```
✓ 'toilet lid':                    1.0000
  'a toilet lid':                   0.9365  (-6.4%)
  'indoor toilet lid':              0.9107  (-8.9%)
  'toilet lid in an indoor scene':  0.8362  (-16.4%)
  'a toilet lid in a room':         0.8093  (-19.1%)
  'a photo of a toilet lid':        0.7043  (-29.6%)
```

### 相似度降低程度排名

1. **最小影响** (< 10%): `a {}`
2. **中等影响** (10-20%): `indoor {}`, `{} in an indoor scene`, `a {} in a room`
3. **最大影响** (> 25%): `a photo of a {}`

### 关键洞察

1. **添加任何模板都会降低相似度**
   - 即使是最小的 `a {}` 也降低 4-6%
   - CLIP-style 的 `a photo of a {}` 降低最多 (25-33%)

2. **降低程度相对一致**
   - 所有三个标签的降低幅度类似
   - 表明这是 SigLIP 的系统性特征

3. **含义**：如果 inference pipeline 使用原始标签，那么在 GT 测试中也应该使用原始标签
   - 不应该添加 "a photo of" 等模板
   - 保持一致性最重要

---

## 结论与建议

### ✅ 确认正常的部分

1. **SigLIP 模型加载和运行正常**
   - Text encoder 工作完美
   - 特征提取方法正确
   - 余弦相似度计算正确

2. **Text-text 相似度达到预期**
   - 自匹配: 1.0000
   - 语义相关性排序合理

### ⚠️ 需要注意的问题

1. **Prompt 模板会降低相似度**
   - 不应该使用 CLIP-style 的 "a photo of a {}"
   - 即使是简单的 "a {}" 也会降低 5% 相似度
   - **建议**: 保持 text prompt 与 inference pipeline 完全一致

2. **Visual feature 测试未完成**
   - 需要修正 color 目录路径
   - 需要测试 image-text 相似度是否在合理范围

### 下一步行动

1. **修正 color 路径并重新测试 image-text 相似度**
   ```bash
   # 修改脚本使用正确路径
   --scene-path /media/public_dataset2/multiscan/scene_00005_00/outputs
   ```

2. **在 GT 数据上测试不同 prompt 模板**
   - 运行 `test_gt_prompt_engineering.py`
   - 查看哪个模板给出最好的 rank-1 accuracy
   - **当前运行中...**

3. **如果 prompt engineering 无法解决问题**
   - 深入调查 visual feature extraction 流程
   - 检查 3D-to-2D projection 是否正确
   - 可视化 GT proposal 在 RGB 帧上的投影

---

## 测试脚本

**创建的测试脚本**:
1. `scripts/verify_siglip_baseline.py` - 基础功能验证
2. `scripts/test_gt_prompt_engineering.py` - GT 数据 prompt engineering 测试

**运行命令**:
```bash
# 基础验证
conda run -n 3Dsiglip python scripts/verify_siglip_baseline.py \
  --scene-path /media/public_dataset2/multiscan/scene_00005_00

# Prompt engineering 测试 (运行中)
conda run -n 3Dsiglip python scripts/test_gt_prompt_engineering.py \
  --scene-path /media/public_dataset2/multiscan/scene_00005_00 \
  --annotation-file .../scene_00005_00_obj_part_inst.txt \
  --limit 20 --min-points 150 --topk-views 1 --batch-size 8
```

---

**报告生成时间**: 2025-11-15 19:56  
**测试状态**:
- ✅ Text-text similarity 测试完成
- ✅ Prompt template comparison 完成
- ⏳ GT prompt engineering 测试运行中
- ❌ Image-text similarity 待测试（路径错误）
