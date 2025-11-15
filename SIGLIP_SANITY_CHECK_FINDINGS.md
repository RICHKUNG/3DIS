# SigLIP Feature Sanity Check - 调查结果

**日期**: 2025-11-15
**场景**: scene_00005_00
**测试规模**: 7 个 GT proposals (5 个有对应文本标签)

---

## 执行摘要

GT visual features 与对应 text features 的匹配度**低于预期**，Rank-1 准确率仅为 **60%**（预期 >= 80%）。

**关键发现**：
- ✅ **成功案例** (3/5): door frame, toilet lid, toilet tank_cover - 匹配度高 (0.54-0.88)
- ❌ **失败案例** (2/5): cabinet door (两个实例) - 匹配度极低 (0.009)
- ⚠️ **Softmax 影响有限**：with/without softmax 的 rank-1 准确率相同 (60%)
- 🔴 **问题本质**：某些类别的 visual features 与 text features 根本不匹配

---

## 详细测试结果

### 1. With Softmax (scale_semantic_score=300.0)

```
Dataset: 7 proposals, 4 labels, 5 with text
Rank-1 accuracy: 60.0%

Similarity Statistics:
  Mean correct similarity: 0.4329
  Median correct similarity: 0.5427
  Min/Max: [0.0089, 0.8785]

  Mean incorrect similarity: 0.5359 (HIGHER than correct!)
  Median incorrect similarity: 0.3727

Margin (Correct - Max Incorrect):
  Mean: -0.1030 (NEGATIVE - incorrect scores higher on average)
  Positive margins: 3/5 (60.0%)
```

### 2. Without Softmax (Raw Cosine Similarity)

```
Rank-1 accuracy: 60.0% (SAME)
Mean correct similarity: 0.0449 (much lower, but still same rank!)
Mean margin: -0.0059
```

**结论**: Softmax 放大了相似度差异，但**不改变排名**。问题在于原始特征本身。

---

## 逐类别分析

| Label | Instances | Rank-1 Rate | Mean Similarity | Status |
|-------|-----------|-------------|-----------------|--------|
| **door frame** | 1 | 100% ✅ | 0.88 | 优秀 |
| **toilet lid** | 1 | 100% ✅ | 0.72 | 良好 |
| **toilet tank_cover** | 1 | 100% ✅ | 0.54 | 及格 |
| **cabinet door** | 2 | 0% ❌ | 0.009 | **失败** |

---

## 失败案例深度分析

### Cabinet Door (两个实例全部失败)

**Instance 1** (ID 11002001, 345 points):
- ✗ Correct similarity: **0.0097** (接近 0!)
- ✗ Max incorrect: **0.9841** (接近 1!)
- ✗ Margin: **-0.9744** (错误匹配远好于正确匹配)
- ✗ Rank: **2** (正确标签只排第2)

**Instance 2** (ID 11002002, 347 points):
- ✗ Correct similarity: **0.0089**
- ✗ Max incorrect: **0.9283**
- ✗ Margin: **-0.9193**
- ✗ Rank: **4** (正确标签排第4)

**可能原因**：
1. **视觉特征提取问题**：
   - 可能投影到的 2D 帧中 cabinet door 不明显/遮挡
   - mask 区域太小导致特征不稳定
   - 只使用 topk=1 个最佳视角可能不够

2. **文本标签问题**：
   - "cabinet door" 可能与其他标签（如 "door"）语义混淆
   - SigLIP 的 text encoder 可能对这个组合标签不敏感

3. **3D-to-2D 投影问题**：
   - cabinet door 可能在很多帧中不可见
   - 可见的帧中可能只看到 door 的边缘

---

## 成功案例分析

### Door Frame (ID 4008001, 2149 points)
- ✓ Correct similarity: **0.8785** (非常高!)
- ✓ Max incorrect: 0.1198
- ✓ Margin: **0.7587** (巨大margin)
- ✓ Rank: **1**

**成功因素**：
- 点数多 (2149 vs cabinet door 的 347)
- Door frame 通常在多个视角都可见
- 语义明确，文本特征区分度高

### Toilet Lid & Tank Cover
- 都达到了较高的 similarity (0.72, 0.54)
- margin 都为正 (0.45, 0.17)

---

## 关键问题诊断

### 问题 1: "Incorrect similarity 高于 correct similarity"

从统计数据看：
- Mean **correct** similarity: 0.4329
- Mean **incorrect** similarity: 0.5359

这意味着平均而言，**错误的标签比正确的标签更相似**！

### 问题 2: Softmax 的作用有限

对比实验显示：
- Softmax 放大了 similarity 差异 (0.4329 vs 0.0449)
- 但 rank-1 accuracy 完全相同 (60% vs 60%)
- **结论**: Softmax 只是数值变换，不改变排序

### 问题 3: 少数类别的完全失败

cabinet door 的两个实例都接近 0 similarity，表明：
- 不是随机噪声
- 而是系统性的特征不匹配
- 需要检查这些实例的具体原因

---

## 深入调查建议

### 立即行动

1. **可视化失败案例**
   ```bash
   # 查看 cabinet door 实例的具体帧
   # 检查 mask 投影是否正确
   # 查看提取的 visual features 是什么
   ```

2. **增加 topk views**
   ```bash
   --topk-views 5  # 从 1 增加到 5
   ```
   可能单个视角不够，需要多视角聚合

3. **检查原始的 check_siglip_gt_similarity.py 结果**
   ```bash
   # 运行原始脚本（不带 softmax）对比
   conda run -n 3Dsiglip python scripts/check_siglip_gt_similarity.py \
     --scene-path /media/public_dataset2/multiscan/scene_00005_00 \
     --annotation-file .../scene_00005_00_obj_part_inst.txt \
     --limit 20
   ```

4. **扩大测试规模**
   ```bash
   --limit 50  # 测试更多实例
   --min-points 100  # 降低点数阈值
   ```

### 中期调查

5. **分析其他场景**
   - 测试 scene_00006_00, scene_00018_00 等
   - 看 cabinet door 是否在其他场景也失败

6. **检查 label 映射**
   - 验证 "cabinet door" 在 VALID_JOINT_TUPLE_NAMES 中的编码
   - 确认 label_id = 11002 的解码正确

7. **对比不同 scale_semantic_score**
   - 测试 scale = 1.0, 10.0, 100.0, 300.0, 1000.0
   - 看是否存在最佳值

### 长期研究

8. **实现可视化工具**
   - 渲染 GT mask 在 RGB 帧上
   - 显示提取 feature 的区域
   - 可视化 feature space 的分布

9. **对比不同 aggregation 方法**
   - 当前：mean pooling across frames
   - 尝试：max pooling, attention-weighted

10. **考虑改进 text prompt**
    - 当前："cabinet door"
    - 尝试："a cabinet door", "door of a cabinet", "cabinet with door"

---

## 初步结论

**Sanity Check 结果**: ⚠️ **部分失败**

1. **60% rank-1 准确率不够高**（目标 >= 80%）

2. **问题不在 softmax**：
   - with/without softmax 结果一致
   - 问题在于原始的 visual-text feature alignment

3. **特定类别系统性失败**：
   - cabinet door: 完全失败 (0.009 similarity)
   - 需要深入调查这些案例

4. **成功的类别表现优秀**：
   - door frame, toilet lid/tank_cover: 0.54-0.88 similarity
   - 证明方法本身可行

**下一步**：
1. 增加 topk_views 到 5
2. 可视化失败案例（cabinet door）
3. 扩大测试到 50+ instances
4. 运行原始脚本对比

---

## 附录：测试命令

### 当前测试
```bash
# With softmax (scale=300)
conda run -n 3Dsiglip python scripts/siglip_gt_sanity_check_with_scaling.py \
  --scene-path /media/public_dataset2/multiscan/scene_00005_00 \
  --annotation-file /media/Pluto/richkung/My3DIS/data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations/scene_00005_00_obj_part_inst.txt \
  --limit 20 --min-points 150 --topk-views 1 --batch-size 8 \
  --scale-semantic-score 300.0 \
  --output-json /tmp/test_with_softmax.json

# Without softmax
# ... --no-softmax ...
```

### 诊断分析
```bash
conda run -n 3Dsiglip python scripts/diagnose_siglip_similarity.py \
  --results /tmp/test_with_softmax.json \
  --compare-with /tmp/test_no_softmax.json
```

---

**报告生成**: 2025-11-15
**脚本位置**:
- `/media/Pluto/richkung/My3DIS/scripts/siglip_gt_sanity_check_with_scaling.py`
- `/media/Pluto/richkung/My3DIS/scripts/diagnose_siglip_similarity.py`
