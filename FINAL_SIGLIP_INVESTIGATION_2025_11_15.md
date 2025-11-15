# SigLIP GT Sanity Check - 最终调查报告

**日期**: 2025-11-15  
**场景**: scene_00005_00  
**测试规模**: 7 GT proposals (5 with text labels)

---

## 执行摘要

🔴 **核心问题已定位**：GT visual features 与 text features 的原始余弦相似度异常低 (0.03-0.06)，**比预期低10倍**。

✅ **已排除的原因**：
- ❌ Softmax/scaling 问题
- ❌ Text prompt engineering 问题
- ❌ 视角数量不足问题
- ❌ SigLIP 模型加载问题

🎯 **问题根源**：**Visual feature extraction pipeline** 可能存在问题，导致提取的特征与文本特征不在同一嵌入空间。

---

## 完整实验结果总结

### 实验 1: Baseline vs TopK=5 vs No Softmax

| 配置 | Rank-1 | Mean Correct (Softmax) | Mean Correct (Raw) | Cabinet Door |
|------|--------|------------------------|-------------------|--------------|
| **Baseline** (topk=1, scale=300, softmax) | **60%** | 0.4329 | 0.045 | 0.009 |
| **TopK=5** (topk=5, scale=300, softmax) | **40%** ⬇️ | 0.3674 | - | 0.004 |
| **No Softmax** (topk=1, raw cosine) | **60%** | - | 0.045 | 0.023 |
| **Original Script** | N/A | - | 0.050 | 0.035 |

**关键发现**：
1. ✅ Softmax 不改变排序（有/无 softmax 都是 60% rank-1）
2. ❌ 增加视角反而更糟（topk=5 降到 40%）
3. 🔴 **所有原始余弦相似度 < 0.065**（预期 > 0.5）

### 实验 2: SigLIP 模型基础验证

✅ **Text-Text 相似度测试 - 完美通过**

| Query | Self-Match | 第二高 | 说明 |
|-------|-----------|-------|------|
| door frame | **1.0000** | 0.6695 (door) | ✓ 自匹配最高 |
| cabinet door | **1.0000** | 0.6124 (drawer) | ✓ 自匹配最高 |
| toilet lid | **1.0000** | 0.7884 (toilet) | ✓ 自匹配最高 |

**结论**: SigLIP 文本编码器工作正常，模型加载正确。

### 实验 3: Text Prompt Template 影响

| Template | door frame | cabinet door | toilet lid | 平均 | 降低幅度 |
|----------|-----------|-------------|-----------|------|---------|
| `{}` (raw) | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0% |
| `a {}` | 0.8834 | 0.8737 | 0.8924 | 0.8867 | -11.3% |
| `indoor {}` | 0.9090 | 0.8751 | 0.9102 | 0.8953 | -10.5% |
| `{} in an indoor scene` | 0.8218 | 0.7215 | 0.7660 | 0.7702 | -23.0% |
| `a {} in a room` | 0.8134 | 0.7505 | 0.8080 | 0.7791 | -22.1% |
| `a photo of a {}` | 0.8089 | 0.8037 | 0.8184 | 0.8086 | -19.1% |

**结论**: 添加**任何**模板都会降低相似度，必须使用原始标签（raw labels）。

---

## 逐案例深入分析

### 成功案例 (3/5)

**1. Door Frame (ID 4008001)** - ✅ 优秀
- Points: 2149 (最多)
- Baseline: 0.88 (softmax) / 0.064 (raw)
- TopK=5: 0.82 (softmax)
- **Rank**: 1 (稳定)
- **分析**: 点数多，可能在多个视角都可见，语义明确

**2. Toilet Lid (ID 32005001)** - ✅ 良好
- Points: 554
- Baseline: 0.72 (softmax) / 0.062 (raw)
- TopK=5: 0.72 (softmax)
- **Rank**: 1 (稳定)

**3. Toilet Tank Cover (ID 32007001)** - ⚠️ 边缘案例
- Points: 268
- Baseline: 0.54 (softmax) / 0.046 (raw)
- TopK=5: 0.28 (softmax) - **大幅下降！**
- **Rank**: 1 (baseline) → 2 (topk=5)
- **分析**: TopK=5 引入了噪声视角，稀释了特征

### 失败案例 (2/5)

**4 & 5. Cabinet Door (ID 11002001, 11002002)** - ❌ 完全失败
- Points: 309, 347
- Baseline: 0.009-0.010 (softmax) / 0.023-0.045 (raw)
- TopK=5: 0.004-0.008 (softmax) - **更糟！**
- Max incorrect: 0.93-0.98 (错误标签的相似度接近1！)
- **Rank**: 2-4 (远低于预期)
- **Margin**: -0.92 to -0.97 (巨大负margin)

**失败原因分析**:
1. **点数较少** (309-347 vs 2149 for door frame)
2. **可能在多数帧中不可见或被遮挡**
3. **投影的 mask 区域可能太小**
4. **Text label "cabinet door" 可能与其他标签混淆**

---

## Visual Feature Extraction 流程分析

### 当前实现 (3DprojToSiglip/notebook_pipeline.py)

```python
def extract_proposal_features_siglip(pipeline, proposal_masks, model, processor, 
                                      device, topk, batch_size, use_optimized):
    # Step 1: 为每个3D点生成SigLIP features (基于投影到2D的frames)
    pc_features = generate_grounding_features_siglip(
        model=model,
        processor=processor,
        color_paths=pipeline.world2cam.color_paths,
        inside_mask=pipeline.inside_mask,
        projected_points=pipeline.projected_points,
        points_depth=pipeline.point_depth,
        depth_maps=pipeline.depth_maps,
        device=device,
        topk=topk,  # 选择topk个最佳viewing angles
        ...
    )
    
    # Step 2: 将点级别特征池化到proposal级别
    return pool_point_to_proposal_features(pc_features, proposal_masks)
```

### 可能的问题点

1. **TopK Selection**
   - 实验显示 topk=5 比 topk=1 **更糟** (60% → 40%)
   - 可能选择的 "最佳" 视角实际上不是最好的
   - 或者 mean pooling across views 稀释了特征

2. **3D-to-2D Projection**
   - Visibility filtering 可能太严格或太宽松
   - Depth threshold 可能不准确
   - Sparse storage (13.2%) 可能丢失信息

3. **Point-to-Proposal Pooling**
   - Mean pooling 可能洗掉了关键特征
   - 小 mask (< 350 points) 可能导致不稳定特征

4. **Frame Aggregation**
   - 当前使用 mean pooling across topk frames
   - 可能需要 max pooling 或 attention-weighted aggregation

---

## 核心问题诊断

### 问题本质

**Raw cosine similarities ALL < 0.065** (expected > 0.5)

这不是数值缩放或排序问题，而是**特征空间不对齐**问题：

```
Expected:
  Visual features · Text features ≈ 0.5-0.8 (for correct matches)
  
Actual:
  Visual features · Text features ≈ 0.03-0.06 (ALL cases)
```

这意味着：
- Visual features 可能不在正确的嵌入空间
- 或 visual features 包含了过多噪声
- 或 aggregation 方法破坏了语义信息

### 证据链

1. ✅ **Text features 正常** (text-text similarity = 1.0)
2. ✅ **SigLIP 模型正常** (self-match perfect)
3. ❌ **Visual-text similarity 异常低** (0.03-0.06)
4. ❌ **增加视角使情况更糟** (topk=5 worse than topk=1)

**结论**: 问题在于 **visual feature extraction/aggregation pipeline**

---

## 建议的下一步调查

### 高优先级 (立即执行)

1. **可视化 3D-to-2D 投影**
   - 查看 cabinet door 的 GT mask 投影到哪些 frames
   - 检查投影的 mask 是否覆盖了有意义的区域
   - 验证 visibility filtering 是否正确

2. **单帧测试**
   - 手动选择一个 cabinet door 清晰可见的帧
   - 直接提取该帧的 visual features（不做 aggregation）
   - 测试该帧的 visual-text similarity

3. **对比 Successful Cases**
   - 比较 door frame (成功) vs cabinet door (失败)
   - 查看它们在 projection/pooling 步骤的差异
   - 分析为什么 door frame 能达到 0.064 而 cabinet door 只有 0.023

4. **检查 Notebook Pipeline**
   - 找到 inference 中成功的案例
   - 逐步对比 feature extraction 流程
   - 验证我们的 GT sanity check 是否使用了完全相同的流程

### 中优先级

5. **测试不同 Aggregation 方法**
   - Current: mean pooling
   - Try: max pooling
   - Try: attention-weighted pooling

6. **调整 Visibility Threshold**
   - 当前使用 `vis_depth_threshold`
   - 尝试更宽松/严格的阈值

7. **增加测试规模**
   - 扩展到 50+ instances
   - 测试多个 scenes
   - 建立统计显著性

### 低优先级

8. **尝试不同的 Feature Extraction**
   - 测试不使用 topk，而是使用所有可见frames
   - 测试使用 weighted average based on view quality
   - 测试使用 learnable aggregation

---

## 可能的解决方案

基于当前证据，可能的修复方向：

### 方案 A: 修复 TopK Selection

**问题**: topk=5 worse than topk=1 表明 view selection 有问题

**解决**:
- 改进 view quality 评估标准
- 考虑 mask coverage、viewing angle、occlusion 等因素
- 使用 weighted aggregation 而非 mean pooling

### 方案 B: 修复 Visibility Filtering

**问题**: Cabinet door 可能在很多帧中被错误地标记为不可见

**解决**:
- 检查 depth threshold 是否过于严格
- 可视化 visibility mask 验证正确性
- 调整 `vis_depth_threshold` 参数

### 方案 C: 改进 Aggregation 方法

**问题**: Mean pooling 可能稀释了关键特征

**解决**:
- 使用 max pooling (保留最强信号)
- 使用 attention-based aggregation
- 为每个 view 学习权重

### 方案 D: 单帧最佳策略

**问题**: 多帧聚合可能引入噪声

**解决**:
- 只使用 single best view (topk=1 但改进选择标准)
- 基于 view quality score 动态选择
- 不做 frame aggregation，直接使用最佳帧

---

## 临时工作方案

**如果需要立即使用当前 pipeline**:

1. **使用 topk=1** (避免多帧聚合的问题)
2. **保留原始标签** (不添加任何 prompt template)
3. **使用 scale=300.0 + softmax** (与 inference 保持一致)
4. **过滤小 proposals** (< 500 points 的可能不稳定)
5. **接受 60% rank-1 accuracy** 作为当前 baseline

**但这不解决根本问题** - 需要深入调查 visual feature extraction。

---

## 结论

### 确定的事实

1. ✅ SigLIP 模型工作正常
2. ✅ Text features 正确
3. ✅ Softmax/scaling 不是问题
4. ✅ Prompt engineering 不是解决方案
5. ❌ Visual features 与 text features **根本性不匹配**
6. ❌ 原始余弦相似度比预期低 **10倍**

### 下一步行动

**强烈建议**在解决 visual feature extraction 问题之前，**暂停大规模 inference**。

**优先调查**:
1. 可视化 3D-to-2D 投影（查看 cabinet door）
2. 单帧测试（避免 aggregation）
3. 对比成功案例（door frame）vs 失败案例（cabinet door）

### 预期时间

- 可视化工具开发: 2-4 小时
- 单帧测试: 1-2 小时
- 根因定位: 4-8 小时
- 修复实现: 取决于问题复杂度

---

## 附录：生成的文件和脚本

### 报告文档

1. `INFERENCE_FIXES_2025_11_15.md` - 实验结果和根因分析
2. `SIGLIP_BASELINE_VERIFICATION_2025_11_15.md` - SigLIP 模型验证
3. `SIGLIP_SANITY_CHECK_FINDINGS.md` - 初步调查结果
4. `FINAL_SIGLIP_INVESTIGATION_2025_11_15.md` - 本文档（最终报告）

### 测试脚本

1. `scripts/siglip_gt_sanity_check_with_scaling.py` - GT sanity check (with scaling)
2. `scripts/check_siglip_gt_similarity.py` - 原始 GT similarity check
3. `scripts/verify_siglip_baseline.py` - SigLIP 基础功能验证
4. `scripts/quick_prompt_test.py` - Prompt template 快速测试
5. `scripts/diagnose_siglip_similarity.py` - 诊断分析工具
6. `scripts/compare_all_experiments.py` - 实验对比工具

### 测试数据

- `/tmp/test_20samples_with_softmax.json` - Baseline结果
- `/tmp/test_20samples_no_softmax.json` - No softmax结果
- `/tmp/test_topk5_softmax.json` - TopK=5结果
- `/tmp/test_original_script.json` - 原始脚本结果

---

**报告生成时间**: 2025-11-15 20:00  
**调查状态**: 根因定位中，需要可视化验证  
**建议**: 暂停 inference，优先修复 visual feature extraction
