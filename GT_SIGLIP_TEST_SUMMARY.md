# GT SigLIP Similarity Test - Summary

## 修改内容

### 1. 更新 `check_siglip_gt_similarity.py`

**主要改进：**
- ✅ 使用 pipeline.ipynb 中的 `extract_proposal_features_siglip` 函数（已通过 `notebook_pipeline.py` 导入）
- ✅ 添加优化选项：`--no-optimized` 和 `--no-sparse`（默认启用优化）
- ✅ 支持批处理优化（`batch_size` 默认32，可调整）
- ✅ 支持稀疏存储（只处理活跃点，节省内存）

**参数变更：**
```diff
+ --batch-size (default: 32, 从8改为32)
+ --no-optimized (默认启用批处理优化)
+ --no-sparse (默认启用稀疏存储)
```

**函数调用更新：**
```python
proposal_features = extract_proposal_features_siglip(
    pipeline=pipeline,
    proposal_masks=proposal_masks,
    model=model,
    processor=processor,
    device=torch_device,
    topk=args.topk_views,
    batch_size=args.batch_size,
    use_optimized=args.use_optimized,    # NEW
    use_sparse=args.use_sparse,          # NEW
)
```

## 测试结果

### 环境
- **Conda 环境**: 3Dsiglip
- **场景**: scene_00005_00
- **设备**: CUDA:0
- **GT 文件**: ov_part_annotations (object + part 联合标注)

### 测试配置
```bash
--min-points 150
--topk-views 1
--batch-size 32
--use-optimized (enabled)
--use-sparse (enabled)
```

### 性能指标

**处理时间：**
- 总时间: ~50秒
  - WORLD_2_CAM 初始化: ~16秒
  - SigLIP 模型加载: ~2秒
  - 特征提取 (7个实例): ~19秒
  - 其他: ~13秒

**内存优化：**
- 稀疏存储: 10,543 / 79,614 活跃点 (13.2%)
- 节省内存: ~86.8%

**处理统计：**
- GT 实例总数: 7
- 匹配文本标签: 5 (71.4%)
- 缺失文本标签: 2 (28.6% - label_id 4002)

### 相似度结果

| Raw Label | Label ID | Instance | Points | Similarity | Text | Status |
|-----------|----------|----------|--------|------------|------|--------|
| 4008001 | 4008 | 001 | 2149 | 0.0640 | door frame | low |
| 32005001 | 32005 | 001 | 554 | 0.0606 | toilet lid | low |
| 32007001 | 32007 | 001 | 268 | 0.0469 | toilet tank_cover | low |
| 11002001 | 11002 | 001 | 309 | 0.0441 | cabinet door | low |
| 11002002 | 11002 | 002 | 347 | 0.0340 | cabinet door | low |
| 4002001 | 4002 | 001 | 3696 | - | UNKNOWN | missing |
| 4002002 | 4002 | 002 | 3220 | - | UNKNOWN | missing |

**相似度统计：**
- Mean: 0.0499
- Min: 0.0340
- Max: 0.0640
- 达到阈值 (0.5): 0 / 5 (0%)

## 问题分析

### 1. 低相似度问题
所有匹配的实例相似度都很低 (< 0.1)，远低于阈值 0.5。

**可能原因：**
- ❓ GT 标注质量问题
- ❓ SigLIP 特征提取配置需要调整
- ❓ 3D 到 2D 投影精度问题
- ❓ 文本 embedding 与视觉特征不匹配

### 2. 缺失标签
Label ID 4002 在 joint label map 中未找到。

**建议：**
- 检查 `multiscan_search3d_constants.py` 中的 label 定义
- 确认是否为有效的 object-part 组合

## 优化效果预估

**预期加速比（大规模测试）：**

假设处理 100+ GT 实例：
- **原始版本**（逐个处理）: ~N × 20秒 = ~33分钟
- **优化版本**（batch=32）: ~(N/32) × 20秒 = ~2-3分钟
- **加速比**: ~10-15x

**内存节省：**
- 稀疏存储可节省 ~80-90% 内存（取决于场景）

## 使用方法

### 快速测试（10个实例）
```bash
conda run -n 3Dsiglip bash test_gt_siglip.sh
```

### 完整测试（所有实例）
```bash
conda run -n 3Dsiglip bash test_gt_siglip_full.sh
```

### 自定义测试
```bash
export PYTHONPATH=src
python scripts/check_siglip_gt_similarity.py \
  --scene-path /path/to/scene \
  --annotation-file /path/to/annotations.txt \
  --siglip-model /path/to/siglip \
  --local-files-only \
  --device cuda:0 \
  --batch-size 32 \
  --output-json output.json
```

### 禁用优化（调试）
```bash
python scripts/check_siglip_gt_similarity.py \
  ... \
  --no-optimized \
  --no-sparse
```

## 输出文件

**JSON 结果：**
```
outputs/gt_siglip_check/scene_00005_00/
├── similarity_results.json       # 测试结果 (limit=10)
└── similarity_results_full.json  # 完整结果
```

**日志文件：**
```
/tmp/gt_siglip_test.log      # 测试日志
/tmp/gt_siglip_full_test.log # 完整测试日志
```

## 下一步

### 建议调查
1. **低相似度问题**
   - 可视化 GT mask 与提取的 2D crops
   - 检查 3D→2D 投影是否准确
   - 尝试调整 `--topk-views` 和 `--vis-depth-threshold`

2. **标签映射问题**
   - 补全 label_id 4002 的定义
   - 验证 joint label map 完整性

3. **性能基准**
   - 在更大的场景上测试（如 scene 包含 100+ GT 实例）
   - 对比优化前后的性能差异

### 参数调优
```bash
# 尝试更多视图
--topk-views 3

# 调整批次大小（根据 GPU 内存）
--batch-size 16  # 低内存
--batch-size 64  # 高内存

# 调整可见性阈值
--vis-depth-threshold 0.2
```

## 参考

- **脚本位置**: `scripts/check_siglip_gt_similarity.py`
- **Pipeline 实现**: `3DprojToSiglip/notebook_pipeline.py`
- **优化实现**: `3DprojToSiglip/utils_new/feature_extraction.py`
- **测试脚本**: `test_gt_siglip.sh`, `test_gt_siglip_full.sh`
