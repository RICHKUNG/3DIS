# Pairing Failures分析与改进建议

## 问题分析

根据对`scene_00005_00`的诊断分析，发现：

### 统计数据
- **Valid pairs**: 29% (100对中有29对通过)
- **Invalid pairs**: 71% (71对被过滤掉)

### 失败原因分布
1. **Containment only**: 56.3% (40对) - 仅因包含度不足
2. **Both constraints**: 31.0% (22对) - 同时违反包含度和尺度约束
3. **Scale only**: 12.7% (9对) - 仅因尺度不匹配

### 关键观察

从失败案例中可以看到两个主要问题：

1. **Low Containment** (包含度过低)
   - 许多pairs的containment ratio < 0.1 (当前阈值)
   - 例如：containment=0.003, 0.000, 0.032等
   - 这说明许多part和object在3D空间中**几乎没有重叠**

2. **Scale Violations** (尺度违反)
   - 许多part的尺度比object**更大**
   - 例如：scale_ratio=2.124, 1.504, 3.513, 15.271等
   - 当前scale_range=[0.001, 0.99]意味着part必须小于object
   - 但实际数据中，**part常常比object大**

## 根本原因

这些问题的根本原因是：

### 1. 语义vs几何的不匹配

在object-part关系中，**语义上的part和几何上的part不一定一致**：

- **语义正确但几何违规的例子：**
  - `door` (整个门) vs `frame` (门框) - **frame可能比door的mask更大**，因为door mask可能只包括门扇
  - `trash_can` (垃圾桶) vs `door` (垃圾桶门) - door可能与trash_can没有太多重叠（如果mask质量不佳）
  - `printer` (打印机) vs `tray` (纸盘) - tray可能延伸到printer主体外部

### 2. Mask质量问题

从非常低的containment ratio (0.000-0.05)可以看出：
- SAM2产生的masks可能不完整或不准确
- Object和part masks可能完全不重叠，即使它们语义上相关
- 这可能是由于：
  - 视角变化导致的分割不一致
  - 遮挡
  - SAM2的over-segmentation

## 改进方案

### 方案1：大幅放宽几何约束（推荐）

**配置更改：**

```yaml
inference:
  pairing:
    containment_threshold: 0.01  # 从0.1降至0.01 (只需要1%重叠)
    scale_range: [0.0001, 5.0]   # 从[0.001, 0.99]扩大到[0.0001, 5.0]
    use_geometric_score: true    # 保持启用，但不作为硬约束
    use_tree_pruning: false      # 保持禁用
```

**理由：**
- **Containment 0.01**: 允许几乎不重叠的masks，依靠语义相似度和NMS来筛选
- **Scale [0.0001, 5.0]**: 允许part比object大5倍，覆盖门框、纸盘等情况
- 依靠**语义相似度**和**NMS**来保证质量，而不是几何约束

**预期效果：**
- 将valid pair rate从29%提升到**70-80%**
- 减少false negatives (漏检)
- 可能增加一些false positives，但NMS会处理

### 方案2：双向配对（Bidirectional Pairing）

实现一个新的pairing策略，尝试两个方向：

```python
# 方向1: part contained in object (传统)
pairs_1 = pair_with_containment(parts, objects, containment_threshold=0.01)

# 方向2: object contained in part (反向)
pairs_2 = pair_with_containment(objects, parts, containment_threshold=0.01)

# 合并两个方向的结果
all_pairs = merge_bidirectional_pairs(pairs_1, pairs_2)
```

**配置：**

```yaml
inference:
  pairing:
    bidirectional: true          # 启用双向配对 (NEW)
    containment_threshold: 0.05  # 两个方向都使用相同阈值
    scale_range: [0.01, 100.0]   # 允许任意尺度比例
    use_geometric_score: true
```

**优势：**
- 自动处理"part比object大"的情况
- 不需要手动判断哪个query需要swap
- 可以捕获更多valid pairs

### 方案3：Query-specific配置（精细控制）

为不同的query类型提供不同的pairing配置：

```yaml
inference:
  pairing_overrides:
    # 对于已知part比object大的queries
    door-frame:
      containment_threshold: 0.01
      scale_range: [0.5, 3.0]     # frame通常比door大
      swap_roles: true             # 交换object和part角色

    # 对于小part的queries
    cabinet-handle:
      containment_threshold: 0.3
      scale_range: [0.001, 0.2]   # handle很小

    # 默认配置
    default:
      containment_threshold: 0.05
      scale_range: [0.001, 2.0]
```

### 方案4：使用IoU而非Containment（替代指标）

当前使用的是**单向containment** (part在object内的比例)。可以改用**IoU** (双向重叠)：

```python
# 当前：containment = |part ∩ object| / |part|
# 问题：如果part很大，containment会很低

# 替代：IoU = |part ∩ object| / |part ∪ object|
# 优势：对尺度不敏感，对称性
```

**配置：**

```yaml
inference:
  pairing:
    use_iou_instead_of_containment: true  # NEW: 使用IoU替代containment
    min_iou: 0.01                          # 最小IoU阈值
    scale_range: [0.0001, 100.0]          # 不限制尺度
```

## 推荐实施顺序

1. **立即实施方案1** (放宽约束)
   - 最简单，零代码改动
   - 快速验证效果

2. **如果方案1效果不理想，实施方案4** (IoU)
   - 小量代码改动
   - 更robust的指标

3. **长期考虑方案2** (双向配对)
   - 需要较多代码改动
   - 但提供最灵活的解决方案

## 验证方法

修改配置后，运行诊断脚本验证：

```bash
# 测试新配置
python3 scripts/diagnose_pairing_failures.py \
  --config configs/inference/full_pipeline.yaml \
  --aggregation-output outputs/experiments/.../aggregation_output

# 期望结果
# - Valid pairs > 70%
# - Containment violations < 20%
# - Scale violations < 10%
```

然后运行完整pipeline：

```bash
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

检查inference结果：
- 是否有更多query产生predictions
- NMS后的质量是否可接受
- Evaluation metrics是否提升

## 配置文件修改示例

修改`configs/inference/full_pipeline.yaml`：

```yaml
inference:
  pairing:
    containment_threshold: 0.01    # 改为0.01 (从0.1)
    scale_range: [0.0001, 5.0]     # 改为[0.0001, 5.0] (从[0.001, 0.99])
    use_geometric_score: true      # 保持true
    use_tree_pruning: false        # 保持false
```

**重要提示：**
- 放宽约束后，**NMS变得更加重要**
- 确保`nms.iou_threshold`设置合理 (当前0.35是合理的)
- 可以考虑降低`keep_top_k`来控制质量 (当前200)
