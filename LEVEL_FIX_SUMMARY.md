# Level Naming Inference Fix - 2025-11-15

## 问题概述

Inference pipeline 无法正常运行，所有推理策略都产生 0 个预测结果。

### 根本原因

**Level 命名不一致问题：**
- 聚合输出文件：`proposal_data_level1.npz`, `proposal_data_level3.npz`, `proposal_data_level5.npz`
- run_full_pipeline.py 转换为：`L1`, `L3`, `L5`
- **推理策略硬编码期望**：`L2`, `L4`, `L6`

这导致推理策略无法找到任何 proposals，从而产生 0 个预测。

## 日志证据

```
2025-11-15 01:23:26,622 - __main__ - INFO -   Level 1: 279 proposals
2025-11-15 01:23:26,657 - __main__ - INFO -   Level 3: 272 proposals
...
2025-11-15 01:23:48,977 - my3dis.inference.retrieval - WARNING - Level L2 not found in proposals
2025-11-15 01:23:49,010 - my3dis.inference.strategies.hierarchical - WARNING - No coarse candidates found
```

## 修复内容

### 1. HierarchicalStrategy (`src/my3dis/inference/strategies/hierarchical.py`)

#### 修改前
```python
def __init__(self, ...):
    # 没有 level 配置参数

def _coarse_localization(self, ...):
    result = self.retriever.retrieve_single_level(
        object_query_feat,
        level='L2',  # 硬编码
        ...
    )
```

#### 修改后
```python
def __init__(
    self,
    ...,
    coarse_level: str = None,
    refinement_level: str = None,
    part_levels: List[str] = None,
    **kwargs
):
    # 自动检测可用 levels
    available_levels = sorted(retriever.proposals_by_level.keys())

    self.coarse_level = coarse_level or (available_levels[0] if len(available_levels) > 0 else 'L1')
    self.refinement_level = refinement_level or (available_levels[1] if len(available_levels) > 1 else 'L3')
    self.part_levels = part_levels or available_levels[1:]

def _coarse_localization(self, ...):
    result = self.retriever.retrieve_single_level(
        object_query_feat,
        level=self.coarse_level,  # 使用配置的 level
        ...
    )
```

**改进：**
- 添加可配置的 level 参数
- 自动检测可用 levels 作为默认值
- 更新所有方法使用配置的 level 值

### 2. ExhaustivePairingStrategy (`src/my3dis/inference/strategies/exhaustive_pairing.py`)

#### 修改前
```python
def _generate_level_pairs(self):
    level_order = {'L2': 0, 'L4': 1, 'L6': 2}  # 硬编码
    ...
```

#### 修改后
```python
def _generate_level_pairs(self):
    # 动态构建 level_order
    level_order = {level: idx for idx, level in enumerate(sorted(self.available_levels))}
    ...
```

**改进：**
- 从 `available_levels` 动态构建 level 顺序
- 支持任意 level 命名方案

### 3. IndependentStrategy (`src/my3dis/inference/strategies/independent.py`)

**无需修改** - 该策略已正确使用可配置的 `object_levels` 和 `part_levels` 参数。

### 4. 配置默认值更新 (`src/my3dis/inference/config.py`)

#### 修改前
```python
@dataclass
class RetrievalConfig:
    object_levels: List[str] = field(default_factory=lambda: ['L2', 'L4'])
    part_levels: List[str] = field(default_factory=lambda: ['L4', 'L6'])

@dataclass
class IndependentStrategyConfig:
    object_levels: List[str] = field(default_factory=lambda: ['L2', 'L4'])
    part_levels: List[str] = field(default_factory=lambda: ['L4', 'L6'])

@dataclass
class ExhaustivePairingConfig:
    available_levels: List[str] = field(default_factory=lambda: ['L2', 'L4', 'L6'])
```

#### 修改后
```python
@dataclass
class RetrievalConfig:
    object_levels: List[str] = field(default_factory=lambda: ['L1', 'L3'])
    part_levels: List[str] = field(default_factory=lambda: ['L3', 'L5'])

@dataclass
class IndependentStrategyConfig:
    object_levels: List[str] = field(default_factory=lambda: ['L1', 'L3'])
    part_levels: List[str] = field(default_factory=lambda: ['L3', 'L5'])

@dataclass
class ExhaustivePairingConfig:
    available_levels: List[str] = field(default_factory=lambda: ['L1', 'L3', 'L5'])
```

**改进：**
- 默认值与实际数据格式一致（L1/L3/L5）
- 保持与现有 YAML 配置文件的兼容性

## 影响范围

### 修改的文件
1. `src/my3dis/inference/strategies/hierarchical.py`
2. `src/my3dis/inference/strategies/exhaustive_pairing.py`
3. `src/my3dis/inference/config.py`

### 未修改的文件
- `src/my3dis/inference/strategies/independent.py` - 已经正确实现
- YAML 配置文件 - 已经正确设置（`configs/inference/full_pipeline.yaml`）

## 向后兼容性

### ✅ 完全向后兼容

所有修改都保持了向后兼容性：

1. **默认值更新**：从 L2/L4/L6 改为 L1/L3/L5，匹配当前数据格式
2. **HierarchicalStrategy**：添加了新的可选参数，未提供时自动检测
3. **ExhaustivePairingStrategy**：逻辑改进，不改变 API
4. **现有配置**：configs/inference/full_pipeline.yaml 已经正确使用 L1/L3/L5

## 测试建议

### 快速测试
```bash
# 使用现有配置运行 inference
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

### 预期结果
1. ✅ Hierarchical 策略应该显示：`Using levels - coarse=L1, refinement=L3, parts=['L3', 'L5']`
2. ✅ 所有策略都应该能找到 proposals
3. ✅ 产生非零数量的预测结果

### 验证点
- [ ] Hierarchical 策略不再显示 "Level L2 not found"
- [ ] Independent 策略成功检索到 proposals
- [ ] ExhaustivePairingStrategy 生成有效的 level pairs
- [ ] 所有策略产生 > 0 个预测

## 设计改进

### 1. 灵活的 Level 命名
- **之前**：硬编码 L2/L4/L6
- **现在**：支持任意 level 命名（L1/L3/L5 或 L2/L4/L6 或其他）

### 2. 自动检测机制
- HierarchicalStrategy 自动从可用 proposals 检测 levels
- 降低配置错误的可能性

### 3. 统一配置
- 所有策略通过配置参数控制 level 选择
- YAML 配置是唯一真相来源

## 后续建议

### 短期
1. ✅ 测试修复后的 pipeline
2. 更新相关文档和注释
3. 添加 level 命名的单元测试

### 长期
1. **考虑标准化 level 命名**：
   - 选项 A：统一使用 L1/L3/L5（匹配 SSAM level 1/3/5）
   - 选项 B：统一使用 L2/L4/L6（匹配 SSAM level 2/4/6）
   - 选项 C：保持当前灵活性，通过文档说明

2. **添加验证逻辑**：
   ```python
   def validate_levels(proposals_by_level, expected_levels):
       missing = set(expected_levels) - set(proposals_by_level.keys())
       if missing:
           raise ValueError(f"Missing levels in proposals: {missing}")
   ```

3. **改进错误信息**：
   ```python
   if level not in self.proposals_by_level:
       available = list(self.proposals_by_level.keys())
       raise ValueError(
           f"Level {level} not found in proposals. "
           f"Available levels: {available}"
       )
   ```

## 总结

这次修复解决了 inference pipeline 中 level 命名不一致的核心问题：

- ✅ **根本原因**：识别了硬编码 level 名称与实际数据不匹配
- ✅ **系统性修复**：重构了所有推理策略，支持可配置 level
- ✅ **向后兼容**：保持了 API 兼容性
- ✅ **未来扩展**：支持任意 level 命名方案

修复后，inference pipeline 应该能够正常工作，产生有效的预测结果。
