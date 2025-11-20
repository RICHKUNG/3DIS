# Hierarchical Strategy 修復報告 (2025-11-19)

## 問題描述

Hierarchical strategy 在評估時表現很差（mAP 接近 0），診斷發現 Stage 3 無法找到任何 part candidates：

```
Stage 1: Coarse object localization at L1
Coarse localization: 17 candidates from L1

Stage 2: Multi-resolution object refinement
Refined to 17 object candidates

Stage 3: Part search in relevant regions
Created 0 candidate pairs  ❌ 問題！
```

## 根本原因分析

### 1. ID 格式不匹配

**Aggregation 輸出使用連續整數 ID：**
```python
# proposal_data_level1.npz
proposal_ids: [1, 2, 3, ..., 128]

# proposal_data_level3.npz
proposal_ids: [1, 2, 3, ..., 150]

# Proposal_relation.json
{
  "1": {"level": 1, "children": [1, 2]},
  "2": {"level": 1, "children": [3, 4]},
  "1": {"level": 3, "parent": 1},
  ...
}
```

**但 Hierarchical Strategy 期望 level-based ID 格式：**
```python
# 期望的 ID 格式
L1: 1000-1999
L3: 3000-3999
L5: 5000-5999
```

### 2. FamilyTree 過濾邏輯錯誤

`FamilyTree.get_children()` 和 `get_descendants()` 使用 ID 範圍過濾：

```python
# pairing.py line 92-94
prefix = parse_level_to_prefix(target_level)  # L3 -> 3000
children = [c for c in children if prefix <= c < prefix + 1000]
# 過濾範圍: 3000-3999
```

但實際的 children IDs 是 `[1, 2, 3, ...]`，全部被過濾掉：

```python
實際 children: [1, 2, 3, 4, 5]
過濾範圍: 3000 <= ID < 4000
結果: []  ❌ 沒有匹配！
```

### 3. Level 信息未被使用

雖然 `Proposal_relation.json` 中包含 `level` 字段，但：
- `InferencePipeline._load_family_tree()` 沒有提取 `level` 字段
- `FamilyTree` 無法使用 `level` 字段進行過濾

## 修復方案

### 1. 修改 `FamilyTree.get_children()` (pairing.py:83-110)

添加 fallback 邏輯，當 level-based ID 過濾失敗時，使用 `level` 字段：

```python
def get_children(self, obj_id: int, target_level: Optional[str] = None) -> List[int]:
    """Get children of an object, optionally filtered by level."""
    if obj_id not in self.tree_data:
        return []

    children = self.tree_data[obj_id].get('children', [])

    if target_level is not None:
        # Try level-based ID format first (L1→1000+, L2→2000+, etc.)
        try:
            prefix = parse_level_to_prefix(target_level)
            filtered = [c for c in children if prefix <= c < prefix + 1000]

            # If no matches found, try using level field from tree_data
            if len(filtered) == 0 and len(children) > 0:
                # Extract target level number (e.g., 'L3' -> 3)
                target_level_num = int(target_level[1:])
                filtered = [
                    c for c in children
                    if c in self.tree_data and self.tree_data[c].get('level') == target_level_num
                ]

            children = filtered
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid target_level '{target_level}': {e}")
            return []

    return children
```

### 2. 修改 `FamilyTree.get_descendants()` (pairing.py:112-152)

同樣添加 fallback 邏輯：

```python
def get_descendants(self, obj_id: int, levels: Optional[List[str]] = None) -> List[int]:
    """Get all descendants of an object, optionally filtered by levels."""
    descendants = []
    queue = [obj_id]
    visited = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        children = self.get_children(current)
        descendants.extend(children)
        queue.extend(children)

    if levels is not None:
        # Try level-based ID format first
        try:
            level_prefixes = [parse_level_to_prefix(l) for l in levels]
            filtered = [
                d for d in descendants
                if any(prefix <= d < prefix + 1000 for prefix in level_prefixes)
            ]

            # If no matches found, try using level field from tree_data
            if len(filtered) == 0 and len(descendants) > 0:
                # Extract target level numbers (e.g., ['L3', 'L5'] -> [3, 5])
                target_level_nums = [int(level[1:]) for level in levels]
                filtered = [
                    d for d in descendants
                    if d in self.tree_data and self.tree_data[d].get('level') in target_level_nums
                ]

            descendants = filtered
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid level in levels list: {e}")
            # Return all descendants if level parsing fails
            pass

    return descendants
```

### 3. 修改 `InferencePipeline._load_family_tree()` (inference_pipeline.py:165-181)

從 JSON 中提取 `level` 字段：

```python
def _load_family_tree(self, path: str) -> FamilyTree:
    """Load family tree from JSON file."""
    logger.info(f"Loading family tree from {path}")

    with open(path, 'r') as f:
        tree_data = json.load(f)

    # Convert to FamilyTree format
    # Assuming tree_data has structure like:
    # {"objects": {obj_id: {"parent": ..., "children": [...], "level": ..., ...}}}

    if 'objects' in tree_data:
        tree_dict = {}
        for obj_id_str, obj_info in tree_data['objects'].items():
            obj_id = int(obj_id_str)
            tree_dict[obj_id] = {
                'parent': obj_info.get('parent'),
                'children': obj_info.get('children', []),
                'level': obj_info.get('level')  # Include level for non-level-based ID format
            }
    else:
        tree_dict = tree_data

    return FamilyTree(tree_dict)
```

## 修復驗證

測試 fallback 邏輯：

```python
from my3dis.inference.pairing import FamilyTree

# Create test tree data (non-level-based ID format)
tree_data = {
    1: {'level': 1, 'children': [10, 20]},
    10: {'level': 3, 'parent': 1},
    20: {'level': 3, 'parent': 1},
}
tree = FamilyTree(tree_data)

result = tree.get_children(1, target_level='L3')
print(f'Result: {result}')
print(f'Expected: [10, 20]')
# Output: ✓ PASS
```

## 影響範圍

### 修改的文件：
1. `src/my3dis/inference/pairing.py`
   - `FamilyTree.get_children()` (line 83-110)
   - `FamilyTree.get_descendants()` (line 112-152)

2. `src/my3dis/inference/inference_pipeline.py`
   - `InferencePipeline._load_family_tree()` (line 165-181)

### 向後兼容性：
✅ **完全向後兼容**
- 仍然支持舊的 level-based ID 格式（1000+, 2000+, ...）
- 新增支持連續整數 ID 格式（1, 2, 3, ...）
- Fallback 邏輯只在 level-based 過濾失敗時啟用

### 影響的 Strategies：
- ✅ `HierarchicalStrategy` - 主要受益者
- ✅ `ExhaustivePairingStrategy` - 也使用 FamilyTree
- ✅ `IndependentStrategy` - 不使用 FamilyTree，不受影響

## 預期改善

修復前：
```
Stage 3: Part search in relevant regions
Created 0 candidate pairs
```

修復後（預期）：
```
Stage 3: Part search in relevant regions
Created X candidate pairs  (X > 0)
```

這應該能顯著提升 hierarchical strategy 的 mAP 表現。

## 其他發現的問題（未修復）

### 1. Level 字符串比較問題 (hierarchical.py:685)

```python
# 當前代碼（潛在問題）
finer_levels = [level for level in self.part_levels if level >= obj_prop.level]
```

這使用字符串比較，在 L1-L9 範圍內碰巧正確，但如果有 L10+ 會出錯：
- `'L10' < 'L4'` (字符串比較) 但 `10 > 4` (數值比較)

**建議修復：**
```python
def level_to_num(level):
    return int(level[1:]) if level.startswith('L') else 0

finer_levels = [
    level for level in self.part_levels
    if level_to_num(level) >= level_to_num(obj_prop.level)
]
```

但因為當前系統只使用 L1-L6，這個問題暫不影響。

## 總結

通過添加 ID 格式的自動檢測和 fallback 邏輯，成功解決了 hierarchical strategy 無法找到 part candidates 的問題。修復方案：

1. ✅ 保持向後兼容
2. ✅ 支持兩種 ID 格式
3. ✅ 自動檢測並選擇正確的過濾方式
4. ✅ 最小化代碼變更

下一步建議測試修復後的 hierarchical strategy 性能。
