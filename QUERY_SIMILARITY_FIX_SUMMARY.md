# Query Similarity 修复总结

## 修改日期
2025-11-14

## 修改目的
将 evaluation 阶段的文字 query 取得方式改成和 `utils_ov_inference.py:48` 一样的实现方式：
```python
predicted_class = (scale_semantic_score * similarity).softmax(dim=-1)
```

## 参考实现
文件: `/media/Pluto/richkung/My3DIS/3DprojToSiglip/utils_new/utils_ov_inference.py:48`

关键步骤:
1. 计算 cosine similarity (normalized dot product)
2. 乘以 `scale_semantic_score=300`
3. 应用 softmax 在所有 proposals 上

## 已修改的文件

### 1. ✅ `src/my3dis/inference/retrieval.py`

**修改函数**: `compute_cosine_similarity()` (line 35-74)

**修改前** (错误):
```python
# 只使用 temperature scaling
similarities = proposal_norms.dot(query_norm)
if temperature != 1.0:
    similarities = similarities / temperature
return similarities
```

**修改后** (正确):
```python
# 1. 计算余弦相似度
similarities = proposal_norms.dot(query_norm)

# 2. 乘以 scale_semantic_score (默认300)
scaled_similarities = scale_semantic_score * similarities

# 3. 应用 softmax
exp_scores = np.exp(scaled_similarities - np.max(scaled_similarities))
softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)
return softmax_scores
```

**影响范围**: 所有通过 `MultiLevelRetriever.retrieve_single_level()` 的查询

---

### 2. ✅ `src/my3dis/inference/retrieval.py`

**修改类**: `MultiLevelRetriever.__init__()` (line 87-106)

**添加参数**:
- `scale_semantic_score: float = 300.0`

**修改方法**: `retrieve_single_level()` (line 108-144)
- 在调用 `compute_cosine_similarity()` 时传递 `scale_semantic_score` 参数

---

### 3. ✅ `src/my3dis/inference/strategies/hierarchical.py`

**修改位置**: 4处直接调用 `compute_cosine_similarity()` 的地方

1. **Line 253-258**: `_refine_objects()` 中计算 L4 children 的 scores
2. **Line 378-383**: `_get_part_candidates_for_object()` 中计算 part scores
3. **Line 562-567**: `_refine_objects_with_combined_feature()` 中计算 L4 children scores
4. **Line 687-692**: `_get_part_candidates_for_object_with_combined_feature()` 中计算 part scores

**修改内容**: 在所有调用中添加 `scale_semantic_score` 参数
```python
child_scores = compute_cosine_similarity(
    query_feat,
    child_features,
    temperature=self.retriever.temperature,  # 保留向后兼容
    scale_semantic_score=self.retriever.scale_semantic_score  # ✨ 新增
)
```

---

### 4. ✅ `src/my3dis/inference/strategies/combined_query_mixin.py`

**修改方法**: `update_proposals_with_combined_feature()` (line 117-156)

**修改前** (错误):
```python
# 直接计算 cosine similarity (未使用 scale + softmax)
for prop in proposals:
    prop_norm = prop.feature / (np.linalg.norm(prop.feature) + 1e-8)
    similarity = float(np.dot(prop_norm, combined_norm))
    prop.score = similarity
```

**修改后** (正确):
```python
# 收集所有 proposal features
proposal_feats = np.stack([p.feature for p in proposals])

# 使用 compute_cosine_similarity 获取 scaled + softmax scores
from ..retrieval import compute_cosine_similarity

scores = compute_cosine_similarity(
    combined_feat,
    proposal_feats,
    temperature=1.0,
    scale_semantic_score=scale_semantic_score
)

# 更新 proposal scores
for prop, score in zip(proposals, scores):
    prop.score = float(score)
```

**重要性**: 这个修改非常关键，因为这个方法会**覆盖**之前 retrieval 中计算的 scores

**影响范围**: 所有使用 combined query 的策略
- `HierarchicalStrategy` (with `use_combined_query=True`)
- `IndependentStrategy` (with `use_combined_query=True`)
- `ExhaustivePairingWithCombinedQuery`

---

## 已验证正确的文件 (无需修改)

### ✅ `src/my3dis/siglip_assignment/assignment_utils.py`

**已正确实现的函数**:

1. **`assign_labels()`** (line 50-108)
   ```python
   similarity = scale_factor * (features @ text_features.T)  # [N_proposals, N_labels]
   similarity = F.softmax(similarity, dim=-1)
   ```

2. **`assign_part_labels()`** (line 111-167)
   ```python
   similarity_object = scale_factor * (features_object @ text_features_object.T)
   similarity_object = F.softmax(similarity_object, dim=-1)

   similarity_part = scale_factor * (features_part @ text_features_part.T)
   similarity_part = F.softmax(similarity_part, dim=-1)
   ```

3. **`compute_mask_scores()`** (line 170-202)
   ```python
   predicted_class = scale_factor * (point_features @ text_features.T)
   predicted_class = F.softmax(predicted_class, dim=-1)
   ```

这个文件已经在之前就正确实现了参考方法，无需修改。

---

### ✅ 其他 Strategy 文件

经检查，以下文件都**不直接计算** similarity，而是通过 `MultiLevelRetriever` 间接获取分数：
- `src/my3dis/inference/strategies/exhaustive_pairing.py`
- `src/my3dis/inference/strategies/independent.py`
- `src/my3dis/inference/strategies/base.py`

这些文件会自动受益于 `compute_cosine_similarity()` 的修复。

---

## 测试验证

### 测试脚本
文件: `scripts/test_similarity_implementation.py`

### 测试结果
```
✓ NEW IMPLEMENTATION MATCHES REFERENCE!
Max absolute difference: 0.0000000000
Mean absolute difference: 0.0000000000
```

测试确认新实现与参考实现完全一致。

---

## 关键改进

1. **Scale Factor**: 从 temperature scaling 改为使用 `scale_semantic_score=300`
2. **Softmax 归一化**: 在所有 proposals 上应用 softmax，使分数总和为 1.0
3. **数值稳定性**: 使用 `exp(x - max(x))` 避免 overflow
4. **向后兼容**: 保留 `temperature` 参数但不再使用

---

## 影响范围

### 直接影响
- ✅ 所有 inference pipeline 中的 query retrieval
- ✅ Hierarchical strategy 的所有层级检索
- ✅ Combined query 模式的所有策略
- ✅ Independent strategy (通过 retriever)
- ✅ Exhaustive pairing strategy (通过 retriever)

### 不受影响
- ✅ `siglip_assignment/` 模块 (已经使用正确方法)
- ✅ 直接使用 GT proposals 的 evaluation

---

## 使用建议

### 默认配置
```python
# MultiLevelRetriever 初始化
retriever = MultiLevelRetriever(
    proposals_by_level=proposals,
    temperature=0.2,  # 保留但不使用
    min_similarity=0.2,
    scale_semantic_score=300.0  # ✨ 使用这个参数
)
```

### 如需调整 scale factor
```python
# 可以通过参数调整（不推荐，除非有特殊需求）
retriever = MultiLevelRetriever(
    proposals_by_level=proposals,
    scale_semantic_score=100.0  # 降低分数的锐度
)
```

---

## 总结

现在**所有** evaluation 阶段的文字 query 相似度计算都使用与 `utils_ov_inference.py` 一致的方法：
1. Cosine similarity
2. Scale by 300
3. Softmax across all proposals

没有任何地方还在使用旧方法（temperature scaling without softmax）。
