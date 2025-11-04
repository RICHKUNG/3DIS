# Inference Module Overview

## Purpose

This module implements **Open Vocabulary 3D Instance Segmentation** for the MultiScan dataset, designed for evaluation with the Search3D benchmark. It handles multi-level object-part retrieval with geometric constraints and multi-instance awareness.

## Architecture

```
my3dis/inference/
├── __init__.py              # Main exports
├── retrieval.py             # Multi-level feature retrieval
├── pairing.py               # Object-Part pairing logic
├── nms.py                   # Multi-instance aware NMS
├── scoring.py               # Score aggregation strategies
├── geometry.py              # Geometric constraint checking
├── formatter.py             # Search3D output formatting
├── inference_pipeline.py    # End-to-end pipeline
├── config.py                # Configuration dataclasses
└── strategies/
    ├── __init__.py
    ├── base.py              # Strategy base class
    ├── independent.py       # Independent retrieval strategy
    └── hierarchical.py      # Hierarchical coarse-to-fine strategy
```

## Module Dependencies

```
retrieval.py
    └─> geometry.py (for IoU computation in deduplication)

pairing.py
    ├─> retrieval.py (Proposal dataclass)
    ├─> geometry.py (geometric validation)
    └─> scoring.py (score aggregation)

nms.py
    └─> geometry.py (IoU, centroid distance)

strategies/
    ├─> retrieval.py
    ├─> pairing.py
    ├─> nms.py
    └─> scoring.py

inference_pipeline.py
    ├─> retrieval.py
    ├─> pairing.py
    ├─> nms.py
    ├─> scoring.py
    ├─> formatter.py
    └─> strategies/

formatter.py
    ├─> nms.py (Prediction dataclass)
    └─> pairing.py (ObjectPartPair dataclass)
```

## Key Data Structures

### Proposal
```python
@dataclass
class Proposal:
    level: str              # 'L2', 'L4', or 'L6'
    proposal_id: int        # Index in level
    mask: np.ndarray        # Binary mask (N_points,)
    feature: np.ndarray     # SigLIP embedding (D,)
    object_id: int          # From tracking (for family tree)
    score: float            # Similarity score
```

### ObjectPartPair
```python
@dataclass
class ObjectPartPair:
    object_proposal: Proposal
    part_proposal: Proposal
    object_score: float
    part_score: float
    geometric_score: float
    combined_score: float
    label_id: int           # Joint semantic label
```

### Prediction
```python
@dataclass
class Prediction:
    mask: np.ndarray        # Binary mask
    score: float            # Confidence
    label_id: int           # Semantic label
    metadata: dict          # Additional info
```

## Execution Flow

### Independent Strategy

```
1. Retrieve Object Candidates
   ├─> Retrieve from L2 (top-K)
   ├─> Retrieve from L4 (top-K)
   └─> Deduplicate (IoU > 0.95)
       └─> Object Pool (M candidates)

2. Retrieve Part Candidates
   ├─> Retrieve from L4 (top-K)
   ├─> Retrieve from L6 (top-K)
   └─> Deduplicate
       └─> Part Pool (N candidates)

3. Geometric Pairing
   For each (object, part) in Object Pool × Part Pool:
       ├─> Check containment ratio > threshold
       ├─> Check scale ratio in range
       ├─> Compute geometric score
       └─> Aggregate scores
           └─> Valid Pairs (P pairs)

4. Multi-Instance NMS
   Sort pairs by score:
       For each pair:
           ├─> Compute IoU with kept pairs
           ├─> If IoU > threshold:
           │   ├─> Check centroid distance
           │   └─> Suppress if close
           └─> Keep if unique or spatially separated
               └─> Final Predictions (Q predictions)
```

### Hierarchical Strategy

```
1. Coarse Localization (L2)
   └─> Top-K L2 proposals
       └─> Coarse Candidates

2. Multi-Resolution Refinement
   For each L2 candidate:
       ├─> Keep L2 proposal
       ├─> Get L4 children (from family tree)
       ├─> Score children with object query
       └─> Keep if score > threshold or score > parent + margin
           └─> Object Candidates (L2 + L4 mix)

3. Part Search in Relevant Regions
   For each object candidate:
       ├─> Get descendants (L4, L6) from tree
       ├─> Score with part query
       ├─> Filter by threshold
       └─> Pair with object
           └─> Candidate Pairs

4. Cross-Path NMS
   └─> Apply multi-instance NMS
       └─> Final Predictions
```

## Configuration Hierarchy

```
InferenceConfig
├── strategy: str
├── retrieval: RetrievalConfig
│   ├── temperature
│   ├── min_similarity
│   └── top_k_per_level
├── pairing: PairingConfig
│   ├── containment_threshold
│   └── scale_range
├── nms: NMSConfig
│   ├── iou_threshold
│   └── centroid_distance_ratio
├── scoring: ScoringConfig
│   ├── method
│   └── weights
├── formatter: FormatterConfig
├── independent: IndependentStrategyConfig
└── hierarchical: HierarchicalStrategyConfig
```

## Output Format (Search3D)

```python
{
    'pred_classes': np.ndarray,  # Shape: (M,), dtype: int32
    'pred_scores': np.ndarray,   # Shape: (M,), dtype: float
    'pred_masks': np.ndarray     # Shape: (P, M), dtype: uint8
}
```

Where:
- `P` = number of points in scene
- `M` = number of predictions
- Each column in `pred_masks` is a binary mask for one prediction

## Key Algorithms

### Multi-Instance NMS

**Problem**: Traditional NMS suppresses all overlapping predictions, but in multi-instance scenarios (e.g., multiple cabinet doors), adjacent instances may have high IoU.

**Solution**: Add spatial separation check:
```python
if IoU(pred1, pred2) > threshold:
    centroid_dist = distance(centroid1, centroid2)
    avg_size = (size1 + size2) / 2
    if centroid_dist < ratio * avg_size:
        suppress(pred2)  # Close together → suppress
    else:
        keep(pred2)      # Spatially separated → keep both
```

### Geometric Pairing

**Constraints**:
1. **Containment**: Part should be mostly inside object
   ```python
   containment_ratio = |part ∩ object| / |part|
   valid if containment_ratio > 0.5
   ```

2. **Scale**: Part should be smaller than object
   ```python
   scale_ratio = |part| / |object|
   valid if 0.01 < scale_ratio < 0.9
   ```

3. **Proximity** (optional): Centroids should be close
   ```python
   distance(centroid_part, centroid_object) < max_distance
   ```

### Score Aggregation

**Weighted Sum** (default):
```python
score = w_obj * sim_obj + w_part * sim_part + w_geo * geo_score
```

**Geometric Mean**:
```python
score = (sim_obj * sim_part * geo_score) ^ (1/3)
```

**With Hierarchy Bonus**:
```python
if object_level < part_level:  # e.g., L2 < L6
    score += hierarchy_bonus
```

## Performance Considerations

### Computational Complexity

**Independent Strategy**:
- Retrieval: O(L × N × D) where L=levels, N=proposals/level, D=embedding dim
- Pairing: O(M × N) where M=object candidates, N=part candidates
- NMS: O(P²) where P=pairs

**Hierarchical Strategy**:
- Coarse: O(N_L2 × D)
- Refinement: O(K × C × D) where K=coarse candidates, C=children/candidate
- Part search: O(K × D × D) where D=descendants
- NMS: O(P²)

**Memory**:
- Proposals: L × N × (P + D) where P=points, D=embedding dim
- Pairs: O(M × N) worst case (geometric pruning reduces this)

### Optimization Tips

1. **Use tree pruning**: Reduces pairing from O(M×N) to O(M×D) where D << N
2. **Batch similarity**: Vectorize cosine similarity computations
3. **Early filtering**: Apply thresholds early to reduce candidates
4. **Soft NMS**: Can be faster than repeated IoU checks

## Testing

### Unit Tests (TODO)

```python
# test_retrieval.py
- test_cosine_similarity()
- test_multi_level_retrieval()
- test_deduplication()

# test_pairing.py
- test_geometric_validation()
- test_tree_guided_pairing()

# test_nms.py
- test_multi_instance_nms()
- test_spatial_separation()

# test_strategies.py
- test_independent_strategy()
- test_hierarchical_strategy()

# test_formatter.py
- test_format_predictions()
- test_validation()
```

### Integration Tests (TODO)

```python
# test_pipeline.py
- test_end_to_end_inference()
- test_multi_query_inference()
- test_config_loading()
```

## Future Enhancements

1. **Learned Fusion**: Train MLP to combine scores instead of fixed weights
2. **Image Re-ranking**: Use RGB crops for second-stage verification
3. **Mask Splitting**: Oversegment large masks that cover multiple instances
4. **Graph Clustering**: Use proposal graph for better grouping
5. **Adaptive Thresholds**: Learn thresholds per query category
6. **Parallel Processing**: Multi-GPU inference for large scenes

## References

- **Algorithm Design**: [`docs/inference/algo_advice_md.md`](../../../docs/inference/algo_advice_md.md)
- **Inference Plan**: [`docs/inference/inference_plan.md`](../../../docs/inference/inference_plan.md)
- **Usage Guide**: [`docs/inference/README.md`](../../../docs/inference/README.md)
- **Search3D Paper**: arXiv:2409.18431v2
