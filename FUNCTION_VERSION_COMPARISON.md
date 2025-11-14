# Function Version Comparison: pipeline.ipynb vs notebook_pipeline.py

## Summary

**Key Finding:** `notebook_pipeline.py` is actually the **IMPROVED** version with better flexibility.

## Detailed Comparison

### 1. `generate_grounding_features_siglip`

#### pipeline.ipynb (Cell 16)
```python
def generate_grounding_features_siglip(
    model,
    processor,
    color_paths,
    inside_mask,
    projected_points,
    points_depth,
    depth_maps,
    device,
    scaling_params,
    vis_depth_threshold,
    proposal_masks,
    topk=1,
    use_optimized=True,  # NEW: Use optimized batched version
    batch_size=32,       # NEW: Batch size for optimization
):
    # ...
    if use_optimized:
        return generate_grounding_features_siglip_batched(
            # ... parameters ...
            use_sparse=True,  # ⚠️ HARDCODED
        )
```

**Issue:** `use_sparse` is **HARDCODED** to `True`, not configurable.

#### notebook_pipeline.py
```python
def generate_grounding_features_siglip(
    model,
    processor,
    color_paths: List[str],
    inside_mask: torch.Tensor,
    projected_points: torch.Tensor,
    points_depth: torch.Tensor,
    depth_maps,
    device: torch.device,
    scaling_params,
    vis_depth_threshold: float,
    proposal_masks: torch.Tensor,
    topk: int = 1,
    use_optimized: bool = True,
    batch_size: int = 32,
    use_sparse: bool = True,  # ✅ PARAMETERIZED
) -> torch.Tensor:
    # ...
    if use_optimized:
        return generate_grounding_features_siglip_batched(
            # ... parameters ...
            use_sparse=use_sparse,  # ✅ USES PARAMETER
        )
```

**Improvement:** `use_sparse` is a **configurable parameter** with type hints.

### 2. `pool_point_to_proposal_features`

#### Both versions are IDENTICAL ✅

```python
def pool_point_to_proposal_features(pc_features, proposal_masks):
    M = proposal_masks.to(pc_features.device).float()
    X = pc_features
    counts = M.sum(dim=0, keepdim=True).clamp_min(1.0)
    proposal_feats = (M.T @ X) / counts.T
    proposal_feats = F.normalize(proposal_feats, dim=1)
    return proposal_feats
```

### 3. `extract_proposal_features_siglip`

#### pipeline.ipynb
**❌ DOES NOT EXIST**

Users must manually call:
1. `generate_grounding_features_siglip()`
2. `pool_point_to_proposal_features()`

#### notebook_pipeline.py
**✅ EXISTS** - High-level wrapper function

```python
def extract_proposal_features_siglip(
    pipeline,
    proposal_masks: torch.Tensor,
    model,
    processor,
    device: torch.device,
    topk: int,
    batch_size: int = 32,
    use_optimized: bool = True,
    use_sparse: bool = True,
) -> torch.Tensor:
    """
    Adapter around the notebook utilities to match the GT similarity script.
    """
    pc_features = generate_grounding_features_siglip(
        model=model,
        processor=processor,
        color_paths=pipeline.world2cam.color_paths,
        inside_mask=pipeline.inside_mask,
        projected_points=pipeline.projected_points,
        points_depth=pipeline.point_depth,
        depth_maps=pipeline.depth_maps,
        device=device,
        scaling_params=pipeline.scaling_params,
        vis_depth_threshold=pipeline.vis_depth_threshold,
        proposal_masks=proposal_masks,
        topk=topk,
        use_optimized=use_optimized,
        batch_size=batch_size,
        use_sparse=use_sparse,
    )

    return pool_point_to_proposal_features(pc_features, proposal_masks)
```

**Benefit:** Simplified API for external scripts.

## Feature Matrix

| Feature | pipeline.ipynb | notebook_pipeline.py |
|---------|----------------|----------------------|
| `generate_grounding_features_siglip` | ✅ | ✅ |
| Type hints | ❌ | ✅ |
| Configurable `use_sparse` | ❌ (hardcoded) | ✅ |
| `pool_point_to_proposal_features` | ✅ | ✅ |
| `extract_proposal_features_siglip` wrapper | ❌ | ✅ |
| Pipeline integration | Manual | Automatic |

## Recommendation

**Current state is CORRECT** ✅

`notebook_pipeline.py` provides:
1. **Better flexibility** - Configurable `use_sparse` parameter
2. **Type safety** - Full type hints
3. **Convenience** - High-level wrapper function
4. **Compatibility** - Works with both notebook and script workflows

## Usage in `check_siglip_gt_similarity.py`

The current implementation is **already using the best version**:

```python
# From check_siglip_gt_similarity.py (line 340)
proposal_features = extract_proposal_features_siglip(
    pipeline=pipeline,
    proposal_masks=proposal_masks,
    model=model,
    processor=processor,
    device=torch_device,
    topk=args.topk_views,
    batch_size=args.batch_size,
    use_optimized=args.use_optimized,  # ✅ Configurable
    use_sparse=args.use_sparse,        # ✅ Configurable
)
```

This provides full control over both optimization features.

## Conclusion

**No changes needed!** The current implementation in `check_siglip_gt_similarity.py` is using the most recent and flexible version from `notebook_pipeline.py`, which is actually an **improvement** over the notebook version.

The notebook version (`pipeline.ipynb`) was likely created for quick experimentation and hardcoded `use_sparse=True` for convenience. The script version (`notebook_pipeline.py`) properly parameterized it for production use.
