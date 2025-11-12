# Pipeline Configuration Changelog

## 2025-11-11 - Bug Fixes and Enhancements

### Fixed Issues

1. **Pooling Mode Not Displaying Correctly** ✓
   - **Problem**: Config specified `pooling_mode: voting`, but logs showed "averaging"
   - **Root Cause**: `InferenceConfig.from_yaml()` was being called with a dict instead of a file path
   - **Solution**: Added `_dict_to_inference_config()` helper function to properly construct InferenceConfig from nested dict
   - **Files Modified**: `scripts/run_full_pipeline.py`

2. **Ground Truth Path Not Being Used** ✓
   - **Problem**: `gt_path` from evaluation config was not being passed through
   - **Root Cause**: Evaluation stage only checked if path was None but didn't display it
   - **Solution**:
     - Added detailed logging for evaluation settings in config summary
     - Added `--gt-path` CLI argument for easy override
   - **Files Modified**: `scripts/run_full_pipeline.py`

### New Features

1. **Enhanced CLI Arguments**
   - `--gt-path`: Override ground truth path from command line
   - `--pooling-mode`: Override feature pooling mode (average/voting)

   Example:
   ```bash
   python scripts/run_full_pipeline.py \
       --config configs/inference/full_pipeline.yaml \
       --gt-path /path/to/ground_truth \
       --pooling-mode voting
   ```

2. **Improved Configuration Logging**
   - Now displays comprehensive settings at pipeline start:
     - Aggregation settings (IoU, area fraction, **pooling mode**, SigLIP model)
     - Inference settings (strategy, NMS IoU, keep top-k)
     - Evaluation settings (**GT path**, query file)
   - Makes it easy to verify configuration is loaded correctly

3. **Updated Documentation**
   - Added explanation of `voting` vs `average` pooling modes
   - **Recommendation**: Use `voting` for OV-3DIS (preserves magnitude)
   - Updated all example configs to use `voting` by default
   - Added CLI override examples to README

### Configuration Files Updated

1. **`configs/inference/full_pipeline.yaml`**
   - Changed `pooling_mode: voting` (user modification)
   - Changed `evaluation.enabled: true` (user modification)

2. **`configs/inference/example_test_1108.yaml`**
   - Updated `pooling_mode: voting` with comment
   - Ready to use with your experiment

3. **`configs/inference/pipeline_template.yaml`**
   - Updated `pooling_mode: voting` as default
   - Added clarifying comment

4. **`configs/inference/README.md`**
   - Added CLI override examples for `--gt-path` and `--pooling-mode`
   - Added detailed explanation of pooling modes
   - Explained why `voting` is recommended for OV-3DIS

### Technical Details

#### Pooling Mode Behavior

**`voting` (Recommended for OV-3DIS):**
```python
# Average point features WITHOUT L2 normalization
proposal_features = point_features.mean(dim=0)  # Preserves magnitude
```

**`average` (Standard retrieval):**
```python
# Average point features WITH L2 normalization
proposal_features = point_features.mean(dim=0)
proposal_features = F.normalize(proposal_features, p=2, dim=-1)  # Unit length
```

The `voting` mode matches the behavior in `utils_ov_inference.py` (lines 54-55) and is required for proper score calibration in multi-instance detection.

#### Config Loading Flow

Before fix:
```python
# ❌ Incorrect: from_yaml expects a file path, not a dict
inference_config = InferenceConfig.from_yaml(config['inference'])
```

After fix:
```python
# ✓ Correct: Use helper to construct from dict
inference_config = _dict_to_inference_config(config['inference'])
```

### Migration Guide

If you have existing configs:

1. **Update pooling_mode**:
   ```yaml
   aggregation:
     pooling_mode: voting  # Change from average to voting
   ```

2. **Add gt_path if using evaluation**:
   ```yaml
   evaluation:
     enabled: true
     gt_path: /path/to/multiscan_annotations_search3d/ov_part_annotations
   ```

3. **Run with new script**:
   ```bash
   python scripts/run_full_pipeline.py --config your_config.yaml
   ```

### Verification

Check that the configuration summary shows correct values:

```
PIPELINE CONFIGURATION
================================================================================
...
Aggregation settings:
  IoU threshold:   0.8
  Area fraction:   0.002
  Pooling mode:    voting          ← Should match your config
  Extract features: True
  SigLIP model:    google/siglip-so400m-patch14-384

Inference settings:
  Strategy:        independent
  NMS IoU:         0.35
  Keep top-k:      200

Evaluation settings:
  GT path:         /your/gt/path   ← Should be set if evaluation enabled
  Query file:      None
```

### Related Files

- `scripts/run_full_pipeline.py` - Main pipeline runner (modified)
- `configs/inference/*.yaml` - Configuration files (updated)
- `configs/inference/README.md` - Documentation (enhanced)
- `src/my3dis/aggregation/aggregation_pipeline.py` - Pooling implementation (unchanged)

### Testing

All changes are backward compatible. Existing scripts and configs will continue to work.

To test the fixes:
```bash
# 1. Verify pooling mode is correctly loaded
python scripts/run_full_pipeline.py \
    --config configs/inference/example_test_1108.yaml \
    | grep "Pooling mode"

# Expected output: "Pooling mode:    voting"

# 2. Verify gt_path override works
python scripts/run_full_pipeline.py \
    --config configs/inference/example_test_1108.yaml \
    --gt-path /test/path \
    | grep "GT path"

# Expected output: "GT path:         /test/path"
```
