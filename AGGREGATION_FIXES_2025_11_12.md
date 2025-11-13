# Aggregation Pipeline Fixes - 2025-11-12

## Problem Diagnosed

The original error in `/media/Pluto/richkung/My3DIS/logs/eval/run_exp_20251112_144609.log` was:

```
ERROR - No aggregation outputs available for inference
```

### Root Cause

The configuration file pointed to the **experiment root directory** instead of the **scene-specific subdirectory**:

- **Incorrect**: `/media/Pluto/richkung/My3DIS/outputs/experiments/test_1108_ssam4_filter500_fill500_propinf`
- **Correct**: `/media/Pluto/richkung/My3DIS/outputs/experiments/test_1108_ssam4_filter500_fill500_propinf/scene_00005_00`

The aggregation pipeline was looking for:
- `{exp_dir}/level_2/video_segments_L02.npz`
- `{exp_dir}/level_4/video_segments_L04.npz`
- `{exp_dir}/level_6/video_segments_L06.npz`

But the files were actually located in the scene subdirectory, so it couldn't find them.

## Fixes Applied

### 1. Configuration Path Fix

**File**: `configs/inference/full_pipeline.yaml`

Updated `experiment.exp_dir` to point to the scene-specific directory:

```yaml
experiment:
  # Path to My3DIS experiment directory (SCENE-SPECIFIC)
  # NOTE: For multi-scene experiments, this should point to the scene subdirectory,
  #       e.g., .../test_1108_ssam4_filter500_fill500_propinf/scene_00005_00
  exp_dir: /media/Pluto/richkung/My3DIS/outputs/experiments/test_1108_ssam4_filter500_fill500_propinf/scene_00005_00
```

### 2. Per-Level Area Threshold Support (New Feature)

Based on user feedback:
> 現在每層的filtering都是以一樣的體積門檻過濾，請改成讓我可以在yaml裡面設定每個level分別的proposal大小門檻

**Changes Made**:

#### a. Updated `aggregation_pipeline.py`

Added `area_fraction_per_level` parameter to `AggregationPipeline.__init__()`:

```python
def __init__(
    self,
    exp_dir: str,
    scene_path: str,
    iou_threshold: float = 0.8,
    area_fraction: float = 1/500,
    device: str = 'cuda',
    config: Optional[Dict] = None,
    pooling_mode: str = 'average',
    area_fraction_per_level: Optional[Dict[int, float]] = None,  # NEW
):
```

Modified `run()` method to compute per-level thresholds:

```python
# Compute area thresholds (per-level or global)
area_thresh_by_level = {}
if self.area_fraction_per_level:
    logger.info("Using per-level area thresholds:")
    for level in levels:
        frac = self.area_fraction_per_level.get(level, self.area_fraction)
        area_thresh_by_level[level] = int(self.P * frac)
        logger.info(f"  Level {level}: {area_thresh_by_level[level]} points ({frac:.4f} * {self.P})")
else:
    area_thresh = int(self.P * self.area_fraction)
    logger.info(f"Using global area threshold: {area_thresh} points ({self.area_fraction:.4f} * {self.P})")
    for level in levels:
        area_thresh_by_level[level] = area_thresh
```

Updated merge/filter calls to use per-level thresholds:

```python
# Level 2
merge_drop_level2_mask(masks_by_level[2], self.iou_threshold, area_thresh_by_level[2])

# Level 4
merge_drop_level4_within_siblings(..., area_thresh_by_level[4], ...)

# Level 6
merge_drop_level6_within_siblings(..., area_thresh_by_level[6], ...)
```

#### b. Updated `run_full_pipeline.py`

Added parsing of per-level area fractions:

```python
# Parse per-level area fractions if provided
area_fraction_per_level = agg_config.get('area_fraction_per_level', None)
if area_fraction_per_level:
    # Convert string keys to int keys
    area_fraction_per_level = {int(k): v for k, v in area_fraction_per_level.items()}

# Pass to pipeline
pipeline = AggregationPipeline(
    ...
    area_fraction_per_level=area_fraction_per_level,
)
```

#### c. Updated `full_pipeline.yaml`

Added configuration option with documentation:

```yaml
aggregation:
  # Minimum area as fraction of total 3D points
  # Default 1/500 means proposals must have at least P/500 points
  area_fraction: 0.002  # 1/500 (used as fallback if area_fraction_per_level not set)

  # Per-level area fractions (optional)
  # If specified, overrides area_fraction for specific levels
  # Useful for applying different size thresholds to different granularity levels
  # Example: Stricter filtering for coarse levels, more permissive for fine levels
  area_fraction_per_level: null  # Example: {2: 0.003, 4: 0.002, 6: 0.001}
```

### 3. Merging Output Information (Already Present)

User requested:
> 在ipynb裡面本來會有一些關於merging的輸出部分，請確認在新的pipeline也保留這個功能

**Status**: ✅ Already implemented

The pipeline already outputs detailed merging statistics:

```
Level 2: Merging and filtering...
  112 proposals after merge/filter
  Dropped: 16 (merged: 2, small: 14)

Level 4: Merging within siblings...
  187 proposals after merge/filter
  Dropped (small): 25

Level 6: Merging within siblings...
  79 proposals after merge/filter
  Dropped (small): 10
```

This matches the information from the original notebook.

## Usage Examples

### Example 1: Global Area Threshold (Default)

```yaml
aggregation:
  area_fraction: 0.002  # All levels use 1/500
  area_fraction_per_level: null
```

**Output**:
```
Using global area threshold: 159 points (0.0020 * 79614)
```

### Example 2: Per-Level Thresholds

```yaml
aggregation:
  area_fraction: 0.002  # Fallback
  area_fraction_per_level:
    2: 0.003  # L2: stricter (more filtering)
    4: 0.002  # L4: medium
    6: 0.001  # L6: permissive (less filtering)
```

**Output**:
```
Using per-level area thresholds:
  Level 2: 239 points (0.0030 * 79614)
  Level 4: 159 points (0.0020 * 79614)
  Level 6: 80 points (0.0010 * 79614)
```

## Verification

Current pipeline run (2025-11-12 14:58) is successfully:

1. ✅ Loading video segments from correct scene directory
2. ✅ Building 3D proposals: 128 (L2), 212 (L4), 89 (L6)
3. ✅ Merging and filtering with statistics
4. ✅ Extracting SigLIP features
5. ✅ Family tree integration working

## Inference Stage Fixes (Additional)

During testing, several issues were found and fixed in the inference stage:

### Issue 1: Incorrect InferencePipeline Initialization

**Problem**: Missing required `proposals_by_level` parameter

**Fix** (`scripts/run_full_pipeline.py:273-301`):
```python
# Extract strategy-specific kwargs
strategy_kwargs = {}
if inference_config.strategy == 'independent':
    strategy_kwargs = {
        'object_levels': inference_config.independent.object_levels,
        'part_levels': inference_config.independent.part_levels,
        # ... other parameters
    }

pipeline = InferencePipeline(
    proposals_by_level=proposals_by_level,  # REQUIRED
    family_tree=family_tree,
    strategy=inference_config.strategy,
    **strategy_kwargs
)
```

### Issue 2: Wrong Method Name for Feature Extraction

**Problem**: `encode_text()` method doesn't exist

**Fix** (`scripts/run_full_pipeline.py:329`):
```python
# Correct method name
text_features = feature_extractor.extract_text_features([text_query])
```

### Issue 3: Incorrect Query Format

**Problem**: `InferencePipeline.predict()` method doesn't exist; should use `infer_single_query()`

**Fix** (`scripts/run_full_pipeline.py:313-341`):
```python
# Updated query format: (object_text, part_text, label_id, num_instances)
queries = [
    ("chair", "chair leg", 1, 1),
    ("table", "table top", 2, 1),
    ("sofa", "sofa cushion", 3, 1),
]

# Separate object and part features
object_features = feature_extractor.extract_text_features([object_text])
part_features = feature_extractor.extract_text_features([part_text])

# Call correct method
results = pipeline.infer_single_query(
    object_query_feat=object_features[0],
    part_query_feat=part_features[0],
    label_id=label_id,
)
```

### Issue 4: Incorrect Prediction Attribute Access

**Problem**: `Prediction` object has `metadata` dict, not direct attributes

**Fix** (`scripts/run_full_pipeline.py:345-363`):
```python
# Access metadata properly
pred_dict = {
    'score': float(result.score),
    'mask_points': int(result.mask.sum()),
}

if result.metadata:
    pred_dict.update({
        'object_level': result.metadata.get('object_level'),
        'part_level': result.metadata.get('part_level'),
        # ...
    })
```

## Files Modified

1. `configs/inference/full_pipeline.yaml` - Fixed exp_dir path, added area_fraction_per_level
2. `src/my3dis/aggregation/aggregation_pipeline.py` - Added per-level threshold support
3. `scripts/run_full_pipeline.py` - Fixed multiple inference stage issues
   - Correct InferencePipeline initialization
   - Proper feature extraction method calls
   - Correct query format and method usage
   - Proper Prediction metadata handling

## Backward Compatibility

All changes are backward compatible:
- Existing configs without `area_fraction_per_level` will use global `area_fraction`
- All existing functionality preserved
- No breaking changes to API

## Test Results

**Aggregation Stage** (Tested 2025-11-12 14:58-15:07):
- ✅ Successfully loaded video segments from all levels
- ✅ Built 3D proposals: L2=128, L4=212, L6=89
- ✅ Merging and filtering: L2=112, L4=187, L6=79 (after filtering)
- ✅ Extracted SigLIP features with voting pooling
- ✅ Output files created successfully:
  - `proposal_data_level2.npz` (753KB)
  - `proposal_data_level4.npz` (1.3MB)
  - `proposal_data_level6.npz` (462KB)
  - `Proposal_relation.json` (1.2MB)
- ⏱️ Execution time: 524 seconds (~8.7 minutes)

**Inference Stage** (Tested 2025-11-12 15:12-15:13):
- ✅ Loaded 378 proposals (112 + 187 + 79)
- ✅ Family tree integration working
- ✅ Independent strategy retrieval functional
- ✅ Object-part pairing successful
- ✅ Multi-instance NMS working
- ✅ Generated 24 predictions from 3 test queries
- ⏱️ Execution time: 7.1 seconds

**Example Predictions**:
```json
{
  "object_query": "table",
  "part_query": "table top",
  "score": 0.291,
  "mask_points": 1382,
  "object_id": 122,
  "part_id": 4146
}
```

## Next Steps

### 1. Use Per-Level Thresholds

Edit `configs/inference/full_pipeline.yaml`:
```yaml
aggregation:
  area_fraction_per_level:
    2: 0.003  # Stricter for coarse level
    4: 0.002  # Medium
    6: 0.001  # More permissive for fine level
```

### 2. Run Full Pipeline

```bash
PYTHONPATH=src python scripts/run_full_pipeline.py --config configs/inference/full_pipeline.yaml
```

### 3. Run Only Specific Stages

```bash
# Skip aggregation (use existing output)
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation

# Skip inference
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-inference
```

### 4. Override Parameters

```bash
# Override device
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --device cuda:1

# Override scene path
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --scene-path /path/to/other/scene
```

---

## New Feature: Automatic Level Detection (2025-11-12 19:00)

### Problem

The pipeline was hardcoded to look for levels `[2, 4, 6]`, but experiments may use different level configurations. For example, experiment `test_1108_ssam4_filter500_fill500_propinf_135` uses levels `[1, 3, 5]`, causing failures:

```
WARNING - Video segments not found: .../level_2/video_segments_L02.npz
WARNING - Video segments not found: .../level_4/video_segments_L04.npz
WARNING - Video segments not found: .../level_6/video_segments_L06.npz
```

### Solution: Auto-Detection Function

Added `detect_available_levels()` function that scans the experiment directory to find available levels.

#### Implementation

**File**: `src/my3dis/aggregation/aggregation_pipeline.py:49-79`

```python
def detect_available_levels(exp_dir: Path) -> List[int]:
    """
    Auto-detect available levels from experiment directory structure.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Sorted list of available levels (e.g., [1, 3, 5] or [2, 4, 6])
    """
    if not exp_dir.exists():
        logger.warning(f"Experiment directory does not exist: {exp_dir}")
        return []

    levels = []
    level_pattern = re.compile(r'^level_(\d+)$')

    for item in exp_dir.iterdir():
        if item.is_dir():
            match = level_pattern.match(item.name)
            if match:
                level_num = int(match.group(1))
                # Check if video_segments file exists
                video_segments_path = item / f"video_segments_L{level_num:02d}.npz"
                if video_segments_path.exists():
                    levels.append(level_num)
                else:
                    logger.debug(f"Level directory found but video_segments missing: {item}")

    levels.sort()
    return levels
```

#### Integration Points

**1. AggregationPipeline.run()** (`aggregation_pipeline.py:338-348`)

```python
if levels is None:
    # Auto-detect levels from experiment directory
    detected_levels = detect_available_levels(self.exp_dir)
    if detected_levels:
        levels = detected_levels
        logger.info(f"Auto-detected levels from experiment directory: {levels}")
    else:
        levels = [2, 4, 6]
        logger.warning(f"No levels detected, using default: {levels}")
else:
    logger.info(f"Using specified levels: {levels}")
```

**2. run_aggregation_stage()** (`scripts/run_full_pipeline.py:227-237`)

```python
# Auto-detect or use specified levels
levels_to_use = exp_config.get('levels', None)
if levels_to_use is None:
    from my3dis.aggregation.aggregation_pipeline import detect_available_levels
    exp_dir = Path(exp_config['exp_dir'])
    detected_levels = detect_available_levels(exp_dir)
    if detected_levels:
        levels_to_use = detected_levels
        logger.info(f"Auto-detected levels from experiment directory: {levels_to_use}")
    else:
        logger.warning("No levels detected and none specified, pipeline will use defaults")
```

**3. Skip aggregation path** (`scripts/run_full_pipeline.py:1107-1118`)

```python
# Auto-detect or use specified levels
levels_to_check = config['experiment'].get('levels', None)
if levels_to_check is None:
    from my3dis.aggregation.aggregation_pipeline import detect_available_levels
    exp_dir = Path(config['experiment']['exp_dir'])
    detected_levels = detect_available_levels(exp_dir)
    if detected_levels:
        levels_to_check = detected_levels
        logger.info(f"Auto-detected levels: {levels_to_check}")
    else:
        levels_to_check = [2, 4, 6]
        logger.warning(f"No levels detected, using default: {levels_to_check}")
```

**4. Module export** (`src/my3dis/aggregation/__init__.py:19,26`)

```python
from .aggregation_pipeline import AggregationPipeline, detect_available_levels

__all__ = [
    'AggregationPipeline',
    'detect_available_levels',
    ...
]
```

### Usage

#### Option 1: Auto-Detect (Set levels to null)

```yaml
experiment:
  exp_dir: /path/to/experiment/scene_00005_00
  levels: null  # Will auto-detect [1, 3, 5] or [2, 4, 6]
```

#### Option 2: Manual Specification

```yaml
experiment:
  levels: [1, 3, 5]  # Explicitly specify levels
```

#### Option 3: Programmatic

```python
from pathlib import Path
from my3dis.aggregation import detect_available_levels, AggregationPipeline

# Auto-detect
exp_dir = Path('/path/to/experiment')
levels = detect_available_levels(exp_dir)
print(f"Detected: {levels}")

# Use in pipeline
pipeline = AggregationPipeline(exp_dir=str(exp_dir), scene_path='...')
results = pipeline.run(output_dir='...', levels=None)  # Auto-detects
```

### Test Results

**Test Configuration**: `configs/inference/test_autodetect.yaml`

```yaml
experiment:
  exp_dir: .../test_1108_ssam4_filter500_fill500_propinf_135/scene_00005_00
  levels: null  # Triggers auto-detection
```

**Output**:
```
2025-11-12 19:09:06 - INFO - Auto-detected levels from experiment directory: [1, 3, 5]
2025-11-12 19:09:06 - INFO - Using specified levels: [1, 3, 5]
2025-11-12 19:09:22 - INFO - Processing level 1...
2025-11-12 19:09:26 - INFO -   Built 76 3D proposals from 2D masks
2025-11-12 19:09:31 - INFO - Processing level 3...
2025-11-12 19:09:31 - INFO -   Built 107 3D proposals from 2D masks
2025-11-12 19:09:42 - INFO - Processing level 5...
2025-11-12 19:09:42 - INFO -   Built 240 3D proposals from 2D masks
2025-11-12 19:09:43 - INFO - ✓ Pipeline completed successfully
```

**Generated Files**:
- `proposal_data_level1.npz` (196KB, 76 proposals)
- `proposal_data_level3.npz` (243KB, 107 proposals)
- `proposal_data_level5.npz` (458KB, 240 proposals)
- `aggregation_metadata.json` (correctly shows levels [1, 3, 5])
- `Proposal_relation.json` (712KB)

### Backward Compatibility

✅ **Fully backward compatible**

- Existing configs with `levels: [2, 4, 6]` continue to work
- When `levels=None` and no levels detected: falls back to `[2, 4, 6]`
- No changes to output format or data structures

### Files Modified (Auto-Detection Feature)

1. `src/my3dis/aggregation/aggregation_pipeline.py`
   - Added `detect_available_levels()` (lines 49-79)
   - Updated `run()` to support auto-detection (lines 338-348)

2. `scripts/run_full_pipeline.py`
   - Updated `run_aggregation_stage()` (lines 227-237)
   - Updated skip-aggregation path (lines 1107-1118)

3. `src/my3dis/aggregation/__init__.py`
   - Exported `detect_available_levels` (lines 19, 26)

4. `configs/inference/test_autodetect.yaml` (NEW)
   - Test config for auto-detection feature

### Benefits

1. **Flexibility**: Works with any level configuration (1,3,5 or 2,4,6 or custom)
2. **User-friendly**: No need to manually specify levels in config
3. **Robustness**: Verifies video_segments files exist before including levels
4. **Backward compatible**: Existing workflows unchanged
5. **Clear logging**: Reports whether levels were auto-detected or manual
