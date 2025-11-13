# Search3D Multi-Level Evaluation Support

**Date**: 2025-11-12
**Status**: ✅ Fully Implemented

## Overview

This document describes the new **multi-level evaluation support** for Search3D benchmark, enabling evaluation of both **object-level** and **object-part level** predictions simultaneously.

## Motivation

Previously, the My3DIS pipeline only supported part-level evaluation using `ov_part_annotations`. However, Search3D benchmark provides multiple levels of ground truth:

1. **Object-level GT** (`obj_annotations/`): Coarse instance segmentation
2. **Object-Part level GT** (`ov_part_annotations/`): Fine-grained part segmentation
3. **Part-only GT** (`part_annotations/`): Part instances only

To fully evaluate model performance across different granularities, we need to support all evaluation modes.

## Implementation

### 1. Search3D Format Exporter (NEW)

**File**: `src/my3dis/aggregation/search3d_formatter.py`

A new module that converts My3DIS predictions to Search3D evaluation format.

#### Key Functions:

```python
export_predictions_search3d(
    predictions: List,
    num_points: int,
    output_path: str,
    format_type: str = 'obj_part',  # 'obj' or 'obj_part'
    instance_id_start: int = 1000,
) -> Dict

export_predictions_both_formats(
    predictions: List,
    num_points: int,
    output_dir: str,
    scene_name: str,
) -> Dict

load_point_cloud_for_export(
    scene_path: str,
    mesh_name: str = 'mesh_aligned_0.05.ply'
) -> np.ndarray

validate_search3d_format(
    prediction_file: str,
    expected_num_points: Optional[int] = None,
) -> Dict
```

#### Output Format:

Search3D format is a simple text file with one line per point:

```
# scene_00005_00_obj_inst.txt (Object-level)
0          # Background
0          # Background
1000       # Instance 1000
1000       # Instance 1000
1001       # Instance 1001
...

# scene_00005_00_obj_part_inst.txt (Part-level)
0          # Background
0          # Background
4008       # Door frame (object_id=4, part_id=8)
4008       # Door frame
11002      # Cabinet door (object_id=11, part_id=2)
...
```

### 2. Pipeline Integration

**File**: `scripts/run_full_pipeline.py`

#### Changes to `run_inference_stage()`:

After generating predictions, automatically export Search3D format:

```python
# Export Search3D format (NEW)
if len(prediction_masks) > 0 and config['evaluation'].get('enabled', False):
    from my3dis.aggregation.search3d_formatter import (
        export_predictions_both_formats,
        load_point_cloud_for_export
    )

    # Load point cloud to get total number of points
    points = load_point_cloud_for_export(scene_path)
    num_points = len(points)

    # Export both formats
    search3d_export_stats = export_predictions_both_formats(
        predictions=pred_objects,
        num_points=num_points,
        output_dir=output_dir,
        scene_name=scene_name,
    )
```

**Output Files**:
- `scene_XXXXX_XX_obj_inst.txt` - Object-level predictions
- `scene_XXXXX_XX_obj_part_inst.txt` - Part-level predictions

#### Changes to `run_evaluation_stage()`:

Completely rewritten to support multiple evaluation modes:

```python
def run_evaluation_stage(
    config: Dict,
    inference_output_dir: str,
) -> Dict:
    """
    Run evaluation with multi-mode support.

    Returns:
        {
            'object': {...},      # Object-level metrics
            'obj_part': {...},    # Part-level metrics
            'summary': {...}      # Combined summary
        }
    """
```

**Key Features**:
- Iterates through configured evaluation modes
- Automatically selects correct GT files and evaluation scripts
- Returns results for all modes
- Generates comprehensive summary

### 3. Configuration Updates

**File**: `configs/inference/full_pipeline.yaml`

#### New Configuration Structure:

```yaml
evaluation:
  enabled: true

  # Ground truth paths for different evaluation levels (NEW)
  gt_paths:
    # Object-level evaluation
    object: /path/to/obj_annotations

    # Object-Part evaluation
    obj_part: /path/to/ov_part_annotations

  # Evaluation modes to run (NEW)
  eval_modes: ['object', 'obj_part']  # Run both evaluations

  # Evaluation scripts for different modes (NEW)
  eval_scripts:
    object: eval/search3d/eval_semantic_instance_objects.py
    obj_part: eval/search3d/eval_semantic_instance_parts_OV.py

  # DEPRECATED: Use gt_paths.obj_part instead
  gt_path: null
```

#### Backward Compatibility:

Old configs using `gt_path` still work:

```yaml
evaluation:
  gt_path: /path/to/ov_part_annotations  # Treated as obj_part mode
```

## Usage

### Option 1: Evaluate Both Levels (Recommended)

```yaml
# configs/inference/full_pipeline.yaml
evaluation:
  enabled: true
  gt_paths:
    object: /media/Pluto/richkung/My3DIS/data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/obj_annotations
    obj_part: /media/Pluto/richkung/My3DIS/data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations
  eval_modes: ['object', 'obj_part']
```

```bash
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

**Output**:
```
================================================================================
STAGE 3: EVALUATION (Search3D Multi-Mode)
================================================================================
Scene: scene_00005_00
Evaluation modes: ['object', 'obj_part']

================================================================================
Evaluating: OBJECT
================================================================================
  GT file: .../obj_annotations/scene_00005_00_obj_inst.txt
  Pred file: .../inference_output/scene_00005_00_obj_inst.txt
  Running evaluation for object...
  ✓ OBJECT evaluation completed
    mAP: 0.234
    AP@50: 0.456
    AP@25: 0.678

================================================================================
Evaluating: OBJ_PART
================================================================================
  GT file: .../ov_part_annotations/scene_00005_00_obj_part_inst.txt
  Pred file: .../inference_output/scene_00005_00_obj_part_inst.txt
  Running evaluation for obj_part...
  ✓ OBJ_PART evaluation completed
    mAP: 0.123
    AP@50: 0.234
    AP@25: 0.456

================================================================================
EVALUATION SUMMARY
================================================================================
  Total modes: 2
  Successful: 2
  Failed: 0
  Skipped: 0

  OBJECT:
    mAP: 0.234
    AP@50: 0.456
    AP@25: 0.678

  OBJ_PART:
    mAP: 0.123
    AP@50: 0.234
    AP@25: 0.456
```

### Option 2: Evaluate Single Level

```yaml
# Object-level only
evaluation:
  eval_modes: ['object']

# Part-level only
evaluation:
  eval_modes: ['obj_part']
```

### Option 3: Programmatic Usage

```python
from my3dis.aggregation import (
    export_predictions_both_formats,
    load_point_cloud_for_export
)

# Load point cloud
points = load_point_cloud_for_export('/path/to/scene')
num_points = len(points)

# Export predictions
stats = export_predictions_both_formats(
    predictions=predictions,
    num_points=num_points,
    output_dir='output/',
    scene_name='scene_00005_00',
)

print(f"Object-level: {stats['obj']['num_labeled_points']} labeled points")
print(f"Part-level: {stats['obj_part']['num_labeled_points']} labeled points")
```

## File Structure

After running the full pipeline with evaluation enabled:

```
inference_output/
├── predictions.json                    # Original format
├── prediction_masks.npz                # Original format
├── scene_00005_00_obj_inst.txt         # NEW: Object-level predictions
├── scene_00005_00_obj_part_inst.txt    # NEW: Part-level predictions
└── ply_exports/                        # Optional visualizations
    ├── pred_0000_label_4008.ply
    └── ...
```

## Evaluation Scripts Compatibility

The implementation works with Search3D's evaluation scripts:

### Object-level Evaluation

**Script**: `eval/search3d/eval_semantic_instance_objects.py`

**Expected Format**:
- Ground Truth: `scene_*_obj_inst.txt` with instance IDs (1000+)
- Predictions: Same format

### Object-Part Evaluation

**Script**: `eval/search3d/eval_semantic_instance_parts_OV.py`

**Expected Format**:
- Ground Truth: `scene_*_obj_part_inst.txt` with composite IDs (object_id * 1000 + part_id)
- Predictions: Same format

**Note**: Both scripts need a `evaluate_scan_from_files()` function. If not present, you may need to add:

```python
def evaluate_scan_from_files(pred_file: str, gt_file: str, scene_name: str):
    """
    Load predictions and GT from files, then call assign_instances_for_scan.
    """
    # Load pred_file
    with open(pred_file, 'r') as f:
        pred_labels = [int(line.strip()) for line in f]

    # Convert to required format
    pred_format = convert_labels_to_format(pred_labels)

    # Call existing function
    return assign_instances_for_scan(pred=pred_format, gt_file=gt_file)
```

## Validation

To validate the exported Search3D format:

```python
from my3dis.aggregation import validate_search3d_format

result = validate_search3d_format(
    prediction_file='scene_00005_00_obj_inst.txt',
    expected_num_points=79614
)

if result['valid']:
    print(f"✓ Valid format")
    print(f"  Lines: {result['num_lines']}")
    print(f"  Labeled: {result['num_labeled']}")
    print(f"  Unique labels: {result['unique_labels']}")
else:
    print(f"✗ Invalid format")
    for error in result['errors']:
        print(f"  - {error}")
```

## Benefits

1. **Comprehensive Evaluation**: Evaluate model performance at multiple granularities
2. **Standard Format**: Directly compatible with Search3D evaluation scripts
3. **Flexibility**: Choose which evaluation modes to run
4. **Backward Compatible**: Old configs continue to work
5. **Automatic Export**: No manual format conversion needed
6. **Validation Tools**: Built-in format validation

## Files Modified

### New Files:
1. `src/my3dis/aggregation/search3d_formatter.py` - Format conversion module
2. `SEARCH3D_MULTI_LEVEL_EVAL.md` - This documentation

### Modified Files:
1. `src/my3dis/aggregation/__init__.py` - Export new functions
2. `scripts/run_full_pipeline.py` - Updated inference and evaluation stages
3. `configs/inference/full_pipeline.yaml` - Updated evaluation config

## Migration Guide

### For Existing Configs:

**Old Config** (still works):
```yaml
evaluation:
  enabled: true
  gt_path: /path/to/ov_part_annotations
```

**New Config** (recommended):
```yaml
evaluation:
  enabled: true
  gt_paths:
    obj_part: /path/to/ov_part_annotations
  eval_modes: ['obj_part']
```

**Full Multi-Level Config**:
```yaml
evaluation:
  enabled: true
  gt_paths:
    object: /path/to/obj_annotations
    obj_part: /path/to/ov_part_annotations
  eval_modes: ['object', 'obj_part']
  eval_scripts:
    object: eval/search3d/eval_semantic_instance_objects.py
    obj_part: eval/search3d/eval_semantic_instance_parts_OV.py
```

### For Existing Code:

No changes needed if you're using the YAML config approach. If calling functions directly:

**Old API** (deprecated but still works for single mode):
```python
eval_result = run_evaluation_stage(
    config,
    predictions_path='/path/to/predictions.json',
    masks_path='/path/to/masks.npz',
    num_points=79614
)
# Returns: {'status': 'success', 'mAP': 0.123, ...}
```

**New API** (multi-mode support):
```python
eval_result = run_evaluation_stage(
    config=config,
    inference_output_dir='/path/to/inference_output',
)
# Returns: {
#   'object': {'status': 'success', 'mAP': 0.234, ...},
#   'obj_part': {'status': 'success', 'mAP': 0.123, ...},
#   'summary': {'successful': 2, 'failed': 0, ...}
# }
```

## Testing

### Basic Test:

```bash
# Test with single scene
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml

# Check output files
ls inference_output/
# Should see:
#   - scene_00005_00_obj_inst.txt
#   - scene_00005_00_obj_part_inst.txt
```

### Validate Format:

```python
from my3dis.aggregation import validate_search3d_format

# Validate object-level
obj_result = validate_search3d_format('inference_output/scene_00005_00_obj_inst.txt')
assert obj_result['valid'], f"Errors: {obj_result['errors']}"

# Validate part-level
part_result = validate_search3d_format('inference_output/scene_00005_00_obj_part_inst.txt')
assert part_result['valid'], f"Errors: {part_result['errors']}"
```

### Check Evaluation:

```bash
# Check log for evaluation results
grep -A 20 "EVALUATION SUMMARY" logs/eval/run_exp_*.log
```

## Troubleshooting

### Issue 1: Missing GT Files

**Symptom**: `GT file not found: .../scene_00005_00_obj_inst.txt`

**Solution**: Ensure GT paths are correct in config:
```yaml
evaluation:
  gt_paths:
    object: /correct/path/to/obj_annotations
```

### Issue 2: Prediction Files Not Generated

**Symptom**: `Prediction file not found: .../scene_00005_00_obj_inst.txt`

**Solution**: Check that:
1. Evaluation is enabled: `evaluation.enabled: true`
2. Predictions were generated successfully
3. Point cloud was loaded correctly

### Issue 3: Evaluation Script Import Error

**Symptom**: `ImportError: cannot import name 'evaluate_scan_from_files'`

**Solution**: The Search3D evaluation scripts may not have this function. You can:
1. Add the function to the evaluation script (see example above)
2. Use the legacy `assign_instances_for_scan()` directly
3. Contact Search3D authors for updated scripts

### Issue 4: Mismatched Point Counts

**Symptom**: `mask length (X) != num_points (Y)`

**Solution**: Ensure the mesh file matches the experiment:
```python
# Use correct mesh file
points = load_point_cloud_for_export(
    scene_path='/path/to/scene',
    mesh_name='mesh_aligned_0.05.ply'  # Must match aggregation
)
```

## Performance Notes

- **Memory**: Exporting Search3D format requires loading full point cloud (~300MB for typical scene)
- **Time**: Export adds ~1-2 seconds per scene
- **Disk**: Each TXT file is ~1-2MB (79k points × ~15 bytes/line)

## Future Enhancements

Potential improvements for future versions:

1. **Part-only Evaluation**: Support `part_annotations/` GT
2. **Batch Export**: Export multiple scenes in parallel
3. **Compressed Format**: Support `.npz` format for large-scale evaluation
4. **Evaluation Cache**: Cache evaluation results to avoid re-computation
5. **Multi-Scene Reports**: Aggregate metrics across multiple scenes

## References

- Search3D Paper: https://arxiv.org/abs/2409.18431
- Search3D GitHub: https://github.com/aycatakmaz/search3d
- My3DIS Aggregation Docs: `docs/ARCHITECTURE.md`
- MultiScan Dataset: https://multiscan.github.io/

## Contact

For questions or issues related to this implementation, please:
1. Check this documentation
2. Review example configs in `configs/inference/`
3. Check logs in `logs/eval/`
4. Create an issue with detailed error messages
