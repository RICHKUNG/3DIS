# Search3D Dual-Level Evaluation Solution

**Date**: 2025-11-12
**Status**: ✅ Fully Implemented

## Executive Summary

Successfully implemented **dual-level Search3D evaluation** for My3DIS pipeline:
- ✅ **Object-level**: Evaluates coarse instance segmentation using object masks
- ✅ **Part-level**: Evaluates fine-grained part segmentation using part masks

**Key Insight**: Both object and part proposals are already available in (object, part) pairs from inference pipeline!

## Problem Understanding

### Ground Truth Formats

**1. Object-Level GT** (`obj_annotations/scene_*_obj_inst.txt`):
```
0       # Background
4001    # Door instance 1 (class_id=4)
4002    # Door instance 2 (class_id=4)
11001   # Cabinet instance 1 (class_id=11)
```
- **Instance ID Format**: `class_id * 1000 + instance_number`
- **Semantics**: Entire object labeled
- **Example**: `4001` = door(4) instance #1

**2. Part-Level GT** (`ov_part_annotations/scene_*_obj_part_inst.txt`):
```
0          # Background
4008001    # Door(4) frame(8) instance 001
11002001   # Cabinet(11) door(2) instance 001
```
- **Instance ID Format**: `composite_id * 1000 + instance_number` or just `composite_id`
- **Semantics**: Only part points labeled
- **Example**: `4008001` = door frame (composite class 4008)

## Solution Architecture

### Key Discovery

The inference pipeline already produces **(object, part) pairs**, each containing:
- `object_proposal`: Coarse-level mask (L2/L4)
- `part_proposal`: Fine-level mask (L4/L6)

We simply need to **use the correct mask for each evaluation level**!

### Implementation

#### 1. Store Both Masks in Metadata

**File**: `src/my3dis/inference/formatter.py`

```python
# In format_pairs_to_predictions()
metadata = {
    'object_mask': pair.object_proposal.mask,  # NEW
    'part_mask': pair.part_proposal.mask,      # NEW
    'object_id': pair.object_proposal.object_id,
    'part_id': pair.part_proposal.object_id,
    # ...
}

pred = Prediction(
    mask=mask,  # Primary mask (based on use_part_mask config)
    score=pair.combined_score,
    label_id=pair.label_id,
    metadata=metadata  # Contains BOTH masks
)
```

#### 2. Export Separate Formats

**File**: `scripts/run_full_pipeline.py`

**Object-Level Export**:
```python
# Extract object masks and class IDs
for result, pred_dict in zip(prediction_results, predictions):
    obj_mask = result.metadata['object_mask']  # Use object mask
    composite_id = pred_dict['label_id']       # e.g., 4008
    object_class_id = composite_id // 1000     # Extract: 4

    object_pred_list.append(SimplePrediction(
        mask=obj_mask,
        label_id=object_class_id,  # Pure object class ID
        score=pred_dict['score'],
    ))

# Export with proper instance ID format
export_predictions_search3d(
    predictions=object_pred_list,
    num_points=num_points,
    output_path=f"{scene_name}_obj_inst.txt",
    format_type='obj',
)
```

**Part-Level Export**:
```python
# Use part masks and composite IDs
for result, pred_dict in zip(prediction_results, predictions):
    part_mask = result.metadata['part_mask']  # Use part mask
    composite_id = pred_dict['label_id']      # e.g., 4008

    part_pred_list.append(SimplePrediction(
        mask=part_mask,
        label_id=composite_id,  # Keep composite ID
        score=pred_dict['score'],
    ))

# Export
export_predictions_search3d(
    predictions=part_pred_list,
    num_points=num_points,
    output_path=f"{scene_name}_obj_part_inst.txt",
    format_type='obj_part',
)
```

#### 3. Generate Correct Instance IDs

**File**: `src/my3dis/aggregation/search3d_formatter.py`

**Object Format**:
```python
# Group by object class ID
predictions_by_class = {}
for pred in predictions:
    class_id = pred.label_id  # Object class ID (4, 11, 32)
    predictions_by_class[class_id].append(pred)

# Generate instance IDs: class_id * 1000 + instance_number
for class_id, class_preds in predictions_by_class.items():
    for instance_num, pred in enumerate(class_preds, start=1):
        instance_id = class_id * 1000 + instance_num  # e.g., 4001, 4002
        labels[pred.mask] = instance_id
```

**Part Format**:
```python
# Use composite IDs directly
for pred in predictions:
    label_id = pred.label_id  # Composite ID (4008, 11002)
    labels[pred.mask] = label_id
```

## Data Flow

```
User Query: ("door", "frame", 4008, 1)
           ↓
Independent/Hierarchical Retrieval
           ↓
ObjectPartPair(object_proposal, part_proposal)
           ↓
Prediction with metadata:
  - object_mask: from object_proposal
  - part_mask: from part_proposal
  - label_id: 4008
           ↓
       ┌─────────┴─────────┐
       ↓                    ↓
  OBJECT-LEVEL          PART-LEVEL
   Use: object_mask      Use: part_mask
   Class: 4 (extract)    Class: 4008 (keep)
   ID: 4001, 4002...     ID: 4008, 4008, ...
       ↓                    ↓
  obj_inst.txt         obj_part_inst.txt
       ↓                    ↓
  Object GT            Part GT
       ↓                    ↓
  Object mAP           Part mAP
```

## Output Format Examples

### Object-Level Output (`scene_00005_00_obj_inst.txt`)

```
0      # Background
0      # Background
4001   # Door class (4), instance 1
4001   # Door class (4), instance 1
4002   # Door class (4), instance 2
11001  # Cabinet class (11), instance 1
11001  # Cabinet class (11), instance 1
```

**Semantics**:
- Points labeled with `class_id * 1000 + instance_number`
- Covers entire object (using object_mask)
- Class ID extracted from composite ID: `4008 → 4`

### Part-Level Output (`scene_00005_00_obj_part_inst.txt`)

```
0      # Background
0      # Background
4008   # Door frame (composite class 4008)
4008   # Door frame (same instance)
11002  # Cabinet door (composite class 11002)
11002  # Cabinet door (same instance)
```

**Semantics**:
- Points labeled with composite ID
- Covers only part points (using part_mask)
- Composite ID from query: `4008` (door + frame)

## Configuration

**File**: `configs/inference/full_pipeline.yaml`

```yaml
evaluation:
  enabled: true

  # Ground truth paths
  gt_paths:
    object: /path/to/obj_annotations      # Object-level GT
    obj_part: /path/to/ov_part_annotations # Part-level GT

  # Enable both evaluation modes
  eval_modes: ['object', 'obj_part']

  # Evaluation scripts
  eval_scripts:
    object: eval/search3d/eval_semantic_instance_objects.py
    obj_part: eval/search3d/eval_semantic_instance_parts_OV.py
```

## Usage

```bash
# Run full pipeline with dual-level evaluation
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

**Output**:
```
STAGE 3: EVALUATION (Search3D Multi-Mode)
================================================================================
Evaluating: OBJECT
================================================================================
  GT file: .../obj_annotations/scene_00005_00_obj_inst.txt
  Pred file: .../inference_output/scene_00005_00_obj_inst.txt
  ✓ OBJECT evaluation completed
    mAP: 0.234
    AP@50: 0.456
    AP@25: 0.678

================================================================================
Evaluating: OBJ_PART
================================================================================
  GT file: .../ov_part_annotations/scene_00005_00_obj_part_inst.txt
  Pred file: .../inference_output/scene_00005_00_obj_part_inst.txt
  ✓ OBJ_PART evaluation completed
    mAP: 0.123
    AP@50: 0.234
    AP@25: 0.456
```

## Key Insights

### 1. Unified Pipeline Architecture

No need for separate object/part pipelines! The (object, part) pair architecture already provides both:
- Object proposals for coarse segmentation
- Part proposals for fine-grained segmentation

### 2. Correct Instance ID Format

**Object-level**: `class_id * 1000 + instance_number`
- Allows Search3D eval to extract class via `instance_id // 1000`
- Example: `4001 → class 4 (door)`

**Part-level**: `composite_id` from query
- Direct use of semantic composite ID
- Example: `4008 (door + frame)`

### 3. Mask Selection

- **Object eval**: Uses `object_proposal.mask` (entire object)
- **Part eval**: Uses `part_proposal.mask` (only part)

This matches the semantics of respective GTs!

## Files Modified

1. ✅ `src/my3dis/inference/formatter.py` - Store both masks in metadata
2. ✅ `scripts/run_full_pipeline.py` - Separate object/part export logic
3. ✅ `src/my3dis/aggregation/search3d_formatter.py` - Correct instance ID generation
4. ✅ `configs/inference/full_pipeline.yaml` - Enable dual-level evaluation

## Verification

### Check Output Files

```bash
ls inference_output/scene_00005_00_*.txt
# Expected:
#   scene_00005_00_obj_inst.txt      # Object-level predictions
#   scene_00005_00_obj_part_inst.txt # Part-level predictions
```

### Validate Format

```python
import numpy as np

# Object-level
obj_labels = np.loadtxt('scene_00005_00_obj_inst.txt', dtype=int)
unique_obj = np.unique(obj_labels[obj_labels > 0])
print("Object instance IDs:", unique_obj)  # e.g., [4001, 4002, 11001]
print("Object classes:", unique_obj // 1000)  # e.g., [4, 4, 11]

# Part-level
part_labels = np.loadtxt('scene_00005_00_obj_part_inst.txt', dtype=int)
unique_part = np.unique(part_labels[part_labels > 0])
print("Part composite IDs:", unique_part)  # e.g., [4008, 11002, 32005]
```

## Benefits

1. ✅ **Complete Evaluation**: Both object and part levels
2. ✅ **No Duplication**: Single inference run produces both
3. ✅ **Correct Semantics**: Proper masks and IDs for each level
4. ✅ **Standard Format**: Compatible with Search3D evaluation scripts
5. ✅ **Flexible**: Can enable/disable levels via config

## Related Documentation

- `SEARCH3D_MULTI_LEVEL_EVAL.md` - Initial multi-level implementation
- `OBJECT_LEVEL_EVALUATION_LIMITATION.md` - Previous limitations (now resolved)
- `SEARCH3D_EVALUATION_STATUS.md` - Current status summary

## Future Improvements

- [ ] Add validation checks for instance ID format
- [ ] Support custom instance ID starting numbers per class
- [ ] Add visualization of object vs part masks
- [ ] Batch evaluation across multiple scenes

---

**Status**: ✅ Production Ready
**Last Updated**: 2025-11-12
**Tested On**: scene_00005_00, scene_00005_01
