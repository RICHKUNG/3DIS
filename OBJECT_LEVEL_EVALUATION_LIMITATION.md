# Object-Level Evaluation Limitation

**Date**: 2025-11-12
**Status**: ⚠️ Known Limitation

## Problem Summary

The current My3DIS inference pipeline **cannot properly perform Search3D object-level evaluation** due to architectural constraints.

## Root Cause

### Search3D Object-Level Evaluation Requirements

Based on `eval/search3d/eval_semantic_instance_objects.py`:

```python
# Line 21: Uses object class labels
from multiscan_search3d_constants import CLASS_LABELS, VALID_CLASS_IDS

# CLASS_LABELS = ('door', 'cabinet', 'toilet', ...)
# VALID_CLASS_IDS = (4, 11, 32, ...)

# Line 273-277: Requires object class ID for label matching
label_id = int(pred_info[uuid]['label_id'])  # Must be object class ID (4, 11, 32)
label_name = ID_TO_LABEL[label_id]  # Maps to 'door', 'cabinet', 'toilet'

# Line 301: Compares prediction mask to GT instance
intersection = np.count_nonzero(
    np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask)
)
```

**Requirements:**
1. **Prediction `label_id`**: Must be object class ID (e.g., 4 for door, 11 for cabinet)
2. **Prediction mask**: Should cover the entire object, not just a part
3. **GT format**: `obj_inst.txt` with instance IDs (1006, 3001, ...) that map to object classes

### My3DIS Inference Pipeline Limitations

**Current Architecture:**

1. **Query Format**: (object_text, part_text, label_id, num_instances)
   - Example: ("door", "frame", 4008, 1)
   - `label_id = 4008` is a **composite ID** (door=4, frame=8), not a pure object ID

2. **Prediction Masks**: Controlled by `formatter.use_part_mask` config
   ```yaml
   formatter:
     use_part_mask: true  # Uses part mask, not object mask
   ```
   - When `true`: `pred.mask = part_proposal.mask` (only part points)
   - When `false`: `pred.mask = object_mask ∪ part_mask` (union)

3. **Search3D Format Export**:
   ```python
   # In search3d_formatter.py
   labels[mask] = label_id  # Using composite ID (4008) and part mask
   ```

**Mismatch:**

| Requirement | My3DIS Output | Issue |
|-------------|---------------|-------|
| Object class ID (4, 11, 32) | Composite ID (4008, 11002, 32005) | ID format mismatch |
| Object mask (entire object) | Part mask (only part) | Mask coverage mismatch |
| GT: obj_inst.txt | Pred: uses ov_part format | Semantic mismatch |

## Current Workaround

The `export_predictions_search3d()` function with `format_type='obj'` currently:

1. **Ignores the composite label_id**
2. **Assigns sequential instance IDs** (1000, 1001, 1002, ...)
3. **Uses whatever mask is provided** (currently part masks)

This creates a valid Search3D file format, but:
- ❌ Evaluation will likely fail (class ID mismatch)
- ❌ Results will be meaningless (wrong masks)
- ❌ Cannot compare to object-level GT properly

## Example Comparison

### Ground Truth (`obj_inst.txt`):
```
0       # Background
0       # Background
1006    # Door instance 1006 (class: door=4)
1006    # Door instance 1006
3001    # Cabinet instance 3001 (class: cabinet=11)
3001    # Cabinet instance 3001
```

### Our Prediction (`scene_*_obj_inst.txt`):
```
0       # Background
0       # Background
1000    # Sequential ID 1000 (should be class 4)
1000    # Sequential ID 1000 (but uses part mask, not full door!)
1001    # Sequential ID 1001 (should be class 11)
1001    # Sequential ID 1001 (but uses part mask, not full cabinet!)
```

**Problems:**
1. Instance IDs don't carry class information
2. Masks only cover parts, not full objects
3. Evaluation script expects class-aware matching

## Solutions

### Option 1: Disable Object-Level Evaluation (Current)

Simply skip object-level evaluation and only run part-level:

```yaml
# configs/inference/full_pipeline.yaml
evaluation:
  enabled: true
  eval_modes: ['obj_part']  # Remove 'object'
  gt_paths:
    obj_part: /path/to/ov_part_annotations
```

**Pros:**
- No code changes needed
- Part-level evaluation works correctly

**Cons:**
- Cannot evaluate object-level performance

### Option 2: Extract Object Information from Metadata

Modify `run_full_pipeline.py` to extract object masks and IDs:

```python
# In run_full_pipeline.py, after inference
for pred_dict, result in zip(predictions, results):
    # Extract object class ID from composite ID
    composite_id = pred_dict['label_id']  # e.g., 4008
    object_class_id = composite_id // 1000  # 4008 // 1000 = 4

    # Get object mask from metadata
    if result.metadata and 'object_proposal' in result.metadata:
        object_mask = result.metadata['object_proposal'].mask
    else:
        # Fallback: use part mask (not ideal)
        object_mask = result.mask

    # Store separately for object-level export
    object_predictions.append({
        'mask': object_mask,
        'label_id': object_class_id,  # Pure object ID
        'score': pred_dict['score'],
    })
```

**Pros:**
- Enables proper object-level evaluation
- Uses correct masks and IDs

**Cons:**
- Requires pipeline changes
- Metadata may not always contain object_proposal

### Option 3: Implement Pure Object Queries

Add support for object-only queries (without parts):

```python
# New query format
object_only_queries = [
    ("door", None, 4, 1),      # Query object only
    ("cabinet", None, 11, 1),  # Query object only
]

# Inference pipeline checks for None part
if part_text is None:
    # Run object-level retrieval only
    results = pipeline.infer_object_only(
        object_query_feat=object_feat,
        label_id=label_id,
    )
```

**Pros:**
- Cleanest solution
- Separates object and part-level tasks

**Cons:**
- Requires significant pipeline refactoring
- Need to implement object-only inference logic

### Option 4: Post-Process Composite IDs

Extract object class from composite IDs during export:

```python
# In search3d_formatter.py
if format_type == 'obj':
    for idx, pred in enumerate(predictions):
        # Extract object class ID
        if hasattr(pred, 'label_id'):
            composite_id = pred.label_id
            object_class_id = composite_id // 1000
        else:
            # Fallback to sequential
            object_class_id = instance_id_start + idx

        # Use object_class_id instead of sequential ID
        labels[mask] = object_class_id
```

**Pros:**
- Minimal changes
- Preserves class information

**Cons:**
- Still uses part masks (not object masks)
- Only partially solves the problem

## Recommended Approach

For now: **Option 1 (Disable Object-Level Evaluation)**

Reasons:
1. Part-level evaluation is the primary focus of My3DIS
2. Quick fix with no code changes
3. Avoids incorrect/misleading metrics

For future: **Option 2 (Extract from Metadata)** or **Option 3 (Pure Object Queries)**

Reasons:
1. Proper solution for object-level evaluation
2. Maintains architectural integrity
3. Allows both object and part-level evaluation

## Current Configuration

To disable object-level evaluation:

```yaml
# configs/inference/full_pipeline.yaml
evaluation:
  enabled: true

  gt_paths:
    # object: /path/to/obj_annotations  # Comment out or remove
    obj_part: /path/to/ov_part_annotations

  eval_modes: ['obj_part']  # Only run part-level

  eval_scripts:
    # object: eval/search3d/eval_semantic_instance_objects.py  # Not used
    obj_part: eval/search3d/eval_semantic_instance_parts_OV.py
```

## Testing

To verify part-level evaluation works correctly:

```bash
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml

# Check output
ls inference_output/
# Should see:
#   - scene_*_obj_part_inst.txt (part-level predictions)
#   - NO scene_*_obj_inst.txt (object-level disabled)
```

## Future Work

- [ ] Implement Option 2: Extract object masks from metadata
- [ ] Add pure object queries support (Option 3)
- [ ] Test object-level evaluation after fixes
- [ ] Update documentation with verified workflow
- [ ] Add integration tests for both evaluation modes

## Related Files

- `src/my3dis/aggregation/search3d_formatter.py` - Export logic
- `scripts/run_full_pipeline.py` - Inference and evaluation orchestration
- `eval/search3d/eval_semantic_instance_objects.py` - Object-level evaluation script
- `eval/search3d/eval_semantic_instance_parts_OV.py` - Part-level evaluation script
- `configs/inference/full_pipeline.yaml` - Pipeline configuration

## References

- Search3D Paper: https://arxiv.org/abs/2409.18431
- MultiScan Dataset: https://multiscan.github.io/
- Search3D Constants: `eval/search3d/data/multiscan_search3d_constants.py`
