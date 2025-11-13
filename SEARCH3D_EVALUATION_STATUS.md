# Search3D Evaluation Status

**Date**: 2025-11-12
**Last Updated**: 2025-11-12

## Summary

✅ **Part-Level Evaluation**: Fully implemented and working
⚠️ **Object-Level Evaluation**: Disabled due to architectural limitations

## Ground Truth Annotations

Search3D provides two levels of GT annotations in `/media/Pluto/richkung/My3DIS/data/search3d_gt/`:

### 1. Object-Level GT (`obj_annotations/`)

**File Format**: `scene_*_obj_inst.txt`

**Content**:
```
0       # Background
1006    # Object instance 1006 (class: door)
3001    # Object instance 3001 (class: cabinet)
```

**Semantics**:
- One line per point in the scene point cloud
- Instance IDs (≥1000) represent whole objects
- GT file contains class→instance mapping internally

**Usage**: For evaluating **coarse object instance segmentation**

**Example** (scene_00005_00):
- Total points: 79,614
- Labeled points: 46,173 (58%)
- Unique instances: 22 objects

### 2. Object-Part Level GT (`ov_part_annotations/`)

**File Format**: `scene_*_obj_part_inst.txt`

**Content**:
```
0          # Background
4002001    # door(4002) + frame(001) instance
11002001   # cabinet(11002) + door(001) instance
```

**Semantics**:
- Composite ID format: `object_id * 1000 + part_id`
- Only **part points** are labeled (not entire object)
- Represents "object-part" semantic units

**Usage**: For evaluating **fine-grained part segmentation**

**Example** (scene_00005_00):
- Total points: 79,614
- Labeled points: 10,566 (13.3%)
- Unique instances: 8 object-part combinations

## Current Implementation

### Part-Level Evaluation (✅ Working)

**Pipeline**:
```
Inference → (object, part) pairs
         ↓
Predictions: part masks + composite IDs
         ↓
Export: scene_*_obj_part_inst.txt
         ↓
Evaluation: Compare to ov_part_annotations
```

**Configuration**:
```yaml
# configs/inference/full_pipeline.yaml
evaluation:
  enabled: true
  eval_modes: ['obj_part']  # Part-level only
  gt_paths:
    obj_part: /path/to/ov_part_annotations
  eval_scripts:
    obj_part: eval/search3d/eval_semantic_instance_parts_OV.py
```

**Queries**:
```python
queries = [
    ("door", "frame", 4008, 1),         # door + frame
    ("cabinet", "door", 11002, 1),      # cabinet + door
    ("toilet", "lid", 32005, 1),        # toilet + lid
]
```

**Prediction Format**:
- `label_id`: Composite ID from query (e.g., 4008)
- `mask`: Part mask (from `formatter.use_part_mask: true`)

**Evaluation**:
```bash
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

**Output**:
```
STAGE 3: EVALUATION (Search3D Multi-Mode)
Scene: scene_00005_00
Evaluation modes: ['obj_part']

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
```

### Object-Level Evaluation (⚠️ Disabled)

**Problem**: Architectural mismatch

**Required**:
- Object class IDs (e.g., 4 for door, 11 for cabinet)
- Full object masks (entire object, not just parts)

**Current Output**:
- Composite IDs (e.g., 4008 for door+frame)
- Part masks (only part points)

**Consequence**:
- Class ID mismatch → evaluation fails
- Wrong masks → incorrect metrics

**Solution**: Disabled in config

**See**: `OBJECT_LEVEL_EVALUATION_LIMITATION.md` for details and future solutions

## File Structure

After running full pipeline:

```
inference_output/
├── predictions.json                    # Original format
├── prediction_masks.npz                # Original format
├── scene_00005_00_obj_inst.txt         # Generated but not used (disabled)
├── scene_00005_00_obj_part_inst.txt    # Used for part-level evaluation ✅
└── ply_exports/                        # Visualizations (optional)
```

## Evaluation Scripts

### Part-Level (`eval_semantic_instance_parts_OV.py`)

**Constants**:
```python
from multiscan_search3d_constants import (
    VALID_JOINT_TUPLE_NAMES,    # [('door', 'frame'), ...]
    VALID_JOINT_SEMANTIC_IDS    # [4008, 11002, ...]
)

CLASS_LABELS = ['door frame', 'cabinet door', ...]
VALID_CLASS_IDS = [4008, 11002, 32005, ...]
```

**Evaluation Logic**:
```python
# Load GT
gt_ids = util_3d.load_ids(gt_file)  # Per-point labels

# Match predictions
for pred in predictions:
    label_id = pred['label_id']  # Composite ID (4008)
    label_name = ID_TO_LABEL[label_id]  # 'door frame'
    pred_mask = pred['mask']  # Part points only

    # Find GT instances with same label
    for gt_inst in gt_instances[label_name]:
        # Compute IoU
        intersection = np.logical_and(
            gt_ids == gt_inst['instance_id'],
            pred_mask
        ).sum()
```

**Metrics**: mAP, AP@50, AP@25 per object-part class

### Object-Level (`eval_semantic_instance_objects.py`)

**Constants**:
```python
from multiscan_search3d_constants import (
    CLASS_LABELS,      # ('door', 'cabinet', 'toilet', ...)
    VALID_CLASS_IDS    # (4, 11, 32, ...)
)
```

**Currently Disabled**: See limitation doc

## Usage Examples

### Run Full Pipeline (Part-Level Only)

```bash
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml
```

### Check Output Files

```bash
ls inference_output/scene_00005_00_*.txt
# Expected:
#   scene_00005_00_obj_part_inst.txt  # Part-level predictions
#   scene_00005_00_obj_inst.txt       # Generated but not evaluated
```

### Validate Format

```python
from my3dis.aggregation import validate_search3d_format

result = validate_search3d_format(
    'inference_output/scene_00005_00_obj_part_inst.txt',
    expected_num_points=79614
)

if result['valid']:
    print(f"✓ Valid format: {result['num_labeled']} labeled points")
else:
    print(f"✗ Errors: {result['errors']}")
```

### Run Evaluation Manually

```bash
# Part-level
cd /media/Pluto/richkung/My3DIS/eval/search3d
python eval_semantic_instance_parts_OV.py \
  --pred-file ../../inference_output/scene_00005_00_obj_part_inst.txt \
  --gt-file /media/Pluto/richkung/My3DIS/data/search3d_gt/.../ov_part_annotations/scene_00005_00_obj_part_inst.txt
```

## Comparison: Object vs Part-Level

| Aspect | Object-Level | Part-Level |
|--------|--------------|------------|
| **GT File** | `obj_inst.txt` | `obj_part_inst.txt` |
| **Instance ID** | Simple (1006, 3001) | Composite (4008, 11002) |
| **Label Semantics** | Object class | Object-Part class |
| **Mask Coverage** | Entire object | Part only |
| **Evaluation** | Object instance seg | Fine-grained part seg |
| **Status** | ⚠️ Disabled | ✅ Working |

## Known Issues

1. **Object-Level Disabled**: Cannot evaluate due to ID/mask mismatch
2. **Partial Scene Coverage**: Only ~13% of points labeled in part-level GT
3. **Query Dependency**: Results depend on query selection and quality

## Future Work

### Short-Term
- [ ] Verify part-level evaluation metrics
- [ ] Test on multiple scenes
- [ ] Document expected performance baselines

### Mid-Term
- [ ] Fix object-level evaluation (extract object masks from metadata)
- [ ] Add support for pure object queries
- [ ] Implement multi-scene batch evaluation

### Long-Term
- [ ] Integrate with Search3D official benchmark
- [ ] Support additional GT formats (part_annotations)
- [ ] Add visualization tools for predictions vs GT

## Related Documentation

- **Implementation**: `SEARCH3D_MULTI_LEVEL_EVAL.md`
- **Limitation**: `OBJECT_LEVEL_EVALUATION_LIMITATION.md`
- **Format**: `src/my3dis/aggregation/search3d_formatter.py`
- **Pipeline**: `scripts/run_full_pipeline.py`

## Contact & Support

For issues or questions:
1. Check documentation files (listed above)
2. Verify config: `configs/inference/full_pipeline.yaml`
3. Review logs: `logs/eval/run_exp_*.log`
4. See evaluation scripts: `eval/search3d/`

---

**Last Test Run**: 2025-11-12
**Test Scene**: scene_00005_00
**Result**: ✅ Part-level evaluation working correctly
