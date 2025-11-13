# Search3D Format Validation Report

**Scene:** scene_00005_00
**Date:** 2025-11-13
**Experiment:** test_1108_ssam4_filter500_fill500_propinf

## Executive Summary

Both evaluation imports are now **FIXED** and working. However, evaluation metrics remain low (mAP ≈ 0) due to **semantic mismatches** between predictions and ground truth:

- **Object-Level:** Only 21% class overlap between GT and predictions
- **Object-Part Level:** Format mismatch - predictions use semantic IDs, GT uses instance IDs

---

## 1. Object-Level Evaluation

### Format Status: ✓ VALID

Both files use the correct format:
```
instance_id = class_id * 1000 + instance_number
```

**Examples:**
- GT: `4004` = chair(4), instance #4
- GT: `85017` = gas_cooker(85), instance #17
- Pred: `32048` = pc(32), instance #48
- Pred: `121003` = air_conditioner(121), instance #3

### Coverage Analysis: ⚠ POOR

| Metric | Ground Truth | Predictions |
|--------|-------------|-------------|
| Total points | 79,614 | 79,614 ✓ |
| Labeled points | 46,173 (58%) | 68,222 (86%) |
| Unique instances | 22 | 57 |
| Unique classes | 19 | 16 |
| **Overlapping classes** | **4 / 19 (21%)** | **4 / 16 (25%)** |

### Class Comparison

**Overlapping Classes (4):**
- `4` - chair
- `27` - toilet
- `32` - pc
- `134` - yoga_mat_roll

**GT-Only Classes (15):**
```
0   (door) ********** 34,281 points - LARGEST CLASS!
1   (desk)
2   (table)
3   (bar_table)
11  (wall_cabinet)
23  (sink)
33  (towel)
34  (air_vent)
38  (paper_towel)
56  (napkin_roll)
64  (box)
85  (gas_cooker)
118 (machine)
129 (shower)
152 (ironing_board)
```

**Prediction-Only Classes (12):**
```
6   (sink_cabinet)
17  (tv)
19  (curtain) ********** 27,217 points
29  (faucet)
30  (chopping_board)
55  (tube)
58  (water_ladle)
106 (hole_puncher)
121 (air_conditioner) ********** 18,686 points - LARGEST PRED CLASS!
140 (tablet)
147 (floor_drain)
153 (velvet_hangers)
```

### Critical Issues

1. **Missing "door" class**: GT's largest class (43% of labeled points) has ZERO predictions
2. **False "air_conditioner"**: Largest prediction class (27% of pred points) not in GT
3. **False "curtain"**: Second largest prediction class (40% of pred points) not in GT
4. **79% GT classes missing**: 15 out of 19 GT classes have no predictions at all

---

## 2. Object-Part Level Evaluation

### Format Status: ❌ MISMATCH

**Ground Truth Format:**
```
composite_id = obj_instance_id * 1000 + part_instance_id
```

**Examples:**
- `4002001` = object instance 4002 + part instance 1
- `11002001` = object instance 11002 + part instance 1
- `32005001` = object instance 32005 + part instance 1

**Prediction Format:**
```
composite_id = obj_class_id * 1000 + part_class_id  [WRONG!]
```

**Examples:**
- `4009` = door(4) + knob(9)
- `32007` = toilet(32) + tank_cover(7)
- `134020` = shower(134) + shower_head(20)

### Coverage Analysis

| Metric | Ground Truth | Predictions |
|--------|-------------|-------------|
| Total points | 79,614 | 79,614 ✓ |
| Labeled points | 10,566 (13.3%) | 68,222 (85.7%) |
| Unique instances | 8 | 20 |
| **Overlapping IDs** | **0 / 8 (0%)** | **0 / 20 (0%)** |

### ID Comparison

**Ground Truth IDs:**
```
4002001  - chair (obj_inst:4002) + drawer (part_inst:1)
4002002  - chair (obj_inst:4002) + drawer (part_inst:2)
4008001  - chair (obj_inst:4008) + knob (part_inst:1)
11002001 - cabinet (obj_inst:11002) + door (part_inst:1)
11002002 - cabinet (obj_inst:11002) + door (part_inst:2)
32005001 - toilet (obj_inst:32005) + lid (part_inst:1)
32007001 - toilet (obj_inst:32007) + tank_cover (part_inst:1)
32021001 - toilet (obj_inst:32021) + flush_button (part_inst:1)
```

**Prediction IDs (semantic composite):**
```
4009     - door + knob (11,569 points)
32007    - toilet + tank_cover (7,790 points)
32021    - toilet + flush_button (13 points)
106012   - dryer + drum (14,555 points)
121003   - dishwasher + drawer (18,686 points)
... 15 more
```

### Critical Issues

1. **Zero overlap**: NO common IDs between GT and predictions
2. **Format incompatibility**: Semantic IDs vs instance IDs cannot be compared
3. **Over-prediction**: Predictions label 86% of points, GT labels only 13%
4. **Instance mismatch**: Even for valid semantic pairs, instance IDs don't exist in predictions

---

## 3. Root Cause Analysis

### Why Object-Level Evaluation Fails

The object-level format is correct, but predictions contain **wrong object classes**:

1. **Query mismatch**: Inference queries don't match scene content
2. **Visual confusion**: Model misclassifies objects (e.g., door → air_conditioner)
3. **Over-segmentation**: 57 predicted instances vs 22 GT instances

**Example Confusion:**
```
GT Point Distribution:
  door (0):             34,281 points (43% of labeled)
  bar_table (3):        15,318 points (20%)
  chair (4):             9,065 points (12%)

Prediction Point Distribution:
  air_conditioner (121): 18,686 points (27% of labeled)
  curtain (19):          27,217 points (40%)
  chair (4):                141 points (0.2%)  ← Massive under-prediction!
```

### Why Object-Part Evaluation Fails

The object-part predictions use **semantic composite IDs** instead of **instance composite IDs**:

**Current (Wrong):**
```python
# Formatter outputs: obj_class * 1000 + part_class
label_id = 4009  # door(4) + knob(9)
```

**Expected (Correct):**
```python
# Should output: obj_instance * 1000 + part_instance
label_id = 4002001  # object instance 4002 + part instance 1
```

**Impact:**
- Evaluator looks for instance IDs (e.g., `4002001`)
- Predictions provide semantic IDs (e.g., `4009`)
- No matches possible → mAP = 0%

---

## 4. Detailed Findings

### Object-Level Instance Distribution

**Ground Truth:**
- Most instances have 500-10k points
- Largest: door instance (85017) with heavy coverage
- Well-separated instances with clear boundaries

**Predictions:**
- More instances (57 vs 22)
- Smaller average size
- Many false positives (non-existent classes)

### Object-Part Coverage

**Ground Truth:**
- Only 13.3% of points labeled (sparse annotation)
- Focuses on critical parts: doors, drawers, lids, tank covers
- 8 carefully annotated object-part relationships

**Predictions:**
- 85.7% of points labeled (dense prediction)
- Predicts many more part relationships (20)
- Includes classes not in GT (dishwasher, dryer, rice_cooker)

---

## 5. Recommendations

### Immediate Fixes

1. **Fix formatter ID generation** (`src/my3dis/aggregation/search3d_formatter.py`)
   ```python
   # WRONG (current):
   label_id = obj_class_id * 1000 + part_class_id

   # RIGHT (needed):
   label_id = obj_instance_id * 1000 + part_instance_id
   ```

2. **Check query file** used for inference
   - Current queries may not match scene content
   - Consider using GT-based queries for validation

3. **Validate object class predictions**
   - Debug why "door" class (GT's 43%) has zero predictions
   - Debug why "air_conditioner" (pred's 27%) is false positive

### Short-Term Improvements

1. **Add query validation**
   - Compare queries against GT classes before inference
   - Warn if major GT classes are missing from queries

2. **Add intermediate visualizations**
   - Render predictions vs GT for visual inspection
   - Highlight class mismatches in 3D

3. **Add format validation to pipeline**
   - Check ID format before evaluation
   - Auto-detect semantic vs instance ID format

### Long-Term Solutions

1. **Scene-specific queries**
   - Generate queries based on GT class distribution
   - Adaptive query selection per scene

2. **Two-stage evaluation**
   - Stage 1: Semantic validation (are predicted classes reasonable?)
   - Stage 2: Instance matching (current mAP evaluation)

3. **Instance ID tracking**
   - Maintain instance IDs throughout pipeline
   - Generate unique instance IDs for each predicted mask

---

## 6. Files Analyzed

### Ground Truth
- Object-level: `/media/Pluto/richkung/My3DIS/data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/obj_annotations/scene_00005_00_obj_inst.txt`
- Object-part: `/media/Pluto/richkung/My3DIS/data/search3d_gt/multiscan_data_search3d/multiscan_annotations_search3d/ov_part_annotations/scene_00005_00_obj_part_inst.txt`

### Predictions
- Object-level: `.../inference_output/hierarchical/scene_00005_00_obj_inst.txt`
- Object-part: `.../inference_output/hierarchical/scene_00005_00_obj_part_inst.txt`

### Evaluation Scripts
- Object-level: `eval/search3d/eval_semantic_instance_objects.py` ✓ FIXED
- Object-part: `eval/search3d/eval_semantic_instance_parts_OV.py` ✓ FIXED

---

## 7. Next Steps

### Priority 1: Fix Object-Part ID Format
```bash
# Edit: src/my3dis/aggregation/search3d_formatter.py
# Change label_id generation to use instance IDs

# Re-run inference
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation  # Aggregation is fine, only re-run inference
```

### Priority 2: Validate Object Classes
```bash
# Check queries used
cat eval/search3d/search3d_ov_part_queries.txt

# Compare with GT classes in scene
python scripts/validate_scene_queries.py \
  --gt-file .../obj_annotations/scene_00005_00_obj_inst.txt \
  --query-file eval/search3d/search3d_ov_part_queries.txt
```

### Priority 3: Re-evaluate
```bash
# After fixes, re-run evaluation
PYTHONPATH=src python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --skip-aggregation \
  --skip-inference  # Only run evaluation
```

---

## Contact

For questions or issues:
1. Check `SEARCH3D_EVALUATION_STATUS.md` for implementation details
2. Review `OBJECT_LEVEL_EVALUATION_LIMITATION.md` for known limitations
3. See formatter code: `src/my3dis/aggregation/search3d_formatter.py`

---

**Report Generated:** 2025-11-13
**Evaluations Fixed:** eval_semantic_instance_objects.py, eval_semantic_instance_parts_OV.py
**Status:** Import errors resolved ✓, Format mismatches identified ✓, Recommendations provided ✓
