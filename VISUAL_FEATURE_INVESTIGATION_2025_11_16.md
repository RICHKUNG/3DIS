# Visual Feature Extraction Investigation - Progress Report

**Date**: 2025-11-16
**Status**: Investigation in progress - GPU memory constraints encountered
**Objective**: Determine why GT visual features have extremely low similarity with text features

---

## Summary of Work Completed

### ✅ What We've Established

1. **SigLIP Model is Working Correctly**
   - Text-text similarity: Perfect 1.0000 self-match
   - Model loading: Verified correct
   - Inference validated on known text pairs
   - Source: `SIGLIP_BASELINE_VERIFICATION_2025_11_15.md`

2. **Text Processing is Correct**
   - Raw labels give best performance (no templates needed)
   - Any prompt engineering reduces similarity by 10-30%
   - Scale=300 and softmax don't fix the core issue
   - Source: `quick_prompt_test.py` results in `/tmp/quick_prompt_test.log`

3. **Core Problem Identified**
   - **Raw cosine similarities are 10x too low**: 0.03-0.06 instead of expected >0.5
   - This is NOT a ranking/normalization issue
   - This IS a fundamental feature space alignment problem
   - Visual features appear to be in a different embedding space than text features

4. **Specific Failure Pattern**
   - ✅ **Successful**: door frame (0.064 raw, 2149 points), toilet lid (0.062 raw, 554 points)
   - ❌ **Complete failure**: cabinet door (0.023-0.045 raw, 309-347 points)
   - Cabinet door instances rank 2-4 with huge negative margins (-0.92 to -0.97)
   - Wrong labels score higher (0.93-0.98) than correct labels!

5. **TopK Impact**
   - topk=1: 60% rank-1 accuracy
   - topk=5: 40% rank-1 accuracy (WORSE!)
   - **Conclusion**: Adding more views degrades performance
   - Indicates either: (a) view selection is wrong, or (b) aggregation method is faulty

---

## Investigation Tools Created

### Analysis Scripts

1. **`scripts/siglip_gt_sanity_check_with_scaling.py`**
   - Main sanity check with proper scaling and softmax
   - Tests GT proposals against text features
   - Generates detailed per-instance analysis

2. **`scripts/verify_siglip_baseline.py`**
   - Validates SigLIP model functionality
   - Tests text-text similarity
   - Compares prompt templates

3. **`scripts/quick_prompt_test.py`**
   - Fast prompt template comparison
   - Self-similarity testing
   - No GPU required (text-only)

4. **`scripts/diagnose_siglip_similarity.py`**
   - Statistical analysis of similarity distributions
   - Per-category breakdown
   - Confusion pattern identification

5. **`scripts/compare_all_experiments.py`**
   - Side-by-side experiment comparison
   - Improvement tracking
   - Aggregated statistics

6. **`scripts/analyze_gt_projection.py`** (NEW)
   - Analyzes 3D-to-2D projection quality
   - Counts visible frames per proposal
   - Calculates coverage ratios and bbox areas
   - **Status**: Created but encountered CUDA OOM

7. **`scripts/visualize_gt_projection.py`** (NEW)
   - Visualizes projected masks on RGB frames
   - Requires opencv (cv2) - not available in environment
   - **Status**: Not functional due to missing dependency

### Documentation Generated

1. `SIGLIP_SANITY_CHECK_FINDINGS.md` - Initial findings
2. `INFERENCE_FIXES_2025_11_15.md` - Comprehensive experiment analysis
3. `SIGLIP_BASELINE_VERIFICATION_2025_11_15.md` - Model verification results
4. `FINAL_SIGLIP_INVESTIGATION_2025_11_15.md` - Complete investigation report

---

## Current Blocker: GPU Memory

### Attempted Analysis

Tried to run projection quality analysis but encountered:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 5.80 GiB.
GPU 0 has a total capacity of 23.68 GiB of which 164.88 MiB is free.
```

### Root Cause

Multiple background processes running simultaneously:
- Multiple instances of `siglip_gt_sanity_check_with_scaling.py`
- Original `check_siglip_gt_similarity.py`
- Various test configurations (topk=1, topk=5, scale=100, scale=300, etc.)

All these processes have loaded WORLD_2_CAM projections into GPU memory.

### Implication

**This reveals an important finding**: The 3D-to-2D projection computation itself is very memory-intensive (~5.8 GiB just for the projection matrix), which may be contributing to the quality issues if aggressive downsampling or approximations are being used to fit in memory.

---

## Next Steps (Prioritized)

### HIGH PRIORITY (Execute Immediately)

#### 1. Clear GPU Memory & Re-run Projection Analysis

**Goal**: Understand projection quality differences between successful vs failed cases

**Actions**:
```bash
# Kill all background processes
pkill -f siglip_gt_sanity_check
pkill -f check_siglip_gt_similarity

# Wait for GPU to clear
nvidia-smi

# Run projection analysis with clean GPU
conda run -n 3Dsiglip python scripts/analyze_gt_projection.py \
  --scene-path /media/public_dataset2/multiscan/scene_00005_00 \
  --annotation-file .../scene_00005_00_obj_part_inst.txt \
  --limit 7 \
  --min-points 150 \
  --output-file /tmp/gt_projection_analysis.txt
```

**Expected Output**:
- Number of visible frames per proposal
- Average points per frame
- Coverage ratio (% of 3D points visible in at least one frame)
- Bbox area statistics

**Hypothesis to Test**:
- Cabinet door has fewer visible frames than door frame
- Cabinet door projections have smaller bbox areas (less visual information)
- Cabinet door may have low coverage ratio (many 3D points never visible)

#### 2. Investigate `extract_proposal_features_siglip()` Implementation

**Goal**: Understand exactly how visual features are extracted and aggregated

**File to Read**: `3DprojToSiglip/notebook_pipeline.py`

**Key Questions**:
1. How are topK frames selected? (Current method seems to make things worse)
2. What aggregation method is used? (Mean pooling may dilute features)
3. How are visibility masks applied?
4. Are there any normalization steps that might cause feature space mismatch?

**Actions**:
```bash
# Read the implementation
less 3DprojToSiglip/notebook_pipeline.py

# Search for key functions
grep -n "def extract_proposal_features_siglip" 3DprojToSiglip/notebook_pipeline.py
grep -n "def generate_grounding_features_siglip" 3DprojToSiglip/notebook_pipeline.py
grep -n "def pool_point_to_proposal_features" 3DprojToSiglip/notebook_pipeline.py
grep -n "topk" 3DprojToSiglip/notebook_pipeline.py
```

#### 3. Single-Frame Test (Bypass Aggregation)

**Goal**: Test if feature extraction works correctly for a single, clear view

**Approach**:
1. Manually identify a frame where cabinet door is clearly visible
2. Extract SigLIP features directly from that frame's cropped region
3. Compare with text feature for "cabinet door"
4. If similarity is still low (~0.03), problem is in SigLIP image encoding
5. If similarity is high (>0.5), problem is in aggregation/projection

**Pseudo-code**:
```python
# Load one RGB frame where cabinet door is visible
frame_idx = <manually selected>
rgb_image = load_image(frame_idx)

# Crop to cabinet door region (using GT bbox)
bbox = get_gt_bbox_for_cabinet_door(frame_idx)
cropped = rgb_image[bbox]

# Extract SigLIP visual feature
visual_feat = siglip_model.get_image_features(cropped)
visual_feat = visual_feat / visual_feat.norm()

# Extract text feature
text_feat = siglip_model.get_text_features("cabinet door")
text_feat = text_feat / text_feat.norm()

# Compute similarity
similarity = visual_feat @ text_feat.T
print(f"Single-frame similarity: {similarity:.4f}")
```

### MEDIUM PRIORITY

#### 4. Compare with Inference Pipeline

**Goal**: Verify our GT test uses exact same feature extraction as inference

**Actions**:
- Find actual inference code that generates proposals
- Trace through feature extraction path
- Compare parameters: topk, batch_size, resize_ratio, vis_depth_threshold
- Verify normalization steps match exactly

#### 5. Test Different Aggregation Methods

**Current**: Mean pooling across topK frames
**Alternatives to test**:
- Max pooling (preserve strongest signal)
- Weighted average by view quality
- Median pooling (robust to outliers)
- Single best frame (no aggregation)

#### 6. Adjust Visibility Filtering Parameters

**Current parameters** (from our scripts):
```python
vis_depth_threshold = 0.03
frequency = 1  # or 100 in some tests
resize_ratio = 0.3
```

**Test variations**:
- Looser depth threshold: 0.05, 0.10
- Higher resize ratio: 0.5, 1.0 (more detail)
- Different frame sampling frequencies

### LOW PRIORITY

#### 7. Expand Test Scale

- Increase to 50+ GT instances
- Test across multiple scenes
- Establish statistical significance

#### 8. Alternative Feature Extraction

- Test without topK selection (use all visible frames)
- Test learned aggregation weights
- Test attention-based pooling

---

## Hypotheses Under Investigation

### Hypothesis A: TopK Selection is Broken

**Evidence**:
- topk=5 performs WORSE than topk=1
- Suggests "best" views are actually worst

**Test**: Inspect view quality scoring function
**Fix**: Improve view selection criteria (angle, coverage, occlusion)

### Hypothesis B: Visibility Filtering Too Strict

**Evidence**:
- Cabinet door (smaller object) fails completely
- Door frame (larger object) succeeds
- May indicate small objects filtered out in many frames

**Test**: Check what % of cabinet door points are visible per frame
**Fix**: Relax vis_depth_threshold or coverage requirements

### Hypothesis C: Mean Pooling Dilutes Features

**Evidence**:
- Adding more frames (topk=5) makes things worse
- Small objects with few visible points may get averaged to noise

**Test**: Try max pooling or weighted aggregation
**Fix**: Use aggregation method that preserves strong signals

### Hypothesis D: 3D-to-2D Projection Inaccurate

**Evidence**:
- CUDA OOM suggests projection is very memory-intensive
- May use approximations or downsampling that lose precision

**Test**: Visualize actual projected masks on RGB frames
**Fix**: Increase projection resolution or reduce approximations

---

## Resource Usage Notes

### Memory Requirements

The investigation revealed that WORLD_2_CAM initialization requires significant GPU memory:
- **Per-scene projection**: ~5.8 GiB
- **SigLIP model**: ~2-3 GiB
- **Feature extraction batches**: Variable

This means:
- Can't run multiple scenes in parallel on 24GB GPU
- Need to carefully manage process lifecycle
- Visualization tools may need CPU fallback

### Recommended Workflow

1. **Clear GPU before each major experiment**:
   ```bash
   pkill -f python
   nvidia-smi  # verify clean
   ```

2. **Use sequential execution** instead of parallel background processes

3. **Save intermediate results** to avoid recomputation:
   - Cache WORLD_2_CAM projections
   - Save extracted features to disk
   - Load from cache for analysis

---

## Key Files Reference

### Scripts
- `scripts/siglip_gt_sanity_check_with_scaling.py` - Main GT test
- `scripts/analyze_gt_projection.py` - Projection quality analysis
- `scripts/quick_prompt_test.py` - Text-only prompt testing

### Data
- `/tmp/test_20samples_with_softmax.json` - Baseline results
- `/tmp/test_20samples_no_softmax.json` - No softmax comparison
- `/tmp/test_topk5_softmax.json` - TopK=5 results

### Documentation
- `FINAL_SIGLIP_INVESTIGATION_2025_11_15.md` - Complete findings to date
- `SIGLIP_BASELINE_VERIFICATION_2025_11_15.md` - Model verification

### Source Code to Investigate
- `3DprojToSiglip/notebook_pipeline.py` - Feature extraction implementation
- `src/my3dis/aggregation/aggregation_pipeline.py` - WORLD_2_CAM initialization

---

## Recommendations

### For Immediate Action

1. **DO NOT proceed with large-scale inference** until visual feature extraction is fixed
2. **Clear GPU and run projection analysis** to understand visibility patterns
3. **Read `extract_proposal_features_siglip` implementation** to find root cause

### For Quick Wins

1. **Try topk=1 with improved view selection** instead of topk=5
2. **Test max pooling** instead of mean pooling
3. **Use raw labels** (no prompt templates)
4. **Filter out small proposals** (< 500 points) as potentially unstable

### For Long-term Fix

1. **Redesign aggregation method** based on findings
2. **Optimize memory usage** to enable larger-scale testing
3. **Add projection quality metrics** to pipeline monitoring
4. **Create regression tests** with known good/bad cases

---

## Timeline Estimate

- **Projection analysis** (with clean GPU): 1-2 hours
- **Code review of feature extraction**: 2-3 hours
- **Single-frame testing**: 1-2 hours
- **Root cause identification**: 4-8 hours total
- **Fix implementation**: Depends on complexity (4-16 hours)
- **Validation**: 2-4 hours

**Total estimate**: 2-4 days for complete investigation and fix

---

**Report Generated**: 2025-11-16
**Next Update**: After GPU memory cleared and projection analysis completed
