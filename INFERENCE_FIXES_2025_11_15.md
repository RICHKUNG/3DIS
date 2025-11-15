# SigLIP GT Sanity Check - Comprehensive Analysis

**Date**: 2025-11-15
**Scene**: scene_00005_00
**Test Scale**: 7 GT proposals (5 with text labels)

---

## Executive Summary

🔴 **CRITICAL FINDING**: GT visual features show **fundamentally low raw cosine similarity** (0.03-0.06) with their corresponding text labels. This indicates a **serious problem in the feature extraction pipeline**, not just a ranking or scaling issue.

**Key Results**:
- ✅ Baseline (topk=1, scale=300, softmax): **60% rank-1** accuracy
- ❌ TopK=5 views: **40% rank-1** - **WORSE than baseline!**
- ✅ No softmax: **60% rank-1** - same as baseline
- ⚠️ Raw similarities: **ALL below 0.065** (expected > 0.5 for correct matches)

**Conclusion**: The problem is **NOT** viewing angles or softmax, but **fundamental feature misalignment**.

---

## Experiment Results Summary

### Configuration Comparison

| Configuration | Rank-1 | Mean Correct Sim | Cabinet Door Sim | Notes |
|--------------|--------|------------------|------------------|-------|
| **Baseline** (topk=1, scale=300, softmax) | **60%** | 0.433 | 0.009 | 3/5 correct |
| **TopK=5** (topk=5, scale=300, softmax) | **40%** | 0.367 | 0.004 | **WORSE!** |
| **No Softmax** (topk=1, raw cosine) | **60%** | 0.045 | 0.023 | Same ranking |
| **Original Script** (raw similarity only) | N/A | 0.050 | 0.035 | No ranking |
| **Scale=100** | **OOM** | - | - | CUDA error |

### Key Observations

1. **TopK=5 Made Things WORSE**
   - Rank-1 dropped from 60% to 40%
   - Toilet tank_cover dropped from rank-1 to rank-2
   - Cabinet door similarities got LOWER (0.009 → 0.004)
   - **Conclusion**: Problem is NOT insufficient viewing angles

2. **Softmax Doesn't Change Rankings**
   - With softmax: 60% rank-1 (mean sim 0.433)
   - Without softmax: 60% rank-1 (mean sim 0.045)
   - **Conclusion**: Softmax only amplifies differences, doesn't fix alignment

3. **Raw Cosine Similarities Are ALL Low**
   - Original script shows: door frame 0.064, toilet lid 0.062, cabinet door 0.035
   - **ALL similarities < 0.065**
   - **Expected**: Correct matches should be > 0.5 (ideally > 0.7)
   - **Conclusion**: Fundamental feature extraction problem

---

## Per-Instance Detailed Analysis

### Successful Cases (Still Problematic)

**1. Door Frame (ID 4008001)**
- Points: 2149
- Baseline: 0.88 (softmax) / 0.064 (raw)
- TopK=5: 0.82 (softmax)
- **Status**: Rank-1 in both tests
- **Note**: High after softmax, but raw similarity only 0.064!

**2. Toilet Lid (ID 32005001)**
- Points: 554
- Baseline: 0.72 (softmax) / 0.062 (raw)
- TopK=5: 0.72 (softmax)
- **Status**: Rank-1 in both tests

**3. Toilet Tank Cover (ID 32007001)**
- Points: 268
- Baseline: 0.54 (softmax) / 0.046 (raw)
- TopK=5: 0.28 (softmax) - **DEGRADED!**
- **Status**: Rank-1 in baseline, **Rank-2 in topk=5**
- **Note**: Performance got WORSE with more views!

### Failed Cases (Complete Failures)

**4. Cabinet Door Instance 1 (ID 11002001)**
- Points: 309
- Baseline: 0.010 (softmax) / 0.045 (raw)
- TopK=5: 0.008 (softmax) - **WORSE!**
- Max incorrect: 0.98 (baseline), 0.97 (topk=5)
- **Status**: Rank-2 (baseline), Rank-3 (topk=5)
- **Margin**: -0.97 (wrong labels much higher!)

**5. Cabinet Door Instance 2 (ID 11002002)**
- Points: 347
- Baseline: 0.009 (softmax) / 0.035 (raw)
- TopK=5: 0.004 (softmax) - **MUCH WORSE!**
- Max incorrect: 0.93 (baseline), 0.66 (topk=5)
- **Status**: Rank-4 in both tests (last place)
- **Margin**: -0.92 (baseline), -0.65 (topk=5)

---

## Root Cause Analysis

### What We've Ruled Out

1. ❌ **Insufficient Viewing Angles**
   - TopK=5 performed WORSE than topk=1
   - More views degraded both cabinet door AND toilet tank_cover
   - Suggests averaging more views dilutes signal

2. ❌ **Softmax/Scaling Issues**
   - With/without softmax: same rank-1 accuracy (60%)
   - Softmax is just a monotonic transformation
   - Doesn't change relative ordering

3. ❌ **Temperature Parameter**
   - Scale=100 test failed (OOM)
   - But scaling only affects magnitude, not underlying similarity

### What the Evidence Points To

1. ✅ **Fundamental Feature Misalignment**
   - **Raw cosine similarities ALL < 0.065**
   - Expected correct matches: > 0.5 (ideally > 0.7)
   - Gap is **10x too low**!

2. ✅ **Visual Features May Not Match Text Features**
   - Door frame: 0.064 raw similarity with "door frame" text
   - This is barely above random chance
   - Suggests visual and text features are in different spaces

3. ✅ **Possible Causes**:

   **A. Feature Extraction Issues**
   - `extract_proposal_features_siglip()` may not be using the correct method
   - 3D-to-2D projection may be losing critical information
   - Frame aggregation (mean pooling) may be washing out features

   **B. Text Feature Issues**
   - Text prompts may need template engineering
   - Current: "cabinet door"
   - Should try: "a photo of a cabinet door", "cabinet door in a room"

   **C. 3D-to-2D Projection Issues**
   - `WORLD_2_CAM` projection may not be correctly mapping 3D masks to 2D
   - Visibility checking may be too aggressive
   - Sparse storage (13.2% of points) may be insufficient

   **D. Model/Tokenizer Mismatch**
   - SigLIP model may not be correctly loaded
   - Preprocessing may differ between visual and text paths

---

## Recommended Next Steps

### High Priority (Immediate)

1. **Verify Feature Extraction Correctness**
   - Test SigLIP on a known image-text pair
   - Verify raw cosine similarities are in expected range (0.3-0.8)
   - Check if visual encoder is properly loaded

2. **Check Text Prompt Engineering**
   - Try different text templates:
     - "a photo of a {label}"
     - "{label} in an indoor scene"
     - "indoor {label}"

3. **Visualize 3D-to-2D Projections**
   - For cabinet door instances:
     - Which frames they project to
     - What the projected masks look like
     - Whether the masks capture meaningful features

4. **Compare with Notebook Pipeline**
   - Run exact same GT through notebook pipeline
   - Compare feature extraction step-by-step
   - Identify where differences occur

### Medium Priority

5. **Test Different Aggregation Methods**
   - Current: mean pooling across frames
   - Try: max pooling, attention-weighted aggregation

6. **Check Model Loading**
   - Verify SigLIP visual and text encoders are from same checkpoint
   - Check model configuration and frozen layers

7. **Test on Known-Good Examples**
   - Find examples where inference worked well
   - Extract their GT and test similarity
   - Establish baseline for "correct" similarity values

### Low Priority

8. **Increase Test Scale**
   - Test on 50+ instances across multiple scenes
   - Verify pattern is consistent

9. **Try Different Scenes**
   - Test scene_00006_00, scene_00018_00
   - Check if cabinet door fails everywhere

---

## Hypothesis: The Real Problem

Based on the evidence, I hypothesize:

**The visual features extracted from GT 3D proposals and the text features from SigLIP are NOT in the same embedding space.**

### Evidence:
1. **All raw similarities extremely low** (0.03-0.06)
2. **Even "successful" cases have low raw similarity** (door frame: 0.064)
3. **More views make it worse** (suggests averaging in wrong space)
4. **Softmax amplifies differences but doesn't fix alignment**

### Possible Root Causes:

**A. Different Preprocessing**
- Visual features: 3D → 2D projection → crop → resize → SigLIP
- Text features: tokenize → SigLIP
- May have different normalization, scaling, or preprocessing

**B. Wrong Feature Extraction Method**
- `extract_proposal_features_siglip()` may not match inference pipeline
- May be using wrong pooling, wrong crops, or wrong frames

**C. Model Loading Issue**
- Visual and text encoders may not be from the same checkpoint
- May have different configurations or frozen layers

**D. 3D-to-2D Projection Artifacts**
- Sparse storage (13.2%) may lose critical visual information
- Visibility filtering may be too strict
- Frame selection may miss important views

---

## Conclusion

**The SigLIP sanity check has FAILED** - not due to softmax or viewing angles, but due to **fundamental low similarity between GT visual and text features**.

**Critical Issue**: Raw cosine similarities are 10x lower than expected (0.03-0.06 vs expected > 0.5).

**Immediate Action Required**:
1. Verify feature extraction code matches inference pipeline
2. Test SigLIP on known image-text pairs to verify correct operation
3. Check if text prompt engineering is needed
4. Visualize what visual features are actually being extracted
5. Compare with known-good examples from inference

**Recommendation**: **Do NOT proceed with full-scale inference** until this is resolved, as the low similarity suggests the retrieval pipeline may not work correctly.

---

**Generated**: 2025-11-15  
**Test Data**:
- Baseline: `/tmp/test_20samples_with_softmax.json`
- TopK=5: `/tmp/test_topk5_softmax.json`
- No Softmax: `/tmp/test_20samples_no_softmax.json`
- Original: `/tmp/test_original_script.json`

**Scripts**:
- `scripts/siglip_gt_sanity_check_with_scaling.py`
- `scripts/check_siglip_gt_similarity.py`
- `scripts/compare_all_experiments.py`
