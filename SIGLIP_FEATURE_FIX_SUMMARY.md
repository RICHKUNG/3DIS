# SigLIP Feature Extraction – Consolidated Fix History

**Last updated:** 2025-11-16  
**Scope:** All investigations and fixes related to SigLIP feature extraction inside `My3DIS/`

---

## 1. Current Status (2025-11-16)
- ✅ **Real SigLIP embeddings are now used across aggregation/inference runs.** Runs executed in the `3Dsiglip` environment report normalized embeddings (‖f‖≈1.0) instead of the earlier random vectors ([`MAP_OPTIMIZATION_REPORT_2025_11_16.md`](dump/siglip_reports/MAP_OPTIMIZATION_REPORT_2025_11_16.md), [`SCENE_COVERAGE_ANALYSIS_2025_11_16.md`](dump/siglip_reports/SCENE_COVERAGE_ANALYSIS_2025_11_16.md)).
- ✅ **Aggregation pipeline imports the fixed extractor.** `src/my3dis/aggregation/aggregation_pipeline.py:488` now pulls `generate_grounding_features_siglip_batched` from `3DprojToSiglip/utils_new/feature_extraction_fixed.py`, the implementation that averages features correctly.
- ✅ **Per-instance sanity check reflects ~10× higher similarity.** Mean correct similarity improved from 0.03-0.06 to **0.4606** while maintaining 60% rank-1 accuracy (see [`SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md)).
- ⚠️ **Small/occluded proposals still underperform.** Cabinet-door instance #2 (0.1449 similarity) and toilet tank_cover (0.0212) remain problematic; fixes here are outside the averaging bug (same reference as above).
- ⚠️ **`src/my3dis/siglip_assignment/feature_extraction.py` is still a placeholder** (identity projection, heuristic frame scores, multi-scale crops). Treat it as experimental until it receives the same fixes.

---

## 2. Timeline of Key Findings & Actions
| Date | Artifact | Key Takeaways |
|------|----------|---------------|
| **Nov 15** | [`SIGLIP_SANITY_CHECK_FINDINGS.md`](dump/siglip_reports/SIGLIP_SANITY_CHECK_FINDINGS.md), [`INFERENCE_FIXES_2025_11_15.md`](dump/siglip_reports/INFERENCE_FIXES_2025_11_15.md), [`FINAL_SIGLIP_INVESTIGATION_2025_11_15.md`](dump/siglip_reports/FINAL_SIGLIP_INVESTIGATION_2025_11_15.md) | GT visual-text similarity only 0.03-0.06 raw with 60% rank-1; softmax, top-k views, or prompt tweaks did not help → issue isolated to visual feature extraction. |
| **Nov 15** | [`SIGLIP_BASELINE_VERIFICATION_2025_11_15.md`](dump/siglip_reports/SIGLIP_BASELINE_VERIFICATION_2025_11_15.md) | Text encoder self-matches at 1.0000; SigLIP model + tokenizer confirmed healthy, so bug is not in the model weights. |
| **Nov 16** | [`FEATURE_EXTRACTION_BUGS_FOUND.md`](dump/siglip_reports/FEATURE_EXTRACTION_BUGS_FOUND.md) | Deep dive revealed three coupled bugs: accumulation without averaging, noisy multi-scale crops, and a >20 visible-point threshold that discards small objects. |
| **Nov 16** | [`SIGLIP_AUDIT_REPORT_2025_11_16.md`](dump/siglip_reports/SIGLIP_AUDIT_REPORT_2025_11_16.md) | Code audit confirmed buggy implementation still wired into production aggregation; inference extractor already OK; `siglip_assignment` path remains incomplete. |
| **Nov 16** | [`SIGLIP_FIXES_COMPLETE_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXES_COMPLETE_2025_11_16.md), [`3DprojToSiglip/utils_new/feature_extraction_fixed.py`](3DprojToSiglip/utils_new/feature_extraction_fixed.py) | Fixed implementation added (proper averaging, single crop, dynamic thresholds) and exported under the original API name. Aggregation pipeline updated to use it. |
| **Nov 16** | [`SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md) | Validation script `scripts/siglip_gt_sanity_check_fixed.py` logged ~10× similarity gains; remaining failures attributed to view/visibility limits. |
| **Nov 16** | [`CONFIG_FIX_REPORT_2025_11_16.md`](dump/siglip_reports/CONFIG_FIX_REPORT_2025_11_16.md) → [`MAP_OPTIMIZATION_REPORT_2025_11_16.md`](dump/siglip_reports/MAP_OPTIMIZATION_REPORT_2025_11_16.md) | Early Nov-16 runs still fell back to random features because `transformers` was missing in the SAM2 environment. Later that day the team switched to the `3Dsiglip` environment where `transformers` is available, so real embeddings with norms≈1.0 are now observed. |

---

## 3. Root Cause & Fix Summary
| Bug | Symptom | Fix Implemented | References |
|-----|---------|-----------------|------------|
| **Accumulation instead of averaging** | Points seen in multiple crops/views accumulated raw features and only normalized once, causing direction bias and semantic dilution. | Track per-point counts (`pc_counts`) and divide before final normalization. Implemented in `feature_extraction_fixed.py` and validated via aggregated similarity jump. | [`FEATURE_EXTRACTION_BUGS_FOUND.md`](dump/siglip_reports/FEATURE_EXTRACTION_BUGS_FOUND.md), [`SIGLIP_FIXES_COMPLETE_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXES_COMPLETE_2025_11_16.md) |
| **Multi-scale crops injected background noise** | Each view contributed three progressively looser crops; background dominated small proposals. | Use a single tight crop per view; remove expanding loop. | Same as above. |
| **Hard-coded visibility threshold (>20 pts)** | Small objects (cabinet doors ~340 pts) rarely met the threshold, so their best views were skipped entirely. | Adopt dynamic threshold `max(5, proposal_size×0.05)` inside the batched extractor. | Same as above. |
| **Environment lacked transformers** | Pipeline log printed “transformers not available... random features,” leading to meaningless retrieval scores. | Run aggregation inside `3Dsiglip` env (or install `transformers` in SAM2). Subsequent analyses confirm real SigLIP vectors. | [`CONFIG_FIX_REPORT_2025_11_16.md`](dump/siglip_reports/CONFIG_FIX_REPORT_2025_11_16.md), [`MAP_OPTIMIZATION_REPORT_2025_11_16.md`](dump/siglip_reports/MAP_OPTIMIZATION_REPORT_2025_11_16.md) |

---

## 4. Validation Snapshot
| Metric | Before Fix (buggy extractor) | After Fix (fixed extractor) |
|--------|-----------------------------|-----------------------------|
| Mean correct similarity | 0.03 – 0.06 | **0.4606** |
| Rank-1 accuracy | 60% (3/5) | 60% (3/5) – same scenes, but similarities align with expectations |
| Door frame similarity | 0.064 | **0.9964** |
| Toilet lid similarity | 0.062 | **0.6232** |
| Cabinet door #1 similarity | 0.023 – 0.045 | **0.5174** |
| Cabinet door #2 similarity | 0.009 – 0.010 | 0.1449 (needs further work) |
| Toilet tank_cover similarity | 0.046 | 0.0212 (still failing) |

_Source: [`SIGLIP_SANITY_CHECK_FINDINGS.md`](dump/siglip_reports/SIGLIP_SANITY_CHECK_FINDINGS.md) and [`SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md)._

---

## 5. Operational & Workflow Notes
- **Validation tooling.** Use `scripts/siglip_gt_sanity_check_fixed.py` for per-instance sanity checks and `scripts/verify_siglip_baseline.py` for model-level verification. These scripts share the fixed extractor path and act as regression tests.
- **Config-driven inference.** `src/my3dis/inference/inference_pipeline.py` now honors YAML configs (see [`CONFIG_FIX_REPORT_2025_11_16.md`](dump/siglip_reports/CONFIG_FIX_REPORT_2025_11_16.md)), so SigLIP improvements propagate to downstream pairing.
- **Environment hygiene.** Keep `transformers`, `torch`, and `Pillow` installed in the environment that runs aggregation. Missing packages revert to random features, negating all fixes.

---

## 6. Outstanding Risks & Follow-Up Tasks
1. **Hard examples still mis-rank.** Cabinet door #2 and toilet tank_cover require better view selection or projection quality. Suggested follow-ups live in [`SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md) (e.g., inspect selected frames, improve visibility heuristics, test more GT).  
2. **`siglip_assignment` path is incomplete.** `src/my3dis/siglip_assignment/feature_extraction.py` still uses identity projections, placeholder frame scores, and multi-scale crops. It should either be updated to reuse the fixed extractor or clearly documented as experimental.  
3. **Scene selection matters.** Scene `scene_00005_01` has sparse/unknown GT labels, so mAP remains 0 despite real features. `SCENE_COVERAGE_ANALYSIS_2025_11_16.md` and `MAP_OPTIMIZATION_REPORT_2025_11_16.md` recommend switching to scenes like `scene_00093_01` for meaningful metrics.  
4. **Broader validation.** Current regression only covers 7 GT instances. Expand to 20-50 instances across multiple scenes to guard against regressions before declaring the pipeline production-ready.

---

## 7. Reference Index
- Bug analyses: [`FEATURE_EXTRACTION_BUGS_FOUND.md`](dump/siglip_reports/FEATURE_EXTRACTION_BUGS_FOUND.md)
- Early investigations: [`SIGLIP_SANITY_CHECK_FINDINGS.md`](dump/siglip_reports/SIGLIP_SANITY_CHECK_FINDINGS.md), [`INFERENCE_FIXES_2025_11_15.md`](dump/siglip_reports/INFERENCE_FIXES_2025_11_15.md), [`FINAL_SIGLIP_INVESTIGATION_2025_11_15.md`](dump/siglip_reports/FINAL_SIGLIP_INVESTIGATION_2025_11_15.md)
- Model verification: [`SIGLIP_BASELINE_VERIFICATION_2025_11_15.md`](dump/siglip_reports/SIGLIP_BASELINE_VERIFICATION_2025_11_15.md)
- Fix implementations: [`SIGLIP_FIXES_COMPLETE_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXES_COMPLETE_2025_11_16.md), [`SIGLIP_AUDIT_REPORT_2025_11_16.md`](dump/siglip_reports/SIGLIP_AUDIT_REPORT_2025_11_16.md), [`3DprojToSiglip/utils_new/feature_extraction_fixed.py`](3DprojToSiglip/utils_new/feature_extraction_fixed.py)
- Validation logs: [`SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md`](dump/siglip_reports/SIGLIP_FIXED_TEST_RESULTS_2025_11_16.md)
- Pipeline health: [`CONFIG_FIX_REPORT_2025_11_16.md`](dump/siglip_reports/CONFIG_FIX_REPORT_2025_11_16.md), [`MAP_OPTIMIZATION_REPORT_2025_11_16.md`](dump/siglip_reports/MAP_OPTIMIZATION_REPORT_2025_11_16.md), [`SCENE_COVERAGE_ANALYSIS_2025_11_16.md`](dump/siglip_reports/SCENE_COVERAGE_ANALYSIS_2025_11_16.md)

> Use this summary as the canonical entry point. Each linked document retains the full raw logs, math, and code listings for deeper dives.
