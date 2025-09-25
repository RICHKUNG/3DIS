3DIS Pipeline Log (Semantic-SAM × SAM2)

Reference
- All pipeline goals, environment details, dataset policy, and execution guidance now live in `README.md`.
- This file tracks decisions, progress, and outstanding work items.

Status
- Plan agreed: levels [2,4,6], frame slice 1200:1600:20, quick demos use the first three sampled frames.
- `run_pipeline.py` now reuses the same candidate/track modules as `run_experiment.sh`, so a single-environment run yields identical artifacts without duplicating logic; the default workflow still swaps between dedicated Semantic-SAM and SAM2 conda envs.
- `run_experiment.sh` coordinates the two-stage execution (generate → track); dataset/output paths are fixed in the script so the CLI only adjusts levels, frame ranges, thresholds, Semantic-SAM cadence, and SAM2 propagation depth.
- Progressive refinement now back-fills large uncovered regions at the coarsest level, so SAM2 always receives explicit masks for gaps that exceed `min_area`.
- Per-level outputs must persist raw and filtered candidate lists alongside SAM2 tracking artifacts.
- YAML 驅動的 `run_workflow.py` 可依配置自動執行 SSAM → filter → SAM2 → 報告，並在 YAML 中指定 GPU/參數、產出 Markdown 紀錄。

Progress Log
- Added `My3DIS/run_pipeline.py` to glue Semantic-SAM candidate generation with SAM2 tracking.
- Added `generate_candidates.py` (Semantic-SAM stage) and `track_from_candidates.py` (SAM2 stage) for two-environment execution.
- Added `run_experiment.sh` to orchestrate stage execution across environments with sensible defaults.
- Completed demo run on first three frames of scene_00065_00 via the two-stage path; results saved under `My3DIS/outputs/scene_00065_00/demo`.
- README updated with consolidated operational guidance from the earlier planning document, plus the fixed-path orchestrator workflow.
- 2025-09-18: Unified pipeline code paths (run_pipeline → generate_candidates/track_from_candidates, progressive candidates now honor stability thresholds) and executed `./run_experiment.sh --levels "1,3,5" --frames "1200:1600:20" --min-area 400 --stability 0.8`; full run finished in ~22m with outputs under `My3DIS/outputs/scene_00065_00/20250918_095823`.
- 2025-09-18: Investigated why `python run_pipeline.py` is slower than `./run_experiment.sh`; root cause is single-environment execution missing Semantic-SAM/SAM2 specific CUDA builds and the combined process double-buffering both models in memory. Recommendation kept: prefer `run_experiment.sh` or reintroduce per-stage environments when running the Python orchestrator directly.
- 2025-09-22: Added `--ssam-freq` to throttle Semantic-SAM usage and record the subset of frames that actually run segmentation; `generate_candidates.py` now only builds artifacts for those frames.
- 2025-09-22: Added SAM2 propagation cap (`--sam2-max-propagate`) throughout CLI/orchestrators and reworked tracking calls to honor frame budgets without hitting the legacy API signature.
- 2025-09-22: Implemented gap-fill mask synthesis in `ssam_progressive_adapter.py` so uncovered regions larger than `min_area` enter downstream filtering/tracking even if Semantic-SAM misses them in the first pass.
- 2025-09-23: Streamlined persistence—progressive refinement now runs inside temp dirs (no `_progressive_tmp` artifacts), filtered masks are packed into `filtered.json`, tracking objects emit JSON-only metadata, and viz renders only keep the `compare/` panels.
- 2025-09-24: Reviewed recent SAM2 tracking failures; noted ~5k mask prompts per long run (logs `nohupGPU0.out`/`nohupGPU1.out`) which translate to ~24–36 GB of GPU tensor memory because each mask prompt persists as a 1024² float tensor. Advised throttling mask additions (filtering, chunked propagation, or converting to boxes) to stay within GPU limits.
- 2025-09-25: Added CLI overrides for SAM2 IoU threshold and box-prompt policies (`run_experiment.sh` → `track_from_candidates.py`); long-tail objects default to area ≤ max(3×min_area, min_area+1) with `MY3DIS_LONG_TAIL_AREA` override, and scripts log the active prompt strategy.
- 2025-09-26: 建立 `configs/scene_00065_00.yaml` 與 `run_workflow.py`，將 SSAM/Filter/SAM2/報告拆成 stage，可在 YAML 中調整 level、frame freq、SSAM freq、GPU 配置；新增 `filter_candidates.py`、`generate_report.py` 支援重複篩選與 Markdown 報告（含每層第一/中位/最後 compare 圖、時間摘要）。
- 2025-09-26: Refined tracking artifacts—gap-fill僅在第一個 level 啟用、SAM2 僅輸出 `video_segments.npz` / `object_segments.npz`、viz 比較圖改為每 10 張 SSAM 幀儲存；README / Agent 記錄同步更新。

Next Actions
1) Create the shared environment from `Algorithm1_env.yml` (optional but recommended).
2) Update or clone the notebook (`algorithm1.ipynb`) to reuse the helper functions if interactive analysis is needed.
3) Execute the refreshed pipeline on a larger slice using the new `--ssam-freq` / `--sam2-max-propagate` knobs to validate runtime savings. ✅ small-scale verification pending.
4) Push code to https://github.com/RICHKUNG/3DIS when credentials are ready.

Open Items
- Confirm the initial MultiScan scene(s) for full processing beyond the demo slice.
- Tuning knobs: `min_area`, `stability_threshold`, and SAM2 IoU threshold (currently 0.6).
- Explore safe mask downsampling/quantization once SAM2 propagation completes (ensure reprojection back to RGB-D resolution).

GitHub
- Initialize/push sequence from `My3DIS/`:
  - `git init && git checkout -b main`
  - `git remote add origin https://github.com/RICHKUNG/3DIS.git`
  - `git add Agent.md run_pipeline.py Algorithm1_env.yml algorithm1.ipynb`
  - `git commit -m "Init: multi-level Semantic-SAM → SAM2 pipeline"`
  - `git push -u origin main`
