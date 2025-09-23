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

Next Actions
1) Create the shared environment from `Algorithm1_env.yml` (optional but recommended).
2) Update or clone the notebook (`algorithm1.ipynb`) to reuse the helper functions if interactive analysis is needed.
3) Execute the refreshed pipeline on a larger slice using the new `--ssam-freq` / `--sam2-max-propagate` knobs to validate runtime savings. ✅ small-scale verification pending.
4) Push code to https://github.com/RICHKUNG/3DIS when credentials are ready.

Open Items
- Confirm the initial MultiScan scene(s) for full processing beyond the demo slice.
- Tuning knobs: `min_area`, `stability_threshold`, and SAM2 IoU threshold (currently 0.6).

GitHub
- Initialize/push sequence from `My3DIS/`:
  - `git init && git checkout -b main`
  - `git remote add origin https://github.com/RICHKUNG/3DIS.git`
  - `git add Agent.md run_pipeline.py Algorithm1_env.yml algorithm1.ipynb`
  - `git commit -m "Init: multi-level Semantic-SAM → SAM2 pipeline"`
  - `git push -u origin main`
