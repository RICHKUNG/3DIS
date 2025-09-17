3DIS Pipeline Log (Semantic-SAM × SAM2)

Reference
- All pipeline goals, environment details, dataset policy, and execution guidance now live in `README.md`.
- This file tracks decisions, progress, and outstanding work items.

Status
- Plan agreed: levels [2,4,6], frame slice 1200:1600:20, quick demos use the first three sampled frames.
- `run_pipeline.py` still supports single-environment runs with baked-in checkpoints, but the default workflow now swaps between dedicated Semantic-SAM and SAM2 conda envs.
- `run_experiment.sh` coordinates the two-stage execution (generate → track); dataset/output paths are fixed in the script so the CLI only adjusts levels, frame ranges, and thresholds.
- Per-level outputs must persist raw and filtered candidate lists alongside SAM2 tracking artifacts.

Progress Log
- Added `My3DIS/run_pipeline.py` to glue Semantic-SAM candidate generation with SAM2 tracking.
- Added `generate_candidates.py` (Semantic-SAM stage) and `track_from_candidates.py` (SAM2 stage) for two-environment execution.
- Added `run_experiment.sh` to orchestrate stage execution across environments with sensible defaults.
- Completed demo run on first three frames of scene_00065_00 via the two-stage path; results saved under `My3DIS/outputs/scene_00065_00/demo`.
- README updated with consolidated operational guidance from the earlier planning document, plus the fixed-path orchestrator workflow.

Next Actions
1) Create the shared environment from `Algorithm1_env.yml` (optional but recommended).
2) Update or clone the notebook (`algorithm1.ipynb`) to reuse the helper functions if interactive analysis is needed.
3) Execute the full frame selection via `./run_experiment.sh` once resource scheduling is available.
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
