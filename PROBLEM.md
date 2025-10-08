# Known Issues and Recommendations

This document aggregates the outstanding risks, maintainability concerns, and optimisation opportunities identified during the latest review.

## High-Priority Fixes (ASAP)

- Avoid global `os.chdir` inside the Semantic-SAM adapter; switch to a scoped context (e.g., `with` + `os.chdir`) or rely on absolute paths to prevent cross-thread side effects. `src/my3dis/ssam_progressive_adapter.py:134`
- Replace hard-coded external paths (dataset, checkpoints, output roots) with environment-variable overrides and document the variables in the README/configs. `src/my3dis/pipeline_defaults.py:8`, `src/my3dis/pipeline_defaults.py:13`, `src/my3dis/pipeline_defaults.py:19`, `src/my3dis/pipeline_defaults.py:20`, `configs/multiscan/base.yaml:3`, `configs/multiscan/base.yaml:25`
- Improve report fallbacks so the Markdown output still shows representative artefacts when comparison images are absent (e.g., surface colored instance maps or list fallback frames). `outputs/experiments/all_1008/scene_00048_01/246_filter_fill_prop30/report.md:11`, `src/my3dis/generate_report.py:117`

## Security & Stability

- Future-proof against the upstream `torch.load(..., weights_only=False)` change and its security implications; ensure local load points opt into safe defaults or handle allowlisting explicitly. `logs/new/run_exp_20251008_144940.log:8`, `logs/new/run_exp_20251008_144940.log:12`
- Multi-scene orchestration uses `ProcessPoolExecutor` while relying on `CUDA_VISIBLE_DEVICES` alone to route GPUs; add scheduling or per-stage GPU assignment to avoid contention/OOM in concurrent runs. `src/my3dis/workflow/executor.py:60`, `src/my3dis/workflow/executor.py:140`
- When OOM watcher reports missing `memory.events`, surface a more actionable warning and consider auto-downgrading concurrency instead of continuing silently. `logs/new/run_exp_20251008_144940.log:2`

## Maintainability & Architecture

- Centralise configuration parsing/validation (levels, frame ranges, GPU hints) into a shared schema to eliminate duplicated logic across modules and scripts. `src/my3dis/common_utils.py:203`, `src/my3dis/common_utils.py:220`, `src/my3dis/progressive_refinement.py:314`, `src/my3dis/progressive_refinement.py:320`, `scripts/pipeline/run_pipeline.py:51`
- Reduce repeated `sys.path` injection blocks by packaging `my3dis` properly (editable install / console scripts) and relying on `python -m` entry points. `src/my3dis/generate_candidates.py:7`, `src/my3dis/track_from_candidates.py:7`, `src/my3dis/filter_candidates.py:7`, `src/my3dis/generate_report.py:7`
- Split `src/my3dis/progressive_refinement.py` into clearer layers (core algorithm vs. CLI vs. visualisation). The current script mixes IO, subprocess Git calls, and heavy logic in a single module. `src/my3dis/progressive_refinement.py:1-260`
- Add manifest entries capturing external repo commit hashes/environment metadata to improve reproducibility (the helper exists but output omits it). `src/my3dis/progressive_refinement.py:152`, `src/my3dis/workflow/scene_workflow.py:385`

## Performance & Resource Management

- Optimise gap-fill union computation by vectorising mask aggregation (e.g., stack + `np.any`) instead of Python loops with per-mask resizing. `src/my3dis/generate_candidates.py:566-612`
- Consider batching or indexing raw candidate dumps to avoid thousands of tiny files that pressure the filesystem (JSON + NPZ per frame). `src/my3dis/generate_candidates.py:214-258`
- Expose visualisation sampling density (every N frames) through config so report generation scales with need. `src/my3dis/tracking/outputs.py:92-107`

## Observability & Reporting

- Capture GPU memory peaks and include them in `stage_timings.json` for easier debugging of resource spikes. `src/my3dis/workflow/summary.py:19-78`
- When the tracker renders no comparisons, emit a structured warning (and report fallback) instead of a bare console print. `src/my3dis/tracking/outputs.py:252-253`
- Persist the active environment snapshot (Python/Torch/CUDA versions) into workflow summaries for downstream audits. `src/my3dis/workflow/scene_workflow.py:385`, `src/my3dis/workflow/summary.py:38`

## Tooling & UX

- Allow `run_experiment.sh` to pick up defaults from environment variables or a `.env` file to better support CI/multi-machine setups. `run_experiment.sh:17-24`
- Document the new environment overrides and the recommended workflow in the README once implemented. `README.md:1-120`

