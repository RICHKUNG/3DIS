# My3DIS Risk & Follow-up Tracker

This note records the open risks, verification tasks, and mid-term improvements after the latest streaming tracker refactor. Items are organised by priority and theme so we can revisit them sprint by sprint.

## Recently Shipped

- Added environment overrides for all external paths (repos, checkpoints, datasets, outputs) to avoid hard-coded mounts. `src/my3dis/pipeline_defaults.py:9-52`, `src/my3dis/workflow/executor.py:32-108`
- Reworked SAM2 tracking to stream results: down-sampled IoU dedup is kept in memory while full-resolution payloads are written per frame/object through manifests. `src/my3dis/track_from_candidates.py:93-233`, `src/my3dis/track_from_candidates.py:547-689`, `src/my3dis/track_from_candidates.py:883-998`
- Replaced `np.savez_compressed` bundles with manifest-backed archives (`video_segments*.npz` / `object_segments*.npz`) so large scenes no longer spike RAM when persisting artifacts. `src/my3dis/tracking/outputs.py:81-158`
- Raw SSAM frames now stream into chunked tar archives with a manifest, allowing filter reruns without rehydrating giant JSON blobs. `src/my3dis/raw_archive.py` (`RawCandidateArchiveWriter` / `RawCandidateArchiveReader`), `src/my3dis/generate_candidates.py` (`persist_raw_frame`)
- Tracker comparison previews honour stride/max knobs (`tracker.comparison_sampling`) and fall back to structured warnings when nothing renders. `src/my3dis/workflow/scene_workflow.py` (`_run_tracker_stage`), `src/my3dis/tracking/outputs.py` (`save_comparison_proposals`)
- Stage execution records CPU/GPU peaks and drops `environment_snapshot.json` + `workflow_summary.json` snapshots for each run. `src/my3dis/workflow/summary.py` (`StageResourceMonitor`, `StageRecorder`), `src/my3dis/workflow/scene_workflow.py` (`_finalize`)

## Immediate Next Actions

- **E2E verification** – Run one large MultiScan scene with the new streaming pipeline and confirm downstream consumers (`summary`, reporting, notebooks) can read the manifest layout without loading everything into memory. `src/my3dis/track_from_candidates.py` (`run_tracking`), `src/my3dis/tracking/outputs.py` (`build_video_segments_archive`)
- **Manifest tooling** – Provide a CLI/helper to iterate the new archives (list objects, dump masks) to unlock smoke tests and ad-hoc inspections. Current scripts still assume `np.load` on dense dictionaries. `src/my3dis/raw_archive.py`, `tools/` (new)
- **Regression test gap** – Add a unit/integration test that exercises `FrameResultStore` + `build_video_segments_archive` to prevent accidental regressions in the streaming writer. `src/my3dis/tracking/stores.py`, `src/my3dis/tracking/outputs.py`

## Security & Stability

- **URGENT**: Audit every `torch.load` call and opt into `weights_only=True` (or an allowlist) before the upstream default flips. `logs/new/run_exp_20251008_144940.log:8`, `logs/new/run_exp_20251008_144940.log:12`
- Multi-scene execution still maps GPUs via `CUDA_VISIBLE_DEVICES` only; add explicit scheduling or per-stage GPU selection to avoid collisions when `ProcessPoolExecutor` fans out. `src/my3dis/workflow/executor.py:132-215`
- When the OOM watcher reports missing `memory.events`, surface actionable warnings and consider auto-reducing concurrency instead of proceeding silently. `logs/new/run_exp_20251008_144940.log:2`

## Maintainability & Architecture

- **HIGH PRIORITY**: stand up a single `my3dis.config.schema` module (pydantic/dataclass) and route all CLI + workflow config loading through it so overrides, defaults, and validation logic stop drifting. `src/my3dis/common_utils.py:166-191`, `src/my3dis/progressive_refinement.py:314-360`, `scripts/pipeline/run_pipeline.py:51-126`
- **HIGH PRIORITY**: convert the repo into an installable package with console entry points so the orchestration scripts drop ad-hoc `sys.path` mutations and import resolution becomes reproducible. `src/my3dis/generate_candidates.py:1-18`, `src/my3dis/track_from_candidates.py:1-40`, `src/my3dis/filter_candidates.py:1-18`, `src/my3dis/generate_report.py:1-18`
- **MEDIUM PRIORITY**: split `src/my3dis/progressive_refinement.py` (~1000 lines) into `core.py`, `cli.py`, and `viz.py` (or similar) to isolate the algorithm from presentation code and make unit coverage feasible. `src/my3dis/progressive_refinement.py:1-260`
- Fix: inject commit hashes and env metadata into manifest headers with a tolerant fallback path so downstream runs can reproduce a scene without manual bookkeeping. `src/my3dis/progressive_refinement.py:152-188`, `src/my3dis/workflow/scene_workflow.py:360-410`

## Performance & Resource Management

- Fix: vectorise the gap-fill union (`np.stack` + `np.any`) with preallocated buffers to kill the Python resize loop and shave peak RSS during dense scenes. `src/my3dis/generate_candidates.py` (`_coerce_union_shape`)
- Add knobs for raw archive chunk sizing/compression (currently fixed to 32 frames, gzip) so long sequences can trade throughput vs disk usage. `src/my3dis/raw_archive.py`, `src/my3dis/generate_candidates.py`
- Fix: wire the OOM watcher into pipeline orchestration so when memory.events are missing or `oom_kill` increments we automatically back off concurrency. `logs/new/run_exp_20251008_144940.log:2`, `src/my3dis/workflow/executor.py:132-215`

## Observability & Reporting

- Surface `StageResourceMonitor` stats inside generated Markdown reports and CLI summaries so peaks are visible outside JSON payloads. `src/my3dis/generate_report.py`, `src/my3dis/workflow/summary.py`
- Provide a lightweight `report resources` helper that prints recent `workflow_summary.json` metrics for quick regressions (no spreadsheet wrangling). `src/my3dis/workflow/summary.py`, `tools/` (new)

## Tooling & UX

- Allow `run_experiment.sh` to read defaults from environment variables or a `.env` file so CI / multi-machine runs do not need local edits. `run_experiment.sh:17-24`
- Expand the README section on streaming outputs with a short “How to read manifests” snippet once verification is complete. `README.md:18-120`
