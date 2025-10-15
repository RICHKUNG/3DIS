# Workflow Submodule Guide (Step 4)

> The workflow package parses YAML configurations, resolves which scenes to run, executes each stage sequentially or in parallel, and records output summaries. The sections below follow that order: configuration parsing → scene selection → single-scene execution → multi-scene scheduling → summary/export helpers.

## Execution Diagram (YAML Mode)

```
python -m my3dis.run_workflow --config <CONFIG>
    └─ main()
       ├─ _record_pid_map(config_path)
       └─ execute_workflow(config, override_output, config_path)
             ├─ parse experiment / stages (YAML already loaded via io.load_yaml)
             ├─ scenes.normalize_scene_list() + path expansion
             ├─ (optional) initialise memory-event monitoring
             └─ for each scene:
                   ▼ run_scene_workflow(...)
                     └─ SceneWorkflow.run()
                        ├─ StageRecorder('ssam')
                        │   └─ my3dis.generate_candidates.run_generation()
                        │       └─ ssam_progressive_adapter.generate_with_progressive()
                        │           └─ progressive_refinement.progressive_refinement_masks()
                        ├─ StageRecorder('filter') [if enabled]
                        │   └─ my3dis.filter_candidates.run_filtering()
                        ├─ StageRecorder('tracker') [if enabled]
                        │   └─ my3dis.track_from_candidates.run_tracking()
                        │       └─ tracking.level_runner.run_level_tracking() × each level
                        │           └─ tracking.sam2_runner.sam2_tracking()
                        ├─ StageRecorder('report') [if enabled]
                        │   └─ my3dis.generate_report.build_report()
                        └─ Finalise
                            ├─ summary.export_stage_timings()
                            ├─ summary.apply_scene_level_layout()
                            └─ summary.append_run_history()
```

## Step 4-0 – Public Exports

### `__init__.py`
- Re-exports key classes and helpers (`SceneWorkflow`, `execute_workflow`, etc.) for `run_workflow.py`.

### `core.py`
- Thin compatibility layer that forwards `SceneContext`, `SceneWorkflow`, `run_scene_workflow`, and `execute_workflow` to preserve the legacy import surface.

## Step 4-1 – Configuration & Error Types

### `errors.py`
- `WorkflowError` – base class for workflow-related failures.
- `WorkflowConfigError` – raised when the configuration file is invalid.
- `WorkflowRuntimeError` – raised for I/O or runtime issues during execution.

### `io.py`
- `load_yaml(path)` reads YAML files, enforces a mapping at the top level, and raises `WorkflowConfigError` when the file is missing or malformed.

### `utils.py`
- `_collect_gpu_tokens(spec)` normalises GPU specifications (string or sequence) into a cleaned token list.
- `normalise_gpu_spec(gpu)` converts GPU specifications to a list of integer indices while dropping duplicates and negatives.
- `serialise_gpu_spec(gpu)` formats GPU indices back into comma-separated strings for environment variables.
- `using_gpu(gpu)` is a context manager that temporarily sets `CUDA_VISIBLE_DEVICES` so concurrent stages do not step on each other.
- `now_local_iso()` / `now_local_stamp()` provide local-time ISO strings and filename-safe timestamps.

## Step 4-2 – Scene Resolution & Path Expansion

### `scenes.py`
- `expand_output_path_template(path_value, experiment_cfg)` expands `{name}` templates inside `experiment.output_root`, raising when the name is missing.
- `discover_scene_names(dataset_root)` finds scene directories under the dataset root (prefers `scene_*` naming).
- `normalize_scene_list(raw_scenes, dataset_root, scene_start, scene_end)` handles lists, single tokens, `all`, and range filters while validating existence.
- `resolve_levels(stage_cfg, manifest, fallback)` picks level lists per stage, preferring explicit stage overrides, then manifests, then experiment defaults.
- `stage_frames_string(stage_cfg, experiment_cfg=None)` merges experiment defaults with stage overrides to produce the canonical `start:end:step` string.
- `resolve_stage_gpu(stage_cfg, default_gpu)` yields the GPU specification for a stage (falling back to experiment-wide settings).
- `derive_scene_metadata(data_path)` infers scene names and dataset roots from paths and injects them into the workflow summary.

## Step 4-3 – Execution Records & Summary Helpers

### `summary.py`
- `_bytes_to_mib(value)` converts byte counts to MiB for human-readable reporting.
- `_normalise_gpu_indices(spec)` wraps `normalise_gpu_spec` for resource monitoring.
- `_gather_git_snapshot()` captures current git commit/branch/dirty state when the repo is under version control.
- `collect_environment_snapshot()` collects Python, platform, CUDA, torch, numpy, and environment details for `environment_snapshot.json`.
- `StageResourceMonitor` tracks CPU/GPU peaks on a background thread.
  - `__init__(stage_name, gpu_spec, poll_interval)` configures monitoring.
  - `start()` launches psutil samples and GPU memory polling.
  - `stop()` stops sampling and returns collected metrics.
  - `_setup_cpu_monitor()`, `_collect_process_rss()`, `_update_cpu_peak()`, `_poll_cpu_usage()` handle CPU memory/time sampling.
  - `_setup_gpu_monitor()`, `_safe_cuda_call(fn, device)`, `_finalise_gpu_metrics()` gather CUDA metrics safely.
  - `_build_cpu_summary()` formats CPU metrics for summaries.
- `StageRecorder(summary, name, gpu)` records start/end timestamps, exceptions, and resource usage through a context manager.
  - `__post_init__()` normalises GPU settings.
  - `__enter__()` registers stage order, start time, and kicks off resource monitoring.
  - `__exit__(...)` records completion, duration, exceptions, and metrics.
- `export_stage_timings(summary, output_path)` writes per-stage timing JSON for reports or external analysis.
- `update_summary_config(summary, config)` embeds a snapshot of the original configuration into the summary.
- `load_manifest(run_dir)` safely loads `manifest.json`, returning `None` on failure.
- `_lock_file_handle(handle)` / `_unlock_file_handle(handle)` provide cross-platform file locks so run history updates remain atomic.
- `append_run_history(summary, manifest, history_root)` appends this run to `logs/workflow_history.csv`.
- `_move_file(src, dst)` moves files with error handling when reorganising output.
- `_load_json_if_exists(path)` reads JSON if present (returns `None` otherwise), used to collect `stage_timings.json`.
- `apply_scene_level_layout(run_dir, summary, manifest)` rearranges per-level outputs, promotes tracking artifacts, and rewrites manifests/summaries accordingly.

### `logging.py`
- `build_completion_log_entry(...)` assembles status messages (including scene/error summaries) suitable for append-only log files.
- `log_completion_event(status, config_path, message)` appends completion notifications to `logs/workflow_notifications.log`.

## Step 4-4 – Single Scene Execution

### `scene_workflow.py`
- `SceneContext` encapsulates the raw configuration, resolved paths, and metadata required to execute a scene.
- `SceneWorkflow`
  - `__init__(context)` builds the summary skeleton and configures output locations.
  - `run()` drives the four stages (Semantic-SAM → Filter → Tracker → Report) and returns the summary dictionary.
  - `_stage_cfg(name)` retrieves per-stage configuration blocks.
  - `_stage_summary(name)` initialises or fetches the per-stage section inside the summary.
  - `_determine_layout_mode()` inspects `experiment.output_layout`.
  - `_populate_experiment_metadata()` enriches the summary with dataset/scene metadata, including experiment-level parents.
  - `_ensure_run_dir()` validates/creates the Semantic-SAM run directory when reusing prior outputs.
  - `_ensure_manifest()` loads and caches the manifest emitted by the Semantic-SAM stage.
  - `_run_ssam_stage()` orchestrates candidate generation, supporting `persist_raw`, `skip_filtering`, mask downscaling, and reuse of existing runs.
  - `_run_filter_stage()` re-applies filtering when enabled.
  - `_run_tracker_stage()` calls `track_from_candidates.run_tracking` with prompt modes, downscale ratios, comparison sampling, and render flags.
  - `_run_report_stage()` invokes report generation, honours `report_name` / `max_width`, and optionally emits `stage_timings.json`.
  - `_finalize()` drops `environment_snapshot.json`, writes `workflow_summary.json` (including stage resources), applies layout changes, and appends run history.
- `run_scene_workflow(**kwargs)` convenience wrapper that constructs `SceneWorkflow` and returns its `run()` result.

## Step 4-5 – Multi-scene Scheduling & Parallelism

### `executor.py`
- `_SceneJob` dataclass describes a single scene job with identifiers and `SceneContext` constructor arguments.
- `_run_scene_job(job)` executes a scene inside the main process (for thread/process pools).
- `_scene_job_worker(conn, job)` worker entrypoint for spawned processes; sends results or errors through a pipe.
- `_run_scene_job_isolated(job)` spawns a dedicated process, waits, and returns `(success, payload)`.
- `_resolve_path_override(env_var, configured)` allows environment variables to override configured dataset/output roots.
- `_prepare_memory_event_readers(paths)` loads `memory.events` watchers and reports which ones are usable.
- `_read_memory_snapshots(readers)` reads current OOM counters.
- `_detect_memory_events(readers, previous, current)` compares counters to detect OOM occurrences.
- `_format_oom_events(events)` renders OOM events into log-friendly strings.
- `execute_workflow(config, override_output, config_path, memory_event_paths)` is the primary entrypoint: parse YAML → resolve scene list → run jobs serially or in parallel → return run summaries along with OOM detection state and timestamps.

## Step 4-6 – Auxiliary Data

- `logs/` – the workflow package appends summaries to `logs/workflow_history.csv`; no executable code lives here.

---

Need a broader context? See `src/my3dis/OVERVIEW.md`.  
Want the tracking internals? Head to `tracking/TRACKING_GUIDE.md`.
