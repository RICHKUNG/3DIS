# My3DIS Core Modules (Execution Flow)

> This document walks through the modules under `src/my3dis` in the order they are executed during a run. For deep dives into the tracking subsystem and the YAML workflow orchestrator, see `tracking/TRACKING_GUIDE.md` and `workflow/WORKFLOW_GUIDE.md`.

## Execution Diagram (Modes)

```
                   ┌──────────────────────────────────────┐
                   │            Execution Modes            │
                   └──────────────────────────────────────┘
                         │                     │
              YAML orchestrator           Two-stage CLI
               (multi-scene)             (single scene / ad hoc)
                         │                     │
            python -m my3dis.run_workflow     ├─────▶ Semantic-SAM candidates
                         │                     │          generate_candidates.run_generation
          ▼ execute_workflow(...)              │          ├─ ssam_progressive_adapter.generate_with_progressive
              for scene in scenes              │          │   └─ semantic_refinement.progressive_refinement_masks
              ▼ SceneWorkflow.run()            │          └─ raw_archive.persist_raw_frame / filtered.json
              ├─ SSAM → generate_candidates    │
              ├─ Filter → filter_candidates    │          ▶ SAM2 tracking:
              ├─ Tracker → track_from_candidates          track_from_candidates.run_tracking
              ├─ Report → generate_report                 ├─ prepare_tracking_context / ensure_subset_video
              └─ Finalise → apply_scene_level_layout      ├─ level_runner.run_level_tracking (per level)
                         │                                │    ├─ candidate_loader.iter_candidate_batches
                         │                                │    ├─ sam2_runner.sam2_tracking
                         │                                │    │   ├─ _prepare_prompt_candidates
                         │                                │    │   ├─ _filter_new_candidates (DedupStore)
                         │                                │    │   ├─ _add_prompts_to_predictor
                         │                                │    │   └─ _propagate_frame_predictions
                         │                                │    ├─ persist_level_outputs
                         │                                │    │   ├─ outputs.build_video_segments_archive
                         │                                │    │   └─ outputs.build_object_segments_archive
                         │                                │    └─ outputs.save_comparison_proposals (optional)
                         │                                └─ update_manifest
```

## Step 0 – Shared Environment & Utilities

### `__init__.py`
- Exposes `__version__`, which reports the installed package version or falls back to `0.0.0` when running from source.

### `common_utils.py`
- `ensure_dir(path)` creates an output directory (including parents) and returns the absolute path; every stage uses it before writing artifacts.
- `format_duration(seconds)` converts raw seconds into `MM:SS` / `HH:MM:SS` strings for logs and reports.
- `encode_mask(mask)` / `pack_binary_mask(mask, full_resolution_shape)` compress boolean masks into JSON-friendly blobs reused across the pipeline.
- `downscale_binary_mask(mask, ratio)` applies a box filter downsampling used by the tracker and visualisation layers.
- `is_packed_mask(entry)` / `unpack_binary_mask(entry)` detect and restore packed masks so each stage can read/write consistently.
- `numeric_frame_sort_key(fname)` sorts frame names by numeric index to avoid lexicographic drift.
- `list_to_csv(values)` converts a value list into the comma-separated string stored in manifests and logs.
- `parse_levels(levels)` normalises CLI/YAML level declarations into integer lists.
- `parse_range(range_str)` parses `start:end:step` strings into Python slice tuples for frame sampling.
- `bbox_from_mask_xyxy(mask)` / `bbox_xyxy_to_xywh(bounds)` translate between mask-derived boxes and SAM/SAM2 expectations.
- `build_subset_video(frames_dir, selected, selected_indices, out_root, folder_name)` prepares the sampled frame subset (preferring symlinks) shared by Semantic-SAM and SAM2.
- `setup_logging(explicit_level, env_var, logger_names_to_quiet)` configures the root logger and silences noisy third-party loggers.
- `configure_entry_log_format(explicit_level)` applies PID/timestamp formatting to CLI entrypoints such as `run_workflow.py`.

### `pipeline_defaults.py`
- `_path_from_env(env_var, fallback)` resolves repository, checkpoint, dataset, or output paths with environment-variable overrides.
- `expand_default(path)` turns the built-in defaults into absolute strings even when the path does not exist yet.

## Step 1 – Semantic-SAM Candidate Generation

### `generate_candidates.py`
- `_coerce_packed_mask(entry)` normalises incoming masks into the packed format shared by metadata and NPZ archives.
- `_mask_to_bool(entry)` expands packed masks back to boolean arrays when needed (e.g., for gap filling).
- `_coerce_union_shape(mask, target_shape)` resizes masks to a shared shape so unions and fill operations can succeed.
- `persist_raw_frame(level_root, frame_idx, frame_name, candidates, chunk_writer)` writes metadata + NPZ payloads for each frame following the pipeline layout.
- `configure_logging(explicit_level)` initialises the module logger and suppresses verbose Semantic-SAM output.
- `run_generation(...)` drives frame sampling, calls the progressive adapter, populates `level_*` directories, and builds manifests/statistics—this is the Stage 1 entrypoint.
- `main()` exposes a CLI wrapper around `run_generation`.

### `ssam_progressive_adapter.py`
- `_semantic_sam_workdir()` temporarily switches into the Semantic-SAM repository to satisfy relative imports.
- `_extract_gap_components(segs, fill_area)` identifies uncovered regions above the gap-fill threshold.
- `generate_with_progressive(...)` wraps `progressive_refinement_masks`, yields per-level candidates, synthesises gap-fill masks, and returns packed payloads ready for persistence.

### `progressive_refinement.py` (compatibility layer)
- Emits deprecation warnings but re-exports the public API from `semantic_refinement`; legacy imports continue to work for now.

### `semantic_refinement.py` (core implementation)
- `progressive_refinement_masks()` implements the progressive refinement algorithm.
- `create_masked_image()` / `prepare_image_from_pil()` generate masked overlays without spilling intermediate files.
- `setup_output_directories()` / `save_original_image_info()` standardise output layout and preserve source metadata.
- `bbox_from_mask()` / `instance_map_to_color_image()` provide helper transformations for inspection.
- Child masks are intersected with their parent segmentation so refinements stay within bounds; duplicates identical to the parent are discarded.

### `semantic_refinement_cli.py`
- `parse_args()` covers single-image or scene batches, checkpoint overrides, and gap-fill tuning.
- `main()` wires logging, model loading, manifest/index creation, and finally calls the core refinement routine.

### `raw_archive.py`
- `_PendingFrame` buffers metadata, mask bytes, and statistics for deferred writes.
- `RawCandidateArchiveWriter` converts per-frame candidates into chunked tar archives to avoid millions of small files.
  - `__init__(level_root, chunk_size, compression)` initialises output directories and buffering policy.
  - `__enter__()` / `__exit__()` support context-manager usage and flush pending chunks automatically.
  - `manifest_path` exposes the final manifest location.
  - `add_frame(...)` accumulates frame payloads and triggers chunk flushes when thresholds are met.
  - `close()` writes the manifest and clears all buffers.
- `RawCandidateArchiveReader` mirrors the writer and understands both the chunked format and the legacy JSON/NPZ layout.
  - `has_manifest()` checks for chunked manifests.
  - `frame_indices()` lists available frame indices.
  - `load_frame(frame_idx)` restores metadata and mask stacks for downstream filtering or reprocessing.
  - `_legacy_frame_indices` / `_legacy_load_frame` provide fallbacks for older runs.

## Step 1b – Candidate Re-filtering

### `filter_candidates.py`
- `FilterStats` accumulates kept/dropped counts per frame and exposes a JSON serialisable view.
- `bbox_from_mask(mask)` rebuilds XYXY bounding boxes compatible with the progressive refinement output.
- `filter_level(...)` reloads the raw archive, applies area/stability thresholds, and emits `filtered.json`.
- `run_filtering(...)` iterates the configured levels, optionally updating manifests when thresholds change.
- `parse_args()` / `main()` expose a CLI that re-runs filtering without regenerating Semantic-SAM candidates.

## Step 2 – SAM2 Tracking

> `track_from_candidates.py` orchestrates the tracking stage. For the internals of the `tracking` package see `tracking/TRACKING_GUIDE.md`.

### `track_from_candidates.py`
- `resolve_sam2_config_path(config_arg)` interprets CLI arguments and produces Hydra-style SAM2 config paths or absolute files.
- `configure_logging(explicit_level)` sets up module-level logging.
- `run_tracking(...)` merges manifests, loads the SAM2 predictor, calls `tracking.level_runner.run_level_tracking` for each level, and collects artifacts/warnings—this is the Stage 2 entrypoint.
- `main()` provides the standalone SAM2 CLI.

## Step 3 – Reporting & Visualisation

### `generate_report.py`
- `StageTiming` wraps per-stage durations and exposes a human-readable `duration_text`.
- `load_json(path)` performs tolerant JSON reads.
- `collect_stage_timings(summary)` pulls stage timings from the workflow summary and sorts them.
- `pick_first_mid_last(items)` keeps the first/middle/last examples to avoid bloated reports.
- `downscale_image(src, dst, max_width)` resizes preview images before embedding them in Markdown.
- `render_level_section(...)` renders per-level tables alongside image previews.
- `build_report(run_dir, report_name, max_preview_width)` ties everything together—this is the Stage 3 entrypoint.
- `parse_args()` / `main()` expose the CLI for any completed run directory.

## Step 4 – Workflow Orchestration (Multi-scene)

> Detailed behaviour of the workflow package (scene scheduling, parallelism, summaries) lives in `workflow/WORKFLOW_GUIDE.md`. This section documents the public entrypoint.

### `run_workflow.py`
- `_configure_entry_logging()` ensures CLI logs include timestamps and PIDs.
- `_stdout_descriptor()` inspects the current stdout target (file vs pipe) so PID maps stay useful.
- `_record_pid_map(config_path)` appends the invocation metadata to `logs/run_pid_map.tsv`.
- `main()` parses `--config`, `--dry-run`, and related flags, loads YAML, calls `workflow.execute_workflow`, and wraps execution with the OOM monitor helpers.

## Step 5 – Auxiliary Tools

### `prepare_tracking_run.py`
- `build_destination(source, dest_root, name)` chooses the destination folder name (supporting timestamps or explicit overrides).
- `clone_directory(source, dest, mode)` clones a run via hard links or deep copies.
- `update_config(config_path, new_run_dir)` rewrites `experiment.run_dir` inside a YAML config.
- `parse_args()` / `main()` expose the CLI used to duplicate runs or re-run the SAM2 stage.

---

- Tracking module breakdown: `tracking/TRACKING_GUIDE.md`
- Workflow queue and multi-scene execution details: `workflow/WORKFLOW_GUIDE.md`
