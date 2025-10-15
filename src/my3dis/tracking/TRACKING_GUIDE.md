# Tracking Submodule Guide (Step 2)

> The tracking package covers everything that happens after Semantic-SAM filtering: loading candidates, preparing prompts, invoking SAM2, persisting artifacts, and generating previews. Modules are documented in execution order.

## Tracking Stage Flow (Per Level)

```
run_level_tracking(level, ...) ─────────────────────────────────────────────┐
  ├─ load_filtered_manifest(level_root)                                    │
  ├─ frames_meta = filtered.json['frames']                                 │
  ├─ preview_local_indices = select_preview_indices(...)                   │
  ├─ candidate_iter = iter_candidate_batches(level_root, frames_meta, ...) │
  ├─ dedup_store = DedupStore()                                            │
  ├─ frame_store = FrameResultStore()                                      │
  └─ sam2_tracking(subset_dir, predictor, candidate_iter, ...)             │
       ├─ predictor.init_state(video_path=subset_dir)                      │
       ├─ for each FrameCandidateBatch:                                    │
       │    ├─ prepared = _prepare_prompt_candidates(batch.candidates)     │
       │    ├─ filtered = _filter_new_candidates(prepared, dedup_store)    │
       │    ├─ _add_prompts_to_predictor(predictor, state, filtered, ...)  │
       │    └─ frame_segments = _propagate_frame_predictions(...)          │
       │         └─ result: {abs_idx: {obj_id: packed_mask}}               │
       │              ├─ dedup_store.add_packed(abs_idx, ...)              │
       │              └─ frame_store.update(abs_idx, frame_name, ...)      │
       └─ return TrackingArtifacts(object_refs, preview_segments, ...)     │

  ├─ artifacts = persist_level_outputs(..., frame_store, ...)              │
  │    ├─ build_video_segments_archive(iter_frames(), video_segments.npz)  │
  │    └─ build_object_segments_archive(object_refs, object_segments.npz)  │
  ├─ if render_viz:                                                        │
  │    └─ save_comparison_proposals(viz_dir, base_frames_dir, ...)         │
  └─ return LevelRunResult(artifacts, comparison, warnings, stats, timer)  │
```

## Step 2-0 – Exports & Shared Helpers

### `__init__.py`
- Re-exports timing helpers, bounding-box utilities, and output wrappers so `track_from_candidates.py` can import them in one place.

### `helpers.py`
- `ProgressPrinter`
  - `__init__(total)` initialises the lightweight progress bar.
  - `update(index, abs_frame)` refreshes the current progress including absolute frame indices.
  - `close()` emits a newline so subsequent logs are not overwritten.
- `_TimingSection` powers `TimingAggregator.track()` as a context manager that measures durations.
  - `__enter__()` / `__exit__()` accumulate elapsed time under the chosen stage name.
- `TimingAggregator`
  - `__init__()` creates empty storage.
  - `add(stage, duration)` aggregates time for a stage.
  - `track(stage)` returns a `_TimingSection` for use with `with`.
  - `total(stage)`, `total_prefix(prefix)`, `total_all()` expose totals.
  - `items()` preserves insertion order when iterating `(stage, duration)` pairs.
  - `merge(other)` combines results from another aggregator.
  - `format_breakdown()` renders a human-readable summary string.
- `format_scale_suffix(ratio)` produces filename-friendly scale suffixes such as `_scale0.3x`.
- `scaled_npz_path(path, ratio)` rewrites NPZ paths with the appropriate scale suffix.
- `resize_mask_to_shape(mask, target_shape)` resizes masks (nearest neighbour) so IoU calculations remain consistent.
- `infer_relative_scale(mask_entry)` infers the stored down-sampling ratio from a packed mask.
- `determine_mask_shape(mask_entry, fallback)` returns the original mask shape, required for SAM2 outputs and comparisons.
- `format_duration_precise(seconds)` renders durations with minute/hour precision for logging.
- `bbox_transform_xywh_to_xyxy(bboxes)` converts batches of XYWH boxes to XYXY for SAM2 prompts.
- `bbox_scalar_fit(bboxes, scalar_x, scalar_y)` scales bounding boxes to match SAM2 input resolution.
- `compute_iou(mask1, mask2)` computes IoU (upsampling when needed) for deduplication and QA.

## Step 2-1 – Loading Semantic-SAM Candidates

### `candidate_loader.py`
- `FrameCandidateBatch` bundles per-frame candidates together with local/absolute indices.
- `load_filtered_manifest(level_root)` reads `filtered/filtered.json`.
- `_load_frame_candidates(...)` loads candidates for a frame, optionally rescales masks, and powers both iteration and preview helpers.
- `iter_candidate_batches(...)` yields `FrameCandidateBatch` objects for SAM2 to consume frame-by-frame.
- `load_filtered_frame_by_index(...)` retrieves a single frame by local index for visualisation sampling.
- `_build_preview_segment(...)` and `_build_preview_stub(...)` assemble lightweight preview payloads (`area`, `score`, etc.).

## Step 2-2 – Deduplication & Intermediate Stores

### `stores.py`
- `_frame_entry_name(frame_idx)` standardises frame naming for archives.
- `DedupStore` tracks previously emitted masks to avoid duplicates.
  - `add_packed(abs_idx, obj_id, mask_entry)` registers newly emitted masks.
  - `seen_packed(mask_entry, iou_threshold)` checks if a packed mask overlaps an existing one beyond the IoU threshold.
  - `seen(box, mask_entry, iou_threshold)` performs the same check when raw masks are not available.
- `FrameResultStore` collects per-frame tracking results and preview material.
  - `update(abs_idx, frame_name, packed_masks, preview)` inserts frame outputs.
  - `iter_frames()` yields data for archive builders.
  - `iter_preview_segments()` emits reduced preview payloads for reporting/visualisation.
  - `_iter_sorted(items)` ensures deterministic ordering across runs.
- `TrackingArtifacts` dataclass groups object references, preview segments, and dedup stats for later stages.

## Step 2-3 – Tracking Context & Manifest Updates

### `pipeline_context.py`
- `PipelineContext` holds the filtered manifest, run directory paths, level configuration, and preview sampling parameters.
- `build_pipeline_context(...)` validates inputs and prepares the context before entering the tracking loop.
- `select_preview_indices(...)` determines which frames to use for comparison renders based on stride/max sample settings.
- `update_tracking_manifest(...)` merges tracking outputs back into the manifest (levels, warnings, derived metadata).
- `summarise_tracking_stats(...)` aggregates per-level statistics for inclusion in workflow summaries.

## Step 2-4 – SAM2 Tracking Core

### `sam2_runner.py`
- `_prepare_prompt_candidates(...)` converts Semantic-SAM candidates into SAM2 prompts (mask-first with bounding-box fallback).
- `_filter_new_candidates(...)` drops prompts already seen by the deduplication store.
- `_add_prompts_to_predictor(predictor, state, candidates, prompt_mode)` pushes prompts into the SAM2 predictor according to the configured mode.
- `_propagate_frame_predictions(...)` runs forward/backward propagation with cadence limits and collects packed masks.
- `sam2_tracking(subset_dir, predictor, candidate_iter, ...)` manages SAM2 state, iterates batches, updates stores, and returns `TrackingArtifacts`.
- `_build_tracking_result(...)` fuses dedup statistics, per-frame records, and preview data into the final `TrackingArtifacts`.

## Step 2-5 – Level Execution & Output Packaging

### `level_runner.py`
- `LevelRunResult` dataclass captures artifacts, comparison info, warnings, stats, and timers.
- `run_level_tracking(...)` orchestrates manifest loading, preview selection, candidate iteration, SAM2 tracking, and persistence.
- `_build_level_stats(...)` collects high-level metrics (counts, durations, dedup info) for summaries.
- `_persist_level_outputs(...)` writes NPZ archives, comparison data, and manifest updates.
- `_render_comparison(...)` delegates to `outputs.save_comparison_proposals` when visualisation is enabled.

## Step 2-6 – Output Encoding & Comparison Visuals

### `outputs.py`
- `encode_packed_mask_for_json` / `decode_packed_mask_from_json` convert between JSON payloads and numpy-packed masks.
- `_ensure_frames_dir(path)` creates the directory that hosts NPZ archives.
- `_frame_entry_name(frame_idx)` mirrors the naming rules used by the stores module.
- `_normalize_stride(value)` / `_normalize_max_samples(value)` parse sampling cadence and output limits.
- `_downsample_evenly(values, target)` performs even subsampling while keeping first/last frames.
- `_apply_sampling_to_frames(frames, sample_stride, max_samples)` combines stride and cap logic to pick the final preview frames.
- `build_video_segments_archive(frames, path, mask_scale_ratio, metadata)` writes frame-major NPZ/ZIP archives from `FrameResultStore`.
- `build_object_segments_archive(object_manifest, path, mask_scale_ratio, metadata)` writes object-major archives (object → frame references).
- `save_comparison_proposals(...)` renders Semantic-SAM vs SAM2 overlays:
  - Chooses which frames to render.
  - Locates base imagery (source frames vs sampled subset).
  - Draws masks and bounding boxes.
  - Saves PNGs plus JSON metadata, emits structured warnings when fallbacks are used (consumed by `apply_scene_level_layout` and report generation).

---

To follow the tracking stage in code, read `track_from_candidates.py → pipeline_context.py → level_runner.py → sam2_runner.py → outputs.py`.  
For the full pipeline context or other stages, refer to `src/my3dis/OVERVIEW.md` and `workflow/WORKFLOW_GUIDE.md`.
