# Tracking Submodule Guide (Step 2)

> The tracking package covers everything that happens after Semantic-SAM filtering: loading candidates, preparing prompts, invoking SAM2, persisting artifacts, and generating previews. Modules are documented in execution order.

## Tracking Stage Flow (Per Level)

```
run_tracking(...) ──────────────────────────────────────────────────────────────┐
  ├─ context = prepare_tracking_context(candidates_root, levels, ...)          │
  ├─ predictor = build_sam2_video_predictor(...)                               │
  ├─ subset_dir, subset_map = ensure_subset_video(context, data_path, output)  │
  ├─ long_tail_area = resolve_long_tail_area_threshold(context.manifest, ...)  │
  └─ for level in context.level_list:                                          │
       run_level_tracking(level, ...) ───────────────────────────────────┐     │
         ├─ filtered_manifest = load_filtered_manifest(level_root)       │     │
         ├─ frames_meta = filtered_manifest['frames']                    │     │
         ├─ preview_indices = select_preview_indices(...)                │     │
         ├─ candidate_iter = iter_candidate_batches(...)                 │     │
         ├─ dedup_store = DedupStore()                                   │     │
         ├─ frame_store = FrameResultStore()                             │     │
         └─ sam2_tracking(subset_dir, predictor, candidate_iter, ...)    │     │
               ├─ state = predictor.init_state(video_path=subset_dir)    │     │
               ├─ for each FrameCandidateBatch                           │     │
               │    ├─ prepared = _prepare_prompt_candidates(...)        │     │
               │    ├─ filtered = _filter_new_candidates(...)            │     │
               │    ├─ _add_prompts_to_predictor(...)                    │     │
               │    └─ frame_segments = _propagate_frame_predictions(...)│     │
               │         └─ {abs_idx: {obj_id: packed_mask}}             │     │
               │              ├─ dedup_store.add_packed(...)             │     │
               │              └─ frame_store.update(...)                 │     │
               └─ return TrackingArtifacts(object_refs, preview_segments, …)   │
         ├─ artifacts = persist_level_outputs(frame_store, ...)          │     │
         ├─ if render_viz: save_comparison_proposals(...)                │     │
         └─ LevelRunResult(artifacts, comparison, warnings, stats, timer)│     │
  └─ update_manifest(context, level_results, ...)                              │
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

## Step 2-3 – Tracking Context & Manifest Updates

### `pipeline_context.py`
- `TrackingContext` dataclass stores manifest metadata, level list, SSAM cadence, subset video paths, and propagation limits.
- `prepare_tracking_context(...)` loads `manifest.json`, normalises indices, and reconciles `sam2_max_propagate` between CLI and manifest values.
- `resolve_long_tail_area_threshold(...)` picks the area cut-off for long-tail box prompts, honouring `MY3DIS_LONG_TAIL_AREA`.
- `ensure_subset_video(...)` recreates the subset frame folder when files are missing or stale, then updates the manifest.
- `update_manifest(...)` writes tracking outputs (artifact relpaths, comparison summaries, warnings, mask scaling info) back to disk.

## Step 2-4 – SAM2 Tracking Core

### `sam2_runner.py`
- `_coerce_mask_bool(...)` and `_prepare_prompt_candidates(...)` turn Semantic-SAM payloads into mask/box prompts with area and bbox metadata.
- `_filter_new_candidates(...)` defers to `DedupStore` to eliminate overlapping prompts per frame based on IoU.
- `_should_use_box_prompt(...)` and `_add_prompts_to_predictor(...)` decide between direct mask prompts, bounding boxes for small objects, or box-only mode.
- `_propagate_frame_predictions(...)` runs forward/backward propagation with optional budgets, packing masks (and downscaling when requested).
- `sam2_tracking(...)` drives the SAM2 predictor under autocast, keeps per-frame results in `FrameResultStore`, tracks object references, and returns a `TrackingArtifacts` bundle.

## Step 2-5 – Level Execution & Output Packaging

### `level_runner.py`
- `LevelRunResult` carries artifacts, comparison metadata, warnings, timing, and summary stats for a level.
- `run_level_tracking(...)` coordinates candidate loading, preview sampling, SAM2 tracking, persistence, and optional visualisation renders.
- `persist_level_outputs(...)` packages frame-major/object-major NPZ archives and cleans up temporary frame stores.
- Comparison rendering is handled through `outputs.save_comparison_proposals(...)`, with warnings surfaced in `LevelRunResult.warnings`.

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
