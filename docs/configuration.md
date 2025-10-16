# Configuration Reference

> This guide explains how the YAML configuration drives My3DIS runs, which values map to CLI flags, and how environment variables can override defaults.

## Experiment Block

```yaml
experiment:
  name: sample_multiscan_run
  dataset_root: ./data/multiscan
  scenes: null
  parallel_scenes: 3
  levels: [2, 4, 6]
  tag: ssam2_filter_fill
  tag_in_path: false
  aggregate_output: false
  output_layout: scene_level
  output_root: outputs/experiments/{name}
  frames:
    step: 50
```

- `name` – experiment label used inside metadata and, when templated, in output paths.
- `dataset_root` – root directory that contains scene folders (default MultiScan layout).
- `scenes` – `null`, `"all"`, a single string, or a list of scene names. Use `scene_start` / `scene_end` to select a range.
- `parallel_scenes` – maximum number of scenes executed concurrently (bounded by GPU availability).
- `levels` – Semantic-SAM levels to process; individual stages can override this.
- `tag` / `tag_in_path` – optional suffixes appended to manifest metadata and run directories.
- `aggregate_output` – when `true`, stages collapse their results into a shared directory; default `scene_level` keeps per-scene folders.
- `output_layout` – `scene_level` (default) or `flat`. The layout determines how `apply_scene_level_layout` organises artifacts.
- `output_root` – target root for run directories. Supports `{name}` templating.
- `frames` – accepts `start`, `end`, `step` keys mirroring Python slice syntax. Setting only `step` spans the full sequence.

## Stage Blocks

### Semantic-SAM (`stages.ssam`)
- `enabled` – toggle the stage entirely.
- `ssam_freq` – run Semantic-SAM on every Nth frame (`1` processes every sampled frame).
- `min_area` – drop masks smaller than this pixel count.
- `stability_threshold` – discard masks with `stability_score` below the threshold.
- `persist_raw` – keep raw Semantic-SAM outputs (tar chunks + manifest).
- `skip_filtering` – if `true`, bypasses `filter_candidates` after generation.
- `add_gaps` – synthesise masks for uncovered regions exceeding `fill_area`.
- `fill_area` – minimum pixel count for gap-fill regions (often aligned with `min_area`).
- Progressive refinement now clamps every child mask to its parent and drops children identical to the parent mask, preventing spill-over between levels.
- `append_timestamp` – force timestamped run directories even when reusing outputs.
- `downscale_masks` / `downscale_ratio` – control optional mask downsampling at persistence time.

### Candidate Filtering (`stages.filter`)
- `enabled` – run the filter stage when raw outputs already exist.
- `min_area`, `stability_threshold` – same semantics as the Semantic-SAM stage (can diverge from generation settings when re-filtering).

### SAM2 Tracker (`stages.tracker`)
- `enabled` – toggle tracking entirely (useful for Semantic-SAM-only runs).
- `prompt_mode` – `all_mask`, `mask_first_box_fallback`, `box_only`, etc. (see `track_from_candidates.py` for supported values).
- `max_propagate` – cap forward/backward propagation steps per prompt.
- `iou_threshold` – deduplication IoU threshold.
- `downscale_masks` / `downscale_ratio` – persist SAM2 masks at lower resolution (filenames gain `_scale{ratio}x`).
- `render_viz` – save Semantic-SAM vs SAM2 comparison images.
- `comparison_sampling` – `stride` / `max_frames` keys bound the number of preview frames rendered.

### Report (`stages.report`)
- `enabled` – toggle Markdown report generation.
- `name` – output filename (written into the run directory).
- `max_width` – downscale comparison images before embedding.
- `record_timings` – when `true`, writes `stage_timings.json`.
- `timing_output` – custom filename for stage timings.

## CLI Overrides

- `run_experiment.sh` exposes a subset of parameters as flags (levels, frame range, thresholds, propagation depth, Semantic-SAM cadence). The script resolves CLI values first, then YAML defaults.
- `python -m my3dis.run_workflow --config ...` honours CLI `--output`, `--dry-run`, and memory-event flags. Stage-specific overrides must be edited in the YAML or provided via environment variables.
- Stage CLIs (`python -m my3dis.generate_candidates`, `python -m my3dis.track_from_candidates`) mirror the options used inside the orchestrator when run standalone.

## Environment Variables

Set these to override hard-coded paths without touching YAML:

```
MY3DIS_SEMANTIC_SAM_ROOT
MY3DIS_SEMANTIC_SAM_CKPT
MY3DIS_SAM2_ROOT
MY3DIS_SAM2_CFG
MY3DIS_SAM2_CKPT
MY3DIS_DATASET_ROOT
MY3DIS_DATA_PATH
MY3DIS_OUTPUT_ROOT
```

CLI arguments override environment variables, which in turn override YAML defaults.

## Customisation Tips

- When sweeping multiple scenes, raise `parallel_scenes` only if you have headroom on VRAM and disk I/O.
- Use stage-level `levels` overrides when you want to generate candidates at `[2,4,6]` but track only a subset.
- Keep gap filling (`add_gaps`) enabled on the coarsest level when you expect large uncovered regions; disable it to benchmark raw Semantic-SAM recall.
- Set `comparison_sampling.max_frames` to a small number for long videos to keep report generation manageable.
- If you share run directories across machines, prefer `tag_in_path=true` to encode experiment metadata directly into folder names.
