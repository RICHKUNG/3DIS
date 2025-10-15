# My3DIS Pipeline (Semantic-SAM × SAM2)

My3DIS links Semantic-SAM candidate generation with SAM2 mask propagation to build dense instance masks for indoor video sequences (MultiScan by default). The repository ships with CLI helpers, YAML-driven orchestration, and reporting utilities so you can run multi-level experiments, capture reproducible metadata, and inspect or re-run any stage offline.

## Highlights
- **Two-stage orchestration:** `run_experiment.sh` swaps between dedicated Semantic-SAM and SAM2 conda environments; `python -m my3dis.run_workflow` consumes YAML for multi-scene sweeps.
- **Configurable sampling:** Drive Semantic-SAM at levels such as `[2, 4, 6]`, throttle expensive frames via `--ssam-freq`, and reuse Python slice syntax (`start:end:step`) across CLIs and YAML.
- **Streaming-friendly persistence:** Raw Semantic-SAM outputs are chunked into tar archives, filtered masks stay in JSON/NPZ pairs, and SAM2 tracking emits frame-major and object-major archives (`*_scale{ratio}x.npz`) together with on-demand comparison renders.
- **Reproducibility tooling:** Each stage records CPU/GPU peaks, stores an environment snapshot, appends to `logs/workflow_history.csv`, and locks workflow history writes to survive concurrent runs.
- **Gap filling and previews:** Progressive refinement back-fills large uncovered regions, while sampling knobs limit visualization render cost without losing first/middle/last coverage.

## Repository Layout
- `configs/` – experiment templates (e.g., `multiscan/base.yaml`) and per-scene definitions.
- `docs/` – focused guides such as `docs/downstream_mask_loading.md`.
- `scripts/` – helper CLIs for batch orchestration and maintenance.
- `src/my3dis/` – pipeline implementation with submodule guides under `workflow/` and `tracking/`.
- `outputs/` – run artifacts (created at runtime, safe to ignore in git).
- `logs/` – global history (`workflow_history.csv`) and script PID map entries.
- `env/` & `archive/env/` – current export of environment packages and frozen snapshots.

## Requirements
- **Hardware:** NVIDIA GPU with sufficient VRAM for Semantic-SAM and SAM2 (tested on 24 GB). The tracker publishes downscaled overlays by default to reduce memory pressure.
- **External repositories:** Clone Semantic-SAM and SAM2 alongside this repo (paths configurable via environment variables). Install them in editable mode after environment setup.
- **Python environments:** We recommend two dedicated conda envs (`Semantic-SAM` with Detectron2 0.6 + Torch 1.13, and `SAM2` with Torch 2.x). A single unified environment is possible but increases RAM/GPU usage.
- **Data:** MultiScan color frames should live under `data/multiscan/<scene>/outputs/color` (or any path exposed via `MY3DIS_DATA_PATH` / `MY3DIS_DATASET_ROOT`). Copy or symlink your dataset locally and set environment overrides if needed.

## Initial Setup
1. Create the recommended conda environments and install dependencies:
   ```bash
   # Example: install the shared requirements (Torch wheel may need a CUDA-specific URL).
   conda create -n Semantic-SAM python=3.10
   conda create -n SAM2 python=3.10

   conda run -n Semantic-SAM pip install -r requirements.txt
   conda run -n SAM2 pip install -r requirements.txt
   ```
2. Install sibling repositories:
   ```bash
   conda run -n Semantic-SAM pip install -e ../Semantic-SAM
   conda run -n SAM2 pip install -e ../SAM2
   ```
3. Export optional overrides when your layout differs from the defaults (any unset path falls back to the baked-in values):
   ```bash
   export MY3DIS_SEMANTIC_SAM_ROOT=/path/to/Semantic-SAM
   export MY3DIS_SEMANTIC_SAM_CKPT=/path/to/swinl_only_sam_many2many.pth
   export MY3DIS_SAM2_ROOT=/path/to/SAM2
   export MY3DIS_SAM2_CFG=/path/to/sam2_config.yaml
   export MY3DIS_SAM2_CKPT=/path/to/sam2_weights.pt
   export MY3DIS_DATASET_ROOT=/data/multiscan
   export MY3DIS_OUTPUT_ROOT=/results/my3dis
   ```
4. When running Python modules directly from the checkout, prepend `PYTHONPATH=src` or `export PYTHONPATH=$(pwd)/src`.

## Running the Pipeline

### Option A – Two-stage shell script
`run_experiment.sh` toggles between the two conda environments and writes timestamped run folders under the configured output root.
```bash
./run_experiment.sh \
  --levels 2,4,6 \
  --frames 1200:1600:20 \
  --ssam-freq 2 \
  --sam2-max-propagate 30 \
  --min-area 500 \
  --stability 0.9
```
Useful flags:
- `--experiment-tag <tag>` appends identifiers to the run directory name.
- `--no-timestamp` writes directly into the fixed output directory (overwrites previous runs).
- `--dry-run` prints commands without executing them.

### Option B – YAML orchestrator
The orchestrator handles multi-scene experiments, parallelism, and stage toggles.
```bash
python -m my3dis.run_workflow --config configs/multiscan/base.yaml
# or use the repo wrapper
PYTHONPATH=src python run_workflow.py --config configs/multiscan/base.yaml
```
Key CLI options:
- `--dry-run` validates the configuration without executing stages.
- `--output <path>` overrides `experiment.output_root` at runtime.
- `--memory-events /path/to/memory.events` turns on OOM detection (per cgroup).

## Configuration
- `configs/multiscan/base.yaml` is the canonical template for MultiScan sweeps. Important fields:
  - `experiment.dataset_root` and `experiment.scenes` select the dataset and scene list. Use `all`, explicit lists, or `scene_start`/`scene_end`.
  - `experiment.levels` lists Semantic-SAM resolution levels (per stage).
  - `experiment.frames` accepts `start`, `end`, and `step`; when only `step` is set the pipeline spans the entire sequence.
  - Stage sections (`stages.ssam`, `stages.filter`, `stages.tracker`, `stages.report`) toggle execution and specialise parameters such as stabilization thresholds or SAM2 propagation depth.
  - `experiment.parallel_scenes` controls multi-process parallelism when GPUs allow it.
- `configs/scenes/` hosts per-scene overrides produced by `scripts/prepare_scene_configs.py`.
- Environment variables prefixed with `MY3DIS_` override the same settings without editing YAML; command-line flags take the highest precedence.

## Outputs
Each run produces a scene-level directory under `experiment.output_root/<scene>/<run_name>/` with the following structure:
- `level_{L}/raw/` – Semantic-SAM raw outputs (`manifest.json`, `chunk_*.tar`).
- `level_{L}/filtered/` – Filtered mask metadata (`filtered.json`, compressed masks).
- `level_{L}/tracking/` – SAM2 artifacts: frame-major archives (`video_segments_scale0.3x.npz`), object-major archives, and prompts used for tracking.
- `level_{L}/comparison/` – Optional visualization PNGs and accompanying JSON manifests.
- `selected_frames/` – Subset of frames linked or copied for the run.
- `workflow_summary.json` – Per-stage configuration, timings, resource peaks, and environment snapshot.
- `report.md` (configurable) – Markdown summary generated by `src/my3dis/generate_report.py`.
- `stage_timings.json`, `environment_snapshot.json`, and `logs/workflow_history.csv` entries for reproducibility.

## Reporting and Monitoring
- `src/my3dis/generate_report.py` turns run directories into Markdown summaries with representative images.
- `StageResourceMonitor` tracks CPU/GPU peaks per stage; collected metrics surface in `workflow_summary.json`.
- `oom_monitor` can follow cgroup memory events and log imminent OOM conditions when configured.
- The PID map recorded by `run_experiment.sh` (`logs/run_pid_map.tsv`) maps script invocations back to their output targets.

## Additional Documentation
- `docs/downstream_mask_loading.md` – loading masks from the streaming NPZ/ZIP format.
- `docs/configuration.md` – experiment YAML settings, CLI overrides, and environment variables.
- `src/my3dis/OVERVIEW.md` – module-level overview of the pipeline (per stage).
- `src/my3dis/workflow/WORKFLOW_GUIDE.md` – YAML orchestration internals.
- `src/my3dis/tracking/TRACKING_GUIDE.md` – SAM2 tracking implementation details.
- `Agent.md` – ongoing project log and rationale.

## Troubleshooting
- Set `PYTHONPATH=src` whenever running modules without installing the package.
- Adjust `stages.tracker.downscale_ratio` or `stages.tracker.max_propagate` if GPU memory usage is high.
- Use `--dry-run` to inspect commands and validate configuration before submitting long sweeps.
- Retain only the minimal artifacts for public releases; large raw dumps can be regenerated from filtered manifests and tracking NPZs.
