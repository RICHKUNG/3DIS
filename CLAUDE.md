# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

My3DIS is a two-stage pipeline that combines Semantic-SAM candidate generation with SAM2 mask propagation to build dense instance masks for indoor video sequences (primarily MultiScan dataset). The pipeline is designed for reproducible multi-scene experiments with YAML-driven orchestration.

**Key Architecture:**
- **Stage 1 (Semantic-SAM):** Progressive multi-level candidate generation with gap filling
- **Stage 2 (SAM2):** Temporal mask propagation and tracking with deduplication
- **Multi-environment:** Requires two conda environments (Semantic-SAM with Detectron2/Torch 1.13, SAM2 with Torch 2.x)

## Environment Setup

### Python Path
When running modules directly from the checkout without installing:
```bash
export PYTHONPATH=$(pwd)/src
# OR
PYTHONPATH=src python -m my3dis.run_workflow --config configs/multiscan/base.yaml
```

### Environment Variables
All external dependencies can be overridden via `MY3DIS_*` environment variables (highest precedence):
- `MY3DIS_SEMANTIC_SAM_ROOT` - Path to Semantic-SAM repository
- `MY3DIS_SEMANTIC_SAM_CKPT` - Semantic-SAM checkpoint path
- `MY3DIS_SAM2_ROOT` - Path to SAM2 repository
- `MY3DIS_SAM2_CFG` - SAM2 config YAML path
- `MY3DIS_SAM2_CKPT` - SAM2 checkpoint path
- `MY3DIS_DATASET_ROOT` - MultiScan dataset root directory
- `MY3DIS_DATA_PATH` - Specific scene color frames directory
- `MY3DIS_OUTPUT_ROOT` - Output directory root

Default paths are defined in `src/my3dis/pipeline_defaults.py:9` and point to local paths on the dev machine.

## Running the Pipeline

### Option A: Two-Stage Shell Script (Single Scene)
```bash
./run_experiment.sh \
  --levels 2,4,6 \
  --frames 1200:1600:20 \
  --ssam-freq 2 \
  --sam2-max-propagate 30 \
  --min-area 500 \
  --stability 0.9
```
- Toggles between conda environments automatically
- Produces timestamped run directories unless `--no-timestamp` is used
- Use `--dry-run` to preview commands

### Option B: YAML Orchestrator (Multi-Scene)
```bash
python -m my3dis.run_workflow --config configs/multiscan/base.yaml
# OR
PYTHONPATH=src python run_workflow.py --config configs/multiscan/base.yaml
```
- Handles multi-scene experiments with parallelism control
- Key config file: `configs/multiscan/base.yaml`
- Stage toggles via `stages.{ssam,filter,tracker,report}.enabled`

## Important Code Architecture

### Pipeline Flow
1. **Candidate Generation** (`src/my3dis/generate_candidates.py:71`)
   - Entry: `run_generation()`
   - Calls `ssam_progressive_adapter.generate_with_progressive()` which wraps `semantic_refinement.progressive_refinement_masks()`
   - Outputs: Raw candidates in chunked tar archives (`raw_archive.py:15`) + filtered JSON manifests

2. **Filtering** (`src/my3dis/filter_candidates.py`)
   - Re-applies area/stability thresholds to raw candidates
   - Can be re-run without regenerating Semantic-SAM outputs

3. **SAM2 Tracking** (`src/my3dis/track_from_candidates.py:73`)
   - Entry: `run_tracking()`
   - Core: `tracking/level_runner.py` → `tracking/sam2_runner.py`
   - Key components:
     - `DedupStore` (`tracking/stores.py`) - IoU-based deduplication
     - `FrameResultStore` - Collects per-frame tracking outputs
     - Streaming manifest writers (`tracking/outputs.py:132`)
   - Outputs: Frame-major and object-major NPZ archives with `_scale{ratio}x` suffix

4. **Workflow Orchestration** (`src/my3dis/workflow/`)
   - `executor.py` - Multi-scene scheduling and parallelism
   - `scene_workflow.py` - Single-scene stage execution with `SceneWorkflow.run()`
   - `summary.py` - Resource monitoring (`StageResourceMonitor`), environment snapshots, and history tracking

### Configuration System
- **Precedence:** CLI flags > environment variables > YAML config
- **YAML Structure:**
  - `experiment.*` - Dataset, scenes, levels, frames, output paths
  - `stages.{ssam,filter,tracker,report}.*` - Per-stage parameters and toggles
- **Scene Selection:**
  - `scenes: all` - All scenes in dataset
  - `scenes: [scene_00065_00, ...]` - Explicit list
  - `scene_start` / `scene_end` - Range filtering
- **Frame Sampling:** Python slice syntax `start:end:step` (e.g., `1200:1600:20`)

### Output Layout
Each run produces:
```
<output_root>/<scene>/<run_name>/
├── level_{L}/
│   ├── raw/                    # Raw Semantic-SAM (manifest.json, chunk_*.tar)
│   ├── filtered/               # Filtered masks (filtered.json, *.npz)
│   └── tracking/               # SAM2 outputs (video_segments_scale0.3x.npz, etc.)
├── workflow_summary.json       # Per-stage config, timings, resource peaks
├── environment_snapshot.json   # Python/CUDA/torch versions, git state
└── report.md                   # Generated summary with visualizations
```

### Streaming Architecture
- **Raw SSAM:** Chunked tar archives to avoid millions of small files (`raw_archive.py`)
- **SAM2 Output:** Manifest-backed NPZ/ZIP archives, never materialize entire arrays in memory
- **Mask Encoding:** Packed binary format (`common_utils.py:encode_mask`) used throughout

## Project Status & Recent Changes

**Recent Cleanup (2025-10-21):**
- ✅ Removed 10 broken/unused files (~17.7KB of technical debt)
- ✅ Simplified progressive_refinement architecture (removed triple-wrapper chain)
- ✅ Unified configuration system (removed unused dataclass-based config)
- ✅ All modules now import successfully without deprecation warnings
- ✅ Codebase reduced from ~4,866 lines to ~4,100 lines

**Packaging Approach:**
- No `setup.py` or package installation - project runs directly from source
- Always use `PYTHONPATH=src` when running modules
- Module files contain `sys.path` setup for backward compatibility

**Working Modules:**
All CLI entrypoints are now functional:
- `run_workflow.py` - Main workflow orchestrator
- `generate_candidates.py` - Stage 1: Semantic-SAM candidate generation
- `filter_candidates.py` - Re-filter raw candidates
- `track_from_candidates.py` - Stage 2: SAM2 tracking
- `generate_report.py` - Generate reports
- `prepare_tracking_run.py` - Prepare tracking runs

## Development Commands

### Run Single Scene (Development)
```bash
# Stage 1: Semantic-SAM
PYTHONPATH=src python src/my3dis/generate_candidates.py \
  --data-path data/multiscan/scene_00065_00/outputs/color \
  --levels 2,4,6 \
  --frames 1200:1600:20 \
  --output outputs/dev_run

# Stage 2: SAM2
PYTHONPATH=src python src/my3dis/track_from_candidates.py \
  --data-path data/multiscan/scene_00065_00/outputs/color \
  --candidates-root outputs/dev_run \
  --output outputs/dev_run \
  --levels 2,4,6
```

### Generate Report
```bash
PYTHONPATH=src python src/my3dis/generate_report.py \
  --run-dir outputs/experiments/<scene>/<run_name> \
  --report-name report.md \
  --max-width 640
```

### Re-filter Candidates (No Regeneration)
```bash
PYTHONPATH=src python src/my3dis/filter_candidates.py \
  --candidates-root outputs/experiments/<scene>/<run_name> \
  --levels 2,4,6 \
  --min-area 500 \
  --stability-threshold 0.9
```

### Prepare Scene Configs
```bash
PYTHONPATH=src python scripts/prepare_scene_configs.py \
  --dataset-root /path/to/multiscan \
  --output-dir configs/scenes/
```

## Key Files Reference

- **Main Entrypoints:**
  - `run_experiment.sh` - Two-stage orchestration script
  - `src/my3dis/run_workflow.py:64` - YAML workflow CLI
  - `src/my3dis/generate_candidates.py` - Stage 1 standalone
  - `src/my3dis/track_from_candidates.py` - Stage 2 standalone

- **Configuration:**
  - `configs/multiscan/base.yaml` - Canonical experiment template
  - `src/my3dis/pipeline_defaults.py` - Default paths and environment overrides
  - `src/my3dis/workflow/scenes.py` - Scene discovery and path expansion

- **Core Modules:**
  - `src/my3dis/OVERVIEW.md` - Module execution flow diagram
  - `src/my3dis/workflow/WORKFLOW_GUIDE.md` - Orchestration internals
  - `src/my3dis/tracking/TRACKING_GUIDE.md` - SAM2 tracking implementation

- **Utilities:**
  - `src/my3dis/common_utils.py` - Shared helpers (mask encoding, frame sorting, etc.)
  - `src/my3dis/workflow/summary.py:94` - Resource monitoring and environment snapshots

## Conda Environment Switching

The pipeline requires toggling between two conda environments:
- **Semantic-SAM env:** For Stage 1 (candidate generation)
  - Requires Detectron2 0.6 + Torch 1.13
- **SAM2 env:** For Stage 2 (tracking)
  - Requires Torch 2.x

When using `run_experiment.sh`, environment switching is automatic via `conda run --live-stream -n <env>`.

When running stages manually, activate the appropriate environment first:
```bash
conda activate Semantic-SAM
PYTHONPATH=src python src/my3dis/generate_candidates.py ...

conda activate SAM2
PYTHONPATH=src python src/my3dis/track_from_candidates.py ...
```

## Reproducibility Features

- **Resource Monitoring:** `StageResourceMonitor` tracks CPU/GPU peaks (`workflow/summary.py:94`)
- **Environment Snapshots:** `environment_snapshot.json` captures Python/CUDA/torch/git state
- **Workflow History:** `logs/workflow_history.csv` records all runs with flock-based locking
- **PID Mapping:** `logs/run_pid_map.tsv` tracks script invocations and their outputs
- **Manifests:** Every stage emits `manifest.json` with provenance metadata

## Visualization & Reporting

- **Comparison Renders:** `stages.tracker.render_viz: true` generates Semantic-SAM vs SAM2 overlays
- **Report Generation:** `stages.report.enabled: true` creates Markdown summaries with representative images
- **Sampling Control:** `comparison_sampling.max_frames` limits visualization cost
- **Resource Summaries:** Stage timings surfaced in `report.md` and `stage_timings.json`

## Security Notes

- **torch.load:** Audit all `torch.load` calls (including `third_party/`) and migrate to `weights_only=True` before upstream defaults change
- **GPU Isolation:** Multi-scene execution relies on `CUDA_VISIBLE_DEVICES`; concurrent workers can collide when `parallel_scenes > 1`
- **OOM Detection:** `oom_monitor` watches cgroup memory events when configured via `--memory-events`
