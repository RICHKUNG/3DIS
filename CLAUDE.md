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
   - **Cascade Filtering** (NEW): Parent-aware filtering prevents orphan creation
     - If parent mask is filtered, all descendants are also filtered
     - Enabled by default in SSAM stage (`cascade_filtering: true`)
     - Can be re-applied via `filter_candidates.py --cascade`
     - Requires `unique_id` and `parent_unique_id` in data
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
- **Cascade Filtering (NEW):**
  - `stages.ssam.cascade_filtering: true` (default) - Enable parent-aware filtering
  - `stages.ssam.stability_threshold: 0.9` - Stability threshold for cascade filter
  - Prevents orphan creation by filtering children when parent is filtered

### Output Layout
Each run produces:
```
<output_root>/<scene>/<run_name>/
├── level_{L}/
│   ├── raw/                    # Raw Semantic-SAM (manifest.json, chunk_*.tar)
│   ├── filtered/               # Filtered masks (filtered.json, *.npz)
│   └── tracking/               # SAM2 outputs (video_segments_scale0.3x.npz, object_segments_scale0.3x.npz)
├── relations/                  # Family tree outputs (NEW)
│   ├── provenance_tree_L2.json # Per-level parent-child relationships
│   ├── provenance_tree_L4.json
│   ├── provenance_tree_L6.json
│   └── family_tree.json        # Unified cross-level family hierarchy
├── visualizations/             # Family comparison images (NEW, optional)
│   ├── family_001_frame_XXXXX.png
│   ├── family_002_frame_XXXXX.png
│   └── ...                     # Side-by-side L2/L4/L6 renders
├── workflow_summary.json       # Per-stage config, timings, resource peaks
├── environment_snapshot.json   # Python/CUDA/torch versions, git state
└── report.md                   # Generated summary with visualizations
```

**Object ID Format (NEW):**
- L2 objects: 2000-2999
- L4 objects: 4000-4999
- L6 objects: 6000-6999
- Level immediately visible from ID alone

### Streaming Architecture
- **Raw SSAM:** Chunked tar archives to avoid millions of small files (`raw_archive.py`)
- **SAM2 Output:** Manifest-backed NPZ/ZIP archives, never materialize entire arrays in memory
- **Mask Encoding:** Packed binary format (`my3dis.mask.encoding.encode_mask`) used throughout

### Module Structure (NEW - 2025-11-07)

The codebase uses a modular structure for better organization:

**Mask Operations** (`src/my3dis/mask/`):
- `encoding.py` - Pack/unpack binary masks, downscaling
- `geometry.py` - Bounding box extraction and conversion

**I/O Utilities** (`src/my3dis/io/`):
- `fs_utils.py` - Directory creation, frame subset building

**General Utilities** (`src/my3dis/utils/`):
- `time_utils.py` - Duration formatting
- `parsing.py` - Config parsing (levels, ranges, CSV)
- `sorting.py` - Numeric filename sorting
- `logging.py` - Logging configuration

**Legacy Compatibility**:
- `common_utils.py` - Re-exports from new modules for backward compatibility
- Will be maintained until v2.2 (approximately 9 months)
- New code should import from specific modules for clarity

**Example**:
```python
# Preferred (new code)
from my3dis.mask.encoding import encode_mask, pack_binary_mask
from my3dis.utils.parsing import parse_levels

# Still works (legacy code)
from my3dis.common_utils import encode_mask, parse_levels
```

### StageRunner Pattern (NEW - 2025-11-07)

The workflow now uses a **StageRunner pattern** for better modularity and testability. Each pipeline stage is implemented as an independent class that implements the `StageRunner` interface.

**Architecture** (`src/my3dis/workflow/`):
- `stage_runner.py` - Abstract base class defining the stage interface
- `stages/` - Directory containing all stage implementations
  - `ssam_stage.py` - Semantic-SAM candidate generation
  - `filter_stage.py` - Candidate filtering
  - `tracker_stage.py` - SAM2 tracking
  - `family_tree_stage.py` - Family tree generation
  - `family_viz_stage.py` - Family visualization
  - `report_stage.py` - Report generation

**Benefits**:
- Each stage is independently testable
- Easy to add new stages
- Clear separation of concerns
- Consistent stage execution interface

**Creating a Custom Stage**:
```python
from my3dis.workflow.stage_runner import StageRunner

class CustomStageRunner(StageRunner):
    @property
    def name(self) -> str:
        """Return stage name for configuration lookup."""
        return "custom"

    def should_run(self) -> bool:
        """Determine if stage should execute based on config."""
        return self._stage_cfg().get('enabled', True)

    def execute(self) -> None:
        """Execute stage-specific logic."""
        self._log(f"Starting {self.name} stage...")

        # Your stage logic here
        # Access context via: self.context
        # Access config via: self._stage_cfg()
        # Record timing via: self._record_stage_timing()

        self._log(f"{self.name} stage completed")
```

**Usage in SceneWorkflow**:
```python
from my3dis.workflow.scene_workflow import SceneWorkflow, SceneContext

# SceneWorkflow automatically registers all standard stages
context = SceneContext(
    config=config,
    experiment_cfg=experiment_cfg,
    stages_cfg=stages_cfg,
    default_stage_gpu=None,
    data_path="/path/to/scene/color",
    output_root="/path/to/output",
    config_path=Path("config.yaml"),
)

workflow = SceneWorkflow(context)
summary = workflow.run()  # Executes all enabled stages
```

**Backward Compatibility**:
- The refactored workflow maintains 100% API compatibility
- `run_scene_workflow()` function still works as before
- Original implementation backed up as `scene_workflow_legacy.py`

## Project Status & Recent Changes

**Module Refactoring (2025-11-07):**

**Phase 1 - common_utils.py Modularization (COMPLETE):**
- ✅ **Reduced from 306 lines to 109 lines (-64.4%)**
- ✅ Split into specialized modules: `mask/`, `io/`, `utils/`
  - `my3dis.mask.encoding` - Mask encoding/decoding operations (187 lines)
  - `my3dis.mask.geometry` - Bounding box and geometric operations (32 lines)
  - `my3dis.io.fs_utils` - Filesystem utilities (76 lines)
  - `my3dis.utils.time_utils` - Time formatting (30 lines)
  - `my3dis.utils.parsing` - Configuration parsing (78 lines)
  - `my3dis.utils.sorting` - Sorting utilities (33 lines)
  - `my3dis.utils.logging` - Logging configuration (69 lines)
- ✅ **100% Backward Compatibility:** All existing imports continue to work
- ✅ **Comprehensive Testing:** Test suite (`scripts/test_refactoring_phase1.py`) validates all modules

**Phase A - scene_workflow.py StageRunner Refactoring (COMPLETE):**
- ✅ **Reduced from 664 lines to 264 lines (-60.2%)**
- ✅ **StageRunner Pattern:** Extracted 6 independent stage implementations
  - `stages/ssam_stage.py` - Semantic-SAM candidate generation (162 lines)
  - `stages/filter_stage.py` - Candidate filtering (81 lines)
  - `stages/tracker_stage.py` - SAM2 tracking (99 lines)
  - `stages/family_tree_stage.py` - Family tree generation (71 lines)
  - `stages/family_viz_stage.py` - Family visualization (127 lines)
  - `stages/report_stage.py` - Report generation (78 lines)
- ✅ **Abstract Interface:** `stage_runner.py` defines consistent stage execution pattern
- ✅ **100% Backward Compatibility:** All existing APIs preserved, legacy version backed up
- ✅ **Integration Tested:** All imports and executor integration verified

**Overall Progress:**
- 📊 **50% Complete** (2/4 phases finished)
- 📚 **Documentation:** See [`REFACTORING_STATUS_2025_11_07.md`](REFACTORING_STATUS_2025_11_07.md) and [`REFACTORING_PROGRESS_2025_11_07.md`](REFACTORING_PROGRESS_2025_11_07.md)
- 🎯 **Impact:** Significantly improved code organization, modularity, and maintainability without breaking changes

**Cascade Filtering & Orphan Fix (2025-11-04):**
- ✅ **Cascade Filtering:** Parent-aware filtering prevents orphan creation at SSAM stage
  - Enabled by default (`stages.ssam.cascade_filtering: true`)
  - Filters children when parent is filtered, maintaining tree integrity
  - Detailed statistics: area_rejected, stability_rejected, cascade_rejected
- ✅ **Dedup Timing Fix:** Fixed race condition in `sam2_runner.py`
  - Deferred rejection processing until after prompt mapping is complete
  - Resolves ~80% of timing-related orphans
- ✅ **Virtual Children Tracking:** Complete family tree semantics (NEW)
  - Tracks SSAM masks rejected (deduped) as "virtual children" of matched parent
  - Enables complete provenance: SSAM → SAM2 with dedup relationships
  - Query API: `get_virtual_children()`, `has_virtual_children()`, `get_all_children()`
  - Statistics: `virtual_children_count`, `parents_with_virtual_children`
- ✅ **Unresolved Rejection Tracking:** Added to `provenance_tracker.py`
  - Tracks cases where parent mapping fails
  - Helps diagnose remaining orphan sources
- ✅ **End-to-End Integration:** YAML config → workflow → SSAM → tracking
  - `filter_candidates.py` supports `--cascade` for re-filtering
  - Complete parameter chain validation
- ✅ **Testing:** Comprehensive test suite with 100% pass rate
  - Unit tests: ProvenanceTracker, FamilyTreeBuilder, QueryInterface
  - Test script: `scripts/test_virtual_children.py`
- 📚 **Documentation:** See [`ORPHAN_FIX_PROGRESS.md`](ORPHAN_FIX_PROGRESS.md) and [`ORPHAN_FIX_IMPLEMENTATION_SUMMARY.md`](ORPHAN_FIX_IMPLEMENTATION_SUMMARY.md)
- 🎯 **Expected Impact:** Orphan rate reduction from 2.2% to <0.5%

**Enhanced Family Tree System (2025-10-30):**
- ✅ **Orphan Fix:** Propagated mask tracking eliminates ~80% of orphans (60% → <10%)
- ✅ **Level-Based Object IDs:** L2=2XXX, L4=4XXX, L6=6XXX for instant level identification
- ✅ **Unified Family Tree:** Per-scene `family_tree.json` with complete parent-child hierarchy
- ✅ **Query Interface:** `family_tree_query.py` for frame/mask lookup with caching
- ✅ **Family Visualization:** Side-by-side L2/L4/L6 rendering in common frames
- ✅ **Workflow Integration:** Automatic family tree + visualization generation after tracking
- ✅ **YAML Configuration:** Control visualization with `visualize_families` and `family_viz.*` parameters
- ✅ **SAM2 Tree Archived:** Old containment-based approach moved to `archive/sam2_tree_containment/`
- 📚 **Documentation:** Updated [`docs/PROVENANCE_TRACKING.md`](docs/PROVENANCE_TRACKING.md)

**Provenance-Based Tree Building (2025-10-29):**
- ✅ **Globally Unique SSAM IDs:** Composite key format `{frame:04d}_{level}_{seq:04d}` eliminates ID collisions
- ✅ **ProvenanceTracker:** Unified SSAM → SAM2 mapping tracks both accepted and rejected prompts
- ✅ **DedupStore Enhancements:** Returns rejection info with matched mask indices
- ✅ **Automatic Parent Tracking:** O(1) parent lookup preserves SSAM hierarchies through SAM2
- ✅ **Per-Level Provenance Trees:** `relations/provenance_tree_L{level}.json` with orphan statistics
- ✅ **Comparison Script:** `scripts/compare_trees.py` analyzes provenance vs containment approaches

**Recent Optimizations (2025-10-22):**
- ✅ **OOM 監控修復：** 修正 `run_workflow.py` 路徑設置，OOM 監控現在可正常工作
- ✅ **無限制參數支持：** `max_masks_per_level: 0` 或 `-1` 現在表示無上限
- ✅ **強制並行選項：** 新增 `force_parallel: true` 可繞過 OOM 檢查
- ✅ **家族關係修復：** 修正 cross-level 父子關係計算，family 統計現在準確
- ✅ **Tree.json 生成修復：** 關係樹現在正確保存到 `relations/tree.json`
- ✅ **Check.md 整合：** 實驗級報告自動生成，包含場景統計和計時

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

## Family Tree System (NEW)

### Automatic Generation
Family trees are automatically generated after SAM2 tracking completes. Output includes:
- `relations/provenance_tree_L{2,4,6}.json` - Per-level parent-child relationships
- `relations/family_tree.json` - Unified cross-level hierarchy
- `visualizations/` - Side-by-side L2/L4/L6 comparison images (if enabled in YAML)

### Configuration (YAML)

Enable/disable visualization and control parameters in your config:

```yaml
stages:
  tracker:
    visualize_families: true      # Enable automatic visualization
    family_viz:
      num_families: 10             # Max families to visualize (null = all)
      max_frames_per_family: 3     # Frames per family
      output_subdir: visualizations
      random_seed: 42
      min_levels: 2                # Only families with >= N levels
```

**Disable visualization:**
```yaml
stages:
  tracker:
    visualize_families: false
```

### Query and Visualization

**Query family tree:**
```bash
PYTHONPATH=src python -m my3dis.family_tree_query \
  --tree outputs/experiments/<scene>/<run>/relations/family_tree.json \
  --stats  # Show statistics
  --obj-id 2050  # Query specific object
  --list-families  # List all families
```

**Visualize families (manual, if not auto-generated):**
```bash
python scripts/visualize_families.py \
  --tree outputs/experiments/<scene>/<run>/relations/family_tree.json \
  --data-path /path/to/scene/color \
  --output-dir visualizations/ \
  --num-families 5 \
  --max-frames 3
```

**Test visualization workflow:**
```bash
python scripts/test_family_viz_workflow.py \
  --run-dir outputs/experiments/<scene>/<run> \
  --data-path /path/to/scene/color
```

### Querying from Python

```python
from my3dis.family_tree_query import FamilyTreeQuery

# Load tree
query = FamilyTreeQuery('path/to/family_tree.json')

# Get object info
obj_info = query.get_object_info(2050)
frames = query.get_object_frames(2050)
parent = query.get_parent(2050)
children = query.get_children(2050)

# Get family members
family_members = query.get_family_members(2050)

# Load masks
mask = query.load_mask(obj_id=2050, frame_idx=1200)
masks_batch = query.load_masks_batch(obj_id=2050)

# Find common frames across objects
common_frames = query.get_common_frames([2050, 4100, 6200])

# Query virtual children (NEW in v2)
virtual_children = query.get_virtual_children(2050)
has_virtual = query.has_virtual_children(2050)
all_children = query.get_all_children(2050, include_virtual=True)
```

### Virtual Children (NEW)

**Virtual children** are SSAM masks that were rejected (deduped) during SAM2 tracking but conceptually belong to an object's family tree. They represent "family merging" where child-level masks were identified as the same object as the parent.

**Use Cases:**
- Understanding which child-level masks merged into parent objects
- Complete provenance tracking from SSAM to SAM2
- Debugging deduplication behavior

**Example:**
```python
from my3dis.family_tree_query import FamilyTreeQuery

query = FamilyTreeQuery('path/to/family_tree.json')

# Check if object has virtual children
if query.has_virtual_children(2050):
    virtual_children = query.get_virtual_children(2050)
    print(f"Object 2050 merged {len(virtual_children)} child-level masks:")
    for ssam_id in virtual_children:
        print(f"  - {ssam_id}")

# Get all children (both real and virtual)
all_children = query.get_all_children(2050, include_virtual=True)
print(f"Real children: {all_children['children']}")
print(f"Virtual children: {all_children['virtual_children']}")
```

**Statistics:**
Virtual children statistics are automatically included in:
- `provenance_tree_L{level}.json`: Per-level virtual children
- `family_tree.json`: Aggregated statistics
  - `virtual_children_count`: Total virtual children across all objects
  - `parents_with_virtual_children`: Number of parents with virtual children

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
# Traditional filtering
PYTHONPATH=src python src/my3dis/filter_candidates.py \
  --candidates-root outputs/experiments/<scene>/<run_name> \
  --levels 2,4,6 \
  --min-area 500 \
  --stability-threshold 0.9

# With cascade filtering (requires unique_id in data)
PYTHONPATH=src python src/my3dis/filter_candidates.py \
  --candidates-root outputs/experiments/<scene>/<run_name> \
  --levels 2,4,6 \
  --min-area 500 \
  --stability-threshold 0.9 \
  --cascade \
  --update-manifest
```

**Note:** Cascade filtering requires data generated with `cascade_filtering: true`. If unique_id fields are missing, it will automatically fall back to traditional filtering with a warning.

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

- **Family Tree System (NEW):**
  - `src/my3dis/family_tree_builder.py` - Build unified family trees from provenance data
  - `src/my3dis/family_tree_query.py` - Query interface for family trees and mask loading
  - `scripts/visualize_families.py` - Side-by-side L2/L4/L6 visualization tool
  - `src/my3dis/tracking/provenance_tracker.py` - SSAM→SAM2 mapping with O(1) parent lookup

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

## Configuration Best Practices

### Parallel Execution Modes

**Mode 1: OOM-Protected Parallel (Recommended for Production)**
```yaml
experiment:
  parallel_scenes: 2
  force_parallel: false  # Default; requires OOM monitoring
```
- Automatically downgrades to sequential if OOM monitoring unavailable
- Logs OOM events to `logs/oom_monitor.log`
- Best for long-running experiments

**Mode 2: Force Parallel (Fast but Unprotected)**
```yaml
experiment:
  parallel_scenes: 2
  force_parallel: true  # Bypass OOM check
```
- Runs parallel execution even without OOM monitoring
- Suitable for trusted environments with sufficient memory
- Faster but no OOM protection

**Mode 3: Sequential (Safest)**
```yaml
experiment:
  parallel_scenes: 1
```
- Processes one scene at a time
- Lowest memory footprint
- Best for debugging or memory-constrained systems

### Mask Limits Configuration

**Default (Capped at 2000):**
```yaml
stages:
  ssam:
    max_masks_per_level: 2000  # Default value
```

**Unlimited (No Cap):**
```yaml
stages:
  ssam:
    max_masks_per_level: 0  # Or -1 for unlimited
```
- Allows all possible masks regardless of quantity
- Use for high-resolution or complex scenes
- May significantly increase memory usage and processing time

**Conservative (Lower Memory):**
```yaml
stages:
  ssam:
    max_masks_per_level: 500  # Reduce for memory savings
```

### Report Generation

**Enable Experiment-Level Summary:**
```yaml
stages:
  report:
    enabled: true
    generate_check_report: true  # Creates check.md with statistics
```
- Generates `check.md` at experiment root
- Includes scene timings and output statistics (objects, families, orphans)
- Useful for comparing multiple scene results

## Security Notes

- **torch.load:** Audit all `torch.load` calls (including `third_party/`) and migrate to `weights_only=True` before upstream defaults change
- **GPU Isolation:** Multi-scene execution relies on `CUDA_VISIBLE_DEVICES`; concurrent workers can collide when `parallel_scenes > 1`
- **OOM Detection:** `oom_monitor` (local module in `oom_monitor/`) watches cgroup memory events; see `docs/OOM_MONITORING.md` for setup details

## Documentation

> **📚 Documentation Index**: See **[`docs/README.md`](docs/README.md)** for complete navigation guide

### Architecture & Refactoring (New!)
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** - Complete architecture overview: current (v1) vs refactored (v2) design
  - High-level architecture diagrams (mermaid)
  - Module導覽與職責劃分
  - Data flow illustrations
  - Key design decisions rationale
- **[`docs/重構.md`](docs/重構.md)** - V2 Refactoring Proposal (繁體中文)
  - Staged refactoring strategy (Week 0 → Week 12)
  - Quantitative success metrics
  - Risk mitigation strategies
  - Parallel development workflow
- **[`docs/BACKWARD_COMPATIBILITY.md`](docs/BACKWARD_COMPATIBILITY.md)** - v1 → v2 Migration Guide
  - Version semantics & compatibility guarantees
  - CLI/YAML/Output format compatibility
  - Migration tools (`my3dis migrate`, `my3dis doctor`)
  - 6-month transition timeline
- **[`docs/REFACTORING_METRICS.md`](docs/REFACTORING_METRICS.md)** - Progress Tracking
  - Baseline metrics (test coverage, complexity, etc.)
  - Weekly progress tracking
  - Automated measurement scripts
- **[`docs/adr/`](docs/adr/)** - Architecture Decision Records
  - [ADR-0001: Adopt Staged Refactoring](docs/adr/0001-adopt-staged-refactoring.md)
  - [ADR Template](docs/adr/template.md)
  - [ADR Guidelines](docs/adr/README.md)

### Comprehensive Guides
- **[`docs/WORKFLOW_STATUS_2025_10_22.md`](docs/WORKFLOW_STATUS_2025_10_22.md)** - Complete workflow status, recent changes, and optimization strategies
- **[`docs/OOM_MONITORING.md`](docs/OOM_MONITORING.md)** - OOM monitoring setup, configuration, and troubleshooting
- **[`docs/TREE_ALGORITHM_ANALYSIS.md`](docs/TREE_ALGORITHM_ANALYSIS.md)** - Tree building algorithm, why tree nodes ≠ tracked objects, path discrepancy analysis
- **[`docs/SAM2_TREE_USAGE.md`](docs/SAM2_TREE_USAGE.md)** - SAM2-based tree building usage guide (containment algorithm, hyperparameters, examples)
- **[`src/my3dis/OVERVIEW.md`](src/my3dis/OVERVIEW.md)** - Module-by-module execution flow
- **[`src/my3dis/workflow/WORKFLOW_GUIDE.md`](src/my3dis/workflow/WORKFLOW_GUIDE.md)** - Multi-scene orchestration internals
- **[`src/my3dis/tracking/TRACKING_GUIDE.md`](src/my3dis/tracking/TRACKING_GUIDE.md)** - SAM2 tracking implementation details

### User Guides
- **[`docs/guides/BATCH_SAM2_TREE_GUIDE.md`](docs/guides/BATCH_SAM2_TREE_GUIDE.md)** - Batch SAM2 tree building guide
- **[`docs/guides/OPTIMIZATION_GUIDE.md`](docs/guides/OPTIMIZATION_GUIDE.md)** - Performance optimization techniques
- **[`docs/guides/downstream_mask_loading.md`](docs/guides/downstream_mask_loading.md)** - Loading masks for downstream tasks
- **[`docs/guides/QUICK_START_RECOVERY.md`](docs/guides/QUICK_START_RECOVERY.md)** - Quick start guide for data recovery

### Format and Data Standards
- **[`docs/FORMAT_STANDARDS_v3.md`](docs/FORMAT_STANDARDS_v3.md)** - Complete v3.0 format specification (JSON schemas, NPZ structures, naming conventions)
- **[`docs/troubleshooting/RECOVERY_FORMAT_ISSUE.md`](docs/troubleshooting/RECOVERY_FORMAT_ISSUE.md)** - Format mismatch issue timeline, root causes, and solutions
- **[`docs/proposals/SAM2_TREE_PROPOSALS.md`](docs/proposals/SAM2_TREE_PROPOSALS.md)** - Four proposals for SAM2-based tree building (comparison matrix, recommendations)

### Quick References
- **[`docs/PROVENANCE_TRACKING.md`](docs/PROVENANCE_TRACKING.md)** - Provenance-based tree building system (NEW)
- **[`docs/configuration.md`](docs/configuration.md)** - Configuration system documentation
- **[`docs/RELATION_INDEX.md`](docs/RELATION_INDEX.md)** - Relation indexing system
- **[`docs/POST_RUN_RECOVERY_v4.md`](docs/POST_RUN_RECOVERY_v4.md)** - Post-run data recovery (v4 format)
- **[`configs/multiscan/base.yaml`](configs/multiscan/base.yaml)** - Production configuration template
- **[`configs/multiscan/test_65.yaml`](configs/multiscan/test_65.yaml)** - Test/debug configuration
- **[`scripts/compare_trees.py`](scripts/compare_trees.py)** - Compare provenance vs containment trees (NEW)
- **[`scripts/check_scene_timings.py`](scripts/check_scene_timings.py)** - Performance analysis tool
- **[`scripts/validate_formats.py`](scripts/validate_formats.py)** - Format validation tool for relations.json, tree.json, index.json
