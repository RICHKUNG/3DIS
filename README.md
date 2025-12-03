# FamilyPart: Hierarchical 3D Instance Segmentation with Family Provenance Tracking

FamilyPart is a research system for building hierarchical 3D instance proposals from indoor RGB sequences, where objects and their parts maintain explicit family relationships through provenance tracking. The system addresses a key challenge in 3D scene understanding: **how to simultaneously capture both coarse objects and fine-grained parts while preserving their semantic relationships across space and time**.

---

## Research Motivation

Traditional 3D instance segmentation approaches treat objects at a single granularity level, missing the hierarchical nature of real-world scenes:
- A **chair** contains **armrests**, **seat**, and **legs**
- A **desk** includes **drawers** and a **tabletop**
- These part-whole relationships are critical for fine-grained scene understanding, robotic manipulation, and open-vocabulary retrieval

**Key Challenge**: When generating proposals at multiple granularity levels (coarse, medium, fine), how do we:
1. Maintain consistent identity across frames?
2. Track which fine-level masks belong to which coarse-level objects?
3. Handle deduplication when the same object is proposed at different levels?
4. Preserve complete provenance from 2D masks → tracked objects → 3D proposals?

FamilyPart solves these challenges through a **provenance-aware two-stage pipeline** that builds an explicit **family tree** connecting objects and parts across levels.

---

## Core Innovation

### 1. Multi-Level Candidate Generation with Cascade Filtering
- **Progressive Semantic-SAM**: Generates candidates at multiple semantic levels (L2=coarse, L4=medium, L6=fine)
- **Cascade Filtering**: Parent-aware filtering prevents orphan creation—if a parent is rejected, all children are automatically rejected
- **Gap Filling**: Optional infilling ensures continuous coverage across frames
- **Result**: Clean hierarchical candidates with minimal orphans

### 2. Provenance-Aware Temporal Tracking
- **SAM2-Based Propagation**: Prompts from Semantic-SAM are propagated across frames via SAM2
- **Deduplication with Memory**: IoU-based deduplication tracks rejected masks as "virtual children"
- **Global Unique IDs**: Level-encoded object IDs (L2: 2000-2999, L4: 4000-4999, L6: 6000-6999) enable instant level identification
- **Result**: Every tracked object knows its SSAM parent, SAM2 propagation history, and which masks merged into it

### 3. Family Tree Construction
- **Provenance Tracker**: Maintains SSAM → SAM2 mapping with O(1) parent lookup
- **Virtual Children**: Tracks deduped masks as "virtual children" for complete lineage
- **Cross-Level Hierarchy**: Unified family tree links L2/L4/L6 objects through parent-child relationships
- **Result**: Query interface supports "find all parts of object X" or "which object does this part belong to?"

### 4. 3D Aggregation with Family Awareness
- **Temporal Fusion**: Projects tracked 2D masks into 3D space via multi-view geometry
- **Hierarchical Proposals**: Exports per-level proposals (NPZ/JSON) with family tree metadata
- **Search3D Integration**: Family-aware proposal format enables part-level open-vocabulary retrieval
- **Result**: 3D proposals preserve 2D→3D provenance and part-whole semantics

### 5. Open-Vocabulary Retrieval with Family Context
- **SigLIP Feature Extraction**: Optimized multi-resolution feature extraction for tracked masks
- **Geometry-Aware Pairing**: Uses family tree to avoid pairing parent-child pairs incorrectly
- **Hierarchical Scoring**: Combines text-similarity and 3D geometry for ranking
- **Result**: "Find all chair armrests" returns fine-level instances with correct parent context

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: RGB-D Video Sequence                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼────────────┐
                │  Stage 1: Semantic-SAM  │
                │  Multi-Level Candidates │
                └────────────┬────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Cascade Filter  │
                    │  (Parent-Aware)  │
                    └────────┬─────────┘
                             │
                ┌────────────▼────────────┐
                │   Stage 2: SAM2 Track   │
                │   + Dedup + Provenance  │
                └────────────┬────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Family Tree     │
                    │  Construction    │
                    └────────┬─────────┘
                             │
                ┌────────────▼────────────┐
                │  Stage 3: 3D Aggregate  │
                │  Multi-View Fusion      │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │  Stage 4: SigLIP Assign │
                │  Open-Vocab Features    │
                └────────────┬────────────┘
                             │
            ┌────────────────▼─────────────────┐
            │  Output: Hierarchical 3D Proposals│
            │  + Family Tree + Feature Embeddings│
            └──────────────────────────────────┘
```

---

## Key Features

### Hierarchical Object Identity
- **Level-Encoded IDs**: Object IDs immediately reveal their semantic level
  - `2050` → L2 (coarse object)
  - `4123` → L4 (medium part)
  - `6789` → L6 (fine detail)

### Complete Provenance Tracking
Every object maintains:
- **Birth**: Which Semantic-SAM mask(es) created it
- **Lineage**: Parent-child relationships across levels
- **Merging**: Which child-level masks were deduped into it (virtual children)
- **Tracking**: Frame-by-frame mask evolution

### Reproducible Experimentation
- **YAML-Driven**: All experiments configured via YAML (no hardcoded paths)
- **Resource Monitoring**: Automatic CPU/GPU/memory tracking
- **Workflow History**: Complete audit trail of all runs
- **Environment Snapshots**: Captures Python/CUDA/library versions

---

## Output Examples

### Family Tree Structure
```json
{
  "2050": {
    "level": 2,
    "children": [4100, 4101, 4102],
    "virtual_children": ["1500_4_0023"],
    "frames": [1200, 1220, 1240, ...],
    "parent": null
  },
  "4100": {
    "level": 4,
    "children": [6200, 6201],
    "parent": 2050,
    "frames": [1200, 1220, ...]
  }
}
```

### Family Visualization
FamilyPart can generate side-by-side visualizations showing how objects decompose across levels:

```
Frame 1200:
┌────────────┬────────────┬────────────┐
│   L2: 2050 │  L4: 4100  │  L6: 6200  │
│   (chair)  │  (armrest) │  (edge)    │
└────────────┴────────────┴────────────┘
```

---

## Getting Started

### Prerequisites
- Two conda environments (Semantic-SAM requires Detectron2 + Torch 1.13, SAM2 requires Torch 2.x)
- MultiScan dataset (or custom RGB-D sequences)
- Semantic-SAM and SAM2 checkpoints

### Quick Start

**Stage 1-2: Segmentation + Tracking (Single Scene)**:
```bash
./run_experiment.sh \
  --levels 2,4,6 \
  --frames 1200:1600:20 \
  --ssam-freq 2 \
  --sam2-max-propagate 30
```

**Stage 1-2: Multi-Scene Batch Processing**:
```bash
PYTHONPATH=src python -m my3dis.run_workflow \
  --config configs/multiscan/base.yaml
```

**Stage 3-4: Aggregation + Inference (After Tracking)**:
```bash
python scripts/run_full_pipeline.py \
  --config configs/inference/experiment_pipeline.yaml \
  --exp-dir outputs/experiments/<your_experiment_name>
```

**Complete Pipeline**: Run workflow first (Stages 1-2), then run full_pipeline (Stages 3-4) with the output directory.

### Configuration

Example YAML configuration:
```yaml
experiment:
  dataset_root: /path/to/multiscan
  scenes: [scene_00065_00, scene_00093_01]
  levels: [2, 4, 6]
  frames: "1200:1600:20"

stages:
  ssam:
    enabled: true
    ssam_freq: 2
    cascade_filtering: true  # Parent-aware filtering

  tracker:
    enabled: true
    sam2_max_propagate: 30
    visualize_families: true  # Generate family visualizations

  aggregation:
    enabled: true
    voxel_size: 0.02

  inference:
    enabled: true
    feature_extractor: "siglip"
    scoring_strategy: "exhaustive_pairing"
```

---

## Project Structure

```
FamilyPart/
├── src/my3dis/           # Core package (workflow, tracking, aggregation, inference)
│   ├── workflow/         # Multi-scene orchestration
│   ├── tracking/         # SAM2 tracking + provenance
│   ├── aggregation/      # 3D proposal generation
│   ├── inference/        # SigLIP features + retrieval
│   ├── mask/             # Mask encoding/geometry utilities
│   └── utils/            # Shared utilities
│
├── configs/              # YAML configuration templates
│   ├── multiscan/        # Dataset-specific configs
│   └── inference/        # Retrieval strategy configs
│
├── scripts/              # Automation and analysis tools
│   ├── run_full_pipeline.py
│   ├── visualize_families.py
│   └── benchmark_*.py
│
├── run_experiment.sh     # Single-scene orchestrator
├── env/                  # Conda environment specs
└── README.md             # This file
```

---

## Output Format

### Per-Scene Outputs
Each experimental run produces:

```
outputs/<experiment_name>/<scene_id>/
├── level_2/
│   ├── raw/              # Raw Semantic-SAM candidates
│   ├── filtered/         # Filtered candidates
│   └── tracking/         # SAM2 tracked objects
├── level_4/
│   └── ...
├── level_6/
│   └── ...
├── relations/
│   ├── provenance_tree_L2.json  # Per-level lineage
│   ├── provenance_tree_L4.json
│   ├── provenance_tree_L6.json
│   └── family_tree.json         # Unified hierarchy
├── visualizations/       # Family comparison images (optional)
├── proposals_L*.npz      # 3D aggregated proposals
├── features_L*.npz       # SigLIP embeddings
└── workflow_summary.json # Timing/resource statistics
```

### Search3D Inference Outputs
```
outputs/inference/<experiment_name>/
├── predictions.json      # Standard Search3D format
├── object_counts.json    # Per-scene statistics
└── inference_summary.json
```

---

## Module Overview

### Workflow Orchestration (`my3dis.workflow`)
- **SceneWorkflow**: Single-scene pipeline executor
- **MultiSceneExecutor**: Parallel batch processing
- **StageRunner Pattern**: Modular stage implementations (SSAM, Filter, Tracker, Aggregation, etc.)

### Tracking System (`my3dis.tracking`)
- **SAM2Runner**: Temporal mask propagation
- **ProvenanceTracker**: SSAM → SAM2 mapping with O(1) parent lookup
- **DedupStore**: IoU-based deduplication with rejection tracking
- **FamilyTreeBuilder**: Cross-level hierarchy construction

### Aggregation (`my3dis.aggregation`)
- **MaskAggregator**: 2D→3D multi-view fusion
- **ProposalExporter**: NPZ/JSON export with family metadata
- **VoxelGridProcessor**: Spatial hashing and clustering

### Inference (`my3dis.inference`)
- **SigLIPFeatureExtractor**: Optimized multi-resolution feature extraction
- **ExhaustivePairingStrategy**: Geometry-aware proposal-query pairing
- **FamilyAwareScorer**: Uses family tree to improve ranking

---

## Design Philosophy

### Provenance Over Heuristics
Rather than relying on post-hoc spatial containment to infer part-whole relationships, FamilyPart tracks provenance from the moment masks are generated. This ensures:
- **No false negatives**: Child objects are never orphaned due to tracking failures
- **No false positives**: Spatially overlapping but unrelated objects aren't incorrectly linked
- **Complete lineage**: Every relationship is traceable back to original Semantic-SAM proposals

### Reproducibility First
Every experiment generates:
- **Workflow Summary**: Stage timings, resource peaks, parameters used
- **Environment Snapshot**: Python/CUDA/library versions, git commit
- **Audit Trail**: Complete history of all runs in `workflow_history.csv`

This ensures experiments can be:
- Debugged months later with full context
- Reproduced on different machines
- Compared across parameter variations

### Modular Architecture
FamilyPart uses the **StageRunner pattern** for extensibility:
- Each pipeline stage is an independent class implementing `StageRunner`
- Easy to add custom stages (e.g., mesh generation, semantic labeling)
- Stages can be toggled via YAML without code changes

---

## Technical Details

### Multi-Environment Design
The pipeline requires two conda environments due to dependency conflicts:
- **Semantic-SAM**: Requires Detectron2 0.6 + PyTorch 1.13
- **SAM2**: Requires PyTorch 2.x

`run_experiment.sh` automatically toggles environments via `conda run`, or use the YAML workflow for seamless multi-scene processing.

### Streaming Architecture
To handle large scenes efficiently:
- **Chunked TAR Archives**: Raw SSAM candidates stored in tar chunks (not millions of individual files)
- **Manifest-Backed NPZ**: Tracking outputs use NPZ archives with JSON manifests
- **Lazy Loading**: Masks loaded on-demand, never materialize entire scenes in memory

### Cascade Filtering Algorithm
Traditional filtering (area + stability thresholds) can create orphans when parents are rejected but children pass. Cascade filtering fixes this:

```
if parent_rejected(mask):
    reject_all_descendants(mask)
else:
    apply_traditional_filters(mask)
```

This simple change reduced orphan rates from ~2% to <0.5% in our experiments.

---

## Python Package Name

The repository is published as **FamilyPart**, but the Python package remains `my3dis` for backward compatibility:

```python
# All imports use my3dis
from my3dis.workflow import SceneWorkflow
from my3dis.tracking import ProvenanceTracker
```

When running scripts directly:
```bash
export PYTHONPATH=src
python -m my3dis.run_workflow --config config.yaml
```

## Acknowledgments

FamilyPart builds upon:
- **Semantic-SAM** for multi-level candidate generation
- **SAM2** for temporal mask tracking
- **SigLIP** for open-vocabulary feature extraction
- **MultiScan** dataset for evaluation
