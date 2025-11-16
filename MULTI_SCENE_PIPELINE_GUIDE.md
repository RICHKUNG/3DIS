# Multi-Scene Full Pipeline Guide

**Date**: 2025-11-16
**Version**: 1.0

## Overview

The full pipeline (`run_full_pipeline.py`) now supports multi-scene processing, allowing you to run aggregation, inference, and evaluation across multiple scenes in a single experiment root directory.

## Features

- ✅ **Multi-scene processing**: Process all scenes in an experiment root directory
- ✅ **Per-scene metrics**: Save individual metrics for each scene
- ✅ **Aggregated metrics**: Compute overall statistics across all scenes
- ✅ **Summary reports**: Generate markdown reports with detailed results
- ✅ **Backward compatible**: Single-scene mode still works as before

## Architecture

### Multi-Scene Mode

When enabled, the pipeline:
1. Discovers all scene directories in the experiment root
2. Processes each scene sequentially through all stages
3. Saves per-scene results to `{scene_dir}/pipeline_results.json`
4. Aggregates metrics across all scenes
5. Generates summary reports

### Output Structure

```
{exp_root}/
├── scene_00005_00/
│   ├── aggregation_output/
│   ├── inference_output/
│   └── pipeline_results.json          # Per-scene results
├── scene_00093_01/
│   ├── aggregation_output/
│   ├── inference_output/
│   └── pipeline_results.json          # Per-scene results
├── aggregated_metrics.json            # Overall metrics (NEW)
└── multi_scene_summary.md             # Summary report (NEW)
```

## Configuration

### YAML Configuration

Add these settings to your config file (e.g., `configs/inference/full_pipeline.yaml`):

```yaml
experiment:
  # Multi-scene mode settings
  multi_scene_mode: true

  # Experiment root directory (contains multiple scene subdirectories)
  exp_root: /path/to/experiment/root

  # Dataset root (for scene data paths)
  dataset_root: /media/public_dataset2/multiscan

  # Scene selection
  # Options:
  #   - "all": Process all scenes
  #   - List: ["scene_00005_00", "scene_00093_01"]
  #   - Pattern: "scene_0009*"
  scenes: all
```

### Legacy Single-Scene Mode

For backward compatibility, single-scene mode is still supported:

```yaml
experiment:
  multi_scene_mode: false  # or omit this field

  # Scene-specific paths
  exp_dir: /path/to/experiment/scene_00005_00
  scene_path: /media/public_dataset2/multiscan/scene_00005_00
```

## Usage

### Basic Multi-Scene Pipeline

```bash
# Run full pipeline (aggregation + inference + evaluation)
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene
```

### Skip Stages

```bash
# Skip aggregation (use existing aggregation outputs)
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene \
  --skip-aggregation

# Skip inference
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene \
  --skip-inference

# Skip evaluation
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene \
  --skip-evaluation
```

### Override Configuration

```bash
# Override experiment root
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene \
  --exp-root /path/to/different/experiment

# Override dataset root
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene \
  --dataset-root /path/to/dataset

# Process specific scenes
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene \
  --scenes scene_00005_00 scene_00093_01
```

## Output Files

### Per-Scene Results

Each scene directory contains `pipeline_results.json`:

```json
{
  "scene_name": "scene_00005_00",
  "status": "success",
  "aggregation": {
    "status": "success",
    "outputs": {...}
  },
  "inference": {
    "status": "success",
    "strategies": {
      "exhaustive_pairing": {
        "num_predictions": 42,
        "elapsed_time": 12.5,
        ...
      }
    }
  },
  "evaluation": {
    "status": "success",
    "results": {
      "exhaustive_pairing": {
        "obj_part": {
          "mAP": 0.234,
          "AP50": 0.345,
          "AP25": 0.456
        }
      }
    }
  }
}
```

### Aggregated Metrics

The experiment root contains `aggregated_metrics.json`:

```json
{
  "total_scenes": 2,
  "successful_scenes": 2,
  "failed_scenes": 0,
  "scenes": {
    "scene_00005_00": {
      "status": "success",
      "exhaustive_pairing": {
        "mAP": 0.234,
        "AP50": 0.345,
        "AP25": 0.456
      }
    },
    "scene_00093_01": {
      "status": "success",
      "exhaustive_pairing": {
        "mAP": 0.267,
        "AP50": 0.378,
        "AP25": 0.489
      }
    }
  },
  "overall_metrics": {
    "exhaustive_pairing": {
      "mean_mAP": 0.2505,
      "mean_AP50": 0.3615,
      "mean_AP25": 0.4725,
      "std_mAP": 0.0165,
      "std_AP50": 0.0165,
      "std_AP25": 0.0165,
      "num_scenes": 2
    }
  }
}
```

### Summary Report

The experiment root contains `multi_scene_summary.md`:

```markdown
# Multi-Scene Pipeline Summary Report

**Generated**: 2025-11-16 17:07:30

## Overview

- **Total Scenes**: 2
- **Successful**: 2
- **Failed**: 0

## Overall Metrics

| Strategy | Mean mAP | Mean AP50 | Mean AP25 | Std mAP | Std AP50 | Std AP25 | Scenes |
|----------|----------|-----------|-----------|---------|----------|----------|--------|
| **exhaustive_pairing** | 0.251 | 0.362 | 0.473 | 0.017 | 0.017 | 0.017 | 2 |

## Per-Scene Results

| Scene | Status | mAP | AP50 | AP25 | Error |
|-------|--------|-----|------|------|-------|
| scene_00005_00 | success | 0.234 | 0.345 | 0.456 |  |
| scene_00093_01 | success | 0.267 | 0.378 | 0.489 |  |

...
```

## Test Configuration

A test configuration is provided at `configs/inference/full_pipeline_multi_scene_test.yaml`.

To run the test:

```bash
# Test multi-scene discovery (dry run)
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline_multi_scene_test.yaml \
  --multi-scene \
  --skip-aggregation \
  --skip-inference \
  --skip-evaluation

# Run full test with inference and evaluation
conda run -n 3Dsiglip python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline_multi_scene_test.yaml \
  --multi-scene \
  --skip-aggregation
```

## Implementation Details

### New Functions

1. **`discover_scenes()`**: Discovers scene directories in experiment root
   - Supports pattern matching and explicit scene lists
   - Auto-resolves scene data paths from dataset root

2. **`process_single_scene()`**: Processes a single scene through all stages
   - Handles aggregation, inference, and evaluation
   - Saves per-scene results
   - Error handling and graceful degradation

3. **`aggregate_scene_metrics()`**: Aggregates metrics across all scenes
   - Computes mean and std for each strategy
   - Groups results by strategy
   - Saves to `aggregated_metrics.json`

4. **`generate_multi_scene_report()`**: Generates summary markdown report
   - Overall metrics table
   - Per-scene results table
   - Detailed scene-by-scene breakdown

### Modified Functions

1. **`setup_paths()`**: Now uses `.get()` for safer key access
2. **`main()`**: Added multi-scene mode detection and branching

## Backward Compatibility

The single-scene mode is fully preserved. If `multi_scene_mode: false` (or omitted), the pipeline works exactly as before.

```bash
# Legacy single-scene mode (still works)
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --exp-dir /path/to/scene_00005_00 \
  --scene-path /media/public_dataset2/multiscan/scene_00005_00
```

## Performance Considerations

- **Sequential Processing**: Scenes are processed sequentially (not in parallel)
- **Memory**: Each scene is processed independently, so memory usage is per-scene
- **Disk I/O**: Per-scene results are saved immediately, aggregation happens at the end

## Troubleshooting

### Issue: "No scenes found to process"

**Solution**: Check that scene directories exist in `exp_root` and follow the naming pattern `scene_*`:

```bash
ls -d {exp_root}/scene_*
```

### Issue: "KeyError: 'aggregation_output_dir'"

**Solution**: This issue has been fixed in the latest version. Use `.get()` for safe key access.

### Issue: Scene data path not found

**Solution**: Verify that `dataset_root` is correctly configured and scene directories exist:

```bash
ls -d {dataset_root}/scene_*
```

## Example Workflow

Here's a complete example of processing multiple scenes:

```bash
# 1. Set up environment
conda activate 3Dsiglip

# 2. Configure multi-scene pipeline
# Edit configs/inference/full_pipeline.yaml:
#   experiment.multi_scene_mode: true
#   experiment.exp_root: /path/to/experiment
#   experiment.scenes: all

# 3. Run full pipeline
python scripts/run_full_pipeline.py \
  --config configs/inference/full_pipeline.yaml \
  --multi-scene \
  --skip-aggregation  # If aggregation already done

# 4. Check results
cat {exp_root}/aggregated_metrics.json
cat {exp_root}/multi_scene_summary.md

# 5. Check per-scene results
cat {exp_root}/scene_00005_00/pipeline_results.json
cat {exp_root}/scene_00093_01/pipeline_results.json
```

## Future Enhancements

Potential improvements for future versions:

- [ ] Parallel scene processing (multi-GPU support)
- [ ] Resume from checkpoint (skip already-processed scenes)
- [ ] Progressive reporting (update summary as scenes complete)
- [ ] Visualization dashboard for multi-scene results
- [ ] Export to CSV/Excel for analysis

## Summary

The multi-scene pipeline provides a scalable way to process entire experiment directories with automatic metric aggregation. It maintains backward compatibility while adding powerful new features for large-scale experiments.

For questions or issues, please refer to the main project documentation or create an issue in the GitHub repository.
