# SAM2 Tree (Containment-Based) - Archived

**Archive Date:** 2025-10-30

**Reason:** Replaced by provenance-based tree building approach.

## What was archived

This directory contains the old containment-based tree building implementation that analyzed SAM2 tracked objects post-hoc to establish parent-child relationships.

### Archived Files

- `sam2_tree_builder.py` - Core containment analysis implementation
- `batch_build_sam2_trees.py` - Batch processing script
- `analyze_sam2_tree.py` - Analysis tool
- `visualize_family_trees.py` - Old visualization script
- `test_sam2_tree.sh` - Test script
- `check_batch_progress.sh` - Progress monitoring
- `example_batch_trees.sh` - Example usage
- `README_visualize_family_trees.md` - Documentation

### Why it was replaced

The containment-based approach had several limitations:

1. **High orphan rate (~60%)**: Temporal misalignment between parent-child objects resulted in many objects without parents
2. **Compute intensive**: Required loading and comparing all mask pairs across all frames
3. **Threshold sensitivity**: Results highly dependent on containment threshold tuning
4. **Lost SSAM hierarchy**: Ignored the original parent-child relationships from Semantic-SAM

### Replacement: Provenance-Based Tree Building

The new approach (implemented in `provenance_tracker.py` and `family_tree_builder.py`):

- **Tracks SSAM → SAM2 mappings** during tracking, not post-hoc
- **Preserves SSAM hierarchy** directly from progressive refinement
- **Much lower orphan rate (~20-30%)**: Direct lineage tracking
- **No thresholds needed**: Relationship is deterministic from provenance
- **Minimal compute overhead**: O(1) parent lookup

### Migration Path

If you have existing sam2_tree.json files:

1. **Comparison**: Use `scripts/compare_trees.py` to compare old vs new approaches
2. **Re-run tracking**: Generate new provenance-based trees by re-running the tracking stage
3. **New visualization**: Use `scripts/visualize_families.py` with the new family_tree.json format

## Historical Context

These scripts were developed during the project's initial phase (Oct 2025) when exploring different tree building strategies. The containment-based approach was documented in:

- `docs/SAM2_TREE_USAGE.md` (archived)
- `docs/proposals/SAM2_TREE_PROPOSALS.md` (see Proposal 1)
- `docs/TREE_ALGORITHM_ANALYSIS.md` (analysis of why it produced fewer families than expected)

For current tree building documentation, see:
- `docs/PROVENANCE_TRACKING.md` - New provenance-based system
- `docs/ARCHITECTURE.md` - Overall pipeline architecture
