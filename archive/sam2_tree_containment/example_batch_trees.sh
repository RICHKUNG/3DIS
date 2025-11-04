#!/bin/bash
# Example: Batch build SAM2 trees for all scenes in a specific experiment
# 範例：對特定實驗的所有 scenes 批量建立 SAM2 trees

set -e  # Exit on error

# ============================================================
# Configuration (修改這裡的設定)
# ============================================================

# 實驗目錄（修改為你的實驗路徑）
EXPERIMENT_ROOT="/media/Pluto/richkung/My3DIS/outputs/experiments/v4_246_ssam2_filter500_fill5k_prop30_iou06_ds03"

# Hyperparameters
CONTAINMENT_THRESHOLD=0.95       # Parent-child containment threshold
TEMPORAL_OVERLAP_THRESHOLD=0.3   # Temporal overlap threshold
LEVELS="2,4,6"                   # Levels to process

# 其他選項
FORCE=false                      # Set to true to rebuild existing trees
DRY_RUN=false                    # Set to true for preview only
SCENE_FILTER=""                  # e.g., "scene_00065_*" to filter scenes
VERBOSE=false                    # Set to true for debug logging

# ============================================================
# Script (不需要修改)
# ============================================================

# 啟動 SAM2 環境
echo "Activating SAM2 environment..."
conda activate SAM2 2>/dev/null || true

# 檢查實驗目錄
if [ ! -d "$EXPERIMENT_ROOT" ]; then
    echo "Error: Experiment directory not found: $EXPERIMENT_ROOT"
    echo "Please update EXPERIMENT_ROOT variable in this script"
    exit 1
fi

echo "========================================"
echo "Batch SAM2 Tree Building"
echo "========================================"
echo "Experiment: $(basename "$EXPERIMENT_ROOT")"
echo "Levels: $LEVELS"
echo "Containment threshold: $CONTAINMENT_THRESHOLD"
echo "Temporal overlap threshold: $TEMPORAL_OVERLAP_THRESHOLD"
echo ""

# 構建命令
CMD="python scripts/batch_build_sam2_trees.py"
CMD="$CMD --experiment-root $EXPERIMENT_ROOT"
CMD="$CMD --levels $LEVELS"
CMD="$CMD --containment-threshold $CONTAINMENT_THRESHOLD"
CMD="$CMD --temporal-overlap-threshold $TEMPORAL_OVERLAP_THRESHOLD"

if [ "$FORCE" = true ]; then
    CMD="$CMD --force"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ -n "$SCENE_FILTER" ]; then
    CMD="$CMD --scene-filter $SCENE_FILTER"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# 執行
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "✅ Batch processing completed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Verify results:"
echo "     ls -la $EXPERIMENT_ROOT/scene_*/relations/sam2_tree.json"
echo ""
echo "  2. Check statistics:"
echo "     cat $EXPERIMENT_ROOT/scene_00005_00/relations/sam2_tree.json | jq '.statistics'"
echo ""
echo "  3. Compare with tracked objects:"
echo "     python3 -c \""
echo "       import json"
echo "       tree = json.load(open('$EXPERIMENT_ROOT/scene_00005_00/relations/sam2_tree.json'))"
echo "       index = json.load(open('$EXPERIMENT_ROOT/scene_00005_00/level_2/index.json'))"
echo "       print(f'Tree nodes: {tree[\\\"statistics\\\"][\\\"total_objects\\\"]}')"
echo "       print(f'Tracked objects: {len(index[\\\"objects\\\"])}')"
echo "     \""
echo ""
