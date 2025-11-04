#!/bin/bash
# Regenerate family trees and reports for an experiment
# Usage: ./scripts/regenerate_family_trees.sh <experiment_dir>

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 outputs/experiments/test_1030_1_ssam4_filter1000"
    exit 1
fi

EXPERIMENT_DIR="$1"

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory $EXPERIMENT_DIR does not exist"
    exit 1
fi

echo "=========================================="
echo "Regenerating family trees and reports"
echo "Experiment: $EXPERIMENT_DIR"
echo "=========================================="

# Set PYTHONPATH
export PYTHONPATH=$(pwd)/src

# Find all scene directories
SCENE_DIRS=$(find "$EXPERIMENT_DIR" -mindepth 1 -maxdepth 1 -type d -name "scene_*" | sort)

if [ -z "$SCENE_DIRS" ]; then
    echo "Warning: No scene directories found in $EXPERIMENT_DIR"
    exit 1
fi

TOTAL_SCENES=$(echo "$SCENE_DIRS" | wc -l)
CURRENT=0

# Regenerate family trees for each scene
for SCENE_DIR in $SCENE_DIRS; do
    CURRENT=$((CURRENT + 1))
    SCENE_NAME=$(basename "$SCENE_DIR")

    echo ""
    echo "[$CURRENT/$TOTAL_SCENES] Processing $SCENE_NAME..."

    # Check if provenance trees exist
    if [ ! -d "$SCENE_DIR/relations" ]; then
        echo "  ⚠ No relations directory, skipping"
        continue
    fi

    PROV_TREES=$(find "$SCENE_DIR/relations" -name "provenance_tree_L*.json" 2>/dev/null | wc -l)
    if [ "$PROV_TREES" -eq 0 ]; then
        echo "  ⚠ No provenance trees found, skipping"
        continue
    fi

    # Regenerate family tree
    echo "  → Regenerating family_tree.json..."
    python3 -m my3dis.family_tree_builder \
        --run-dir "$SCENE_DIR" \
        --levels 2 4 6 \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ Family tree regenerated"
    else
        echo "  ✗ Failed to regenerate family tree"
        continue
    fi

    # Check statistics
    if [ -f "$SCENE_DIR/relations/family_tree.json" ]; then
        STATS=$(python3 -c "
import json
with open('$SCENE_DIR/relations/family_tree.json') as f:
    tree = json.load(f)
    stats = tree['statistics']
    print(f\"Objects: {stats['total_objects']}, Families: {stats['total_families']}, Orphans: {stats['orphan_count']}\")
" 2>/dev/null)
        echo "  $STATS"
    fi
done

echo ""
echo "=========================================="
echo "Regenerating experiment report..."
echo "=========================================="

# Regenerate check.md
python3 -c "
from pathlib import Path
from my3dis.workflow.reporting import generate_experiment_check_report

exp_dir = Path('$EXPERIMENT_DIR')
output_path = generate_experiment_check_report(exp_dir)
print(f'✓ Report saved: {output_path}')
"

echo ""
echo "=========================================="
echo "Regeneration complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the updated check.md:"
echo "   cat $EXPERIMENT_DIR/check.md"
echo ""
echo "2. Regenerate visualizations (if needed):"
echo "   PYTHONPATH=src python3 scripts/visualize_families.py \\"
echo "     --tree <scene_dir>/relations/family_tree.json \\"
echo "     --data-path <data_path> \\"
echo "     --output-dir <output_dir>"
