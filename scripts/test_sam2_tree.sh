#!/bin/bash
# Test script for SAM2-based tree building
# Tests the containment-based tree algorithm on a sample scene

set -e  # Exit on error

echo "========================================"
echo "SAM2 Tree Building Test Script"
echo "========================================"
echo ""

# Configuration
SCENE="scene_00065_00"
TEST_ROOT="/media/Pluto/richkung/My3DIS/outputs/test_sam2_tree"
DATA_PATH="/media/public_dataset2/multiscan/${SCENE}/outputs/color"
LEVELS="2,4,6"

# Hyperparameters (可調整)
CONTAINMENT_THRESHOLD=0.95
TEMPORAL_OVERLAP_THRESHOLD=0.3

echo "Configuration:"
echo "  Scene: ${SCENE}"
echo "  Levels: ${LEVELS}"
echo "  Containment threshold: ${CONTAINMENT_THRESHOLD}"
echo "  Temporal overlap threshold: ${TEMPORAL_OVERLAP_THRESHOLD}"
echo ""

# Check if tracking already completed
if [ -d "${TEST_ROOT}/${SCENE}/level_2" ]; then
    echo "Tracking outputs already exist at ${TEST_ROOT}/${SCENE}"
    echo "Skipping tracking stage..."
    echo ""
else
    echo "Error: Tracking outputs not found"
    echo "Please run SAM2 tracking first, or use an existing experiment directory"
    echo ""
    echo "Example:"
    echo "  conda activate SAM2"
    echo "  PYTHONPATH=src python src/my3dis/track_from_candidates.py \\"
    echo "    --data-path ${DATA_PATH} \\"
    echo "    --candidates-root outputs/experiments/some_experiment/${SCENE} \\"
    echo "    --output ${TEST_ROOT}/${SCENE} \\"
    echo "    --levels ${LEVELS}"
    exit 1
fi

# Build SAM2 tree
echo "Step 1: Building SAM2-based containment tree..."
echo "----------------------------------------"

conda activate SAM2 2>/dev/null || true

PYTHONPATH=src python3 << EOF
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from my3dis.sam2_tree_builder import build_sam2_containment_tree

tree_path = build_sam2_containment_tree(
    run_dir='${TEST_ROOT}/${SCENE}',
    levels=[int(x) for x in '${LEVELS}'.split(',')],
    containment_threshold=${CONTAINMENT_THRESHOLD},
    temporal_overlap_threshold=${TEMPORAL_OVERLAP_THRESHOLD},
    output_filename='sam2_tree.json',
)

print(f"\n✅ SAM2 tree saved → {tree_path}")
EOF

echo ""
echo "Step 2: Comparing node counts..."
echo "----------------------------------------"

python3 << EOF
import json
from pathlib import Path

run_dir = Path('${TEST_ROOT}/${SCENE}')

# Load SSAM tree (if exists)
ssam_tree_path = run_dir / 'relations' / 'tree.json'
if ssam_tree_path.exists():
    with ssam_tree_path.open('r') as f:
        ssam_tree = json.load(f)
    ssam_nodes = sum(len(v) for v in ssam_tree.values() if isinstance(v, list))
    print(f"SSAM tree nodes: {ssam_nodes}")
else:
    print("SSAM tree not found")
    ssam_nodes = 0

# Load SAM2 tree
sam2_tree_path = run_dir / 'relations' / 'sam2_tree.json'
with sam2_tree_path.open('r') as f:
    sam2_tree = json.load(f)

# Extract statistics
stats = sam2_tree.get('statistics', {})
hierarchy = sam2_tree.get('hierarchy', {})

print(f"SAM2 tree nodes: {stats.get('total_objects', 0)}")
print(f"  - Roots: {stats.get('roots', 0)}")
print(f"  - Families (with children): {stats.get('families', 0)}")
print(f"  - Max depth: {stats.get('max_depth', 0)}")
print("")

# Per-level breakdown
print("Per-level breakdown:")
for level_str in sorted(hierarchy.keys(), key=lambda x: int(x)):
    level_objs = hierarchy[level_str]
    print(f"  Level {level_str}: {len(level_objs)} objects")

print("")

# Load tracked objects from index.json
print("Tracked objects (from index.json):")
for level in [2, 4, 6]:
    index_path = run_dir / f'level_{level}' / 'index.json'
    if index_path.exists():
        with index_path.open('r') as f:
            index = json.load(f)
        tracked_objs = len(index.get('objects', {}))
        tree_objs = len(hierarchy.get(str(level), {}))
        match = "✅" if tracked_objs == tree_objs else "❌"
        print(f"  Level {level}: {tracked_objs} tracked, {tree_objs} in tree {match}")

print("")

# Ratio
if ssam_nodes > 0:
    ratio = stats.get('total_objects', 0) / ssam_nodes
    print(f"Tree expansion ratio: {ratio:.1f}x (SAM2 / SSAM)")
EOF

echo ""
echo "Step 3: Validating tree format..."
echo "----------------------------------------"

if [ -f "scripts/validate_formats.py" ]; then
    PYTHONPATH=src python3 scripts/validate_formats.py \
        --file "${TEST_ROOT}/${SCENE}/relations/sam2_tree.json" \
        --type relations
else
    echo "Validation script not found, skipping validation"
fi

echo ""
echo "========================================"
echo "✅ Test completed successfully!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - SAM2 tree: ${TEST_ROOT}/${SCENE}/relations/sam2_tree.json"
echo "  - SSAM tree: ${TEST_ROOT}/${SCENE}/relations/tree.json (if exists)"
echo ""
echo "Next steps:"
echo "  1. Inspect the tree: cat ${TEST_ROOT}/${SCENE}/relations/sam2_tree.json | jq '.statistics'"
echo "  2. Compare with SSAM tree"
echo "  3. Adjust hyperparameters if needed:"
echo "     - containment_threshold (default: 0.95)"
echo "     - temporal_overlap_threshold (default: 0.3)"
echo ""
