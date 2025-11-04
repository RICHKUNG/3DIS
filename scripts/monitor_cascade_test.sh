#!/bin/bash
# Monitor cascade filtering test progress

LOG_FILE="logs/test_cascade/run_$(ls -t logs/test_cascade/ | head -1)"
OUTPUT_DIR="outputs/experiments/test_cascade_filtering_1104/scene_00005_00"

echo "=== Cascade Filtering Test Monitor ==="
echo "Log file: $LOG_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if test is running
if ps aux | grep -q "[r]un_workflow.py.*test_cascade"; then
    echo "Status: RUNNING"
    echo ""

    # Show recent progress
    echo "Recent progress:"
    tail -20 "$LOG_FILE" | grep -E "INFO|Cascade filtering|Level.*candidates|objects.*families"
    echo ""

    # Check SSAM stage
    if [ -d "$OUTPUT_DIR/level_2/raw" ]; then
        echo "SSAM Stage: COMPLETE"
        echo "  Raw masks generated:"
        ls -lh "$OUTPUT_DIR"/level_*/raw/manifest.json 2>/dev/null | wc -l
    elif grep -q "Semantic-SAM candidate generation started" "$LOG_FILE"; then
        echo "SSAM Stage: IN PROGRESS"
        grep "Frame.*added.*gap-fill" "$LOG_FILE" | tail -3
    fi

    echo ""

    # Check SAM2 tracking stage
    if [ -f "$OUTPUT_DIR/level_2/video_segments_L02.npz" ]; then
        echo "SAM2 Tracking: COMPLETE or IN PROGRESS"
        ls -lh "$OUTPUT_DIR"/level_*/video_segments_*.npz 2>/dev/null
    fi

else
    echo "Status: COMPLETED or NOT RUNNING"
    echo ""

    # Show final results
    if [ -f "$OUTPUT_DIR/relations/family_tree.json" ]; then
        echo "=== Final Results ==="
        python3 << 'EOF'
import json

with open("outputs/experiments/test_cascade_filtering_1104/scene_00005_00/relations/family_tree.json") as f:
    tree = json.load(f)
    stats = tree.get('statistics', {})

print(f"Total objects: {stats.get('total_objects', 'N/A')}")
print(f"Total families: {stats.get('total_families', 'N/A')}")
print(f"Orphans: {stats.get('orphan_count', 'N/A')} ({stats.get('orphan_count', 0) / max(stats.get('total_objects', 1), 1) * 100:.2f}%)")
print(f"Objects per level:")
for level, count in sorted(stats.get('objects_per_level', {}).items()):
    print(f"  {level}: {count}")

# Check visualization
import os
viz_dir = "outputs/experiments/test_cascade_filtering_1104/scene_00005_00/visualizations"
if os.path.exists(viz_dir):
    viz_count = len([f for f in os.listdir(viz_dir) if f.endswith('.png')])
    print(f"\nVisualization images: {viz_count}")
EOF
    else
        echo "Results not yet available"
    fi
fi

echo ""
echo "To view full log: tail -f $LOG_FILE"
