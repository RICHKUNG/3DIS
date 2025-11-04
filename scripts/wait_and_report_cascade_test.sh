#!/bin/bash
# Wait for cascade filtering test to complete and report results

OUTPUT_DIR="outputs/experiments/test_cascade_filtering_1104/scene_00005_00"
LOG_FILE="logs/scene_workers/test_cascade_20251104_201936/000_scene_00005_00_pid2053524.log"
FAMILY_TREE="$OUTPUT_DIR/relations/family_tree.json"

echo "Waiting for cascade filtering test to complete..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Wait for family tree to be generated (indicates completion)
while [ ! -f "$FAMILY_TREE" ]; do
    # Check if process is still running
    if ! ps aux | grep -q "[r]un_workflow.py.*test_cascade"; then
        echo "ERROR: Workflow process ended but no family_tree.json found!"
        echo "Check logs for errors:"
        echo "  $LOG_FILE"
        exit 1
    fi

    # Show current stage
    CURRENT_STAGE=$(tail -1 "$LOG_FILE" | grep -oE "Stage [A-Za-z ]+:" || echo "Processing...")
    if [ -n "$CURRENT_STAGE" ]; then
        echo -ne "\rCurrent: $CURRENT_STAGE                    "
    fi

    sleep 10
done

echo ""
echo "✅ Test completed!"
echo ""

# Extract and display results
PYTHONPATH=src python3 << 'EOF'
import json
import os

output_dir = "outputs/experiments/test_cascade_filtering_1104/scene_00005_00"
family_tree_path = os.path.join(output_dir, "relations/family_tree.json")

if not os.path.exists(family_tree_path):
    print("ERROR: family_tree.json not found!")
    exit(1)

with open(family_tree_path) as f:
    tree = json.load(f)

stats = tree.get('statistics', {})
total_objs = stats.get('total_objects', 0)
orphan_count = stats.get('orphan_count', 0)
orphan_rate = (orphan_count / max(total_objs, 1)) * 100

print("=" * 60)
print("CASCADE FILTERING TEST RESULTS")
print("=" * 60)
print()
print(f"Total Objects: {total_objs}")
print(f"Total Families: {stats.get('total_families', 'N/A')}")
print(f"Orphan Count: {orphan_count}")
print(f"Orphan Rate: {orphan_rate:.2f}%")
print()
print("Objects per Level:")
for level in ['L2', 'L4', 'L6']:
    count = stats.get('objects_per_level', {}).get(level, 0)
    print(f"  {level}: {count}")
print()

# Check visualizations
viz_dir = os.path.join(output_dir, "visualizations")
if os.path.exists(viz_dir):
    viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
    print(f"Visualization Images: {len(viz_files)}")
    if viz_files:
        print(f"  Sample: {viz_files[0]}")
else:
    print("Visualization: Not generated")

print()
print("=" * 60)

# Save summary to file
summary_path = os.path.join(output_dir, "CASCADE_TEST_SUMMARY.txt")
with open(summary_path, 'w') as f:
    f.write("CASCADE FILTERING TEST SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  - cascade_filtering: true\n")
    f.write(f"  - stability_threshold: null\n")
    f.write(f"  - min_area: 100\n\n")
    f.write(f"Results:\n")
    f.write(f"  - Total Objects: {total_objs}\n")
    f.write(f"  - Total Families: {stats.get('total_families', 'N/A')}\n")
    f.write(f"  - Orphan Count: {orphan_count}\n")
    f.write(f"  - Orphan Rate: {orphan_rate:.2f}%\n\n")
    f.write(f"Objects per Level:\n")
    for level in ['L2', 'L4', 'L6']:
        count = stats.get('objects_per_level', {}).get(level, 0)
        f.write(f"  - {level}: {count}\n")

print(f"Summary saved to: {summary_path}")
EOF

echo ""
echo "To view detailed logs:"
echo "  tail -100 $LOG_FILE"
