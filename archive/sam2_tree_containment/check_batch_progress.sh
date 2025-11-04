#!/bin/bash
# Check progress of all batch SAM2 tree building jobs

echo "========================================="
echo "Batch SAM2 Tree Building Progress"
echo "========================================="
echo ""
echo "Timestamp: $(date)"
echo ""

# Check all log files
for log_file in logs/batch_sam2_trees_*.log; do
    if [ -f "$log_file" ]; then
        exp_name=$(basename "$log_file" .log | sed 's/batch_sam2_trees_//')

        # Get last few lines to check progress
        last_scene=$(tail -50 "$log_file" | grep -oP 'Processing \K[^\.]+' | tail -1)
        total_scenes=$(grep -oP 'Found \K\d+' "$log_file" | head -1)
        success=$(grep -oP 'Successfully built: \K\d+' "$log_file" | tail -1)
        skipped=$(grep -oP 'Skipped: \K\d+' "$log_file" | tail -1)
        failed=$(grep -oP 'Failed: \K\d+' "$log_file" | tail -1)
        completed=$(grep -q "Batch tree building completed" "$log_file" && echo "✅ DONE" || echo "⏳ Running")

        echo "[$completed] $exp_name"
        if [ -n "$total_scenes" ]; then
            echo "  Total scenes: $total_scenes"
        fi
        if [ -n "$last_scene" ]; then
            echo "  Current: $last_scene"
        fi
        if [ -n "$success" ]; then
            echo "  Built: $success | Skipped: $skipped | Failed: $failed"
        fi
        echo ""
    fi
done

echo "========================================="
echo "To check individual logs:"
echo "  tail -f logs/batch_sam2_trees_<experiment>.log"
echo "========================================="
