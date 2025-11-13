#!/bin/bash
# Quick test script to verify pairing improvements
# Tests the queries that previously had 0 valid pairs

set -e

CONFIG="configs/inference/full_pipeline.yaml"
LOG_DIR="logs/pairing_test"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "Testing Pairing Improvements"
echo "================================================================================"
echo ""
echo "Target queries (previously 0 valid pairs):"
echo "  1. door-frame (label=4008)"
echo "  2. trash_can-door (label=25002)"
echo "  3. printer-tray (label=140017)"
echo ""
echo "Configuration:"
echo "  - containment_threshold: 0.001 (was 0.1)"
echo "  - scale_range: [0.0001, 20.0] (was [0.001, 0.99])"
echo ""
echo "================================================================================"
echo ""

# Check if aggregation output exists
AGG_DIR="outputs/experiments/test_1108_ssam4_filter500_fill500_propinf/scene_00005_00/aggregation_output"
if [ ! -d "$AGG_DIR" ]; then
    echo "ERROR: Aggregation output not found at $AGG_DIR"
    echo "Please run aggregation first or update the path in this script"
    exit 1
fi

# Run inference with new config
echo "Running inference with improved pairing config..."
echo "Log: $LOG_DIR/inference_test_$(date +%Y%m%d_%H%M%S).log"
echo ""

conda run -n SAM2 --live-stream \
    python scripts/run_full_pipeline.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/inference_test_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "================================================================================"
echo "Test Complete"
echo "================================================================================"
echo ""
echo "Please check the log for:"
echo "  - Queries that now have predictions (was 0 before)"
echo "  - Valid pair counts for each query"
echo "  - Final prediction counts after NMS"
echo ""
echo "Key log lines to look for:"
echo "  'Created N valid pairs' - should be > 0 for previously failing queries"
echo "  'Found N predictions' - final count after NMS"
echo ""
