#!/usr/bin/env bash
# Sweep run_experiment.sh across multiple frame steps, min-area, and stability values.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_EXPERIMENT="$SCRIPT_DIR/run_experiment.sh"

if [[ ! -x "$RUN_EXPERIMENT" ]]; then
  echo "run_experiment.sh not found or not executable at $RUN_EXPERIMENT" >&2
  exit 1
fi

# Edit the lists below or export environment variables with the same names
# before launching the script (e.g. FREQUENCIES="20 30").
FREQUENCIES=(${FREQUENCIES:-20})
MIN_AREAS=(${MIN_AREAS:-200})
STABILITIES=(${STABILITIES:-0.8})

# Frame range prefix (start:end) is kept separate from the step so that only
# the sampling frequency needs to change for each sweep item.
FRAME_RANGE_PREFIX=${FRAME_RANGE_PREFIX:-"1000:4000"}
LEVELS=${LEVELS:-"2,4,6"}

LOG_ROOT=${LOG_ROOT:-"$SCRIPT_DIR/logs"}
mkdir -p "$LOG_ROOT"

SWEEP_STAMP=${SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}
SWEEP_LOG_DIR="$LOG_ROOT/$SWEEP_STAMP"
mkdir -p "$SWEEP_LOG_DIR"

print_ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

SWEEP_SUMMARY="$SWEEP_LOG_DIR/sweep_summary.log"
: > "$SWEEP_SUMMARY"
echo "[$(print_ts)] Sweep logs directory: $SWEEP_LOG_DIR" | tee -a "$SWEEP_SUMMARY"

run_count=0
fail_count=0

for freq in "${FREQUENCIES[@]}"; do
  frames="${FRAME_RANGE_PREFIX}:${freq}"
  for min_area in "${MIN_AREAS[@]}"; do
    for stability in "${STABILITIES[@]}"; do
      ((run_count++)) || true
      label="freq${freq}_area${min_area}_stab${stability}"
      stamp=$(date +%Y%m%d_%H%M%S)
      log_file="$SWEEP_LOG_DIR/run_${label}_${stamp}.log"

      echo "[$(print_ts)] Starting $label → frames=$frames min-area=$min_area stability=$stability" | tee -a "$log_file"

      if "$RUN_EXPERIMENT" \
        --levels "$LEVELS" \
        --frames "$frames" \
        --min-area "$min_area" \
        --stability "$stability" \
        >>"$log_file" 2>&1; then
        echo "[$(print_ts)] Completed $label" | tee -a "$log_file"
      else
        status=$?
        echo "[$(print_ts)] FAILED $label (exit $status)" | tee -a "$log_file"
        ((fail_count++)) || true
      fi

      echo | tee -a "$log_file"
    done
  done
done

echo "[$(print_ts)] Sweep finished. Runs attempted: $run_count, failures: $fail_count" | tee -a "$SWEEP_SUMMARY"
