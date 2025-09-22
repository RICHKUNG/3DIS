#!/usr/bin/env bash
# Orchestrate the current 3DIS experiment: Semantic-SAM candidate generation + SAM2 tracking.
# Mirrors the workflow described in Agent.md / README.

set -euo pipefail

DEFAULT_SEMANTIC_SAM_CKPT="/media/Pluto/richkung/Semantic-SAM/checkpoints/swinl_only_sam_many2many.pth"
DEFAULT_SAM2_CFG="/media/Pluto/richkung/SAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CKPT="/media/Pluto/richkung/SAM2/checkpoints/sam2.1_hiera_large.pt"
DEFAULT_DATA_PATH="/media/public_dataset2/multiscan/scene_00065_00/outputs/color"
DEFAULT_OUTPUT_ROOT="/media/Pluto/richkung/My3DIS/outputs/scene_00065_00"

usage() {
  cat <<'USAGE'
Usage: run_experiment.sh [options]

Fixed paths:
  data path   /media/public_dataset2/multiscan/scene_00065_00/outputs/color
  output root /media/Pluto/richkung/My3DIS/outputs/scene_00065_00

Options:
  --levels L1,L2,...          Levels to process (default: 2,4,6)
  --frames start:end:step     Frame sampling string (default: 1200:1600:20)
  --min-area PIXELS           Filter masks below this area (default: 300)
  --stability THR             Minimum stability_score (default: 0.9)
  --ssam-freq N               Run Semantic-SAM every N frames (default: 1)
  --sam2-max-propagate N      Max frames to propagate in each direction (default: unlimited)
  --sam-ckpt PATH             Override Semantic-SAM checkpoint (default set in script)
  --sam2-cfg PATH             Override SAM2 YAML config (default set in script)
  --sam2-ckpt PATH            Override SAM2 checkpoint (default set in script)
  --semantic-env NAME         Conda env for Semantic-SAM stage (default: Semantic-SAM)
  --sam2-env NAME             Conda env for SAM2 stage (default: SAM2)
  --no-timestamp              Write directly into output-root (otherwise timestamped subdir)
  --dry-run                   Print the commands without executing
  -h, --help                  Show this help

Examples:
  # 基本使用
  ./run_experiment.sh

  # 每2張圖片執行一次SSAM，SAM2最多向前後各傳播30幀
  ./run_experiment.sh --ssam-freq 2 --sam2-max-propagate 30

  # 自定義路徑和參數
  ./run_experiment.sh \
    --levels 2,4 \
    --frames 1000:2000:10 \
    --ssam-freq 3 \
    --sam2-max-propagate 50 \
    --min-area 500

  # 乾運行檢查命令
  ./run_experiment.sh --ssam-freq 2 --sam2-max-propagate 30 --dry-run
USAGE
}

DATA_PATH="$DEFAULT_DATA_PATH"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
SAM_CKPT="$DEFAULT_SEMANTIC_SAM_CKPT"
SAM2_CFG="$DEFAULT_SAM2_CFG"
SAM2_CKPT="$DEFAULT_SAM2_CKPT"
LEVELS="2,4,6"
FRAMES="1200:1600:20"
MIN_AREA="300"
STABILITY="1.0"
SSAM_FREQ="1"
SAM2_MAX_PROPAGATE=""
SEM_ENV="Semantic-SAM"
SAM2_ENV="SAM2"
NO_TIMESTAMP=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sam-ckpt) SAM_CKPT="$2"; shift 2;;
    --sam2-cfg) SAM2_CFG="$2"; shift 2;;
    --sam2-ckpt) SAM2_CKPT="$2"; shift 2;;
    --levels) LEVELS="$2"; shift 2;;
    --frames) FRAMES="$2"; shift 2;;
    --min-area) MIN_AREA="$2"; shift 2;;
    --stability) STABILITY="$2"; shift 2;;
    --ssam-freq) SSAM_FREQ="$2"; shift 2;;
    --sam2-max-propagate) SAM2_MAX_PROPAGATE="$2"; shift 2;;
    --semantic-env) SEM_ENV="$2"; shift 2;;
    --sam2-env) SAM2_ENV="$2"; shift 2;;
    --no-timestamp) NO_TIMESTAMP=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GEN_SCRIPT="$SCRIPT_DIR/generate_candidates.py"
TRACK_SCRIPT="$SCRIPT_DIR/track_from_candidates.py"

if [[ ! -f "$GEN_SCRIPT" || ! -f "$TRACK_SCRIPT" ]]; then
  echo "Error: expected scripts not found under $SCRIPT_DIR" >&2
  exit 1
fi

PYTHON_ABS=${PYTHON_ABS:-python3}

OUTPUT_ROOT_ABS=$($PYTHON_ABS -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_ROOT")
DATA_PATH_ABS=$($PYTHON_ABS -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$DATA_PATH")

mkdir -p "$OUTPUT_ROOT_ABS"

stage1_cmd=(conda run --live-stream -n "$SEM_ENV" python -u "$GEN_SCRIPT" \
  --data-path "$DATA_PATH_ABS" \
  --levels "$LEVELS" \
  --frames "$FRAMES" \
  --sam-ckpt "$SAM_CKPT" \
  --output "$OUTPUT_ROOT_ABS" \
  --min-area "$MIN_AREA" \
  --stability-threshold "$STABILITY" \
  --ssam-freq "$SSAM_FREQ")

if [[ -n "$SAM2_MAX_PROPAGATE" ]]; then
  stage1_cmd+=(--sam2-max-propagate "$SAM2_MAX_PROPAGATE")
fi

if [[ $NO_TIMESTAMP -eq 1 ]]; then
  stage1_cmd+=(--no-timestamp)
fi

stage2_cmd_base=(conda run --live-stream -n "$SAM2_ENV" python -u "$TRACK_SCRIPT" \
  --data-path "$DATA_PATH_ABS" \
  --sam2-cfg "$SAM2_CFG" \
  --sam2-ckpt "$SAM2_CKPT" \
  --levels "$LEVELS")

if [[ -n "$SAM2_MAX_PROPAGATE" ]]; then
  stage2_cmd_base+=(--sam2-max-propagate "$SAM2_MAX_PROPAGATE")
fi

format_duration() {
  local total=$1
  local hours=$((total / 3600))
  local mins=$(((total % 3600) / 60))
  local secs=$((total % 60))
  if [[ $hours -gt 0 ]]; then
    printf '%02d:%02d:%02d' "$hours" "$mins" "$secs"
  else
    printf '%02d:%02d' "$mins" "$secs"
  fi
}

declare -a STAGE_SUMMARY=()

run_stage() {
  local label="$1"
  shift || true

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] $label"
    if [[ $# -gt 0 ]]; then
      printf '  %q' "$@"
      printf '\n'
    fi
    return 0
  fi

  echo "→ $label"
  local start_ts
  start_ts=$(date +%s)
  if "$@"; then
    local end_ts duration friendly
    end_ts=$(date +%s)
    duration=$((end_ts - start_ts))
    friendly=$(format_duration "$duration")
    echo "  completed in $friendly"
    STAGE_SUMMARY+=("$label: $friendly")
  else
    local status=$?
    local end_ts duration friendly
    end_ts=$(date +%s)
    duration=$((end_ts - start_ts))
    friendly=$(format_duration "$duration")
    echo "  failed after $friendly" >&2
    return "$status"
  fi
}

SCRIPT_START_TS=$(date +%s)

run_stage "Stage 1: Semantic-SAM candidate generation" "${stage1_cmd[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
  if [[ $NO_TIMESTAMP -eq 1 ]]; then
    RUN_DIR="$OUTPUT_ROOT_ABS"
  else
    RUN_DIR="$OUTPUT_ROOT_ABS/<timestamp>"
  fi
else
  if [[ $NO_TIMESTAMP -eq 1 ]]; then
    RUN_DIR="$OUTPUT_ROOT_ABS"
  else
    latest=$(ls -1dt "$OUTPUT_ROOT_ABS"/* 2>/dev/null | head -n1 || true)
    if [[ -z "$latest" ]]; then
      echo "Could not locate timestamped output under $OUTPUT_ROOT_ABS" >&2
      exit 1
    fi
    RUN_DIR="$latest"
  fi

  if [[ ! -f "$RUN_DIR/manifest.json" ]]; then
    echo "manifest.json not found in $RUN_DIR; stage 1 may have failed" >&2
    exit 1
  fi
fi

echo "Outputs located at: $RUN_DIR"

stage2_cmd=(${stage2_cmd_base[@]} --candidates-root "$RUN_DIR" --output "$RUN_DIR")

echo
run_stage "Stage 2: SAM2 tracking" "${stage2_cmd[@]}"

if [[ $DRY_RUN -ne 1 ]]; then
  echo
  echo "Timing summary:"
  for item in "${STAGE_SUMMARY[@]}"; do
    echo "  - $item"
  done
  total_elapsed=$(format_duration $(( $(date +%s) - SCRIPT_START_TS )))
  echo "  - Total: $total_elapsed"
fi

echo "\nDone. Final artifacts under: $RUN_DIR"
