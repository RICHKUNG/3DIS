#!/bin/bash
# GPU 記憶體使用測試腳本
# 用於測試優化後的配置是否解決 OOM 問題

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🧪 GPU 記憶體使用測試"
echo "=========================================="

# 檢查 GPU
echo -e "\n${YELLOW}1. 檢查 GPU 狀態...${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 記錄初始記憶體
echo -e "\n${YELLOW}2. 記錄初始 GPU 記憶體...${NC}"
INITIAL_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "初始 GPU 記憶體使用: ${INITIAL_MEM} MB"

# 測試單場景
TEST_SCENE="${1:-scene_00005_00}"
CONFIG="${2:-configs/multiscan/base.yaml}"

echo -e "\n${YELLOW}3. 開始測試單場景: ${TEST_SCENE}${NC}"
echo "配置檔: ${CONFIG}"

# 在背景啟動記憶體監控
LOG_FILE="logs/memory_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
python scripts/monitor_gpu.py --interval 2 --log-file "$LOG_FILE" --alert-threshold 85 &
MONITOR_PID=$!

echo -e "${GREEN}記憶體監控已啟動 (PID: $MONITOR_PID)${NC}"
echo -e "${GREEN}日誌檔: $LOG_FILE${NC}"

# 捕捉中斷信號以清理監控進程
cleanup() {
    echo -e "\n${YELLOW}停止記憶體監控...${NC}"
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true

    # 檢查最終記憶體
    FINAL_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo -e "\n${YELLOW}記憶體統計:${NC}"
    echo "  初始: ${INITIAL_MEM} MB"
    echo "  最終: ${FINAL_MEM} MB"
    echo "  峰值: 請查看 $LOG_FILE"

    # 檢查是否有 OOM
    if grep -q "out of memory" logs/*.log 2>/dev/null; then
        echo -e "\n${RED}⚠️  檢測到 OOM 錯誤！${NC}"
        echo -e "${YELLOW}建議：${NC}"
        echo "  1. 進一步降低 max_masks_per_level"
        echo "  2. 考慮使用較小的 SAM2 模型"
        echo "  3. 套用記憶體清理補丁: patches/add_memory_cleanup.patch"
        exit 1
    else
        echo -e "\n${GREEN}✅ 測試完成，未檢測到 OOM${NC}"
        exit 0
    fi
}

trap cleanup EXIT INT TERM

# 執行測試
echo -e "\n${YELLOW}4. 執行 workflow...${NC}"
PYTHONPATH=src python -m my3dis.run_workflow \
    --config "$CONFIG" \
    --scene "$TEST_SCENE" 2>&1 | tee "logs/workflow_test_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n${GREEN}測試執行完成${NC}"
