#!/bin/bash
set -euo pipefail

NUM_GPUS="${1:-2}"
DEVICE_IDS="${2:-"0,1"}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

export WORLD_SIZE="${NUM_GPUS}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export HCCL_IF_IP="${HCCL_IF_IP:-$(hostname -I | awk '{print $1}')}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-23500}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1200}"
export RANK_OFFSET="${RANK_OFFSET:-0}"

IFS=',' read -ra DEVS <<< "${DEVICE_IDS}"

echo "=== HCCL Communication Test ==="
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "DEVICE_IDS=${DEVICE_IDS} (NPU devices to use)"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}, HCCL_IF_IP=${HCCL_IF_IP}"
echo "HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PIDS=()
for i in "${!DEVS[@]}"; do
    export LOCAL_RANK=$i
    export RANK_ID=$(expr $i + $RANK_OFFSET)
    export NPU_DEVICE_IDX=${DEVS[$i]}
    echo "Launching rank ${RANK_ID} (npu:${NPU_DEVICE_IDX})..."
    python3 "${SCRIPT_DIR}/test_hccl.py" &
    PIDS+=($!)
done

echo ""
echo "All ${NUM_GPUS} processes launched, waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
        echo "Process $pid FAILED"
    fi
done

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "=== All tests PASSED ==="
else
    echo ""
    echo "=== Some tests FAILED ==="
    exit 1
fi
