#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-configs/server/qwen14b_7gpu.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "[vllm] env file not found: ${ENV_FILE}" >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-14b-awq}"
GPU_GROUPS="${GPU_GROUPS:-0}"
BASE_PORT="${BASE_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-key}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

mkdir -p logs/server pids/server

port_is_listening() {
  local port="$1"
  python - "${port}" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sys.exit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
finally:
    sock.close()
PY
}

IFS=';' read -ra GROUP_ARRAY <<< "${GPU_GROUPS}"
for index in "${!GROUP_ARRAY[@]}"; do
  group="${GROUP_ARRAY[$index]}"
  port=$((BASE_PORT + index))
  tp_size=$(awk -F',' '{print NF}' <<< "${group}")
  log_file="logs/server/vllm_${port}.log"
  pid_file="pids/server/vllm_${port}.pid"

  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "[vllm] port ${port} already has running pid $(cat "${pid_file}")"
    continue
  fi
  if [[ -f "${pid_file}" ]]; then
    echo "[vllm] removing stale pid $(cat "${pid_file}") for port ${port}"
    rm -f "${pid_file}"
  fi
  if port_is_listening "${port}"; then
    echo "[vllm] port ${port} already has a listener but no live pid file; skipping start"
    continue
  fi

  echo "[vllm] starting ${MODEL_ID} on CUDA_VISIBLE_DEVICES=${group}, port=${port}, tp=${tp_size}"
  setsid nohup env \
    CUDA_VISIBLE_DEVICES="${group}" \
    HF_HOME="${HF_HOME:-/data/hf}" \
    HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
    vllm serve "${MODEL_ID}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --api-key "${VLLM_API_KEY}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${tp_size}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --dtype "${VLLM_DTYPE}" \
      ${VLLM_EXTRA_ARGS} \
      > "${log_file}" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
  echo "[vllm] pid $(cat "${pid_file}") -> ${log_file}"
done

echo "[vllm] started. Check logs with: tail -f logs/server/vllm_${BASE_PORT}.log"
