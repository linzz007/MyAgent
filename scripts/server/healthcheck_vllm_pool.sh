#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-configs/server/qwen14b_7gpu.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "[health] env file not found: ${ENV_FILE}" >&2
  exit 1
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-14b-awq}"
GPU_GROUPS="${GPU_GROUPS:-0}"
BASE_PORT="${BASE_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-key}"

IFS=';' read -ra GROUP_ARRAY <<< "${GPU_GROUPS}"
for index in "${!GROUP_ARRAY[@]}"; do
  port=$((BASE_PORT + index))
  echo "[health] testing port ${port}"
  curl -sS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${VLLM_API_KEY}" \
    --data "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Return exactly: ok\"}],
      \"temperature\": 0,
      \"max_tokens\": 8,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" | python -m json.tool
done
