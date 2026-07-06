#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-configs/server/qwen14b_7gpu.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "[download] env file not found: ${ENV_FILE}" >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
HF_HOME="${HF_HOME:-/data/hf}"
export HF_HOME

echo "[download] HF_HOME=${HF_HOME}"
echo "[download] MODEL_ID=${MODEL_ID}"

if [[ -d "${MODEL_ID}" ]]; then
  echo "[download] local model directory exists, skip HuggingFace download: ${MODEL_ID}"
  echo "[download] done"
  exit 0
fi

mkdir -p "${HF_HOME}"

python -m pip install --upgrade "huggingface-hub[cli]"
hf download "${MODEL_ID}"

echo "[download] done"
