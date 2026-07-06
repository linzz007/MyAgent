#!/usr/bin/env bash
set -euo pipefail

mkdir -p pids/server
shopt -s nullglob

for pid_file in pids/server/vllm_*.pid; do
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[vllm] stopping pid ${pid} (${pid_file})"
    kill "${pid}" || true
  fi
  rm -f "${pid_file}"
done

echo "[vllm] stopped"
