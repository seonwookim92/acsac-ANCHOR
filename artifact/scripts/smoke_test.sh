#!/usr/bin/env bash
#
# Deterministic checks that need no LLM endpoint, no GPU, and no network.
# Backs claims/claim3.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
PYTHON_BIN="${ANCHOR_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found. Run ./install.sh from the package root first." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

cd "${ROOT_DIR}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" scripts/preflight.py
"${PYTHON_BIN}" main.py --help >"${tmp_dir}/help.txt"
"${PYTHON_BIN}" main.py --graph >"${tmp_dir}/graph.txt"
HF_CACHE_DIR="${tmp_dir}/hf" \
LOCAL_MEDIA_PATH="${tmp_dir}/media" \
ANCHOR_VLLM_DRY_RUN=1 \
  "${PKG_ROOT}/infrastructure/start_vllm_qwen35.sh" >"${tmp_dir}/vllm-command.txt"
ANCHOR_VLLM_SETUP_DRY_RUN=1 \
  "${PKG_ROOT}/infrastructure/prepare_vllm_node.sh" >"${tmp_dir}/vllm-setup.txt"

grep -q -- '--schema' "${tmp_dir}/help.txt"
grep -q 'mapper_init' "${tmp_dir}/graph.txt"
grep -q 'Qwen/Qwen3.5-35B-A3B-FP8' "${tmp_dir}/vllm-command.txt"
grep -q 'spark-vllm-docker' "${tmp_dir}/vllm-setup.txt"

echo "PASS: deterministic functional checks completed"
