#!/usr/bin/env bash
set -euo pipefail

# Reference vLLM server configuration used for the ANCHOR experiments.
# Environment variables may override deployment-specific values.
PORT="${PORT:-8081}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
MODEL_REVISION="${MODEL_REVISION:-9d1823d2dee688a6b25e77009dc727688c44936e}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-35b}"
GPU_UTIL="${GPU_UTIL:-0.60}"
MAX_LEN="${MAX_LEN:-262144}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm-node:latest}"
VLLM_IMAGE_SOURCE_URL="${VLLM_IMAGE_SOURCE_URL:-https://github.com/eugr/spark-vllm-docker}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
LOCAL_MEDIA_PATH="${LOCAL_MEDIA_PATH:-/tmp/vllm-media}"

mkdir -p "${HF_CACHE_DIR}" "${LOCAL_MEDIA_PATH}"

echo "Starting ANCHOR reference vLLM server"
echo "Model: ${MODEL}@${MODEL_REVISION}"
echo "Served name: ${SERVED_NAME}"
echo "External port: ${PORT}"
echo "GPU utilization: ${GPU_UTIL}"
echo "Maximum model length: ${MAX_LEN}"
echo "Container image: ${VLLM_IMAGE}"
echo "Container image source: ${VLLM_IMAGE_SOURCE_URL}"

COMMAND=(
  docker run --gpus all --rm
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
  -v "${LOCAL_MEDIA_PATH}:${LOCAL_MEDIA_PATH}:ro"
  -p "${PORT}:8000"
  "${VLLM_IMAGE}"
  vllm serve "${MODEL}"
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_NAME}"
  --allowed-local-media-path "${LOCAL_MEDIA_PATH}"
  --port 8000
  --host 0.0.0.0
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --enable-prefix-caching
  --limit-mm-per-prompt '{"image": 1, "video": 0}'
)

if [[ "${ANCHOR_VLLM_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
