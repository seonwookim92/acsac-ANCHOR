#!/usr/bin/env bash
set -euo pipefail

# Prepare the public DGX Spark runner under its conventional local image tag.
INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(dirname "${INFRA_DIR}")"
UPSTREAM_URL="${SPARK_VLLM_URL:-https://github.com/eugr/spark-vllm-docker.git}"
UPSTREAM_REVISION="${SPARK_VLLM_REVISION:-e5174b8c56c96e2ec2b5d63a8022075d1c60bad3}"
SETUP_ROOT="${ANCHOR_VLLM_SETUP_ROOT:-${ARTIFACT_ROOT}/../anchor-vllm-runtime}"
UPSTREAM_DIR="${SETUP_ROOT}/spark-vllm-docker"
VLLM_IMAGE="${VLLM_IMAGE:-vllm-node:latest}"
SETUP_LOG="${ANCHOR_VLLM_SETUP_LOG:-${ARTIFACT_ROOT}/results/vllm_server_setup.txt}"

print_plan() {
  printf '%s\n' \
    "git clone ${UPSTREAM_URL} ${UPSTREAM_DIR}" \
    "git -C ${UPSTREAM_DIR} fetch --depth 1 origin ${UPSTREAM_REVISION}" \
    "git -C ${UPSTREAM_DIR} checkout --detach ${UPSTREAM_REVISION}" \
    "(cd ${UPSTREAM_DIR} && ./build-and-copy.sh)" \
    "docker image inspect ${VLLM_IMAGE}" \
    "setup record: ${SETUP_LOG}"
}

if [[ "${ANCHOR_VLLM_SETUP_DRY_RUN:-0}" == "1" ]]; then
  print_plan
  exit 0
fi

for command_name in git docker uname; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${command_name}" >&2
    exit 1
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is unavailable to the current user" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; configure the NVIDIA driver first" >&2
  exit 1
fi

machine_arch="$(uname -m)"
case "${machine_arch}" in
  aarch64|arm64) ;;
  *)
    echo "ERROR: vllm-node is the DGX Spark/GB10 ARM64 path; found ${machine_arch}" >&2
    echo "Use a compatible vLLM image and set VLLM_IMAGE for another GPU platform." >&2
    exit 1
    ;;
esac

mkdir -p "${SETUP_ROOT}" "$(dirname "${SETUP_LOG}")"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
else
  configured_remote="$(git -C "${UPSTREAM_DIR}" remote get-url origin)"
  if [[ "${configured_remote}" != "${UPSTREAM_URL}" ]]; then
    echo "ERROR: existing checkout has unexpected origin: ${configured_remote}" >&2
    exit 1
  fi
fi

git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${UPSTREAM_REVISION}"
git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_REVISION}"

(
  cd "${UPSTREAM_DIR}"
  ./build-and-copy.sh
)

if ! docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: upstream preparation did not create ${VLLM_IMAGE}" >&2
  exit 1
fi

{
  echo "ANCHOR reviewer vLLM setup"
  echo "prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "upstream_url=${UPSTREAM_URL}"
  echo "upstream_revision=$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
  echo "machine_arch=${machine_arch}"
  echo "container_image=${VLLM_IMAGE}"
  docker image inspect "${VLLM_IMAGE}" \
    --format 'container_image_id={{.Id}} repo_digests={{json .RepoDigests}} created={{.Created}}'
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits
} | tee "${SETUP_LOG}"

echo
echo "vllm-node preparation complete. Next run:"
echo "  ./infrastructure/start_vllm_qwen35.sh"
