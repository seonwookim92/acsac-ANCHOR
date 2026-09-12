#!/usr/bin/env bash
#
# Run ANCHOR end to end on one fixed CTINexus report and validate the emitted
# graph structure. Requires a configured LLM endpoint (artifact/.env).
#
#   ./scripts/run_example.sh uco | stix | malont
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ANCHOR_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SCHEMA="${1:-uco}"

case "${SCHEMA}" in
  uco|stix|malont) ;;
  *) echo "ERROR: schema must be uco, stix, or malont" >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found. Run ./install.sh from the package root first." >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  echo "ERROR: artifact/.env is missing." >&2
  echo "       cp artifact/.env.example artifact/.env, then set LLM_PROVIDER and its block." >&2
  exit 1
fi

cd "${ROOT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p output

output_name="functional-${SCHEMA}.json"
"${PYTHON_BIN}" scripts/check_endpoint.py
"${PYTHON_BIN}" main.py \
  --file dataset/ctinexus_buckeye-windows-zero-day-exploit.txt \
  --schema "${SCHEMA}" \
  --output "${output_name}" \
  --timer

"${PYTHON_BIN}" scripts/check_output.py "output/${output_name}" "${SCHEMA}"
