#!/usr/bin/env bash
# Demonstrate an intentional UCO SHACL violation and its correction.
# Pass --llm to exercise ANCHOR's LLM-based correction loop.
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ANCHOR_PYTHON:-${PKG_ROOT}/artifact/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found. Run ./install.sh from the package root first." >&2
  exit 1
fi

cd "${PKG_ROOT}/artifact"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN}" scripts/shacl_violation_demo.py "$@"
