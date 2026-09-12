#!/usr/bin/env bash
#
# ANCHOR — one-click installation for artifact evaluation.
#
# Creates artifact/.venv, installs the dependencies, and runs the deterministic
# package checks. It never contacts an LLM endpoint and needs no GPU, so it can
# be run before any inference endpoint is configured.
#
# Python 3.11, 3.12, and 3.13 are all supported and are auto-detected.
# Override the interpreter with PYTHON_BIN=/path/to/python ./install.sh
#
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${PKG_ROOT}/artifact"
VENV_DIR="${ANCHOR_VENV_DIR:-${ARTIFACT_DIR}/.venv}"

MIN_MINOR=11
MAX_MINOR=13

version_of() {
  "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true
}

supported() {
  local ver="$1" major minor
  major="${ver%%.*}"
  minor="${ver##*.}"
  [[ "${major}" == "3" ]] || return 1
  [[ "${minor}" =~ ^[0-9]+$ ]] || return 1
  (( minor >= MIN_MINOR && minor <= MAX_MINOR ))
}

PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -n "${PYTHON_BIN}" ]]; then
  # An explicit choice is honoured, but still sanity-checked.
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN='${PYTHON_BIN}' was not found on PATH" >&2
    exit 1
  fi
  chosen_version="$(version_of "${PYTHON_BIN}")"
  if ! supported "${chosen_version}"; then
    echo "WARNING: PYTHON_BIN='${PYTHON_BIN}' is Python ${chosen_version:-unknown}," >&2
    echo "         outside the tested 3.${MIN_MINOR}-3.${MAX_MINOR} range. Continuing as instructed." >&2
  fi
else
  for candidate in python3.11 python3.12 python3.13 python3 python; do
    command -v "${candidate}" >/dev/null 2>&1 || continue
    candidate_version="$(version_of "${candidate}")"
    if supported "${candidate_version}"; then
      PYTHON_BIN="${candidate}"
      chosen_version="${candidate_version}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  cat >&2 <<EOF
ERROR: no supported Python found. ANCHOR needs Python 3.${MIN_MINOR}-3.${MAX_MINOR}.

Interpreters checked on PATH: python3.11, python3.12, python3.13, python3, python

Install one, or point the installer at an existing interpreter:

  PYTHON_BIN=/path/to/python3.12 ./install.sh

Ways to get a supported interpreter:

  Debian/Ubuntu   sudo apt install python3.12 python3.12-venv
  Fedora/RHEL     sudo dnf install python3.12
  macOS           brew install python@3.12
  conda           conda create -n anchor python=3.12 && conda activate anchor
  pyenv           pyenv install 3.12 && pyenv shell 3.12
  uv              uv python install 3.12
EOF
  exit 1
fi

echo "[install] using ${PYTHON_BIN} (Python ${chosen_version})"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel 'setuptools<82'

# requirements-lock.txt pins every transitive dependency and is the exact
# environment the artifact was tested with. If a resolver conflict appears on a
# Python version other than the one it was generated on, fall back to the pinned
# direct dependencies and let pip resolve the rest.
if "${VENV_DIR}/bin/python" -m pip install -r "${ARTIFACT_DIR}/requirements-lock.txt"; then
  echo "[install] installed the fully locked environment"
else
  echo "[install] the lock file did not resolve on Python ${chosen_version};" >&2
  echo "[install] falling back to requirements-core.txt (pinned direct dependencies)" >&2
  "${VENV_DIR}/bin/python" -m pip install -r "${ARTIFACT_DIR}/requirements-core.txt"
fi

ANCHOR_PYTHON="${VENV_DIR}/bin/python" "${ARTIFACT_DIR}/scripts/smoke_test.sh"

echo
echo "Next: configure an LLM endpoint and run a claim."
echo "  cp artifact/.env.example artifact/.env   # a dotfile: use 'ls -a artifact/'"
echo "  # edit artifact/.env: set LLM_PROVIDER and fill in that one provider block"
echo "  ./claims/claim1/run.sh"
echo
echo "No endpoint yet? ./claims/claim3/run.sh needs none."
