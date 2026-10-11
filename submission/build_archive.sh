#!/usr/bin/env bash
#
# Build the submission archive of the whole package, excluding environments,
# credentials, generated output, and caches. Writes a .tar.gz and its SHA-256
# outside the package directory.
#
#   ./submission/build_archive.sh [ARCHIVE_PATH]
#
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${PKG_ROOT}/artifact"
PYTHON_BIN="${ANCHOR_PYTHON:-${ARTIFACT_DIR}/.venv/bin/python}"
ARCHIVE_PATH="${1:-${PKG_ROOT}/../anchor-acsac-artifact.tar.gz}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found. Run ./install.sh first." >&2
  exit 1
fi

case "${ARCHIVE_PATH}" in
  "${PKG_ROOT}"/*)
    echo "ERROR: write the archive outside the package directory" >&2
    exit 1
    ;;
esac

cd "${ARTIFACT_DIR}"
"${PYTHON_BIN}" scripts/preflight.py --strict

parent_dir="$(dirname "${PKG_ROOT}")"
root_name="$(basename "${PKG_ROOT}")"

tar -C "${parent_dir}" -czf "${ARCHIVE_PATH}" \
  --exclude="${root_name}/.git" \
  --exclude="${root_name}/artifact/.venv" \
  --exclude="${root_name}/artifact/.env" \
  --exclude="${root_name}/artifact/output/*" \
  --exclude="${root_name}/.venv" \
  --exclude="${root_name}/.env" \
  --exclude="${root_name}/results" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  "${root_name}"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${ARCHIVE_PATH}" >"${ARCHIVE_PATH}.sha256"
else
  shasum -a 256 "${ARCHIVE_PATH}" >"${ARCHIVE_PATH}.sha256"
fi

echo "PASS: archive written to ${ARCHIVE_PATH}"
echo "PASS: checksum written to ${ARCHIVE_PATH}.sha256"
