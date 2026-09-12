#!/usr/bin/env bash
#
# Claim 3: deterministic package completeness checks. No LLM endpoint needed.
#
set -euo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PKG_ROOT}/artifact/scripts/smoke_test.sh"
