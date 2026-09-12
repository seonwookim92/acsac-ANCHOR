#!/usr/bin/env bash
#
# Claim 2: the same executable drives UCO, STIX, and MALOnt via --schema.
# Runs the two schemas not already covered by claim1.
#
set -euo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"${PKG_ROOT}/artifact/scripts/run_example.sh" stix
"${PKG_ROOT}/artifact/scripts/run_example.sh" malont
echo "PASS: schema switch validated for stix and malont"
