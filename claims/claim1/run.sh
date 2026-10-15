#!/usr/bin/env bash
#
# Claim 1: ANCHOR processes one CTI report end to end and emits a structurally
# valid, ontology-aligned graph (UCO).
#
set -euo pipefail
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"${PKG_ROOT}/artifact/scripts/run_example.sh" uco
"${PKG_ROOT}/claims/claim1/shacl_demo.sh"
