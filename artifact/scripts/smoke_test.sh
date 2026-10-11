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
"${PYTHON_BIN}" - <<'PY'
import src.nodes.mapper as mapper

def fail_if_called(*args, **kwargs):
    raise AssertionError("graph construction initialized an LLM provider")

mapper.get_llm = fail_if_called
from src.graph import build_graph
build_graph([])
PY
echo "PASS: graph construction did not initialize an LLM provider"
"${PYTHON_BIN}" - "${tmp_dir}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

tmp_dir = Path(sys.argv[1])
checker = Path("scripts/check_output.py")
owl_thing = "http://www.w3.org/2002/07/owl#Thing"
rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
uco_class = "https://ontology.unifiedcyberontology.org/uco/tool/MaliciousTool"
uco_property = "https://ontology.unifiedcyberontology.org/uco/core/tag"

def diagnostics(entity, dt_prop, relation, *, missing=0):
    phases = {"entity": entity, "dt_prop": dt_prop, "relation": relation}
    total = sum(values[0] for values in phases.values())
    semantic = sum(values[1] for values in phases.values())
    operational = sum(values[2] for values in phases.values())
    completed = total - operational
    return {
        "provider": "test",
        "model": "test",
        "total_items": total,
        "completed_items": completed,
        "semantic_fallbacks": semantic,
        "operational_fallbacks": operational,
        "timeouts": 0,
        "exceptions": 0,
        "missing_tool_calls": missing,
        "missing_entity_uris": 0,
        "by_phase": {
            phase: {
                "total": values[0],
                "completed": values[0] - values[2],
                "semantic_fallbacks": values[1],
                "operational_fallbacks": values[2],
            }
            for phase, values in phases.items()
        },
    }

base = {
    "file": "diagnostic.json",
    "text": "synthetic smoke-test input",
    "ontology": "uco",
}
cases = {
    "partial-semantic-fallback": ({
        **base,
        "extracted_entities": [
            {"name": "Mapped", "class": uco_class},
            {"name": "Unmapped", "class": owl_thing},
        ],
        "extracted_triplets": [
            {"subject": "Mapped", "relation": "tag", "property": uco_property,
             "property_type": "DatatypeProperty", "object": "malware"},
            {"subject": "Unmapped", "relation": "label", "property": rdfs_label,
             "property_type": "DatatypeProperty", "object": "unknown"},
        ],
        "mapping_diagnostics": diagnostics((2, 1, 0), (2, 1, 0), (0, 0, 0)),
    }, True),
    "all-generic": ({
        **base,
        "extracted_entities": [{"name": "Unmapped", "class": owl_thing}],
        "extracted_triplets": [
            {"subject": "Unmapped", "relation": "label", "property": rdfs_label,
             "property_type": "DatatypeProperty", "object": "unknown"},
        ],
        "mapping_diagnostics": diagnostics((1, 1, 0), (1, 1, 0), (0, 0, 0)),
    }, False),
    "operational-fallback": ({
        **base,
        "extracted_entities": [{"name": "Mapped", "class": uco_class}],
        "extracted_triplets": [
            {"subject": "Mapped", "relation": "tag", "property": uco_property,
             "property_type": "DatatypeProperty", "object": "malware"},
        ],
        "mapping_diagnostics": diagnostics((1, 0, 0), (1, 0, 1), (0, 0, 0), missing=1),
    }, False),
}

for name, (payload, should_pass) in cases.items():
    path = tmp_dir / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(checker), str(path), "uco"],
        text=True,
        capture_output=True,
    )
    if (result.returncode == 0) != should_pass:
        raise AssertionError(
            f"{name}: expected pass={should_pass}, returncode={result.returncode}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
PY
echo "PASS: semantic fallbacks allowed and operational fallbacks rejected"
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
