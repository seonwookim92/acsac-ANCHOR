#!/usr/bin/env python3
"""Demonstrate detection and correction of an intentional UCO SHACL violation."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
UCO_DIR = ARTIFACT_ROOT / "ontology" / "uco"
MCP_SERVER_SCRIPT = ARTIFACT_ROOT / "mcp" / "ontology-mcp-server" / "main.py"
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))
DEMO_ENTITY_ID = "shacl-violation-demo"
DEMO_ENTITY_URI = f"http://example.org/entities/{DEMO_ENTITY_ID}"
DEMO_CLASS_URI = (
    "https://ontology.unifiedcyberontology.org/uco/analysis/ArtifactClassification"
)
REQUIRED_PROPERTY_URI = (
    "https://ontology.unifiedcyberontology.org/uco/analysis/class"
)


def _text(value) -> str:
    content = getattr(value, "content", value)
    return content if isinstance(content, str) else str(content)


def _json(value) -> dict:
    return json.loads(_text(value))


def _tool(tools: list, name: str):
    match = next((tool for tool in tools if getattr(tool, "name", "") == name), None)
    if match is None:
        raise RuntimeError(f"Required MCP tool is unavailable: {name}")
    return match


def _print_violations(label: str, result: dict) -> None:
    print(f"{label}: FAIL ({result.get('error_count', 0)} SHACL violation(s))")
    for error in result.get("errors", []):
        prop = error.get("property_uri", "unknown property")
        count = error.get("current_count", "?")
        required = error.get("min_required", "?")
        print(f"  - {error.get('type', 'violation')}: {prop} "
              f"(current={count}, required>={required})")


async def _run(use_llm: bool) -> int:
    # The demonstration is specifically for UCO, regardless of the schema used
    # by a preceding ANCHOR run. Endpoint/model settings still come from .env.
    from main import _mcp_config
    from src.nodes.mapper import make_shacl_correction_node

    os.environ["MCP_SERVER_SCRIPT"] = str(MCP_SERVER_SCRIPT)
    os.environ["ONTOLOGY_DIR"] = str(UCO_DIR)
    config = _mcp_config()
    client = MultiServerMCPClient(config)

    async with client.session("ontology-mcp-server") as session:
        tools = await load_mcp_tools(session)
        reset_graph = _tool(tools, "reset_graph")
        create_entity = _tool(tools, "create_entity")
        validate_entity = _tool(tools, "validate_entity")
        set_property = _tool(tools, "set_property")

        await reset_graph.ainvoke({})
        await create_entity.ainvoke({
            "entity_id": DEMO_ENTITY_ID,
            "class_uris": [DEMO_CLASS_URI],
        })

        before = _json(await validate_entity.ainvoke({"entity_uri": DEMO_ENTITY_URI}))
        if before.get("valid"):
            print("ERROR: the deliberately incomplete entity did not trigger a violation")
            return 1
        if not any(
            error.get("type") == "missing_required_property"
            and error.get("property_uri") == REQUIRED_PROPERTY_URI
            for error in before.get("errors", [])
        ):
            print("ERROR: the expected analysis:class violation was not reported")
            return 1

        print("SHACL VIOLATION DEMONSTRATION (UCO)")
        print(f"Injected: {DEMO_CLASS_URI} without its required analysis:class value")
        _print_violations("BEFORE", before)

        if use_llm:
            print("CORRECTION: invoking ANCHOR's existing LLM-based correction loop")
            correction = make_shacl_correction_node(tools)
            node_result = await correction({
                "entities_resolved": [{
                    "name": DEMO_ENTITY_ID,
                    "entity_uri": DEMO_ENTITY_URI,
                    "class_uri": DEMO_CLASS_URI,
                }]
            })
            warnings = node_result.get("validation_warnings", [])
        else:
            print("CORRECTION: adding the required analysis:class value deterministically")
            await set_property.ainvoke({
                "entity_uri": DEMO_ENTITY_URI,
                "property_uri": REQUIRED_PROPERTY_URI,
                "value": "malware",
            })
            warnings = []

        after = _json(await validate_entity.ainvoke({"entity_uri": DEMO_ENTITY_URI}))
        if after.get("valid") and after.get("shacl_status") == "passed":
            print("AFTER: PASS - all applicable SHACL constraints satisfied")
            mode = "LLM correction" if use_llm else "deterministic correction"
            print(f"PASS: intentional SHACL violation detected and resolved ({mode})")
            return 0

        _print_violations("AFTER", after)
        if use_llm and warnings:
            print("WARNING: violation remains unresolved and was recorded in validation_warnings")
            print("PASS: intentional SHACL violation detected and unresolved warning emitted")
            return 0

        print("ERROR: violation was neither corrected nor reported as a validation warning")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create an intentionally invalid UCO entity and demonstrate SHACL "
            "detection followed by correction or an explicit warning."
        )
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="use ANCHOR's LLM-based correction loop (requires a configured endpoint)",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.llm))


if __name__ == "__main__":
    raise SystemExit(main())
