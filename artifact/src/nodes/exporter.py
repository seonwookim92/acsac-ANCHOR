"""
Node 5: Exporter

Assembles the final JSON result from state and writes it to disk.

If ENABLE_EVAL_TOOLS=true and the MCP tool export_eval_json is available,
it delegates to the MCP tool (which also writes a file on the server side).
Otherwise, writes the JSON directly using Python.

Output format matches reference/3am_ransomware_analysis_final.json:
{
  "file": "<output_file>",
  "text": "<original input text>",
  "ontology": "uco",
  "extracted_entities": [{"name": "...", "class": "..."}],
  "extracted_triplets": [{"subject": "...", "relation": "...", "relation_class": "...", "object": "..."}],
  "mapping_diagnostics": {"provider": "...", "model": "...", ...}
}
"""

import json
import os
from collections import Counter
from pathlib import Path
from src.state import AnchorState
from src.utils.cypher_export import result_to_cypher


def _mapping_diagnostics(measurements: list[dict]) -> dict:
    """Summarize mapper completion without treating generic URIs as failures."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    model_key, default_model = {
        "anthropic": ("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-6"),
        "openai": ("OPENAI_LLM_MODEL", "gpt-4o"),
        "vllm": ("VLLM_LLM_MODEL", "default"),
    }.get(provider, ("VLLM_LLM_MODEL", "default"))

    statuses = Counter(m.get("completion_status", "missing_tool_call") for m in measurements)
    phases = {}
    for phase in ("entity", "dt_prop", "relation"):
        items = [m for m in measurements if m.get("phase") == phase]
        phases[phase] = {
            "total": len(items),
            "completed": sum(m.get("completion_status") == "completed" for m in items),
            "semantic_fallbacks": sum(bool(m.get("semantic_fallback")) for m in items),
            "operational_fallbacks": sum(bool(m.get("operational_fallback")) for m in items),
        }

    return {
        "provider": provider,
        "model": os.getenv(model_key, "").strip() or default_model,
        "total_items": len(measurements),
        "completed_items": statuses["completed"],
        "semantic_fallbacks": sum(bool(m.get("semantic_fallback")) for m in measurements),
        "operational_fallbacks": sum(bool(m.get("operational_fallback")) for m in measurements),
        "timeouts": statuses["timeout"],
        "exceptions": statuses["exception"],
        "missing_tool_calls": statuses["missing_tool_call"],
        "missing_entity_uris": statuses["missing_entity_uri"],
        "by_phase": phases,
    }


def make_exporter_node(tools: list | None = None):
    """
    Factory that optionally closes over MCP tools for export_eval_json.
    If tools is None or export_eval_json is not present, writes JSON directly.
    """
    # Find export_eval_json tool if available
    export_tool = None
    if tools:
        for t in tools:
            if getattr(t, "name", "") == "export_eval_json":
                export_tool = t
                break

    async def exporter_node(state: AnchorState) -> dict:
        output_file = state.get("output_file", "result.json")
        input_text = state.get("input_text", "")
        mapped_entities = state.get("mapped_entities", [])
        mapped_triplets = state.get("mapped_triplets", [])
        measurements = state.get("measurements", [])

        # Determine ontology name from ONTOLOGY_DIR env
        ontology_dir = os.getenv("ONTOLOGY_DIR", "").lower()
        if "uco" in ontology_dir:
            ontology_name = "uco"
        elif "stix" in ontology_dir:
            ontology_name = "stix"
        elif "malont" in ontology_dir:
            ontology_name = "malont"
        else:
            ontology_name = "unknown"

        result = {
            "file": output_file,
            "text": input_text,
            "ontology": ontology_name,
            "extracted_entities": mapped_entities,
            "extracted_triplets": mapped_triplets,
            "mapping_diagnostics": _mapping_diagnostics(measurements),
        }

        validation_warnings = state.get("validation_warnings")
        if validation_warnings:
            result["validation_warnings"] = validation_warnings

        # Write output JSON
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / output_file

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Optional Cypher export (EXPORT_CYPHER=true)
        if os.getenv("EXPORT_CYPHER", "").lower() in ("1", "true"):
            cypher_path = out_path.with_suffix(".cypher")
            with open(cypher_path, "w", encoding="utf-8") as f:
                f.write(result_to_cypher(result))

        # Also invoke export_eval_json via MCP if available
        if export_tool:
            try:
                export_tool.invoke({
                    "file": str(out_path),
                    "text": input_text,
                    "entities": mapped_entities,
                    "triplets": mapped_triplets,
                })
            except Exception:
                pass  # MCP export is best-effort; local file is already written

        return {"result": result}

    return exporter_node
