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
  "extracted_triplets": [{"subject": "...", "relation": "...", "relation_class": "...", "object": "..."}]
}
"""

import json
import os
from pathlib import Path
from src.state import AnchorState
from src.utils.cypher_export import result_to_cypher


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
