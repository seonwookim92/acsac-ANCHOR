"""
Cypher export utility — converts ANCHOR result JSON to Neo4j-importable Cypher.

Usage (standalone):
    from src.utils.cypher_export import result_to_cypher
    cypher_text = result_to_cypher(result_dict)

The generated file can be imported with:
    cypher-shell -f result.cypher
    # or pasted into Neo4j Browser
"""

import re


def _neo4j_label(class_str: str) -> str:
    """
    Convert ontology class string or full URI to a valid Neo4j node label.

    Examples:
        "uco-observable:File"                                   → "File"
        "https://ontology.unifiedcyberontology.org/uco/tool/MaliciousTool" → "MaliciousTool"
        "http://docs.oasis-open.org/cti/ns/stix/malware#Malware" → "Malware"
        "File"                                                  → "File"
    """
    # Full URI: take fragment (#) first, then last path segment
    if class_str.startswith("http://") or class_str.startswith("https://"):
        if "#" in class_str:
            return class_str.split("#")[-1]
        return class_str.rstrip("/").split("/")[-1]
    # Prefixed name (namespace:Local) — take the local part
    if ":" in class_str:
        return class_str.split(":")[-1]
    return class_str or "Thing"


def _neo4j_rel_type(relation_class: str) -> str:
    """
    Convert relation class string to a Neo4j relationship type (UPPER_SNAKE_CASE).

    Examples:
        "uco-action:performs"      → "PERFORMS"
        "stix:uses"                → "USES"
        "drops"                    → "DROPS"
    """
    local = relation_class.split(":")[-1] if ":" in relation_class else relation_class
    # camelCase → UPPER_SNAKE: insert _ before uppercase runs
    snaked = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", local).upper()
    # Replace any non-alphanumeric (except _) with _
    snaked = re.sub(r"[^A-Z0-9_]", "_", snaked)
    return snaked or "RELATED_TO"


def _cypher_str(value: str) -> str:
    """Escape a string value for embedding in a Cypher literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def result_to_cypher(result: dict) -> str:
    """
    Convert an ANCHOR result dict to a Cypher script string.

    Args:
        result: dict with keys file, text, ontology, extracted_entities, extracted_triplets

    Returns:
        Multi-line Cypher string ready to write to a .cypher file.
    """
    ontology   = result.get("ontology", "unknown")
    source     = result.get("file", "")
    entities   = result.get("extracted_entities", [])
    triplets   = result.get("extracted_triplets", [])

    lines = [
        f"// ANCHOR — {ontology.upper()} ontology",
        f"// Source : {source}",
        f"// Entities: {len(entities)}   Triplets: {len(triplets)}",
        "",
    ]

    # ── Entities ────────────────────────────────────────────────────────────────
    if entities:
        lines.append("// ── Entities " + "─" * 60)
        for ent in entities:
            name       = _cypher_str(ent.get("name", ""))
            cls_raw    = ent.get("class", "")
            label      = _neo4j_label(cls_raw)
            cls_escaped = _cypher_str(cls_raw)
            lines.append(
                f"MERGE (:`{label}` {{name: '{name}', "
                f"ontology_class: '{cls_escaped}', ontology: '{ontology}'}});"
            )
        lines.append("")

    # ── Relationships ────────────────────────────────────────────────────────────
    if triplets:
        lines.append("// ── Relationships " + "─" * 55)
        for tri in triplets:
            subj    = _cypher_str(tri.get("subject", ""))
            obj     = _cypher_str(tri.get("object", ""))
            rel     = _cypher_str(tri.get("relation", ""))
            rel_cls = tri.get("relation_class", tri.get("relation", ""))
            rel_type = _neo4j_rel_type(rel_cls)
            lines.append(
                f"MATCH (s {{name: '{subj}'}}), (o {{name: '{obj}'}})"
            )
            lines.append(
                f"MERGE (s)-[:`{rel_type}` {{label: '{rel}'}}]->(o);"
            )
        lines.append("")

    return "\n".join(lines)
