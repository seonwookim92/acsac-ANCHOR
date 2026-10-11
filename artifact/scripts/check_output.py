#!/usr/bin/env python3
"""Validate the stable structure of one ANCHOR functional-example output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS


ROOT = Path(__file__).resolve().parents[1]
OWL_THING = str(OWL.Thing)
GENERIC_PROPERTIES = {str(RDFS.label), str(RDFS.seeAlso)}


def load_vocabulary(schema: str) -> tuple[set[str], set[str]]:
    """Load declared class/property URIs from the selected packaged ontology."""
    graph = Graph()
    ontology_dir = ROOT / "ontology" / schema
    paths = sorted(
        path for path in ontology_dir.rglob("*")
        if path.suffix.lower() in {".ttl", ".owl", ".rdf", ".n3", ".nt", ".trig", ".jsonld"}
    )
    if not paths:
        fail(f"no packaged ontology files found for {schema}")
    for path in paths:
        try:
            graph.parse(path)
        except Exception as exc:
            fail(f"cannot parse packaged ontology {path.relative_to(ROOT)}: {exc}")

    class_types = (OWL.Class, RDFS.Class)
    property_types = (
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        URIRef("http://www.w3.org/2002/07/owl#DataProperty"),
        RDF.Property,
    )
    classes = {
        str(subject)
        for class_type in class_types
        for subject in graph.subjects(RDF.type, class_type)
        if isinstance(subject, URIRef)
    }
    properties = {
        str(subject)
        for property_type in property_types
        for subject in graph.subjects(RDF.type, property_type)
        if isinstance(subject, URIRef)
    }
    return classes, properties


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: check_output.py OUTPUT.json {uco|stix|malont}")

    output_path = Path(sys.argv[1])
    expected_schema = sys.argv[2]
    if expected_schema not in {"uco", "stix", "malont"}:
        fail(f"unsupported schema: {expected_schema}")
    if not output_path.is_file():
        fail(f"output file not found: {output_path}")

    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid UTF-8 JSON: {exc}")

    required = {
        "file", "text", "ontology", "extracted_entities", "extracted_triplets",
        "mapping_diagnostics",
    }
    missing = sorted(required - result.keys())
    if missing:
        fail(f"missing top-level fields: {', '.join(missing)}")
    if result["ontology"] != expected_schema:
        fail(f"expected ontology {expected_schema!r}, found {result['ontology']!r}")
    if not isinstance(result["text"], str) or not result["text"].strip():
        fail("input text is empty")
    if not isinstance(result["extracted_entities"], list):
        fail("extracted_entities is not an array")
    if not result["extracted_entities"]:
        fail("no entities were extracted")
    if not isinstance(result["extracted_triplets"], list):
        fail("extracted_triplets is not an array")
    if not result["extracted_triplets"]:
        fail("no triplets were extracted")

    classes, properties = load_vocabulary(expected_schema)
    generic_entity_count = 0

    for index, entity in enumerate(result["extracted_entities"]):
        if not isinstance(entity, dict):
            fail(f"entity {index} is not an object")
        for field in ("name", "class"):
            if not isinstance(entity.get(field), str) or not entity[field].strip():
                fail(f"entity {index} has no nonempty {field}")
        class_uri = entity["class"]
        if class_uri == OWL_THING:
            generic_entity_count += 1
        elif class_uri not in classes:
            fail(f"entity {index} class is not declared by {expected_schema}: {class_uri}")

    if generic_entity_count == len(result["extracted_entities"]):
        fail("all entity classes are owl:Thing fallbacks")

    triplet_fields = ("subject", "relation", "property", "property_type", "object")
    generic_property_count = 0
    for index, triplet in enumerate(result["extracted_triplets"]):
        if not isinstance(triplet, dict):
            fail(f"triplet {index} is not an object")
        missing_fields = [field for field in triplet_fields if field not in triplet]
        if missing_fields:
            fail(f"triplet {index} is missing: {', '.join(missing_fields)}")
        for field in triplet_fields:
            if not isinstance(triplet[field], str) or not triplet[field].strip():
                fail(f"triplet {index} has no nonempty string {field}")
        if triplet["property_type"] not in {"DatatypeProperty", "ObjectProperty", "Property"}:
            fail(f"triplet {index} has invalid property_type: {triplet['property_type']!r}")
        property_uri = triplet["property"]
        if property_uri in GENERIC_PROPERTIES:
            generic_property_count += 1
        elif property_uri not in properties:
            fail(f"triplet {index} property is not declared by {expected_schema}: {property_uri}")

    if generic_property_count == len(result["extracted_triplets"]):
        fail("all triplet properties are generic fallbacks")

    diagnostics = result["mapping_diagnostics"]
    if not isinstance(diagnostics, dict):
        fail("mapping_diagnostics is not an object")
    count_fields = (
        "total_items", "completed_items", "semantic_fallbacks",
        "operational_fallbacks", "timeouts", "exceptions",
        "missing_tool_calls", "missing_entity_uris",
    )
    for field in count_fields:
        value = diagnostics.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"mapping_diagnostics.{field} is not a nonnegative integer")
    for field in ("provider", "model"):
        if not isinstance(diagnostics.get(field), str) or not diagnostics[field].strip():
            fail(f"mapping_diagnostics.{field} is empty")
    if diagnostics["total_items"] == 0:
        fail("mapping diagnostics contain no mapper items")
    expected_items = len(result["extracted_entities"]) + len(result["extracted_triplets"])
    if diagnostics["total_items"] != expected_items:
        fail(
            "mapping diagnostic item count does not match emitted entities/triplets "
            f"({diagnostics['total_items']} != {expected_items})"
        )
    if diagnostics["completed_items"] + diagnostics["operational_fallbacks"] != diagnostics["total_items"]:
        fail("mapping diagnostic totals are inconsistent")
    operational_reasons = (
        diagnostics["timeouts"] + diagnostics["exceptions"]
        + diagnostics["missing_tool_calls"] + diagnostics["missing_entity_uris"]
    )
    if diagnostics["operational_fallbacks"] != operational_reasons:
        fail("operational fallback reason counts are inconsistent")
    if diagnostics["semantic_fallbacks"] > diagnostics["completed_items"]:
        fail("semantic fallback count exceeds completed mapper items")

    by_phase = diagnostics.get("by_phase")
    if not isinstance(by_phase, dict):
        fail("mapping_diagnostics.by_phase is not an object")
    for phase in ("entity", "dt_prop", "relation"):
        phase_counts = by_phase.get(phase)
        if not isinstance(phase_counts, dict):
            fail(f"mapping_diagnostics.by_phase.{phase} is missing")
        for field in ("total", "completed", "semantic_fallbacks", "operational_fallbacks"):
            value = phase_counts.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                fail(f"mapping_diagnostics.by_phase.{phase}.{field} is invalid")
        if phase_counts["completed"] + phase_counts["operational_fallbacks"] != phase_counts["total"]:
            fail(f"mapping_diagnostics.by_phase.{phase} totals are inconsistent")
        if phase_counts["semantic_fallbacks"] > phase_counts["completed"]:
            fail(f"mapping_diagnostics.by_phase.{phase} semantic fallback count is inconsistent")
    if by_phase["entity"]["total"] != len(result["extracted_entities"]):
        fail("entity diagnostic count does not match emitted entities")
    if by_phase["dt_prop"]["total"] + by_phase["relation"]["total"] != len(result["extracted_triplets"]):
        fail("property diagnostic count does not match emitted triplets")
    if diagnostics["operational_fallbacks"]:
        fail(
            "mapper used operational fallbacks "
            f"(total={diagnostics['operational_fallbacks']}, "
            f"timeouts={diagnostics['timeouts']}, exceptions={diagnostics['exceptions']}, "
            f"missing_tool_calls={diagnostics['missing_tool_calls']}, "
            f"missing_entity_uris={diagnostics['missing_entity_uris']})"
        )

    print(
        "PASS: mapper diagnostics "
        f"({diagnostics['completed_items']}/{diagnostics['total_items']} items completed; "
        f"{diagnostics['semantic_fallbacks']} semantic fallback(s); 0 operational fallbacks)"
    )
    print(f"PASS: valid {expected_schema} ANCHOR output")


if __name__ == "__main__":
    main()
