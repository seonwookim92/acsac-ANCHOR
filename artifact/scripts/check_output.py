#!/usr/bin/env python3
"""Validate the stable structure of one ANCHOR functional-example output."""

from __future__ import annotations

import json
import sys
from pathlib import Path


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

    required = {"file", "text", "ontology", "extracted_entities", "extracted_triplets"}
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

    for index, entity in enumerate(result["extracted_entities"]):
        if not isinstance(entity, dict):
            fail(f"entity {index} is not an object")
        for field in ("name", "class"):
            if not isinstance(entity.get(field), str) or not entity[field].strip():
                fail(f"entity {index} has no nonempty {field}")

    triplet_fields = ("subject", "relation", "property", "property_type", "object")
    for index, triplet in enumerate(result["extracted_triplets"]):
        if not isinstance(triplet, dict):
            fail(f"triplet {index} is not an object")
        missing_fields = [field for field in triplet_fields if field not in triplet]
        if missing_fields:
            fail(f"triplet {index} is missing: {', '.join(missing_fields)}")

    print(f"PASS: valid {expected_schema} ANCHOR output")


if __name__ == "__main__":
    main()
