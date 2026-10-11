from typing import Annotated, Optional
from operator import add
from typing_extensions import TypedDict


class RawEntity(TypedDict):
    name: str           # normalized display name
    type_hint: str      # ontology search query hint (e.g. "malware ransomware tool")
    properties: dict    # {property_hint: value} e.g. {"programming language": "Rust"}


class RawTriplet(TypedDict):
    subject: str        # entity name
    predicate: str      # human-readable verb (e.g. "stops", "observed")
    object: str         # entity name OR literal value (for DatatypeProperty)
    evidence: str       # original sentence
    property_type: str  # "ObjectProperty" | "DatatypeProperty"


class AnchorState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    input_text: str
    output_file: str    # filename for export_eval_json (e.g. "result.json")

    # ── Node 1: Chunker ───────────────────────────────────────────────────────
    chunks: list[str]

    # ── Node 2: Extractor (accumulated across chunks via add) ─────────────────
    raw_entities: Annotated[list[RawEntity], add]
    raw_triplets: Annotated[list[RawTriplet], add]

    # ── Node 3: Preprocessor (deduplicated, IoC hint-enriched) ──────────────
    entities: list[RawEntity]   # clean, unique entities (IoC type_hint overridden if detected)
    triplets: list[RawTriplet]  # clean, deduplicated triplets

    # ── Node 4: Mapper — per-item queues for the LangGraph loop ──────────────
    # Each queue shrinks (replaced) as items are consumed; resolved lists grow (append).
    entities_pending:   list[dict]                  # RawEntity items yet to be class-resolved
    entities_resolved:  Annotated[list[dict], add]  # {name, entity_uri, class_uri}
    dt_props_pending:   list[dict]                  # {entity_name, hint, value} yet to be mapped
    dt_props_resolved:  Annotated[list[dict], add]  # {entity_name, hint, property_uri, value}
    relations_pending:  list[dict]                  # {subject, predicate, object, evidence} yet to be mapped
    relations_resolved: Annotated[list[dict], add]  # {subject, predicate, property_uri, property_type, object}

    # ── Node 4: Mapper (agent output, assembled by mapper_finalize) ──────────
    mapped_entities: list[dict]   # [{name, class}]  e.g. class="tool:MaliciousTool"
    mapped_triplets: list[dict]   # ObjectProperty:   {subject, relation, property, property_type="ObjectProperty",  object}
                                  # DatatypeProperty: {subject, relation, property, property_type="DatatypeProperty", object}

    # ── Node 4b: SHACL Correction ────────────────────────────────────────────
    # [{entity, uri, errors}] for entities that still have violations after max retries
    validation_warnings: list[dict]

    # ── Node 5: Exporter ─────────────────────────────────────────────────────
    result: Optional[dict]        # final assembled JSON dict

    # ── Cross-cutting ─────────────────────────────────────────────────────────
    errors: Annotated[list[str], add]

    # ── Measurements (--measurement flag) ─────────────────────────────────────
    # One record per mapper item, including tool strategy, completion_status,
    # semantic_fallback, operational_fallback, and optional search scores.
    measurements: Annotated[list[dict], add]
