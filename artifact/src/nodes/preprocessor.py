"""
Node 3: Preprocessor

Deterministic cleanup after raw LLM extraction:
1. IoC detection: entity names that match CVE/IP/hash/URL/domain patterns
   get their UCO class pre-assigned, skipping search_classes later.
2. Noise filter: remove entities whose names are clearly not ontology-worthy
   (pure dates, bare numbers, very short/empty strings).
3. Entity deduplication: merge entities with identical or near-identical names
   across chunks, combining their properties.
4. Triplet deduplication: drop (subject, predicate, object) duplicates.
5. Triplet reference check: drop triplets where subject or object is not in
   the entity list (prevents dangling references to filtered-out entities).
"""

import re
from src.state import AnchorState, RawEntity, RawTriplet
from src.utils.ioc_patterns import detect_ioc

# Patterns that indicate a string is NOT a meaningful entity name
_NOISE_PATTERNS = [
    re.compile(r'^\d{4}$'),                        # bare year: "2021"
    re.compile(r'^\d+$'),                           # bare number
    re.compile(r'^(january|february|march|april|may|june|july|august|'
               r'september|october|november|december)\s+\d{4}$', re.I),
    re.compile(r'^(q[1-4]\s+)?\d{4}$', re.I),     # "Q1 2024"
    re.compile(r'^over the (past|last)', re.I),     # "Over the past two years"
    re.compile(r'^(recently|currently|previously)$', re.I),
    re.compile(r'^.{1,2}$'),                        # 1-2 char strings
]

# Articles and vague starters that indicate a descriptor, not an entity name
_DESCRIPTOR_STARTS = re.compile(
    r'^(a |an |the |some |several |various |multiple |numerous |certain |'
    r'such |many |few |one |two |three |)'
    r'(new |old |recent |existing |specific |particular |unique |'
    r'malicious |suspicious |malevolent |)',
    re.I,
)


def _is_noise(name: str) -> bool:
    name_stripped = name.strip()
    for pat in _NOISE_PATTERNS:
        if pat.match(name_stripped):
            return True
    # Descriptor-only names (e.g. "a completely new malware family")
    # Heuristic: if after stripping articles/adjectives there's < 2 meaningful words left
    # and the name contains >= 4 words, it's likely a description
    words = name_stripped.split()
    if len(words) >= 5 and _DESCRIPTOR_STARTS.match(name_stripped):
        return True
    return False


def _normalize_name(name: str) -> str:
    """Strip leading articles for consistent deduplication keys."""
    return re.sub(r'^(a |an |the )', '', name.strip(), flags=re.I).strip()


# Generic CTI suffixes that don't distinguish entities — stripped for dedup keys only.
_GENERIC_SUFFIX = re.compile(
    r'\s+(group|team|gang|apt|organization|organisation|'
    r'hacking group|hacking team|threat actor|threat group)$',
    re.I,
)


def _dedup_key(name: str) -> str:
    """Produce a normalized key for entity deduplication.

    More aggressive than _normalize_name: also normalizes underscores to spaces
    and strips generic CTI suffixes (e.g. 'Fancy Bear Hacking Group' → 'fancy bear').
    Used only for matching — display name is preserved separately.
    """
    key = re.sub(r'^(a |an |the )', '', name.strip(), flags=re.I)
    key = key.replace('_', ' ')
    key = re.sub(r'\s+', ' ', key).strip().lower()
    key = _GENERIC_SUFFIX.sub('', key).strip()
    return key


async def preprocessor_node(state: AnchorState) -> dict:
    import os
    raw_entities: list[RawEntity] = state.get("raw_entities", [])
    raw_triplets: list[RawTriplet] = state.get("raw_triplets", [])
    print(f"[preprocessor] raw_entities={len(raw_entities)}, raw_triplets={len(raw_triplets)}")

    no_ioc          = os.getenv("ANCHOR_NO_IOC",          "").lower() in ("1", "true")
    no_noise_filter = os.getenv("ANCHOR_NO_NOISE_FILTER", "").lower() in ("1", "true")
    if no_ioc:
        print("[preprocessor] IoC pre-assignment DISABLED (--no-ioc)")
    if no_noise_filter:
        print("[preprocessor] Noise filter DISABLED (--no-noise-filter)")

    # ── 1. Deduplicate + IoC-map entities ─────────────────────────────────────
    # Key: normalized lowercase name → merged entity dict
    merged: dict[str, dict] = {}

    for entity in raw_entities:
        name = entity["name"].strip()
        if not name:
            continue
        if not no_noise_filter and _is_noise(name):
            continue

        norm_key = _dedup_key(name)

        if norm_key not in merged:
            merged[norm_key] = {
                "name": _normalize_name(name),  # use cleaned name (display form)
                "type_hint": entity["type_hint"],
                "properties": dict(entity.get("properties", {})),
            }
        else:
            # Merge properties from duplicate mentions
            existing = merged[norm_key]
            for k, v in entity.get("properties", {}).items():
                if k not in existing["properties"]:
                    existing["properties"][k] = v
            # Keep the longer / more descriptive type_hint
            if len(entity["type_hint"]) > len(existing["type_hint"]):
                existing["type_hint"] = entity["type_hint"]

    # ── 2. IoC type_hint override (skipped when --no-ioc) ────────────────────
    # Replaces the LLM-extracted type_hint with a precise, schema-agnostic
    # description (e.g. "MD5 hash", "IPv4 address") so search_classes can find
    # the correct class in any ontology (UCO, STIX, MALOnt).
    entities: list[RawEntity] = []
    for entity_dict in merged.values():
        if not no_ioc:
            ioc = detect_ioc(entity_dict["name"])
            if ioc:
                entity_dict["type_hint"] = ioc.type_hint
        entities.append(entity_dict)  # type: ignore[arg-type]

    # ── 3. Build canonical name set for triplet reference check ───────────────
    # Map dedup key → canonical display name
    name_map: dict[str, str] = {
        _dedup_key(e["name"]): e["name"] for e in entities
    }

    def _resolve_name(raw: str) -> str | None:
        """Return canonical entity name or None if not in entity list."""
        return name_map.get(_dedup_key(raw))

    # ── 4. Deduplicate + filter triplets ──────────────────────────────────────
    seen_triplets: set[tuple[str, str, str, str]] = set()
    triplets: list[RawTriplet] = []

    for t in raw_triplets:
        prop_type = t.get("property_type", "ObjectProperty")
        pred = t["predicate"].strip()
        subj = _resolve_name(t["subject"])

        if not subj:
            continue  # subject must always be a known entity

        if prop_type == "ObjectProperty":
            obj = _resolve_name(t["object"])
            if not obj:
                continue  # dangling reference
            if subj == obj:
                continue  # self-loop
            key = (prop_type, subj, pred.lower(), obj)
        else:
            # DatatypeProperty: object is a literal, no entity resolution needed
            obj = t["object"].strip()
            if not obj:
                continue
            key = (prop_type, subj, pred.lower(), obj.lower())

        if key in seen_triplets:
            continue
        seen_triplets.add(key)

        triplets.append(RawTriplet(
            subject=subj,
            predicate=pred,
            object=obj,
            evidence=t.get("evidence", ""),
            property_type=prop_type,
        ))

    op_triplets = [t for t in triplets if t.get("property_type") == "ObjectProperty"]
    dp_triplets = [t for t in triplets if t.get("property_type") == "DatatypeProperty"]

    print(f"[preprocessor] after filter: {len(entities)} entities, "
          f"{len(op_triplets)} ObjectProperty + {len(dp_triplets)} DatatypeProperty triplets")
    for e in entities:
        print(f"  entity: {e['name']!r} ({e['type_hint']})")
    for t in op_triplets:
        print(f"  [ObjProp]  {t['subject']!r} --[{t['predicate']}]--> {t['object']!r}")
    for t in dp_triplets:
        print(f"  [DataProp] {t['subject']!r}  [{t['predicate']}] = {t['object']!r}")

    return {
        "entities": entities,
        "triplets": triplets,
    }
