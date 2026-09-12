"""
Node 2: Raw Extractor (2-pass)

Pass 1 — Entity extraction (one LLM call per chunk):
  Extract all named entities with type hints and literal properties.
  No behavior-technique abstraction; no triplets.

Pass 2 — Relationship extraction (one LLM call per chunk):
  Given the full entity list from Pass 1 and each chunk, extract:
  - ObjectProperty  : entity → entity relationships
  - DatatypeProperty: entity → literal value attributes

Separating the two tasks removes subject-bias and dangling-reference issues
that arise when the LLM must discover entities and relationships simultaneously.
"""

import asyncio
import difflib
import json
import os
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AnchorState, RawEntity, RawTriplet
from src.utils.llm import get_llm

# ── Pass 1: Entity extraction ──────────────────────────────────────────────────

_ENTITY_SYSTEM_PROMPT = """\
You are a Cyber Threat Intelligence (CTI) knowledge extraction specialist.
Extract all named entities from the CTI text.

## Entity Types to Extract
- Threat actors, APT groups, intrusion sets (e.g. "Lazarus", "Quantum")
- Malware, ransomware, spyware, tools (e.g. "Predator", "LockBit", "3AM")
- Attack patterns, exploit chains, campaigns, operations, attack methods
  (e.g. "0-day exploit chain for iPhones", "Operation Interception", "BazarCall",
   "in-the-wild exploit chain", "spear-phishing campaign")
- Vulnerabilities: CVE identifiers and named vulnerabilities
- Organizations, companies, research groups (e.g. "Google TAG", "Apple", "Broadcom")
- Software, operating systems, file artifacts (e.g. "iOS 16.7", "safarifontagent")
- Locations (countries, regions)
- Victim groups / targets (e.g. "Engineers", "iOS users")

## Rules
- Normalize: remove leading articles ("a", "an", "the")
- Anonymous roles become named entities: "ransomware affiliate" → "Ransomware Affiliate"
- Named campaigns/operations are standalone entities: "Quantum BazarCall Campaign"
- DO NOT extract: bare dates, durations, pure numeric strings, generic descriptions
- DO NOT use date/time expressions as entity names, even as campaign proxies.
  "January 2023", "Q3 2024", "since early 2022" are NOT entity names.
  If a campaign has no proper name, use a descriptive label (e.g., "Destructive Cyberattack Campaign")
  and record the date as a property: {"first seen": "2023-01"}.
- If the text refers to an unnamed threat actor or attacker as the persistent subject of
  cyber actions (e.g., "the attackers", "the group", "they deployed malware"), extract it
  as a generic entity using the most specific label inferable from context.
  Examples: "Threat Actors", "Attackers", "Unidentified APT Group"
  This allows coreference resolution to ground subsequent pronoun/descriptor references.
- DO NOT abstract behaviors into entities.
  "encrypts files", "stops services", "installs spyware" are NOT entities.
  If the text says malware *drops* specific named files, extract those files as entities.
- Named or described attack artifacts ARE entities even without a proper name:
  "0-day exploit chain for iPhones" → name it exactly as described in the text
  "in-the-wild exploit" → extract with the full descriptive phrase as its name

**type_hint**: short phrase for ontology class search.
  Good: "malware ransomware", "apt intrusion set", "vulnerability cve", "attack pattern exploit chain"
  Bad: "a kind of malware that...", "organization that..."

**properties**: only clearly stated literal attributes of this entity.
  Examples: {"implementation language": "Rust"}, {"alias": "Quantum Locker"},
            {"first seen": "2021-08"}, {"architecture": "arm,x86-64"}
  Do NOT put relationships to other entities here.

## Output Format
Return ONLY valid JSON:
{
  "entities": [
    {
      "name": "3AM",
      "type_hint": "malware ransomware",
      "properties": {"implementation language": "Rust"}
    },
    {
      "name": "Ransomware Affiliate",
      "type_hint": "threat actor",
      "properties": {}
    },
    {
      "name": "0-day exploit chain for iPhones",
      "type_hint": "attack pattern exploit chain",
      "properties": {}
    }
  ]
}
"""


# ── Pass 2: Relationship extraction ───────────────────────────────────────────

_RELATION_SYSTEM_PROMPT = """\
You are a Cyber Threat Intelligence (CTI) knowledge extraction specialist.
Given an entity list and CTI text, extract ALL relationships between those entities
and ALL literal attributes of those entities that appear in the text.

## Entity List
The user message will provide the confirmed entity list.
**STRICT RULE**: Use ONLY the exact names from this list as subjects and objects
for ObjectProperty. Do NOT use dates, time references, unnamed events, or any
other string not present in the entity list. If a relationship involves an entity
not in the list, OMIT the triplet entirely — do not substitute or invent names.

## Two Relationship Types

### 1. ObjectProperty — entity-to-entity relationships
Both subject AND object must be names from the entity list.
- predicate: concise verb phrase ("uses", "targets", "exploits", "drops",
  "attributed to", "discovered", "observed", "part of", "variant of",
  "impersonates", "originates from", "mitigates", "patched", "installs")
- evidence: the exact sentence or clause from the text

**Be exhaustive.** For each entity, check every other entity for possible relationships.
Pay attention to:
- What does this entity USE, TARGET, EXPLOIT, DROP, INSTALL?
- Who DISCOVERED, OBSERVED, ATTRIBUTED, or INVESTIGATED this entity?
- What ORGANIZATION does this entity BELONG TO or PARTNER WITH?
- What VULNERABILITY does an attack pattern EXPLOIT?
- What SOFTWARE VERSION MITIGATES or PATCHES a vulnerability?

### 2. DatatypeProperty — entity literal attributes
Facts about an entity that are NOT relationships to other entities.
- entity: must be a name from the entity list
- attribute: descriptive label for the property
  Common CTI attributes: "malware type", "alias", "implementation language",
  "first seen", "description", "country", "objective",
  "architecture execution environment", "is family"
- value: the literal value (string, ISO 8601 date, boolean "true"/"false")

**DO NOT extract** type/class descriptors as DatatypeProperty — these are already
captured during entity classification:
  - "identity class = organization/group/individual" → skip
  - "type = malware/software/organization" → skip (unless it is a specific subtype
    like malware_type = "ransomware" for a Malware entity)

For each entity in the list, check what attributes the text reveals:
- Malware: malware type (ransomware/spyware/dropper/backdoor/etc.), implementation
  language, is_family (true/false), aliases, description
- ThreatActor/IntrusionSet: aliases, first_seen (ISO 8601), description
- Identity/Organization: country (if mentioned)
- Campaign: objective, first_seen (ISO 8601)
- Location: country code (ISO 3166-1 alpha-2, e.g. "KP", "US")

## Output Format
Return ONLY valid JSON:
{
  "object_properties": [
    {
      "subject": "Quantum",
      "predicate": "uses",
      "object": "BazarCall",
      "evidence": "Quantum was observed employing their version of BazarCall"
    }
  ],
  "datatype_properties": [
    {
      "entity": "3AM",
      "attribute": "malware type",
      "value": "ransomware"
    },
    {
      "entity": "3AM",
      "attribute": "implementation language",
      "value": "Rust"
    }
  ]
}
"""


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _items(data, key: str, *, allow_bare_list: bool = False) -> list[dict]:
    """Read a list-of-objects field out of a parsed LLM JSON payload, defensively.

    The prompts ask for an object such as ``{"entities": [...]}``, but a model is
    free to answer with something else, and weaker models routinely do: a bare
    array instead of the object, or an array of plain strings instead of objects.
    Reaching straight for ``data.get(key)`` turns that into an AttributeError and
    an opaque traceback, which reads as a broken artifact rather than a model that
    did not follow the schema.

    ``allow_bare_list`` accepts a top-level array only where the response carries a
    single list and the mapping is therefore unambiguous (entity extraction). For
    responses that carry several lists, a bare array is ignored rather than guessed
    at. Non-object elements are always dropped.
    """
    if isinstance(data, dict):
        value = data.get(key, [])
    elif isinstance(data, list) and allow_bare_list:
        value = data
    else:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _parse_json(raw: str) -> dict | list | None:
    if not raw:
        return None
    fence = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', raw)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]+\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


async def _llm_call(llm, system: str, human: str) -> dict | None:
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=human),
    ])
    content = response.content
    meta = getattr(response, "response_metadata", {})
    usage = meta.get("token_usage", meta.get("usage", {}))

    if isinstance(content, list):
        raw = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    else:
        raw = str(content).strip()

    return _parse_json(raw), meta.get("finish_reason"), usage


def _normalize_name(name: str) -> str:
    return re.sub(r'^(a |an |the )', '', name.strip(), flags=re.I).strip()


_GENERIC_SUFFIX = re.compile(
    r'\s+(group|team|gang|apt|organization|organisation|'
    r'hacking group|hacking team|threat actor|threat group)$',
    re.I,
)


def _dedup_key(name: str) -> str:
    key = re.sub(r'^(a |an |the )', '', name.strip(), flags=re.I)
    key = key.replace('_', ' ')
    key = re.sub(r'\s+', ' ', key).strip().lower()
    key = _GENERIC_SUFFIX.sub('', key).strip()
    return key


# ── Pass 1.5: Coreference resolution ──────────────────────────────────────────

_COREF_SYSTEM_PROMPT = """\
You are a coreference resolution specialist for Cyber Threat Intelligence (CTI) text.

Given a list of known entities and a CTI text, replace anaphoric references with
the exact entity name from the list.

Anaphoric references to resolve:
- Pronouns: "it", "they", "them", "its", "their"
- Generic descriptors used as stand-ins: "the group", "the actor", "the threat actor",
  "the malware", "the ransomware", "the tool", "the campaign", "the operator",
  "the attacker", "the attackers", "the victim", "the organization", "the company", "the vendor"
- Demonstratives: "this malware", "this group", "that campaign"

Rules:
- Replace ONLY when you are confident about the referent from context
- Use ONLY names exactly as they appear in the entity list
- If the entity list contains a generic actor entity (e.g., "Threat Actors", "Attackers",
  "Unidentified APT Group"), resolve "the attackers", "the group", "they", "them" to it
  when those references clearly denote the threat actor performing actions in the text
- If a reference is ambiguous or could refer to multiple entities, leave it unchanged
- Do not change any other part of the text
- Do not add or remove sentences

Return ONLY the resolved text, with no explanation or preamble.
"""


async def _resolve_coref(llm, chunk: str, entity_names: list[str]) -> str:
    """Replace anaphoric references in chunk with canonical entity names."""
    entity_list = "\n".join(f'- "{n}"' for n in entity_names)
    human = f"Known entities:\n{entity_list}\n\nCTI text:\n{chunk}"
    response = await llm.ainvoke([
        SystemMessage(content=_COREF_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ])
    content = response.content
    if isinstance(content, list):
        resolved = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ).strip()
    else:
        resolved = str(content).strip()
    # Fallback to original if response looks malformed (empty or too short)
    return resolved if len(resolved) > len(chunk) * 0.5 else chunk


# ── Main node ──────────────────────────────────────────────────────────────────

# ── Single-pass combined prompt ────────────────────────────────────────────────

_COMBINED_SYSTEM_PROMPT = """\
You are a Cyber Threat Intelligence (CTI) knowledge extraction specialist.
Extract ALL named entities AND their relationships from the CTI text in a single pass.

## Entities
- Same rules as for entity extraction (threat actors, malware, CVEs, orgs, software, locations, etc.)
- Normalize: remove leading articles. Provide type_hint and literal properties.

## Relationships
- object_properties: entity → entity links (both must be from your extracted entity list)
- datatype_properties: entity → literal value attributes

## Output Format
Return ONLY valid JSON:
{
  "entities": [
    {"name": "...", "type_hint": "...", "properties": {}}
  ],
  "object_properties": [
    {"subject": "...", "predicate": "...", "object": "...", "evidence": "..."}
  ],
  "datatype_properties": [
    {"entity": "...", "attribute": "...", "value": "..."}
  ]
}
"""


async def extractor_node(state: AnchorState) -> dict:
    chunks = state.get("chunks", [])
    print(f"[extractor] input_text len={len(state.get('input_text',''))}, chunks={len(chunks)}")

    no_coref     = os.getenv("ANCHOR_NO_COREF",   "").lower() in ("1", "true")
    single_pass  = os.getenv("ANCHOR_SINGLE_PASS","").lower() in ("1", "true")
    if single_pass:
        print("[extractor] mode=SINGLE-PASS (entities+relations in one call, no coref)")
    elif no_coref:
        print("[extractor] mode=2-PASS, coref=DISABLED")

    llm_pass1 = get_llm(temperature=0)
    llm_pass2 = get_llm(temperature=0, max_tokens=int(os.getenv("MAPPER_MAX_TOKENS", 4096)))
    errors: list[str] = []

    # ── Single-pass mode ──────────────────────────────────────────────────────
    if single_pass:
        async def _combined_chunk(i: int, chunk: str):
            print(f"[extractor/combined] chunk {i}: {len(chunk)} chars — calling LLM...")
            data, finish, usage = await _llm_call(
                llm_pass2, _COMBINED_SYSTEM_PROMPT,
                f"Extract entities and relationships from this CTI text:\n\n{chunk}",
            )
            print(f"[extractor/combined] chunk {i}: finish={finish}, usage={usage}")
            return i, data

        combined_results = await asyncio.gather(*[_combined_chunk(i, c) for i, c in enumerate(chunks)])

        all_entities: list[RawEntity] = []
        all_triplets: list[RawTriplet] = []
        for i, data in combined_results:
            if data is None:
                errors.append(f"Extractor combined chunk {i}: could not parse JSON")
                continue
            for e in _items(data, "entities"):
                if not e.get("name") or not e.get("type_hint"):
                    continue
                all_entities.append(RawEntity(
                    name=_normalize_name(e["name"]),
                    type_hint=e["type_hint"].strip(),
                    properties=e.get("properties", {}),
                ))
            # Build lookup from this chunk's entities for validation
            chunk_lookup = {_dedup_key(e["name"]): e["name"] for e in all_entities}
            def _res(raw): return chunk_lookup.get(_dedup_key(raw))  # noqa: E731
            for t in _items(data, "object_properties"):
                s = _res(t.get("subject", "")); o = _res(t.get("object", ""))
                pred = (t.get("predicate") or "").strip()
                if s and o and pred and s != o:
                    all_triplets.append(RawTriplet(subject=s, predicate=pred, object=o,
                                                    evidence=t.get("evidence",""), property_type="ObjectProperty"))
            for dp in _items(data, "datatype_properties"):
                ent = _res(dp.get("entity", "")); attr = (dp.get("attribute") or "").strip()
                val = str(dp.get("value","")).strip()
                if ent and attr and val:
                    all_triplets.append(RawTriplet(subject=ent, predicate=attr, object=val,
                                                    evidence="", property_type="DatatypeProperty"))
        print(f"[extractor] single-pass total: {len(all_entities)} entities, {len(all_triplets)} triplets")
        return {"raw_entities": all_entities, "raw_triplets": all_triplets, "errors": errors}

    # ── Pass 1: extract entities from all chunks in parallel ──────────────────
    async def _pass1_chunk(i: int, chunk: str):
        print(f"[extractor/pass1] chunk {i}: {len(chunk)} chars — calling LLM...")
        data, finish, usage = await _llm_call(
            llm_pass1, _ENTITY_SYSTEM_PROMPT,
            f"Extract entities from this CTI text:\n\n{chunk}",
        )
        print(f"[extractor/pass1] chunk {i}: finish={finish}, usage={usage}")
        return i, data

    pass1_results = await asyncio.gather(*[_pass1_chunk(i, c) for i, c in enumerate(chunks)])

    pass1_entities: list[RawEntity] = []
    for i, data in pass1_results:
        if data is None:
            errors.append(f"Extractor pass1 chunk {i}: could not parse JSON")
            continue
        for e in _items(data, "entities", allow_bare_list=True):
            if not e.get("name") or not e.get("type_hint"):
                continue
            pass1_entities.append(RawEntity(
                name=_normalize_name(e["name"]),
                type_hint=e["type_hint"].strip(),
                properties=e.get("properties", {}),
            ))

    print(f"[extractor/pass1] done: {len(pass1_entities)} entities across all chunks")
    for e in pass1_entities:
        props = f" {dict(e['properties'])}" if e.get("properties") else ""
        print(f"  entity: {e['name']!r} ({e['type_hint']}){props}")

    # ── Deduplicate entities for pass2 context ─────────────────────────────────
    # name-normalized → entity  (longest type_hint wins)
    deduped: dict[str, RawEntity] = {}
    for e in pass1_entities:
        key = _dedup_key(e["name"])
        if key not in deduped:
            deduped[key] = dict(e)
        else:
            if len(e["type_hint"]) > len(deduped[key]["type_hint"]):
                deduped[key]["type_hint"] = e["type_hint"]
            for k, v in e.get("properties", {}).items():
                if k not in deduped[key]["properties"]:
                    deduped[key]["properties"][k] = v

    deduped_list: list[RawEntity] = list(deduped.values())

    if not deduped_list:
        print("[extractor/pass2] no entities — skipping relationship extraction")
        return {"raw_entities": pass1_entities, "raw_triplets": [], "errors": errors}

    # Format entity list for pass2 prompt
    entity_lines = "\n".join(
        f'- "{e["name"]}" ({e["type_hint"]})'
        for e in deduped_list
    )

    # ── Pass 1.5: Coreference resolution (parallel per chunk) ─────────────────
    if no_coref:
        print("[extractor/coref] SKIPPED (--no-coref)")
        resolved_chunks = list(chunks)
    else:
        entity_names = [e["name"] for e in deduped_list]
        llm_coref = get_llm(temperature=0)

        async def _coref_chunk(i: int, chunk: str) -> tuple[int, str]:
            print(f"[extractor/coref] chunk {i}: resolving coreferences...")
            resolved = await _resolve_coref(llm_coref, chunk, entity_names)
            if resolved != chunk:
                before_words = chunk.split()
                after_words = resolved.split()
                matcher = difflib.SequenceMatcher(None, before_words, after_words, autojunk=False)
                changes = [
                    (' '.join(before_words[i1:i2]), ' '.join(after_words[j1:j2]))
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                    if tag in ('replace', 'delete', 'insert')
                ]
                print(f"[extractor/coref] chunk {i}: {len(changes)} substitution(s)")
                for before, after in changes:
                    print(f'  "{before}" → "{after}"')
            else:
                print(f"[extractor/coref] chunk {i}: no changes")
            return i, resolved

        coref_results = await asyncio.gather(*[_coref_chunk(i, c) for i, c in enumerate(chunks)])
        resolved_chunks = [text for _, text in sorted(coref_results)]

    # ── Pass 2: extract relationships from all chunks in parallel ─────────────
    # Build a lookup: normalized name → canonical name (for ObjectProperty validation)
    name_lookup = {_dedup_key(e["name"]): e["name"] for e in deduped_list}

    def _resolve(raw: str) -> str | None:
        return name_lookup.get(_dedup_key(raw))

    valid_names = ", ".join(f'"{e["name"]}"' for e in deduped_list)

    async def _pass2_chunk(i: int, chunk: str):
        print(f"[extractor/pass2] chunk {i}: {len(chunk)} chars — calling LLM...")
        human = (
            f"Allowed entity names (use ONLY these as subject/object):\n{entity_lines}\n\n"
            f"Valid name set: [{valid_names}]\n\n"
            f"CTI text:\n{chunk}"
        )
        data, finish, usage = await _llm_call(llm_pass2, _RELATION_SYSTEM_PROMPT, human)
        print(f"[extractor/pass2] chunk {i}: finish={finish}, usage={usage}")
        return i, data

    pass2_results = await asyncio.gather(*[_pass2_chunk(i, c) for i, c in enumerate(resolved_chunks)])

    all_triplets: list[RawTriplet] = []
    for i, data in pass2_results:
        if data is None:
            errors.append(f"Extractor pass2 chunk {i}: could not parse JSON")
            continue

        # ObjectProperty — both subject and object must be in entity list
        op_count = 0
        for t in _items(data, "object_properties"):
            subj = _resolve(t.get("subject", ""))
            obj  = _resolve(t.get("object", ""))
            pred = (t.get("predicate") or "").strip()
            if not subj or not obj or not pred or subj == obj:
                continue
            all_triplets.append(RawTriplet(
                subject=subj,
                predicate=pred,
                object=obj,
                evidence=t.get("evidence", "").strip(),
                property_type="ObjectProperty",
            ))
            op_count += 1

        # DatatypeProperty — subject must be in entity list; object is a literal value
        dp_count = 0
        for dp in _items(data, "datatype_properties"):
            entity_name = _resolve(dp.get("entity", ""))
            attr  = (dp.get("attribute") or "").strip()
            value = str(dp.get("value", "")).strip()
            if not entity_name or not attr or not value:
                continue
            all_triplets.append(RawTriplet(
                subject=entity_name,
                predicate=attr,
                object=value,
                evidence="",
                property_type="DatatypeProperty",
            ))
            dp_count += 1

        print(f"[extractor/pass2] chunk {i}: {op_count} object_properties, {dp_count} datatype_properties")
        for t in all_triplets[-(op_count + dp_count):]:
            tag = "[OP]" if t["property_type"] == "ObjectProperty" else "[DP]"
            print(f"  {tag} {t['subject']!r} --[{t['predicate']}]--> {t['object']!r}")

    op_total = sum(1 for t in all_triplets if t["property_type"] == "ObjectProperty")
    dp_total = sum(1 for t in all_triplets if t["property_type"] == "DatatypeProperty")
    print(f"[extractor] total: {len(pass1_entities)} raw entities, "
          f"{op_total} ObjectProperty + {dp_total} DatatypeProperty triplets")

    return {
        "raw_entities": pass1_entities,
        "raw_triplets": all_triplets,
        "errors": errors,
    }
