# Universal Ontology MCP

**The Intelligent Bridge between Unstructured Data and High-Fidelity Knowledge Graphs.**

Universal Ontology MCP is a powerful tool designed for AI assistants to explore, navigate, and populate complex ontologies. It transforms raw text into structured relationships while adhering to strict semantic standards.

## Why Universal Ontology MCP?

Existing ontology tools often struggle with semantic ambiguity and rigid keyword matching. This MCP solves these problems by providing:

- **Dual-Mode Class Discovery**: Two complementary search strategies — fast embedding-based search (`search_classes`) and LLM-driven hierarchical navigation (`recurse_classes`) — cover each other's failure modes.
- **LOW CONFIDENCE Detection**: `search_*` tools automatically flag results below the confidence threshold and suggest the appropriate `recurse_*` fallback tool.
- **Scoped Property Search via `domain_class`**: When a property search returns LOW CONFIDENCE, `recurse_*` tools show a collapsed summary of large property groups grouped by their defining class (`rdfs:domain`). Pass any group's class URI as `domain_class` to `search_attributes` or `search_relations` to re-run embedding search within that scoped pool — works identically across UCO, STIX, MalOnt, and any other ontology.
- **Embedding Cache**: Ontology embeddings are computed once and cached to disk. Subsequent startups skip re-encoding entirely, dramatically reducing load time for large ontologies. Cache is automatically invalidated when ontology files change.
- **Batch Embedding Encoding**: All class and property embeddings are computed in a single batched pass at startup rather than one-by-one, making initial load significantly faster.
- **Balanced Semantic Search**: Uses state-of-the-art embeddings (default: `all-MiniLM-L6-v2`) with a **50/50 weighted scoring** between class names and descriptions for intuitive matching.
- **Context-Aware Property Search**: The engine analyzes **Domain/Range constraints** of your entities and recommends the most valid `DatatypeProperty` or `ObjectProperty`, preventing schema violations.
- **Advanced Facet Discovery**: Locates related Facet classes through four complementary strategies — direct property range, `rdfs:subPropertyOf` chain, SHACL `sh:node` resolution, and name heuristics.
- **Structured Validation Output**: `validate_entity` returns structured JSON detailing exactly which properties are missing or exceed limits, along with actionable fix instructions the AI can act on immediately.
- **Duplicate-Safe Entity Creation**: `create_entity` detects pre-existing entities and reports which types were added, skipped (already assigned), or invalid — no silent overwrites.
- **Automatic Property Normalization**: Detects and normalizes non-standard types like `owl:DataProperty` into the standard `owl:DatatypeProperty` for consistent mapping.
- **Component-Based Modeling**: Simplifies complex modeling (like UCO Facets) by recommending relevant components for any given class.
- **ASCII Graph Visualization**: Preview your knowledge graph structure directly in the chat interface before exporting.
- **Built-in SHACL Validation**: Ensures data integrity with schema constraint checking before export.
- **Session Control & Sanitization**: Surgically remove entities, reset the session, prune disconnected islands, and enjoy automatic URI sanitization.
- **Multi-Format Ontology Support**: Loads `.ttl`, `.owl`, `.rdf`, `.n3`, `.nt`, `.trig`, `.jsonld` files simultaneously with automatic format detection.

---

## Supported Ontology Formats

The engine automatically detects and loads all recognized ontology files under `ONTOLOGY_DIR` recursively.

| Extension | Format | Description |
|-----------|--------|-------------|
| `.ttl` | Turtle | Human-readable RDF syntax. Used by UCO, CASE, schema.org |
| `.owl` | OWL/RDF-XML | Protégé-native format. Used by STIX 2.1 OWL, TAC, TAL |
| `.rdf` | RDF/XML | Generic RDF/XML format |
| `.n3` | Notation3 | Extended Turtle superset |
| `.nt` | N-Triples | Line-based RDF format |
| `.trig` | TriG | Named graph RDF format |
| `.jsonld` | JSON-LD | JSON-based RDF format |

These are all different **serialization formats** of the same underlying RDF/OWL data model — the engine merges them into a single graph regardless of format.

---

## Ontology Examples

### UCO (Turtle / `.ttl`)

[Unified Cyber Ontology](https://github.com/ucoProject/UCO) uses Turtle format. Point `ONTOLOGY_DIR` to the `ontology/` subdirectory:

```
UCO/ontology/          ← point here
├── core/
├── observable/
├── action/
└── ...
```

### STIX 2.1 (OWL / `.owl`)

The [tac-ontology](https://github.com/oasis-open/tac-ontology) repository provides a formal OWL formalization of STIX 2.1. For pure STIX 2.1 CTI work, point to the `stix/` subdirectory:

```
tac-ontology/stix/     ← point here for STIX 2.1 only
├── core-objects/
│   ├── sdo/           ← AttackPattern, ThreatActor, Malware, ...
│   ├── sco/           ← IPv4Address, File, NetworkTraffic, ...
│   └── sro/           ← Relationship, Sighting
├── meta-objects/
├── bundle-object/
└── vocabularies/
```

> **Note on ATT&CK TTPs**: MITRE ATT&CK Techniques (e.g. T1059.001) are instances of `stix:AttackPattern`, with the T-number stored in `external_references`. The full ATT&CK framework is officially distributed as STIX 2.1 data.

---

## Discovery Strategy

### Class Discovery

| Situation | Tool |
|-----------|------|
| You know the concept name | `search_classes("Vulnerability")` — one concept per call |
| Search returns [LOW CONFIDENCE] | `recurse_classes("vulnerability")` — read descriptions, pick branch, repeat |
| Want a structural overview | `show_class_tree()` or `show_class_tree('<uri>', depth=3)` |

**Important**: Always pass **one concept per query** to `search_classes`. Multi-concept queries dilute the embedding vector and return poor results. Run one call per concept, in parallel if possible.

#### `recurse_classes` — Iterative Hierarchical Navigation

When `search_classes` returns [LOW CONFIDENCE], use `recurse_classes` to navigate the class tree yourself:

```
1. recurse_classes("vulnerability")
   → Shows root classes with full descriptions. Read and pick the best branch.

2. recurse_classes("vulnerability", "https://.../UcoObject")
   → Shows UcoObject's subclasses. Pick the next branch.

3. Repeat until you reach the right class or a leaf node.

4. Confirm with get_class_details('<uri>') before creating entities.
```

Unlike `search_classes`, this tool computes **no embeddings** — you (the LLM) read the descriptions and decide.

---

### Property Discovery

Property search uses a two-tier fallback that works identically across any ontology (UCO, STIX, MalOnt, or custom):

```
Tier 1 — Embedding search (always try first)
  search_attributes(entity_uri, query, value?, context?)
  search_relations(subject_uri, object_uri, query, context?)

  → If HIGH CONFIDENCE: use the result directly → set_property
  → If LOW CONFIDENCE: proceed to Tier 2

Tier 2 — Group inspection
  recurse_attributes(entity_uri, query)
  recurse_relations(subject_uri, object_uri, query)

  Direct / Inherited / Subject-side / Object-side / Inverse sections
  are shown in full. Sections exceeding RECURSE_SECTION_CAP items are
  collapsed into a per-defining-class summary:

    [Other Properties — collapsed]
      • FileFacet       (12 props)  Metadata about a file on disk
        → search_attributes(entity_uri, query, domain_class="https://.../FileFacet")
      • (no domain declared)  (143 props)
        → included in normal search_attributes without domain_class

  → If a collapsed group looks promising: call search_* with domain_class=<URI>
    to re-run embedding search scoped to that class's properties.
```

The `domain_class` parameter is a plain `rdfs:domain` URI — it carries no ontology-specific meaning and works regardless of whether the ontology uses Facets, flat properties, or sparse declarations.

---

## Tool Reference

### Schema Inspection

| Tool | Description |
|------|-------------|
| `get_ontology_summary` | Quick high-level overview: class count, property count, SHACL shapes, namespaces. |
| `search_classes(query, limit, context, namespace_filter)` | Embedding + keyword hybrid search. **One concept per query.** Flags [LOW CONFIDENCE] when score is below threshold. |
| `recurse_classes(query, class_uri?)` | LLM-driven top-down navigation. Fallback when `search_classes` is LOW CONFIDENCE. No embeddings — shows names and descriptions for your judgment. |
| `get_class_details(class_uri)` | Detailed usage instructions, SHACL-derived property requirements, and connectivity rules for a class. |
| `show_class_tree(class_uri?, depth)` | Displays the class hierarchy as an indented tree for structural orientation. |
| `list_subclasses(class_uri)` | Lists immediate subclasses of a class. |
| `list_root_classes` | Lists top-level classes with no parents. |

### Property Search

| Tool | Description |
|------|-------------|
| `search_attributes(entity_uri, query, value?, context?, domain_class?)` | Finds the best `DatatypeProperty` for an entity. Context-aware, domain/range-aware, facet-traversing. Flags [LOW CONFIDENCE] → use `recurse_attributes`. Pass `domain_class=<URI>` to scope the search to a specific defining class. |
| `recurse_attributes(entity_uri, query, value?)` | Browses DatatypeProperties grouped by Direct / Inherited (shown in full) and collapsed groups by defining class. Use `search_attributes(..., domain_class=<URI>)` to search within a collapsed group. |
| `search_relations(subject_uri, object_uri, query, context?, domain_class?)` | Finds the best `ObjectProperty` to connect two entities. Analyzes domain/range constraints and structural paths. Flags [LOW CONFIDENCE] → use `recurse_relations`. Pass `domain_class=<URI>` to scope the search. |
| `recurse_relations(subject_uri, object_uri, query)` | Browses ObjectProperties grouped by Direct / Subject-side / Object-side / Inverse (shown in full) and collapsed groups by defining class. Use `search_relations(..., domain_class=<URI>)` to search within a collapsed group. |

### Graph Construction

| Tool | Description |
|------|-------------|
| `create_entity(entity_id, class_uris)` | Creates a multi-typed entity. Reports added, duplicate-skipped, and invalid class URIs separately. |
| `set_property(entity_uri, property_uri, value)` | Sets a property value. Values starting with `http` are automatically treated as URI links. |
| `attach_component(entity_uri, component_class, connection_prop, attributes)` | Groups metadata into a Facet/Component node. Uses UUID-based URIs to prevent collisions. |
| `remove_entity(entity_uri)` | Deletes an entity and all its incoming/outgoing triples, including attached components. |
| `reset_graph` | Clears all instance data while preserving the loaded ontology schema. |

### Graph Analysis & Export

| Tool | Description |
|------|-------------|
| `validate_entity(entity_uri)` | Validates an entity against SHACL constraints. Returns structured JSON with error type, missing property details, and fix instructions. |
| `visualize_graph(verbose?)` | ASCII tree preview of the current instance graph. `verbose=True` shows full URIs. |
| `get_graph_data` | Returns the current graph as JSON (nodes + edges) for programmatic use. |
| `prune_islands(min_size?)` | Removes disconnected entity clusters smaller than `min_size`. |
| `export_graph(filename?)` | Exports all instance data to a Turtle (`.ttl`) file. |

---

## Evaluation Export

`export_eval_json` writes a structured JSON file for automated evaluation against ground truth annotations. This tool is **disabled by default** and must be explicitly enabled via the `ENABLE_EVAL_TOOLS` environment variable. Only use it when explicitly requested for evaluation purposes.

The LLM reads the current graph (via `visualize_graph`) and supplies the entities and triplets directly — no automatic extraction. This keeps the tool ontology-agnostic: Action nodes, Relationship nodes, or any other reification pattern is interpreted by the LLM rather than hard-coded logic.

**Output format:**

```json
{
  "file": "analysis.json",
  "text": "Symantec's Threat Hunter Team observed a cyber attack...",
  "ontology": "stix",
  "extracted_entities": [
    {"name": "3AM Ransomware",              "class": "http://docs.oasis-open.org/cti/ns/stix#Malware"},
    {"name": "Symantec's Threat Hunter Team","class": "http://docs.oasis-open.org/cti/ns/stix#Identity"}
  ],
  "extracted_triplets": [
    {
      "subject": "Symantec's Threat Hunter Team",
      "relation": "observed",
      "property": "http://docs.oasis-open.org/cti/ns/stix#sighting_of_ref",
      "property_type": "ObjectProperty",
      "object": "3AM Ransomware"
    },
    {
      "subject": "3AM Ransomware",
      "relation": "programming language",
      "property": "http://docs.oasis-open.org/cti/ns/stix#languages",
      "property_type": "DatatypeProperty",
      "object": "Rust"
    }
  ]
}
```

- **entities**: Real-world entities only. Omit auxiliary graph nodes (Action nodes, Relationship nodes, reification bridges) — those become triplets instead. `class` is the full ontology class URI.
- **triplets**: Derive subject → relation → object by interpreting Action nodes, Relationship nodes, or any other reification pattern in the graph. `property` is the full ontology property URI; `property_type` distinguishes `ObjectProperty` from `DatatypeProperty`.
- **ontology**: Auto-detected from `ONTOLOGY_DIR` path or loaded namespace URIs.
- **property_type**: Verified against the loaded schema when the property URI is known; falls back to the LLM-supplied value otherwise.

---

## Architecture

```
universal-ontology-mcp/
├── main.py                  # Entry point — runs the FastMCP server
├── requirements.txt
├── settings_template.json   # MCP client configuration template
└── mcp_server/
    ├── engine.py            # Core engine: ontology loading, embedding cache,
    │                        # schema cache, SHACL extraction, semantic search,
    │                        # facet discovery, candidate relation finding
    ├── server.py            # FastMCP tool definitions (all @mcp.tool() handlers)
    └── config.py            # Persona instructions, environment variable defaults
```

### Engine Internals

- **Embedding Cache**: On startup, checks `.embedding_cache.pkl` in the ontology directory. If the cache key (MD5 of all file paths + sizes + mtimes) matches, embeddings are restored from disk — no re-encoding. On cache miss, all embeddings are batch-encoded in a single pass and saved.
- **SHACL Integration**: `_build_schema_cache()` extracts `sh:NodeShape` constraints, resolves `sh:node → sh:targetClass` chains for facet discovery, and augments property domain lists from shape definitions.
- **Facet Discovery** (`get_related_facets`): Four strategies in order — (1) direct hasFacet-like property range, (2) `rdfs:subPropertyOf` chain, (3) SHACL `sh:node` resolution, (4) shape path name heuristic.
- **Domain-Class Grouping** (`_domain_class_summary`): When a `recurse_*` section exceeds `RECURSE_SECTION_CAP`, properties are grouped by their `rdfs:domain` class and rendered as a count summary. Each group includes the class description and a ready-to-copy `search_*(..., domain_class=<URI>)` call for scoped re-search.

---

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your ontology directory (the engine will recursively load all recognized files):
   ```bash
   export ONTOLOGY_DIR="/path/to/your/ontology/folder"
   ```

---

## MCP Configuration

Add this configuration to your MCP-compatible client (e.g., Claude Desktop, Claude Code, Gemini CLI).

### Claude Desktop / Claude Code / Gemini CLI

```json
{
  "mcpServers": {
    "universal-ontology-mcp": {
      "command": "python",
      "args": ["/absolute/path/to/universal-ontology-mcp/main.py"],
      "env": {
        "ONTOLOGY_DIR": "/absolute/path/to/your/ontology/folder",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

### OpenCode

```json
{
  "mcp": {
    "universal-ontology-mcp": {
      "type": "local",
      "command": ["python", "/absolute/path/to/universal-ontology-mcp/main.py"],
      "environment": {
        "ONTOLOGY_DIR": "/absolute/path/to/your/ontology/folder",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ONTOLOGY_DIR` | `./ontology` | Path to directory containing ontology files (recursive) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for semantic search |
| `RECURSE_SECTION_CAP` | `20` | Max properties shown per section in `recurse_*` tools. Sections exceeding this threshold are collapsed into a domain-class summary. |
| `ENABLE_EVAL_TOOLS` | `false` | Set to `true` to enable `export_eval_json`. Off by default — only for evaluation use. |

---

## License

This project is licensed under the [MIT License](LICENSE).
