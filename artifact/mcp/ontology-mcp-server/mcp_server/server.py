import json
import uuid
import rdflib
import urllib.parse
from rdflib.namespace import RDF
from mcp.server.fastmcp import FastMCP
from .engine import OntologyEngine, _local_name
from .config import MAPPING_GUIDELINES, DEFAULT_ONTOLOGY_DIR, EMBEDDING_MODEL, ENABLE_EVAL_TOOLS, RECURSE_SECTION_CAP

import logging

# Score adjustment constants — all overridable via environment variables.
# Variable names follow the same convention as config.py (no project-specific prefix).
import os as _os
KEYWORD_BONUS            = float(_os.getenv("KEYWORD_BONUS",            "0.3"))
SEMANTIC_MATCH_THRESHOLD = float(_os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.6"))
DATATYPE_MATCH_BOOST     = float(_os.getenv("DATATYPE_MATCH_BOOST",     "0.2"))
DIRECT_RELATION_BOOST    = float(_os.getenv("DIRECT_RELATION_BOOST",    "0.1"))
INVERSE_RELATION_PENALTY = float(_os.getenv("INVERSE_RELATION_PENALTY", "0.05"))
LOW_CONFIDENCE_CLASS_THRESHOLD = float(_os.getenv("LOW_CONFIDENCE_CLASS_THRESHOLD", "0.45"))
LOW_CONFIDENCE_PROP_THRESHOLD  = float(_os.getenv("LOW_CONFIDENCE_PROP_THRESHOLD",  "0.3"))
RANGE_MATCH_BOOST        = float(_os.getenv("RANGE_MATCH_BOOST",        "0.15"))
_NAME_WEIGHT             = float(_os.getenv("EMB_NAME_WEIGHT",          "0.5"))
_EMB_WEIGHTS             = (_NAME_WEIGHT, 1.0 - _NAME_WEIGHT)

# Silence noisy logs from FastMCP and underlying libraries
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize FastMCP server
mcp = FastMCP("Universal Ontology Core", instructions=MAPPING_GUIDELINES)

# Global manager instance
engine = OntologyEngine(DEFAULT_ONTOLOGY_DIR, model_name=EMBEDDING_MODEL)


def _rdf_value(value: str):
    """Returns an rdflib URIRef if value looks like a URI, otherwise a Literal."""
    return rdflib.URIRef(value) if str(value).startswith("http") else rdflib.Literal(value)


def _fmt_prop(uri: str, note: str = "", show_domain: bool = True) -> list[str]:
    """Formats a single property for display in recurse_* tools."""
    info = engine.properties.get(uri, {})
    dom_short = [_local_name(d) for d in info.get('domain', [])]
    rng_short  = [_local_name(r) for r in info.get('range', [])]
    out = [f"  • {info.get('name', uri)}",
           f"    URI: {uri}",
           f"    Description: {info.get('comment', 'No description')}"]
    if show_domain and dom_short:
        out.append(f"    Domain: {', '.join(dom_short)}")
    if rng_short:
        out.append(f"    Range/XSD Type: {', '.join(rng_short)}")
    if note:
        out.append(f"    Note: {note}")
    return out


def _section(title: str, items: list, note: str = "", max_items: int = 0) -> list[str]:
    """Renders a labeled section of properties, with optional cap.

    items: list of str (URI) or tuple (URI, note_str).
    max_items: if > 0, caps the displayed items and appends a count of omitted ones.
    """
    if not items:
        return []
    total = len(items)
    cap = max_items if max_items > 0 else total
    header = f"[{title}]  ({total} properties)" + (f"  — {note}" if note else "")
    out = [header]
    for item in items[:cap]:
        if isinstance(item, tuple):
            p_uri, item_note = item
            out.extend(_fmt_prop(p_uri, note=item_note))
        else:
            out.extend(_fmt_prop(item, show_domain=False))
        out.append("")
    if total > cap:
        out.append(f"  ... {total - cap} more properties not shown.")
        out.append("")
    return out


def _domain_class_summary(prop_uris: list[str], search_tool: str, search_args: str) -> list[str]:
    """Collapses a large property list into a per-domain-class count summary.

    Groups properties by their first rdfs:domain class (or '(no domain declared)').
    Instructs the LLM to call search_tool with domain_class=<URI> to narrow the search.
    """
    groups: dict[str, list[str]] = {}  # domain_class_uri_or_sentinel -> [prop_uris]
    NO_DOMAIN = "(no domain declared)"

    for uri in prop_uris:
        info = engine.properties.get(uri, {})
        domain = info.get('domain', [])
        key = domain[0] if domain else NO_DOMAIN
        groups.setdefault(key, []).append(uri)

    total = len(prop_uris)
    out = [f"  {total} properties across {len(groups)} defining classes — collapsed to avoid flooding."]
    out.append("  Identify the relevant class below, then narrow the search with domain_class=.")
    out.append("")

    for cls_uri, uris in sorted(groups.items(), key=lambda x: -len(x[1])):
        if cls_uri == NO_DOMAIN:
            cls_label = NO_DOMAIN
            cls_desc = "no rdfs:domain declared — already included in normal search_* calls"
            domain_arg = ""
        else:
            cls_info = engine.classes.get(cls_uri, {})
            cls_label = cls_info.get('name') or _local_name(cls_uri)
            cls_desc = cls_info.get('comment', '') or ''
            if len(cls_desc) > 80:
                cls_desc = cls_desc[:77] + "..."
            domain_arg = f', domain_class="{cls_uri}"'

        out.append(f"  • {cls_label}  ({len(uris)} props)  {cls_uri if cls_uri != NO_DOMAIN else ''}")
        if cls_desc:
            out.append(f"    {cls_desc}")
        if domain_arg:
            out.append(f"    → {search_tool}({search_args}{domain_arg})")
        out.append("")

    return out


@mcp.tool()
def list_root_classes() -> str:
    """Lists the top-level classes in the ontology (those with no parents)."""
    roots = []
    for uri, info in engine.classes.items():
        if not info['superclasses']:
            roots.append(f"- {info['name']} ({uri}): {info['comment']}")
    return "\n".join(sorted(roots)) or "No root classes found."

@mcp.tool()
def list_subclasses(class_uri: str) -> str:
    """Lists the immediate subclasses of a given class."""
    if class_uri not in engine.classes: return "Error: Class not found."
    subs = engine.classes[class_uri]['subclasses']
    result = []
    for uri in subs:
        cls = engine.classes[uri]
        result.append(f"- {cls['name']} ({uri}): {cls['comment']}")
    return "\n".join(sorted(result)) or "No subclasses found."

@mcp.tool()
def recurse_classes(query: str, class_uri: str = None) -> str:
    """LLM-driven hierarchical class finder. Use when search_classes returns LOW CONFIDENCE.

    YOU read class names and descriptions at each level and decide which branch to follow.
    No embeddings — pure LLM judgment based on descriptions.

    Each call shows TWO levels at once (current level + their children previewed),
    so fewer calls are needed to reach the right class.

    Workflow:
      1. Call with only query → see root classes + their children previewed.
      2. Pick the class whose description best matches your concept.
      3. Call again with that class URI → go deeper, see its children + grandchildren.
      4. Repeat until you find the right class (leaf or clear match).
      5. Confirm with get_class_details('<uri>') before creating entities.

    Args:
        query: What you are looking for. Shown as context — you do the reasoning.
        class_uri: Class to expand. Omit to start from root classes.
    """
    def _render_class(uri: str, indent: str = "") -> list[str]:
        """Renders one class with its direct children previewed."""
        cls = engine.classes.get(uri, {})
        children = sorted(cls.get('subclasses', []), key=lambda u: engine.classes.get(u, {}).get('name', u))
        n_children = len(children)

        lines = []
        child_hint = f"[{n_children} subclasses]" if n_children else "[leaf]"
        lines.append(f"{indent}• {cls.get('name', uri)}  {child_hint}")
        lines.append(f"{indent}  URI: {uri}")
        lines.append(f"{indent}  {cls.get('comment', '')}")

        # Preview children (names only, so LLM can see what's inside this branch)
        if children:
            preview_names = ", ".join(
                engine.classes.get(c, {}).get('name', _local_name(c))
                for c in children[:8]
            )
            if len(children) > 8:
                preview_names += f" ... (+{len(children) - 8} more)"
            lines.append(f"{indent}  Children: {preview_names}")
        return lines

    if class_uri:
        if class_uri not in engine.classes:
            return f"Error: Class '{class_uri}' not found."
        info = engine.classes[class_uri]
        children = sorted(info['subclasses'], key=lambda u: engine.classes.get(u, {}).get('name', u))

        lines = [
            f"Looking for: '{query}'",
            f"Current: {info['name']}  ({class_uri})",
            f"Description: {info['comment']}",
            "",
        ]

        if not children:
            lines.append("This is a leaf class (no subclasses).")
            lines.append("→ If this matches your concept, use get_class_details to confirm.")
            lines.append("→ If not, go back and try a different branch.")
            return "\n".join(lines)

        lines.append(f"Subclasses ({len(children)}) — read descriptions and pick the best branch:")
        lines.append("-" * 60)
        for uri in children:
            lines.extend(_render_class(uri, indent=""))
            lines.append("")

    else:
        roots = sorted(
            [u for u, i in engine.classes.items() if not i['superclasses']],
            key=lambda u: engine.classes[u]['name']
        )
        lines = [
            f"Looking for: '{query}'",
            f"Root classes ({len(roots)}) — pick the branch most likely to contain your concept:",
            "-" * 60,
        ]
        for uri in roots:
            lines.extend(_render_class(uri, indent=""))
            lines.append("")

    lines.append("→ recurse_classes(query, '<chosen_uri>') to go deeper into a branch.")
    lines.append("→ get_class_details('<uri>') to confirm a match before creating an entity.")
    return "\n".join(lines)


@mcp.tool()
def show_class_tree(class_uri: str = None, depth: int = 2) -> str:
    """Displays the class hierarchy as an indented tree (read-only, no navigation).

    Use this to get a structural overview of the ontology or a specific subtree.
    For interactive top-down search, use recurse_classes instead.

    Args:
        class_uri: Root of the subtree to display. If omitted, shows all top-level classes.
        depth: How many levels to expand (1–4 recommended). Branches beyond this depth
               show a '[N subclasses]' count instead of expanding further.
    """
    def _render(uri: str, current_depth: int, is_last: bool, prefix: str, visited: set) -> list[str]:
        if uri not in engine.classes or uri in visited:
            return []
        visited.add(uri)
        info = engine.classes[uri]
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        n_children = len(info['subclasses'])
        count_hint = f"  [{n_children} subclasses]" if n_children and current_depth >= depth else ""
        lines = [f"{prefix}{connector}{info['name']}{count_hint}  ({uri})"]

        if current_depth < depth and n_children:
            children = sorted(info['subclasses'], key=lambda u: engine.classes.get(u, {}).get('name', u))
            for i, child_uri in enumerate(children):
                lines.extend(_render(child_uri, current_depth + 1, i == len(children) - 1, child_prefix, visited))
        return lines

    # Determine starting nodes
    if class_uri:
        if class_uri not in engine.classes:
            return f"Error: Class '{class_uri}' not found."
        roots = [class_uri]
        header = f"Class tree from: {engine.classes[class_uri]['name']} (depth={depth})\n"
    else:
        roots = sorted(
            [u for u, info in engine.classes.items() if not info['superclasses']],
            key=lambda u: engine.classes[u]['name']
        )
        header = f"Top-level class tree (depth={depth})\n"

    lines = [header]
    visited: set = set()
    for i, root_uri in enumerate(roots):
        info = engine.classes.get(root_uri, {})
        n_children = len(info.get('subclasses', []))
        count_hint = f"  [{n_children} subclasses]" if n_children and depth == 0 else ""
        lines.append(f"{info.get('name', root_uri)}{count_hint}  ({root_uri})")
        if depth > 0:
            children = sorted(info.get('subclasses', []), key=lambda u: engine.classes.get(u, {}).get('name', u))
            for j, child_uri in enumerate(children):
                lines.extend(_render(child_uri, 1, j == len(children) - 1, "", visited))
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def search_classes(query: str, limit: int = 5, context: str | None = None, namespace_filter: str | None = None) -> str:
    """Search for classes using Hybrid Search (Keyword + Semantic Embeddings).

    IMPORTANT: Query must contain ONE concept only (e.g. "Vulnerability", "Organization", "Malware").
    Multi-concept queries dilute the embedding — call this tool once per concept, in parallel if possible.

    Args:
        query: The concept to search for. One concept only.
        limit: Number of results to return (default 5).
        context: Original source sentence containing the entity. Providing this significantly
                 improves accuracy for technical terms (e.g. "Apple patched CVE-2023-41991 in iOS").
        namespace_filter: Restrict search to a specific namespace (e.g. "observable", "identity",
                          "tool"). Use get_ontology_summary to see available filter keys.
    """
    # Apply namespace filter
    if namespace_filter:
        class_uris = [u for u in engine.classes if namespace_filter.lower() in u.lower()]
        if not class_uris:
            return (f"No classes found matching namespace_filter='{namespace_filter}'. "
                    f"Use get_ontology_summary to see valid filter keys.")
    else:
        class_uris = list(engine.classes.keys())

    # Combine query with context for richer embedding
    combined_query = f"{query} {context}" if context else query

    name_embeddings = [engine.classes[u]['name_embedding'] for u in class_uris]
    comment_embeddings = [engine.classes[u]['comment_embedding'] for u in class_uris]
    sim_scores = engine.weighted_similarity(combined_query, name_embeddings, comment_embeddings, weights=_EMB_WEIGHTS)

    query_lower = query.lower()
    final_scores = []
    for i, uri in enumerate(class_uris):
        cls = engine.classes[uri]
        base_score = sim_scores[i]
        keyword_bonus = KEYWORD_BONUS if query_lower in cls['name'].lower() or query_lower in cls['comment'].lower() else 0
        final_scores.append((base_score + keyword_bonus, uri))

    sorted_res = sorted(final_scores, key=lambda x: x[0], reverse=True)[:limit]
    if not sorted_res:
        return f"No results found for '{query}'."

    top_score = sorted_res[0][0]
    low_confidence = top_score < LOW_CONFIDENCE_CLASS_THRESHOLD

    lines = []
    if low_confidence:
        lines.append(f"[LOW CONFIDENCE — top score: {top_score:.4f}, threshold: {LOW_CONFIDENCE_CLASS_THRESHOLD}]")
        lines.append(f"Embedding search could not find a strong match for '{query}'.")
        lines.append("Recommended actions:")
        if not context:
            lines.append(f"  1. Retry with context: search_classes(\"{query}\", context=\"<original sentence>\") ")
        if not namespace_filter:
            lines.append(f"  {'2' if not context else '1'}. Narrow scope: search_classes(\"{query}\", namespace_filter=\"<key from get_ontology_summary>\")")
        lines.append(f"  3. Navigate manually: recurse_classes(\"{query}\")")
        lines.append(f"  4. If no class fits after all attempts, use owl:Thing as fallback:")
        lines.append(f"     create_entity(\"<id>\", [\"http://www.w3.org/2002/07/owl#Thing\"])")
        lines.append(f"     This preserves the entity and its relations without forcing a wrong type.")
        lines.append("")
        lines.append("Low-confidence candidates (for reference only):")
    else:
        ns_note = f" in namespace '{namespace_filter}'" if namespace_filter else ""
        ctx_note = f" [with context]" if context else ""
        lines.append(f"Found {len(sorted_res)} candidates for '{query}'{ns_note}{ctx_note}:")

    for i, (score, uri) in enumerate(sorted_res):
        cls = engine.classes[uri]
        lines.append(f"{i+1}. {cls['name']} (Score: {score:.4f})")
        lines.append(f"   URI: {uri}")
        lines.append(f"   Description: {cls['comment']}")

        # Show parent class for hierarchy context
        parents = cls.get('superclasses', [])
        if parents:
            parent_names = [engine.classes[p]['name'] if p in engine.classes else _local_name(p)
                            for p in parents[:2]]
            lines.append(f"   Parent: {', '.join(parent_names)}")

        # Show subclasses so mapper knows to go deeper for specificity
        subclasses = cls.get('subclasses', [])
        if subclasses:
            sub_names = [engine.classes[s]['name'] if s in engine.classes else _local_name(s)
                         for s in subclasses[:8]]
            suffix = f" (+{len(subclasses) - 8} more)" if len(subclasses) > 8 else ""
            lines.append(f"   Subclasses [{len(subclasses)}]: {', '.join(sub_names)}{suffix}")
            lines.append(f"   ↳ Prefer a subclass if it fits more precisely.")

        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def search_attributes(entity_uri: str, query: str, value: str | None = None, context: str | None = None, domain_class: str | None = None) -> str:
    """
    Recommends the best DatatypeProperty to set on an entity using embedding-based search.
    Call this for EVERY literal value you want to attach after create_entity.
    It is context-aware, facet-traversing, and domain/range-validated.

    If the top result score is LOW CONFIDENCE, use recurse_attributes for manual inspection.
    recurse_attributes may show collapsed property groups — use domain_class to search within one.

    - entity_uri: The URI of the already-created entity.
    - query: What the value represents (e.g. 'version number', 'identifier', 'timestamp').
    - value: The actual data value (e.g. '16.7', '2023-09-21'). Used for type inference.
    - context: The original source sentence or field name for better semantic matching.
    - domain_class: Optional URI of a defining class to restrict the candidate pool.
                    Use when recurse_attributes shows a collapsed group and you want to search
                    within a specific class's properties (e.g. domain_class="https://.../FileFacet").
                    Omit to search all properties (including those with no domain constraint).
    """
    subj_uri = rdflib.URIRef(entity_uri)
    types = [str(t) for t in engine.graph.objects(subj_uri, RDF.type)]

    if not types:
        return f"Error: No type defined for entity '{entity_uri}'. Please use create_entity first."

    # 1. Collect all valid DatatypeProperties for the entity's types (including facets)
    candidate_uris = []
    main_types = set()
    for t in types:
        main_types.add(t)
        main_types.update(engine.get_transitive_superclasses(t))
        candidate_uris.extend(engine.get_properties_for_class(t, "DatatypeProperty", include_facets=True))
    candidate_uris = list(set(candidate_uris))

    # 1b. Scope to a specific defining class if requested
    if domain_class:
        scoped = [u for u in candidate_uris if domain_class in engine.properties.get(u, {}).get('domain', [])]
        if scoped:
            candidate_uris = scoped
        else:
            return (f"No DatatypeProperties found with domain_class='{domain_class}' for this entity.\n"
                    f"Check the class URI from recurse_attributes output, or omit domain_class to search all.")

    if not candidate_uris:
        return f"No valid DatatypeProperties found for the classes: {types}"

    # 2. Refine query with context if provided
    combined_query = f"{query} {context}" if context else query

    # 3. Rank candidates by semantic similarity
    name_embs = [engine.properties[u]['name_embedding'] for u in candidate_uris]
    comment_embs = [engine.properties[u]['comment_embedding'] for u in candidate_uris]
    sim_scores = engine.weighted_similarity(combined_query, name_embs, comment_embs, weights=_EMB_WEIGHTS)

    # 4. Boost score if range matches inferred value type
    inferred_type = engine.infer_datatype(value) if value else None

    final_results = []
    for score, uri in zip(sim_scores, candidate_uris):
        info = engine.properties[uri]
        if inferred_type and any(inferred_type in r for r in info['range']):
            score += DATATYPE_MATCH_BOOST
        final_results.append((score, uri))

    results = sorted(final_results, key=lambda x: x[0], reverse=True)[:5]
    top_score = results[0][0] if results else 0.0
    low_confidence = top_score < LOW_CONFIDENCE_PROP_THRESHOLD

    scope_note = f"  Scoped to: {_local_name(domain_class)}" if domain_class else ""
    res = [f"Recommended attributes for {entity_uri}{scope_note}"]
    if value:
        res.append(f"   Target Value: '{value}' (Inferred Type: {inferred_type.split('#')[-1] if inferred_type else 'Unknown'})")
    res.append(f"   Intent/Context: '{combined_query}'")

    if low_confidence:
        res.append(f"[LOW CONFIDENCE — top score: {top_score:.4f}, threshold: {LOW_CONFIDENCE_PROP_THRESHOLD}]")
        res.append(f"Embedding search could not find a strong match for '{query}'.")
        res.append(f"Recommended action: recurse_attributes(\"{entity_uri}\", \"{query}\")"
                   + (f", value=\"{value}\"" if value else "") + " to browse property groups.")
        res.append("")
        res.append("Low-confidence candidates (for reference only):")
    res.append("-" * 40)

    for i, (score, uri) in enumerate(results):
        info = engine.properties[uri]
        res.append(f"{i+1}. {info['name']} ({uri}) [Score: {score:.4f}]")
        res.append(f"   Description: {info['comment'] or 'No description'}")

        is_indirect = not any(d in main_types for d in info['domain'])
        if is_indirect and info['domain']:
            dom_name = _local_name(info['domain'][0])
            res.append(f"   TIP: Defined on [{dom_name}]. Use attach_component for this class first.")
        res.append("")

    return "\n".join(res)


@mcp.tool()
def recurse_attributes(entity_uri: str, query: str, value: str = None) -> str:
    """
    LLM-driven DatatypeProperty browser. Use when search_attributes returns LOW CONFIDENCE.

    Direct and Inherited properties are shown in full.
    Larger groups (properties from other defining classes, unconstrained properties) are
    collapsed into a per-class summary to avoid flooding — use search_attributes with
    domain_class=<URI> to run a scoped embedding search within a specific group.

    Workflow:
      1. Call this tool with the entity URI and what you want to express.
      2. Direct/Inherited properties are shown in full — pick one if it matches.
      3. If not found, check the collapsed groups summary for a likely defining class.
      4. Call search_attributes(entity_uri, query, domain_class="<URI>") to search within it.
      5. Call set_property (or attach_component first if the class is a composition type).

    Args:
        entity_uri: URI of the already-created entity.
        query: What the value represents (e.g. 'version number', 'file path').
        value: Optional actual value — used only to show inferred XSD type as a hint.
    """
    subj_uri = rdflib.URIRef(entity_uri)
    types = [str(t) for t in engine.graph.objects(subj_uri, RDF.type)]

    if not types:
        return f"Error: No type defined for entity '{entity_uri}'. Please use create_entity first."

    own_type_set = set(types)
    for t in types:
        own_type_set.update(engine.get_transitive_superclasses(t))

    inferred_type = engine.infer_datatype(value) if value else None
    type_hint = (f"  (Value hint: '{value}' → inferred XSD type: "
                 f"{inferred_type.split('#')[-1] if inferred_type else 'unknown'})") if value else ""

    lines = [
        f"recurse_attributes: Browse DatatypeProperties for {entity_uri}",
        f"Looking for: '{query}'{type_hint}",
        f"Entity types: {[_local_name(t) for t in types]}",
        "",
    ]

    direct_types = set(types)
    ancestor_types = own_type_set - direct_types

    direct_props = []
    inherited_props = []
    other_props = []  # everything else: unconstrained + composition classes

    seen = set()
    # Direct + Inherited (non-facet)
    for t in types:
        for p_uri in engine.get_properties_for_class(t, "DatatypeProperty", include_facets=False):
            if p_uri in seen:
                continue
            seen.add(p_uri)
            info = engine.properties[p_uri]
            domain = set(info.get('domain', []))
            if not domain:
                other_props.append(p_uri)
            elif domain & direct_types:
                direct_props.append(p_uri)
            elif domain & ancestor_types:
                inherited_props.append(p_uri)
            else:
                other_props.append(p_uri)

    # Composition/facet properties
    for t in types:
        for p_uri in engine.get_properties_for_class(t, "DatatypeProperty", include_facets=True):
            if p_uri in seen:
                continue
            seen.add(p_uri)
            other_props.append(p_uri)

    def _sort(lst):
        return sorted(lst, key=lambda u: engine.properties.get(u, {}).get('name', u))

    lines.extend(_section("Direct",    _sort(direct_props),    "domain directly includes this entity's class", RECURSE_SECTION_CAP))
    lines.extend(_section("Inherited", _sort(inherited_props), "domain includes an ancestor class",            RECURSE_SECTION_CAP))

    if other_props:
        if len(other_props) <= RECURSE_SECTION_CAP:
            lines.extend(_section("Other", _sort(other_props), "unconstrained or via composition class", RECURSE_SECTION_CAP))
        else:
            lines.append(f"[Other Properties — collapsed]")
            search_args = f'"{entity_uri}", "{query}"' + (f', value="{value}"' if value else '')
            lines.extend(_domain_class_summary(other_props, "search_attributes", search_args))

    if not direct_props and not inherited_props and not other_props:
        lines.append("No DatatypeProperties found for this entity's types.")
        lines.append("Consider setting rdfs:label or rdfs:comment as a fallback.")

    lines.append("→ Pick a property URI and call set_property(entity_uri, property_uri, value).")
    lines.append("  If the defining class requires composition, call attach_component first.")
    return "\n".join(lines)

@mcp.tool()
def search_relations(subject_uri: str, object_uri: str, query: str, context: str | None = None, domain_class: str | None = None) -> str:
    """
    Recommends the best ObjectProperty to link two entities using embedding-based search.
    Call this for EVERY relationship between entities after create_entity.
    Discovers direct, composition-bridged, and inverse relations automatically.

    If the top result score is LOW CONFIDENCE, use recurse_relations for manual inspection.
    recurse_relations may show collapsed property groups — use domain_class to search within one.

    - subject_uri: URI of the source entity (already created).
    - object_uri: URI of the target entity (already created).
    - query: The relationship intent (e.g. 'installed by', 'targets', 'patches', 'discovered').
    - context: The original source sentence for better semantic matching.
    - domain_class: Optional URI of a defining class to restrict the candidate pool.
                    Use when recurse_relations shows a collapsed group and you want to search
                    within a specific class's properties (e.g. domain_class="https://.../FileFacet").
                    Omit to search all properties (including those with no domain constraint).
    """
    s_uri = rdflib.URIRef(subject_uri)
    o_uri = rdflib.URIRef(object_uri)

    _OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
    s_types = [str(t) for t in engine.graph.objects(s_uri, RDF.type)] or [_OWL_THING]
    o_types = [str(t) for t in engine.graph.objects(o_uri, RDF.type)] or [_OWL_THING]

    # 1. Candidate discovery (direct, composition-bridged, inverse)
    candidates = engine.get_candidate_relations(s_types, o_types)

    if not candidates:
        return (f"No ObjectProperties found between {s_types} and {o_types}.\n"
                f"Try recurse_relations(\"{subject_uri}\", \"{object_uri}\", \"{query}\") "
                f"to browse all available ObjectProperties regardless of domain/range.")

    # 1b. Scope to a specific defining class if requested
    if domain_class:
        scoped = [c for c in candidates if domain_class in engine.properties.get(c['uri'], {}).get('domain', [])]
        if scoped:
            candidates = scoped
        else:
            return (f"No ObjectProperties found with domain_class='{domain_class}' for these entity types.\n"
                    f"Check the class URI from recurse_relations output, or omit domain_class to search all.")

    # 2. Query reformulation: anchor the verb in subject/object type context.
    s_names = [engine.classes.get(t, {}).get('name', _local_name(t)) for t in s_types]
    o_names = [engine.classes.get(t, {}).get('name', _local_name(t)) for t in o_types]
    type_context = " ".join(s_names[:2] + o_names[:2])
    combined_query = f"{query} {type_context}"
    if context:
        combined_query = f"{combined_query} {context}"

    # 3. Semantic similarity ranking
    candidate_uris = [c['uri'] for c in candidates]
    name_embs    = [engine.properties[u]['name_embedding']    for u in candidate_uris]
    comment_embs = [engine.properties[u]['comment_embedding'] for u in candidate_uris]
    sim_scores = engine.weighted_similarity(combined_query, name_embs, comment_embs, weights=_EMB_WEIGHTS)

    # Pre-compute transitive object types for range matching
    all_o_types = set(o_types)
    for ot in o_types:
        all_o_types.update(engine.get_transitive_superclasses(ot))

    query_lower = query.lower()

    # 4. Apply score adjustments
    final_results = []
    for score, cand in zip(sim_scores, candidates):
        info = engine.properties[cand['uri']]

        if query_lower in info['name'].lower() or query_lower in info['comment'].lower():
            score += KEYWORD_BONUS
        if info['range'] and any(r in all_o_types for r in info['range']):
            score += RANGE_MATCH_BOOST
        if cand['path_type'] == 'direct':
            score += DIRECT_RELATION_BOOST
        elif cand['path_type'] == 'inverse':
            score -= INVERSE_RELATION_PENALTY

        final_results.append((score, cand))

    results = sorted(final_results, key=lambda x: x[0], reverse=True)[:5]
    top_score = results[0][0] if results else 0.0
    low_confidence = top_score < LOW_CONFIDENCE_PROP_THRESHOLD

    s_label = s_names[0] if s_names else subject_uri
    o_label = o_names[0] if o_names else object_uri
    scope_note = f"  [scoped to: {_local_name(domain_class)}]" if domain_class else ""
    res = [f"Recommended relations: {s_label} --[{query}]--> {o_label}{scope_note}"]
    res.append(f"   Query: '{combined_query}'  |  Candidates evaluated: {len(candidates)}")

    if low_confidence:
        res.append(f"[LOW CONFIDENCE — top score: {top_score:.4f}, threshold: {LOW_CONFIDENCE_PROP_THRESHOLD}]")
        res.append(f"Embedding search could not find a strong match for '{query}'.")
        res.append(f"Recommended action: recurse_relations(\"{subject_uri}\", \"{object_uri}\", \"{query}\") "
                   f"to browse property groups.")
        res.append("")
        res.append("Low-confidence candidates (for reference only):")

    res.append("-" * 50)

    for i, (score, cand) in enumerate(results):
        uri = cand['uri']
        info = engine.properties[uri]
        res.append(f"{i+1}. {info['name']} ({uri}) [Score: {score:.4f}]")
        res.append(f"   Description: {info['comment'] or 'No description'}")
        if cand['path_type'] == 'facet':
            bridge_name = _local_name(cand['bridge'])
            res.append(f"   Note: Bridged via [{bridge_name}]. Use attach_component first.")
        elif cand['path_type'] == 'inverse':
            res.append(f"   Note: INVERSE relation (defined as {o_names} -> {s_names}). Check direction.")
        res.append("")

    return "\n".join(res)


@mcp.tool()
def recurse_relations(subject_uri: str, object_uri: str, query: str) -> str:
    """
    LLM-driven ObjectProperty browser. Use when search_relations returns LOW CONFIDENCE
    or returns no candidates.

    Direct, Subject-side, Object-side, and Inverse properties are shown in full.
    Unconstrained properties (no domain/range) and composition-bridged properties are
    collapsed into a per-class summary when large — use search_relations with
    domain_class=<URI> to run a scoped embedding search within a specific group.

    Workflow:
      1. Call this tool with both entity URIs and the relationship you want to express.
      2. Direct/Subject-side/Object-side/Inverse are shown in full — pick one if it matches.
      3. If not found, check the collapsed groups summary for a likely defining class.
      4. Call search_relations(subject_uri, object_uri, query, domain_class="<URI>").
      5. If "composition-bridged", use attach_component first; then set_property.
      6. If "Inverse", flip direction: set_property(object_uri, property_uri, subject_uri).

    Args:
        subject_uri: URI of the source entity.
        object_uri: URI of the target entity.
        query: The relationship you want to express (e.g. 'targets', 'exploits', 'authored by').
    """
    s_uri = rdflib.URIRef(subject_uri)
    o_uri = rdflib.URIRef(object_uri)

    _OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
    s_types = [str(t) for t in engine.graph.objects(s_uri, RDF.type)] or [_OWL_THING]
    o_types = [str(t) for t in engine.graph.objects(o_uri, RDF.type)] or [_OWL_THING]

    # Expand type hierarchies
    all_s_types = set(s_types)
    for t in s_types:
        all_s_types.update(engine.get_transitive_superclasses(t))

    all_o_types = set(o_types)
    for t in o_types:
        all_o_types.update(engine.get_transitive_superclasses(t))

    s_names = [engine.classes.get(t, {}).get('name', _local_name(t)) for t in s_types]
    o_names = [engine.classes.get(t, {}).get('name', _local_name(t)) for t in o_types]
    s_label = s_names[0] if s_names else subject_uri
    o_label = o_names[0] if o_names else object_uri

    lines = [
        f"recurse_relations: Browse ObjectProperties for {s_label} --[{query}]--> {o_label}",
        f"Subject types: {[_local_name(t) for t in s_types]}",
        f"Object types:  {[_local_name(t) for t in o_types]}",
        "",
    ]

    # Categorize all ObjectProperties.
    # Properties with BOTH domain and range empty go to "other" (not "direct"),
    # preventing sparse ontologies from flooding the Direct section.
    direct = []
    subj_side = []
    obj_side = []
    inverse = []
    other = []   # unconstrained + composition-bridged (collapsed when large)

    seen = set()
    for p_uri, info in engine.properties.items():
        if info.get('type') != 'ObjectProperty':
            continue
        if p_uri in seen:
            continue

        p_domain = set(info.get('domain', []))
        p_range = set(info.get('range', []))
        dom_empty = not p_domain
        rng_empty = not p_range

        has_dom_s = bool(p_domain & all_s_types)
        has_rng_o = bool(p_range & all_o_types)
        has_dom_o = bool(p_domain & all_o_types)
        has_rng_s = bool(p_range & all_s_types)

        if dom_empty and rng_empty:
            other.append(p_uri)
        elif (has_dom_s or dom_empty) and (has_rng_o or rng_empty) and (has_dom_s or has_rng_o):
            direct.append(p_uri)
        elif has_dom_s:
            subj_side.append(p_uri)
        elif has_rng_o:
            obj_side.append(p_uri)
        elif has_dom_o and has_rng_s:
            inverse.append(p_uri)

        seen.add(p_uri)

    # Composition-bridged properties
    rel_candidates = engine.get_candidate_relations(s_types, o_types)
    for c in rel_candidates:
        if c['path_type'] == 'facet' and c['uri'] not in seen:
            other.append(c['uri'])
            seen.add(c['uri'])

    def _sort(lst):
        return sorted(lst, key=lambda u: engine.properties.get(u if isinstance(u, str) else u[0], {}).get('name', ''))

    lines.extend(_section("Direct",       _sort(direct),    "at least one side explicitly matches the entity types", RECURSE_SECTION_CAP))
    lines.extend(_section("Subject-side", _sort(subj_side), "domain matches subject; range declared but doesn't match object", RECURSE_SECTION_CAP))
    lines.extend(_section("Object-side",  _sort(obj_side),  "range matches object; domain declared but doesn't match subject", RECURSE_SECTION_CAP))

    if inverse:
        inv_sorted = _sort(inverse)
        lines.append(f"[Inverse]  ({len(inverse)} properties)  — defined in the reverse direction")
        lines.append("  Use set_property(object_uri, property_uri, subject_uri) to apply these.")
        lines.append("")
        for p_uri in inv_sorted[:RECURSE_SECTION_CAP]:
            lines.extend(_fmt_prop(p_uri))
            lines.append("")
        if len(inverse) > RECURSE_SECTION_CAP:
            lines.append(f"  ... {len(inverse) - RECURSE_SECTION_CAP} more not shown.")
            lines.append("")

    if other:
        if len(other) <= RECURSE_SECTION_CAP:
            lines.extend(_section("Other", _sort(other), "unconstrained or via composition class", RECURSE_SECTION_CAP))
        else:
            lines.append(f"[Other Properties — collapsed]")
            search_args = f'"{subject_uri}", "{object_uri}", "{query}"'
            lines.extend(_domain_class_summary(other, "search_relations", search_args))

    if not direct and not subj_side and not obj_side and not inverse and not other:
        lines.append("No ObjectProperties found at all for these type combinations.")
        lines.append("Consider using rdfs:seeAlso or owl:sameAs as a generic fallback link.")

    lines.append("→ Pick a property URI and call set_property(subject_uri, property_uri, object_uri).")
    return "\n".join(lines)

@mcp.tool()
def get_ontology_summary() -> str:
    """
    Returns a high-level summary of the loaded ontology, including namespaces with class counts
    and example class names. Use this first to understand available categories and to identify
    namespace_filter values for search_classes.
    """
    class_count = len(engine.classes)
    prop_count = len(engine.properties)
    shape_count = len(engine.shapes)

    # Group classes by namespace prefix
    ns_map = {}  # namespace_uri -> [local_name, ...]
    for uri, info in engine.classes.items():
        if '#' in uri:
            ns = uri.split('#')[0] + '#'
            local = uri.split('#')[-1]
        elif '/' in uri:
            ns = '/'.join(uri.split('/')[:-1]) + '/'
            local = uri.split('/')[-1]
        else:
            ns = ''
            local = uri
        if ns not in ns_map:
            ns_map[ns] = []
        ns_map[ns].append(local)

    # Build namespace lines sorted by class count descending
    ns_lines = []
    for ns, locals_ in sorted(ns_map.items(), key=lambda x: -len(x[1])):
        examples = ', '.join(sorted(locals_)[:4])
        # Extract a short label from the namespace URI
        short = _local_name(ns.rstrip('/#')) or ns
        ns_lines.append(f"  {short:<20} ({len(locals_):>3} classes)  e.g. {examples}")
        ns_lines.append(f"    filter key: \"{short}\"  |  full prefix: {ns}")

    ns_block = "\n".join(ns_lines[:30])  # cap at 15 namespaces (2 lines each)

    return f"""Ontology Summary
================
  Total Classes   : {class_count}
  Total Properties: {prop_count}
  SHACL Shapes    : {shape_count}

Namespaces (use the filter key in search_classes namespace_filter):
{ns_block}

Next steps:
  - search_classes(query, namespace_filter="<filter key>") to find a class
  - If search score < {LOW_CONFIDENCE_CLASS_THRESHOLD}, add context= or use recurse_classes
"""


@mcp.tool()
def get_class_details(class_uri: str) -> str:
    """Retrieves schema, hierarchy and connectivity guidance for a class."""
    if class_uri not in engine.classes: return "Error: Class not found."
    
    info = engine.classes[class_uri]
    supers = engine.get_transitive_superclasses(class_uri)
    
    res = [f"Class: {info['name']}", f"Comment: {info['comment']}", f"Hierarchy: {' -> '.join(supers[::-1] + [class_uri])}"]
    
    res.append("\n[Properties & Usage Guidance]")
    all_targets = [class_uri] + supers
    for target in all_targets:
        if target in engine.shapes:
            for s in engine.shapes[target]:
                p_info = engine.properties.get(s['path'], {})
                p_type = p_info.get('type', "Property")
                p_range = p_info.get('range', [])
                
                line = f"- {p_info.get('name', s['path'])} ({s['path']})\n  - Type: {p_type}"
                if p_type == "ObjectProperty":
                    line += f" (MUST link to an Entity of type: {', '.join(p_range) or 'Any'})"
                else:
                    line += f" (Literal value)"
                
                if s['minCount']: line += f" | ⚠️ REQUIRED"
                res.append(line)
    
    return "\n".join(res)

# --- [Entity Manipulation Tools] ---

@mcp.tool()
def create_entity(entity_id: str, class_uris: list[str]) -> str:
    """Creates a multi-typed entity. Base classes must exist in the loaded ontology."""
    safe_id = urllib.parse.quote(str(entity_id).replace(" ", "_"))
    entity_uri = rdflib.URIRef(engine.instance_base_uri + safe_id)

    # Detect existing entity
    existing_types = {str(t) for t in engine.graph.objects(entity_uri, RDF.type)}
    is_existing = len(existing_types) > 0

    STANDARD_NS = (
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
    )

    added_uris = []
    skipped_duplicate = []
    invalid_uris = []

    for c_uri in class_uris:
        c_uri_str = str(c_uri)
        if c_uri_str in engine.classes or any(c_uri_str.startswith(ns) for ns in STANDARD_NS):
            if c_uri_str in existing_types:
                skipped_duplicate.append(c_uri_str)
            else:
                engine.graph.add((entity_uri, RDF.type, rdflib.URIRef(c_uri_str)))
                added_uris.append(c_uri_str)
        else:
            invalid_uris.append(c_uri_str)

    lines = []
    if is_existing:
        existing_short = [_local_name(t) for t in existing_types]
        lines.append(f"WARNING: Entity '{entity_uri}' already exists (types: {existing_short}).")
        if added_uris:
            lines.append(f"  Added new types: {added_uris}")
        if skipped_duplicate:
            lines.append(f"  Skipped (already assigned): {skipped_duplicate}")
    else:
        lines.append(f"Created: {entity_uri} (Original ID: {entity_id})")

    if invalid_uris:
        lines.append(f"WARNING: Classes not found in ontology and skipped: {invalid_uris}")

    return "\n".join(lines)

@mcp.tool()
def remove_entity(entity_uri: str) -> str:
    """Deletes an entity, all its associated links, and any attached components (cascade)."""
    subj = rdflib.URIRef(entity_uri)

    # Cascade: find components created by attach_component for this entity.
    # They follow the pattern {entity_uri}_comp_{class}_{uuid}.
    comp_prefix = entity_uri + "_comp_"
    components = [s for s in engine.graph.subjects() if str(s).startswith(comp_prefix)]

    for comp in components:
        engine.graph.remove((comp, None, None))
        engine.graph.remove((None, None, comp))

    engine.graph.remove((subj, None, None))
    engine.graph.remove((None, None, subj))

    msg = f"Success: Removed {entity_uri}"
    if components:
        msg += f" and {len(components)} attached component(s)"
    return msg + "."

@mcp.tool()
def reload_ontology() -> str:
    """Reloads the ontology files from disk and rebuilds the schema cache."""
    engine.load_ontology()
    return "Success: Ontology reloaded and schema cache rebuilt."

@mcp.tool()
def reset_graph() -> str:
    """Removes all instance triples from the graph while preserving the ontology schema and embeddings."""
    to_remove = [
        (s, p, o) for s, p, o in engine.graph
        if str(s).startswith(engine.instance_base_uri)
    ]
    for triple in to_remove:
        engine.graph.remove(triple)
    return f"Success: Removed {len(to_remove)} instance triples. Schema and embeddings preserved."


@mcp.tool()
def set_property(entity_uri: str, property_uri: str, value: str) -> str:

    """
    Sets a property value. The property_uri MUST exist in the loaded ontology
    or be from standard RDF/RDFS/OWL/SHACL namespaces.
    - value: The literal value OR the URI of another entity (must start with 'http').
    """
    if str(property_uri) not in engine.properties:
        # Check standard namespaces as well
        if not any(str(property_uri).startswith(ns) for ns in ["http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#", "http://www.w3.org/2002/07/owl#", "http://www.w3.org/ns/shacl#"]):
             return f"ERROR: The property URI '{property_uri}' does not exist in the current ontology or standard namespaces."

    subj = rdflib.URIRef(entity_uri)
    pred = rdflib.URIRef(property_uri)
    engine.graph.add((subj, pred, _rdf_value(value)))
    return f"Success: {entity_uri} --({property_uri})--> {value}"

@mcp.tool()
def attach_component(entity_uri: str, component_class: str, connection_prop: str, attributes: dict) -> str:
    """
    Groups metadata into a Facet/Component (e.g. FileFacet, NetworkConnectionFacet).
    - attributes: A dictionary of {property_uri: value}.
    - Note: Values starting with 'http' are automatically treated as URIs.
    """
    subj = rdflib.URIRef(entity_uri)
    cls_name = _local_name(component_class)
    comp_uri = rdflib.URIRef(f"{entity_uri}_comp_{cls_name}_{uuid.uuid4().hex[:8]}")
    engine.graph.add((comp_uri, RDF.type, rdflib.URIRef(component_class)))
    engine.graph.add((subj, rdflib.URIRef(connection_prop), comp_uri))
    for p_uri, val in attributes.items():
        engine.graph.add((comp_uri, rdflib.URIRef(p_uri), _rdf_value(val)))
    return f"Success: Attached {component_class} to {entity_uri}. Component URI: {comp_uri}"

@mcp.tool()
def visualize_graph(verbose: bool = False) -> str:
    """
    Returns a text-based (ASCII tree) visualization of the current session's graph.
    - verbose: If True, displays full URIs for classes, properties, and entities.
    """
    if not engine.graph:
        return "The graph is currently empty."

    # Identify instance subjects
    subjects = set()
    for s in engine.graph.subjects():
        if str(s).startswith(engine.instance_base_uri):
            subjects.add(s)

    if not subjects:
        return "No instances found in the graph."

    mode_text = "(Full URIs)" if verbose else "(Names Only)"
    output = [f"### Current Knowledge Graph Preview {mode_text} ###\n"]
    
    # Simple grouping by type or subject
    for subj in sorted(subjects):
        subj_label = str(subj) if verbose else str(subj).replace(engine.instance_base_uri, "")
        types = [str(t) if verbose else _local_name(str(t)) for t in engine.graph.objects(subj, RDF.type)]
        output.append(f"● {subj_label} [{', '.join(types)}]")
        
        # Outgoing properties
        for p, o in engine.graph.predicate_objects(subj):
            if p == RDF.type: continue
            
            p_label = str(p) if verbose else _local_name(str(p))
            if str(o).startswith(engine.instance_base_uri):
                o_label = str(o) if verbose else str(o).replace(engine.instance_base_uri, "")
                output.append(f"  └── {p_label} ➔ {o_label}")
            else:
                o_val = str(o)
                output.append(f"  └── {p_label}: \"{o_val}\"")
        output.append("")

    return "\n".join(output)

@mcp.tool()
def get_graph_data() -> str:
    """Returns the current knowledge graph as a JSON string containing nodes and edges."""
    import json
    nodes_map = {}
    edges = []
    
    # Base URI check
    def is_instance(uri):
        return str(uri).startswith(engine.instance_base_uri)

    # 1. First pass: Collect all instances as subjects
    for s in engine.graph.subjects():
        if is_instance(s):
            s_uri = str(s)
            if s_uri not in nodes_map:
                nodes_map[s_uri] = {
                    "id": s_uri,
                    "label": s_uri.replace(engine.instance_base_uri, ""),
                    "types": []
                }
            
            # Attributes and types
            for p, o in engine.graph.predicate_objects(s):
                p_str = str(p)
                o_str = str(o)
                
                if p == RDF.type:
                    short_type = _local_name(o_str)
                    if short_type not in nodes_map[s_uri]["types"]:
                        nodes_map[s_uri]["types"].append(short_type)
                elif is_instance(o):
                    # It's an edge to another instance
                    edges.append({
                        "from": s_uri,
                        "to": o_str,
                        "label": _local_name(p_str)
                    })
                    # Ensure the target node exists in nodes_map too
                    if o_str not in nodes_map:
                        nodes_map[o_str] = {
                            "id": o_str,
                            "label": o_str.replace(engine.instance_base_uri, ""),
                            "types": []
                        }
                else:
                    # It's an attribute
                    key = _local_name(p_str)
                    # Don't overwrite if multiple, just append or cap
                    nodes_map[s_uri][key] = o_str

    return json.dumps({"nodes": list(nodes_map.values()), "edges": edges}, indent=2)




@mcp.tool()
def prune_islands(min_size: int = 1) -> str:
    """
    Removes disconnected 'islands' (small clusters of entities) from the graph.
    - min_size: Islands with total entity count <= min_size will be removed.
    - Helps clean up semantic noise/side-talk that doesn't connect to the core analysis.
    """
    import networkx as nx
    G = nx.Graph()
    
    def is_instance(uri):
        return str(uri).startswith(engine.instance_base_uri)
        
    instance_nodes = set()
    # Find all instances
    for s, p, o in engine.graph:
        if is_instance(s):
            instance_nodes.add(s)
            if is_instance(o):
                G.add_edge(s, o)
                instance_nodes.add(o)
        elif is_instance(o):
            instance_nodes.add(o)
            
    # Add isolated nodes to the graph
    for node in instance_nodes:
        if node not in G:
            G.add_node(node)
            
    # Identify components to prune
    components = list(nx.connected_components(G))
    to_remove = set()
    for comp in components:
        if len(comp) <= min_size:
            to_remove.update(comp)
            
    # Delete triples associated with pruned entities
    triples_removed = 0
    for entity in to_remove:
        # Outgoing
        for p, o in list(engine.graph.predicate_objects(entity)):
            # If o is a facet/component (we hash them in attach_component), we should follow it?
            # For now, let's keep it simple: any triple where subject or object is a pruned instance
            engine.graph.remove((entity, p, o))
            triples_removed += 1
        # Incoming
        for s, p in list(engine.graph.subject_predicates(entity)):
            engine.graph.remove((s, p, entity))
            triples_removed += 1
            
    return f"Success: Pruned {len(to_remove)} entities across {len([c for c in components if len(c) <= min_size])} islands. Total triples removed: {triples_removed}."


@mcp.tool()
def validate_entity(entity_uri: str) -> str:
    """Validates entity against SHACL constraints. Returns structured JSON."""
    subj = rdflib.URIRef(entity_uri)
    types = [str(t) for t in engine.graph.objects(subj, RDF.type)]
    schema_constraint_count = sum(len(shapes) for shapes in engine.shapes.values())

    if not types:
        result = {
            "valid": False,
            "entity": entity_uri,
            "shacl_status": "failed",
            "constraints_checked": 0,
            "schema_constraint_count": schema_constraint_count,
            "errors": [{
                "type": "no_type",
                "message": "Entity has no rdf:type assigned.",
                "fix": f"Use create_entity('{entity_uri.split('/')[-1]}', ['<class_uri>']) to assign a class."
            }]
        }
        return json.dumps(result, indent=2)

    all_targets = []
    for t in types:
        all_targets.extend([t] + engine.get_transitive_superclasses(t))

    errors = []
    constraints_checked = 0
    for target in set(all_targets):
        if target not in engine.shapes:
            continue
        target_short = _local_name(target)
        for shape in engine.shapes[target]:
            constraints_checked += 1
            values = list(engine.graph.objects(subj, rdflib.URIRef(shape['path'])))
            min_count = int(shape['minCount']) if shape['minCount'] is not None else None
            max_count = int(shape['maxCount']) if shape['maxCount'] is not None else None

            if min_count is not None and len(values) < min_count:
                p_info = engine.properties.get(shape['path'], {})
                p_name = p_info.get('name', _local_name(shape['path']))
                p_type = p_info.get('type', 'Property')
                p_range = p_info.get('range', [])
                range_short = [_local_name(r) for r in p_range]

                if p_type == "ObjectProperty":
                    fix = (f"Use set_property('{entity_uri}', '{shape['path']}', '<target_entity_uri>') "
                           f"to link an entity of type: {range_short or 'Any'}.")
                else:
                    fix = f"Use set_property('{entity_uri}', '{shape['path']}', '<value>') to set the missing literal."

                errors.append({
                    "type": "missing_required_property",
                    "property_uri": shape['path'],
                    "property_name": p_name,
                    "property_type": p_type,
                    "expected_range": range_short,
                    "required_by": target_short,
                    "current_count": len(values),
                    "min_required": min_count,
                    "fix": fix,
                })

            if max_count is not None and len(values) > max_count:
                p_info = engine.properties.get(shape['path'], {})
                p_name = p_info.get('name', _local_name(shape['path']))
                errors.append({
                    "type": "excess_values",
                    "property_uri": shape['path'],
                    "property_name": p_name,
                    "required_by": target_short,
                    "current_count": len(values),
                    "max_allowed": max_count,
                    "fix": f"Remove extra values for '{shape['path']}' until only {max_count} remain.",
                })

    if not errors:
        if schema_constraint_count == 0:
            status = "not_available"
            message = "The selected ontology defines no SHACL shapes; validation was skipped."
        elif constraints_checked == 0:
            status = "not_applicable"
            message = "No SHACL shape applies to this entity's classes."
        else:
            status = "passed"
            message = "All applicable SHACL constraints satisfied."
        return json.dumps({
            "valid": True,
            "entity": entity_uri,
            "shacl_status": status,
            "constraints_checked": constraints_checked,
            "schema_constraint_count": schema_constraint_count,
            "message": message,
        })
    return json.dumps({
        "valid": False,
        "entity": entity_uri,
        "shacl_status": "failed",
        "constraints_checked": constraints_checked,
        "schema_constraint_count": schema_constraint_count,
        "error_count": len(errors),
        "errors": errors,
    }, indent=2)

@mcp.tool()
def export_graph(filename: str = "knowledge_graph.ttl") -> str:
    """Exports all instance data to a Turtle file. Fails if invalid URIs exist."""
    output = rdflib.Graph()
    for s, p, o in engine.graph:
        if str(s).startswith(engine.instance_base_uri):
            output.add((s, p, o))
    
    try:
        output.serialize(destination=filename, format="turtle")
        return f"Successfully exported to {filename}"
    except Exception as e:
        return f"Export Error: {str(e)}. This usually means an invalid URI was created. Consider using `reset_graph` and avoiding special characters in IDs."

if ENABLE_EVAL_TOOLS:
    @mcp.tool()
    def export_eval_json(
        file: str,
        text: str,
        entities: list[dict],
        triplets: list[dict],
    ) -> str:
        """
        [EVALUATION ONLY] Writes a structured evaluation JSON file. Not part of the standard mapping workflow.
        Only call this tool when the user explicitly requests it for evaluation purposes.

        You must supply the entities and triplets yourself based on the current graph (use visualize_graph first).

        Parameters
        ----------
        file     : Output filename, e.g. "analysis.json"
        text     : The original source text that was analyzed.
        entities : List of entity dicts. Each entry must have:
                     - "name"  : display name of the entity (use the name property if set, else the entity ID)
                     - "class" : full class URI, e.g. "http://docs.oasis-open.org/cti/ns/stix#Malware"
                   Include only real-world entities. Omit auxiliary graph nodes (Actions,
                   Relationships, or any reification/bridge nodes) that exist solely to
                   encode relations — those become triplets instead.
        triplets : List of relation dicts. Each entry must have:
                     - "subject"       : display name of the subject entity
                     - "relation"      : a concise, human-readable verb phrase, e.g. "uses", "targets", "part of"
                     - "property"      : full URI of the ontology property used, e.g. "http://...#uses"
                     - "property_type" : "ObjectProperty" or "DatatypeProperty"
                     - "object"        : display name of the object entity (or literal value for DatatypeProperty)
                   Derive these by interpreting Action nodes, Relationship nodes, or any
                   other reification pattern in the graph as subject→relation→object triples.

        Example call
        ------------
        export_eval_json(
            file="report.json",
            text="Symantec observed ...",
            entities=[
                {"name": "3AM Ransomware", "class": "http://docs.oasis-open.org/cti/ns/stix#Malware"},
                {"name": "Symantec Threat Hunter Team", "class": "http://docs.oasis-open.org/cti/ns/stix#Identity"}
            ],
            triplets=[
                {"subject": "Symantec Threat Hunter Team", "relation": "observed",
                 "property": "http://docs.oasis-open.org/cti/ns/stix#sighting_of_ref",
                 "property_type": "ObjectProperty", "object": "3AM Ransomware"}
            ]
        )
        """
        # ── Detect ontology name from path or namespace URIs ────────────────
        ontology_name = "unknown"
        path_lower = engine.ontology_path.lower()
        if "uco" in path_lower:
            ontology_name = "uco"
        elif "stix" in path_lower:
            ontology_name = "stix"
        else:
            for _, namespace in engine.graph.namespace_manager.namespaces():
                ns = str(namespace).lower()
                if "unifiedcyberontology" in ns or "/uco/" in ns:
                    ontology_name = "uco"
                    break
                if "stix" in ns:
                    ontology_name = "stix"
                    break

        def _resolve_property_type(prop_uri: str, fallback: str) -> str:
            """Look up property type from engine; fall back to LLM-supplied value."""
            info = engine.properties.get(prop_uri, {})
            return info.get("type") or fallback or ""

        def _normalize_entity(e: dict) -> dict:
            """Canonical entity: name + full class URI. Drops unknown keys."""
            return {
                "name": e.get("name", ""),
                "class": e.get("class", e.get("type", "")),
            }

        def _normalize_triplet(t: dict) -> dict:
            """Canonical triplet: subject / relation / property (full URI) / property_type / object.

            Accepts 'property' or legacy 'relation_class' as the property URI key.
            property_type is verified against the engine schema when the URI is known.
            """
            prop_uri = t.get("property") or t.get("relation_class") or t.get("property_uri") or ""
            prop_type = _resolve_property_type(prop_uri, t.get("property_type", ""))
            return {
                "subject": t.get("subject", ""),
                "relation": t.get("relation", ""),
                "property": prop_uri,
                "property_type": prop_type,
                "object": t.get("object", ""),
            }

        normalized_entities = [_normalize_entity(e) for e in entities]
        normalized_triplets = [_normalize_triplet(t) for t in triplets]

        result = {
            "file": file,
            "text": text,
            "ontology": ontology_name,
            "extracted_entities": normalized_entities,
            "extracted_triplets": normalized_triplets,
        }
        try:
            with open(file, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            return (
                f"Evaluation JSON exported to '{file}' "
                f"({len(normalized_entities)} entities, {len(normalized_triplets)} triplets)."
            )
        except Exception as e:
            return f"Export Error: {e}"


def run():
    mcp.run()
