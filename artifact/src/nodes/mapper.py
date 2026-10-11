"""
Node 4: Ontology Mapper — LangGraph loop-based implementation

Instead of one monolithic agent that tries to process all entities and relations
in a single prompt (which overwhelms weaker models), the mapper is split into a
LangGraph loop of small, focused nodes:

  mapper_init → resolve_entity (×N) → map_dt_prop (×M) → map_relation (×K) → mapper_finalize

Each mini-node handles exactly ONE item, keeping the LLM task simple enough for
any model (Qwen3, GPT-4o, Claude, etc.) to follow reliably.

Tool subsets per phase:
  resolve_entity : search_classes, recurse_classes, get_class_details, create_entity
  map_dt_prop    : search_attributes, recurse_attributes, set_property
  map_relation   : search_relations, recurse_relations, set_property
"""

import ast
import asyncio
import json
import os
import re
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
from src.state import AnchorState
from src.utils.llm import get_llm

_ENTITY_BASE = "http://example.org/entities/"
_OWL_THING   = "http://www.w3.org/2002/07/owl#Thing"
_RDFS_LABEL  = "http://www.w3.org/2000/01/rdf-schema#label"
_RDFS_SEE_ALSO = "http://www.w3.org/2000/01/rdf-schema#seeAlso"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _local(uri: str) -> str:
    return uri.split("#")[-1].split("/")[-1]


def _filter_tools(tools: list, names: set) -> list:
    return [t for t in tools if getattr(t, "name", "") in names]


def _get_namespace() -> str:
    """Derive namespace_filter from ONTOLOGY_DIR env var (e.g. 'stix', 'uco')."""
    d = os.getenv("ONTOLOGY_DIR", "")
    return os.path.basename(d.rstrip("/\\")) if d else ""


async def _fetch_ontology_summary(tools: list) -> str:
    for t in tools:
        if getattr(t, "name", "") == "get_ontology_summary":
            try:
                result = await t.ainvoke({})
                return str(result.content) if hasattr(result, "content") else str(result)
            except Exception:
                return ""
    return ""


def _get_tool_call_args(messages, tool_name: str) -> dict | None:
    """Return args of the LAST call to tool_name found in messages."""
    result = None
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            if tc.get("name") == tool_name:
                result = tc.get("args", {})
    return result


def _extract_entity_uri(messages) -> str | None:
    """Extract the created entity URI from a create_entity tool response."""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and _ENTITY_BASE in content:
            m = re.search(r'http://example\.org/entities/[^\s\'"` ]+', content)
            if m:
                return m.group(0).rstrip(")`*.")
    return None


def _extract_entity_uri_from_content(content: str) -> str | None:
    """Extract entity URI from a raw create_entity tool response string."""
    if _ENTITY_BASE in content:
        m = re.search(r'http://example\.org/entities/[^\s\'"` ]+', content)
        if m:
            return m.group(0).rstrip(")`*.")
    return None


def _entity_uri_map(state: AnchorState) -> dict[str, str]:
    """Build {entity_name → entity_uri} from entities_resolved."""
    return {e["name"]: e["entity_uri"] for e in state.get("entities_resolved", [])}


# ── Measurement helpers ────────────────────────────────────────────────────────

_SEARCH_TOOL_NAMES  = {"search_classes", "search_attributes", "search_relations"}
_RECURSE_TOOL_NAMES = {"recurse_classes", "recurse_attributes", "recurse_relations"}
_FINAL_TOOL_NAMES   = {"create_entity", "set_property"}


def _parse_search_scores(content: str) -> list[dict]:
    """Extract (name, score) pairs from a search_* tool response string.

    Handles both formats:
      search_classes    → "1. ClassName (Score: 0.7823)"
      search_attributes/relations → "1. propName (uri) [Score: 0.8234]"
    """
    candidates = []
    for m in re.finditer(
        r'^\s*\d+\.\s+(.+?)\s+(?:\(Score:\s*([\d.]+)\)|\[Score:\s*([\d.]+)\])',
        content, re.MULTILINE
    ):
        score_str = m.group(2) or m.group(3)
        if score_str:
            try:
                candidates.append({"name": m.group(1).strip(), "score": float(score_str)})
            except ValueError:
                pass
    return candidates


def _collect_measurements(
    messages: list,
    phase: str,
    query: str,
    failure_reason: str | None = None,
    selected_uri: str | None = None,
    generic_uri: str | None = None,
) -> dict:
    """Build one measurement record from a mapper agent's message history.

    final_tool is the last search_* or recurse_* tool called immediately before
    the final create_entity / set_property call.  A generic URI chosen through
    a completed final tool call is a legitimate semantic fallback.  A value
    injected because the agent timed out, raised, or omitted its final tool call
    is an operational fallback.
    """
    # Map tool_call_id → tool_name so ToolMessages can be identified
    call_id_to_name: dict[str, str] = {}
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            call_id_to_name[tc.get("id", "")] = tc.get("name", "")

    search_called = False
    recurse_called = False
    last_lookup: str | None = None   # "search" | "recurse"
    final_action_called = False
    search_calls: list[dict] = []

    for msg in messages:
        # Determine ordering of lookups → final action from AIMessage tool_calls
        for tc in (getattr(msg, "tool_calls", None) or []):
            name = tc.get("name", "")
            if name in _SEARCH_TOOL_NAMES:
                search_called = True
                last_lookup = "search"
            elif name in _RECURSE_TOOL_NAMES:
                recurse_called = True
                last_lookup = "recurse"
            elif name in _FINAL_TOOL_NAMES:
                final_action_called = True

        # Parse scores from ToolMessage responses to search_* calls
        if isinstance(msg, ToolMessage):
            tool_name = call_id_to_name.get(getattr(msg, "tool_call_id", ""), "")
            if tool_name in _SEARCH_TOOL_NAMES:
                candidates = _parse_search_scores(str(msg.content))
                search_calls.append({"tool": tool_name, "candidates": candidates})

    top_score: float | None = None
    for sc in search_calls:
        for c in sc["candidates"]:
            if top_score is None or c["score"] > top_score:
                top_score = c["score"]

    if failure_reason is None and not final_action_called:
        failure_reason = "missing_tool_call"
    completion_status = failure_reason or "completed"
    semantic_fallback = (
        completion_status == "completed"
        and generic_uri is not None
        and selected_uri == generic_uri
    )
    operational_fallback = completion_status != "completed"
    if completion_status == "timeout":
        final_tool = "timeout"
    elif operational_fallback:
        final_tool = "fallback"
    else:
        final_tool = last_lookup or "direct"

    return {
        "phase": phase, "query": query,
        "search_called": search_called, "recurse_called": recurse_called,
        "final_tool": final_tool,
        "final_action_called": final_action_called,
        "completion_status": completion_status,
        "semantic_fallback": semantic_fallback,
        "operational_fallback": operational_fallback,
        "search_calls": search_calls,
        "top_score": top_score,
    }


# ── Verbose callback ───────────────────────────────────────────────────────────

class _VerboseCallback(BaseCallbackHandler):
    """Prints tool calls and decision highlights to stdout in real time."""

    def __init__(self, label: str = ""):
        super().__init__()
        self._step  = 0
        self._label = label
        self._rel_query:  dict[tuple, str] = {}
        self._attr_query: dict[str, str]   = {}

    @staticmethod
    def _parse(input_str) -> dict:
        try:
            if isinstance(input_str, dict):
                return input_str
            if isinstance(input_str, str):
                try:
                    return json.loads(input_str)
                except Exception:
                    return ast.literal_eval(input_str)
        except Exception:
            pass
        return {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._step += 1
        kw = serialized.get("kwargs", {})
        model = kw.get("model_name") or kw.get("model", "llm")
        lbl = f" ({self._label})" if self._label else ""
        print(f"  [step {self._step}{lbl}] LLM call ({model})", flush=True)

    def on_llm_end(self, response, **kwargs):
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None) or getattr(gen, "text", "")
            tool_calls = getattr(msg, "tool_calls", []) or []
            if tool_calls:
                names = ", ".join(tc.get("name", "?") for tc in tool_calls)
                print(f"  [step {self._step}] → wants tools: [{names}]", flush=True)
            else:
                content = getattr(msg, "content", str(msg))
                preview = str(content)[:120].replace("\n", " ")
                print(f"  [step {self._step}] → final text: {preview!r}", flush=True)
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?")
        inp  = self._parse(input_str)

        if name in ("search_relations", "recurse_relations"):
            s = inp.get("subject_uri", ""); o = inp.get("object_uri", ""); q = inp.get("query", "")
            if s and o and q:
                self._rel_query[(s, o)] = q

        elif name in ("search_attributes", "recurse_attributes"):
            e = inp.get("entity_uri", ""); q = inp.get("query", "")
            if e and q:
                self._attr_query[e] = q

        elif name == "create_entity":
            eid  = inp.get("entity_id", "?")
            uris = inp.get("class_uris", [])
            uri  = uris[0] if uris else "?"
            print(f"  ★ CLASS    : \"{eid}\"  →  {_local(uri)}", flush=True)
            print(f"               ({uri})", flush=True)
            return

        elif name == "set_property":
            subj  = inp.get("entity_uri", "?")
            prop  = inp.get("property_uri", "?")
            value = inp.get("value", "?")
            slbl  = _local(subj).replace("_", " ")
            if str(value).startswith(_ENTITY_BASE):
                vlbl  = _local(value).replace("_", " ")
                orig  = self._rel_query.get((subj, value), "")
                arrow = orig if orig else _local(prop)
                print(f"  ★ RELATION : \"{slbl}\"  --[{arrow}]-->  \"{vlbl}\"", flush=True)
            else:
                orig  = self._attr_query.get(subj, "")
                albl  = orig if orig else _local(prop)
                print(f"  ★ ATTR     : \"{slbl}\"  [{albl}]  =  \"{value}\"", flush=True)
            print(f"               property: {prop}", flush=True)
            return

        inp_str = str(input_str)
        print(f"  [tool→] {name}  {inp_str}", flush=True)

    def on_tool_end(self, output, **kwargs):
        out   = str(output)
        lines = out.strip().splitlines()
        shown = lines[:2]
        for line in shown:
            print(f"  [←tool] {line[:300]}", flush=True)
        remaining = len(lines) - len(shown)
        if remaining > 0:
            print(f"  [←tool] ... (+{remaining} more lines)", flush=True)

    def on_tool_error(self, error, **kwargs):
        print(f"  [tool✗] {error}", flush=True)


# ── Shared agent runner ────────────────────────────────────────────────────────

async def _run_agent(agent, prompt: str, timeout: int, verbose: bool, label: str = "") -> list:
    """Run a mini-agent and return the final message list."""
    callbacks = [_VerboseCallback(label)] if verbose else []
    result = await asyncio.wait_for(
        agent.ainvoke(
            {"messages": [("human", prompt)]},
            config={"recursion_limit": 30, "callbacks": callbacks},
        ),
        timeout=timeout,
    )
    return result.get("messages", [])


def _make_lazy_agent_getter(tools: list):
    """Create and cache a mapper agent only when its node actually executes.

    Graph inspection needs the node topology but no provider or credentials.
    Delaying construction keeps ``main.py --graph`` and Claim 3 independent of
    the user's LLM configuration while preserving one agent per compiled node.
    """
    agent = None

    def get_agent():
        nonlocal agent
        if agent is None:
            llm = get_llm(
                temperature=0,
                max_tokens=int(os.getenv("MAPPER_MAX_TOKENS", 4096)),
                thinking=True,
            )
            agent = create_react_agent(llm, tools)
        return agent

    return get_agent


# ── Node: mapper_init ──────────────────────────────────────────────────────────

def make_mapper_init_node(tools: list):
    """Reset graph, fetch ontology summary, and populate entity/property/relation queues."""

    async def mapper_init_node(state: AnchorState) -> dict:
        entities = state.get("entities", [])
        triplets = state.get("triplets", [])

        # Reset the MCP graph
        for t in tools:
            if getattr(t, "name", "") == "reset_graph":
                await t.ainvoke({})
                break

        # Fetch ontology summary (logged for debugging)
        summary = await _fetch_ontology_summary(tools)
        ns = _get_namespace()
        if summary:
            print(f"[mapper_init] ontology_summary fetched ({len(summary)} chars), namespace={ns!r}")
        else:
            print("[mapper_init] WARNING: could not fetch ontology summary")

        # DatatypeProperty: from triplets flagged as DatatypeProperty
        dt_props = [
            {"entity_name": t["subject"], "hint": t["predicate"], "value": t["object"]}
            for t in triplets
            if t.get("property_type") == "DatatypeProperty"
        ]
        # Also include literal properties stored directly on entities (legacy / pass1 properties)
        dt_props += [
            {"entity_name": e["name"], "hint": hint, "value": str(val)}
            for e in entities
            for hint, val in e.get("properties", {}).items()
        ]

        # ObjectProperty: from triplets flagged as ObjectProperty (or unlabeled, for backward compat)
        relations = [
            {
                "subject":   t["subject"],
                "predicate": t["predicate"],
                "object":    t["object"],
                "evidence":  t.get("evidence", ""),
            }
            for t in triplets
            if t.get("property_type", "ObjectProperty") == "ObjectProperty"
        ]

        print(f"[mapper_init] queues: {len(entities)} entities, "
              f"{len(dt_props)} dt_props, {len(relations)} relations")

        return {
            "entities_pending":  list(entities),
            "dt_props_pending":  dt_props,
            "relations_pending": relations,
        }

    return mapper_init_node


# ── Node: resolve_entity ───────────────────────────────────────────────────────

def _mapper_tool_mode(phase: str = "both") -> str:
    """Returns 'search', 'recurse', or 'both' for the given phase.

    phase='class'  → checks ANCHOR_MAPPER_TOOLS_CLASS, falls back to ANCHOR_MAPPER_TOOLS
    phase='prop'   → checks ANCHOR_MAPPER_TOOLS_PROP,  falls back to ANCHOR_MAPPER_TOOLS
    """
    fallback = os.getenv("ANCHOR_MAPPER_TOOLS", "both").lower().strip()
    if phase == "class":
        return os.getenv("ANCHOR_MAPPER_TOOLS_CLASS", fallback).lower().strip()
    if phase == "prop":
        return os.getenv("ANCHOR_MAPPER_TOOLS_PROP", fallback).lower().strip()
    return fallback


def _max_tool_calls() -> int:
    return int(os.getenv("ANCHOR_MAX_TOOL_CALLS", "4"))


def make_resolve_entity_node(tools: list):
    """Pop ONE entity from entities_pending, classify it, and create it in the graph."""
    mode = _mapper_tool_mode("class")
    if mode == "search":
        entity_tool_names = {"search_classes", "get_class_details", "create_entity"}
    elif mode == "recurse":
        entity_tool_names = {"recurse_classes", "get_class_details", "create_entity"}
    else:
        entity_tool_names = {"search_classes", "recurse_classes", "get_class_details", "create_entity"}
    entity_tools = _filter_tools(tools, entity_tool_names)
    create_tool = next((t for t in tools if getattr(t, "name", "") == "create_entity"), None)
    ns      = _get_namespace()
    ns_arg  = f', namespace_filter="{ns}"' if ns else ""
    timeout = int(os.getenv("MAPPER_TIMEOUT", 120))
    verbose = os.getenv("ANCHOR_VERBOSE", "").lower() in ("1", "true")
    get_agent = _make_lazy_agent_getter(entity_tools)

    async def resolve_entity_node(state: AnchorState) -> dict:
        pending = list(state.get("entities_pending", []))
        if not pending:
            return {}

        entity = pending.pop(0)
        name       = entity["name"]
        type_hint  = entity.get("type_hint", name)

        prompt = f"""\
Classify this ONE entity and create it in the ontology graph.

Entity name : "{name}"
Type hint   : "{type_hint}"

Step 1 — Find the best class:
  Call search_classes("{type_hint}"{ns_arg})
  • score >= 0.45 → use that class URI (from line "   URI: http://...")
  • score < 0.45 or LOW CONFIDENCE → navigate the hierarchy:
      a. Call recurse_classes("{type_hint}")           ← root level, NO class_uri arg
      b. Root branches are ABSTRACT containers — you MUST go deeper.
         Pick the branch whose description best fits "{type_hint}".
      c. Call recurse_classes("{type_hint}", "<chosen_uri>")
      d. Repeat (c) until you reach a leaf class or a clearly matching class.
      e. Call get_class_details("<uri>") to confirm if unsure.
      f. If nothing fits → use "{_OWL_THING}"
  • No match at all → use "{_OWL_THING}"

Step 2 — Create the entity:
  Call create_entity("{name}", ["<chosen_class_uri>"])
  You MUST call create_entity. There is no valid reason to skip it.

CRITICAL: Use class URIs EXACTLY as they appear in tool output (lines starting with "URI:").
  NEVER construct or guess a URI — only copy verbatim from the tool response.
"""

        print(f"[resolve_entity] '{name}'  hint='{type_hint}'")
        failure_reason = None
        try:
            messages = await _run_agent(get_agent(), prompt, timeout, verbose,
                                        label=f"entity:{name}")
        except asyncio.TimeoutError:
            print(f"[resolve_entity] TIMEOUT for '{name}'")
            messages = []
            failure_reason = "timeout"
        except Exception as ex:
            print(f"[resolve_entity] ERROR for '{name}': {ex}")
            messages = []
            failure_reason = "exception"

        # Extract results from message history
        create_args = _get_tool_call_args(messages, "create_entity")
        class_uri   = (create_args.get("class_uris") or [_OWL_THING])[0] if create_args else _OWL_THING
        entity_uri  = _extract_entity_uri(messages) or (
            _ENTITY_BASE + re.sub(r"[^\w]", "_", name)
        )

        # Fallback: if agent never called create_entity (timeout/error/skip),
        # register the entity directly so map_relation can reference it later.
        if not create_args and create_tool:
            print(f"[resolve_entity] FALLBACK create_entity for '{name}' (agent did not call it)")
            try:
                fb = await create_tool.ainvoke({"entity_id": name, "class_uris": [class_uri]})
                fb_uri = _extract_entity_uri_from_content(str(fb))
                if fb_uri:
                    entity_uri = fb_uri
            except Exception as fb_ex:
                print(f"[resolve_entity] FALLBACK failed for '{name}': {fb_ex}")

        resolved = {"name": name, "entity_uri": entity_uri, "class_uri": class_uri}
        print(f"[resolve_entity] ✓ '{name}' → {_local(class_uri)}  ({entity_uri})")

        measurement = _collect_measurements(
            messages, "entity", f"{name} [{type_hint}]",
            failure_reason=failure_reason,
            selected_uri=class_uri,
            generic_uri=_OWL_THING,
        )
        return {
            "entities_pending":  pending,
            "entities_resolved": [resolved],
            "measurements":      [measurement],
        }

    return resolve_entity_node


# ── Node: map_dt_prop ──────────────────────────────────────────────────────────

def make_map_dt_prop_node(tools: list):
    """Pop ONE DatatypeProperty candidate, search for the best property, and set it."""
    mode = _mapper_tool_mode("prop")
    if mode == "search":
        dt_tool_names = {"search_attributes", "set_property"}
    elif mode == "recurse":
        dt_tool_names = {"recurse_attributes", "set_property"}
    else:
        dt_tool_names = {"search_attributes", "recurse_attributes", "set_property"}
    dt_tools = _filter_tools(tools, dt_tool_names)
    timeout = int(os.getenv("MAPPER_TIMEOUT", 120))
    verbose = os.getenv("ANCHOR_VERBOSE", "").lower() in ("1", "true")
    get_agent = _make_lazy_agent_getter(dt_tools)

    async def map_dt_prop_node(state: AnchorState) -> dict:
        pending = list(state.get("dt_props_pending", []))
        if not pending:
            return {}

        cand       = pending.pop(0)
        entity_name = cand["entity_name"]
        hint        = cand["hint"]
        value       = cand["value"]
        uri_map     = _entity_uri_map(state)
        entity_uri  = uri_map.get(entity_name, "")

        if not entity_uri:
            print(f"[map_dt_prop] SKIP '{entity_name}'.{hint} — entity URI not found")
            return {
                "dt_props_pending": pending,
                "dt_props_resolved": [{"entity_name": entity_name, "hint": hint,
                                       "property_uri": _RDFS_LABEL, "value": value}],
                "measurements": [_collect_measurements(
                    [], "dt_prop", f"{entity_name}.{hint}",
                    failure_reason="missing_entity_uri",
                    selected_uri=_RDFS_LABEL,
                    generic_uri=_RDFS_LABEL,
                )],
            }

        prompt = f"""\
Set ONE DatatypeProperty (literal value) on this entity.

Entity : "{entity_name}"
URI    : {entity_uri}
Hint   : "{hint}"
Value  : "{value}"

Follow these steps IN ORDER. Stop and call set_property as soon as you have a property URI.
Maximum tool calls: {_max_tool_calls()} total (including set_property).

Step 1 — Embedding search:
  Call search_attributes("{entity_uri}", "{hint}", value="{value}")
  • score >= 0.45 → set_property → DONE
  • LOW CONFIDENCE → Step 2

Step 2 — Browse properties:
  Call recurse_attributes("{entity_uri}", "{hint}")
  Read the output:
  • [Direct]/[Inherited] sections: scan each property name against "{hint}".
    If any property semantically fits → set_property with it → DONE.
    An approximate match is always preferred over the fallback.
  • [Collapsed: <class_uri>] line: note ONLY THE FIRST such URI for Step 3.
  • If no match AND there was a [Collapsed: <URI>] line → Step 3.
  • If no match AND no [Collapsed] line → go to Fallback.

Step 3 — Scoped search (only if Step 2 found a [Collapsed: <URI>] line):
  Use the FIRST [Collapsed: <URI>] value from Step 2 as domain_class.
  Do NOT use the entity's own class URI as domain_class.
  Call search_attributes("{entity_uri}", "{hint}", value="{value}",
                         domain_class="<first Collapsed URI from Step 2>")
  • score >= 0.45 → set_property → DONE
  • No match → Fallback

Fallback — call set_property with the best available URI:
  1. Best semantically-related property from Step 2's [Direct]/[Inherited] section (if any listed)
  2. Last resort: "{_RDFS_LABEL}"

CRITICAL: You MUST use property URIs EXACTLY as they appear in tool output (lines starting with "URI:").
  NEVER construct or guess a URI — only copy verbatim from the tool response.
You MUST call set_property(entity_uri="{entity_uri}", property_uri="<URI>", value="{value}").
"""

        print(f"[map_dt_prop] '{entity_name}'.'{hint}' = '{value}'")
        failure_reason = None
        try:
            messages = await _run_agent(get_agent(), prompt, timeout, verbose,
                                        label=f"attr:{entity_name}")
        except asyncio.TimeoutError:
            print(f"[map_dt_prop] TIMEOUT for '{entity_name}'.'{hint}'")
            messages = []
            failure_reason = "timeout"
        except Exception as ex:
            print(f"[map_dt_prop] ERROR: {ex}")
            messages = []
            failure_reason = "exception"

        sp_args      = _get_tool_call_args(messages, "set_property")
        property_uri = (sp_args.get("property_uri") if sp_args else None) or _RDFS_LABEL
        print(f"[map_dt_prop] ✓ '{entity_name}'.'{hint}' → {_local(property_uri)}")

        measurement = _collect_measurements(
            messages, "dt_prop", f"{entity_name}.{hint}",
            failure_reason=failure_reason,
            selected_uri=property_uri,
            generic_uri=_RDFS_LABEL,
        )
        return {
            "dt_props_pending":  pending,
            "dt_props_resolved": [{"entity_name": entity_name, "hint": hint,
                                   "property_uri": property_uri, "value": value}],
            "measurements":      [measurement],
        }

    return map_dt_prop_node


# ── Node: map_relation ─────────────────────────────────────────────────────────

def make_map_relation_node(tools: list):
    """Pop ONE ObjectProperty candidate, search for the best relation, and set it."""
    mode = _mapper_tool_mode("prop")
    if mode == "search":
        rel_tool_names = {"search_relations", "set_property"}
    elif mode == "recurse":
        rel_tool_names = {"recurse_relations", "set_property"}
    else:
        rel_tool_names = {"search_relations", "recurse_relations", "set_property"}
    rel_tools = _filter_tools(tools, rel_tool_names)
    create_tool = next((t for t in tools if getattr(t, "name", "") == "create_entity"), None)
    timeout = int(os.getenv("MAPPER_TIMEOUT", 120))
    verbose = os.getenv("ANCHOR_VERBOSE", "").lower() in ("1", "true")
    get_agent = _make_lazy_agent_getter(rel_tools)

    async def map_relation_node(state: AnchorState) -> dict:
        pending = list(state.get("relations_pending", []))
        if not pending:
            return {}

        rel       = pending.pop(0)
        subject   = rel["subject"]
        predicate = rel["predicate"]
        obj       = rel["object"]
        evidence  = rel.get("evidence", "")
        uri_map   = _entity_uri_map(state)
        subj_uri  = uri_map.get(subject, "")
        obj_uri   = uri_map.get(obj, "")

        if not subj_uri or not obj_uri:
            missing = subject if not subj_uri else obj
            print(f"[map_relation] SKIP '{subject}'→'{obj}' — URI not found for '{missing}'")
            return {
                "relations_pending":  pending,
                "relations_resolved": [{"subject": subject, "predicate": predicate,
                                        "property_uri": _RDFS_SEE_ALSO,
                                        "property_type": "ObjectProperty", "object": obj}],
                "measurements": [_collect_measurements(
                    [], "relation", f"{subject} --[{predicate}]--> {obj}",
                    failure_reason="missing_entity_uri",
                    selected_uri=_RDFS_SEE_ALSO,
                    generic_uri=_RDFS_SEE_ALSO,
                )],
            }

        # ── Pre-check: ensure both entities are registered in the MCP graph ──────
        # search_relations / recurse_relations require both entities to have types.
        # If resolve_entity timed out or the entity was never in entities_pending,
        # the URI may exist in uri_map but not in the graph → force-create with
        # the resolved class (or owl:Thing as fallback) before the agent runs.
        if create_tool:
            class_map = {e["name"]: e["class_uri"] for e in state.get("entities_resolved", [])}
            for ent_name, ent_uri in ((subject, subj_uri), (obj, obj_uri)):
                class_uri = class_map.get(ent_name, _OWL_THING)
                try:
                    await create_tool.ainvoke({"entity_id": ent_name, "class_uris": [class_uri]})
                except Exception:
                    pass  # already exists or unrecoverable — proceed anyway

        prompt = f"""\
Set ONE ObjectProperty (entity → entity link) in the graph.

Subject  : "{subject}" (URI: {subj_uri})
Object   : "{obj}"     (URI: {obj_uri})
Predicate: "{predicate}"
Evidence : "{evidence}"

Follow these steps IN ORDER. Stop and call set_property as soon as you have a property URI.
Maximum tool calls: {_max_tool_calls()} total (including set_property).

Step 1 — Embedding search:
  Call search_relations("{subj_uri}", "{obj_uri}", "{predicate}", context="{evidence}")
  • score >= 0.45 → set_property → DONE
  • LOW CONFIDENCE → Step 2

Step 2 — Browse properties:
  Call recurse_relations("{subj_uri}", "{obj_uri}", "{predicate}")
  Read the output:
  • [Direct] section: scan each property name against "{predicate}".
    If any property semantically fits → set_property with it → DONE.
    An approximate match from [Direct] is always preferred over the fallback.
  • [Inverse] section: if a property fits in reverse direction →
    set_property("{obj_uri}", "<URI>", "{subj_uri}") → DONE.
  • [Collapsed: <class_uri>] line: note ONLY THE FIRST such URI for Step 3.
  • If [Direct]/[Inverse] had no match AND there was a [Collapsed: <URI>] line → Step 3.
  • If [Direct]/[Inverse] had no match AND no [Collapsed] line → go to Fallback.

Step 3 — Scoped search (only if Step 2 found a [Collapsed: <URI>] line):
  Use the FIRST [Collapsed: <URI>] value from Step 2 output as domain_class.
  Do NOT use the subject's or object's own class URI ({subj_uri}, {obj_uri}) as domain_class.
  Call search_relations("{subj_uri}", "{obj_uri}", "{predicate}",
                        context="{evidence}", domain_class="<first Collapsed URI from Step 2>")
  • score >= 0.45 → set_property → DONE
  • No match → Fallback

Fallback — call set_property with the best available URI:
  1. Best semantically-related property from Step 2's [Direct] section (if any listed)
  2. Last resort: "{_RDFS_SEE_ALSO}"

CRITICAL: You MUST use property URIs EXACTLY as they appear in tool output (lines starting with "URI:").
  NEVER construct or guess a URI — only copy verbatim from the tool response.
You MUST call set_property exactly once. Use subject URI and object URI exactly as given:
  set_property("{subj_uri}", "<property_uri>", "{obj_uri}")
"""

        print(f"[map_relation] '{subject}' --[{predicate}]--> '{obj}'")
        failure_reason = None
        try:
            messages = await _run_agent(get_agent(), prompt, timeout, verbose,
                                        label=f"rel:{predicate}")
        except asyncio.TimeoutError:
            print(f"[map_relation] TIMEOUT for '{subject}'→'{obj}'")
            messages = []
            failure_reason = "timeout"
        except Exception as ex:
            print(f"[map_relation] ERROR: {ex}")
            messages = []
            failure_reason = "exception"

        sp_args      = _get_tool_call_args(messages, "set_property")
        property_uri = (sp_args.get("property_uri") if sp_args else None) or _RDFS_SEE_ALSO
        print(f"[map_relation] ✓ '{subject}' --[{_local(property_uri)}]--> '{obj}'")

        measurement = _collect_measurements(
            messages, "relation",
            f"{subject} --[{predicate}]--> {obj}",
            failure_reason=failure_reason,
            selected_uri=property_uri,
            generic_uri=_RDFS_SEE_ALSO,
        )
        return {
            "relations_pending":  pending,
            "relations_resolved": [{"subject": subject, "predicate": predicate,
                                    "property_uri": property_uri,
                                    "property_type": "ObjectProperty", "object": obj}],
            "measurements":       [measurement],
        }

    return map_relation_node


# ── Node: mapper_finalize ──────────────────────────────────────────────────────

async def mapper_finalize_node(state: AnchorState) -> dict:
    """Assemble mapped_entities and mapped_triplets from resolved queues."""
    mapped_entities = [
        {"name": e["name"], "class": e["class_uri"]}
        for e in state.get("entities_resolved", [])
    ]

    mapped_triplets = []
    for r in state.get("dt_props_resolved", []):
        mapped_triplets.append({
            "subject":       r["entity_name"],
            "relation":      r["hint"],
            "property":      r["property_uri"],
            "property_type": "DatatypeProperty",
            "object":        r["value"],
        })
    for r in state.get("relations_resolved", []):
        mapped_triplets.append({
            "subject":       r["subject"],
            "relation":      r["predicate"],
            "property":      r["property_uri"],
            "property_type": r.get("property_type", "ObjectProperty"),
            "object":        r["object"],
        })

    print(f"[mapper_finalize] {len(mapped_entities)} entities, "
          f"{len(mapped_triplets)} triplets assembled")

    return {
        "mapped_entities": mapped_entities,
        "mapped_triplets": mapped_triplets,
    }


# ── Node 4b: SHACL Correction ──────────────────────────────────────────────────

_SHACL_MAX_RETRIES = 3


def make_shacl_correction_node(tools: list):
    """
    After mapper_finalize, validate each entity against SHACL constraints and
    attempt to fix violations via LLM agent (max 3 retries per entity).
    Entities with unresolved violations are flagged in validation_warnings.
    Only active in MCP mode (requires validate_entity tool).
    """
    validate_tool = next((t for t in tools if getattr(t, "name", "") == "validate_entity"), None)
    if not validate_tool:
        async def noop(state: AnchorState) -> dict:
            return {}
        return noop

    fix_tool_names = {
        "validate_entity", "set_property",
        "search_attributes", "recurse_attributes",
        "search_relations",  "recurse_relations",
    }
    fix_tools = _filter_tools(tools, fix_tool_names)
    timeout = int(os.getenv("MAPPER_TIMEOUT", 120))
    verbose = os.getenv("ANCHOR_VERBOSE", "").lower() in ("1", "true")
    get_agent = _make_lazy_agent_getter(fix_tools)

    async def shacl_correction_node(state: AnchorState) -> dict:
        entities_resolved = state.get("entities_resolved", [])
        if not entities_resolved:
            return {"validation_warnings": []}

        warnings = []
        passed = 0
        not_applicable = 0

        for entity in entities_resolved:
            entity_uri = entity.get("entity_uri", "")
            name       = entity.get("name", "")
            if not entity_uri:
                continue

            resolved = False
            for attempt in range(_SHACL_MAX_RETRIES):
                # Direct tool call — no agent overhead for the check step
                try:
                    raw = await validate_tool.ainvoke({"entity_uri": entity_uri})
                    result = json.loads(raw if isinstance(raw, str) else str(raw))
                except Exception as exc:
                    print(f"[shacl_correction] validate_entity failed for '{name}': {exc}")
                    break

                if result.get("valid"):
                    status = result.get("shacl_status", "passed")
                    if status == "not_available":
                        print("[shacl_correction] skipped: selected ontology contains no SHACL shapes")
                        return {"validation_warnings": []}
                    if status == "not_applicable":
                        not_applicable += 1
                        print(f"[shacl_correction] - '{name}': no applicable SHACL shape")
                    else:
                        passed += 1
                        print(f"[shacl_correction] ✓ '{name}' passed")
                    resolved = True
                    break

                errors = result.get("errors", [])
                if not errors:
                    resolved = True
                    break

                print(f"[shacl_correction] '{name}': {len(errors)} violation(s) — "
                      f"attempt {attempt + 1}/{_SHACL_MAX_RETRIES}")

                violation_text = json.dumps(errors, indent=2)
                prompt = f"""\
Fix SHACL constraint violations for entity "{name}" ({entity_uri}).

Violations detected:
{violation_text}

For each missing_required_property violation:
  1. Use search_attributes or search_relations to find the correct property URI.
  2. Use set_property("{entity_uri}", "<property_uri>", "<value>") to add the missing value.

After fixing, call validate_entity("{entity_uri}") to confirm all violations are resolved.
Fix every violation listed above before finishing.
"""
                try:
                    await _run_agent(get_agent(), prompt, timeout, verbose,
                                     label=f"shacl/{name}/attempt{attempt + 1}")
                except Exception as exc:
                    print(f"[shacl_correction] ERROR '{name}' attempt {attempt + 1}: {exc}")
                    break

            if not resolved:
                # Final check after last retry
                try:
                    raw = await validate_tool.ainvoke({"entity_uri": entity_uri})
                    final = json.loads(raw if isinstance(raw, str) else str(raw))
                    if not final.get("valid"):
                        print(f"[shacl_correction] ✗ '{name}': "
                              f"unresolved after {_SHACL_MAX_RETRIES} retries — flagged")
                        warnings.append({
                            "entity": name,
                            "uri":    entity_uri,
                            "errors": final.get("errors", []),
                        })
                except Exception:
                    pass

        total = len(entities_resolved)
        flagged = len(warnings)
        print(f"[shacl_correction] {passed}/{total} entities passed applicable SHACL constraints; "
              f"{not_applicable} had no applicable shape; {flagged} flagged")
        return {"validation_warnings": warnings}

    return shacl_correction_node


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_init(state: AnchorState) -> str:
    return "resolve_entity" if state.get("entities_pending") else "map_dt_prop"

def route_entity(state: AnchorState) -> str:
    return "resolve_entity" if state.get("entities_pending") else "map_dt_prop"

def route_dt_prop(state: AnchorState) -> str:
    return "map_dt_prop" if state.get("dt_props_pending") else "map_relation"

def route_relation(state: AnchorState) -> str:
    return "map_relation" if state.get("relations_pending") else "mapper_finalize"
