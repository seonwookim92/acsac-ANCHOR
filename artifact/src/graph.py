"""
LangGraph StateGraph definition for the ANCHOR text-to-ontology pipeline.

Graph topology:
  START → chunker → extractor → preprocessor
        → mapper_init
        → resolve_entity (loop until entities_pending empty)
        → map_dt_prop    (loop until dt_props_pending empty)
        → map_relation   (loop until relations_pending empty)
        → mapper_finalize
        → exporter → END

The mapper phase is a LangGraph loop: each node handles exactly ONE item from a
queue, then routes back to itself or forwards to the next phase.
"""

import time
from langgraph.graph import StateGraph, START, END
from src.state import AnchorState
from src.nodes.chunker import chunker_node
from src.nodes.extractor import extractor_node
from src.nodes.preprocessor import preprocessor_node
from src.nodes.mapper import (
    make_mapper_init_node,
    make_resolve_entity_node,
    make_map_dt_prop_node,
    make_map_relation_node,
    mapper_finalize_node,
    make_shacl_correction_node,
    route_after_init,
    route_entity,
    route_dt_prop,
    route_relation,
)
from src.nodes.exporter import make_exporter_node

# Shared dict written to by _timed wrappers; read by main.py after ainvoke.
node_elapsed: dict[str, float] = {}


def _timed(name: str, fn):
    """Wrap an async node to record its wall-clock elapsed time."""
    async def wrapper(state):
        t0 = time.perf_counter()
        result = await fn(state)
        node_elapsed[name] = time.perf_counter() - t0
        return result
    wrapper.__name__ = name
    return wrapper


def _timed_accum(name: str, fn):
    """Wrap a looping node to accumulate elapsed time across iterations."""
    async def wrapper(state):
        t0 = time.perf_counter()
        result = await fn(state)
        node_elapsed[name] = node_elapsed.get(name, 0.0) + (time.perf_counter() - t0)
        return result
    wrapper.__name__ = name
    return wrapper


def build_graph(tools: list):
    """
    Build and compile the pipeline graph.

    Args:
        tools: MCP tool objects from MultiServerMCPClient.get_tools()

    Returns:
        Compiled LangGraph graph ready for ainvoke / invoke.
    """
    node_elapsed.clear()

    mapper_init_node      = make_mapper_init_node(tools)
    resolve_entity_node   = make_resolve_entity_node(tools)
    map_dt_prop_node      = make_map_dt_prop_node(tools)
    map_relation_node     = make_map_relation_node(tools)
    shacl_correction_node = make_shacl_correction_node(tools)
    exporter_node         = make_exporter_node(tools)

    graph = StateGraph(AnchorState)

    # ── Pre-mapper nodes ───────────────────────────────────────────────────────
    graph.add_node("chunker",      _timed("chunker",      chunker_node))
    graph.add_node("extractor",    _timed("extractor",    extractor_node))
    graph.add_node("preprocessor", _timed("preprocessor", preprocessor_node))

    # ── Mapper nodes (loop-based) ──────────────────────────────────────────────
    graph.add_node("mapper_init",       _timed("mapper_init",          mapper_init_node))
    graph.add_node("resolve_entity",    _timed_accum("resolve_entity", resolve_entity_node))
    graph.add_node("map_dt_prop",       _timed_accum("map_dt_prop",    map_dt_prop_node))
    graph.add_node("map_relation",      _timed_accum("map_relation",   map_relation_node))
    graph.add_node("mapper_finalize",   _timed("mapper_finalize",      mapper_finalize_node))
    graph.add_node("shacl_correction",  _timed("shacl_correction",     shacl_correction_node))

    # ── Post-mapper node ───────────────────────────────────────────────────────
    graph.add_node("exporter", _timed("exporter", exporter_node))

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "chunker")
    graph.add_edge("chunker",      "extractor")
    graph.add_edge("extractor",    "preprocessor")
    graph.add_edge("preprocessor", "mapper_init")

    # mapper_init → first item or skip phases if queues are empty
    graph.add_conditional_edges("mapper_init", route_after_init,
                                 ["resolve_entity", "map_dt_prop"])

    # resolve_entity loop
    graph.add_conditional_edges("resolve_entity", route_entity,
                                 ["resolve_entity", "map_dt_prop"])

    # map_dt_prop loop
    graph.add_conditional_edges("map_dt_prop", route_dt_prop,
                                 ["map_dt_prop", "map_relation"])

    # map_relation loop
    graph.add_conditional_edges("map_relation", route_relation,
                                 ["map_relation", "mapper_finalize"])

    graph.add_edge("mapper_finalize",  "shacl_correction")
    graph.add_edge("shacl_correction", "exporter")
    graph.add_edge("exporter", END)

    return graph.compile()


def build_ext_ref_graph(tools: list):
    """
    Reduced pipeline graph for --ext-ref mode.

    Skips chunker / extractor / preprocessor entirely and starts directly at
    mapper_init, expecting ``entities`` and ``triplets`` to be pre-populated
    in the initial state (loaded from a previous run's ``_entities`` /
    ``_triplets`` fields).
    """
    node_elapsed.clear()

    mapper_init_node    = make_mapper_init_node(tools)
    resolve_entity_node = make_resolve_entity_node(tools)
    map_dt_prop_node    = make_map_dt_prop_node(tools)
    map_relation_node   = make_map_relation_node(tools)
    exporter_node       = make_exporter_node(tools)

    graph = StateGraph(AnchorState)

    graph.add_node("mapper_init",     _timed("mapper_init",          mapper_init_node))
    graph.add_node("resolve_entity",  _timed_accum("resolve_entity", resolve_entity_node))
    graph.add_node("map_dt_prop",     _timed_accum("map_dt_prop",    map_dt_prop_node))
    graph.add_node("map_relation",    _timed_accum("map_relation",   map_relation_node))
    graph.add_node("mapper_finalize", _timed("mapper_finalize",      mapper_finalize_node))
    graph.add_node("exporter",        _timed("exporter",             exporter_node))

    graph.add_edge(START, "mapper_init")

    graph.add_conditional_edges("mapper_init",    route_after_init, ["resolve_entity", "map_dt_prop"])
    graph.add_conditional_edges("resolve_entity", route_entity,     ["resolve_entity", "map_dt_prop"])
    graph.add_conditional_edges("map_dt_prop",    route_dt_prop,    ["map_dt_prop",    "map_relation"])
    graph.add_conditional_edges("map_relation",   route_relation,   ["map_relation",   "mapper_finalize"])

    graph.add_edge("mapper_finalize", "exporter")
    graph.add_edge("exporter", END)

    return graph.compile()


def print_graph(graph) -> None:
    """Print the Mermaid diagram of the compiled graph to stdout."""
    print(graph.get_graph().draw_mermaid())


def build_extended_mermaid(tools: list) -> str:
    """
    Build a Mermaid TD diagram showing the full pipeline with MCP tool nodes
    expanded under the mapper nodes (tool call → dotted return edges).
    """
    pipeline_nodes = [
        "chunker", "extractor", "preprocessor",
        "mapper_init", "resolve_entity", "map_dt_prop", "map_relation", "mapper_finalize",
        "exporter",
    ]
    mapper_nodes = {"mapper_init", "resolve_entity", "map_dt_prop", "map_relation", "mapper_finalize"}
    tool_names = [getattr(t, "name", str(t)) for t in tools]

    def safe(name: str) -> str:
        return name.replace("-", "_").replace("/", "_").replace(".", "_")

    lines = [
        "---",
        "config:",
        "  flowchart:",
        "    curve: linear",
        "---",
        "graph TD;",
        '\t__start__([<p>__start__</p>]):::first',
    ]
    for n in pipeline_nodes:
        lines.append(f"\t{n}({n})")
    lines.append('\t__end__([<p>__end__</p>]):::last')

    for name in tool_names:
        lines.append(f'\t{safe(name)}["{name}"]:::tool')

    lines.append("\t__start__ --> chunker;")
    seq = [
        ("chunker", "extractor"),
        ("extractor", "preprocessor"),
        ("preprocessor", "mapper_init"),
        ("mapper_init", "resolve_entity"),
        ("resolve_entity", "map_dt_prop"),
        ("map_dt_prop", "map_relation"),
        ("map_relation", "mapper_finalize"),
        ("mapper_finalize", "exporter"),
    ]
    for a, b in seq:
        lines.append(f"\t{a} --> {b};")
    lines.append("\texporter --> __end__;")

    for name in tool_names:
        s = safe(name)
        for mn in mapper_nodes:
            lines.append(f"\t{mn} --> {s};")
            lines.append(f"\t{s} -.-> {mn};")

    lines += [
        "\tclassDef default fill:#f2f0ff,line-height:1.2",
        "\tclassDef first fill-opacity:0",
        "\tclassDef last fill:#bfb6fc",
        "\tclassDef tool fill:#dbeafe,stroke:#3b82f6,color:#1e40af",
    ]
    return "\n".join(lines)
