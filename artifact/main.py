"""
ANCHOR — Text-to-Ontology Pipeline Entry Point

══════════════════════════════════════════════════════════════════════
 CTI Text Pipeline  (evaluation / CTINexus dataset)
══════════════════════════════════════════════════════════════════════
  python main.py --text "A new ransomware family..."   # single text
  python main.py --file input.txt                      # from .txt file
  python main.py --file-dir ./dataset/ --output-dir ./output/  # batch .txt
  python main.py --batch ./data/                       # batch .json (legacy)
  python main.py --graph                               # print Mermaid to stdout
  python main.py --graph-png graph.png                 # save graph as PNG

══════════════════════════════════════════════════════════════════════
 Flags
══════════════════════════════════════════════════════════════════════
  --timer          Print per-node timing after a run
  --measurement    Print search-vs-recurse breakdown and score distribution
  --schema SCHEMA  Override ontology schema: uco | stix | malont
  --output FILE    Output filename (single-run mode, default: result.json)
  --output-dir DIR Output directory  (batch mode,    default: ./output)
  --export-cypher  Also write a .cypher file (Neo4j MERGE/MATCH statements)

Environment variables (see .env):
  LLM_PROVIDER / ANTHROPIC_LLM_MODEL / OPENAI_LLM_MODEL / VLLM_LLM_MODEL
  ANTHROPIC_API_KEY / OPENAI_API_KEY / VLLM_BASE_URL / VLLM_API_KEY
  MCP_SERVER_SCRIPT  — path to ontology-mcp-server/main.py
  ONTOLOGY_DIR       — path to ontology folder (uco / stix / malont)
  EMBEDDING_MODEL    — sentence-transformer model (default: all-MiniLM-L6-v2)
  ENABLE_EVAL_TOOLS  — "true" to enable export_eval_json MCP tool
  OUTPUT_DIR         — where to write results (default: ./output)
  LLM_MAX_TOKENS     — extractor token cap (default: 2048)
  MAPPER_MAX_TOKENS  — mapper token cap    (default: 4096)
"""

import asyncio
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)


# ── Timing ────────────────────────────────────────────────────────────────────

def _print_measurements(measurements: list[dict]) -> None:
    """Print search-vs-recurse breakdown and score distribution to stderr."""
    import statistics as _stats

    phases = [
        ("entity",   "Entity (class)   "),
        ("dt_prop",  "DatatypeProperty "),
        ("relation", "ObjectProperty   "),
    ]

    print("\n── Measurement ─────────────────────────────────────────────────", file=sys.stderr)

    print("\n  Tool Resolution by Phase:", file=sys.stderr)
    print(f"  {'Phase':<20} {'search':>7} {'recurse':>8} {'fallback':>9} {'timeout':>8} {'total':>6}",
          file=sys.stderr)
    print(f"  {'-'*63}", file=sys.stderr)

    phase_score_data: dict[str, list[float]] = {}
    all_top_scores:   list[float] = []
    all_cand_scores:  list[float] = []

    for phase_key, phase_label in phases:
        items = [m for m in measurements if m["phase"] == phase_key]
        if not items:
            continue
        counts: dict[str, int] = {"search": 0, "recurse": 0, "fallback": 0, "timeout": 0}
        top_scores: list[float] = []
        for m in items:
            ft = m.get("final_tool", "fallback")
            counts[ft] = counts.get(ft, 0) + 1
            ts = m.get("top_score")
            if ts is not None:
                top_scores.append(ts)
                all_top_scores.append(ts)
            for sc in m.get("search_calls", []):
                for c in sc.get("candidates", []):
                    all_cand_scores.append(c["score"])
        phase_score_data[phase_key] = top_scores
        print(
            f"  {phase_label:<20} {counts['search']:>7} {counts['recurse']:>8}"
            f" {counts['fallback']:>9} {counts['timeout']:>8} {len(items):>6}",
            file=sys.stderr,
        )

    if all_top_scores:
        class_thr = float(os.getenv("LOW_CONFIDENCE_CLASS_THRESHOLD", "0.45"))
        prop_thr  = float(os.getenv("LOW_CONFIDENCE_PROP_THRESHOLD",  "0.3"))
        _phase_thr = {
            "entity":   class_thr,
            "dt_prop":  prop_thr,
            "relation": prop_thr,
        }
        print(
            f"\n  Search Score Stats — top-1 per item "
            f"(class-thr={class_thr}, prop-thr={prop_thr}):",
            file=sys.stderr,
        )
        print(
            f"  {'Phase':<20} {'n':>4}  {'min':>5}  {'Q1':>5}  {'med':>5}  "
            f"{'Q3':>5}  {'max':>5}  {'mean':>5}  {'≥thr':>8}",
            file=sys.stderr,
        )
        print(f"  {'-'*78}", file=sys.stderr)

        for phase_key, phase_label in phases:
            scores = phase_score_data.get(phase_key, [])
            if not scores:
                continue
            thr = _phase_thr.get(phase_key, class_thr)
            ss = sorted(scores)
            n  = len(ss)
            q1 = ss[n // 4]
            q2 = _stats.median(ss)
            q3 = ss[3 * n // 4]
            mn = _stats.mean(ss)
            above = sum(1 for s in scores if s >= thr)
            print(
                f"  {phase_label:<20} {n:>4}  {min(ss):>5.3f}  {q1:>5.3f}  "
                f"{q2:>5.3f}  {q3:>5.3f}  {max(ss):>5.3f}  {mn:>5.3f}  "
                f"{above:>3}/{n:<3}",
                file=sys.stderr,
            )

        ss_all = sorted(all_top_scores)
        n = len(ss_all)
        q1 = ss_all[n // 4]; q2 = _stats.median(ss_all); q3 = ss_all[3 * n // 4]
        mn = _stats.mean(ss_all)
        above = sum(1 for s in all_top_scores if s >= class_thr)
        print(f"  {'-'*78}", file=sys.stderr)
        print(
            f"  {'ALL':<20} {n:>4}  {min(ss_all):>5.3f}  {q1:>5.3f}  "
            f"{q2:>5.3f}  {q3:>5.3f}  {max(ss_all):>5.3f}  {mn:>5.3f}  "
            f"{above:>3}/{n:<3}",
            file=sys.stderr,
        )

    if all_cand_scores:
        n  = len(all_cand_scores)
        mn = _stats.mean(all_cand_scores)
        sd = _stats.stdev(all_cand_scores) if n >= 2 else 0.0
        ss = sorted(all_cand_scores)
        q1 = ss[n // 4]; q2 = _stats.median(ss); q3 = ss[3 * n // 4]
        print(f"\n  All candidate scores (n={n}, includes all search_* results):",
              file=sys.stderr)
        print(f"    mean={mn:.3f}  stdev={sd:.3f}  min={min(ss):.3f}  "
              f"Q1={q1:.3f}  median={q2:.3f}  Q3={q3:.3f}  max={max(ss):.3f}",
              file=sys.stderr)

    if all_top_scores:
        print("\n  Score Histogram (top-1 per item, all phases):", file=sys.stderr)
        buckets = [0] * 10
        for s in all_top_scores:
            buckets[min(int(s * 10), 9)] += 1
        max_count = max(buckets) if buckets else 1
        _c_thr = float(os.getenv("LOW_CONFIDENCE_CLASS_THRESHOLD", "0.45"))
        _p_thr = float(os.getenv("LOW_CONFIDENCE_PROP_THRESHOLD",  "0.3"))
        for i, count in enumerate(buckets):
            lo  = i * 0.1
            bar = "█" * int(count / max_count * 24)
            markers = []
            if lo <= _c_thr < lo + 0.1:
                markers.append(f"class-thr ({_c_thr})")
            if lo <= _p_thr < lo + 0.1:
                markers.append(f"prop-thr ({_p_thr})")
            marker = (" ← " + ", ".join(markers)) if markers else ""
            print(f"    [{lo:.1f}–{lo+0.1:.1f})  {bar:<24}  {count:>3}{marker}",
                  file=sys.stderr)

    print("\n────────────────────────────────────────────────────────────────", file=sys.stderr)


def _print_timing(node_times: dict[str, float], total: float) -> None:
    nodes = ["chunker", "extractor", "preprocessor",
             "mapper_init", "resolve_entity", "map_dt_prop", "map_relation",
             "mapper_finalize", "shacl_correction", "exporter"]
    print("\n── Timing ──────────────────────────────────────", file=sys.stderr)
    for n in nodes:
        t = node_times.get(n)
        bar = "█" * int((t or 0) / total * 30) if total > 0 else ""
        print(f"  {n:<14} {t:6.1f}s  {bar}" if t else f"  {n:<14}    —",
              file=sys.stderr)
    print(f"  {'TOTAL':<14} {total:6.1f}s", file=sys.stderr)
    print("────────────────────────────────────────────────", file=sys.stderr)


# ── MCP config ────────────────────────────────────────────────────────────────

def _mcp_config() -> dict:
    script = os.getenv("MCP_SERVER_SCRIPT")
    if not script:
        print("ERROR: MCP_SERVER_SCRIPT not set in environment.", file=sys.stderr)
        sys.exit(1)

    script_path = Path(script).resolve()
    if not script_path.exists():
        print(f"ERROR: MCP_SERVER_SCRIPT not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    ontology_dir = os.getenv("ONTOLOGY_DIR", "")
    if not ontology_dir:
        print("WARNING: ONTOLOGY_DIR not set — MCP server may use its default.",
              file=sys.stderr)
    print(f"[config] ONTOLOGY_DIR = {ontology_dir!r}", file=sys.stderr)

    env = {
        "ONTOLOGY_DIR": ontology_dir,
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cpu"),
        "ENABLE_EVAL_TOOLS": os.getenv("ENABLE_EVAL_TOOLS", "true"),
    }
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]

    return {
        "ontology-mcp-server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(script_path)],
            "env": env,
        }
    }


# ── PNG rendering ─────────────────────────────────────────────────────────────

def _render_mermaid_png(mermaid_code: str, output_path: str) -> None:
    clean = mermaid_code
    if mermaid_code.startswith("---"):
        end = mermaid_code.find("---", 3)
        if end != -1:
            clean = mermaid_code[end + 3:].lstrip()

    try:
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png
        png_bytes = draw_mermaid_png(mermaid_code)
        Path(output_path).write_bytes(png_bytes)
        return
    except Exception:
        pass

    encoded = base64.urlsafe_b64encode(clean.encode()).decode()
    url = f"https://mermaid.ink/img/{encoded}?bgColor=white&padding=20"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "anchor"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            Path(output_path).write_bytes(resp.read())
        return
    except Exception as e:
        raise RuntimeError(
            f"Could not render PNG (tried LangChain renderer and mermaid.ink): {e}\n"
            "Install playwright:  pip install playwright && playwright install chromium\n"
            "Or check network connectivity for mermaid.ink."
        ) from e


def save_graph_png(output_path: str) -> None:
    from src.graph import build_graph, print_graph as _print_graph
    compiled = build_graph([])
    mermaid_code = compiled.get_graph().draw_mermaid()
    _render_mermaid_png(mermaid_code, output_path)
    print(f"Graph saved → {output_path}")


# ── Pipeline runs ────────────────────────────────────────────────────────────

async def run_pipeline(text: str, output_file: str, *, timer: bool = False) -> tuple[dict, list[dict]]:
    from src.graph import build_graph, node_elapsed
    from langchain_mcp_adapters.tools import load_mcp_tools

    config = _mcp_config()
    mcp_client = MultiServerMCPClient(config)

    async with mcp_client.session("ontology-mcp-server") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(tools)

        initial_state = {
            "input_text": text,
            "output_file": output_file,
            "chunks": [],
            "raw_entities": [],
            "raw_triplets": [],
            "entities": [],
            "triplets": [],
            "entities_pending":   [],
            "entities_resolved":  [],
            "dt_props_pending":   [],
            "dt_props_resolved":  [],
            "relations_pending":  [],
            "relations_resolved": [],
            "mapped_entities": [],
            "mapped_triplets": [],
            "result": None,
            "errors": [],
            "measurements": [],
        }

        t0 = time.perf_counter()
        try:
            final_state = await graph.ainvoke(initial_state,
                                              config={"recursion_limit": 500})
        finally:
            total = time.perf_counter() - t0
            if timer:
                _print_timing(node_elapsed, total)

    if final_state.get("errors"):
        print("\n⚠ Errors during pipeline:", file=sys.stderr)
        for e in final_state["errors"]:
            print(f"  - {e}", file=sys.stderr)

    return final_state.get("result", {}), final_state.get("measurements", [])


def _output_suffix() -> str:
    """Build '{provider}_{model}_{schema}' tag for output filenames."""
    provider = os.getenv("LLM_PROVIDER", "llm").lower().strip()
    model_key = {"anthropic": "ANTHROPIC_LLM_MODEL",
                 "openai":    "OPENAI_LLM_MODEL",
                 "vllm":      "VLLM_LLM_MODEL"}.get(provider, "VLLM_LLM_MODEL")
    model = os.getenv(model_key, "").strip()
    model_safe = model.replace("/", "-").replace("\\", "-") if model else provider
    ontology_dir = os.getenv("ONTOLOGY_DIR", "")
    schema = os.path.basename(ontology_dir.rstrip("/\\")) if ontology_dir else "ontology"
    return f"{provider}_{model_safe}_{schema}"


_DATASET_PREFIXES = ("ctinexus_", "gtikg_", "ttpdrill_")


def _strip_dataset_prefix(stem: str) -> str:
    for prefix in _DATASET_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _find_ext_ref(ext_ref_path: Path, stem: str) -> list[Path]:
    def _candidates(s: str) -> list[Path]:
        suffixed = sorted(ext_ref_path.glob(f"{s}_*.json"))
        if suffixed:
            return suffixed
        exact = ext_ref_path / f"{s}.json"
        if exact.exists():
            return [exact]
        return []

    hits = _candidates(stem)
    if hits:
        return hits

    for prefix in _DATASET_PREFIXES:
        if stem.startswith(prefix):
            hits = _candidates(stem[len(prefix):])
            if hits:
                return hits

    return []


def _make_initial_state(text: str, output_file: str) -> dict:
    return {
        "input_text": text, "output_file": output_file,
        "chunks": [], "raw_entities": [], "raw_triplets": [],
        "entities": [], "triplets": [],
        "entities_pending": [], "entities_resolved": [],
        "dt_props_pending": [], "dt_props_resolved": [],
        "relations_pending": [], "relations_resolved": [],
        "mapped_entities": [], "mapped_triplets": [],
        "result": None, "errors": [], "measurements": [],
    }



async def run_file_dir(input_dir: str, output_dir: str, *, timer: bool = False, limit: int | None = None) -> list[dict]:
    from src.graph import build_graph, node_elapsed
    from langchain_mcp_adapters.tools import load_mcp_tools

    input_path = Path(input_dir)
    txt_files = sorted(input_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_dir}", file=sys.stderr)
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = output_dir

    pending = []
    for txt_file in txt_files:
        output_file = f"{_strip_dataset_prefix(txt_file.stem)}.json"
        if (output_path / output_file).exists():
            print(f"Skipping (already done): {txt_file.name}  →  {output_file}")
        else:
            pending.append((txt_file, output_file))

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        print("All files already processed — nothing to do.")
        return []

    config = _mcp_config()
    mcp_client = MultiServerMCPClient(config)
    all_measurements: list[dict] = []

    async with mcp_client.session("ontology-mcp-server") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(tools)

        for txt_file, output_file in pending:
            text = txt_file.read_text(encoding="utf-8")
            print(f"Processing: {txt_file.name}  →  {output_file}")
            node_elapsed.clear()
            t0 = time.perf_counter()

            try:
                final_state = await graph.ainvoke(_make_initial_state(text, output_file),
                                                  config={"recursion_limit": 500})
            finally:
                total = time.perf_counter() - t0
                if timer:
                    _print_timing(node_elapsed, total)

            if final_state.get("errors"):
                print(f"  ⚠ Errors: {final_state['errors']}")

            all_measurements.extend(final_state.get("measurements", []))
            result = final_state.get("result", {})
            print(f"  → {len(result.get('extracted_entities', []))} entities, "
                  f"{len(result.get('extracted_triplets', []))} triplets")

    return all_measurements


async def run_file_dir_ext_ref(
    input_dir: str, output_dir: str, ext_ref_dir: str, *, timer: bool = False, limit: int | None = None
) -> list[dict]:
    from src.graph import build_ext_ref_graph, node_elapsed
    from langchain_mcp_adapters.tools import load_mcp_tools

    input_path   = Path(input_dir)
    ext_ref_path = Path(ext_ref_dir)
    txt_files    = sorted(input_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_dir}", file=sys.stderr)
        return
    if not ext_ref_path.is_dir():
        print(f"ERROR: --ext-ref directory not found: {ext_ref_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = output_dir

    pending = []
    for txt_file in txt_files:
        output_file = f"{_strip_dataset_prefix(txt_file.stem)}.json"
        if (output_path / output_file).exists():
            print(f"Skipping (already done): {txt_file.name}  →  {output_file}")
            continue

        matches = _find_ext_ref(ext_ref_path, txt_file.stem)
        if not matches:
            print(f"  [!] No reference file for '{txt_file.stem}' in {ext_ref_dir} — skipping")
            continue
        if len(matches) > 1:
            print(f"  [!] Multiple reference files for '{txt_file.stem}': "
                  f"{[m.name for m in matches]} — using first")
        pending.append((txt_file, output_file, matches[0]))

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        print("All files already processed or no reference files found — nothing to do.")
        return []

    config = _mcp_config()
    mcp_client = MultiServerMCPClient(config)
    all_measurements: list[dict] = []

    async with mcp_client.session("ontology-mcp-server") as session:
        tools = await load_mcp_tools(session)
        graph = build_ext_ref_graph(tools)

        for txt_file, output_file, ref_file in pending:
            with open(ref_file, encoding="utf-8") as f:
                ref_data = json.load(f)

            text = ref_data.get("text", "") or txt_file.read_text(encoding="utf-8")

            raw_extracted = ref_data.get("extracted_entities", [])
            if not raw_extracted:
                print(f"  [!] '{ref_file.name}' has no 'extracted_entities' — skipping")
                continue
            entities = [
                {
                    "name": e["name"],
                    "type_hint": (e.get("class", "").rstrip("/").split("/")[-1]
                                  or e["name"]),
                    "properties": {},
                }
                for e in raw_extracted
            ]

            triplets = [
                {
                    "subject":       t["subject"],
                    "predicate":     t.get("relation", ""),
                    "object":        t["object"],
                    "evidence":      "",
                    "property_type": t.get("property_type", "ObjectProperty"),
                }
                for t in ref_data.get("extracted_triplets", [])
                if t.get("subject") and t.get("object")
            ]

            print(f"Processing (ext-ref): {txt_file.name}  →  {output_file}")
            print(f"  ref: {ref_file.name}  ({len(entities)} entities, {len(triplets)} triplets)")
            node_elapsed.clear()
            t0 = time.perf_counter()

            state = _make_initial_state(text, output_file)
            state["entities"] = entities
            state["triplets"] = triplets

            try:
                final_state = await graph.ainvoke(state, config={"recursion_limit": 500})
            finally:
                total = time.perf_counter() - t0
                if timer:
                    _print_timing(node_elapsed, total)

            if final_state.get("errors"):
                print(f"  ⚠ Errors: {final_state['errors']}")

            all_measurements.extend(final_state.get("measurements", []))
            result = final_state.get("result", {})
            print(f"  → {len(result.get('extracted_entities', []))} entities, "
                  f"{len(result.get('extracted_triplets', []))} triplets")

    return all_measurements


async def run_batch(input_dir: str, output_dir: str, *, timer: bool = False) -> list[dict]:
    from src.graph import build_graph, node_elapsed
    from langchain_mcp_adapters.tools import load_mcp_tools

    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        print(f"No .json files found in {input_dir}", file=sys.stderr)
        return

    config = _mcp_config()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mcp_client = MultiServerMCPClient(config)

    all_measurements: list[dict] = []

    async with mcp_client.session("ontology-mcp-server") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(tools)

        for json_file in json_files:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text:
                print(f"  Skipping {json_file.name} — no 'text' field")
                continue

            output_file = json_file.stem + "_result.json"
            os.environ["OUTPUT_DIR"] = output_dir

            print(f"Processing: {json_file.name}")
            node_elapsed.clear()
            t0 = time.perf_counter()

            try:
                final_state = await graph.ainvoke(_make_initial_state(text, output_file),
                                                  config={"recursion_limit": 500})
            finally:
                total = time.perf_counter() - t0
                if timer:
                    _print_timing(node_elapsed, total)

            if final_state.get("errors"):
                print(f"  ⚠ Errors: {final_state['errors']}")

            all_measurements.extend(final_state.get("measurements", []))
            result = final_state.get("result", {})
            print(f"  → {len(result.get('extracted_entities', []))} entities, "
                  f"{len(result.get('extracted_triplets', []))} triplets")

    return all_measurements


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ANCHOR — Text-to-Ontology Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input modes:\n"
            "  --text / --file / --file-dir / --batch\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    # ── Input: CTI text (evaluation / CTINexus dataset) ──────────────────────
    mode.add_argument("--text",      type=str, help="Input CTI text string")
    mode.add_argument("--file",      type=str, help="Path to .txt input file")
    mode.add_argument("--file-dir",  type=str, help="Directory of .txt files (batch)")
    mode.add_argument("--batch",     type=str, help="Directory of .json files (legacy batch)")

    # ── Graph utilities ───────────────────────────────────────────────────────
    mode.add_argument("--graph",     action="store_true", help="Print Mermaid diagram to stdout")
    mode.add_argument("--graph-png", type=str, metavar="PATH",
                      help="Save full pipeline graph as PNG")

    # ── Shared options ────────────────────────────────────────────────────────
    parser.add_argument("--output",     type=str, default="result.json",
                        help="Output filename (single-run mode)")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory (batch mode)")
    parser.add_argument("--timer",      action="store_true",
                        help="Print per-node timing after the run")
    parser.add_argument("--measurement", action="store_true",
                        help="Print search-vs-recurse tool usage and embedding score "
                             "distribution after the run (MCP mode only)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process at most N files (batch modes; skipped files don't count)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print mapper agent tool calls and LLM steps in real time")
    parser.add_argument("--schema", type=str, choices=["uco", "stix", "malont"],
                        help="Override ontology schema")
    parser.add_argument("--llm", type=str, metavar="PROVIDER",
                        help="Override LLM provider (anthropic | openai | vllm)")
    parser.add_argument("--llm-model", type=str, metavar="MODEL",
                        help="Override model name for the active provider")
    parser.add_argument("--ext-ref", type=str, metavar="DIR",
                        help="(--file-dir only) Reuse extracted entities/triplets from DIR, "
                             "skipping chunker/extractor/preprocessor")
    parser.add_argument("--export-cypher", action="store_true",
                        help="Also write a .cypher file alongside the JSON output "
                             "(Neo4j-importable MERGE/MATCH statements)")

    # ── Ablation / experiment flags ───────────────────────────────────────────
    abl = parser.add_argument_group("Ablation / experiment flags")

    abl.add_argument("--single-pass", action="store_true",
                     help="[Ablation ①] 1-pass joint entity+relation extraction")
    abl.add_argument("--no-coref", action="store_true",
                     help="[Ablation ②] Disable coreference resolution (Pass 1.5)")
    abl.add_argument("--no-noise-filter", action="store_true",
                     help="[Ablation ③] Disable noise entity filter in preprocessor")
    abl.add_argument("--no-ioc", action="store_true",
                     help="[Ablation ④] Disable IoC pre-assignment")
    abl.add_argument("--mapper-tools", choices=["search", "recurse", "both"], default=None,
                     metavar="MODE",
                     help="[Ablation ⑤] Restrict mapper tool strategy: search | recurse | both")
    abl.add_argument("--mapper-tools-class", choices=["search", "recurse", "both"], default=None,
                     metavar="MODE",
                     help="[Ablation ⑤a] Tool mode for entity class resolution only")
    abl.add_argument("--mapper-tools-prop", choices=["search", "recurse", "both"], default=None,
                     metavar="MODE",
                     help="[Ablation ⑤b] Tool mode for attribute/relation mapping only")
    abl.add_argument("--max-tool-calls", type=int, default=None, metavar="N",
                     help="[Ablation ⑥] Max MCP tool calls per mapper item (default: 4)")
    abl.add_argument("--class-threshold", type=float, default=None, metavar="T",
                     help="[Ablation ⑦a] LOW_CONFIDENCE_CLASS_THRESHOLD for search_classes (default: 0.45)")
    abl.add_argument("--prop-threshold", type=float, default=None, metavar="T",
                     help="[Ablation ⑦b] LOW_CONFIDENCE_PROP_THRESHOLD for search_attributes/search_relations (default: 0.3)")
    abl.add_argument("--name-weight", type=float, default=None, metavar="W",
                     help="[Ablation ⑧] Weight for class NAME embedding vs. COMMENT embedding")
    abl.add_argument("--keyword-bonus", type=float, default=None, metavar="B",
                     help="[Ablation ⑨] Bonus score for verbatim name match")
    abl.add_argument("--no-thinking", action="store_true",
                     help="[Ablation ⑩] Disable extended thinking for the mapper LLM")
    abl.add_argument("--mapper-timeout", type=int, default=None, metavar="SEC",
                     help="[Ablation ⑪] Timeout in seconds per mapper item (default: 120)")

    args = parser.parse_args()

    # ── Apply CLI overrides before any LLM imports ────────────────────────────
    if args.schema:
        ontology_dir = os.getenv("ONTOLOGY_DIR", "")
        if ontology_dir:
            new_dir = str(Path(ontology_dir).parent / args.schema)
            os.environ["ONTOLOGY_DIR"] = new_dir
            print(f"[config] --schema {args.schema} → ONTOLOGY_DIR={new_dir!r}", file=sys.stderr)
        else:
            print("WARNING: ONTOLOGY_DIR not set in .env — --schema ignored", file=sys.stderr)

    if args.llm:
        os.environ["LLM_PROVIDER"] = args.llm.lower().strip()

    if args.llm_model:
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower().strip()
        model_key = {"anthropic": "ANTHROPIC_LLM_MODEL",
                     "openai":    "OPENAI_LLM_MODEL",
                     "vllm":      "VLLM_LLM_MODEL"}.get(provider, "VLLM_LLM_MODEL")
        os.environ[model_key] = args.llm_model

    if args.verbose:
        os.environ["ANCHOR_VERBOSE"] = "1"

    # ── Ablation env vars ─────────────────────────────────────────────────────
    if args.single_pass:
        os.environ["ANCHOR_SINGLE_PASS"] = "1"
    if args.no_coref:
        os.environ["ANCHOR_NO_COREF"] = "1"
    if args.no_noise_filter:
        os.environ["ANCHOR_NO_NOISE_FILTER"] = "1"
    if args.no_ioc:
        os.environ["ANCHOR_NO_IOC"] = "1"
    if args.mapper_tools:
        os.environ["ANCHOR_MAPPER_TOOLS"] = args.mapper_tools
    if args.mapper_tools_class:
        os.environ["ANCHOR_MAPPER_TOOLS_CLASS"] = args.mapper_tools_class
    if args.mapper_tools_prop:
        os.environ["ANCHOR_MAPPER_TOOLS_PROP"] = args.mapper_tools_prop
    if args.max_tool_calls is not None:
        os.environ["ANCHOR_MAX_TOOL_CALLS"] = str(args.max_tool_calls)
    if args.class_threshold is not None:
        os.environ["LOW_CONFIDENCE_CLASS_THRESHOLD"] = str(args.class_threshold)
    if args.prop_threshold is not None:
        os.environ["LOW_CONFIDENCE_PROP_THRESHOLD"] = str(args.prop_threshold)
    if args.name_weight is not None:
        os.environ["EMB_NAME_WEIGHT"] = str(args.name_weight)
    if args.keyword_bonus is not None:
        os.environ["KEYWORD_BONUS"] = str(args.keyword_bonus)
    if args.no_thinking:
        os.environ["VLLM_THINKING"] = "false"
    if args.mapper_timeout is not None:
        os.environ["MAPPER_TIMEOUT"] = str(args.mapper_timeout)
    if args.export_cypher:
        os.environ["EXPORT_CYPHER"] = "true"

    # ── Graph utilities ───────────────────────────────────────────────────────
    if args.graph:
        from src.graph import build_graph, print_graph
        print_graph(build_graph([]))
        return

    if args.graph_png:
        save_graph_png(args.graph_png)
        return

    wall_t0 = time.perf_counter()
    measurements: list[dict] = []

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.file_dir:
        if args.ext_ref:
            measurements = asyncio.run(run_file_dir_ext_ref(
                args.file_dir, args.output_dir, args.ext_ref,
                timer=args.timer, limit=args.limit,
            ))
        else:
            measurements = asyncio.run(run_file_dir(
                args.file_dir, args.output_dir,
                timer=args.timer, limit=args.limit,
            ))

    elif args.batch:
        measurements = asyncio.run(run_batch(args.batch, args.output_dir, timer=args.timer))

    elif args.text or args.file:
        text = open(args.file, encoding="utf-8").read() if args.file else args.text
        result, measurements = asyncio.run(
            run_pipeline(text, args.output, timer=args.timer)
        )
        print("\n── Result ──────────────────────────────────────")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.measurement and measurements:
        _print_measurements(measurements)
    elif args.measurement:
        print("\n[--measurement] No measurement data collected (no items processed).",
              file=sys.stderr)

    wall_total = time.perf_counter() - wall_t0
    h = int(wall_total // 3600)
    m = int((wall_total % 3600) // 60)
    s = wall_total % 60
    print(f"\n── Total elapsed: {h:02d}:{m:02d}:{s:05.2f} ({wall_total:.1f}s) ──────────────────")


if __name__ == "__main__":
    main()
