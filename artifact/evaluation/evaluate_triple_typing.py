#!/usr/bin/env python3
"""
evaluate_triple_typing.py — Relation Type Matching Evaluation

Evaluates how accurately relation/property types are predicted. Uses LLM-Full
matching to find (pred, gold) triple pairs, then applies hierarchical partial
scoring via ontology subClassOf chains:
  exact match  → 1.0
  ±1 level     → 0.6
  ±2 levels    → 0.3
  ±3+ levels   → 0.0

Both ground truth and predictions use the 'property' field, which may be:
  - A full URI:    https://ontology.unifiedcyberontology.org/uco/core/target
  - A short form:  core:target
  - null / empty   (treated as untyped, excluded from accuracy calculation)

Ground truth layout expected:
  ground_truth/{ontology}/json/{slug}_{ontology}.json
  Each file contains: {"extracted_triplets": [{"subject","relation","property",...}, ...], ...}

Usage:
  python evaluate_triple_typing.py \\
      --results  output/vllm_qwen3.5-35b_uco/ \\
      --ground-truth ground_truth/
"""

import asyncio
import argparse
import io
import json
import os
import re
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).parent.resolve()

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(ROOT.parent / ".env", override=True)
except ImportError:
    pass



# ── Slug helpers ───────────────────────────────────────────────────────────────

def _extract_gt_slug(stem: str) -> str:
    return re.sub(r'_(uco|stix|malont)$', '', stem)


def _extract_output_slug(stem: str) -> str:
    # 1. Strip schema suffix
    stem = re.sub(r'_(uco|stix|malont|none)$', '', stem)
    # 2. Strip method/provider suffix
    idx = stem.find('_vllm_')
    if idx != -1:
        stem = stem[:idx]
    else:
        for method in ('_ttpdrill', '_ctinexus', '_gtikg', '_anthropic', '_openai'):
            if stem.endswith(method):
                stem = stem[: -len(method)]
                break
    # 3. Strip leading dataset prefix (longest match first)
    for prefix in ('ctinexus_ctinexus_', 'ctinexus_gtikg_', 'ctinexus_ttpdrill_',
                   'ctinexus_', 'gtikg_', 'ttpdrill_'):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _infer_ontology_from_dir(dirpath: Path) -> str:
    name = dirpath.name.lower()
    for ont in ('malont', 'stix', 'uco'):
        if name.endswith(f'_{ont}'):
            return ont
    return 'none'


# ── GT loader ──────────────────────────────────────────────────────────────────

def _load_gt(gt_base: str, ontology: str) -> dict:
    """Load ground truth triples (with property field) indexed by slug."""
    base = Path(gt_base)
    candidate_json = base / ontology / "json" if ontology and ontology != 'none' else None
    candidate_ont  = base / ontology            if ontology and ontology != 'none' else None
    if candidate_json and candidate_json.is_dir():
        gt_dir = candidate_json
    elif candidate_ont and candidate_ont.is_dir():
        gt_dir = candidate_ont
    elif base.is_dir():
        gt_dir = base
    else:
        raise ValueError(f"GT dir not found: {base}")

    def _read_dir(d: Path) -> dict:
        out: dict = {}
        for f in sorted(d.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fp:
                    raw = json.load(fp)
            except Exception as e:
                print(f"[!] Could not read GT file {f.name}: {e}")
                continue
            slug = _extract_gt_slug(f.stem)
            raw_triples = raw.get("extracted_triplets") or raw.get("triplets") or []
            all_triples = [
                {
                    "subject":  t.get("subject", ""),
                    "relation": t.get("relation", ""),
                    "object":   t.get("object", ""),
                    "property": t.get("property") or ""}
                for t in raw_triples
            ]
            filtered = [t for t in all_triples
                        if t["property"] not in FALLBACK_PROPERTY_URIS]
            n_excluded = len(all_triples) - len(filtered)
            if n_excluded:
                print(f"  [GT filter] {f.name}: excluded {n_excluded} triple(s) "
                      f"with fallback property URI")
            out[slug] = {
                "id": slug,
                "ground_truth_triples": filtered}
        return out

    gt_map = _read_dir(gt_dir)

    if not gt_map and gt_dir == base:
        for schema in ('stix', 'uco', 'malont'):
            sub = base / schema / "json"
            if sub.is_dir():
                gt_map = _read_dir(sub)
                if gt_map:
                    print(f"[*] ontology=none — using GT from '{schema}' schema ({sub})")
                    break

    if not gt_map:
        raise ValueError(f"No GT JSON files found under: {gt_dir}")
    return gt_map


# ── Ontology Hierarchy (subClassOf) ───────────────────────────────────────────

class OntologyHierarchy:
    DIST_SCORES = {0: 1.0, 1: 0.6, 2: 0.3}
    _BLANK = re.compile(r"^N[0-9a-f]{24}$", re.I)

    def __init__(self, ontology_root: Path) -> None:
        self._root    = ontology_root
        self._parents: Dict[str, Dict[str, Set[str]]] = {}

    def _local(self, uri: str) -> str:
        local = str(uri).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        return re.sub(r"[^a-z0-9]+", "", local.casefold())

    def _load(self, ontology: str) -> Dict[str, Set[str]]:
        if ontology in self._parents: return self._parents[ontology]
        import rdflib
        from rdflib import RDFS
        warnings.filterwarnings("ignore")
        g = rdflib.Graph()
        ont_dir = self._root / ontology
        if ontology in ("uco"):
            for f in ont_dir.rglob("*.ttl"):
                try: g.parse(str(f), format="turtle")
                except Exception: pass
        else:
            for f in ont_dir.rglob("*.owl"):
                try: g.parse(str(f), format="xml")
                except Exception: pass
        parents: Dict[str, Set[str]] = {}
        for s, o in g.subject_objects(RDFS.subClassOf):
            s_loc = self._local(str(s)); o_loc = self._local(str(o))
            if s_loc and o_loc and s_loc != o_loc and not self._BLANK.match(s_loc) and not self._BLANK.match(o_loc):
                parents.setdefault(s_loc, set()).add(o_loc)
        self._parents[ontology] = parents
        return parents

    def _ancestors(self, start: str, parents: Dict[str, Set[str]], max_depth: int) -> Dict[str, int]:
        visited: Dict[str, int] = {start: 0}
        queue: deque = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth: continue
            for parent in parents.get(node, set()):
                if parent not in visited:
                    visited[parent] = depth + 1; queue.append((parent, depth + 1))
        return visited

    def score(self, pred_type: str, gold_type: str, ontology: str) -> float:
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        if not pred_norm or not gold_norm: return 0.0
        if pred_norm == gold_norm:         return 1.0
        parents = self._load(ontology); max_d = max(self.DIST_SCORES)
        pred_anc = self._ancestors(pred_norm, parents, max_d)
        if gold_norm in pred_anc: return self.DIST_SCORES.get(pred_anc[gold_norm], 0.0)
        gold_anc = self._ancestors(gold_norm, parents, max_d)
        if pred_norm in gold_anc: return self.DIST_SCORES.get(gold_anc[pred_norm], 0.0)
        return 0.0


_ONTOLOGY_HIERARCHY: Optional[OntologyHierarchy] = None


def _get_hierarchy() -> OntologyHierarchy:
    global _ONTOLOGY_HIERARCHY
    if _ONTOLOGY_HIERARCHY is None:
        _ONTOLOGY_HIERARCHY = OntologyHierarchy(ROOT.parent / "ontology")
    return _ONTOLOGY_HIERARCHY


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _prf(tp, predicted: int, gold: int) -> dict:
    tp_f = float(tp)
    p  = tp_f / predicted if predicted > 0 else 0.0
    r  = tp_f / gold      if gold      > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "predicted": predicted, "gold": gold}


def _aggregate(metrics_list: List[dict]) -> dict:
    if not metrics_list: return {}
    result: dict = {}
    for k in ["p", "r", "f1"]:
        vals = [m[k] for m in metrics_list if k in m]
        result[f"macro_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    tp   = sum(m.get("tp",        0) for m in metrics_list)
    pred = sum(m.get("predicted", 0) for m in metrics_list)
    gold = sum(m.get("gold",      0) for m in metrics_list)
    micro = _prf(tp, pred, gold)
    result["micro_p"]  = micro["p"];  result["micro_r"]  = micro["r"]
    result["micro_f1"] = micro["f1"]; result["micro_tp"] = tp
    result["micro_fp"] = pred - tp;   result["micro_fn"] = gold - tp
    return result


SUPPORTED_ONTOLOGIES = {"uco", "malont", "stix"}
EMPTY_TYPE_MARKERS   = {"", "none", "null", "nil", "unknown", "unmapped", "nonmatch", "nomatch"}

# Ground-truth triples whose property is one of these generic W3C URIs were not
# successfully mapped to a schema-specific property and are excluded from typing
# evaluation (they carry no meaningful ontology signal).
FALLBACK_PROPERTY_URIS: set[str] = {
    "http://www.w3.org/2000/01/rdf-schema#seeAlso",        # UCO + MALOnt: unmapped relation
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",     # MALOnt: unmapped relation
    "http://www.w3.org/2000/01/rdf-schema#label",          # fallback DatatypeProperty
    "http://www.w3.org/2002/07/owl#Thing",                 # (edge case)
}


def _normalize_ontology(ontology: str) -> str:
    return (ontology or "").strip().lower()


def _resolve_ontology(item: dict, override: str = None) -> str:
    ontology = _normalize_ontology(override or item.get("ontology"))
    if ontology not in SUPPORTED_ONTOLOGIES:
        raise ValueError(
            f"Unsupported or missing ontology '{item.get('ontology')}'. "
            "Use --ontology with one of: uco, stix, malont."
        )
    return ontology


def _normalize_type_label(label) -> str:
    """Normalise a property URI or short-form to a plain lowercase token.

    Handles:
      - Full URI:   https://...uco/core/target  → "target"
      - Short form: core:target                 → "target"
      - Plain name: ObjectProperty              → "objectproperty"
      - null/empty/unknown                      → ""
    """
    if label is None: return ""
    value = str(label).strip()
    if not value: return ""
    if "://" in value:   value = value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    elif ":" in value:   value = value.rsplit(":", 1)[-1]
    norm = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return "" if norm in EMPTY_TYPE_MARKERS else norm


def _triple_display(t: dict) -> str:
    return f"({t.get('subject','?')}, {t.get('relation','?')}, {t.get('object','?')})"


def _relation_pred_type(obj: dict) -> str:
    """Get predicted triple property (URI or short form)."""
    return (obj.get("property") or "").strip()


def _relation_gold_type(obj: dict) -> str:
    """Get gold triple property (URI or short form)."""
    return (obj.get("property") or "").strip()


def _count_relation_type_hier_stats(
    pred_triples: List[dict], gold_triples: List[dict],
    pairs: List[Tuple[int, int]], ontology: str,
) -> Tuple[float, int]:
    hier = _get_hierarchy()
    score_sum = 0.0; eligible = 0; used_g: set = set()
    for pi, gj in pairs:
        if gj in used_g: continue
        if not (0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples)): continue
        used_g.add(gj)
        pred_type = _relation_pred_type(pred_triples[pi])
        gold_type = _relation_gold_type(gold_triples[gj])
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        if pred_norm and gold_norm:
            eligible  += 1
            score_sum += hier.score(pred_type, gold_type, ontology)
    return score_sum, eligible


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_llm(provider: str, model: str, base_url: str = None):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY", "dummy")
        if base_url:  # OpenAI-compatible server (vLLM etc.) — use its own key
            api_key = os.getenv("VLLM_API_KEY", api_key)
        kwargs = {"model": model, "api_key": api_key, "temperature": 0,
                  "model_kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
    elif provider in ("claude", "anthropic"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model,
                          base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                          temperature=0)


def _parse_llm_pairs(content: str) -> list:
    """Parse LLM response into [{pred, gold}] handling array, bare-object, multi-object, or parallel-arrays formats."""
    m = re.search(r'\[\s*(?:\{[^{}\[\]]*\}(?:\s*,\s*\{[^{}\[\]]*\})*\s*)?\]', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    objs = []
    for obj_str in re.findall(r'\{[^{}]*\}', content, re.DOTALL):
        try:
            obj = json.loads(obj_str)
            if "pred" in obj and "gold" in obj:
                # Handle parallel-arrays format: {"pred": [1,2,...], "gold": [1,2,...]}
                if isinstance(obj["pred"], list) and isinstance(obj["gold"], list):
                    for p, g in zip(obj["pred"], obj["gold"]):
                        if p is not None and g is not None:
                            objs.append({"pred": p, "gold": g})
                else:
                    objs.append(obj)
        except Exception:
            pass
    return objs


async def _llm_batch_match_triples_full(
    pred_triples: List[dict], gold_triples: List[dict], llm, timeout: float = 180.0,
) -> List[Tuple[int, int]]:
    from langchain_core.messages import SystemMessage, HumanMessage
    if not pred_triples or not gold_triples: return []
    task = (
        "Match each predicted (Subject, Relation, Object) triple to the semantically "
        "equivalent gold triple.\n"
        "Rules: Match if subject+object refer to the same entities AND the relation "
        "expresses the same semantic relationship. Each gold at most once."
    )
    pred_lines = "\n".join(f"{i+1}. {_triple_display(t)}" for i, t in enumerate(pred_triples))
    gold_lines = "\n".join(f"{j+1}. {_triple_display(t)}" for j, t in enumerate(gold_triples))
    prompt = (
        f"{task}\n\nPredicted:\n{pred_lines}\n\nGold:\n{gold_lines}\n\n"
        'Output JSON: [{"pred": <1-indexed>, "gold": <1-indexed>}, ...] or []'
    )
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content="Precise semantic evaluation assistant. Output only valid JSON."),
                HumanMessage(content=prompt),
            ]),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        print(f"  [!] LLM timeout ({timeout:.0f}s) for triple matching — returning []", flush=True)
        return []
    try:
        data = _parse_llm_pairs(resp.content)
        if not data and resp.content.strip():
            try:
                json.loads(resp.content.strip())
            except Exception:
                print(f"  [!] LLM response unparseable: {resp.content[:200]!r}", flush=True)
        raw_pairs = [
            (int(d["pred"]) - 1, int(d["gold"]) - 1)
            for d in data
            if "pred" in d and "gold" in d and d["pred"] is not None and d["gold"] is not None
        ]
        used_g: set = set(); pairs = []
        for pi, gj in raw_pairs:
            if 0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples) and gj not in used_g:
                pairs.append((pi, gj)); used_g.add(gj)
        return pairs
    except Exception:
        return []


# ── Per-sample evaluation ──────────────────────────────────────────────────────

async def evaluate_sample(
    item: dict, gt: dict, llm,
    ontology_override: str = None, llm_timeout: float = 180.0,
) -> dict:
    ontology = _resolve_ontology(item, ontology_override)
    preds = [
        {k: t.get(k, "") for k in ("subject", "relation", "object", "property")}
        for t in item.get("extracted_triplets", [])
        if t.get("subject") or t.get("object")
    ]
    golds = list(gt.get("ground_truth_triples", []))

    base = {
        "id": gt["id"], "ontology": ontology,
        "n_pred": len(preds), "n_gold": len(golds)}
    gold_type_list = [_relation_gold_type(t) or "NON_MATCH" for t in golds]
    zero_type = _prf(0.0, 0, 0)

    if not preds or not golds:
        return {**base, "gold_type_list": gold_type_list,
                "n_llm_matched": 0, "n_eligible": 0, "type_hier": zero_type}

    llm_full_pairs = await _llm_batch_match_triples_full(preds, golds, llm, llm_timeout)
    llm_full_tp    = sum(1 for _ in llm_full_pairs)
    type_score, type_n = _count_relation_type_hier_stats(preds, golds, llm_full_pairs, ontology)

    # Per-pair type detail log
    hier = _get_hierarchy()
    type_log = []
    n_untyped_pred = 0
    used_g: set = set()
    for pi, gj in llm_full_pairs:
        if gj in used_g: continue
        if not (0 <= pi < len(preds) and 0 <= gj < len(golds)): continue
        used_g.add(gj)
        pred_type = _relation_pred_type(preds[pi])
        gold_type = _relation_gold_type(golds[gj])
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        sc = hier.score(pred_type, gold_type, ontology) if (pred_norm and gold_norm) else None
        if gold_norm and not pred_norm:
            n_untyped_pred += 1
        type_log.append({
            "pred_triple":   _triple_display(preds[pi]),
            "gold_triple":   _triple_display(golds[gj]),
            "pred_rel_type": pred_type or "NON_MATCH",
            "gold_rel_type": gold_type or "NON_MATCH",
            "eligible":      bool(pred_norm and gold_norm),
            "pred_untyped":  bool(gold_norm and not pred_norm),
            "hier_score":    round(sc, 4) if sc is not None else None})

    return {
        **base,
        "gold_type_list":  gold_type_list,
        "n_llm_matched":   llm_full_tp,
        "n_eligible":      type_n,
        "n_untyped_pred":  n_untyped_pred,
        "type_hier":       _prf(type_score, type_n, type_n),
        "type_log":        type_log}


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(results: List[dict], llm_tag: str) -> None:
    sid_w = 32; SEP = "  │"
    COL   = "  Score  Elig Untyp  Cov   Acc"
    TITLE = "RELATION TYPE MATCHING — HIERARCHICAL (subClassOf: exact=1.0 / ±1=0.6 / ±2=0.3)"
    W     = max(sid_w + 2 + len(COL), len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)
    print(f"  {'Sample':<{sid_w}}{SEP}{'── LLM-Full matched pairs ──':^{len(COL)}}")
    print(f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Untyp':>6}{'Cov':>5}{'Acc':>6}")
    print(f"  {'':>{sid_w}}{SEP}{'':>7}{'':>6}{'(pred∅)':>6}{'':>5}{'':>6}")
    print("─" * W)

    for r in results:
        sid     = r["id"][:sid_w]
        m       = r.get("type_hier", {})
        sc      = m.get("tp", 0.0)
        eli     = m.get("predicted", 0)
        untyp   = r.get("n_untyped_pred", 0)
        typeable = eli + untyp
        cov     = eli / typeable if typeable > 0 else 0.0
        acc     = m.get("f1", 0.0)
        print(f"  {sid:<{sid_w}}{SEP}  {sc:>6.2f}  {eli:>4}  {untyp:>4}  {cov:>4.2f}  {acc:>5.4f}")

    print("─" * W)
    all_m        = [r.get("type_hier", {}) for r in results]
    tot_sc       = sum(m.get("tp", 0.0) for m in all_m)
    tot_eli      = sum(m.get("predicted", 0) for m in all_m)
    tot_untyp    = sum(r.get("n_untyped_pred", 0) for r in results)
    tot_typeable = tot_eli + tot_untyp
    micro_acc    = tot_sc / tot_eli if tot_eli > 0 else 0.0
    micro_cov    = tot_eli / tot_typeable if tot_typeable > 0 else 0.0
    macro_acc    = sum(m.get("f1", 0.0) for m in all_m) / len(all_m) if all_m else 0.0

    print()
    print(f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Untyp':>6}{'Cov':>5}{'Acc':>6}")
    print(f"  {'Micro total':<{sid_w}}{SEP}  {tot_sc:>6.2f}  {tot_eli:>4}  {tot_untyp:>4}  {micro_cov:>4.2f}  {micro_acc:>5.4f}")
    print(f"  {'Macro-Acc  ':<{sid_w}}{SEP}  {'':>6}  {'':>4}  {'':>4}  {'':>4}  {macro_acc:>5.4f}")
    print(f"\n  Elig   = matched pairs where BOTH pred and gold have a property type")
    print(f"  Untyp  = matched pairs where gold has a type but pred property is empty/unknown")
    print(f"  Cov    = Elig / (Elig + Untyp)  — type coverage rate")
    print(f"  Acc    = Score / Elig  (partial credits: 1.0/0.6/0.3 by subClassOf distance)")
    print("=" * W)


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relation Type Matching Evaluation (hierarchical subClassOf scoring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",          required=True,
                        help="Output directory (one model+schema) or single JSON file")
    parser.add_argument("--ground-truth",     required=True, help="GT root directory (ground_truth/)")
    parser.add_argument("--ontology",         default=None, choices=sorted(SUPPORTED_ONTOLOGIES) + ["none"],
                        help="Override ontology (default: infer from dir name / file contents)")
    parser.add_argument("--llm-provider",     default=None)
    parser.add_argument("--llm-model",        default=None)
    parser.add_argument("--llm-base-url",     default=None)
    parser.add_argument("--llm-timeout",      type=float, default=180.0)
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--output",           default=None)
    # Kept for API compatibility with evaluate_all.py
    parser.add_argument("--include-implicit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    _lp = os.getenv("LLM_PROVIDER", "openai").lower()
    if _lp == "vllm":
        _default_provider = "openai"
        _default_model    = os.getenv("VLLM_LLM_MODEL", "gpt-4o-mini")
        _default_base_url = os.getenv("VLLM_BASE_URL")
    elif _lp in ("anthropic", "claude"):
        _default_provider = "anthropic"
        _default_model    = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-6")
        _default_base_url = None
    else:
        _default_provider = "openai"
        _default_model    = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        _default_base_url = None
    llm_provider = args.llm_provider or _default_provider
    llm_model    = args.llm_model    or _default_model
    llm_base_url = args.llm_base_url or _default_base_url or None
    llm_tag      = f"{llm_provider}/{llm_model}"

    results_path = Path(args.results)

    # ── Load output items ──
    if results_path.is_dir():
        sample_files = sorted(
            f for f in results_path.glob("*.json") if not f.name.startswith("eval_") and not f.stem.endswith("_results")
        )
        if not sample_files:
            raise ValueError(f"No JSON files found in: {results_path}")

        items_with_slugs: List[Tuple[str, dict]] = []
        for f in sample_files:
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                items_with_slugs.append((_extract_output_slug(f.stem), data))
            except Exception as e:
                print(f"[!] Error loading {f.name}: {e}")

        key = results_path.name
        out_dir = Path(args.output) if args.output else results_path.parent / "eval_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"eval_triple_typing_{key}.json"

        resolved_ont = args.ontology or _infer_ontology_from_dir(results_path)
        if resolved_ont == 'none' and items_with_slugs:
            resolved_ont = _normalize_ontology(items_with_slugs[0][1].get("ontology", "none"))

        print(f"[*] Results dir  : {results_path}  ({len(sample_files)} files)")
        print(f"[*] Output       : {output_path}")
    else:
        with open(results_path, encoding="utf-8") as fp:
            raw = json.load(fp)
        if isinstance(raw, list):
            items_with_slugs = [(
                _extract_output_slug(Path(item.get("file", results_path.stem)).stem), item,
            ) for item in raw]
            resolved_ont = args.ontology or _normalize_ontology(
                raw[0].get("ontology", "") if raw else ""
            )
        else:
            items_with_slugs = [(_extract_output_slug(results_path.stem), raw)]
            resolved_ont = args.ontology or _normalize_ontology(raw.get("ontology", ""))

        key = results_path.stem
        output_path = Path(args.output) if args.output else None
        print(f"[*] Results      : {args.results}")

    # Skip if ontology is not uco/stix/malont
    if resolved_ont not in SUPPORTED_ONTOLOGIES:
        print(f"[skip] ontology='{resolved_ont}' — relation type matching requires uco/stix/malont")
        return

    print(f"[*] Ground truth : {args.ground_truth}")
    print(f"[*] Ontology     : {resolved_ont}")
    print(f"[*] LLM judge    : {llm_tag}")
    if llm_base_url:
        print(f"[*] LLM endpoint : {llm_base_url}")
    if args.limit:
        print(f"[*] Limit        : {args.limit} samples")

    gt_map = _load_gt(args.ground_truth, resolved_ont)
    llm    = _build_llm(llm_provider, llm_model, llm_base_url)

    items_to_process = items_with_slugs[:args.limit] if args.limit else items_with_slugs
    results: List[dict] = []
    skipped = 0

    for i, (slug, item) in enumerate(items_to_process):
        if "error" in item:
            skipped += 1; continue

        gt = gt_map.get(slug)
        if not gt:
            print(f"  [!] No GT for slug '{slug}'", flush=True)
            skipped += 1; continue

        print(f"\n[{i+1}/{len(items_to_process)}] {slug}", flush=True)
        result = await evaluate_sample(item, gt, llm, resolved_ont, args.llm_timeout)
        results.append(result)
        m = result["type_hier"]
        print(
            f"  LLM matched={result['n_llm_matched']}  eligible={result['n_eligible']}"
            f"  TYPE hier Acc={m['f1']:.3f}  [Score={m['tp']:.1f}/{m['predicted']}]"
        )

    if skipped:
        print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")
    if not results:
        print("[!] No results to report.")
        return

    _buf = io.StringIO()
    _old, sys.stdout = sys.stdout, _buf
    _print_report(results, llm_tag)
    sys.stdout = _old
    _table_str = _buf.getvalue()
    print(_table_str, end="")

    all_m     = [r.get("type_hier", {}) for r in results]
    tot_sc    = sum(m.get("tp", 0.0) for m in all_m)
    tot_eli   = sum(m.get("predicted", 0) for m in all_m)
    micro_acc = tot_sc / tot_eli if tot_eli > 0 else 0.0
    macro_acc = sum(m.get("f1", 0.0) for m in all_m) / len(all_m) if all_m else 0.0

    if output_path:
        out = {
            "report_table": _table_str,
            "task": "triple_typing", "results_dir": str(results_path),
            "results_key": key, "ground_truth": args.ground_truth,
            "ontology": resolved_ont, "llm": llm_tag,
            "num_samples": len(results), "skipped": skipped,
            "micro_acc": round(micro_acc, 4), "macro_acc": round(macro_acc, 4),
            "n_eligible": sum(r.get("n_eligible", 0) for r in results),
            "n_untyped":  sum(r.get("n_untyped_pred", 0) for r in results),
            "type_hier": _aggregate(all_m),
            "samples": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
