#!/usr/bin/env python3
"""
evaluate_entity_extraction.py — Entity Extraction Evaluation

Compares extracted entities against ground truth using:
  1. Jaccard similarity  (token-overlap, greedy matching)
  2. Embedding similarity (sentence-transformers, greedy cosine matching)
  3. LLM-as-a-Judge      (one batch LLM call per sample)

Ground truth layout expected:
  ground_truth/{ontology}/json/{slug}_{ontology}.json
  Each file contains: {"extracted_entities": [{"name": "...", "class": "..."}, ...], ...}

Output layout expected (one dir = one model+schema combo):
  output/{model_schema}/{prefix}_{slug}_vllm_{model}_{ontology}.json

Usage:
  # Directory mode (one model+schema directory):
  python evaluate_entity_extraction.py \\
      --results  output/vllm_qwen3.5-35b_uco/ \\
      --ground-truth ground_truth/

  # Single file:
  python evaluate_entity_extraction.py \\
      --results  output/vllm_qwen3.5-35b_uco/some_file.json \\
      --ground-truth ground_truth/

  # Human-in-the-loop review:
  python evaluate_entity_extraction.py ... --hitl
"""

import asyncio
import argparse
import io
import json
import os
import re
import sys
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
    """Remove _{ontology} suffix from GT filename stem."""
    return re.sub(r'_(uco|stix|malont)$', '', stem)


def _extract_output_slug(stem: str) -> str:
    """Extract article slug from output filename stem.

    Handles multiple naming conventions:
      {prefix}_{slug}_vllm_{model}_{ontology}   (anchor/vllm)
      {prefix}_{slug}_ttpdrill_{ontology}        (TTPDrill baseline)
      {prefix}_{slug}_ctinexus_{ontology}        (CTINexus baseline)
      {prefix}_{slug}_gtikg_{ontology}           (GTIKG baseline)

    Strategy:
      1. Strip trailing schema token  (_uco | _stix | _malont | _none)
      2. Strip trailing method token  (_vllm_* | _ttpdrill | _ctinexus | _gtikg | ...)
      3. Strip leading dataset prefix (ctinexus_ | gtikg_ | ttpdrill_ | ...)
    """
    # 1. Strip schema suffix
    stem = re.sub(r'_(uco|stix|malont|none)$', '', stem)

    # 2. Strip method/provider suffix
    #    _vllm_{model} — cut everything from _vllm_ onward
    idx = stem.find('_vllm_')
    if idx != -1:
        stem = stem[:idx]
    else:
        # Other known single-token method suffixes
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
    """Infer ontology from output directory name (e.g. 'vllm_qwen3.5-35b_uco' → 'uco')."""
    name = dirpath.name.lower()
    for ont in ('malont', 'stix', 'uco'):   # longer names first
        if name.endswith(f'_{ont}'):
            return ont
    return 'none'


# ── GT loader ──────────────────────────────────────────────────────────────────

def _load_gt(gt_base: str, ontology: str) -> dict:
    """Load ground truth indexed by article slug.

    Looks for ground_truth/{ontology}/json/ first; falls back to gt_base directly.
    When ontology is 'none', auto-discovers the first available schema subdirectory
    (e.g. ground_truth/stix/json/) so schema-free outputs can still be evaluated.
    Returns {slug: {"id", "text", "ground_truth_entities": [{"name"}]}}
    """
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
                    data = json.load(fp)
            except Exception as e:
                print(f"[!] Could not read GT file {f.name}: {e}")
                continue
            slug = _extract_gt_slug(f.stem)
            out[slug] = {
                "id":   slug,
                "text": data.get("text", ""),
                "ground_truth_entities": [
                    {"name": e["name"].strip()}
                    for e in data.get("extracted_entities", [])
                    if e.get("name")
                ],
            }
        return out

    samples = _read_dir(gt_dir)

    # When ontology is 'none', gt_dir falls back to the base directory which may
    # contain no JSON files directly (they live in schema subdirs like stix/json/).
    # Auto-discover the first available schema subdir and use it as GT.
    if not samples and gt_dir == base:
        for schema in ('stix', 'uco', 'malont'):
            sub = base / schema / "json"
            if sub.is_dir():
                samples = _read_dir(sub)
                if samples:
                    print(f"[*] ontology=none — using GT from '{schema}' schema ({sub})")
                    break

    if not samples:
        raise ValueError(f"No GT JSON files found under: {gt_dir}")
    return samples


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _prf(tp, predicted: int, gold: int) -> dict:
    tp_f = float(tp)
    p  = tp_f / predicted if predicted > 0 else 0.0
    r  = tp_f / gold      if gold      > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "predicted": predicted, "gold": gold}


def _aggregate(metrics_list: List[dict]) -> dict:
    if not metrics_list:
        return {}
    result: dict = {}
    for k in ["p", "r", "f1"]:
        vals = [m[k] for m in metrics_list if k in m]
        result[f"macro_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    tp   = sum(m.get("tp",        0) for m in metrics_list)
    pred = sum(m.get("predicted", 0) for m in metrics_list)
    gold = sum(m.get("gold",      0) for m in metrics_list)
    micro = _prf(tp, pred, gold)
    result["micro_p"]  = micro["p"]
    result["micro_r"]  = micro["r"]
    result["micro_f1"] = micro["f1"]
    result["micro_tp"] = tp
    result["micro_fp"] = pred - tp
    result["micro_fn"] = gold - tp
    return result


SUPPORTED_ONTOLOGIES = {"uco", "malont", "stix"}


def _normalize_ontology(ontology: str) -> str:
    return (ontology or "").strip().lower()


def _resolve_ontology(item: dict, override: str = None) -> str:
    return _normalize_ontology(override or item.get("ontology") or "none")


# ── Matchers ──────────────────────────────────────────────────────────────────

class EmbeddingMatcher:
    def __init__(self, threshold: float = 0.75, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print(f"[emb] Loading '{model_name}'...", flush=True)
        self.model     = SentenceTransformer(model_name)
        self.threshold = threshold
        self._np       = np

    def _cos(self, a, b) -> float:
        np = self._np
        d  = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

    def score_matrix(self, preds: List[str], golds: List[str]) -> List[List[float]]:
        if not preds or not golds:
            return []
        embs = self.model.encode(preds + golds, show_progress_bar=False)
        pe, ge = embs[:len(preds)], embs[len(preds):]
        return [[self._cos(p, g) for g in ge] for p in pe]

    def greedy_match(self, matrix: List[List[float]], threshold: float = None) -> Tuple[int, List[Tuple[int, int, float]]]:
        t = threshold if threshold is not None else self.threshold
        if not matrix:
            return 0, []
        pairs = sorted(
            [(i, j, matrix[i][j])
             for i in range(len(matrix)) for j in range(len(matrix[i]))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= t and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc))
                used_p.add(pi); used_g.add(gj)
        return len(matched), matched


class JaccardMatcher:
    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    @staticmethod
    def _tokens(s: str) -> set:
        return set(re.sub(r"[^\w]+", " ", s.lower()).split())

    def similarity(self, a: str, b: str) -> float:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta and not tb: return 1.0
        if not ta or  not tb: return 0.0
        return len(ta & tb) / len(ta | tb)

    def greedy_match(self, preds: List[str], golds: List[str]) -> Tuple[int, List[Tuple[int, int, float]]]:
        if not preds or not golds:
            return 0, []
        pairs = sorted(
            [(i, j, self.similarity(preds[i], golds[j]))
             for i in range(len(preds)) for j in range(len(golds))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= self.threshold and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc))
                used_p.add(pi); used_g.add(gj)
        return len(matched), matched


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


async def _llm_batch_match_entities(preds: List[str], golds: List[str], llm, timeout: float = 180.0) -> List[Tuple[int, int]]:
    from langchain_core.messages import SystemMessage, HumanMessage
    if not preds or not golds:
        return []
    pred_lines = "\n".join(f"{i+1}. {x}" for i, x in enumerate(preds))
    gold_lines = "\n".join(f"{j+1}. {x}" for j, x in enumerate(golds))
    prompt = (
        "Match each predicted entity name to the semantically equivalent gold entity name.\n"
        "Rules: Same entity if they refer to the same real-world entity "
        "(abbreviations, different phrasings, and added qualifiers are OK).\n"
        "Each gold entity can be matched at most once. Only include confident matches.\n\n"
        f"Predicted:\n{pred_lines}\n\nGold:\n{gold_lines}\n\n"
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
        print(f"  [!] LLM timeout ({timeout:.0f}s) for entity matching — returning []", flush=True)
        return []
    try:
        data = _parse_llm_pairs(resp.content)
        if not data and resp.content.strip():
            try:
                json.loads(resp.content.strip())
            except Exception:
                print(f"  [!] LLM response unparseable: {resp.content[:200]!r}", flush=True)
        pairs = [
            (int(d["pred"]) - 1, int(d["gold"]) - 1)
            for d in data
            if "pred" in d and "gold" in d and d["pred"] is not None and d["gold"] is not None
        ]
        return [(pi, gj) for pi, gj in pairs if 0 <= pi < len(preds) and 0 <= gj < len(golds)]
    except Exception as e:
        print(f"  [!] LLM response parse error ({e}): {resp.content[:200]!r}", flush=True)
        return []


def _parse_llm_pairs(content: str) -> list:
    """Parse LLM response into [{pred, gold}] handling array, bare-object, multi-object, or parallel-arrays formats."""
    # Try proper JSON array first
    m = re.search(r'\[\s*(?:\{[^{}\[\]]*\}(?:\s*,\s*\{[^{}\[\]]*\})*\s*)?\]', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Fall back: extract every {pred/gold} object individually (handles bare or multi-object responses)
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


def _count_llm_tp(pairs: List[Tuple[int, int]], n_pred: int, n_gold: int) -> int:
    used_g: set = set()
    tp = 0
    for pi, gj in pairs:
        if 0 <= pi < n_pred and 0 <= gj < n_gold and gj not in used_g:
            tp += 1
            used_g.add(gj)
    return tp


# ── HITL ──────────────────────────────────────────────────────────────────────

async def hitl_match_entities(
    pred_objs: List[dict], gold_objs: List[dict],
    emb: EmbeddingMatcher, llm, sample_id: str,
) -> dict:
    preds = [e["name"].strip() for e in pred_objs]
    golds = [e["name"].strip() for e in gold_objs]
    if not preds or not golds:
        return {"emb_tp": 0, "llm_tp": 0, "human_tp": 0, "match_log": []}

    matrix      = emb.score_matrix(preds, golds)
    emb_tp, _em = emb.greedy_match(matrix)
    llm_pairs   = await _llm_batch_match_entities(preds, golds, llm)
    llm_tp      = _count_llm_tp(llm_pairs, len(preds), len(golds))
    llm_match_set = {(pi, gj) for pi, gj in llm_pairs if 0 <= pi < len(preds) and 0 <= gj < len(golds)}

    _, low_cands = emb.greedy_match(matrix, threshold=0.35)
    cand_pred_ids = {pi for pi, _, _ in low_cands}
    for pi, gj in llm_pairs:
        if pi not in cand_pred_ids and 0 <= pi < len(preds) and 0 <= gj < len(golds):
            low_cands.append((pi, gj, matrix[pi][gj] if pi < len(matrix) else 0.0))
            cand_pred_ids.add(pi)
    low_cands.sort(key=lambda x: x[2], reverse=True)

    W = 60
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │{'HITL · Entity Extraction Matching':^{W}}│")
    print(f"  │  sample : {sample_id[:W-12]:<{W-12}}│")
    print(f"  └{'─'*W}┘")
    print(f"  Commands: y/↵=match  n=no-match  <num>=pick gold  s=auto(llm)  a=auto-all  q=quit\n")
    print("  Gold entities:")
    for j, g in enumerate(golds):
        print(f"    {j+1:3d}. {g}")
    print()

    used_g: set = set()
    human_tp    = 0
    match_log   = []
    auto_rest   = False

    for idx, (pi, gj, emb_sc) in enumerate(low_cands):
        if gj in used_g:
            continue
        emb_ok   = emb_sc >= emb.threshold
        llm_ok   = (pi, gj) in llm_match_set
        disagree = emb_ok != llm_ok
        tag      = "  ← DISAGREE" if disagree else ""
        print(f"  [{idx+1}/{len(low_cands)}]  PREDICTED : \"{preds[pi]}\"")
        print(f"         GOLD [{gj+1:3d}]: \"{golds[gj]}\"")
        print(f"           Embedding : {emb_sc:.3f}  {'✓ MATCH' if emb_ok else '✗ no-match'}")
        print(f"           LLM Judge : {'✓ MATCH' if llm_ok else '✗ no-match'}{tag}")

        if auto_rest:
            is_match = llm_ok; decision = "auto(llm)"
        else:
            try:
                choice = input("  Decision [y/↵/n/<num>/s/a/q]: ").strip().lower()
            except EOFError:
                choice = "s"
            if choice == "q":
                break
            elif choice == "a":
                auto_rest = True; is_match = llm_ok; decision = "auto(llm)"
            elif choice == "s":
                is_match = llm_ok; decision = "skip→llm"
            elif choice == "n":
                is_match = False; decision = "human:n"
            elif choice.isdigit():
                new_gj = int(choice) - 1
                if 0 <= new_gj < len(golds) and new_gj not in used_g:
                    gj = new_gj; is_match = True; decision = f"human→gold[{new_gj+1}]"
                else:
                    is_match = False; decision = "invalid"
            else:
                is_match = True; decision = "human:y"

        if is_match:
            human_tp += 1; used_g.add(gj)
        match_log.append({
            "pred": preds[pi], "gold": golds[gj], "gold_idx": gj,
            "emb_score": round(emb_sc, 4), "emb_match": emb_ok,
            "llm_match": llm_ok, "decision": decision, "matched": is_match,
        })
        print()

    return {"emb_tp": emb_tp, "llm_tp": llm_tp, "human_tp": human_tp, "match_log": match_log}


# ── Per-sample evaluation ──────────────────────────────────────────────────────

async def evaluate_sample(
    item: dict, gt: dict, jac: JaccardMatcher, emb: EmbeddingMatcher, llm,
    hitl: bool, ontology_override: str = None, llm_timeout: float = 180.0,
) -> dict:
    ontology  = _resolve_ontology(item, ontology_override)
    pred_objs = [e for e in item.get("extracted_entities", []) if e.get("name")]
    gold_objs = [e for e in gt.get("ground_truth_entities",  []) if e.get("name")]
    preds     = [e["name"].strip() for e in pred_objs]
    golds     = [e["name"].strip() for e in gold_objs]

    base = {"id": gt["id"], "ontology": ontology, "n_pred": len(preds), "n_gold": len(golds)}
    zero = _prf(0, len(preds), len(golds))

    if not preds or not golds:
        return {**base, "gold_list": golds, "missed_gold": golds,
                "jaccard": zero, "emb": zero, "llm": zero}

    if hitl:
        counts  = await hitl_match_entities(pred_objs, gold_objs, emb, llm, gt["id"])
        log     = counts["match_log"]
        matched_g = {l["gold_idx"] for l in log if l.get("matched") and "gold_idx" in l}
        missed  = [golds[j] for j in range(len(golds)) if j not in matched_g]
        return {
            **base, "gold_list": golds, "missed_gold": missed,
            "emb":   _prf(counts["emb_tp"],   len(preds), len(golds)),
            "llm":   _prf(counts["llm_tp"],   len(preds), len(golds)),
            "human": _prf(counts["human_tp"], len(preds), len(golds)),
            "match_log": log,
        }
    else:
        jac_tp, jac_matches = jac.greedy_match(preds, golds)
        matrix              = emb.score_matrix(preds, golds)
        emb_tp, emb_matches = emb.greedy_match(matrix)
        llm_pairs           = await _llm_batch_match_entities(preds, golds, llm, llm_timeout)
        llm_tp              = _count_llm_tp(llm_pairs, len(preds), len(golds))

        jac_matched_p = {p for p, g, s in jac_matches}
        emb_matched_p = {p for p, g, s in emb_matches}
        llm_matched_p = {p for p, g in llm_pairs}
        llm_matched_g = {g for p, g in llm_pairs}

        match_log = []
        for pi, p_obj in enumerate(pred_objs):
            jac_gj   = next((g for p, g, s in jac_matches if p == pi), None)
            emb_info = next(((g, s) for p, g, s in emb_matches if p == pi), (None, 0.0))
            llm_gj   = next((g for p, g in llm_pairs if p == pi), None)
            match_log.append({
                "prediction": p_obj["name"],
                "jaccard":    {"matched": pi in jac_matched_p, "gold": golds[jac_gj] if jac_gj is not None else None},
                "emb":        {"matched": pi in emb_matched_p, "gold": golds[emb_info[0]] if emb_info[0] is not None else None, "score": round(emb_info[1], 4)},
                "llm":        {"matched": pi in llm_matched_p, "gold": golds[llm_gj] if llm_gj is not None else None},
            })

        missed = [golds[j] for j in range(len(golds)) if j not in llm_matched_g]
        return {
            **base, "gold_list": golds, "missed_gold": missed,
            "jaccard": _prf(jac_tp, len(preds), len(golds)),
            "emb":     _prf(emb_tp, len(preds), len(golds)),
            "llm":     _prf(llm_tp, len(preds), len(golds)),
            "match_log": match_log,
        }


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(
    results: List[dict], hitl: bool,
    jac_threshold: float, emb_threshold: float, llm_tag: str,
) -> None:
    has_human = hitl and any("human" in r for r in results)
    ext_keys  = ["jaccard", "emb", "llm"] + (["human"] if has_human else [])
    ext_labels = {
        "jaccard": f"Jaccard(≥{jac_threshold})",
        "emb":     f"Embedding(≥{emb_threshold})",
        "llm":     f"LLM({llm_tag})",
        "human":   "Human",
    }

    sid_w = 28
    COL   = "  TP   FP   FN     F1"
    SEP   = "  │"
    W     = sid_w + sum(len(SEP) + len(COL) for _ in ext_keys)
    TITLE = "ENTITY EXTRACTION — EVALUATION REPORT"
    W     = max(W, len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)

    tag_hdr = f"  {'Sample':<{sid_w}}" + "".join(f"{SEP}{ext_labels[k]:^{len(COL)}}" for k in ext_keys)
    col_hdr = f"  {'':>{sid_w}}" + "".join(f"{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}" for _ in ext_keys)
    print(tag_hdr); print(col_hdr); print("─" * W)

    for r in results:
        sid  = r["id"][:sid_w]
        line = f"  {sid:<{sid_w}}"
        for k in ext_keys:
            m = r.get(k, {}); tp = m.get("tp", 0); fp = m.get("predicted", 0) - tp; fn = m.get("gold", 0) - tp
            line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {m.get('f1', 0.0):>6.4f}"
        print(line)

    print("─" * W)
    aggs = {k: _aggregate([r.get(k, {}) for r in results]) for k in ext_keys}
    print()
    print(f"  {'':>{sid_w}}" + "".join(f"{SEP}{'  TP   FP   FN  Micro-F1':^{len(COL)}}" for _ in ext_keys))
    micro_line = f"  {'Micro total':<{sid_w}}"
    macro_line = f"  {'Macro-F1   ':<{sid_w}}"
    pr_line    = f"  {'P / R      ':<{sid_w}}"
    for k in ext_keys:
        a = aggs[k]
        tp = a.get("micro_tp", 0); fp = a.get("micro_fp", 0); fn = a.get("micro_fn", 0)
        micro_line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {a.get('micro_f1', 0.0):>6.4f}"
        macro_line += f"{SEP}  {'':>4}  {'':>4}  {'':>4}  {a.get('macro_f1', 0.0):>6.4f}"
        pr_line    += f"{SEP}  P={a.get('micro_p', 0.0):.4f}  R={a.get('micro_r', 0.0):.4f}{'':>7}"
    print(micro_line); print(macro_line); print(pr_line)
    print("=" * W)


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entity Extraction Evaluation (Jaccard + Embedding + LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",       required=True,
                        help="Output directory (one model+schema) or single JSON file")
    parser.add_argument("--ground-truth",  required=True, help="GT root directory (ground_truth/)")
    parser.add_argument("--emb-threshold", type=float, default=0.75)
    parser.add_argument("--jac-threshold", type=float, default=0.2)
    parser.add_argument("--ontology",      default=None, choices=sorted(SUPPORTED_ONTOLOGIES) + ["none"],
                        help="Override ontology (default: infer from dir name / file contents)")
    parser.add_argument("--llm-provider",  default=None)
    parser.add_argument("--llm-model",     default=None)
    parser.add_argument("--llm-base-url",  default=None)
    parser.add_argument("--llm-timeout",   type=float, default=180.0)
    parser.add_argument("--hitl",          action="store_true")
    parser.add_argument("--limit",         type=int, default=None)
    parser.add_argument("--output",        default=None,
                        help="Output JSON path (single file) or directory")
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
            f for f in results_path.glob("*.json")
            if not f.name.startswith("eval_") and not f.stem.endswith("_results")
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
        output_path = out_dir / f"eval_entity_ext_{key}.json"

        resolved_ont = args.ontology or _infer_ontology_from_dir(results_path)
        if resolved_ont == 'none' and items_with_slugs:
            resolved_ont = _normalize_ontology(items_with_slugs[0][1].get("ontology", "none"))

        print(f"[*] Results dir  : {results_path}  ({len(sample_files)} files)")
        print(f"[*] Output       : {output_path}")
    else:
        # Single file mode
        with open(results_path, encoding="utf-8") as fp:
            raw = json.load(fp)
        if isinstance(raw, list):
            items_with_slugs = [(
                _extract_output_slug(Path(item.get("file", results_path.stem)).stem),
                item,
            ) for item in raw]
            resolved_ont = args.ontology or _normalize_ontology(
                raw[0].get("ontology", "none") if raw else "none"
            )
        else:
            items_with_slugs = [(_extract_output_slug(results_path.stem), raw)]
            resolved_ont = args.ontology or _normalize_ontology(raw.get("ontology", "none"))

        key = results_path.stem
        output_path = Path(args.output) if args.output else None
        print(f"[*] Results      : {args.results}")

    print(f"[*] Ground truth : {args.ground_truth}")
    print(f"[*] Ontology     : {resolved_ont}")
    print(f"[*] Emb threshold: {args.emb_threshold}")
    print(f"[*] Jac threshold: {args.jac_threshold}")
    print(f"[*] LLM judge    : {llm_tag}")
    if llm_base_url:
        print(f"[*] LLM endpoint : {llm_base_url}")
    print(f"[*] HITL         : {'ON' if args.hitl else 'OFF'}")
    if args.limit:
        print(f"[*] Limit        : {args.limit} samples")

    gt_map = _load_gt(args.ground_truth, resolved_ont)
    jac    = JaccardMatcher(threshold=args.jac_threshold)
    emb    = EmbeddingMatcher(threshold=args.emb_threshold)
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
        result = await evaluate_sample(
            item, gt, jac, emb, llm, args.hitl, args.ontology, args.llm_timeout
        )
        results.append(result)

        if not args.hitl:
            print(
                f"  JAC F1={result['jaccard']['f1']:.3f}"
                f"  EMB F1={result['emb']['f1']:.3f}"
                f"  LLM F1={result['llm']['f1']:.3f}"
                f"  (pred={result['n_pred']}, gold={result['n_gold']})"
            )
        else:
            h = result.get("human", result["llm"])
            print(f"  EMB F1={result['emb']['f1']:.3f}  LLM F1={result['llm']['f1']:.3f}"
                  f"  HUMAN F1={h['f1']:.3f}")

    if skipped:
        print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")
    if not results:
        print("[!] No results to report.")
        return

    _buf = io.StringIO()
    _old, sys.stdout = sys.stdout, _buf
    _print_report(results, args.hitl, args.jac_threshold, args.emb_threshold, llm_tag)
    sys.stdout = _old
    _table_str = _buf.getvalue()
    print(_table_str, end="")

    if output_path:
        out = {
            "report_table": _table_str,
            "task": "entity_extraction", "results_dir": str(results_path),
            "results_key": key, "ground_truth": args.ground_truth,
            "emb_threshold": args.emb_threshold, "jac_threshold": args.jac_threshold,
            "ontology": resolved_ont, "llm": llm_tag, "hitl": args.hitl,
            "num_samples": len(results), "skipped": skipped,
            "jaccard":   _aggregate([r["jaccard"] for r in results if "jaccard" in r]),
            "embedding": _aggregate([r["emb"]     for r in results]),
            "llm_judge": _aggregate([r["llm"]     for r in results]),
            "samples": results,
        }
        if args.hitl:
            human_list = [r["human"] for r in results if "human" in r]
            if human_list:
                out["human_hitl"] = _aggregate(human_list)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
