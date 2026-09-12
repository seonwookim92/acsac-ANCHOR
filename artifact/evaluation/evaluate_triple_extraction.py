#!/usr/bin/env python3
"""
evaluate_triple_extraction.py — Triple Extraction Evaluation

Evaluates extracted triples against ground truth using:

  Both modes always run:
    Soft (S+O)  : Jaccard / Embedding / LLM  — subject+object only
    Full (S+R+O): Jaccard / Embedding / LLM  — full triple string

Ground truth layout expected:
  ground_truth/{ontology}/json/{slug}_{ontology}.json
  Each file contains: {"extracted_triplets": [{"subject","relation","object",...}, ...], ...}

Usage:
  python evaluate_triple_extraction.py \\
      --results  output/vllm_qwen3.5-35b_uco/ \\
      --ground-truth ground_truth/

  # Schema-agnostic (gtikg):
  python evaluate_triple_extraction.py \\
      --results  output/gtikg_vllm_qwen3.5-35b/ \\
      --ground-truth ground_truth/ \\
      --ontology uco
"""

import asyncio
import argparse
import json
import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    """Load ground truth triples indexed by slug."""
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
            out[slug] = {
                "id": slug,
                "ground_truth_triples": [
                    {
                        "subject":  t.get("subject", ""),
                        "relation": t.get("relation", ""),
                        "object":   t.get("object", ""),
                    }
                    for t in raw.get("extracted_triplets", [])
                ],
            }
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


# ── Triple string helpers ──────────────────────────────────────────────────────

def _triple_soft(t: dict) -> str:
    return f"{t.get('subject','').strip()} [SEP] {t.get('object','').strip()}"

def _triple_full(t: dict) -> str:
    return f"{t.get('subject','').strip()} [SEP] {t.get('relation','').strip()} [SEP] {t.get('object','').strip()}"

def _triple_display(t: dict) -> str:
    return f"({t.get('subject','?')}, {t.get('relation','?')}, {t.get('object','?')})"


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
        np = self._np; d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

    def score_matrix(self, preds: List[str], golds: List[str]) -> List[List[float]]:
        if not preds or not golds: return []
        embs = self.model.encode(preds + golds, show_progress_bar=False)
        pe, ge = embs[:len(preds)], embs[len(preds):]
        return [[self._cos(p, g) for g in ge] for p in pe]

    def greedy_match(self, matrix: List[List[float]], threshold: float = None) -> Tuple[int, List[Tuple[int, int, float]]]:
        t = threshold if threshold is not None else self.threshold
        if not matrix: return 0, []
        pairs = sorted(
            [(i, j, matrix[i][j]) for i in range(len(matrix)) for j in range(len(matrix[i]))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= t and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc)); used_p.add(pi); used_g.add(gj)
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
        if not preds or not golds: return 0, []
        pairs = sorted(
            [(i, j, self.similarity(preds[i], golds[j])) for i in range(len(preds)) for j in range(len(golds))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= self.threshold and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc)); used_p.add(pi); used_g.add(gj)
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


async def _llm_batch_match_triples(
    pred_triples: List[dict], gold_triples: List[dict], mode: str, llm,
    timeout: float = 180.0,
) -> List[Tuple[int, int]]:
    from langchain_core.messages import SystemMessage, HumanMessage
    if not pred_triples or not gold_triples: return []

    if mode == "soft":
        task = (
            "Match each predicted (Subject, Object) pair to the semantically equivalent "
            "gold (Subject, Object) pair.\n"
            "Rules: Entities match if they refer to the same real-world thing. "
            "Ignore relation differences. Each gold at most once."
        )
        pred_lines = "\n".join(f"{i+1}. ({t.get('subject','?')}, {t.get('object','?')})" for i, t in enumerate(pred_triples))
        gold_lines = "\n".join(f"{j+1}. ({t.get('subject','?')}, {t.get('object','?')})" for j, t in enumerate(gold_triples))
    else:
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
        print(f"  [!] LLM timeout ({timeout:.0f}s) for {mode} matching — returning []", flush=True)
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


def _count_llm_tp(pairs: List[Tuple[int, int]], n_pred: int, n_gold: int) -> int:
    used_g: set = set(); tp = 0
    for pi, gj in pairs:
        if 0 <= pi < n_pred and 0 <= gj < n_gold and gj not in used_g:
            tp += 1; used_g.add(gj)
    return tp


# ── Per-sample evaluation ──────────────────────────────────────────────────────

async def evaluate_sample(
    item: dict, gt: dict, jac: JaccardMatcher, emb: EmbeddingMatcher, llm,
    ontology_override: str = None, llm_timeout: float = 180.0,
) -> dict:
    ontology = _resolve_ontology(item, ontology_override)
    preds = [
        {k: t.get(k, "") for k in ("subject", "relation", "object")}
        for t in item.get("extracted_triplets", [])
        if t.get("subject") or t.get("object")
    ]
    golds = list(gt.get("ground_truth_triples", []))

    base = {
        "id": gt["id"], "ontology": ontology,
        "n_pred": len(preds), "n_gold": len(golds),
    }
    zero = _prf(0, len(preds), len(golds))

    if not preds or not golds:
        return {
            **base,
            "gold_list": [_triple_full(t) for t in golds],
            "missed_gold": [_triple_full(t) for t in golds],
            "soft": {"jaccard": zero, "emb": zero, "llm": zero},
            "full": {"jaccard": zero, "emb": zero, "llm": zero},
        }

    soft_preds = [_triple_soft(t) for t in preds]; soft_golds = [_triple_soft(t) for t in golds]
    full_preds = [_triple_full(t) for t in preds]; full_golds = [_triple_full(t) for t in golds]

    jac_soft_tp, _ = jac.greedy_match(soft_preds, soft_golds)
    jac_full_tp, _ = jac.greedy_match(full_preds, full_golds)
    soft_matrix    = emb.score_matrix(soft_preds, soft_golds)
    full_matrix    = emb.score_matrix(full_preds, full_golds)
    emb_soft_tp, _ = emb.greedy_match(soft_matrix)
    emb_full_tp, _ = emb.greedy_match(full_matrix)
    llm_soft_pairs = await _llm_batch_match_triples(preds, golds, "soft", llm, timeout=llm_timeout)
    llm_full_pairs = await _llm_batch_match_triples(preds, golds, "full", llm, timeout=llm_timeout)
    llm_soft_tp = _count_llm_tp(llm_soft_pairs, len(preds), len(golds))
    llm_full_tp = _count_llm_tp(llm_full_pairs, len(preds), len(golds))

    llm_full_matched_g = {gj for _, gj in llm_full_pairs}
    missed = [full_golds[j] for j in range(len(full_golds)) if j not in llm_full_matched_g]

    return {
        **base,
        "gold_list":   full_golds,
        "missed_gold": missed,
        "soft": {
            "jaccard": _prf(jac_soft_tp, len(preds), len(golds)),
            "emb":     _prf(emb_soft_tp, len(preds), len(golds)),
            "llm":     _prf(llm_soft_tp, len(preds), len(golds)),
        },
        "full": {
            "jaccard": _prf(jac_full_tp, len(preds), len(golds)),
            "emb":     _prf(emb_full_tp, len(preds), len(golds)),
            "llm":     _prf(llm_full_tp, len(preds), len(golds)),
        },
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_extraction_table(
    results: List[dict], mode: str, mode_label: str,
    jac_threshold: float, emb_threshold: float, llm_tag: str,
) -> None:
    ext_keys   = ["jaccard", "emb", "llm"]
    ext_labels = {
        "jaccard": f"Jaccard(≥{jac_threshold})",
        "emb":     f"Emb(≥{emb_threshold})",
        "llm":     f"LLM({llm_tag})",
    }
    sid_w = 28; COL = "  TP   FP   FN     F1"; SEP = "  │"
    TITLE = f"TRIPLE EXTRACTION — {mode_label.upper()} — EVALUATION REPORT"
    W     = max(sid_w + sum(len(SEP) + len(COL) for _ in ext_keys), len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)
    print(f"  {'Sample':<{sid_w}}" + "".join(f"{SEP}{ext_labels[k]:^{len(COL)}}" for k in ext_keys))
    print(f"  {'':>{sid_w}}" + "".join(f"{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}" for _ in ext_keys))
    print("─" * W)

    for r in results:
        sid  = r["id"][:sid_w]; line = f"  {sid:<{sid_w}}"
        for k in ext_keys:
            m  = r.get(mode, {}).get(k, {}); tp = m.get("tp", 0); fp = m.get("predicted", 0) - tp; fn = m.get("gold", 0) - tp
            line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {m.get('f1', 0.0):>6.4f}"
        print(line)

    print("─" * W)
    aggs = {k: _aggregate([r.get(mode, {}).get(k, {}) for r in results]) for k in ext_keys}
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


def _print_report(
    results: List[dict], jac_threshold: float, emb_threshold: float, llm_tag: str,
) -> None:
    _print_extraction_table(results, "soft", "SOFT (S+O)",   jac_threshold, emb_threshold, llm_tag)
    _print_extraction_table(results, "full", "FULL (S+R+O)", jac_threshold, emb_threshold, llm_tag)


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triple Extraction Evaluation (Jaccard + Embedding + LLM, Soft + Full)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",          required=True,
                        help="Output directory (one model+schema) or single JSON file")
    parser.add_argument("--ground-truth",     required=True, help="GT root directory (ground_truth/)")
    parser.add_argument("--emb-threshold",    type=float, default=0.75)
    parser.add_argument("--jac-threshold",    type=float, default=0.2)
    parser.add_argument("--ontology",         default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology (required for schema-agnostic models like gtikg)")
    parser.add_argument("--llm-provider",     default=None)
    parser.add_argument("--llm-model",        default=None)
    parser.add_argument("--llm-base-url",     default=None)
    parser.add_argument("--llm-timeout",      type=float, default=180.0)
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--output",           default=None)
    # Kept for API compatibility with evaluate_all.py; no implicit triples in new GT format
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
        output_path = out_dir / f"eval_triple_ext_{key}.json"

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
    print(f"[*] Modes        : SOFT (S+O)  +  FULL (S+R+O)")
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
        result = await evaluate_sample(item, gt, jac, emb, llm, args.ontology, args.llm_timeout)
        results.append(result)

        s = result["soft"]; f_ = result["full"]
        print(f"  pred={result['n_pred']}  gold={result['n_gold']}")
        print(f"  SOFT  JAC F1={s['jaccard']['f1']:.3f}  EMB F1={s['emb']['f1']:.3f}  LLM F1={s['llm']['f1']:.3f}")
        print(f"  FULL  JAC F1={f_['jaccard']['f1']:.3f}  EMB F1={f_['emb']['f1']:.3f}  LLM F1={f_['llm']['f1']:.3f}")

    if skipped:
        print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")
    if not results:
        print("[!] No results to report.")
        return

    _buf = io.StringIO()
    _old, sys.stdout = sys.stdout, _buf
    _print_report(results, args.jac_threshold, args.emb_threshold, llm_tag)
    sys.stdout = _old
    _table_str = _buf.getvalue()
    print(_table_str, end="")

    def _micro_f1(mode, method):
        return _aggregate([r.get(mode, {}).get(method, {}) for r in results]).get("micro_f1", 0.0)

    if output_path:
        out = {
            "report_table": _table_str,
            "task": "triple_extraction", "results_dir": str(results_path),
            "results_key": key, "ground_truth": args.ground_truth,
            "emb_threshold": args.emb_threshold, "jac_threshold": args.jac_threshold,
            "ontology": resolved_ont, "llm": llm_tag,
            "num_samples": len(results), "skipped": skipped,
            "soft": {
                "jaccard":   _aggregate([r["soft"]["jaccard"] for r in results]),
                "embedding": _aggregate([r["soft"]["emb"]     for r in results]),
                "llm_judge": _aggregate([r["soft"]["llm"]     for r in results]),
            },
            "full": {
                "jaccard":   _aggregate([r["full"]["jaccard"] for r in results]),
                "embedding": _aggregate([r["full"]["emb"]     for r in results]),
                "llm_judge": _aggregate([r["full"]["llm"]     for r in results]),
            },
            "samples": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
