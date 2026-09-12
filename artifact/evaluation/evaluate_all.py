#!/usr/bin/env python3
"""
evaluate_all.py — Run all four evaluation scripts across all model+schema combos.

Expected layout:
  output/                              ← --results points here
    vllm_qwen3.5-35b_uco/             ← one subdir = one model+schema combo
    ctinexus_vllm_qwen3.5-35b_stix/
    gtikg_vllm_qwen3.5-35b/           ← ontology=none → extraction only
    ...
  ground_truth/                         ← --ground-truth points here
    uco/json/*.json
    stix/json/*.json
    malont/json/*.json

For each subdirectory the script runs:
  1. evaluate_entity_extraction.py
  2. evaluate_entity_typing.py      (skipped if ontology=none)
  3. evaluate_triple_extraction.py
  4. evaluate_triple_typing.py      (skipped if ontology=none)

After all tasks complete, prints a combined cross-directory comparison table.

Notes:
  - gtikg / schema-agnostic dirs (ontology=none) are evaluated for extraction only.
  - For gtikg dirs pass --gtikg-ontology uco|stix|malont to select the GT schema
    used for entity/triple extraction comparison (default: uco).

Usage:
  # All subdirs under output/:
  python evaluate_all.py \\
      --results  output/ \\
      --ground-truth ground_truth/

  # Single subdir:
  python evaluate_all.py \\
      --results  output/vllm_qwen3.5-35b_uco/ \\
      --ground-truth ground_truth/

  # Skip specific tasks:
  python evaluate_all.py --results output/ --ground-truth ground_truth/ \\
      --skip entity-typing triple-typing

  # Only specific tasks:
  python evaluate_all.py --results output/ --ground-truth ground_truth/ \\
      --only triple-extraction
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

TASKS = [
    ("entity-extraction", ROOT / "evaluate_entity_extraction.py", "eval_entity_ext_"),
    ("entity-typing",     ROOT / "evaluate_entity_typing.py",     "eval_entity_typing_"),
    ("triple-extraction", ROOT / "evaluate_triple_extraction.py", "eval_triple_ext_"),
    ("triple-typing",     ROOT / "evaluate_triple_typing.py",     "eval_triple_typing_"),
]

TYPING_TASKS = {"entity-typing", "triple-typing"}


def _infer_ontology_from_dir(dirpath: Path) -> str:
    """Infer ontology from output directory name."""
    name = dirpath.name.lower()
    for ont in ('malont', 'stix', 'uco'):
        if name.endswith(f'_{ont}'):
            return ont
    return 'none'


def build_cmd(
    script: Path, task_name: str, args: argparse.Namespace,
    out_dir: Path, results_override: str = None,
    ontology_override: str = None,
) -> list[str]:
    cmd = [sys.executable, str(script)]
    cmd += ["--results",      results_override or args.results]
    cmd += ["--ground-truth", args.ground_truth]
    cmd += ["--output",       str(out_dir)]
    ont = ontology_override or args.ontology
    if ont:
        cmd += ["--ontology", ont]
    if args.llm_provider:
        cmd += ["--llm-provider", args.llm_provider]
    if args.llm_model:
        cmd += ["--llm-model", args.llm_model]
    if args.llm_base_url:
        cmd += ["--llm-base-url", args.llm_base_url]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    cmd += ["--llm-timeout", str(args.llm_timeout)]
    if task_name in ("entity-extraction", "triple-extraction"):
        cmd += ["--emb-threshold", str(args.emb_threshold)]
        cmd += ["--jac-threshold", str(args.jac_threshold)]
    if task_name == "entity-extraction" and args.hitl:
        cmd.append("--hitl")
    if task_name in ("triple-extraction", "triple-typing") and args.include_implicit:
        cmd.append("--include-implicit")
    return cmd


def _collect_outputs(out_dir: Path) -> dict:
    """Read all task output JSON files from out_dir, indexed by (key)."""
    collected: dict = {}   # key → {task_name: metrics_dict}
    prefix_to_task = {
        "eval_entity_ext_":     "entity-extraction",
        "eval_entity_typing_":  "entity-typing",
        "eval_triple_ext_":     "triple-extraction",
        "eval_triple_typing_":  "triple-typing",
    }
    for f in out_dir.glob("eval_*.json"):
        for prefix, task in prefix_to_task.items():
            if f.name.startswith(prefix):
                key = f.stem[len(prefix):]   # e.g., "vllm_qwen3.5-35b_uco"
                try:
                    with open(f, encoding="utf-8") as fp:
                        data = json.load(fp)
                    collected.setdefault(key, {})[task] = data
                    if "ontology" in data and data["ontology"] not in ("auto", None):
                        collected[key].setdefault("_ontology", data["ontology"])
                except Exception:
                    pass
                break
    return collected


def _fmt(val, width=10) -> str:
    if val is None:
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.4f}"


def _print_combined_table(collected: dict) -> None:
    if not collected:
        return

    SCHEMA_ORDER = ["uco", "stix", "malont", "none", "unknown"]
    TYPING_SCHEMAS = {"uco", "stix", "malont"}

    schema_groups: dict = {}
    for key, tasks in collected.items():
        ont = tasks.get("_ontology", "unknown")
        if ont in ("auto", None, "unknown"):
            for task_data in tasks.values():
                if isinstance(task_data, dict):
                    candidate = task_data.get("ontology", "")
                    if candidate and candidate not in ("auto", None):
                        ont = candidate
                        break
        schema_groups.setdefault(ont, []).append(key)

    col_labels = [
        ("EntExt",  "LLM-F1"),
        ("EntTyp",  "Hier-Acc"),
        ("TrpSoft", "LLM-F1"),
        ("TrpFull", "LLM-F1"),
        ("TrpTyp",  "Hier-Acc"),
    ]
    n_cols = len(col_labels)
    c   = 10; SEP = " │"
    TITLE = "COMBINED CROSS-DIR COMPARISON — ALL EVALUATION TASKS"

    def _sort_key(k):
        order = ["anchor", "ctinexus", "ttpdrill", "gtikg"]
        for i, prefix in enumerate(order):
            if k.startswith(prefix): return (i, k)
        return (len(order), k)

    all_keys = [k for ont in SCHEMA_ORDER for k in sorted(schema_groups.get(ont, []), key=_sort_key)]
    if not all_keys:
        return
    key_w = max(len(k) for k in all_keys) + 2
    key_w = max(key_w, 24)
    W = key_w + n_cols * (c + len(SEP))
    W = max(W, len(TITLE) + 4)

    def _row_vals(key):
        tasks = collected[key]
        ee = tasks.get("entity-extraction", {})
        et = tasks.get("entity-typing", {})
        te = tasks.get("triple-extraction", {})
        tt = tasks.get("triple-typing", {})
        return [
            ee.get("llm_judge", {}).get("micro_f1") if ee else None,
            et.get("micro_acc") if et else None,
            te.get("soft", {}).get("llm_judge", {}).get("micro_f1") if te else None,
            te.get("full", {}).get("llm_judge", {}).get("micro_f1") if te else None,
            tt.get("micro_acc") if tt else None,
        ]

    print("\n" + "═" * W)
    print(f"{TITLE:^{W}}")

    hdr1 = f"  {'Results Key':<{key_w}}"
    hdr2 = f"  {'':>{key_w}}"
    for lbl, sub in col_labels:
        hdr1 += f"{SEP}{lbl:^{c}}"
        hdr2 += f"{SEP}{sub:^{c}}"

    for ont in SCHEMA_ORDER:
        keys = sorted(schema_groups.get(ont, []), key=_sort_key)
        if not keys: continue
        print("═" * W)
        print(f"  Schema: {ont.upper()}")
        print("─" * W)
        print(hdr1); print(hdr2)
        print("─" * W)

        rows_data = []
        for key in keys:
            vals = _row_vals(key)
            rows_data.append((key, vals))
            row = f"  {key:<{key_w}}"
            for v in vals:
                row += f"{SEP}{_fmt(v, c)}"
            print(row)

        print("─" * W)
        best_row = f"  {'Best':>{key_w}}"
        for col_i in range(n_cols):
            col_vals = [(k, v) for k, v in [(k, vals[col_i]) for k, vals in rows_data] if v is not None]
            if col_vals:
                _, best_val = max(col_vals, key=lambda x: x[1])
                best_row += f"{SEP}{_fmt(best_val, c)}"
            else:
                best_row += f"{SEP}{_fmt(None, c)}"
        print(best_row)

    print("═" * W)
    print(f"  EntExt = Entity Extraction (LLM judge micro-F1)")
    print(f"  EntTyp = Entity Type Matching (hierarchical micro-Acc, schema-aware only)")
    print(f"  TrpSoft/Full = Triple Extraction Soft(S+O)/Full(S+R+O) (LLM judge micro-F1)")
    print(f"  TrpTyp = Relation Type Matching (hierarchical micro-Acc, schema-aware only)")
    print("═" * W)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all four evaluation scripts across all model+schema output dirs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",          required=True,
                        help="Parent output directory (contains model+schema subdirs) "
                             "or a single model+schema directory")
    parser.add_argument("--ground-truth",     required=True, help="GT root directory (ground_truth/)")
    parser.add_argument("--ontology",         default=None, choices=["uco", "stix", "malont"],
                        help="Force a specific ontology for all dirs (overrides auto-detection)")
    parser.add_argument("--gtikg-ontology",   default="uco", choices=["uco", "stix", "malont"],
                        help="GT ontology used when comparing schema-agnostic (gtikg) output "
                             "(default: uco)")
    parser.add_argument("--llm-provider",     default=None)
    parser.add_argument("--llm-model",        default=None)
    parser.add_argument("--llm-base-url",     default=None)
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--output",           default=None,
                        help="Output directory for eval JSON files (default: {results}/eval_outputs)")
    parser.add_argument("--emb-threshold",    type=float, default=0.75)
    parser.add_argument("--jac-threshold",    type=float, default=0.2)
    parser.add_argument("--llm-timeout",      type=float, default=180.0)
    parser.add_argument("--hitl",             action="store_true")
    parser.add_argument("--include-implicit", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[],
                        metavar="TASK",
                        help=f"Tasks to skip: {[t for t, _, _ in TASKS]}")
    parser.add_argument("--only", nargs="*", default=None,
                        metavar="TASK",
                        help="Run only these tasks (overrides --skip)")
    args = parser.parse_args()

    # Resolve tasks to run
    if args.only is not None:
        tasks_to_run = [(n, p, px) for n, p, px in TASKS if n in args.only]
    else:
        tasks_to_run = [(n, p, px) for n, p, px in TASKS if n not in args.skip]

    if not tasks_to_run:
        print("[!] No tasks to run after applying --skip / --only filters.")
        return

    results_path = Path(args.results)

    # Determine output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = results_path / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect directories to evaluate
    if results_path.is_dir():
        # Check if this dir directly contains JSON files (= single model+schema combo)
        json_files = [f for f in results_path.glob("*.json") if not f.name.startswith("eval_") and not f.stem.endswith("_results")]
        subdirs = [d for d in results_path.iterdir() if d.is_dir() and not d.name.startswith("eval_")]

        if json_files and not subdirs:
            # Single model+schema directory passed directly
            eval_dirs = [results_path]
        elif subdirs:
            # Parent directory containing multiple model+schema subdirs
            eval_dirs = sorted(subdirs)
        else:
            print(f"[!] No subdirectories or JSON files found in: {results_path}")
            return
    else:
        print(f"[!] --results must be a directory: {results_path}")
        return

    print(f"\n{'═'*60}")
    print(f"  evaluate_all.py — {len(eval_dirs)} dir(s), {len(tasks_to_run)} task(s)")
    print(f"{'═'*60}")
    for d in eval_dirs:
        ont = args.ontology or _infer_ontology_from_dir(d)
        typing_note = "" if ont in ("uco", "stix", "malont") else "  [extraction only]"
        print(f"  • {d.name:<40} ontology={ont}{typing_note}")
    print(f"\n  Ground truth : {args.ground_truth}")
    print(f"  Output dir   : {out_dir}")
    print()

    total_start = datetime.now()
    run_results: list[tuple[str, str, int, int]] = []   # (dir_name, task, returncode, elapsed_s)

    for eval_dir in eval_dirs:
        ont = args.ontology or _infer_ontology_from_dir(eval_dir)
        is_typed = ont in ("uco", "stix", "malont")

        # For schema-agnostic dirs, use gtikg_ontology for extraction GT comparison
        extraction_ont = ont if is_typed else args.gtikg_ontology

        print(f"\n{'━'*60}")
        print(f"  Dir: {eval_dir.name}  (ontology={ont})")
        print(f"{'━'*60}")

        for task_name, script_path, _ in tasks_to_run:
            # Skip typing tasks for schema-agnostic models
            if task_name in TYPING_TASKS and not is_typed:
                print(f"  [skip] {task_name}: ontology=none")
                continue

            # For extraction tasks on gtikg, force the GT ontology
            ont_override = extraction_ont if not is_typed else (args.ontology or None)

            cmd = build_cmd(
                script_path, task_name, args, out_dir,
                results_override=str(eval_dir),
                ontology_override=ont_override,
            )
            started = datetime.now()

            print(f"\n  [{task_name}]")
            print(f"  cmd: {' '.join(str(c) for c in cmd)}")
            print()

            proc = subprocess.run(cmd, cwd=str(ROOT))
            elapsed = int((datetime.now() - started).total_seconds())
            run_results.append((eval_dir.name, task_name, proc.returncode, elapsed))

    total_elapsed = int((datetime.now() - total_start).total_seconds())
    mm, ss = divmod(total_elapsed, 60)

    # Task summary
    print(f"\n{'═'*60}")
    print(f"  TASK SUMMARY")
    print(f"{'═'*60}")
    n_ok   = sum(1 for _, _, rc, _ in run_results if rc == 0)
    n_fail = len(run_results) - n_ok
    for dir_name, task_name, rc, elapsed in run_results:
        m_, s_ = divmod(elapsed, 60)
        status = "OK  " if rc == 0 else f"FAIL (exit {rc})"
        label  = f"{dir_name}/{task_name}"
        print(f"  {label:<50}  {status}  {m_}m {s_:02d}s")
    print(f"{'─'*60}")
    print(f"  {n_ok}/{len(run_results)} succeeded  |  Total: {mm}m {ss:02d}s")
    print(f"{'═'*60}")

    # Combined comparison table
    collected = _collect_outputs(out_dir)
    if collected:
        _print_combined_table(collected)
    else:
        print(f"\n[!] No eval output JSON files found in {out_dir} — combined table skipped.")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
