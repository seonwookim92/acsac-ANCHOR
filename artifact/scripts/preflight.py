#!/usr/bin/env python3
"""Deterministic integrity checks for the Available/Functional package.

Verifies the ACSAC packaging layout, that every distributed Python source
parses, that the 149 CTINexus inputs align with 149 author-created records per
schema, and that all packaged ontologies parse as RDF. No LLM endpoint, GPU, or
network access is required.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]        # artifact/
PKG_ROOT = ROOT.parent                            # package root
SCHEMAS = ("uco", "stix", "malont")

# Top-level layout required by the ACSAC packaging guidelines.
REQUIRED_LAYOUT = (
    "install.sh",
    "README.txt",
    "license.txt",
    "use.txt",
    "artifact",
    "infrastructure",
    "infrastructure/constraints",
    "infrastructure/access",
    "infrastructure/resources",
    "infrastructure/model_snapshot.json",
    "claims",
    "claims/claim1/claim.txt",
    "claims/claim1/run.sh",
    "claims/claim1/expected",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def report_id(path: Path) -> str:
    name = path.stem
    return name.removeprefix("ctinexus_")


def check_layout() -> None:
    missing = [rel for rel in REQUIRED_LAYOUT if not (PKG_ROOT / rel).exists()]
    if missing:
        fail(f"required ACSAC layout entries are missing: {missing}")

    claim_dirs = sorted(p for p in (PKG_ROOT / "claims").iterdir()
                        if p.is_dir() and p.name.startswith("claim"))
    if not claim_dirs:
        fail("no claims/claimN directories found")
    for claim_dir in claim_dirs:
        for required in ("claim.txt", "run.sh", "expected"):
            if not (claim_dir / required).exists():
                fail(f"{claim_dir.name} is missing {required}")
        run_sh = claim_dir / "run.sh"
        if not run_sh.stat().st_mode & 0o111:
            fail(f"{claim_dir.name}/run.sh is not executable")
    print(f"PASS: ACSAC layout present ({len(claim_dirs)} claim directories)")


def check_source() -> None:
    files = sorted(
        path for path in ROOT.rglob("*.py")
        if ".venv" not in path.parts and "output" not in path.parts
    )
    if not files:
        fail("no Python sources found")
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            fail(f"Python source check failed for {path.relative_to(ROOT)}: {exc}")
    print(f"PASS: {len(files)} Python source files parse")


def check_data() -> None:
    inputs = sorted((ROOT / "dataset").glob("ctinexus_*.txt"))
    if len(inputs) != 149:
        fail(f"expected 149 CTINexus inputs, found {len(inputs)}")
    input_ids = {report_id(path) for path in inputs}

    for schema in SCHEMAS:
        records = sorted((ROOT / "ground_truth" / schema).glob("*.json"))
        if len(records) != 149:
            fail(f"expected 149 {schema} records, found {len(records)}")
        record_ids = {path.stem for path in records}
        if record_ids != input_ids:
            missing = sorted(input_ids - record_ids)
            extra = sorted(record_ids - input_ids)
            fail(f"{schema} identifiers differ; missing={missing[:3]}, extra={extra[:3]}")
        for path in records:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")
            if not isinstance(value, dict):
                fail(f"ground-truth root is not an object: {path.relative_to(ROOT)}")

    print("PASS: 149 aligned inputs and 149 records for each schema")


def check_ontologies() -> None:
    try:
        from rdflib import Graph
    except ImportError:
        fail("rdflib is not installed; run ./install.sh")

    ontology_files = {
        "uco": sorted((ROOT / "ontology" / "uco").rglob("*.ttl")),
        "stix": sorted((ROOT / "ontology" / "stix").rglob("*.owl")),
        "malont": sorted((ROOT / "ontology" / "malont").rglob("*.owl")),
    }
    for schema, paths in ontology_files.items():
        if not paths:
            fail(f"no ontology files found for {schema}")
        for path in paths:
            try:
                Graph().parse(path)
            except Exception as exc:
                fail(f"RDF parse failed for {path.relative_to(ROOT)}: {exc}")
        print(f"PASS: {schema} RDF parses ({len(paths)} files)")


def check_clean_package(strict: bool = False) -> None:
    """Report editor/interpreter droppings in the package tree.

    These files are git-ignored and excluded from the release archive, so they
    are harmless to a reviewer -- merely running `python main.py` once creates
    __pycache__. They are therefore only a WARNING by default, and a hard
    failure under --strict, which the authors' packaging script uses to keep
    them out of the published repository and archive.
    """
    ignored_runtime_dirs = {".git", ".venv", "output", "results"}
    unwanted = [
        path for path in PKG_ROOT.rglob("*")
        if not ignored_runtime_dirs.intersection(path.parts)
        and (path.name == ".DS_Store" or path.suffix == ".pyc" or path.name == "__pycache__")
    ]
    if unwanted:
        listing = ", ".join(str(p.relative_to(PKG_ROOT)) for p in unwanted[:5])
        more = f" (and {len(unwanted) - 5} more)" if len(unwanted) > 5 else ""
        message = (
            f"generated files are present: {listing}{more}\n"
            f"       Harmless for evaluation -- they are git-ignored and excluded "
            f"from the release archive.\n"
            f"       Remove them with:\n"
            f"         find . -name .DS_Store -delete\n"
            f"         find . -name __pycache__ -type d -prune "
            f"-not -path './artifact/.venv/*' -exec rm -rf {{}} +"
        )
        if strict:
            fail(message)
        print(f"WARNING: {message}", file=sys.stderr)
    else:
        print("PASS: no generated files outside ignored runtime directories")
    if (ROOT / ".env").exists():
        print("INFO: local artifact/.env exists and is excluded by .gitignore/build_archive.sh")


def main() -> None:
    strict = "--strict" in sys.argv
    check_layout()
    check_clean_package(strict=strict)
    check_source()
    check_data()
    check_ontologies()
    print("PASS: artifact preflight completed")


if __name__ == "__main__":
    main()
