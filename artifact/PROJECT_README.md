# ANCHOR: Schema-Agnostic Threat Knowledge Graph Construction via Hybrid Ontology Discovery 

ANCHOR (Adaptive Navigation for Cybersecurity Hybrid Ontology Reasoning) is a schema-agnostic CTI knowledge graph construction system that converts unstructured cyber threat reports into ontology-aligned knowledge graphs. It combines LLM-based extraction, hybrid ontology discovery over formal OWL/SHACL schemas, and SHACL-based validation to produce structured entities and relations across multiple cybersecurity ontologies such as UCO, STIX, and MALOnt.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CTI Text Pipeline](#cti-text-pipeline)
  - [Shared Flags](#shared-flags)
  - [Ablation Flags](#ablation-flags)
- [Output Format](#output-format)
- [Evaluation](#evaluation)
- [Supported Ontologies](#supported-ontologies)
- [Citation](#citation)

---

## Repository Structure

This file documents the `artifact/` directory. For the surrounding ACSAC
package layout (`install.sh`, `claims/`, `infrastructure/`, `README.txt`), see
the package-root `README.md`.

```
artifact/                       # this directory
├── main.py                     # Pipeline entry point
├── visualize.py                # Render an output graph to PNG (optional)
├── .env.example                # Environment variable template (a dotfile)
├── requirements-core.txt       # Pinned direct dependencies
├── requirements-lock.txt       # Fully locked environment used by ./install.sh
│
├── src/                        # Pipeline core
│   ├── graph.py                # LangGraph StateGraph definitions
│   ├── state.py                # Shared pipeline state (AnchorState)
│   ├── nodes/
│   │   ├── chunker.py          # Phase 1: text chunking with overlap
│   │   ├── extractor.py        # Phase 1: entity + relation extraction
│   │   ├── preprocessor.py     # Phase 1: noise filtering, IoC tagging
│   │   ├── mapper.py           # Phase 2: MCP-based ontology mapping + SHACL correction
│   │   └── exporter.py         # Phase 3: JSON / Cypher output
│   └── utils/
│       ├── llm.py              # LLM provider abstraction
│       ├── cypher_export.py    # Neo4j Cypher generation
│       └── ioc_patterns.py     # IoC regex patterns
│
├── mcp/
│   └── ontology-mcp-server/    # MCP ontology server
│
├── ontology/                   # OWL/SHACL ontology schemas
│   ├── uco/                    # Unified Cyber Ontology
│   ├── stix/                   # STIX 2.1
│   ├── malont/                 # MalOnt
│
├── dataset/                    # CTINexus evaluation dataset (.txt files)
├── ground_truth/               # Ground truth annotations per ontology
│   └── {uco,stix,malont}/
├── evaluation/                 # Evaluation scripts
│   ├── evaluate_all.py
│   ├── evaluate_entity_extraction.py
│   ├── evaluate_entity_typing.py
│   ├── evaluate_triple_extraction.py
│   └── evaluate_triple_typing.py
├── scripts/                    # Artifact-evaluation checks
│   ├── preflight.py            # Deterministic package integrity checks
│   ├── smoke_test.sh           # Runs preflight + CLI/graph/infra dry runs
│   ├── check_endpoint.py       # Validates the configured LLM endpoint
│   ├── check_output.py         # Validates an emitted graph's structure
│   └── run_example.sh          # One-report end-to-end run per schema
└── output/                     # Default output directory
```

---

## Requirements

- Python 3.11, 3.12, or 3.13 (verified on all three; `install.sh` auto-detects one)
- An LLM provider: [Anthropic API](https://www.anthropic.com/), [OpenAI API](https://platform.openai.com/), or a self-hosted [vLLM](https://github.com/vllm-project/vllm) server

The pipeline has been exercised with `claude-haiku-4-5`, `gpt-4o`,
`gpt-5.4-mini`, `gpt-oss:120B`, `Qwen3.5-35B`, `Qwen3.6-35B`, and `Gemma4-27B`
(the open-weights models via vLLM). Any model with solid multi-step tool-calling
support should work; the mapper drives the ontology MCP server through tool use.

The paper's reported measurements come specifically from `Qwen3.5-35B-A3B-FP8`
in the configuration documented in the package-root `VLLM_SETUP.md`. Other
models exercise the same code path but do not reproduce those numbers.

---

## Installation

### Recommended: the packaged installer

From the **package root** (the directory above this one):

```bash
./install.sh
```

This creates `artifact/.venv`, installs the exact locked environment from
`requirements-lock.txt`, and runs the deterministic package checks. It is the
environment the artifact was tested with, and the one the `claims/` scripts
expect. Use it unless you have a reason not to.

### Alternative: manual environment

If you prefer to manage the environment yourself:

```bash
conda create -n anchor-env python=3.11
conda activate anchor-env
pip install -r requirements-core.txt
```

`requirements-core.txt` pins the direct dependencies; `install.sh` uses
`requirements-lock.txt`, which additionally pins every transitive dependency.
Prefer the lock file for reproducibility.

The `claims/` scripts look for `artifact/.venv/bin/python` by default. With a
manual environment, point them at your interpreter instead:

```bash
ANCHOR_PYTHON=$(which python) ./claims/claim3/run.sh
```

### Configure environment variables

```bash
cp .env.example .env      # a dotfile — list it with `ls -a`
# Edit .env with your values (see Configuration section below)
```

---

## Configuration

All settings are controlled via `.env`. Copy `.env.example` to `.env` and set the values below.

### LLM Provider

Choose one of three providers by setting `LLM_PROVIDER`:

**Anthropic**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_LLM_MODEL=claude-sonnet-4-6
```

**OpenAI**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o
```

**vLLM (self-hosted)**
```env
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=token-abc123        # set to anything if auth is disabled
VLLM_LLM_MODEL=Qwen/Qwen3.6-35B
VLLM_THINKING=false              # true: enable extended thinking (Qwen3)
```

### Token Limits

```env
LLM_MAX_TOKENS=2048      # extractor phase
MAPPER_MAX_TOKENS=16384  # mapper phase (MCP agent)
MAPPER_TIMEOUT=600       # per-item timeout in seconds
```

### MCP Ontology Server

```env
# Path to the MCP server entry point (run commands from the artifact root)
MCP_SERVER_SCRIPT=./mcp/ontology-mcp-server/main.py

# Path to the ontology schema folder
# Available: ontology/uco | ontology/stix | ontology/malont 
ONTOLOGY_DIR=./ontology/uco
```

### Embedding Model

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2   # sentence-transformers model
EMBEDDING_DEVICE=cpu                # cpu | cuda | mps
```

### Output

```env
OUTPUT_DIR=./output
ENABLE_EVAL_TOOLS=true   # required for final JSON export via MCP
```

---

## Usage

### CTI Text Pipeline

Process plain-text CTI reports and produce an ontology-aligned knowledge graph.

```bash
# Single text string
python main.py --text "LockBit ransomware encrypts files using AES-256..."

# From a .txt file
python main.py --file report.txt

# From a .txt file with explicit output filename
python main.py --file report.txt --output report_uco.json

# Batch: all .txt files in a directory
python main.py --file-dir dataset/ --output-dir output/

# Override ontology schema at runtime
python main.py --file report.txt --schema stix

# Use a different LLM for this run
python main.py --file report.txt --llm anthropic --llm-model claude-opus-4-7

# Also write a Neo4j-importable Cypher file
python main.py --file report.txt --export-cypher

# Print pipeline graph as Mermaid diagram
python main.py --graph

# Diagnostic flags
python main.py --file report.txt --timer        # per-node wall-clock time
python main.py --file report.txt --measurement  # embedding score distribution (MCP only)
python main.py --file report.txt --verbose      # stream agent tool calls in real time
```

**Re-running ontology mapping on previously extracted entities** (skips extraction):

```bash
python main.py --file-dir dataset/ --ext-ref output/previous_run/ --output-dir output/remap/
```

### Shared Flags

| Flag | Description |
|---|---|
| `--schema {uco,stix,malont}` | Override `ONTOLOGY_DIR` for this run |
| `--output FILE` | Output filename (single-run mode, default: `result.json`) |
| `--output-dir DIR` | Output directory (batch mode, default: `./output`) |
| `--timer` | Print per-node wall-clock time after the run |
| `--measurement` | Print search-vs-recurse breakdown and embedding score histogram |
| `--limit N` | Process at most N files in batch mode |
| `--verbose` | Stream mapper agent tool calls and LLM steps in real time |
| `--llm PROVIDER` | Override `LLM_PROVIDER` for this run |
| `--llm-model MODEL` | Override the model name for the active provider |
| `--export-cypher` | Write a `.cypher` file alongside the JSON output |

### Ablation Flags

Used for controlled experiments as described in the paper.

| Flag | Description |
|---|---|
| `--single-pass` | ① Joint entity+relation extraction (no two-pass) |
| `--no-coref` | ② Disable coreference resolution |
| `--no-noise-filter` | ③ Disable noise entity filtering in preprocessor |
| `--no-ioc` | ④ Disable IoC pre-assignment |
| `--mapper-tools {search,recurse,both}` | ⑤ Restrict ontology search strategy |
| `--mapper-tools-class MODE` | ⑤a Strategy for entity class resolution only |
| `--mapper-tools-prop MODE` | ⑤b Strategy for attribute/relation mapping only |
| `--max-tool-calls N` | ⑥ Max MCP tool calls per mapper item (default: 4) |
| `--class-threshold T` | ⑦a Embedding confidence cutoff for `search_classes` (default: 0.45) |
| `--prop-threshold T` | ⑦b Embedding confidence cutoff for `search_attributes/relations` (default: 0.3) |
| `--name-weight W` | ⑧ Weight for class name embedding vs. description embedding |
| `--keyword-bonus B` | ⑨ Score bonus for verbatim name match |
| `--no-thinking` | ⑩ Disable extended thinking for the mapper LLM |
| `--mapper-timeout SEC` | ⑪ Per-item timeout in seconds (default: 600) |

---

## Output Format

Each run produces a JSON file (and optionally a `.cypher` file) in `OUTPUT_DIR`.

**JSON schema:**

```json
{
  "file": "result.json",
  "text": "<input text>",
  "ontology": "uco",
  "extracted_entities": [
    {"name": "LockBit", "class": "https://...uco/tool/MaliciousTool"}
  ],
  "extracted_triplets": [
    {
      "subject": "LockBit",
      "relation": "targets",
      "property": "https://...rdfs#seeAlso",
      "property_type": "ObjectProperty",
      "object": "Windows systems"
    }
  ],
  "validation_warnings": []
}
```

`validation_warnings` is populated when an entity fails SHACL validation after all correction retries.

**Cypher output** (`--export-cypher`):

```cypher
MERGE (:`MaliciousTool` {name: 'LockBit', ontology_class: 'https://...', ontology: 'uco'});
MATCH (s {name: 'LockBit'}), (o {name: 'Windows systems'})
MERGE (s)-[:`TARGETS` {label: 'targets'}]->(o);
```

Import into Neo4j: `cypher-shell -f result.cypher` or paste into Neo4j Browser.

---

## Visualizing an output graph

`visualize.py` renders an emitted graph to a PNG for inspection. It is a
convenience utility and is not part of the pipeline or of any artifact claim.

```bash
python visualize.py --file output/functional-uco.json --out output/functional-uco.png

# Also draw the source sentence and the resolved ontology URIs
python visualize.py --file output/functional-uco.json --out graph.png \
    --show-text --show-schema
```

Requires `matplotlib` and `networkx`, both included in `requirements-core.txt`
and `requirements-lock.txt`.

---

## Evaluation

The `evaluation/` directory contains scripts for benchmarking against the CTINexus dataset.

**Run all metrics across all output subdirectories:**

```bash
cd evaluation
python evaluate_all.py \
    --results  ../output/ \
    --ground-truth ../ground_truth/
```

**Run individual metrics:**

```bash
# Entity extraction (schema-agnostic)
python evaluate_entity_extraction.py \
    --results ../output/run_uco/ \
    --ground-truth ../ground_truth/ \
    --ontology uco

# Entity ontology typing
python evaluate_entity_typing.py \
    --results ../output/run_uco/ \
    --ground-truth ../ground_truth/ \
    --ontology uco

# Triple extraction
python evaluate_triple_extraction.py \
    --results ../output/run_uco/ \
    --ground-truth ../ground_truth/ \
    --ontology uco

# Triple ontology typing
python evaluate_triple_typing.py \
    --results ../output/run_uco/ \
    --ground-truth ../ground_truth/ \
    --ontology uco
```

Ground truth files are located at `ground_truth/{uco,stix,malont}/`.

---

## Supported Ontologies

| Key | Name | Path |
|---|---|---|
| `uco` | Unified Cyber Ontology (UCO) | `ontology/uco/` |
| `stix` | STIX 2.1 | `ontology/stix/` |
| `malont` | MalOnt | `ontology/malont/` |

Switch ontologies at runtime with `--schema` or by setting `ONTOLOGY_DIR` in `.env`.

---

## Citation

```bibtex
@article{anchor2025,
  title   = {ANCHOR: Schema-Agnostic Threat Knowledge Graph Construction via Hybrid Ontology Discovery},
  author  = {Seonwoo Kim and Jinwoo Kim and Daegyu Kang and Daeseong Kim and Insup Lee},
  journal = {Annual Computer Security Applications Conference},
  year    = {2026}
}
```
