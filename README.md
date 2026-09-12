# ANCHOR — ACSAC Artifact

**Schema-Agnostic Knowledge Graph Construction via Hybrid Ontology Discovery for Cyber Threat Intelligence**

ANCHOR builds ontology-aligned threat knowledge graphs from unstructured CTI
text. It decouples extraction from any specific schema, discovers relevant
ontology fragments dynamically instead of stuffing a whole schema into the
prompt, and enforces SHACL constraints in a closed correction loop. This
repository contains the implementation, the 149-report CTINexus input corpus,
author-created UCO/STIX/MALOnt annotations, and the packaged ontologies.

> This file is the GitHub-rendered overview. `README.txt` is the same content in
> the plain-text form the ACSAC packaging guidelines require.

The accompanying paper is included as [`paper.pdf`](paper.pdf).
Repository: <https://github.com/seonwookim92/acsac-ANCHOR>
Permanent archive: <https://doi.org/10.5281/zenodo.22724008>

## Requested badges

**Available** and **Functional**.

The **Results Reproduced** badge is *not* requested. This submission does not
ask evaluators to regenerate the numerical values in the paper's evaluation
tables. See [ARTIFACT_SCOPE.md](ARTIFACT_SCOPE.md) for the paper-to-artifact
mapping.

## Quick start

```bash
./install.sh                # creates artifact/.venv, runs deterministic checks
./claims/claim3/run.sh      # package completeness — NO LLM endpoint needed

cp artifact/.env.example artifact/.env   # a dotfile: use `ls -a artifact/`
# edit artifact/.env: set LLM_PROVIDER and fill in that one provider block

./claims/claim1/run.sh      # end-to-end run, UCO
./claims/claim2/run.sh      # schema switch, STIX and MALOnt
```

Start with `claim3`. It is deterministic, takes under two minutes, needs no
endpoint and no GPU, and confirms the package is complete and installable before
any inference is configured.

## No GPU is required to evaluate this artifact

The paper's measurements were produced with a locally served open-weights model
(`Qwen/Qwen3.5-35B-A3B-FP8` on an ASUS Ascent GX10 with 128 GB coherent unified
memory), because paper Challenge 3 concerns privacy-preserving CTI analysis with
a **local** LLM. That reference configuration needs substantial accelerator
memory.

Artifact evaluation does not. ANCHOR resolves its chat model at runtime from
`LLM_PROVIDER` ([`artifact/src/utils/llm.py`](artifact/src/utils/llm.py)), and
all three providers drive the identical pipeline:

| `LLM_PROVIDER` | Endpoint | GPU needed |
| --- | --- | --- |
| `openai` | Hosted OpenAI-compatible API | No |
| `anthropic` | Anthropic Messages API | No |
| `vllm` | Any OpenAI-compatible server that supports tool calling, including the reference deployment | Only if self-hosting |

No model name is pinned in the claim scripts.
[`artifact/scripts/check_endpoint.py`](artifact/scripts/check_endpoint.py)
validates whatever endpoint is configured, reports an actionable error if it is
unreachable or if the configured model is not served (listing the names that
are), and probes tool calling — the one model capability the ontology mapper
genuinely requires — so an unsuitable model is caught in seconds rather than
after a multi-minute run.

Because the claims here assert structural validity rather than the paper's
numbers, **substituting a model does not invalidate any claim in `claims/`** —
it changes the extracted content, not the structural contract being verified.
Configuration for both paths is in
[infrastructure/access](infrastructure/access), the rationale in
[infrastructure/constraints](infrastructure/constraints), and the full
self-hosting procedure — prerequisites, image preparation, the pinned model
snapshot, launch, readiness checks, troubleshooting — in
[VLLM_SETUP.md](VLLM_SETUP.md).

## Claims

| Claim | Assertion | Paper connection | Endpoint needed |
| --- | --- | --- | --- |
| [claim1](claims/claim1/claim.txt) | Processes a CTI report end to end, emits a structurally valid ontology-aligned graph (UCO) | Section IV, Figure 2 | Yes |
| [claim2](claims/claim2/claim.txt) | The same executable is schema-switchable across UCO, STIX, and MALOnt via `--schema` | Section IV-A principle (i); Table I | Yes |
| [claim3](claims/claim3/claim.txt) | The package is internally complete: 149 aligned inputs, 149 records per schema, all ontologies and sources parse | Section V evaluation corpus | **No** |

All three are *functional* claims. None asserts a paper table value. Each
`claims/claimN/expected/README.txt` states the exact acceptance criterion and,
just as importantly, what is **not** an acceptance criterion.


## Model capability is a prerequisite, and it is separable from the artifact

ANCHOR's hybrid ontology discovery (paper Section IV-C) is implemented as an LLM
agent driving the ontology MCP server through **multi-step tool calling**. Typing
one entity or predicate can take several dependent tool calls — an embedding
search, then a recursive walk down the class hierarchy, then a scoped re-search.
This is the mechanism the paper proposes, not an implementation detail, so the
artifact cannot avoid depending on it.

The practical consequence for evaluation: **a model that is weak at tool calling
will produce a poor or empty graph no matter how correct the code is.** Such a
run completes without error and then fails the structural check with
`no entities were extracted` or a near-empty graph. Typical symptoms are a model
that answers in prose instead of emitting a tool call, that invents tool
arguments, or that stops after the first call and never follows up.

This is a property of the chosen model, not a defect in the artifact, and it is
distinguishable from one:

- `artifact/scripts/check_endpoint.py` probes tool calling before every claim run
  and prints `WARNING: the model did not make a tool call when asked to` when the
  model fails the probe. If you saw that warning, a subsequent claim failure is
  explained by model choice.
- `claims/claim3` exercises the packaged code and data with **no LLM at all**. If
  claim3 passes and a claim1/claim2 run yields an empty graph, the package is
  intact and the model is the variable.
- Re-running claim1 with a stronger tool-calling model resolves it. Models this
  pipeline has been run with are listed in `artifact/PROJECT_README.md`.

For a self-hosted server, also confirm it was started with tool calling enabled
(vLLM: `--enable-auto-tool-choice --tool-call-parser ...`); a server without it
silently drops the tool definitions and produces the same symptom.

## Layout

```text
install.sh                  one-click setup (no LLM contact, no GPU)
README.txt                  plain-text overview (ACSAC requirement)
license.txt                 license summary
use.txt                     intended use, limitations, destructive actions
paper.pdf                   the accompanying paper

artifact/                   main code, data, and schemas
  main.py                     ANCHOR command-line entry point
  src/                        extraction, mapping, validation, export
  mcp/ontology-mcp-server/    local ontology discovery MCP server
  ontology/{uco,stix,malont}/ packaged OWL/SHACL schemas
  dataset/                    149 CTINexus report texts
  ground_truth/{uco,stix,malont}/  149 author-created records per schema
  evaluation/                 original evaluation utilities
  scripts/                    functional checks and the one-report example
  PROJECT_README.md           full developer and command reference

infrastructure/             constraints, access, resources,
                            model_snapshot.json, reference vLLM scripts
claims/claim{1,2,3}/        claim.txt, run.sh, expected/
submission/                 AE submission metadata and packaging helper
```

Deliberately **not** included: model weights, a running vLLM service, GPU
hardware, credentials, generated results, and Python environments.

## Requirements

**Client — this is all that is required:** Linux or macOS, Python 3.11, 3.12,
or 3.13 (auto-detected by `install.sh`), Git, outbound HTTPS, about 8 GB RAM
and 5 GB free storage. The first run downloads
the `all-MiniLM-L6-v2` embedding model. No GPU.

**Optional, only to self-host the paper's reference model:** an NVIDIA GPU
server with at least 64 GB of accelerator-addressable VRAM or coherent unified
memory. See [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) and
[VLLM_SETUP.md](VLLM_SETUP.md).

## Functional scope and limitations

- The evaluated path is the text-to-ontology pipeline using the ontology MCP
  server. ANCHOR takes CTI text as input; converting documents to text is out of
  scope and not included.
- The ablation flags and numerical evaluation utilities remain available as
  released research code. Paper-table reproduction is outside the requested
  badge scope.
- Output is a research artifact, not verified intelligence. See
  [use.txt](use.txt).

## Licensing and provenance

Original ANCHOR software is Apache-2.0. Author-created ground-truth labels are
CC BY 4.0. Third-party data and ontologies retain their upstream terms. See
[license.txt](license.txt), [DATA_PROVENANCE.md](DATA_PROVENANCE.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and `LICENSES/`.

## Full documentation

[artifact/PROJECT_README.md](artifact/PROJECT_README.md) documents every
command-line flag, output field, conversion feature, and evaluation utility.
