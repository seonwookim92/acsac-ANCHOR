# Expected outputs

This is the consolidated reference. The authoritative per-claim acceptance
criteria live next to each claim, in `claims/claimN/expected/README.txt`.

## claim3 — deterministic package checks (no endpoint required)

```bash
./claims/claim3/run.sh
```

Checks the ACSAC layout, Python syntax, command-line loading, input/label
alignment, and RDF parsing. It must finish with:

```text
PASS: deterministic functional checks completed
```

It requires no LLM endpoint, no GPU, no credentials, and no network access, and
does not reproduce any paper metric. Run it first.

Full criterion: [`claims/claim3/expected/README.txt`](claims/claim3/expected/README.txt).

## claim1 — end-to-end run on one report (UCO)

```bash
./claims/claim1/run.sh
```

The generated `artifact/output/functional-uco.json` must:

- be valid UTF-8 JSON;
- report `"ontology": "uco"`;
- retain nonempty input text;
- contain `extracted_entities` and `extracted_triplets` arrays;
- contain at least one extracted entity;
- give every entity a nonempty `name` and `class`; and
- use the required `subject`, `relation`, `property`, `property_type`, and
  `object` fields for every emitted triplet.

The validator prints:

```text
PASS: valid uco ANCHOR output
```

Full criterion: [`claims/claim1/expected/README.txt`](claims/claim1/expected/README.txt).

## claim2 — schema switch (STIX and MALOnt)

```bash
./claims/claim2/run.sh
```

Applies the same structural contract to each schema, with `"ontology"` matching
the schema selected on the command line. Must print:

```text
PASS: valid stix ANCHOR output
PASS: valid malont ANCHOR output
PASS: schema switch validated for stix and malont
```

Full criterion: [`claims/claim2/expected/README.txt`](claims/claim2/expected/README.txt).

## Model capability is a prerequisite

ANCHOR's ontology mapper drives the MCP server through multi-step tool calling,
so claim1 and claim2 depend on the chosen model sustaining several dependent tool
calls per entity and predicate. A model weak at that completes the run and emits
a poor or empty graph, failing the structural check — a model limitation, not an
artifact defect. `check_endpoint.py` probes tool calling before each run and warns
when the model fails; `claim3` validates the package with no LLM at all, so the
two causes stay distinguishable.

## What is *not* an acceptance criterion

Because an LLM participates in extraction and mapping, exact entity names,
counts, ordering, and ontology URIs vary between runs, and vary more when a
different model is used. **None of these are acceptance criteria**, and no
reference output file is shipped for value comparison — comparing against one
would not be a meaningful test of a functional claim.

Any model with solid tool-calling support may be substituted for the functional
checks. See `infrastructure/constraints`.

## Common failures

- **Endpoint unreachable:** `check_endpoint.py` prints the provider, base URL,
  and model it resolved from `artifact/.env`. Confirm `LLM_PROVIDER` and that
  the matching block is filled in.
- **Model not served** (`LLM_PROVIDER=vllm`): the error lists the model names
  the server actually serves. Set `VLLM_LLM_MODEL` to one of them, or start the
  server with a matching `--served-model-name`.
- **Missing API key** (`openai` / `anthropic`): the error names the missing
  variable.
- **MCP script not found:** run from the package root and keep the distributed
  relative paths in `artifact/.env.example`.
- **First-run delay:** sentence-transformers is downloading its embedding model.
  Confirm outbound HTTPS access and retry after the download completes.
- **Empty or near-empty graph:** some small models cannot sustain the multi-step
  tool calling the mapper requires. `check_endpoint.py` probes this up front and
  prints `WARNING: the model did not make a tool call when asked to` — if you saw
  that, switch to a model with solid tool-use support. For a self-hosted server,
  confirm it was started with tool calling enabled (vLLM:
  `--enable-auto-tool-choice --tool-call-parser ...`).
- **`Unrecognized request argument: chat_template_kwargs` (HTTP 400):** the
  endpoint is a strict OpenAI-compatible server that rejects vLLM's
  thinking-toggle extension. ANCHOR detects this, warns once, and retries without
  it, so the run continues. Set `VLLM_EXTRA_BODY=false` in `artifact/.env` to
  suppress the field and the warning from the start. Tolerant servers ignore the
  field and need no change — but note that on such a server the `VLLM_THINKING`
  toggle has no effect, so a reasoning model keeps emitting thinking tokens and
  may exhaust `LLM_MAX_TOKENS` before completing its JSON. Raise
  `LLM_MAX_TOKENS`, or use a model with reasoning disabled.
- **Rate limiting (HTTP 429) on a hosted API:** a claim run makes on the order of
  30–50 sequential calls. On a low-tier account, run the claims one at a time
  rather than in parallel, and retry after the limit resets.
- **`no supported Python found`:** `install.sh` accepts Python 3.11, 3.12, or
  3.13 and auto-detects one. Its error message lists install commands per
  platform, or point it at an interpreter with
  `PYTHON_BIN=/path/to/python3.12 ./install.sh`.
- **CUDA out of memory** (only when self-hosting the reference model): stop
  competing jobs, reduce `MAX_LEN`, or use a hosted provider instead — see
  `HARDWARE_REQUIREMENTS.md`.
