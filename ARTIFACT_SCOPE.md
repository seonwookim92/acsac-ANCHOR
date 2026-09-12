# Artifact scope and paper mapping

## Requested badges

- **Available**
- **Functional**

The **Reproduced** badge is not requested. No numerical tolerance or paper-table
acceptance range is asserted by this package.

## Functional claims

Each claim has an executable `claims/claimN/run.sh` and a written acceptance
criterion in `claims/claimN/expected/README.txt`.

| Claim | Paper connection | Artifact component | Reviewer action | Acceptance condition |
| --- | --- | --- | --- | --- |
| **claim1.** ANCHOR implements the CTI-to-ontology pipeline described by the paper. | Section IV; Figure 2 architecture | `artifact/main.py`, `artifact/src/`, `artifact/mcp/ontology-mcp-server/` | `./claims/claim1/run.sh` | Run completes and the output validator prints `PASS: valid uco ANCHOR output`. |
| **claim2.** The implementation is schema-switchable across UCO, STIX, and MALOnt. | Section IV-A principle (i); Table I "Schema dependency: Any OWL/SHACL" | `artifact/ontology/`, the `--schema` flag, `artifact/src/nodes/mapper.py`, `artifact/src/nodes/exporter.py` | `./claims/claim2/run.sh` | Each output's `ontology` field matches the selected schema and satisfies the structural contract. |
| **claim3.** The distributed inputs, labels, schemas, and sources are internally complete. | Section V evaluation corpus and author-created labels | `artifact/dataset/`, `artifact/ground_truth/`, `artifact/ontology/`, `artifact/scripts/preflight.py` | `./claims/claim3/run.sh` | 149 aligned inputs and 149 records per schema are found; all ontologies and sources parse; the ACSAC layout is present. **No LLM endpoint required.** |

## Infrastructure position

The paper's reference configuration self-hosts `Qwen/Qwen3.5-35B-A3B-FP8`,
because paper Challenge 3 is specifically about privacy-preserving CTI analysis
with a local LLM. Model weights, GPU hardware, and a running vLLM service are
not redistributable.

This does not constrain evaluation. ANCHOR resolves its chat model at runtime
from `LLM_PROVIDER`, so every claim above can be executed from a commodity
client with no GPU by pointing the artifact at a hosted OpenAI-compatible or
Anthropic endpoint. `claim3` needs no endpoint at all. Full rationale is in
`infrastructure/constraints`; configuration in `infrastructure/access`.

## Model capability is a prerequisite

ANCHOR's hybrid ontology discovery is an LLM agent driving the ontology MCP
server through multi-step tool calling (paper Section IV-C). claim1 and claim2
therefore depend on the evaluator's chosen model sustaining several dependent
tool calls per entity and predicate. A model weak at tool calling completes a run
and emits a poor or empty graph, failing the structural check — a limitation of
that model, not a defect in the artifact.

The package keeps the two causes separable: `check_endpoint.py` probes tool
calling before each run and warns when the model fails it, and `claim3` validates
the packaged code and data with no LLM at all. If claim3 passes while claim1
yields an empty graph, the artifact is intact and the model is the variable.

## Explicitly outside the requested evaluation

The submission does not ask evaluators to reproduce the numerical values in
Tables III–IX or any ablation ordering. The original evaluation programs and
flags are retained because they are part of the released research code, not
because they are proposed as Reproduced-badge tasks.

Because a model may be substituted for the functional checks, extracted entity
names, counts, ordering, and ontology URIs will differ from the authors' runs.
That is expected and is not a failure of any claim here.

ANCHOR takes CTI text as input. Converting source documents (PDF, images, web
pages) to text is a separate concern, is not described in the paper, and is not
included in this package.
