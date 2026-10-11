EXPECTED OUTPUT -- CLAIM 1

The run writes artifact/output/functional-uco.json and then validates it with
artifact/scripts/check_output.py.

UCO is the packaged schema that supplies SHACL shapes. During this run the
SHACL correction stage checks the constraints applicable to each mapped UCO
class. A class for which UCO defines no applicable shape is reported separately
and is not counted as having passed a SHACL constraint.

ACCEPTANCE CRITERION -- the final line of ./claims/claim1/run.sh must be:

    PASS: valid uco ANCHOR output

The validator enforces this structural contract on the emitted graph:

  - the file is valid UTF-8 JSON;
  - "ontology" equals "uco";
  - the retained input text is nonempty;
  - "extracted_entities" and "extracted_triplets" are both arrays;
  - there is at least one extracted entity and at least one triplet;
  - every entity has a nonempty "name" and a nonempty "class"; and
  - every emitted triplet carries the required "subject", "relation",
    "property", "property_type", and "object" fields as nonempty strings;
  - every non-generic class and property URI is declared by the selected
    packaged ontology;
  - the output is not entirely generic (`owl:Thing`, `rdfs:label`, and
    `rdfs:seeAlso`); and
  - `mapping_diagnostics` reports that every mapper item completed its final
    tool call, with zero operational fallbacks.

The validator prints a diagnostic summary before the final PASS line, for
example:

    PASS: mapper diagnostics (24/24 items completed; 2 semantic fallback(s); 0 operational fallbacks)

An intentional generic URI selected after successful ontology discovery and a
completed final tool call is a semantic fallback and is allowed. A value
inserted because of a timeout, exception, missing final tool call, or missing
entity URI is an operational fallback and fails this Functional claim.

NOT AN ACCEPTANCE CRITERION

An LLM participates in extraction and ontology mapping, so the run is
nondeterministic. Exact entity names, entity and triplet counts, ordering, and
the specific ontology URIs chosen will differ between runs, and will differ more
between models. None of these are acceptance criteria for this claim, and no
numerical tolerance is asserted. A reference output file is deliberately NOT
shipped for comparison, because byte or value comparison against one is not a
meaningful test of this claim.

A representative successful run on the authors' reference deployment produced a
graph on the order of 20-40 entities and 20-40 triplets for this report. This
figure is given only to orient a reviewer; a materially different count is not a
failure as long as the structural contract above holds.

MODEL CAPABILITY IS A PREREQUISITE

The ontology mapper drives the MCP server through multi-step tool calling. A
model weak at that may omit a final tool call or time out, which is recorded as
an operational fallback and fails the criterion above. This does not prohibit a
completed mapping from intentionally choosing a generic semantic fallback when
the ontology contains no suitable term. check_endpoint.py probes tool calling
before the run; claims/claim3 validates the packaged code and data with no LLM.

TROUBLESHOOTING

  Endpoint unreachable
      check_endpoint.py reports the provider, base URL, and model it resolved
      from artifact/.env. Confirm LLM_PROVIDER and the matching block.

  Model not served (LLM_PROVIDER=vllm)
      The error lists the model names the server actually serves. Set
      VLLM_LLM_MODEL to one of them, or start the server with
      --served-model-name matching your setting.

  MCP script not found
      Run from the package root and keep the distributed relative paths in
      artifact/.env.example (MCP_SERVER_SCRIPT=./mcp/ontology-mcp-server/main.py).

  Long first run
      sentence-transformers downloads all-MiniLM-L6-v2 on first use. Confirm
      outbound HTTPS access and retry once the download completes.

  Empty or near-empty graph
      Some small models cannot sustain the multi-step tool calling the mapper
      requires. check_endpoint.py probes this before the run and warns if the
      model did not make a tool call. Use a model with solid tool-use support;
      for a self-hosted server, confirm tool calling was enabled at launch.

  HTTP 400 "Unrecognized request argument: chat_template_kwargs"
      A strict OpenAI-compatible server rejected vLLM's thinking-toggle
      extension. ANCHOR warns once and retries without it, so the run
      continues. Set VLLM_EXTRA_BODY=false in artifact/.env to suppress it.

  HTTP 429 rate limiting on a hosted API
      A claim run makes on the order of 30-50 sequential calls. Run claims one
      at a time and retry after the limit resets.
