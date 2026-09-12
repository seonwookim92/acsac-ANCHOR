EXPECTED OUTPUT -- CLAIM 1

The run writes artifact/output/functional-uco.json and then validates it with
artifact/scripts/check_output.py.

ACCEPTANCE CRITERION -- the final line of ./claims/claim1/run.sh must be:

    PASS: valid uco ANCHOR output

The validator enforces this structural contract on the emitted graph:

  - the file is valid UTF-8 JSON;
  - "ontology" equals "uco";
  - the retained input text is nonempty;
  - "extracted_entities" and "extracted_triplets" are both arrays;
  - there is at least one extracted entity;
  - every entity has a nonempty "name" and a nonempty "class"; and
  - every emitted triplet carries the required "subject", "relation",
    "property", "property_type", and "object" fields.

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
model weak at that completes the run and emits a poor or empty graph, which
fails the criterion above. That is a property of the chosen model, not a defect
in the artifact. check_endpoint.py probes tool calling before the run and prints
a WARNING when the model fails the probe; claims/claim3 validates the packaged
code and data with no LLM at all. If claim3 passes and this claim yields an
empty graph, the package is intact and the model is the variable.

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
