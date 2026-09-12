================================================================================
ANCHOR -- Schema-Agnostic Knowledge Graph Construction via Hybrid Ontology
          Discovery for Cyber Threat Intelligence
ACSAC Artifact Evaluation -- top-level overview and execution guide
================================================================================

The accompanying paper is included as paper.pdf.
Repository: https://github.com/seonwookim92/acsac-ANCHOR
Permanent archive: https://doi.org/10.5281/zenodo.22724008

REQUESTED BADGES

    Available and Functional.

    The Results Reproduced badge is NOT requested. This submission does not ask
    evaluators to regenerate the numerical values in the paper's evaluation
    tables. See ARTIFACT_SCOPE.md.

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------

    ./install.sh                # creates artifact/.venv, runs deterministic checks
    ./claims/claim3/run.sh      # package completeness -- NO LLM endpoint needed

    cp artifact/.env.example artifact/.env
    # .env.example is a dotfile -- list it with `ls -a artifact/`
    # edit artifact/.env: set LLM_PROVIDER and fill in that one provider block

    ./claims/claim1/run.sh      # end-to-end run, UCO
    ./claims/claim2/run.sh      # schema switch, STIX and MALOnt

Start with claim3. It is deterministic, takes under two minutes, needs no
endpoint and no GPU, and confirms the package is complete and installable
before any inference is configured.

--------------------------------------------------------------------------------
NO GPU IS REQUIRED TO EVALUATE THIS ARTIFACT
--------------------------------------------------------------------------------

The paper's measurements were produced with a locally served open-weights model
(Qwen/Qwen3.5-35B-A3B-FP8 on an ASUS Ascent GX10, 128 GB coherent unified
memory), because paper Challenge 3 concerns privacy-preserving CTI analysis with
a LOCAL LLM. That reference configuration needs substantial accelerator memory.

Artifact evaluation does not. ANCHOR resolves its chat model at runtime from
LLM_PROVIDER, and all three providers drive the identical pipeline:

    LLM_PROVIDER=openai      hosted OpenAI-compatible API   -- no GPU
    LLM_PROVIDER=anthropic   Anthropic Messages API         -- no GPU
    LLM_PROVIDER=vllm        any OpenAI-compatible server   -- incl. the
                             reference deployment, or a small local model

No model name is pinned anywhere in the claim scripts. Because the claims here
assert structural validity and not the paper's numbers, substituting a model
does not invalidate them. Configuration for both paths is in
infrastructure/access, the rationale in infrastructure/constraints, and the
full self-hosting procedure -- prerequisites, image preparation, the pinned
model snapshot, launch, readiness checks, troubleshooting -- in VLLM_SETUP.md.

--------------------------------------------------------------------------------
MODEL CAPABILITY IS A PREREQUISITE, AND IT IS SEPARABLE FROM THE ARTIFACT
--------------------------------------------------------------------------------

ANCHOR's hybrid ontology discovery (paper Section IV-C) is implemented as an LLM
agent driving the ontology MCP server through MULTI-STEP TOOL CALLING. Typing one
entity or predicate can take several dependent tool calls -- an embedding search,
then a recursive walk down the class hierarchy, then a scoped re-search. This is
the mechanism the paper proposes, not an implementation detail, so the artifact
cannot avoid depending on it.

The practical consequence for evaluation: A MODEL THAT IS WEAK AT TOOL CALLING
WILL PRODUCE A POOR OR EMPTY GRAPH NO MATTER HOW CORRECT THE CODE IS. Such a run
completes without error and then fails the structural check with "no entities
were extracted" or a near-empty graph. Typical symptoms are a model that answers
in prose instead of emitting a tool call, that invents tool arguments, or that
stops after the first call and never follows up.

This is a property of the chosen model, not a defect in the artifact, and it is
distinguishable from one:

  - artifact/scripts/check_endpoint.py probes tool calling before every claim run
    and prints "WARNING: the model did not make a tool call when asked to" when
    the model fails the probe. If you saw that warning, a subsequent claim
    failure is explained by model choice.

  - claims/claim3 exercises the packaged code and data with NO LLM at all. If
    claim3 passes and a claim1/claim2 run yields an empty graph, the package is
    intact and the model is the variable.

  - Re-running claim1 with a stronger tool-calling model resolves it. Models this
    pipeline has been run with are listed in artifact/PROJECT_README.md.

For a self-hosted server, also confirm it was started with tool calling enabled
(vLLM: --enable-auto-tool-choice --tool-call-parser ...); a server without it
silently drops the tool definitions and produces the same symptom.

--------------------------------------------------------------------------------
DIRECTORY LAYOUT
--------------------------------------------------------------------------------

    install.sh              one-click setup (no LLM contact, no GPU)
    README.txt              this file
    license.txt             license summary
    use.txt                 intended use, limitations, destructive actions
    paper.pdf               the accompanying paper

    artifact/               main code, data, and schemas
        main.py                 ANCHOR command-line entry point
        src/                    extraction, mapping, validation, export
        mcp/ontology-mcp-server/  local ontology discovery MCP server
        ontology/{uco,stix,malont}/  packaged OWL/SHACL schemas
        dataset/                149 CTINexus report texts
        ground_truth/{uco,stix,malont}/  149 author-created records per schema
        evaluation/             original evaluation utilities
        scripts/                functional checks and the one-report example
        PROJECT_README.md       full developer and command reference
        .env.example            endpoint configuration template

    infrastructure/         constraints, access, resources, model_snapshot.json,
                            and the reference vLLM setup scripts
    claims/claim{1,2,3}/    claim.txt, run.sh, expected/ per functional claim
    submission/             AE submission metadata and packaging helper

    ARTIFACT_SCOPE.md       paper-to-artifact claim mapping
    EXPECTED_OUTPUT.md      consolidated expected-output reference
    HARDWARE_REQUIREMENTS.md  reference vs. minimum hardware
    VLLM_SETUP.md           self-hosted vLLM setup, end to end (optional)
    DATA_PROVENANCE.md      data sources, collection, licensing
    ETHICS.md               ethical considerations
    THIRD_PARTY_NOTICES.md  third-party components and terms
    PUBLIC_RELEASE.md       public availability statement
    LICENSE, NOTICE, LICENSES/   full license texts

Deliberately NOT included: model weights, a running vLLM service, GPU hardware,
credentials, generated results, and Python environments.

--------------------------------------------------------------------------------
CLAIMS
--------------------------------------------------------------------------------

    claim1  ANCHOR processes a CTI report end to end and emits a structurally
            valid ontology-aligned graph (UCO).
            Paper: Section IV, Figure 2.

    claim2  The same executable is schema-switchable across UCO, STIX, and
            MALOnt via --schema.
            Paper: Section IV-A principle (i); Table I.

    claim3  The distributed package is internally complete: 149 aligned inputs,
            149 records per schema, all ontologies parse, all sources parse.
            Deterministic; no endpoint required.
            Paper: Section V evaluation corpus.

All three are FUNCTIONAL claims. None asserts a paper table value.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

Client (this is all that is required):
    Linux or macOS, Python 3.11 / 3.12 / 3.13 (auto-detected), Git,
    outbound HTTPS.
    About 8 GB RAM and 5 GB free storage. The first run downloads the
    all-MiniLM-L6-v2 embedding model.
    No GPU.

Optional, only to self-host the paper's reference model:
    An NVIDIA GPU server with at least 64 GB of accelerator-addressable VRAM or
    coherent unified memory. See HARDWARE_REQUIREMENTS.md and VLLM_SETUP.md.

--------------------------------------------------------------------------------
CONTACT
--------------------------------------------------------------------------------

ACSAC artifact evaluation is single-blind and all author-reviewer communication
must go through HotCRP. Please raise any issue as a HotCRP comment. One
knowledgeable author monitors HotCRP throughout the evaluation period.
