# Public availability statement

This artifact is **publicly and permanently available**, not conditionally
released after evaluation.

| | |
| --- | --- |
| Public source repository | <https://github.com/seonwookim92/acsac-ANCHOR> |
| Permanent archive (DOI) | `PENDING_ZENODO_DOI` |
| License — software | Apache-2.0 |
| License — author-created annotations | CC BY 4.0 |

> The archival DOI is filled in once the Zenodo deposit is published, before the
> artifact submission is marked ready for review.

## What is released

- The complete ANCHOR source: the extraction pipeline, hybrid ontology
  discovery, SHACL validation and correction, and the graph exporters.
- The ontology discovery MCP server, reusable on its own for general OWL/SHACL
  exploration.
- The 149 CTINexus report texts used as input, under their upstream terms.
- The author-created UCO, STIX, and MALOnt records — 149 per schema — under
  CC BY 4.0.
- The packaged UCO, STIX, and MALOnt schema files, under their upstream terms.
- The original evaluation utilities.
- Installation, claim, and packaging scripts, plus all documentation in this
  package.
- `paper.pdf`, the accompanying paper, for convenience when reading the claims.

## What is not released, and why

- **Qwen model weights.** Distributed by their upstream publisher, not
  redistributed here. See `VLLM_SETUP.md`.
- **A running vLLM service and the GPU hardware behind it.** Not redistributable.
  This does not block evaluation or reuse: ANCHOR resolves its chat model at
  runtime from `LLM_PROVIDER`, so it runs against any OpenAI-compatible or
  Anthropic endpoint with no GPU. See `infrastructure/constraints`.
- **Credentials and API keys.** Never committed. `artifact/.env` is
  git-ignored; `artifact/.env.example` is the template.
- **Generated outputs and Python environments.** Reproduced locally by
  `./install.sh` and the claim scripts.

## Archival commitment

The authors commit to keeping the artifact permanently available online, as
required by the ACSAC call for paper artifacts. The archival deposit carries a
DOI and is immutable; the source repository may receive post-publication fixes
and improvements.
