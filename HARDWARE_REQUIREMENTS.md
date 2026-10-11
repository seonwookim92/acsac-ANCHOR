# Hardware requirements

## Summary

| Purpose | Hardware |
| --- | --- |
| **Artifact evaluation** (all three claims in `claims/`) | A commodity Linux or macOS client. **No GPU.** |
| Self-hosting the paper's reference model | An NVIDIA GPU server with ≥ 64 GB of accelerator-addressable memory. Optional. |
| Reproducing the paper's reported numbers | The reference configuration below. Outside the requested badge scope. |

**No GPU is required to evaluate this artifact.** ANCHOR resolves its chat model
at runtime from `LLM_PROVIDER`, so `claim1` and `claim2` can be executed against
a hosted OpenAI-compatible or Anthropic endpoint from a laptop, and `claim3` is
fully deterministic and needs no endpoint at all. The GPU requirements below
apply only to a reviewer or future user who chooses to self-host the paper's
reference model. See `infrastructure/constraints`.

## Client system — the only hard requirement

- Linux or macOS with Python 3.11, 3.12, or 3.13, Git, and outbound HTTPS
  access. `install.sh` auto-detects a supported interpreter.
- 8 GB RAM or more.
- 5 GB free storage for the default CPU evaluation environment, installer
  download cache, and embedding-model cache.
- Network access to whatever LLM endpoint is configured in `artifact/.env`.

The ANCHOR client and the local ontology MCP process do not require a GPU. The
`all-MiniLM-L6-v2` embedding model runs on CPU by default
(`EMBEDDING_DEVICE=cpu`). On Linux, `install.sh` explicitly installs PyTorch's
CPU-only wheel so that the client does not download the unused CUDA runtime.
On a clean Ubuntu 25.10 x86-64 image with Python 3.13.7, the resulting virtual
environment measured 1.47 GB and contained no `nvidia-*` packages; the embedding
model added about 92 MB and pip's retained download cache measured 351 MB. The
package, environment, pip cache, and embedding cache therefore occupied about
1.92 GB in total. The 5 GB requirement leaves more than 3 GB of headroom for
temporary installer files.

The 5 GB figure applies only to the default CPU evaluation installation. An
optional CUDA-enabled client can be installed on Linux with
`ANCHOR_TORCH_VARIANT=cuda ./install.sh`, followed by
`EMBEDDING_DEVICE=cuda` in `artifact/.env`. PyTorch then installs its CUDA
runtime packages. In the same clean Ubuntu 25.10/Python 3.13.7 environment, the
CUDA virtual environment measured 5.52 GB and the retained pip cache measured
2.89 GB; including the package and embedding cache, the total was about 8.51 GB.
Allow at least 10 GB free disk for that optional environment. This client-side
CUDA option is distinct from the reference LLM server below.

## Reference research environment

The ANCHOR experiments were conducted on an **ASUS Ascent GX10** desktop AI
system based on the **NVIDIA GB10 Grace Blackwell Superchip**, with **128 GB of
coherent unified system memory** shared by the Grace CPU and Blackwell GPU.

ASUS product specification:
https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/

A local deployment was used deliberately, not incidentally: paper Challenge 3 is
about supporting privacy-preserving CTI analysis with locally deployed
open-source LLMs, so the reference configuration self-hosts rather than calling
an enterprise API.

## Minimum environment for self-hosting the local LLM

The reference model is `Qwen/Qwen3.5-35B-A3B-FP8`. Its official repository
reports approximately **37.5 GB of model files**, 35B total parameters with 3B
activated, and a native context length of 262,144 tokens:

https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8

Model files are not the complete runtime memory requirement. vLLM also needs
accelerator memory for loading and quantization overhead, execution workspaces,
and the KV cache. Accordingly:

- **Minimum practical capacity:** 64 GB or more of accelerator-addressable
  memory, either dedicated GPU VRAM or coherent unified memory available to the
  GPU.
- **Reference/recommended capacity:** 128 GB of coherent unified memory, as on
  the ASUS Ascent GX10 used in the research.

The 64 GB value is a conservative requirement inferred from the 37.5 GB FP8
snapshot plus vLLM runtime and cache headroom; it is not a vendor-published
fixed minimum. On a 64 GB system, `GPU_UTIL` may need to be increased and
`MAX_LEN` reduced from the paper-reference values. The Qwen model card
recommends reducing the context window on out-of-memory errors, and retaining at
least 128K tokens where possible.

The provided launch script defaults to the paper-reference configuration:

```text
GPU_UTIL=0.60
MAX_LEN=262144
```

This exact configuration was validated on the 128 GB unified-memory GX10. A
smaller-memory deployment is suitable for a functional check but is not asserted
to reproduce the reference server configuration or the paper's measurements.

Ordinary host RAM attached to a discrete GPU is not interchangeable with VRAM.
The memory must be directly usable by the accelerator, as dedicated VRAM or a
supported coherent unified-memory architecture.

## If you do not have such a server

You do not need one. Use a hosted provider for the functional claims:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_LLM_MODEL=gpt-4o
```

or

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_LLM_MODEL=claude-sonnet-4-6
```

A smaller open-weights model served by vLLM on a modest GPU also works, provided
it supports multi-step tool calling. Substituting a model exercises the same
code path; it does not reproduce the paper's numbers, which is outside the
requested badge scope.
