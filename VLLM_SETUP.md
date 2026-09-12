# Self-hosted vLLM setup (optional)

## Is this document for you?

**Optional.** This is Option A in [infrastructure/access](infrastructure/access):
self-hosting the model on your own NVIDIA GPU server. It reproduces the
*reference* deployment used for the paper's measurements.

It is **not** a prerequisite for artifact evaluation:

- `claims/claim3` needs no LLM endpoint at all.
- `claims/claim1` and `claims/claim2` run fine against a hosted API
  (`LLM_PROVIDER=openai` or `anthropic`) from a laptop with no GPU — Option B in
  [infrastructure/access](infrastructure/access).

Follow this guide if you have a suitable GPU server and want to run against the
authors' exact model, if you want to keep report text on your own hardware (the
privacy property of paper Challenge 3), or if you want to serve some other
OpenAI-compatible model locally.

The artifact distributes no model weights, no container image, and no running
service. It pins and documents them so you can obtain them yourself.

## What gets deployed

```text
model:            Qwen/Qwen3.5-35B-A3B-FP8
revision:         9d1823d2dee688a6b25e77009dc727688c44936e
served name:      qwen3.5-35b
external port:    8081
max context:      262144
GPU memory util:  0.60
container image:  vllm-node:latest
image source:     https://github.com/eugr/spark-vllm-docker
```

Per-shard SHA-256 checksums are in
[infrastructure/model_snapshot.json](infrastructure/model_snapshot.json).
Hardware detail is in [infrastructure/resources](infrastructure/resources) and
[HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md).

## Prerequisites

- **Linux on an NVIDIA GPU server.** The exact reference path is an ARM64 DGX
  Spark / GB10 system with 128 GB coherent unified memory. The minimum practical
  capacity for this model is 64 GB of accelerator-addressable memory.
- **NVIDIA driver and NVIDIA Container Toolkit**, configured so Docker accepts
  `--gpus all`.
- **Docker and Git.**
- **Outbound HTTPS** to GitHub, Docker Hub, and Hugging Face.
- **At least 100 GB free disk** for the container image, the 37.5 GB FP8 model
  snapshot, caches, and startup compilation artifacts.
- **No Hugging Face token is needed** — the model is public.

All commands below are run **from the package root**.

## 1. Prepare the `vllm-node` image

```bash
./infrastructure/prepare_vllm_node.sh
```

The script checks local tooling, clones the public build project at a pinned
commit, runs its documented single-node `build-and-copy.sh` workflow, and
verifies that `vllm-node:latest` exists. It records the source commit, Docker
image ID, repo digests where available, architecture, and GPU inventory to
`results/vllm_server_setup.txt`.

The external build checkout is placed *beside* the package directory (in
`../anchor-vllm-runtime`), not inside it, so it cannot accidentally enter the
release archive.

To review every command without cloning, pulling, or building anything:

```bash
ANCHOR_VLLM_SETUP_DRY_RUN=1 ./infrastructure/prepare_vllm_node.sh
```

Overrides, if your setup differs:

```bash
SPARK_VLLM_URL=...          # upstream repository
SPARK_VLLM_REVISION=...     # upstream commit
ANCHOR_VLLM_SETUP_ROOT=...  # where the build checkout goes
VLLM_IMAGE=...              # local image tag to verify
```

If you already have a working vLLM image for your hardware, skip this step
entirely and pass its tag as `VLLM_IMAGE` in step 2. The `spark-vllm-docker`
project targets DGX Spark / GB10; on other NVIDIA hardware, using a compatible
stock vLLM image is an infrastructure port rather than an exact reproduction of
the reference server, which is fine for the functional claims.

## 2. Start the server

In terminal A, from the package root:

```bash
mkdir -p results
./infrastructure/start_vllm_qwen35.sh 2>&1 | tee results/vllm_server_startup.log
```

On first launch, vLLM downloads the pinned model into the mounted Hugging Face
cache. Expect this to take a while — it is 37.5 GB.

The script applies the reference reasoning parser, tool-calling parser,
prefix-cache, and multimodal-limit flags. **Tool calling must be enabled**; the
ANCHOR mapper drives the ontology MCP server through multi-step tool use and
will not work against a server without it.

Override only deployment-specific values:

```bash
HF_CACHE_DIR=/data/hf-cache \
VLLM_IMAGE=vllm-node:latest \
./infrastructure/start_vllm_qwen35.sh
```

On a 64 GB server rather than the 128 GB reference, raise the allocation
fraction and shorten the context:

```bash
GPU_UTIL=0.85 MAX_LEN=131072 ./infrastructure/start_vllm_qwen35.sh
```

That smaller configuration is suitable for the functional claims. It is not
presented as a paper-equivalent runtime configuration — for a paper-value
comparison the model revision and inference flags should not be changed. Such a
comparison is outside the requested badge scope.

To print the complete Docker command without starting anything:

```bash
ANCHOR_VLLM_DRY_RUN=1 ./infrastructure/start_vllm_qwen35.sh
```

Keep the startup log with your results: the OpenAI-compatible `/v1/models`
response does not necessarily expose the model revision.

## 3. Check the API

In terminal B on the same server:

```bash
curl -fsS http://127.0.0.1:8081/v1/models
```

The response must contain the served ID `qwen3.5-35b` — or whatever name you
started the server with. Do not start a claim run until the startup log reports
that the API is ready.

## 4. Connect the ANCHOR client

```bash
cp artifact/.env.example artifact/.env
```

> `artifact/.env.example` is a dotfile. It is in the repository and the release
> archive even though `ls` hides it; use `ls -a artifact/`.

Set the vLLM block in `artifact/.env`:

```env
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8081/v1
VLLM_API_KEY=dummy
VLLM_LLM_MODEL=qwen3.5-35b
```

If ANCHOR runs on a different machine from the server, replace `127.0.0.1` with
that server's address and protect the endpoint with your normal VPN, firewall,
or SSH-tunnel policy. **Never commit `artifact/.env`** — it is git-ignored.

Verify the client can reach the endpoint:

```bash
artifact/.venv/bin/python artifact/scripts/check_endpoint.py
```

Expected:

```text
[endpoint] LLM_PROVIDER=vllm  model='qwen3.5-35b'
PASS: vllm endpoint http://127.0.0.1:8081/v1 serves 'qwen3.5-35b'
```

If the configured model is not served, the error lists the names the server does
serve. No model name is pinned by the check — `VLLM_LLM_MODEL` may be anything
your server exposes.

## 5. Run a claim

```bash
./claims/claim1/run.sh      # one report, UCO
./claims/claim2/run.sh      # schema switch, STIX and MALOnt
```

See [EXPECTED_OUTPUT.md](EXPECTED_OUTPUT.md) and each
`claims/claimN/expected/README.txt` for acceptance criteria.

## Serving a different model

Nothing in the client is tied to Qwen. To serve something smaller on a modest
GPU, start any OpenAI-compatible server with tool calling enabled and point
`VLLM_BASE_URL` / `VLLM_LLM_MODEL` at it. The functional claims assert
structural validity, not the paper's numbers, so a substituted model satisfies
them — it changes the extracted content, not the structural contract being
verified. See [infrastructure/constraints](infrastructure/constraints).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `could not select device driver` | Install/configure the NVIDIA Container Toolkit and confirm Docker supports `--gpus all`. |
| `vllm-node:latest` not found | Re-run `prepare_vllm_node.sh` and read its output before starting the model. |
| Out of memory at startup | Stop other GPU workloads, raise `GPU_UTIL`, or lower `MAX_LEN`. The Qwen model card recommends reducing the context window on OOM, retaining at least 128K where possible. |
| Model alias missing from `/v1/models` | Confirm the server command contains `--served-model-name`, and that `VLLM_LLM_MODEL` matches it. |
| Client reports connection refused | The server is not up yet, or `VLLM_BASE_URL` points at the wrong host/port. Check the startup log. |
| Mapper stalls or produces an empty graph | The server was started without tool-calling support, or the model is too small to sustain multi-step tool use. |
| Revision uncertainty | Compare the startup log against `infrastructure/model_snapshot.json` and keep both with your run outputs. |
