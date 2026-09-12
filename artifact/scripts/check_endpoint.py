#!/usr/bin/env python3
"""Verify that the configured LLM endpoint is reachable before a functional run.

ANCHOR is provider-agnostic: ``src/utils/llm.py`` builds its chat model from
``LLM_PROVIDER`` (``vllm``, ``openai``, or ``anthropic``). This preflight check
mirrors that resolution so an evaluator can exercise the pipeline with whatever
OpenAI-compatible or Anthropic endpoint they already have, instead of being
required to stand up the authors' reference server.

The model name is NOT pinned here. The paper's measurements were produced with
a locally served ``Qwen/Qwen3.5-35B-A3B-FP8``, but the Functional check only
asserts that the pipeline runs end-to-end and emits a structurally valid
ontology-aligned graph; any sufficiently capable tool-calling model may be
substituted for that purpose.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

MODEL_ENV_KEY = {
    "vllm": "VLLM_LLM_MODEL",
    "openai": "OPENAI_LLM_MODEL",
    "anthropic": "ANTHROPIC_LLM_MODEL",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def get_json(url: str, headers: dict[str, str], timeout: int = 15) -> dict:
    request = urllib.request.Request(url)
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def list_model_ids(payload: dict) -> set[str]:
    return {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def check_openai_compatible(base_url: str, api_key: str, model: str, label: str,
                            model_env: str) -> None:
    """Check an OpenAI-compatible /models endpoint (vLLM, OpenAI, gateways)."""
    base_url = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        payload = get_json(f"{base_url}/models", headers)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            fail(f"{label}: {base_url} rejected the API key (HTTP {exc.code})")
        fail(f"{label}: cannot query {base_url}/models: HTTP {exc.code} {exc.reason}")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        fail(f"{label}: cannot query {base_url}/models: {exc}")

    model_ids = list_model_ids(payload)
    if not model_ids:
        print(f"PASS: {label} endpoint {base_url} reachable "
              f"(served model list is empty; proceeding with {model!r})")
        return
    if model not in model_ids:
        fail(
            f"{label}: model {model!r} is not served by {base_url}.\n"
            f"       Available: {sorted(model_ids)}\n"
            f"       Set {model_env} in artifact/.env to one of the names above, "
            f"or serve the model under the name you configured."
        )
    print(f"PASS: {label} endpoint {base_url} serves {model!r}")


def check_anthropic(api_key: str, model: str) -> None:
    if not api_key or api_key.startswith("sk-ant-..."):
        fail("anthropic: ANTHROPIC_API_KEY is not set in .env")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    try:
        payload = get_json(f"{base_url}/v1/models", headers)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            fail(f"anthropic: {base_url} rejected ANTHROPIC_API_KEY (HTTP {exc.code})")
        fail(f"anthropic: cannot query {base_url}/v1/models: HTTP {exc.code} {exc.reason}")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        fail(f"anthropic: cannot query {base_url}/v1/models: {exc}")

    model_ids = list_model_ids(payload)
    if model_ids and model not in model_ids:
        print(f"WARNING: model {model!r} is not in the account's listed models; "
              f"proceeding anyway (aliases are not always listed).\n"
              f"         Listed: {sorted(model_ids)[:12]}\n"
              f"         If the run then fails with a model-not-found error, set "
              f"ANTHROPIC_LLM_MODEL in artifact/.env to one of the names above.",
              file=sys.stderr)
    print(f"PASS: anthropic endpoint {base_url} reachable with model {model!r}")


def probe_tool_calling() -> None:
    """Ask the configured model to make one tool call, through the real code path.

    ANCHOR's mapper drives the ontology MCP server through multi-step tool use. A
    model without solid tool-calling support runs to completion but yields an empty
    or near-empty graph -- after several minutes. This probe surfaces that in
    seconds.

    It is advisory: a WARNING, never a hard failure. A capable model can decline to
    call a tool on any single prompt, and blocking an evaluator on a false negative
    would be worse than letting the run proceed. Set ANCHOR_SKIP_TOOL_PROBE=1 to
    skip it.
    """
    if os.getenv("ANCHOR_SKIP_TOOL_PROBE", "").lower() in ("1", "true"):
        return

    sys.path.insert(0, str(ROOT))
    try:
        from langchain_core.tools import tool
        from src.utils.llm import get_llm

        @tool
        def search_ontology_class(term: str) -> str:
            """Look up an ontology class by a search term."""
            return "Malware"

        response = get_llm(max_tokens=256).bind_tools([search_ontology_class]).invoke(
            "Call the search_ontology_class tool with the term 'malware'. "
            "Respond only with the tool call."
        )
    except Exception as exc:
        print(f"WARNING: tool-calling probe could not run: {str(exc)[:200]}\n"
              f"         Continuing; the claim scripts will surface any real problem.",
              file=sys.stderr)
        return

    if getattr(response, "tool_calls", None):
        print(f"PASS: model performed a tool call "
              f"({response.tool_calls[0].get('name')})")
        return

    print(
        "WARNING: the model did not make a tool call when asked to.\n"
        "         ANCHOR's ontology mapper depends on multi-step tool calling, so a\n"
        "         model without solid tool-use support typically produces an empty or\n"
        "         near-empty graph and fails the claim's structural check.\n"
        "         If a claim run then fails, switch to a model with reliable tool use\n"
        "         (see PROJECT_README.md for models this pipeline has been run with),\n"
        "         and for a self-hosted server confirm it was started with tool calling\n"
        "         enabled (vLLM: --enable-auto-tool-choice --tool-call-parser ...).\n"
        "         Set ANCHOR_SKIP_TOOL_PROBE=1 to skip this probe.",
        file=sys.stderr,
    )


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)

    provider = os.getenv("LLM_PROVIDER", "vllm").lower().strip()
    if provider not in MODEL_ENV_KEY:
        fail(f"LLM_PROVIDER={provider!r} is not valid; use vllm, openai, or anthropic")

    model = os.getenv(MODEL_ENV_KEY[provider], "").strip()
    if not model:
        fail(f"{MODEL_ENV_KEY[provider]} is not set in .env for LLM_PROVIDER={provider}")

    print(f"[endpoint] LLM_PROVIDER={provider}  model={model!r}")

    if provider == "vllm":
        check_openai_compatible(
            os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8081/v1"),
            os.getenv("VLLM_API_KEY", ""),
            model,
            "vllm",
            "VLLM_LLM_MODEL",
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-..."):
            fail("openai: OPENAI_API_KEY is not set in .env")
        check_openai_compatible(
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key,
            model,
            "openai",
            "OPENAI_LLM_MODEL",
        )
    else:
        check_anthropic(os.getenv("ANTHROPIC_API_KEY", ""), model)

    probe_tool_calling()


if __name__ == "__main__":
    main()
