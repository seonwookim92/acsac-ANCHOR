"""
LLM provider factory.

Reads LLM_PROVIDER from environment and returns the appropriate
LangChain chat model instance.

  LLM_PROVIDER=anthropic  → ChatAnthropic  (ANTHROPIC_API_KEY required)
  LLM_PROVIDER=openai     → ChatOpenAI     (OPENAI_API_KEY required)
  LLM_PROVIDER=vllm       → ChatOpenAI with custom base_url
                             (VLLM_BASE_URL + VLLM_API_KEY required)

LLM_MAX_TOKENS: max output tokens (default 2048). Critical for vLLM —
  without a limit the model may generate indefinitely.
"""

import os
import sys
from langchain_core.language_models.chat_models import BaseChatModel

_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_MAPPER_MAX_TOKENS = 4096
_DEFAULT_OPENAI_MAX_TOKENS = 16384   # reasoning models consume tokens on <think>; needs headroom


def _max_tokens(env_key: str = "LLM_MAX_TOKENS", default: int = _DEFAULT_MAX_TOKENS) -> int:
    return int(os.getenv(env_key, default))


_MODEL_ENV_KEY = {
    "anthropic": "ANTHROPIC_LLM_MODEL",
    "openai":    "OPENAI_LLM_MODEL",
    "vllm":      "VLLM_LLM_MODEL",
}


_MIN_MAX_TOKENS = 1024


def _is_max_tokens_error(exc: Exception) -> bool:
    """True when a provider rejected the requested output-token budget as too large.

    MAPPER_MAX_TOKENS is a single number applied to whichever model the evaluator
    configured, but every model has its own output ceiling (gpt-4o allows 16384,
    gpt-4-turbo 4096, several Anthropic models 8192). Asking for more than the
    ceiling is an HTTP 400, worded differently by each provider.
    """
    text = str(exc).lower()
    if "max_tokens" not in text and "max_completion_tokens" not in text \
            and "max output tokens" not in text:
        return False
    markers = ("less than", "at most", "maximum", "exceeds", "too large",
               "greater than", "must be", "invalid", "supports at most")
    return any(marker in text for marker in markers)


def _halve_max_tokens(model, exc: Exception) -> bool:
    """Halve the model's output budget and report it. False once the floor is hit."""
    current = getattr(model, "max_tokens", None)
    if not isinstance(current, int) or current <= _MIN_MAX_TOKENS:
        return False
    reduced = max(_MIN_MAX_TOKENS, current // 2)
    print(
        f"WARNING: the endpoint rejected max_tokens={current} "
        f"({str(exc)[:120]}); retrying with {reduced}. "
        f"Set MAPPER_MAX_TOKENS in .env to a value your model accepts.",
        file=sys.stderr,
    )
    model.max_tokens = reduced  # type: ignore[assignment]
    return True


def _is_unknown_argument_error(exc: Exception) -> bool:
    """True when a server rejected a non-OpenAI-standard body field.

    Strict OpenAI-compatible servers answer HTTP 400 with wording such as
    "Unrecognized request argument supplied: chat_template_kwargs",
    "unknown field", or "extra fields not permitted".
    """
    text = str(exc).lower()
    if "chat_template_kwargs" in text:
        return True
    # Deliberately narrow. "invalid_request_error" is OpenAI's generic type for
    # every bad request, so matching it would drop extra_body on unrelated 400s
    # (an oversized max_tokens, for one) and emit a misleading warning.
    markers = ("unrecognized request argument", "unknown field", "unexpected keyword",
               "extra fields not permitted", "additionalproperties", "unknown parameter",
               "unrecognized keys")
    return any(marker in text for marker in markers)


def _build_vllm_chat_model():
    """ChatOpenAI that survives servers rejecting vLLM's `chat_template_kwargs`.

    Defined lazily so that importing this module does not require langchain_openai.
    """
    from langchain_openai import ChatOpenAI

    class _VLLMChat(ChatOpenAI):
        """Drops `extra_body` and retries once if the server rejects it."""

        def _drop_extra_body(self) -> bool:
            if not getattr(self, "extra_body", None):
                return False
            print(
                "WARNING: the endpoint rejected vLLM's 'chat_template_kwargs' extension; "
                "retrying without it. Set VLLM_EXTRA_BODY=false in .env to suppress this.",
                file=sys.stderr,
            )
            self.extra_body = None  # type: ignore[assignment]
            return True

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            while True:
                try:
                    return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except Exception as exc:
                    if _is_max_tokens_error(exc) and _halve_max_tokens(self, exc):
                        continue
                    if _is_unknown_argument_error(exc) and self._drop_extra_body():
                        continue
                    raise

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            while True:
                try:
                    return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except Exception as exc:
                    if _is_max_tokens_error(exc) and _halve_max_tokens(self, exc):
                        continue
                    if _is_unknown_argument_error(exc) and self._drop_extra_body():
                        continue
                    raise

    return _VLLMChat


def _VLLMChatModel(**kwargs):  # noqa: N802 - factory kept call-compatible with a class
    return _build_vllm_chat_model()(**kwargs)


def get_llm(temperature: float = 0, max_tokens: int | None = None,
            thinking: bool = False) -> BaseChatModel:
    """
    thinking=True  : enable Qwen3 reasoning (vLLM only).
                     Use for the mapper agent — needs complex multi-step tool use.
                     Requires large max_tokens (thinking tokens count toward budget
                     but are stripped by vLLM --reasoning-parser before returning).
    thinking=False : disable thinking (default).
                     Use for the extractor — simple JSON generation, fast.

    Provider/model resolution order:
      1. LLM_PROVIDER env var (set by --llm CLI flag or .env default)
      2. Per-provider model env var: ANTHROPIC_LLM_MODEL / OPENAI_LLM_MODEL / VLLM_LLM_MODEL
         (set by --llm-model CLI flag or .env)
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    model_key = _MODEL_ENV_KEY.get(provider, "VLLM_LLM_MODEL")
    model = os.getenv(model_key, "")

    if provider == "anthropic":
        if max_tokens is None:
            max_tokens = _max_tokens()
        from langchain_anthropic import ChatAnthropic

        class _AnthropicChatModel(ChatAnthropic):
            """ChatAnthropic that backs off when the model's output ceiling is lower.

            MAPPER_MAX_TOKENS is one number applied to whichever model is
            configured, but Anthropic models differ in their maximum output
            tokens, and exceeding it is a hard 400 on the first mapper call.
            """

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                while True:
                    try:
                        return super()._generate(messages, stop=stop,
                                                 run_manager=run_manager, **kwargs)
                    except Exception as exc:
                        if _is_max_tokens_error(exc) and _halve_max_tokens(self, exc):
                            continue
                        raise

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                while True:
                    try:
                        return await super()._agenerate(messages, stop=stop,
                                                        run_manager=run_manager, **kwargs)
                    except Exception as exc:
                        if _is_max_tokens_error(exc) and _halve_max_tokens(self, exc):
                            continue
                        raise

        return _AnthropicChatModel(
            model=model or "claude-sonnet-4-6",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if max_tokens is None:
            max_tokens = _max_tokens("OPENAI_MAX_TOKENS", _DEFAULT_OPENAI_MAX_TOKENS)

        class _OpenAIChatModel(ChatOpenAI):
            """ChatOpenAI that retries without temperature if the API rejects it."""

            def _create_chat_result(self, response, generation_info=None):
                return super()._create_chat_result(response, generation_info)

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                while True:
                    try:
                        return super()._generate(messages, stop=stop,
                                                 run_manager=run_manager, **kwargs)
                    except Exception as e:
                        if ("temperature" in str(e) and "unsupported" in str(e).lower()
                                and self.temperature is not None):
                            self.temperature = None  # type: ignore[assignment]
                            continue
                        if _is_max_tokens_error(e) and _halve_max_tokens(self, e):
                            continue
                        raise

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                while True:
                    try:
                        return await super()._agenerate(messages, stop=stop,
                                                        run_manager=run_manager, **kwargs)
                    except Exception as e:
                        if ("temperature" in str(e) and "unsupported" in str(e).lower()
                                and self.temperature is not None):
                            self.temperature = None  # type: ignore[assignment]
                            continue
                        if _is_max_tokens_error(e) and _halve_max_tokens(self, e):
                            continue
                        raise

        return _OpenAIChatModel(
            model=model or "gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "vllm":
        if max_tokens is None:
            max_tokens = _max_tokens()
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "dummy")
        kwargs: dict = dict(
            model=model or "default",
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # VLLM_THINKING=false disables thinking globally (overrides thinking=True callers).
        # Useful for models that don't support reasoning or when speed is preferred.
        #
        # The switch is vLLM's `chat_template_kwargs` extension, which is NOT part of
        # the OpenAI schema. Tolerant servers ignore it (in which case the toggle
        # simply has no effect), strict ones answer HTTP 400. VLLM_EXTRA_BODY=false
        # suppresses it up front; otherwise the model below drops it and retries
        # automatically the first time a server rejects it, so any
        # OpenAI-compatible endpoint works without reconfiguration.
        vllm_thinking_enabled = os.getenv("VLLM_THINKING", "true").lower() != "false"
        extra_body_enabled = os.getenv("VLLM_EXTRA_BODY", "true").lower() != "false"
        if extra_body_enabled and not (thinking and vllm_thinking_enabled):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return _VLLMChatModel(**kwargs)

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Valid values: anthropic, openai, vllm"
    )
