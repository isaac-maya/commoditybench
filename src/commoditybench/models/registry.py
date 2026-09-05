"""A small registry mapping friendly names to configured model adapters.

The CLI accepts these names with ``--models``. Add an entry here to enroll a new model
in the benchmark; no other code needs to change. Model ids are the provider's current
strings — update them as providers ship new versions.
"""

from __future__ import annotations

import os
from typing import Callable

from .base import ClassifierModel

# Endpoint for the self-hosted Qwen3-235B vLLM server. It is account/deploy-specific and
# ephemeral (a RunPod pod or serverless endpoint that gets torn down to stop GPU billing),
# so it is read from the environment rather than hard-coded. Set QWEN3_235B_BASE_URL to a
# running OpenAI-compatible endpoint (launched with --enable-auto-tool-choice
# --tool-call-parser hermes) before running the qwen3-235b model.
QWEN3_235B_BASE_URL = os.environ.get(
    "QWEN3_235B_BASE_URL", "https://api.runpod.ai/v2/REPLACE_WITH_ENDPOINT_ID/openai/v1"
)


def _anthropic(model_id: str, **kw) -> Callable[[], ClassifierModel]:
    from .anthropic_model import AnthropicModel

    return lambda name: AnthropicModel(name=name, model_id=model_id, **kw)


def _openai(model_id: str, **kw) -> Callable[[], ClassifierModel]:
    from .openai_model import OpenAIModel

    return lambda name: OpenAIModel(name=name, model_id=model_id, **kw)


def _gemini(model_id: str, **kw) -> Callable[[], ClassifierModel]:
    from .gemini_model import GeminiModel

    return lambda name: GeminiModel(name=name, model_id=model_id, **kw)


# name -> factory(name) -> ClassifierModel. Edit ids here as providers update models.
_REGISTRY: dict[str, Callable[[str], ClassifierModel]] = {
    # --- Anthropic (Claude) ---
    # Opus generation ladder for the cross-generation (METR-style) trendline. Each
    # generation is run in its STRONGEST NATIVE reasoning config — the reasoning API
    # surface diverges across generations, so a single config can't drive all five:
    #   * 4.1 (2025-08): extended thinking only; no `effort` parameter (sending it 400s).
    #   * 4.5 (2025-11): extended thinking; `effort` supported (adaptive thinking 400s).
    #   * 4.6 (2026-02), 4.7 (2026-04), 4.8 (2026-05): adaptive thinking + `effort`.
    # Verified empirically (see results/generation_ab_findings.md). `max_tokens=8192` on
    # all five gives reasoning headroom so no generation is truncated (a confound that
    # would otherwise penalise the newer adaptive models). This is a capability snapshot
    # in each model's native config — NOT an equalized comparison.
    # See results/generation_trendline_findings.md for the writeup + trendline.
    "claude-opus-4-1": _anthropic(
        "claude-opus-4-1", thinking_mode="extended", effort=None,
        max_tokens=8192, thinking_budget=6000,
    ),
    "claude-opus-4-5": _anthropic(
        "claude-opus-4-5", thinking_mode="extended", effort="high",
        max_tokens=8192, thinking_budget=6000,
    ),
    "claude-opus-4-6": _anthropic(
        "claude-opus-4-6", thinking_mode="adaptive", effort="high", max_tokens=8192,
    ),
    "claude-opus-4-7": _anthropic(
        "claude-opus-4-7", thinking_mode="adaptive", effort="high", max_tokens=8192,
    ),
    "claude-opus-4-8": _anthropic(
        "claude-opus-4-8", thinking_mode="adaptive", effort="high", max_tokens=8192,
    ),
    "claude-sonnet-4-6": _anthropic("claude-sonnet-4-6"),
    # --- OpenAI (GPT) ---
    # GPT-4o is a NON-reasoning chat model (temp 0, no tools were given in the round-1
    # comparison) — kept for back-compat / the original snapshot. For a fair peer to Opus
    # 4.8 (a high-effort reasoning model), use a GPT-5 reasoning id below: these set
    # `reasoning_effort` (which switches the adapter to max_completion_tokens + no
    # temperature) and get the same `--agentic` CCL tools via OpenAIModel.classify_agentic.
    # `gpt-5.5` (2026-04-23) is OpenAI's strongest general reasoning model on this account,
    # the closest contemporary to Opus 4.8 (2026-05); `-pro` is the slower/pricier tier.
    "gpt-4o": _openai("gpt-4o"),
    "gpt-4o-mini": _openai("gpt-4o-mini"),
    "gpt-5.5": _openai("gpt-5.5", reasoning_effort="high", max_tokens=16384),
    "gpt-5.5-pro": _openai("gpt-5.5-pro", reasoning_effort="high", max_tokens=16384),
    "gpt-5-mini": _openai("gpt-5-mini", reasoning_effort="high", max_tokens=16384),
    # --- Google (Gemini) ---
    "gemini-2.0-flash": _gemini("gemini-2.0-flash"),
    "gemini-1.5-pro": _gemini("gemini-1.5-pro"),
    # --- Open weights via RunPod public endpoints (OpenAI-compatible) ---
    # RunPod's hosted LLM menu is small; Qwen3-32B is the strongest text model.
    # Run in thinking mode with the sampling settings the Qwen3 model card requires
    # (temp 0.6 / top_p 0.95 / top_k 20); greedy decoding is explicitly discouraged in
    # thinking mode (repetition/degradation). Thinking is on by default, so the response
    # carries a <think> trace and json_schema can't be enforced -> structured="none";
    # the base parser strips the trace and recovers the answer JSON. Budget is large
    # enough for the trace plus the answer.
    "qwen3-32b": _openai(
        "Qwen/Qwen3-32B-AWQ",
        base_url="https://api.runpod.ai/v2/qwen3-32b-awq/openai/v1",
        api_key_env="RUNPOD_API_KEY",
        structured="none",
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        extra_body={"top_k": 20, "min_p": 0},
    ),
    # Frontier open-weight: Qwen3-235B-A22B-Instruct-2507 (MoE, 22B active), 4-bit AWQ
    # (~124 GB) on a self-deployed RunPod serverless vLLM endpoint, TP=2 on 2xH100-80GB,
    # launched with --enable-auto-tool-choice + the hermes tool parser so it gets the same
    # --agentic CCL tools as Opus 4.8 (via OpenAIModel.classify_agentic chat-completions
    # loop). This is the Instruct (NON-thinking) variant, so no <think> trace; sampling
    # follows the model card (temp 0.7 / top_p 0.8 / top_k 20). The endpoint is ephemeral;
    # set QWEN3_235B_BASE_URL (see above) to your running vLLM endpoint before use.
    "qwen3-235b": _openai(
        "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
        base_url=QWEN3_235B_BASE_URL,
        api_key_env="RUNPOD_API_KEY",
        structured="none",
        temperature=0.7,
        top_p=0.8,
        max_tokens=16384,
        extra_body={"top_k": 20, "min_p": 0},
    ),
    # --- DeepSeek (open weights via the DeepSeek API, OpenAI-compatible) ---
    # DeepSeek-V4 serves an OpenAI-compatible API at https://api.deepseek.com (key:
    # DEEPSEEK_API_KEY). The API does NOT support strict json_schema output, so these use
    # structured="json_object" (the answer JSON is guaranteed valid and the shared parser
    # recovers the ECCN). LESSON LEARNED (2026-09-01 run): DeepSeek counts thinking
    # tokens against max_tokens, and at max_tokens=16384 the default (thinking ON) mode
    # truncated 3/34 answers to an empty trace ("...") — the reasoning_content fallback
    # cannot help when the trace itself is cut off. Configs below control for that:
    #   * flash = thinking OFF (fast/cheap lane; a non-reasoning comparator like GPT-4o)
    #   * pro   = thinking ON with a 64K budget (frontier lane, like GPT-5.5/Opus);
    #             pro was 503-overloaded on first attempt (2026-09-01) — rerun when up.
    "deepseek-v4-flash": _openai(
        "deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        structured="json_object",
        max_tokens=16384,
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "deepseek-v4-pro": _openai(
        "deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        structured="json_object",
        max_tokens=65536,
    ),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str) -> ClassifierModel:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(available_models())}"
        )
    return _REGISTRY[name](name)
