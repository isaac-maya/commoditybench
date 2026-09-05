"""CLI: run one or more models over a dataset, score them, and write results.

Examples
--------
    # List enrolled models
    commoditybench --list-models

    # Evaluate two models on the verified set, 8 concurrent calls each
    commoditybench --dataset data/questions.jsonl \\
        --models claude-opus-4-8 gpt-4o --verified-only --workers 8

    # A/B the RAG condition (uses the pure-Python BM25 retriever over the shipped
    # data/ccl/ccl_index.json — no Chroma index, no new dependencies)
    commoditybench --dataset data/questions.jsonl --models claude-opus-4-8 --rag

    # Abstention + calibration condition (zero-error brief). Schema-strict providers
    # (Anthropic/Gemini/OpenAI json_schema) additionally need the env switch set
    # before start: COMMODITYBENCH_ABSTAIN_AWARE=1 -- see --help.
    commoditybench --dataset data/questions.jsonl --models deepseek-v4-flash --abstain

    # Equalized preset: matched inference settings + a config-deviation check
    commoditybench --dataset data/questions.jsonl \\
        --models deepseek-v4-flash deepseek-v4-pro --equalized

    # Per-call audit log (item, prompts, raw response, tokens, cost estimate)
    commoditybench ... --log-calls results/calls.jsonl

Outputs (under ``results/`` by default):
    - ``<run_id>__<model>.jsonl``  : per-question predictions + scores
    - ``<run_id>__summary.json``   : aggregate metrics per model
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .dataset import Question, load_questions
from .eccn import extract_eccn, score_prediction
from .models import ClassifierModel, build_model, available_models
from .models.base import _strip_reasoning, _try_json
from .prompts import ABSTAIN_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt

# -------------------------------------------------------------------------------------
# Per-call cost-estimate rates (USD per MTok). DeepSeek flash pricing is ~$0.28/M
# input class (observed on prior runs); the exact v4 price card is NOT verified here.
# These constants exist for BUDGET DISCIPLINE ONLY — an estimate, never a bill.
# Override with COMMODITYBENCH_INPUT_USD_PER_MTok / OUTPUT_... when the price card
# is known.
ASSUMED_USD_PER_MTok = {
    "input": float(os.environ.get("COMMODITYBENCH_INPUT_USD_PER_MTok", "0.28")),
    "output": float(os.environ.get("COMMODITYBENCH_OUTPUT_USD_PER_MTok", "1.10")),
}

#: The --equalized preset: matched inference settings across providers "where the API
#: permits" (their cross-model caveat: every model currently runs in its strongest
#: NATIVE config — nothing enforces matched settings; this preset + the deviation check
#: below are that enforcement). Reasoning is matched as DISABLED because that is the one
#: mode every enrolled family supports (a GPT-5.5/Opus/Qwen/DeepSeek all-reason preset
#: is not expressible for GPT-4o-class chat models). Prompt is matched at the run level:
#: one condition = one prompt template for every model in the run.
EQUALIZED_PRESET: dict = {
    "temperature": 0.0,
    "reasoning": "disabled",  # matched across providers where the API permits
    "max_tokens": 8192,
    "prompt": "same template for every model in the run (condition-level)",
    "note": "Defaults are UNCHANGED without --equalized; this dict is the declared "
    "contract the deviation check enforces per model.",
}


def _repo_commit() -> str:
    """Best-effort repo commit the run is based on (rule 6 pinning)."""
    env = os.environ.get("COMMODITYBENCH_REPO_COMMIT")
    if env:
        return env
    try:
        root = Path(__file__).resolve().parents[2]
        return (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - not running inside a git checkout
        return "unset"


def _abstain_env_active() -> bool:
    """True when COMMODITYBENCH_ABSTAIN_AWARE was set before process start."""
    return os.environ.get("COMMODITYBENCH_ABSTAIN_AWARE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _display_path(p) -> str:
    """Path for records in the run summary: repo-relative when possible.

    Absolute local paths in shipped summaries are meaningless to anyone outside
    the machine that ran them; anything under the repo root is recorded relative
    to it (repo convention: existing summaries say ``data/questions.jsonl``).
    """
    if not p:
        return ""
    try:
        root = Path(__file__).resolve().parents[2]
        return str(Path(p).resolve().relative_to(root))
    except (ValueError, OSError):
        return str(p)


def _model_is_schema_strict(model) -> bool:
    """Schema-strict adapters enforce the answer JSON server-side.

    Anthropic/Gemini use native structured outputs (schema captured at import);
    the OpenAI-compatible adapter is strict only in its ``json_schema`` mode
    (``json_object``/``none`` lanes never send a schema, so the prompt alone
    carries the abstain fields).
    """
    if type(model).__name__ in ("AnthropicModel", "GeminiModel"):
        return True
    return getattr(model, "structured", None) == "json_schema"


def _snapshot_model_config(model) -> dict:
    """Read a model instance's effective inference config (no secrets — env var NAMES
    only, never their values). Used for rule-6 config pinning and the equalized check."""
    cfg: dict = {
        "adapter": type(model).__name__,
        "name": getattr(model, "name", None),
        "model_id": getattr(model, "model_id", None),
    }
    for attr in (
        "temperature", "top_p", "reasoning_effort", "max_tokens", "structured",
        "extra_body", "thinking_mode", "effort", "thinking_budget", "base_url",
        "api_key_env",
    ):
        if hasattr(model, attr):
            cfg[attr] = getattr(model, attr)
    return cfg


def _apply_overrides(model, overrides: dict) -> list[str]:
    """Apply CLI/preset config overrides onto a model instance; warn per knob that an
    adapter does not expose. Mutating instance attributes works because the adapters
    read them per call — no registry change needed."""
    name = getattr(model, "name", type(model).__name__)
    warns: list[str] = []
    for attr, val in overrides.items():
        if hasattr(model, attr):
            setattr(model, attr, val)
        else:
            warns.append(
                f"[{name}] cannot set {attr}: {type(model).__name__} exposes no such option"
            )
    return warns


def _apply_equalized(model, preset: dict) -> list[str]:
    """Force the matched-settings preset onto a registered model instance.

    Reasoning-disabled handling is adapter-aware: OpenAI/OpenAI-compatible models get
    ``reasoning_effort=None`` plus the DeepSeek ``thinking: disabled`` body (DeepSeek
    API only); Anthropic gets ``thinking_mode="off"``. For endpoints where the adapter
    cannot force reasoning off, the returned warnings say so — and the deviation check
    will echo them as warnings on the run (never silently).
    """
    warns: list[str] = []
    warns += _apply_overrides(
        model,
        {"temperature": preset["temperature"], "max_tokens": preset["max_tokens"]},
    )
    if preset["reasoning"] != "disabled":
        return warns
    name = getattr(model, "name", type(model).__name__)
    if hasattr(model, "reasoning_effort"):
        model.reasoning_effort = None  # chat lane: no reasoning_effort knob sent
    base_url = str(getattr(model, "base_url", "") or "")
    if "deepseek.com" in base_url and hasattr(model, "extra_body"):
        eb = dict(model.extra_body or {})
        eb["thinking"] = {"type": "disabled"}
        model.extra_body = eb
    elif hasattr(model, "extra_body"):
        warns.append(
            f"[{name}] cannot force 'thinking: disabled': non-DeepSeek endpoint "
            f"({base_url or 'default'}); set the provider's reasoning-off parameter in "
            "the registry entry before an equalized run"
        )
    if hasattr(model, "thinking_mode"):  # Anthropic adapter
        model.thinking_mode = "off"
        if hasattr(model, "effort"):
            model.effort = None
    return warns


def _equalized_check(model, preset: dict) -> dict:
    """Compare a model's effective config to the preset. Returns the effective config
    plus a list of deviations — the harness CHECK that makes the old prose-only
    'not equalized' caveat enforceable. Warnings, not a hard stop."""
    eff = _snapshot_model_config(model)
    name = eff.get("name") or eff.get("model_id") or "?"
    devs: list[str] = []
    temp = eff.get("temperature")
    if temp is None:
        devs.append(f"[{name}] temperature not exposed by adapter — unverifiable")
    elif abs(float(temp) - float(preset["temperature"])) > 1e-9:
        devs.append(f"[{name}] temperature={temp} deviates from preset {preset['temperature']}")
    mt = eff.get("max_tokens")
    if mt is None:
        devs.append(f"[{name}] max_tokens not exposed by adapter — unverifiable")
    elif int(mt) != int(preset["max_tokens"]):
        devs.append(f"[{name}] max_tokens={mt} deviates from preset {preset['max_tokens']}")
    if preset["reasoning"] == "disabled":
        if eff.get("reasoning_effort") is not None:
            devs.append(
                f"[{name}] reasoning_effort={eff.get('reasoning_effort')} — preset disables reasoning"
            )
        eb = eff.get("extra_body") or {}
        thinking = eb.get("thinking") if isinstance(eb, dict) else None
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            pass  # DeepSeek thinking explicitly disabled
        elif eff.get("thinking_mode") not in (None, "off", False):
            devs.append(
                f"[{name}] thinking_mode={eff.get('thinking_mode')} — preset disables reasoning"
            )
        elif "deepseek.com" in str(eff.get("base_url", "")):
            devs.append(f"[{name}] DeepSeek extra_body does not disable thinking: {eb}")
        else:
            devs.append(
                f"[{name}] reasoning-off could not be verified for this endpoint/adapter "
                f"(extra_body={eb}) — confirm before citing as equalized"
            )
    return {"model": eff.get("name"), "effective": eff, "deviations": devs}


# -------------------------------------------------------------------------------------
# Abstain-condition row interpretation
#
# Counting rule (integrity): an abstention is verified against the RAW response —
# non-empty content AND an explicit "abstain": true field with "eccn": null.
# Empty/truncated/unparsable responses are logged as ERRORS, never counted as
# abstentions. Rows carry ``abstain`` / ``abstain_field`` / ``content_nonempty`` so an
# analysis can re-verify every claim against the per-call log (--log-calls).
# -------------------------------------------------------------------------------------


def _coerce_confidence(value) -> int | None:
    """Confidence must be an int in [0, 100]; anything else is not a usable signal."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v <= 100 else None


def _abstain_row_error(q: Question, exc_text: str, usage: dict | None = None) -> dict:
    """An API failure under the abstain condition: an error row, never an abstention."""
    usage = usage or {}
    return {
        "id": q.id, "item_name": q.item_name, "gold_eccn": q.gold_eccn,
        "predicted_eccn": "", "category": q.category, "verified": q.verified,
        "reasoning": "", "parsed_ok": False, "error": exc_text[:500],
        "n_tool_calls": 0, "usage": usage,
        "score": score_prediction("", q.gold_eccn).to_dict(),
        "abstain": False, "confidence": None, "abstain_field": False,
        "content_nonempty": False, "abstain_contradiction": False,
    }


def _abstain_row_interpret(q: Question, text: str, usage: dict) -> dict:
    """Interpret one raw model response under the abstain condition.

    Order of decisions:
      1. empty content -> ERROR (never an abstention);
      2. valid JSON: abstain:true + eccn:null + non-empty content -> ABSTENTION;
         eccn committed -> answered (an abstain:true that ALSO names an ECCN is a
         contradiction — the ECCN wins, flagged); eccn:null WITHOUT abstain:true ->
         ERROR (malformed under the extended schema);
      3. no JSON -> repo prose-fallback convention (a recovered ECCN is scored,
         parsed_ok=False); no ECCN recoverable -> ERROR, not an abstention.
    """
    clean = _strip_reasoning(text).strip()
    base = {
        "id": q.id, "item_name": q.item_name, "gold_eccn": q.gold_eccn,
        "predicted_eccn": "", "category": q.category, "verified": q.verified,
        "reasoning": "", "parsed_ok": False, "error": None, "n_tool_calls": 0,
        "usage": usage, "abstain": False, "confidence": None,
        "abstain_field": False, "content_nonempty": bool(clean),
        "abstain_contradiction": False,
    }
    if not clean:
        return {**base, "error": "empty_or_truncated_content",
                "score": score_prediction("", q.gold_eccn).to_dict()}

    obj = _try_json(clean)
    if obj is not None:
        confidence = _coerce_confidence(obj.get("confidence"))
        reasoning = str(obj.get("reasoning") or "")[:500]
        abstain_field = obj.get("abstain")
        eccn = obj.get("eccn")
        if isinstance(eccn, str):
            eccn_str = eccn.strip()
            nullish = eccn_str.lower() in ("", "null", "none")
        else:
            eccn_str, nullish = "", True
        if abstain_field is True and nullish:
            # Verified abstention: non-empty content + explicit abstain field.
            score = score_prediction("ABSTAIN", q.gold_eccn)
            return {**base, "predicted_eccn": "ABSTAIN", "reasoning": reasoning,
                    "parsed_ok": True, "abstain": True, "confidence": confidence,
                    "abstain_field": True, "score": score.to_dict()}
        if not nullish:
            # An ECCN string was committed — answered, even if abstain was also true.
            score = score_prediction(eccn_str, q.gold_eccn)
            return {**base, "predicted_eccn": eccn_str, "reasoning": reasoning,
                    "parsed_ok": True, "confidence": confidence,
                    "abstain_field": abstain_field is not None,
                    "abstain_contradiction": abstain_field is True,
                    "score": score.to_dict()}
        # eccn null/missing WITHOUT an explicit abstain field -> malformed.
        return {**base, "reasoning": reasoning, "confidence": confidence,
                "abstain_field": abstain_field is not None,
                "error": "eccn_null_without_abstain_field",
                "score": score_prediction("", q.gold_eccn).to_dict()}

    # No JSON: the repo convention recovers an ECCN from prose (parsed_ok=False and
    # still scored); prose declines without a field are NOT counted abstentions.
    recovered = extract_eccn(clean)
    if recovered.is_valid:
        score = score_prediction(str(recovered), q.gold_eccn)
        return {**base, "predicted_eccn": str(recovered), "reasoning": clean[:500],
                "parsed_ok": False, "score": score.to_dict()}
    return {**base, "reasoning": clean[:500],
            "error": "unparsed_no_answer_no_abstain_field",
            "score": score_prediction("", q.gold_eccn).to_dict()}


class _CallLog:
    """Per-call audit log (rule 6): one JSON line per API call — item, condition,
    exact prompts, raw response, tokens, and a cost ESTIMATE (rate assumptions above).
    Thread-safe for the concurrent runner. A 503 that ends the call after the adapter's
    retries appears as a line with ``error`` set; retries recovered inside the adapter
    (tenacity) are not individually visible — the line records the final outcome."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, **fields) -> None:
        tokens_in = int(fields.get("input_tokens") or 0)
        tokens_out = int(fields.get("output_tokens") or 0)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cost_estimate_usd": round(
                tokens_in / 1e6 * ASSUMED_USD_PER_MTok["input"]
                + tokens_out / 1e6 * ASSUMED_USD_PER_MTok["output"],
                6,
            ),
            **fields,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def _evaluate_model(
    model_name: str,
    questions: list[Question],
    *,
    workers: int,
    retriever=None,
    agentic: bool = False,
    ccl_index=None,
    out_dir: Path,
    run_id: str,
    model: ClassifierModel | None = None,
    abstain: bool = False,
    condition: str = "closed_book",
    call_log: _CallLog | None = None,
) -> dict:
    model = model or build_model(model_name)
    if agentic and not hasattr(model, "classify_agentic"):
        raise SystemExit(
            f"Model {model_name!r} does not support --agentic (no classify_agentic). "
            "The agentic CCL-navigation condition is implemented for the Anthropic adapter "
            "and the OpenAI / OpenAI-compatible adapter (GPT + vLLM/RunPod endpoints)."
        )
    rows: list[dict] = []

    def work(q: Question) -> dict:
        context = retriever.retrieve(q) if retriever is not None else None
        if abstain:
            system, user = (
                ABSTAIN_SYSTEM_PROMPT,
                build_user_prompt(q, context=context, abstain_aware=True),
            )
            try:
                text, usage = model._complete(system, user)
            except Exception as exc:  # noqa: BLE001 - per-item provider errors
                row = _abstain_row_error(q, f"{type(exc).__name__}: {exc}")
                if call_log is not None:
                    call_log.add(run_id=run_id, model=model_name, condition=condition,
                                 qid=q.id, item_name=q.item_name, system_prompt=system,
                                 user_prompt=user, raw_response="", parsed_ok=False,
                                 error=str(exc)[:500], input_tokens=0, output_tokens=0)
                return row
            row = _abstain_row_interpret(q, text, usage)
            if call_log is not None:
                call_log.add(run_id=run_id, model=model_name, condition=condition,
                             qid=q.id, item_name=q.item_name, system_prompt=system,
                             user_prompt=user, raw_response=text,
                             parsed_ok=row["parsed_ok"], error=row["error"],
                             input_tokens=usage.get("input_tokens", 0),
                             output_tokens=usage.get("output_tokens", 0))
            return row
        if agentic:
            from .ccl.tools import CCLToolbox

            pred = model.classify_agentic(q, toolbox=CCLToolbox(ccl_index))
            if call_log is not None:
                # Agentic runs interleave several tool turns; the log records the final
                # answer call (the per-tool calls are visible in the row's tool trace).
                call_log.add(run_id=run_id, model=model_name, condition=condition,
                             qid=q.id, item_name=q.item_name,
                             system_prompt="(agentic: AGENTIC_SYSTEM_PROMPT, multi-turn)",
                             user_prompt=build_user_prompt(q), raw_response=pred.raw_response,
                             parsed_ok=pred.parsed_ok, error=pred.error,
                             input_tokens=pred.usage.get("input_tokens", 0),
                             output_tokens=pred.usage.get("output_tokens", 0))
        else:
            pred = model.classify(q, context=context)
            if call_log is not None:
                call_log.add(run_id=run_id, model=model_name, condition=condition,
                             qid=q.id, item_name=q.item_name, system_prompt=SYSTEM_PROMPT,
                             user_prompt=build_user_prompt(q, context=context),
                             raw_response=pred.raw_response, parsed_ok=pred.parsed_ok,
                             error=pred.error,
                             input_tokens=pred.usage.get("input_tokens", 0),
                             output_tokens=pred.usage.get("output_tokens", 0))
        score = score_prediction(pred.predicted_eccn, q.gold_eccn)
        return {
            "id": q.id,
            "item_name": q.item_name,
            "gold_eccn": q.gold_eccn,
            "predicted_eccn": pred.predicted_eccn,
            "category": q.category,
            "verified": q.verified,
            "reasoning": pred.reasoning,
            "parsed_ok": pred.parsed_ok,
            "error": pred.error,
            "n_tool_calls": len(pred.usage.get("tool_calls", [])) if agentic else 0,
            "usage": pred.usage,
            "score": score.to_dict(),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, q): q for q in questions}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"  [{model_name}] {i}/{len(questions)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    rows.sort(key=lambda r: r["id"])
    out_path = out_dir / f"{run_id}__{model_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = _aggregate(model_name, rows)
    summary["config"] = _snapshot_model_config(model)
    return summary


def _aggregate(model_name: str, rows: list[dict]) -> dict:
    """Aggregate per-question rows into headline metrics for one model.

    Headline rates are computed over **all** ``n`` questions: an API error or an
    unparseable answer counts as wrong (0). This keeps two models comparable even when
    they fail on different items — a model that errors out on its hardest questions must
    not get a denominator made only of the questions it managed to answer. We also report
    ``exact_accuracy_attempted`` (over non-errored items only) for diagnostics, but it is
    never the headline.
    """
    n = len(rows) or 1  # guarded upstream (non-empty), but keep division safe
    errors = sum(1 for r in rows if r["error"])
    scored = [r["score"] for r in rows if not r["error"]]
    attempted = len(scored)

    def count(key: str) -> int:
        return sum(1 for s in scored if s[key])

    def rate_all(key: str) -> float:
        return round(count(key) / n, 4)  # errored/unparsed items contribute 0

    return {
        "model": model_name,
        "n": len(rows),
        "attempted": attempted,  # non-errored
        "errors": errors,
        "error_rate": round(errors / n, 4),
        "exact_accuracy": rate_all("exact_match"),  # over ALL n (headline)
        "eccn_accuracy": rate_all("eccn_match"),  # ignores subparagraph
        "group_accuracy": rate_all("group_match"),  # category + product group
        "category_accuracy": rate_all("category_match"),
        "parse_rate": rate_all("parsed"),  # fraction of all items that produced an answer
        "mean_grade": round(sum(s["grade"] for s in scored) / n, 4),
        "exact_accuracy_attempted": (
            round(count("exact_match") / attempted, 4) if attempted else 0.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="commoditybench", description=__doc__)
    parser.add_argument("--dataset", help="Path to a .jsonl dataset file.")
    parser.add_argument(
        "--models", nargs="+", help="Model names to evaluate (see --list-models)."
    )
    parser.add_argument("--out-dir", default="results", help="Where to write results.")
    parser.add_argument("--run-id", default="run", help="Prefix for output filenames.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent calls/model.")
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Only score human-verified questions (required for headline metrics).",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Inject retrieved CCL excerpts. Uses the pure-Python BM25 retriever over "
        "data/ccl/ccl_index.json (see --rag-index/--rag-top-k); the Chroma scaffolding "
        "in commoditybench.rag.retriever remains available for a future vector condition.",
    )
    parser.add_argument(
        "--rag-index",
        default=None,
        help="CCL index JSON for the BM25 retriever (default: data/ccl/ccl_index.json).",
    )
    parser.add_argument(
        "--rag-top-k", type=int, default=6, help="BM25 excerpts injected (--rag)."
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Let the model navigate the CCL via tools before answering (Anthropic + "
        "OpenAI/OpenAI-compatible adapters; requires data/ccl/ccl_index.json — build with "
        "commoditybench.ccl.parse_ecfr).",
    )
    parser.add_argument(
        "--abstain",
        action="store_true",
        help="Abstention + calibration condition (zero-error brief): abstain-instructed "
        "prompt; abstain:true+eccn:null = abstention, empty/truncated = error. Any "
        "registered provider. Schema-strict adapters (Anthropic/Gemini/OpenAI "
        "json_schema) additionally need COMMODITYBENCH_ABSTAIN_AWARE=1 set before "
        "start so the extended answer schema is enforced.",
    )
    parser.add_argument(
        "--equalized",
        action="store_true",
        help="Apply the matched-settings preset (temperature/reasoning/max_tokens/prompt, "
        "see EQUALIZED_PRESET) to every model and run the deviation CHECK: warnings are "
        "printed and recorded whenever a model's effective config deviates from the "
        "preset. Defaults are unchanged without the flag.",
    )
    parser.add_argument(
        "--log-calls",
        default=None,
        metavar="PATH",
        help="Append one JSON line per API call (item, condition, exact prompts, raw "
        "response, tokens, cost estimate) to PATH (audit log; also enables the "
        "reproducibility fields on the summary).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Explicit temperature override (pinned config; recorded per model). "
        "Ignored by reasoning-effort models whose API forbids temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Explicit max_tokens override (pinned config; recorded per model).",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List enrolled models and exit."
    )
    args = parser.parse_args(argv)

    if args.list_models:
        print("\n".join(available_models()))
        return 0

    if not args.dataset or not args.models:
        parser.error("--dataset and --models are required (or use --list-models).")

    # Load .env if present so keys come through without manual export.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    questions = load_questions(args.dataset, verified_only=args.verified_only)
    if not questions:
        print(
            "No questions loaded. If you used --verified-only, note that the bundled "
            "example set is intentionally unverified — see data/schema.md.",
            file=sys.stderr,
        )
        return 1

    if args.rag and args.agentic:
        parser.error("--rag and --agentic are mutually exclusive conditions.")
    if args.abstain and args.agentic:
        parser.error("--abstain and --agentic are mutually exclusive conditions.")

    if args.abstain and not _abstain_env_active():
        # Schema-strict adapters capture ANSWER_SCHEMA at import time; without the
        # env switch they enforce the 3-field schema and the model can never emit
        # abstain:true — the run would silently measure a condition that cannot
        # abstain. Fail loudly instead.
        strict = [
            name for name in args.models
            if _model_is_schema_strict(build_model(name))
        ]
        if strict:
            parser.error(
                "--abstain requires COMMODITYBENCH_ABSTAIN_AWARE=1 before start for "
                f"schema-strict model(s): {', '.join(strict)} "
                "(Anthropic/Gemini/OpenAI json_schema enforce the schema server-side "
                "and capture it at import; the extended abstain schema needs the env "
                "switch). DeepSeek/open json_object models do not need it."
            )

    retriever = None
    if args.rag:
        from .rag.retriever_bm25 import BM25CCLRetriever  # pure Python, zero deps

        kwargs: dict = {"top_k": args.rag_top_k}
        if args.rag_index:
            kwargs["index_path"] = args.rag_index
        retriever = BM25CCLRetriever(**kwargs).build()

    ccl_index = None
    if args.agentic:
        from .ccl import CCLIndex  # lazy: only needed for the agentic condition

        ccl_index = CCLIndex.load()  # shared read-only index across all questions

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    condition = []
    if args.abstain:
        condition.append("abstain")
    if args.rag:
        condition.append("rag")
    if args.agentic:
        condition.append("agentic")
    if not condition:
        condition.append("closed_book")
    if args.equalized:
        condition.append("equalized")
    condition_str = "+".join(condition)

    # Per-model instances with overrides applied (equalized preset and/or explicit
    # --temperature/--max-tokens), so the config snapshot records exactly what ran.
    overrides: dict = {}
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens
    prebuilt: dict[str, ClassifierModel] = {}
    equalized_checks: dict[str, dict] = {}
    if overrides or args.equalized:
        for model_name in args.models:
            model = build_model(model_name)
            warns = _apply_overrides(model, overrides)
            if args.equalized:
                warns += _apply_equalized(model, EQUALIZED_PRESET)
                equalized_checks[model_name] = _equalized_check(model, EQUALIZED_PRESET)
            for msg in warns:
                print(f"WARNING: {msg}", file=sys.stderr)
            prebuilt[model_name] = model
    if args.equalized:
        for name, chk in equalized_checks.items():
            if chk["deviations"]:
                print(
                    f"WARNING [equalized check] {name}: "
                    + "; ".join(chk["deviations"]),
                    file=sys.stderr,
                )
            else:
                print(
                    f"equalized check OK: {name} matches the preset "
                    f"(temp={EQUALIZED_PRESET['temperature']}, "
                    f"reasoning={EQUALIZED_PRESET['reasoning']}, "
                    f"max_tokens={EQUALIZED_PRESET['max_tokens']})",
                    file=sys.stderr,
                )

    call_log = _CallLog(args.log_calls) if args.log_calls else None

    print(
        f"Evaluating {len(args.models)} model(s) on {len(questions)} question(s)"
        f" [{condition_str}].",
        file=sys.stderr,
    )
    summaries = []
    for model_name in args.models:
        summaries.append(
            _evaluate_model(
                model_name,
                questions,
                workers=args.workers,
                retriever=retriever,
                agentic=args.agentic,
                ccl_index=ccl_index,
                out_dir=out_dir,
                run_id=args.run_id,
                model=prebuilt.get(model_name),
                abstain=args.abstain,
                condition=condition_str,
                call_log=call_log,
            )
        )

    # A self-describing run record: pin the inputs so results are reproducible and so a
    # summary file can never be mistaken for a verified, citable result.
    citable = args.verified_only
    report = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "n_questions": len(questions),
        "verified_only": args.verified_only,
        "rag": args.rag,
        "agentic": args.agentic,
        "workers": args.workers,
        "citable": citable,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": _repo_commit(),
        "note": (
            "Verified-only run." if citable else
            "NOT CITABLE: includes verified=false questions. Run with --verified-only "
            "for headline numbers."
        ),
        "models": summaries,
    }
    if condition_str != "closed_book":
        report["condition"] = condition_str
    if args.abstain:
        report["abstain_condition"] = {
            "counting_rule": "abstention = non-empty content AND explicit abstain:true "
            "AND eccn:null; empty/truncated content or a missing abstain field = error, "
            "never an abstention",
            "prompt": "ABSTAIN_SYSTEM_PROMPT + ABSTAIN_USER_TEMPLATE (prompts.py)",
            "rows_carry": ["abstain", "confidence", "abstain_field", "content_nonempty"],
        }
    if args.equalized:
        report["equalized"] = {
            "preset": EQUALIZED_PRESET,
            "per_model_checks": {
                name: {"effective": chk["effective"], "deviations": chk["deviations"]}
                for name, chk in equalized_checks.items()
            },
            "no_overclaim": "This run's equalization applies to the models listed here "
            "under one provider key; it is NOT a cross-provider equalized leaderboard.",
        }
    if args.rag:
        report["retriever"] = {
            "backend": "BM25CCLRetriever (pure Python, zero new deps)",
            "index": _display_path(getattr(retriever, "index_path", "")),
            "top_k": args.rag_top_k,
            "render_cap": getattr(retriever, "render_cap", None),
            "hit_rate": retriever.hit_rate(questions, k=args.rag_top_k),
        }
    if args.log_calls:
        report["call_log"] = args.log_calls
    summary_path = out_dir / f"{args.run_id}__summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(summaries)
    if not citable:
        print(
            "\n⚠️  Results include unverified questions — NOT citable. "
            "Use --verified-only for reportable numbers.",
            file=sys.stderr,
        )
    print(f"\nWrote per-question results and {summary_path}", file=sys.stderr)
    return 0


def _print_table(summaries: list[dict]) -> None:
    # All accuracy columns are over ALL n questions (errors/unparsed = 0). See _aggregate.
    cols = [
        ("model", "model", 22),
        ("n", "n", 5),
        ("exact_accuracy", "exact", 7),
        ("eccn_accuracy", "eccn", 7),
        ("group_accuracy", "group", 7),
        ("category_accuracy", "cat", 7),
        ("mean_grade", "grade", 7),
        ("parse_rate", "parse", 7),
        ("errors", "err", 5),
    ]
    header = "".join(f"{label:<{w}}" for _, label, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        print("".join(f"{str(s[key]):<{w}}" for key, _, w in cols))


if __name__ == "__main__":
    raise SystemExit(main())
