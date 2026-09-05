"""Prompt construction for the classification task.

We keep the system prompt provider-agnostic and ask every model for the same structured
JSON answer, so scoring is uniform. Adapters that support native structured outputs
(e.g. the Anthropic adapter) enforce the schema; others rely on the instruction plus a
robust fallback extractor (see :func:`commoditybench.eccn.extract_eccn`).

Abstain-aware schema (OPT-IN, defaults unchanged)
-------------------------------------------------
The abstention + calibration condition (see ``docs/DESIGN_abstention_calibration.md``
and the runner's ``--abstain`` flag) extends the answer schema with two optional fields:

    "abstain": true | false      # explicit decline when the deciding parameter is absent
    "confidence": 0-100          # stated confidence, measured against reality

and allows ``"eccn": null`` (paired with ``"abstain": true``). The extension is opt-in
so **default runs stay byte-identical** (existing runs remain comparable):

- Set the env var ``COMMODITYBENCH_ABSTAIN_AWARE=1`` BEFORE the process starts to make
  the module-level ``ANSWER_SCHEMA`` (imported by the Anthropic/Gemini/OpenAI adapters
  at import time) the extended schema — this is the switch for adapters that enforce the
  schema server-side.
- ``ABSTAIN_ANSWER_SCHEMA`` is always exported for code that wants the extended schema
  explicitly without the env switch (the DeepSeek ``json_object`` path never sends a
  schema, so our own abstain runs do not need the env).
- The runner's ``--abstain`` condition is independent: it selects the abstain-instructed
  prompt variant and the strict abstention-counting rules regardless of the env.
"""

from __future__ import annotations

import os

from .dataset import Question

#: Env switch that makes the module-level ANSWER_SCHEMA the abstain-aware extension.
ABSTAIN_OPT_IN_ENV = "COMMODITYBENCH_ABSTAIN_AWARE"
_abstain_aware = (
    os.environ.get(ABSTAIN_OPT_IN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
)


def _answer_schema(*, abstain_aware: bool) -> dict:
    """The answer JSON schema; ``abstain_aware=True`` returns the extended variant."""
    eccn: dict = {
        "type": "string",
        "description": "The single most likely ECCN (e.g. '3A001.a.1.a') or 'EAR99' "
        "if the item is subject to the EAR but not listed on the CCL.",
    }
    required = ["eccn", "category", "reasoning"]
    properties: dict = {
        "eccn": eccn,
        "category": {
            "type": "string",
            "description": "The single-digit CCL category (0-9) you assigned, or 'EAR99'.",
        },
        "reasoning": {
            "type": "string",
            "description": "A brief justification citing the controlling CCL text or "
            "technical parameters that drove the decision.",
        },
    }
    if abstain_aware:
        # Abstention is a first-class answer: eccn:null is only valid paired with
        # abstain:true (a model that commits an ECCN must not also claim to abstain).
        eccn["type"] = ["string", "null"]
        eccn["description"] = (
            "The single most likely ECCN (e.g. '3A001.a.1.a') or 'EAR99' if the item "
            "is subject to the EAR but not listed on the CCL, or null (with "
            "'abstain': true) when the description lacks the parameter the controlling "
            "entry requires."
        )
        properties["abstain"] = {
            "type": "boolean",
            "description": "True when the description does not contain the deciding "
            "parameter (frame rate, APP, wavelength, threshold, intended end use, ...) "
            "and the item therefore cannot be classified without guessing.",
        }
        properties["confidence"] = {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Stated confidence that the classification is correct: "
            "100 = certain it cannot be wrong; 0 = pure guess.",
        }
        required = ["eccn", "category", "reasoning", "abstain", "confidence"]
    return {
        "type": "object",
        "properties": properties,
        # All properties are listed in `required`: OpenAI's strict json_schema mode
        # mandates that every key in `properties` also appear in `required` (with
        # additionalProperties false), or the request 400s. Anthropic/Gemini accept this
        # too. In the abstain-aware variant the extra fields are therefore required of
        # every answer (an abstaining answer still states its confidence).
        "required": required,
        "additionalProperties": False,
    }


# JSON Schema for the answer, reused by adapters that support structured outputs.
# Byte-identical to the original 3-field schema unless COMMODITYBENCH_ABSTAIN_AWARE=1
# was set before import (see the module docstring).
ANSWER_SCHEMA = _answer_schema(abstain_aware=_abstain_aware)

# The extended schema, always available for explicit use (e.g. the DeepSeek json_object
# lane documents it; strict adapters opt in via the env switch above).
ABSTAIN_ANSWER_SCHEMA = _answer_schema(abstain_aware=True)

SYSTEM_PROMPT = """\
You are an expert US export-control classification analyst. Your task is the same one \
performed by the Bureau of Industry and Security (BIS): given a description of a \
commodity, determine the single Export Control Classification Number (ECCN) under which \
it falls on the Commerce Control List (CCL, Supplement No. 1 to Part 774 of the EAR).

Rules:
- Return exactly one ECCN, as specific as the description supports (include the \
subparagraph when it is determinable, e.g. "3A001.a.1.a").
- If the item is subject to the EAR but is not described by any CCL entry, return \
"EAR99".
- Base the classification on the item's technical characteristics and function, not on \
its country of origin or end use.
- Respond ONLY with the requested JSON object. Do not add commentary outside it."""

USER_TEMPLATE = """\
Classify the following item and return the JSON object described by the schema.

Item name: {item_name}
Description: {description}
{context_block}
Respond with a JSON object: {{"eccn": "...", "category": "...", "reasoning": "..."}}"""

CONTEXT_TEMPLATE = """
Relevant excerpts from the Commerce Control List (retrieved automatically; they may or \
may not be on point — judge for yourself):
\"\"\"
{context}
\"\"\"
"""

# Abstain-instructed variant for the --abstain condition (zero-error brief). The
# abstain/confidence fields mirror ABSTAIN_ANSWER_SCHEMA; the wording follows the
# repo's existing prompt voice.
ABSTAIN_SYSTEM_PROMPT = """\
You are an expert US export-control classification analyst, the same role performed by \
the Bureau of Industry and Security (BIS): given a description of a commodity, \
determine the single Export Control Classification Number (ECCN) under which it falls \
on the Commerce Control List (CCL, Supplement No. 1 to Part 774 of the EAR), or EAR99 \
if it is subject to the EAR but not described by any CCL entry.

Your employer's standard is that the acceptable level of errors is zero: a wrong ECCN \
becomes an unlicensed export or a wrongly denied one. A human reviewer costs time; a \
wrong classification costs compliance. Therefore:
- If the description contains the technical parameter the controlling entry requires \
(frame rate, Adjusted Peak Performance, wavelength, threshold, intended end use, etc.), \
classify precisely and cite the parameter.
- If the description DOES NOT contain the deciding parameter — if the classification \
genuinely hinges on information you were not given — set "abstain": true and \
"eccn": null rather than guessing. Abstention is the correct professional answer when \
the facts are missing.
- State your confidence honestly on a 0-100 scale: 100 = certain the classification \
cannot be wrong; 0 = pure guess.

Rules:
- Return exactly one ECCN, as specific as the description supports (include the \
subparagraph when it is determinable, e.g. "3A001.a.1.a"), or "EAR99" when the item is \
subject to the EAR but not described by any CCL entry.
- Base the classification on the item's technical characteristics and function, not on \
its country of origin or end use.
- Respond ONLY with the requested JSON object. Do not add commentary outside it."""

ABSTAIN_USER_TEMPLATE = """\
Classify the following item and return the JSON object described by the schema.

Item name: {item_name}
Description: {description}
{context_block}
Respond with a JSON object: {{"eccn": "..." or null, "category": "..." or null, \
"confidence": 0-100, "abstain": true or false, "reasoning": "..."}} — set \
"abstain": true and "eccn": null only when the description genuinely lacks the \
deciding parameter the controlling entry requires."""


def build_user_prompt(
    question: Question,
    context: str | None = None,
    *,
    abstain_aware: bool = False,
) -> str:
    """Build the user prompt. When ``context`` is given (RAG mode), CCL excerpts are
    injected between the item description and the answer instruction. With
    ``abstain_aware=True`` the abstain-instructed template is used (--abstain
    condition); the default is byte-identical to the original template."""
    template = ABSTAIN_USER_TEMPLATE if abstain_aware else USER_TEMPLATE
    context_block = CONTEXT_TEMPLATE.format(context=context) if context else ""
    return template.format(
        item_name=question.item_name,
        description=question.description,
        context_block=context_block,
    )
