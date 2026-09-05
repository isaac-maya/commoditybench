# Abstain-aware answer schema — opt-in note

> 2026-09-05 · run base: repo commit 4d253a3d5b8b939efb3ef83b56d7c06d75ded597.
> Companion to `docs/DESIGN_abstention_calibration.md` and the runner's `--abstain`
> condition. This note documents the `prompts.py` schema extension and its opt-in switch.

## What changed

`src/commoditybench/prompts.py` now builds the answer JSON schema through
`_answer_schema(abstain_aware=...)`:

- `ANSWER_SCHEMA` — the module-level schema the adapters import. **Byte-identical to the
  original 3-field schema** (`eccn`, `category`, `reasoning`) unless the opt-in env is
  set, so default runs and every existing test/comparison are unchanged (verified by
  comparing the built dict against the pre-change schema: identical).
- `ABSTAIN_ANSWER_SCHEMA` — the extended variant, always exported. Adds:
  - `"abstain": boolean` — explicit decline when the description lacks the deciding
    parameter the controlling entry requires (frame rate, APP, wavelength, threshold,
    intended end use, ...);
  - `"confidence": integer 0-100` — stated confidence, measured against reality by the
    calibration analysis (`abstention.py::calibration_report`);
  - `"eccn": ["string", "null"]` — null is only meaningful paired with `abstain: true`
    (a model that commits an ECCN must not also claim to abstain; the runner flags that
    combination as a contradiction and scores the committed ECCN).
  - The extended schema keeps every property in `required` (OpenAI strict-mode rule),
    so abstain/confidence are required of every answer under the extended schema.

## The opt-in switch

- Env var `COMMODITYBENCH_ABSTAIN_AWARE=1` (also `true/yes/on`), read **at import
  time** — set it before the process starts. When set, the module-level `ANSWER_SCHEMA`
  that the Anthropic/Gemini/OpenAI adapters bind at import becomes the extended schema,
  so schema-enforcing providers accept (and emit) `abstain`/`confidence`/`eccn:null`.
- `ABSTAIN_ANSWER_SCHEMA` is importable unconditionally for code that wants the
  extended schema without the env (the DeepSeek `json_object` lane never sends a schema,
  so DeepSeek abstain runs work without the env).

## Relationship to the runner's --abstain condition

The runner condition is independent of the env: `--abstain` selects the
abstain-instructed prompt variant (`ABSTAIN_SYSTEM_PROMPT` / `ABSTAIN_USER_TEMPLATE`)
and applies the strict counting rule (non-empty content + explicit `abstain: true` +
`eccn: null` = abstention; empty/truncated content or a missing abstain field = error,
never an abstention). Schema-strict adapters (Anthropic/Gemini/OpenAI `json_schema`)
additionally need the env var so the schema they enforce matches the prompt — the
runner refuses to start `--abstain` with such a model when the env is unset, because
the model could never emit an abstention under the 3-field schema. For our DeepSeek
runs (`json_object`, no schema sent) the env is not required.

## Verification

- Defaults byte-identical: `ANSWER_SCHEMA`, `SYSTEM_PROMPT`, `USER_TEMPLATE`,
  `CONTEXT_TEMPLATE` all compare equal to the pre-change module (checked against
  `git show HEAD:src/commoditybench/prompts.py`).
- Env opt-in: `COMMODITYBENCH_ABSTAIN_AWARE=1` import makes `ANSWER_SCHEMA ==
  ABSTAIN_ANSWER_SCHEMA` (checked in a fresh interpreter).
- Repo suite: `58 passed` (40 pre-existing + abstention/retriever/guard tests)
  with the env unset (default runs unchanged).
