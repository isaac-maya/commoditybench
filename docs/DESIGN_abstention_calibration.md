# Design note — Abstention + calibration scoring for CommodityBench

**Status:** working implementation in `src/commoditybench/abstention.py` + the
runner's `--abstain` condition (no new dependencies; suite: 58 passed — 40
pre-existing + 18 new abstention/retriever/guard tests).
**Motivating constraint (their brief):** "The acceptable level of errors is zero."
**Open gap this closes:** the graded scorer has no abstain class — a model that guesses
scores the same as a model that knows it does not know.

---

## 1. Why abstention is the compliance-critical extension

The benchmark's own framing: an LLM used by BIS must be *at least as reliable as a
human licensing officer*, and "human review does not save labor." Under that mandate,
the cost structure of an error is asymmetric:

- A **confident wrong ECCN** in a real workflow becomes an unlicensed export (or a
  wrongly denied one) — the failure that matters.
- An **abstention** ("I cannot determine this from the description") routes to a human
  reviewer — a workflow cost, not a compliance violation.

The current scorer cannot see the difference: `score_prediction("3A001.a.1.a", gold)`
and `score_prediction("I don't know", gold)` both land at grade 0 with `level: none`.
Any accuracy threshold set on that scale silently rewards guessing whenever a guess has
any chance of partial credit — which the graded ladder (0.2 for a right category) does
incentivize. The zero-error brief inverts that incentive: the system should abstain
whenever its confidence is below the point where a wrong answer costs more than a
human review.

## 2. What was added

`abstention.py` provides three pieces (see the module docstring for the full API):

1. **Abstention as a first-class outcome** — `is_abstention()` recognizes explicit
   declines (a curated, normalized token list: ABSTAIN, INSUFFICIENT INFORMATION,
   CANNOT DETERMINE, …), *before* the ECCN parser runs (so prose like "I cannot
   determine an ECCN from this description" is not salvaged into a guess by
   `extract_eccn`). `abstention_report()` buckets every row into **three outcomes** —
   answered, abstained (explicit, verified decline), error (API failure / empty /
   truncated / malformed) — and reports:
   - *standard* (abstain AND error = 0 over all n) — directly comparable with every
     published run;
   - *answered* (accuracy over committed items) — the officer-facing "when it commits,
     how often is it right?";
   - *coverage* (commit rate) — the two together are the actual operating point.
   Errors are never counted as abstentions: a 503-heavy run must not read as "the
   model abstains more" (the runner records the same counting rule on every
   `--abstain` summary).
2. **Calibration** — `calibration_report()` bins stated confidence vs exact-match
   accuracy and computes Expected Calibration Error (ECE). Requires the answer schema to
   carry a `confidence` field (0-100) — see §3. This turns "the model said 90% sure"
   into a testable claim and exposes overconfidence, the precondition for trusting any
   confidence-based abstention policy.
3. **Extended error taxonomy** — `classify_error()` buckets non-exact answers into
   `taxonomy_confusion` / `alias_failure` / `hallucinated_subparagraph` /
   `missing_info` (+ `abstained_or_error` for deferrals and plumbing, which are not
   classification errors). This answers the brief's pilot question 4b ("Which types
   of errors are most common? Can scaffolding address them?") with a taxonomy a
   licensing officer can act on per class. There is deliberately no
   reasoning-quality class: judging *why* a trace is wrong needs per-row human
   review, and the module reports mechanically.

## 3. The answer-schema change (required for calibration)

`prompts.py`'s answer JSON schema gains two optional fields:

```json
{
  "eccn": "4A994 or EAR99 or null",
  "abstain": false,
  "confidence": 0-100,
  "reasoning": "..."
}
```

Prompt guidance (consistent with the repo's agentic order-of-review): *"If the
description does not contain the parameter the controlling text requires (e.g. frame
rate, APP, wavelength), set `abstain: true` and `eccn: null` rather than guessing."*
The design deliberately keeps `confidence` as a *stated* confidence (what the model
says) — calibration measures it against reality. Nothing in the scorer changes; the
abstention module sits on top.

## 4. What the metrics should (and shouldn't) claim

- **Never headline a coverage-filtered number.** The denominator discipline of the
  existing scorer (errors count as 0 over all questions) is preserved: abstention does
  not raise any headline number; it *reports* a tradeoff (coverage vs
  answered-accuracy). A model that abstains on everything scores 0 standard exact and
  that is correct.
- **The useful policy output** is the operating curve: for each confidence threshold t
  (abstain below t), the resulting (coverage, answered-accuracy, wrong-answers-per-1000)
  triple. That is the number a BIS pilot would optimize against a human-review budget.
- **ITAR/NRC jurisdiction is a natural abstain case.** While drafting the Cat-0
  backfill we confirmed the current CCL has *restructured* Category 0: reactor hardware
  entries (0A001, 0B001-0B006, 0C001/0C002/0C004…) no longer exist on the CCL — they are
  under NRC licensing authority (10 CFR 110), and 0A002 is ITAR-subject. A model asked
  to classify "reactor-grade heavy water for a reactor" faces a question with **no valid
  ECCN or EAR99 answer** — the honest behavior is abstention with a jurisdiction note.
  The benchmark's answer space (ECCN | EAR99) cannot express "outside EAR jurisdiction";
  abstention is the first-class way to carry that case, and the taxonomy's
  `missing_info` class can absorb it (the deciding fact — jurisdiction — was absent from
  the item description).

## 5. Validation

- `src/commoditybench/abstention.py` imports clean and the suite pins its accounting:
  `tests/test_abstention.py` (error rows never count as abstentions, confidence 100
  lands in the top calibration bin, `total_errors` excludes correct answers and
  deferrals, every declared taxonomy class is reachable).
- Exercised on the real abstain run (`results/abstain_full34__deepseek-v4-flash.jsonl`):
  2 verified abstentions of 34, 0 errors, 32 answered (12 exact). Calibration over the
  answered rows: mean stated confidence ~86.8 vs 0.40 exact accuracy; the ≥90 bin
  averages ~91.5% confidence at 0.40 exact — the overconfidence finding that makes
  calibration worth measuring. All claims re-derivable from the shipped row files.
- The runner's `--abstain` condition records its counting rule on every summary, and
  rows carry `abstain` / `abstain_field` / `content_nonempty` so every counted
  abstention can be re-verified against the raw response (the per-call audit log via
  `--log-calls`).

## 6. Suggested follow-ups (not in this PR)

- Run the abstention condition on the full 34 with a frontier model and the repo's own
  providers; compare coverage/answered-accuracy vs the no-abstain baseline and against
  the DeepSeek numbers here (equalized settings — see the `--equalized` preset).
- A confidence-threshold sweep → the (coverage, wrong-per-1000) operating curve, and a
  recommended abstain-below-t policy threshold for a pilot.
