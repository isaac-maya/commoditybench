"""Abstention, calibration, and error-taxonomy analysis for CommodityBench.

Why this exists
---------------
The CommodityBench brief's core constraint is that "the acceptable level of errors is
zero" and that "human review does not save labor" - an LLM used by a BIS licensing
officer must be at least as reliable as a human officer. The current graded scorer
(``commoditybench.eccn.score_prediction``) has no abstain class: every answer is scored
on the exact/eccn/group/category/none ladder, and an API error or unparseable answer
counts as a wrong 0. A model that guesses is indistinguishable from a model that knows
it does not know - and for a zero-error workflow, **knowing when not to answer is the
most valuable behavior there is** (a wrong guess becomes an unlicensed export; an
abstention becomes a human review).

This module adds three things on top of the existing scorer, without changing it:

1. **Abstention as a first-class outcome.** ``is_abstention`` recognizes a model's
   explicit decline to classify. ``abstention_report`` computes the standard metrics
   (so new runs stay comparable with every existing run) *and* the abstention-aware
   metrics (accuracy over answered items, coverage, effective rate treating abstain
   like an error).

2. **Calibration analysis.** ``calibration_report`` bins stated confidence against
   actual accuracy and reports Expected Calibration Error (ECE) plus a reliability
   table. This is what turns "the model said 90% confident" into a testable claim.
   Requires the answer schema to carry a ``confidence`` field (0-100) - see the
   design note (docs/DESIGN_abstention_calibration.md) for the prompt/schema change.

3. **An extended error taxonomy.** ``classify_error`` buckets a wrong answer into
   four classes: taxonomy_confusion / alias_failure / hallucinated_subparagraph /
   missing_info. This directly answers the brief's pilot question
   4b ("Which types of errors are most common? Can scaffolding address them?").
   There is deliberately no reasoning-quality class: judging *why* a reasoning
   trace is wrong needs per-row review, and this module reports mechanically.

The module is a library plus a small CLI (``python -m commoditybench.abstention
results/run__model.jsonl``) that consumes the same per-question result rows that
``run_eval.py`` writes, so it works on existing and future runs alike.

Design choices
--------------
- Abstention never lowers the graded score of the *answered* items, but it lowers the
  *effective* rates computed over the full question set (same denominator discipline as
  the existing scorer: an abstained item is not a free pass).
- Errors and abstentions are separate outcomes. A row with ``error`` set (API failure,
  empty/truncated content, malformed answer) is a plumbing failure and is reported in
  the ``errors`` bucket — it is never counted as a model abstention. An abstention is
  only an explicit, verified decline (see the counting rule recorded by
  ``run_eval --abstain``). The standard view still scores both as 0 over all n, so run
  comparisons stay unchanged; the report just refuses to mislabel one as the other.
- Confidence is optional. Rows without a ``confidence`` field produce the abstention
  report but skip calibration (and say so).
- Everything here is pure Python + stdlib, no new dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# -------------------------------------------------------------------------------------
# 1. Abstention detection
# -------------------------------------------------------------------------------------

# A model that declines to classify may emit any of these (after normalization:
# uppercase, punctuation stripped, spaces collapsed). Keeping the list explicit beats a
# fuzzy classifier: a licensing officer needs to *see* the rule that fired.
_ABSTAIN_TOKENS = {
    "ABSTAIN",
    "ABSTAINED",
    "ABSTENTION",
    "IDONTKNOW",
    "IDON'TKNOW",
    "IREFUSETOANSWER",
    "DECLINETOANSWER",
    "DECLINED",
    "REFUSED",
    "CANNOTDETERMINE",
    "CANTDETERMINE",
    "CANNOTCLASSIFY",
    "UNABLETODETERMINE",
    "INSUFFICIENTINFORMATION",
    "INSUFFICIENTINFO",
    "MISSINGINFORMATION",
    "NEEDMOREINFORMATION",
    "NOTENOUGHINFORMATION",
    "UNKNOWN",
    "NOANSWER",
    "NA",
    "N/A",
}


def normalize_abstain_text(text: str) -> str:
    """Normalize a raw model output for abstention detection."""
    t = (text or "").strip().upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t


def is_abstention(predicted_eccn: str | None) -> bool:
    """True when the model output is an explicit abstention, not a classification.

    A real ECCN (``4A994``) or ``EAR99`` is never an abstention. The check is on the
    raw prediction *before* the ECCN parser runs, because ``extract_eccn`` would
    otherwise try to salvage prose like "I cannot determine an ECCN from this
    description" into a guess.

    ``None`` (no answer at all) is NOT an abstention: an empty/truncated/unparsable
    response is a plumbing failure and is carried by the row's ``error`` field.
    Callers bucket error rows separately — abstention counts must never absorb
    API errors (see ``abstention_report``).
    """
    if predicted_eccn is None:
        return False
    return normalize_abstain_text(predicted_eccn) in _ABSTAIN_TOKENS


# -------------------------------------------------------------------------------------
# 2. Abstention-aware reporting
# -------------------------------------------------------------------------------------


@dataclass
class AbstentionReport:
    """Standard + abstention-aware metrics for one model run.

    Three outcomes per row: answered, abstained (explicit, verified decline), or
    error (API failure / empty / truncated / malformed). Errors are NEVER counted as
    abstentions — see the module docstring.

    The ``standard_*`` numbers keep the existing denominator discipline (abstain and
    error both count as 0 over all n questions) so they are directly comparable with
    every published run. The ``answered_*`` numbers are the officer-facing view: when
    the model commits, how often is it right? ``coverage`` is how often it commits at
    all. ``abstain_rate`` is abstentions over rows the model actually answered or
    abstained (errors excluded) — model behavior, not plumbing.
    """

    n: int
    n_abstained: int
    n_answered: int
    n_errors: int
    abstain_rate: float
    coverage: float  # answered / n
    error_rate: float  # errors / n
    # Standard view (abstain == error == 0 over n) - comparable with existing runs.
    standard_exact: float
    standard_grade: float
    # Officer view (answered only).
    answered_exact: float
    answered_grade: float
    # Effective view (abstain == wrong over n; what the workflow would actually ship).
    effective_exact: float
    # Abstention is itself an outcome; a perfect run is abstain-on-the-hard + exact-on-
    # the-known. ``deferral_precision`` = exact among answered / (1 - abstain_rate) is
    # implied by the numbers above; kept explicit for the report.
    exact_among_answered: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_abstained": self.n_abstained,
            "n_answered": self.n_answered,
            "n_errors": self.n_errors,
            "abstain_rate": self.abstain_rate,
            "coverage": self.coverage,
            "error_rate": self.error_rate,
            "standard_exact": self.standard_exact,
            "standard_grade": self.standard_grade,
            "answered_exact": self.answered_exact,
            "answered_grade": self.answered_grade,
            "effective_exact": self.effective_exact,
            "exact_among_answered": self.exact_among_answered,
        }


def abstention_report(rows: list[dict]) -> AbstentionReport:
    """Compute the abstention report from run_eval-shaped result rows.

    Each row: {"id", "gold_eccn", "predicted_eccn", "score": {...}} - the shape
    ``run_eval.py`` writes. Buckets:
    - ``error`` set (or empty prediction without an explicit abstain marker) -> error;
    - explicit abstention (``predicted_eccn`` an abstain token, or the row's
      ``abstain`` flag true) -> abstained;
    - everything else -> answered.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("abstention_report: empty rows")

    errors: list[dict] = []
    abstained: list[dict] = []
    answered: list[dict] = []
    for r in rows:
        pred = r.get("predicted_eccn")
        if r.get("error"):
            errors.append(r)
        elif r.get("abstain") is True or is_abstention(pred):
            abstained.append(r)
        elif pred is None or str(pred).strip() == "":
            # No error flag but nothing usable either (e.g. a base-condition run
            # that recorded an empty prediction): plumbing failure, not a decline.
            errors.append(r)
        else:
            answered.append(r)

    n_abstained = len(abstained)
    n_answered = len(answered)
    n_errors = len(errors)
    model_rows = n_answered + n_abstained  # rows where the model produced usable output
    abstain_rate = round(n_abstained / model_rows, 4) if model_rows else 0.0
    coverage = round(n_answered / n, 4)
    error_rate = round(n_errors / n, 4)

    # Standard view: abstained and error rows contribute grade 0 / exact False over n.
    def _standard(key: str) -> float:
        if key == "exact":
            return round(sum(1 for r in answered if r["score"]["exact_match"]) / n, 4)
        return round(sum(r["score"]["grade"] for r in answered) / n, 4)

    # Officer view: answered items only.
    answered_exact = (
        round(sum(1 for r in answered if r["score"]["exact_match"]) / n_answered, 4)
        if n_answered
        else 0.0
    )
    answered_grade = (
        round(sum(r["score"]["grade"] for r in answered) / n_answered, 4)
        if n_answered
        else 0.0
    )

    return AbstentionReport(
        n=n,
        n_abstained=n_abstained,
        n_answered=n_answered,
        n_errors=n_errors,
        abstain_rate=abstain_rate,
        coverage=coverage,
        error_rate=error_rate,
        standard_exact=_standard("exact"),
        standard_grade=_standard("grade"),
        answered_exact=answered_exact,
        answered_grade=answered_grade,
        effective_exact=_standard("exact"),  # abstain == wrong in the standard view
        exact_among_answered=answered_exact,
    )


# -------------------------------------------------------------------------------------
# 3. Calibration
# -------------------------------------------------------------------------------------


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    accuracy: float
    avg_confidence: float
    gap: float  # confidence - accuracy; positive = overconfident


def calibration_report(
    rows: list[dict], *, bins: list[tuple[float, float]] | None = None
) -> dict:
    """Bin stated confidence vs actual exact-match accuracy and compute ECE.

    Reads ``row["confidence"]`` (0-100 float, present only if the answer schema
    captured it). Rows without confidence are ignored with a warning in the caller.
    Default bins: [0,60), [60,80), [80,90), [90,100] - coarse enough to be populated
    on a ~100-item run, fine enough to separate "guessing" from "sure".
    """
    if bins is None:
        bins = [(0, 60), (60, 80), (80, 90), (90, 100)]

    scorable = [
        r
        for r in rows
        if not r.get("error")
        and not is_abstention(r.get("predicted_eccn"))
        and r.get("confidence") is not None
    ]
    if not scorable:
        return {"bins": [], "ece": None, "note": "no rows carried a confidence field"}

    bin_stats: list[CalibrationBin] = []
    for lo, hi in bins:
        members = [
            r for r in scorable
            if lo <= float(r["confidence"]) < hi
            # Confidence is bounded at 100; the top bin must include it.
            or (hi >= 100 and float(r["confidence"]) == 100)
        ]
        if not members:
            continue
        acc = sum(1 for r in members if r["score"]["exact_match"]) / len(members)
        conf = sum(float(r["confidence"]) for r in members) / len(members)
        bin_stats.append(
            CalibrationBin(
                lo=lo, hi=hi, n=len(members), accuracy=round(acc, 4),
                avg_confidence=round(conf, 4),
                # gap in percentage points: conf (0-100) - acc*100 (0-100).
                gap=round(conf - acc * 100, 4),
            )
        )

    # Expected Calibration Error (Naeini et al. 2015): weighted mean |conf - acc|,
    # both on the 0-1 scale (conf / 100).
    total = sum(b.n for b in bin_stats) or 1
    ece = round(
        sum(b.n * abs(b.avg_confidence / 100 - b.accuracy) for b in bin_stats) / total,
        4,
    )

    return {
        "bins": [b.__dict__ for b in bin_stats],
        "ece": ece,
        "n_scored": len(scorable),
        "note": "confidence bins over exact-match accuracy; gap > 0 = overconfident",
    }


# -------------------------------------------------------------------------------------
# 4. Extended error taxonomy
# -------------------------------------------------------------------------------------

ERROR_CLASSES = [
    "taxonomy_confusion",       # wrong CCL category (e.g. Cat 5 for a Cat 4 item)
    "alias_failure",            # right category, wrong group/entry (trade-name alias,
                                #   sibling part, wrong product family)
    "hallucinated_subparagraph",  # right CGNNN head, invented/over-precise subparagraph
    "missing_info",             # the description lacked the deciding parameter and the
                                #   model answered anyway instead of abstaining/asking
    # Deliberately no reasoning-quality class: judging *why* a trace is wrong needs
    # per-row human review (the module reports mechanically).
]


def classify_error(row: dict) -> str:
    """Bucket a wrong prediction into the extended taxonomy.

    Uses the scored structure (category/group/head matches) plus lightweight reasoning-
    text heuristics. Only meaningful for rows where the prediction is NOT exact.
    """
    score = row["score"]
    if score["exact_match"]:
        return "none"  # not an error
    gold = row.get("gold_eccn", "")
    pred = row.get("predicted_eccn", "")

    # Abstentions, API errors, and empty predictions are not classification errors —
    # they are deferrals or plumbing failures, bucketed separately from the taxonomy
    # (same three-bucket rule as abstention_report).
    if (
        row.get("error")
        or is_abstention(pred)
        or pred is None
        or str(pred).strip() == ""
    ):
        return "abstained_or_error"

    reasoning = (row.get("reasoning") or "").lower()
    gold_ear99 = gold == "EAR99"

    # Missing-info: the model itself flags the gap but answered anyway, or the answer
    # is a guess on EAR99-vs-controlled with no controlling text cited.
    missing_signals = (
        "not enough", "insufficient", "no information", "unable to determine",
        "cannot determine", "would need", "missing", "assum",
    )
    if any(s in reasoning for s in missing_signals):
        return "missing_info"

    if score["category_match"]:
        if not score["eccn_match"]:
            # Same category, wrong head: group match -> sibling-entry confusion;
            # else group mismatch. Both are alias/family confusion at this level.
            return "alias_failure"
        # Head matches, subparagraph wrong -> over-precise or invented subparagraph.
        return "hallucinated_subparagraph"

    # Category wrong entirely: an EAR99 <-> controlled flip is a first-order
    # taxonomy error (the scorer already treats it as 0 with no partial credit).
    return "taxonomy_confusion"


def error_taxonomy_report(rows: list[dict]) -> dict:
    """Count error classes across a run (excluding abstentions/errors)."""
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {
        c: [] for c in ERROR_CLASSES + ["abstained_or_error", "none"]
    }
    for r in rows:
        cls = classify_error(r)
        counts[cls] += 1
        if len(examples[cls]) < 3:
            examples[cls].append(f"{r.get('id')}: pred={r.get('predicted_eccn')} gold={r.get('gold_eccn')}")
    total_errors = sum(
        v for k, v in counts.items() if k not in ("abstained_or_error", "none")
    )
    return {
        "total_errors": total_errors,
        "by_class": dict(counts),
        "examples": examples,
        "note": "taxonomy_confusion = wrong category; alias_failure = wrong entry in right "
                "category; hallucinated_subparagraph = right head, invented leaf; "
                "missing_info = answered despite the deciding parameter being absent",
    }


# -------------------------------------------------------------------------------------
# 5. CLI
# -------------------------------------------------------------------------------------


def analyze_results(path: str | Path) -> dict:
    """Full abstention + calibration + taxonomy analysis of one results file."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    report = abstention_report(rows)
    calib = calibration_report(rows)
    taxonomy = error_taxonomy_report(rows)
    return {
        "source": str(path),
        "n_rows": len(rows),
        "abstention": report.to_dict(),
        "calibration": calib,
        "error_taxonomy": taxonomy,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Abstention + calibration + error-taxonomy analysis for a "
                    "CommodityBench results file (run_eval-shaped JSONL)."
    )
    ap.add_argument("results", nargs="+", help="One or more results/*.jsonl files.")
    ap.add_argument("--pretty", action="store_true", help="Indent the JSON output.")
    args = ap.parse_args(argv)

    for path in args.results:
        print(json.dumps(analyze_results(path), indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
