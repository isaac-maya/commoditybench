"""Abstention, calibration, and error-taxonomy accounting.

Pins the three-bucket rule (answered / abstained / error): an API failure,
empty/truncated content, or a null prediction is a plumbing error and must never
inflate abstention counts — run_eval records that counting rule on every
abstain-condition summary, and this suite enforces it on the analysis side.
"""
from commoditybench.abstention import (
    ERROR_CLASSES,
    abstention_report,
    calibration_report,
    classify_error,
    error_taxonomy_report,
    is_abstention,
)

import pytest


def _row(pred, *, error=None, abstain=None, confidence=None, reasoning="",
         exact=False, eccn_match=False, category_match=False, group_match=False,
         grade=0.0):
    return {
        "id": "q-1", "item_name": "Widget", "gold_eccn": "4A994.b",
        "predicted_eccn": pred, "error": error, "abstain": abstain,
        "confidence": confidence, "reasoning": reasoning,
        "score": {
            "exact_match": exact, "eccn_match": eccn_match,
            "group_match": group_match, "category_match": category_match,
            "grade": grade,
        },
    }


def _answered_exact(confidence=None):
    return _row("4A994.b", confidence=confidence, exact=True, eccn_match=True,
                category_match=True, group_match=True, grade=1.0)


def _answered_wrong(confidence=None):
    # Same category, wrong head -> alias failure territory.
    return _row("4A994", confidence=confidence, eccn_match=False,
                category_match=True, group_match=True, grade=0.4)


# ---------------------------------------------------------------------------
# is_abstention
# ---------------------------------------------------------------------------


def test_is_abstention_recognizes_explicit_declines():
    # Exact normalized-token matching is deliberate (the docstring: a licensing
    # officer must be able to see the rule that fired). Long free-text refusals
    # are caught on the run_eval --abstain path via the abstain:true field, not
    # by token matching.
    assert is_abstention("ABSTAIN")
    assert is_abstention("I don't know")
    assert is_abstention("cannot determine")
    assert is_abstention("Insufficient information")
    assert is_abstention("n/a")
    assert not is_abstention("4A994.b")
    assert not is_abstention("EAR99")


def test_is_abstention_none_is_not_an_abstention():
    # Empty/truncated content is an ERROR, never an abstention (run_eval's
    # counting rule). The report buckets such rows via their error field.
    assert not is_abstention(None)
    assert not is_abstention("")


# ---------------------------------------------------------------------------
# abstention_report — three buckets
# ---------------------------------------------------------------------------


def test_error_rows_never_count_as_abstentions():
    rows = [
        _answered_exact(),
        _row("", error="503: overloaded"),
        _row("ABSTAIN", abstain=True, confidence=0),
        _row(None),  # empty prediction without an error flag: still plumbing
    ]
    rep = abstention_report(rows)
    assert rep.n_errors == 2
    assert rep.n_abstained == 1
    assert rep.n_answered == 1
    # Abstain rate is over model-produced rows only (answered + abstained).
    assert rep.abstain_rate == 0.5
    # Errors and abstentions both score 0 in the standard view over all n.
    assert rep.standard_exact == 0.25
    assert rep.error_rate == 0.5


def test_abstention_report_empty_raises():
    with pytest.raises(ValueError):
        abstention_report([])


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_calibration_top_bin_includes_confidence_100():
    rows = [
        _answered_exact(confidence=95),
        _answered_exact(confidence=100),   # must not fall out of every bin
        _answered_wrong(confidence=88),
        _row("ABSTAIN", abstain=True, confidence=0),
    ]
    cal = calibration_report(rows)
    assert cal["n_scored"] == 3
    bins = {b["lo"]: b for b in cal["bins"]}
    assert bins[90]["n"] == 2   # 95 and 100 both land in the [90,100] bin
    assert cal["ece"] is not None


def test_calibration_skips_rows_without_confidence():
    cal = calibration_report([_answered_exact()])
    assert cal["bins"] == []
    assert cal["ece"] is None
    assert "note" in cal


# ---------------------------------------------------------------------------
# error taxonomy
# ---------------------------------------------------------------------------


def test_classify_error_buckets():
    assert classify_error(_answered_exact()) == "none"
    assert classify_error(_row("3A001", category_match=False)) == "taxonomy_confusion"
    # Wrong head in the right category -> alias/family confusion.
    assert classify_error(_answered_wrong()) == "alias_failure"
    # Same head, wrong leaf -> over-precise subparagraph.
    assert classify_error(
        _row("4A994.c", eccn_match=True, category_match=True,
             group_match=True, grade=0.7)) == "hallucinated_subparagraph"
    # Reasoning flags the missing parameter but the model answered anyway.
    assert classify_error(
        _row("4A994.b", reasoning="not enough information to determine the "
                                  "subparagraph", eccn_match=True,
             category_match=True, group_match=True, grade=0.7)) == "missing_info"
    assert classify_error(_row("", error="timeout")) == "abstained_or_error"
    assert classify_error(_row("ABSTAIN", abstain=True)) == "abstained_or_error"
    assert classify_error(_row(None)) == "abstained_or_error"


def test_every_declared_error_class_is_reachable():
    produced = {
        classify_error(_row("3A001", category_match=False)),     # taxonomy_confusion
        classify_error(_answered_wrong()),                       # alias_failure
        classify_error(_row("4A994.c", eccn_match=True,
                            category_match=True, group_match=True,
                            grade=0.7)),                         # hallucinated_subparagraph
        classify_error(_row("4A994.b",
                            reasoning="insufficient information to classify",
                            eccn_match=True, category_match=True,
                            group_match=True, grade=0.7)),       # missing_info
    }
    # Every declared class is produced by at least one input (no dead classes).
    assert set(ERROR_CLASSES) <= produced
    # And every produced class is declared (no undocumented classes).
    assert produced <= set(ERROR_CLASSES)


def test_error_taxonomy_total_excludes_correct_and_deferrals():
    rows = [
        _answered_exact(), _answered_exact(),
        _answered_wrong(),                              # 1 real error
        _row("3A001", category_match=False),            # 1 real error
        _row("", error="503"), _row("ABSTAIN", abstain=True),
    ]
    tax = error_taxonomy_report(rows)
    assert tax["total_errors"] == 2
    assert tax["by_class"]["none"] == 2
    assert tax["by_class"]["abstained_or_error"] == 2
