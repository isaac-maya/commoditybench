"""Leak-check widening: build_dataset.leak_problems.

The historical validator check only caught the gold ECCN (or its CGNNN head)
inside the description, and skipped EAR99-gold rows entirely
(scripts/build_dataset.py, pre-widening). This suite pins the widened behavior:
ANY ECCN-shaped token or the words ECCN/EAR99 in the description OR item_name
are flagged — including on EAR99-gold rows — while benign technical vocabulary
passes untouched.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_dataset import cmd_validate, leak_problems  # noqa: E402
from commoditybench.dataset import Question  # noqa: E402


def _q(**overrides) -> Question:
    base = dict(
        id="leak-test",
        item_name="Generic widget",
        description="A generic widget with physical characteristics only.",
        gold_eccn="4A994.b",
        verified=False,
        category="4",
        difficulty="medium",
        source_url="https://example.invalid/x",
    )
    base.update(overrides)
    return Question(**base)


def test_gold_exact_leak_fires():
    assert leak_problems(_q(description="...classified as 4A994.b by the vendor..."))


def test_related_entry_leak_fires():
    # Gold 4A090.a; description names the related 3A090.a threshold entry.
    assert leak_problems(_q(gold_eccn="4A090.a",
                            description="...meets the threshold of ECCN 3A090.a..."))


def test_item_name_leak_fires():
    assert leak_problems(_q(item_name="Rack server (ECCN 4A994.b)"))


def test_ear99_gold_row_is_checked():
    # Previously the whole check was gated on `not e.is_ear99` — this class
    # passed silently. "classified EAR99" in an EAR99-gold description is a full
    # answer leak.
    assert leak_problems(_q(gold_eccn="EAR99",
                            description="...an uncontrolled part classified EAR99..."))


def test_literal_eccn_word_fires():
    assert leak_problems(_q(description="...is not listed under ECCN 5A002..."))


def test_normalization_variants_fire():
    assert leak_problems(_q(description="...exceeds the 4 A 090 . a thresholds..."))


def test_benign_technical_terms_pass():
    assert not leak_problems(
        _q(description="A numerically controlled oscillator for wideband transmit, "
                       "with embedded motor control and 16-bit resolution."))


def test_clean_row_passes():
    assert not leak_problems(_q())
    assert not leak_problems(_q(gold_eccn="EAR99",
                                description="A passive optical mirror on a kinematic mount."))


def _write_questions(tmp_path, rows):
    p = tmp_path / "questions.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


def test_cmd_validate_fails_on_leaky_file(tmp_path):
    # Integration: the widened leak check must make `build_dataset.py validate`
    # exit nonzero on a file that leaks, not just print.
    leaky = dict(
        id="leak-file", item_name="Generic widget",
        description="A widget classified as 3A090.a by the vendor datasheet.",
        gold_eccn="4A994.b", verified=False, category="4",
        difficulty="medium", source_url="https://example.invalid/x",
    )
    assert cmd_validate(_write_questions(tmp_path, [leaky])) == 1


def test_cmd_validate_passes_on_clean_file(tmp_path):
    clean = dict(
        id="clean-file", item_name="Generic widget",
        description="A generic widget with physical characteristics only.",
        gold_eccn="4A994.b", verified=False, category="4",
        difficulty="medium", source_url="https://example.invalid/x",
    )
    assert cmd_validate(_write_questions(tmp_path, [clean])) == 0
