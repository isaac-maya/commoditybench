"""Scope-conditioned preflight: scripts/scope_preflight.py.

Pins the two-level detector's behavior on the shipped index and dataset:

- scope-conditioned entries are those whose membership turns on an OPERATIONAL
  fact (destination, end-use, jurisdiction, availability) rather than physical
  specs — the flat one-description-one-answer schema cannot represent them;
- overlay-only entries (8A992 etc.) carry sanctions language in License
  Requirements but have context-free item definitions — NOT the problem class;
- run against the published 34 questions, the detector flags the 8 gold-5A992.c
  mass-market crypto rows — mechanically reproducing the Cat-5 input-starvation
  diagnosis in results/future_research_directions.md — plus 3 head-level rows
  whose conditioning sentence does not reach their paragraph (granularity limit,
  documented in the tool's report).
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json  # noqa: E402

import scope_preflight as sp  # noqa: E402

INDEX = str(Path(__file__).resolve().parents[1] / "data" / "ccl" / "ccl_index.json")
Q34 = str(Path(__file__).resolve().parents[1] / "data" / "questions.jsonl")


def _classified():
    idx = json.load(open(INDEX, encoding="utf-8"))
    return [sp.classify_entry(e) for e in idx["entries"]]


def test_index_class_counts():
    counts = Counter(c["class"] for c in _classified())
    assert sum(counts.values()) == 637
    assert counts == {"scope-conditioned": 23, "overlay-only": 117, "clean": 497}


def test_qualifier_distribution():
    quals = Counter(q for c in _classified() for q in c["qualifiers"])
    assert quals == {
        "availability": 12, "end-use": 4, "destination": 3,
        "regulatory-status": 3, "jurisdiction": 2,
    }


def test_acceptance_cases():
    by = {c["eccn"]: c for c in _classified()}
    assert by["0A998"]["class"] == "scope-conditioned"
    assert by["0A998"]["qualifiers"] == ["destination"]
    assert by["8D999"]["class"] == "scope-conditioned"
    assert by["8A992"]["class"] == "overlay-only"  # sanctions overlay, not scope
    assert by["1C298"]["class"] == "scope-conditioned"
    assert set(by["1C298"]["qualifiers"]) == {"end-use", "jurisdiction"}
    assert by["5A992"]["class"] == "scope-conditioned"
    assert by["5A992"]["qualifiers"] == ["availability"]


def test_their_34_flags_the_cat5_cluster():
    idx = json.load(open(INDEX, encoding="utf-8"))
    by = {c["eccn"]: c for c in (sp.classify_entry(e) for e in idx["entries"])}
    rows = sp.scan_questions(Q34, by)
    flagged = [r for r in rows if r["scope_flagged"]]
    assert len(flagged) == 11, [r["id"] for r in flagged]
    cat5 = [r for r in flagged if str(r["gold_eccn"]).startswith("5A992.c")]
    assert len(cat5) == 8  # the mass-market crypto cluster, reproduced mechanically
    assert {r["id"] for r in flagged} >= {
        "espressif-esp32-wroom-32e-n4", "ti-am3358bzcz100", "rpi-sc0668",
    }
