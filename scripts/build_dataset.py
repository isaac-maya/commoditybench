"""Helper for assembling and validating the benchmark dataset.

This is curation tooling, not an automated scraper — manufacturer self-classifications
require human judgment to turn into clean questions (see data/schema.md). What it does:

    validate  : check a .jsonl file against the Question schema and the ECCN format,
                and report coverage (per-category counts, verified vs. unverified, EAR99).
    template  : print a blank question record you can fill in.

Usage:
    python scripts/build_dataset.py validate data/questions.example.jsonl
    python scripts/build_dataset.py template
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Allow running from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commoditybench.dataset import load_questions  # noqa: E402
from commoditybench.eccn import parse  # noqa: E402

#: ECCN-shaped token on RAW text (case-insensitive, word-boundary anchored) so
#: benign technical vocabulary ("numerically controlled oscillator", "motor
#: control") never matches.
_ECCN_TOKEN_RE = re.compile(r"\b[0-9][A-E][0-9]{3}(?:\.[a-z0-9]+)*\b", re.IGNORECASE)
#: Same shape on a whitespace/punct-stripped copy, to catch spaced variants
#: ("4 A 090 . a"). No boundary anchor here on purpose — after merging, the
#: prefix is unreadable anyway; the hit is reported for human review.
_MERGED_TOKEN_RE = re.compile(r"[0-9][A-E][0-9]{3}(?:[a-z0-9]+)*", re.IGNORECASE)
#: The literal vocabulary words, case-insensitive (they only ever denote the answer).
_ECCN_WORD_RE = re.compile(r"\b(eccn|ear99)\b", re.IGNORECASE)


def leak_problems(q) -> list[str]:
    """Curation-leak problems for one question, or [].

    Wider than the historical gold-only check (which caught just the gold ECCN or
    its CGNNN head in the description, and skipped EAR99-gold rows entirely):

    - ANY ECCN-shaped token in the description OR item_name is flagged. A related
      entry is as decisive as the gold itself ("...meets the threshold of ECCN
      3A090.a" against a 4A090.a gold hands the model the answer structure), and
      the model sees both fields.
    - The words ECCN/EAR99 are flagged. "classified EAR99" inside an EAR99-gold
      description is a full answer leak — that class was previously unchecked.
    - Hits are reported "review in context": a same-head sibling mention or a
      part number that merely contains the head is evidence to classify, not
      proof (the old substring check over-flagged those).

    Matches schema.md's curation rule — "Exclude the vendor name and the ECCN
    from the description, or the task becomes a lookup instead of a
    classification" — on both fields the model actually sees.
    """
    problems: list[str] = []
    for field in ("description", "item_name"):
        raw = getattr(q, field)
        if not raw:
            continue
        merged = re.sub(r"[\s\-_.]", "", raw)
        for tok in _ECCN_TOKEN_RE.findall(raw):
            problems.append(f"{q.id}: ECCN-shaped token {tok!r} in {field} (review in context)")
        for tok in _MERGED_TOKEN_RE.findall(merged):
            problems.append(f"{q.id}: ECCN-shaped token {tok!r} in {field} (spaced variant; review in context)")
        for word in _ECCN_WORD_RE.findall(raw):
            problems.append(f"{q.id}: word {word.upper()!r} in {field}")
    return problems

TEMPLATE = {
    "id": "VENDOR-PRODUCT-001",
    "item_name": "",
    "description": "Functional + technical characteristics only. Do NOT include the "
    "vendor name or the ECCN here.",
    "gold_eccn": "",
    "manufacturer": "",
    "source_url": "",
    "source_type": "manufacturer_self_classification",
    "verified": False,
    "category": "",
    "difficulty": "medium",
    "notes": "",
}


def cmd_validate(path: str) -> int:
    questions = load_questions(path)
    problems: list[str] = []
    cats: Counter = Counter()
    verified = 0
    ear99 = 0
    seen_ids: set[str] = set()

    for q in questions:
        if q.id in seen_ids:
            problems.append(f"duplicate id: {q.id}")
        seen_ids.add(q.id)

        e = parse(q.gold_eccn)
        if not e.is_valid:
            problems.append(f"{q.id}: gold_eccn {q.gold_eccn!r} is not a valid ECCN/EAR99")
        if e.is_ear99:
            ear99 += 1
        elif e.category:
            cats[e.category] += 1
        if q.verified:
            verified += 1
        # Leakage check (see leak_problems): the description AND item_name are
        # shown to the model, so both are scanned for any ECCN-shaped token and
        # the words ECCN/EAR99 — including on EAR99-gold rows.
        problems.extend(leak_problems(q))

    print(f"Loaded {len(questions)} questions from {path}")
    print(f"  verified:   {verified}  unverified: {len(questions) - verified}")
    print(f"  EAR99:      {ear99}")
    print("  by category:")
    for c in sorted(cats):
        print(f"    {c} ({_cat_name(c)}): {cats[c]}")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nOK: no schema/format problems found.")
    return 0


def _cat_name(c: str) -> str:
    from commoditybench.eccn import CATEGORY_NAMES

    return CATEGORY_NAMES.get(c, "?")


def cmd_template() -> int:
    print(json.dumps(TEMPLATE, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="Validate a dataset file and report coverage.")
    v.add_argument("path")
    sub.add_parser("template", help="Print a blank question record.")
    args = ap.parse_args()

    if args.cmd == "validate":
        return cmd_validate(args.path)
    if args.cmd == "template":
        return cmd_template()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
