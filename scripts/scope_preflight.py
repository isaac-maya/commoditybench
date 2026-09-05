#!/usr/bin/env python3
"""
scope_preflight.py — context-dependent classification detector for the CCL index.

Question-validity preflight: run it before spending models on new rows
(companion to scripts/build_dataset.py validate). Pure Python 3, stdlib only,
zero API, no network.

WHAT IT DOES
------------
For every CCL entry it classifies the entry's scope-conditioning status:

  scope-conditioned : the ENTRY's membership (what is controlled by it) is a
                      function of an operational fact — destination, end-use
                      (buyer intent), jurisdiction (agency handoff by use or
                      provenance), or market availability — not of physical
                      specifications.
  overlay-only      : the LICENSE REQUIREMENTS region carries destination-
                      regime sanction language (part 746 / Russia / Iraq /
                      Pakistan / UN), but the item definition is context-free;
                      the ECCN itself is stable, licensing varies. This is the
                      normal EAR design and NOT the problem class.
  clean             : neither.

TWO-LEVEL DESIGN
----------------
Level 1 (overlay scan) — LICENSE REQUIREMENTS region only (the section body
between the "License Requirements" and "List Based License Exceptions"
headers). Fires on destination-regime rows: the Russian industry sector
sanction (§ 746.8), UN embargo rows ("See § 746.1"), Iraq/Pakistan
regional-stability rows, part-746 cross-references. These change licensing,
never the ECCN.

Level 2 (item-scope scan) — entry title + the List-of-Items-Controlled region
(Related Controls / Definitions / Items:) + License Requirement Notes (which
can define scope, e.g. 1C298's NRC handoff or 1C355's retail-sale carve-out).
Qualifier families, each fire carrying the verbatim paragraph of evidence:

    destination_structural : the ONLY control row in License Requirements is
                             the § 746.8 Russian industry sector row — the
                             entry is a sanctions vehicle (0A998, 8D999).
    destination_text       : item-definition text conditions scope on a
                             destination ("destined for use in civil
                             aircraft", 9A991.c Note).
    end-use                : buyer-intent language conditions scope. Strong
                             (negative/restrictive) formulations — "not for
                             use in", "for use other than", "when intended
                             for use in a nuclear reactor" — fire the
                             qualifier. Sentences that merely pair intent
                             words with design predicates ("designed or
                             intended for use in a nuclear reactor",
                             "specially designed, prepared, or intended for
                             use with nuclear plants") are flagged
                             design-cooccurring and do NOT fire: the deciding
                             fact is embodied in the item, not operational.
    jurisdiction           : an agency handoff (NRC / 10 CFR part 110 / DOE)
                             whose predicate is operational — buyer intent or
                             regulatory provenance ("byproduct material",
                             "produced in a nuclear reactor"). NRC sentences
                             whose predicate is design language ("specially
                             designed or prepared for … isotope separation")
                             are reported under juris_design_carveouts and do
                             NOT fire.
    availability           : market-availability language in the item
                             definition ("mass market encryption commodities
                             in accordance with § 740.17(b)", 5A992.c;
                             consumer goods "packaged for retail sale for
                             personal use", 1C355).

Class rule: any Level-2 fire -> scope-conditioned (qualifiers + evidence).
Else any Level-1 fire -> overlay-only. Else clean.

HONESTY NOTES / KNOWN LIMITS (by design)
- Level-2 flags are ENTRY-level. Some entries carry scope-conditioning
  language that reaches only a subset of paragraphs (2B999's NRC sentence
  reaches nuclear-service equipment, not its .j lubricating pumps; 9A991's
  civil-aircraft Note reaches .c engines, not .d parts). The dataset-check
  table therefore reports the firing paragraph per flagged row; reach into the
  row's own paragraph is a judgment the tool deliberately does not fake.
- Level-1 is a screen; it counts rows and cross-references alike. The class
  counts that matter (scope-conditioned) are Level-2.
- The tool reports mechanically, and a flag is an entry-level review prompt, not a
  verdict. A small number of flagged entries sit at the design-vs-operational
  boundary (e.g. 3A225's "intended for use in specific industrial machinery" note),
  and the dataset-check table reports the firing paragraph per row so reach into
  the row's own paragraph stays a human judgment. The class counts are mechanical
  output; treat any flagged row as a review prompt, not a proven problem.

Acceptance tests (--selftest): 0A998 -> scope-conditioned (destination);
8D999 -> scope-conditioned (destination); 8A992 -> overlay-only, NOT
scope-conditioned; 1C298 -> scope-conditioned (end-use + jurisdiction);
5A992 -> scope-conditioned (availability). These reproduce, mechanically, the
repo's own Cat-5 mass-market diagnosis (CLAUDE.md) and the input-starvation
analysis in results/future_research_directions.md.
"""

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, OrderedDict

# ---------------------------------------------------------------------------
# Region splitting
# ---------------------------------------------------------------------------

M_LR = "License Requirements"
M_EX = "List Based License Exceptions"
M_IC = "List of Items Controlled"
M_NOTE = "License Requirement"


def split_entry(text):
    """Return (title, lr, ic, note) regions of one flattened entry text.

    Region order in the flattened eCFR text (per entry):
      heading (may contain the phrase "(see List of Items Controlled)")
      License Requirements  [table rows + License Requirement Notes]
      List Based License Exceptions
      List of Items Controlled [Related Controls / Definitions / Items:]
    """
    t = text
    i_lr = t.find(M_LR)
    if i_lr == -1:
        return "", "", t, ""          # heading-only entry (0A521-style)

    title = t[:i_lr].strip()

    i_ex = t.find(M_EX)
    ex_end = i_ex + len(M_EX) if i_ex != -1 else len(t)
    lr = t[i_lr + len(M_LR):ex_end]

    # Real section marker: after the exceptions header (the heading's
    # "(see List of Items Controlled)" occurrence comes first).
    if i_ex != -1:
        i_ic = t.find(M_IC, ex_end)
    else:
        i_ic = t.find(M_IC, i_lr + len(M_LR))
    if i_ic != -1:
        ic = t[i_ic + len(M_IC):]
    else:
        # No real section: entries with heading-only definitions (munitions /
        # 600-series short entries). Everything after the exceptions header
        # (or after License Requirements) is the definition.
        ic = t[ex_end:] if i_ex != -1 else t[i_lr + len(M_LR):]

    # License Requirement Note(s): the scope-defining tail of the LR region.
    # Use the LAST occurrence of the marker (the section header itself is
    # the first occurrence).
    note = ""
    k = lr.rfind(M_NOTE)
    if k != -1:
        note = lr[k:]
    return title, lr, ic, note


def _paragraphs(region):
    """Split a flattened region into paragraphs (blank-line separated).

    Numbered items inside one flattened paragraph ("(1) … (2) … (3) …", the
    flattened form of Related Controls / ECCN Controls / notes lists) are
    split into separate evidence units so citations stay sentence-sized.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", region) if p.strip()]
    out = []
    for p in paras:
        parts = re.split(r"\s(?=\(\d+\)\s)", p)
        out.extend(part.strip() for part in parts if part.strip())
    return out


def _clean(p, cap=420):
    p = re.sub(r"\s+", " ", p).strip(" |")
    if len(p) > cap:
        return p[:cap].rstrip() + " […]"
    return p


def _scan_paragraphs(region, patterns, cap=420):
    """Return OrderedDict pattern -> list of paragraphs containing a match."""
    hits = OrderedDict()
    if not region:
        return hits
    for name, pat in patterns:
        found = []
        for p in _paragraphs(region):
            if re.search(pat, p, re.I):
                c = _clean(p, cap)
                if c not in found:
                    found.append(c)
        if found:
            hits[name] = found
    return hits


def _scan_lines(region, patterns, cap=340):
    """Row mode for the License Requirements table (Level 1).

    A table row flattens to two consecutive non-empty lines (the control
    label and its country-chart cell, possibly wrapped). Evidence joins the
    previous, matched, and next non-empty lines so the row reads whole.
    """
    hits = OrderedDict()
    if not region:
        return hits
    lines = [l.strip() for l in region.split("\n")]
    for name, pat in patterns:
        found = []
        for i, line in enumerate(lines):
            if not re.search(pat, line, re.I):
                continue
            prev = next((lines[j] for j in range(i - 1, -1, -1) if lines[j]), "")
            nxt = next((lines[j] for j in range(i + 1, len(lines)) if lines[j]), "")
            c = _clean(" | ".join(x for x in (prev, line, nxt) if x), cap)
            if c not in found:
                found.append(c)
        if found:
            hits[name] = found
    return hits


# ---------------------------------------------------------------------------
# Pattern sets
# ---------------------------------------------------------------------------

# Level 1 — destination-regime rows in the LICENSE REQUIREMENTS region.
L1_PATTERNS = [
    ("russia_sector", r"Russian industry sector sanction"),
    ("part746_row", r"See\s*§\s*746|See\s*S\s*746|§\s*746|S\s*746"),
    ("iraq_pakistan", r"for export or reexport to Iraq|for export or reexport to Pakistan"
                      r"|transfer within Iraq|transfer within Pakistan"),
    ("part746_ref", r"part 746 of the EAR"),
]

# Level 2 — end-use language (title/IC/note regions).
ENDUSE_PATTERNS = [
    ("strong_negative", r"\bnot for use\b|\bfor use other than\b|\bfor uses other than\b"
                        r"|\bnot intended for\b|\bother than in a nuclear reactor\b"),
    ("when_intended", r"\bwhen intended for\b|\bis intended for\b"),
    ("intended_use", r"\bintended for use\b"),
]

# Design-co-occurrence: sentences pairing intent words with design predicates.
DESIGN_COOCCUR = re.compile(
    r"specially designed|designed or|prepared or|specially prepared|"
    r"designed, prepared|designed for", re.I)

# Level 2 — destination text in the item definition.
DEST_TEXT_PATTERNS = [
    ("destined_for", r"\bdestined for\b"),
    ("russia_belarus", r"\bto or within Russia\b|\bRussia or Belarus\b"),
]

# Level 2 — jurisdiction (agency handoff).
JURIS_PATTERNS = [
    ("nrc", r"Nuclear Regulatory Commission"),
    ("cfr110", r"10 CFR part 110"),
    ("doe", r"Department of Energy \(see 10 CFR part 810\)|"
            r"licensing authority of the Department of Energy"),
    ("lic_authority", r"export licensing authority of the"),
]

# Operational predicates: agency handoff driven by use-intent or provenance.
OPERATIONAL_PRED = re.compile(
    r"intended for use|not for use|when intended|is intended|for use other than|"
    r"byproduct material|produced in a nuclear reactor|"
    r"are byproduct|is byproduct|that are byproduct", re.I)
# Design predicates: handoff driven by design/embodied attributes.
DESIGN_PRED = re.compile(
    r"specially designed[\"']?\s*(?:or prepared|, prepared|, or prepared)|"
    r"specially prepared for|especially designed or prepared|"
    r"designed or intended for use in a nuclear reactor|"
    r"designed for, or equipped with|for use in liquid-metal|"
    r"specially designed for|designed or modified", re.I)

INDEPENDENT_ROW = re.compile(
    r"\b(NS|RS|AT|NP|UN|EI|CB|CC|FC|FP|SS)\s+applies to\b", re.I)
RUSSIA_ROW = re.compile(r"Russian industry sector sanction", re.I)


# ---------------------------------------------------------------------------
# Entry classification
# ---------------------------------------------------------------------------

def classify_entry(entry):
    text = entry["text"]
    title, lr, ic, note = split_entry(text)
    scope_text = "\n\n".join([title, ic, note])   # item-definition regions

    res = {
        "eccn": entry["eccn"],
        "category": entry.get("category"),
        "title": _clean(title, 200),
        "l1": _scan_lines(lr, L1_PATTERNS),
        "l1_para": _scan_paragraphs(lr, L1_PATTERNS),
        "enduse_hits": _scan_paragraphs(scope_text, ENDUSE_PATTERNS),
        "dest_hits": _scan_paragraphs(scope_text, DEST_TEXT_PATTERNS),
        "juris_hits": _scan_paragraphs(scope_text, JURIS_PATTERNS),
        "avail_hits": _scan_paragraphs(scope_text, [
            ("mass_market", r"\bmass[- ]market\b"),
            ("m74017", r"§\s*740\.17|S\s*740\.17"),
            ("from_stock", r"\bsold from stock\b|\bfrom stock\b"),
            ("retail_sale", r"\bpackaged for retail sale\b|\bretail sale\b"),
            ("public_avail", r"\bpublicly available\b|\bavailable to the general public\b"
                             r"|\bgenerally available to\b"),
            ("personal_use", r"\bfor personal use\b|\bpersonal use when\b"),
            ("commercially_avail", r"\bcommercially available before\b"
                                   r"|\bcommercially available prior to\b"),
        ]),
        "cert_hits": _scan_paragraphs(scope_text, [
            ("certified_civil", r"\bcertified for use on\b|\bcertified for use in\b"
                                r"|\bcertified by the civil aviation authority\b"
                                r"|\bcertified by civil aviation authorities\b"),
            ("type_certificate", r"\bcivil type certificate\b|\btype certificate\b|\btype-certificated\b"),
            ("wassenaar_origin", r"\bdesign or production origins\b|\borigins are either not from a Wassenaar\b"),
        ]),
        "juris_design_carveouts": [],
        "design_cooccur_notes": [],
        "class": "clean",
        "qualifiers": [],
        "evidence": [],
    }

    l1_fired = any(res["l1"].values())

    # --- Level 2a: destination_structural (table shape) -------------------
    dest_struct = False
    if RUSSIA_ROW.search(lr) and not INDEPENDENT_ROW.search(lr):
        dest_struct = True
        res["qualifiers"].append("destination")
        res["evidence"].append(
            "destination_structural: License Requirements table's only control row is the "
            "Russian industry sector sanction row (\"See § 746.8 for specific license "
            "requirements and license review policy\") — the entry is a destination-scoped "
            "sanctions vehicle.")

    # --- Level 2b: qualifier families with design-co-occurrence gating -----
    qual_evidence = []  # (qualifier, paragraph)

    def add_ev(q, para):
        qual_evidence.append((q, para))

    for para_list in res["enduse_hits"].values():
        for p in para_list:
            if DESIGN_COOCCUR.search(p):
                res["design_cooccur_notes"].append("end-use: " + p)
                continue
            add_ev("end-use", p)
    for para_list in res["dest_hits"].values():
        for p in para_list:
            add_ev("destination", p)
    for para_list in res["avail_hits"].values():
        for p in para_list:
            add_ev("availability", p)
    for para_list in res["cert_hits"].values():
        for p in para_list:
            add_ev("regulatory-status", p)

    # Jurisdiction: verdict each paragraph operational vs design-predicated.
    juris_seen = []
    for para_list in res["juris_hits"].values():
        juris_seen.extend(para_list)
    for p in juris_seen:
        if OPERATIONAL_PRED.search(p):
            add_ev("jurisdiction", p)
        elif DESIGN_PRED.search(p):
            res["juris_design_carveouts"].append(p)
        # else: unclassified NRC mention — keep in juris_hits for review

    seen = set()
    for q, p in qual_evidence:
        key = (q, p)
        if key in seen:
            continue
        seen.add(key)
        if q not in res["qualifiers"]:
            res["qualifiers"].append(q)
        res["evidence"].append(f"{q}: {p}")

    if dest_struct or qual_evidence:
        res["class"] = "scope-conditioned"
    elif l1_fired:
        res["class"] = "overlay-only"
    return res


# ---------------------------------------------------------------------------
# Dataset check mode
# ---------------------------------------------------------------------------

HEAD_RE = re.compile(r"^([0-9][A-E][0-9]{3})")


def gold_head(gold):
    if not gold:
        return None
    if gold == "EAR99":
        return "EAR99"
    m = HEAD_RE.match(str(gold).strip())
    return m.group(1) if m else str(gold).strip()


def read_questions(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def scan_questions(path, by_eccn):
    rows = read_questions(path)
    out = []
    for r in rows:
        gold = r.get("gold_eccn")
        head = gold_head(gold)
        row = {
            "id": r.get("id"),
            "item_name": r.get("item_name"),
            "gold_eccn": gold,
            "head": head,
            "entry_status": "n/a",
            "scope_flagged": False,
            "qualifiers": [],
            "evidence": [],
        }
        if head in (None, "EAR99"):
            row["entry_status"] = "ear99/na" if head == "EAR99" else "no-gold"
        else:
            ent = by_eccn.get(head)
            if ent is None:
                row["entry_status"] = "head-not-in-index"
            else:
                row["entry_status"] = ent["class"]
                if ent["class"] == "scope-conditioned":
                    row["scope_flagged"] = True
                    row["qualifiers"] = ent["qualifiers"]
                    row["evidence"] = ent.get("evidence", [])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(classified, questions_scans=None, paths=None):
    print("=" * 100)
    print("SCOPE DETECTOR — index scan report")
    print(f"index: {paths['index']} | entries: {len(classified)}")
    print("=" * 100)

    counts = Counter(c["class"] for c in classified)
    print(f"\nclass counts: scope-conditioned={counts['scope-conditioned']}  "
          f"overlay-only={counts['overlay-only']}  clean={counts['clean']}  "
          f"(total {len(classified)})")

    qual = Counter()
    for c in classified:
        if c["class"] == "scope-conditioned":
            for q in c["qualifiers"]:
                qual[q] += 1
    print(f"scope-conditioned by qualifier: "
          + ", ".join(f"{q}={n}" for q, n in qual.most_common()))

    print("\n" + "-" * 100)
    print("SCOPE-CONDITIONED ENTRIES — full evidence")
    print("-" * 100)
    for c in sorted(classified, key=lambda x: x["eccn"]):
        if c["class"] != "scope-conditioned":
            continue
        print(f"\n### {c['eccn']} | qualifiers: {', '.join(c['qualifiers'])}")
        print(f"    title: {c['title'][:170]}")
        for ev in c["evidence"]:
            q, _, p = ev.partition(": ")
            print(f"    [{q}] {p[:300]}")

    print("\n" + "-" * 100)
    print("DESIGN-CO-OCCURRING END-USE NOTES (intent + design wording; NOT fires)")
    print("-" * 100)
    for c in sorted(classified, key=lambda x: x["eccn"]):
        for n in c.get("design_cooccur_notes", []):
            print(f"  {c['eccn']:8s} | {n[:230]}")

    print("\n" + "-" * 100)
    print("DESIGN-PREDICATED NRC CARVE-OUTS (NOT jurisdiction fires)")
    print("-" * 100)
    for c in sorted(classified, key=lambda x: x["eccn"]):
        for s in c.get("juris_design_carveouts", []):
            print(f"  {c['eccn']:8s} | {s[:230]}")

    print("\n" + "-" * 100)
    print("OVERLAY-ONLY ENTRIES — first Level-1 hit each")
    print("-" * 100)
    for c in sorted(classified, key=lambda x: x["eccn"]):
        if c["class"] != "overlay-only":
            continue
        first = next((s for v in c["l1"].values() for s in v), "")
        print(f"  {c['eccn']:8s} | {first[:190]}")

    if questions_scans:
        for (path, scan) in questions_scans:
            flagged = [r for r in scan if r["scope_flagged"]]
            print("\n" + "=" * 100)
            print(f"DATASET CHECK — {path} ({len(scan)} rows; "
                  f"{len(flagged)} scope-flagged)")
            print("=" * 100)
            for r in scan:
                mark = "SCOPE" if r["scope_flagged"] else "-"
                print(f"  [{mark:5s}] {str(r['id'])[:34]:34s} "
                      f"gold={str(r['gold_eccn']):12s} head={str(r['head']):7s} "
                      f"entry={r['entry_status']}")
                if r["scope_flagged"]:
                    print(f"            qualifiers: {', '.join(r['qualifiers'])}")
                    for ev in r["evidence"][:3]:
                        print(f"            {ev[:240]}")


def selftest(classified, by_eccn):
    checks = [
        ("0A998", "scope-conditioned", ["destination"]),
        ("8D999", "scope-conditioned", ["destination"]),
        ("8A992", "overlay-only", []),
        ("1C298", "scope-conditioned", ["end-use", "jurisdiction"]),
        ("5A992", "scope-conditioned", ["availability"]),
    ]
    ok = True
    print("\n" + "=" * 100)
    print("SELFTEST (acceptance assertions)")
    print("=" * 100)
    for eccn, cls, quals in checks:
        c = by_eccn.get(eccn)
        if c is None:
            print(f"  FAIL {eccn}: not in index")
            ok = False
            continue
        got_q = sorted(set(c["qualifiers"]))
        cls_ok = c["class"] == cls
        q_ok = cls != "scope-conditioned" or got_q == sorted(quals)
        status = "PASS" if (cls_ok and q_ok) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status} {eccn}: class={c['class']} (want {cls}) "
              f"qualifiers={got_q} (want {sorted(quals)})")
    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=None,
                    help="path to ccl_index.json (default: <repo>/data/ccl/ccl_index.json)")
    ap.add_argument("--questions", action="append", default=[])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    idx_path = args.index
    if idx_path is None:
        idx_path = str(pathlib.Path(__file__).resolve().parents[1]
                       / "data" / "ccl" / "ccl_index.json")
    idx = json.load(open(idx_path, encoding="utf-8"))
    classified = [classify_entry(e) for e in idx["entries"]]
    by_eccn = {c["eccn"]: c for c in classified}

    paths = {"index": idx_path}
    questions_scans = [(qp, scan_questions(qp, by_eccn)) for qp in args.questions]

    if args.json_out:
        payload = {
            "index": idx_path,
            "n_entries": len(classified),
            "classes": dict(Counter(c["class"] for c in classified)),
            "qualifiers": dict(Counter(q for c in classified for q in c["qualifiers"])),
            "entries": classified,
            "question_scans": [{"path": p, "rows": s} for (p, s) in questions_scans],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\n[json written: {args.json_out}]")

    if args.selftest:
        ok = selftest(classified, by_eccn)
        print_report(classified, questions_scans, paths)
        return 0 if ok else 1

    print_report(classified, questions_scans, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
