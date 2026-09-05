# Future research directions for CommodityBench

Status: draft, 2026-06-26. **Nothing here changes the current dataset or runs.** This is a
parking lot for changes to make *next time the benchmark is re-run* (i.e. when there is API
budget and we are willing to invalidate / re-baseline existing runs). Items are ordered
roughly by expected value-per-effort.

Why this matters: editing `data/questions.jsonl` (descriptions, gold labels, or which items
are included) invalidates every committed run, because runs are scored against the dataset as
it was at run time. So we batch dataset changes and re-run everything at once, rather than
drip-editing. Treat the list below as the spec for the *next* dataset version (call it v2).

---

## 1. Give the model the information the gold label actually depends on (mass-market / decontrol notes)

**Finding that motivates this.** The single largest fixable error cluster for Opus 4.8 *with
tools* is the mass-market info-security split: gold `5A992.c` items predicted as controlled
`5A002.a.x`. Of 8 gold-`5A992.c` items in the agentic run, 5 are wrong. If those flipped, exact
accuracy on that run rises **0.559 → 0.706 (+0.147)** and mean graded score **0.641 → 0.753**
— roughly a third of all remaining error.

**Root cause.** The `5A992.c`-vs-`5A002` distinction turns on the **Cryptography Note (Note 3 to
Category 5, Part 2)** — i.e. whether the item qualifies for *mass-market* treatment: generally
available via retail sale, sold from stock, crypto functionality not user-modifiable, etc. That
is a **commercial/regulatory fact about how the product is sold**, not a technical attribute. The
model is fed only `item_name` + a technical `description` (see `src/commoditybench/prompts.py`),
and for the crypto parts that description foregrounds exactly the signals that pull *toward*
`5A002` (AES, ECDSA/ECDH, TLS engines) while omitting every signal that would justify the
decontrol. The prompt even instructs the model to classify on "technical characteristics and
function, not… end use." So for these items the discriminating fact is simply absent from the
input — the over-classification is partly an under-specified-input artifact, not pure model error.

**This is a fork in benchmark design — decide it explicitly:**

- **(A) Pure technical-classification benchmark.** Keep inputs technical-only. Then accept that
  mass-market/`5A992.c` golds are partly unanswerable from inputs, and either (i) drop those items,
  (ii) re-label them to the technically-correct controlled entry, or (iii) keep them but report
  them as a known "input-starved" subset. Cleanest scientifically, but less faithful to the real
  BIS task.
- **(B) BIS-realistic classification benchmark (recommended).** Add the commercial context BIS
  actually has, so decontrol-note reasoning is *in scope* and testable. Concretely:
  - Add an optional structured field to each question, e.g. `commercial_context` /
    `availability` (free text or enum: "mass-market retail / sold from stock", "custom / made to
    order", "unknown"), surfaced in the prompt between description and instruction.
  - For the crypto items, this lets us measure whether the model can *apply* the Cryptography Note
    when given the predicate facts — a much more meaningful capability signal than guessing.
  - **Run both with and without the field** (A/B) so we can quantify exactly how much of the
    over-control is input-starvation vs. genuine reasoning failure. This is the clean experiment
    flagged at the end of the right/wrong analysis.

**Scope beyond crypto.** The same principle applies to other decontrol/mass-market notes (e.g.
mass-market software, items released by License Exception, de minimis, "not specially designed").
Whatever facts those notes hinge on should be present in inputs if we want to test the reasoning
rather than penalize the model for missing data.

### Design issue vs. model issue — the two Category-5 over-control clusters are not the same kind of error

The two biggest over-control clusters both live in Category 5 and look superficially alike
(model reaches for a heavier control than gold), but they have different root causes, and that
should shape how we fix and report them. The center of gravity differs; neither is pure.

- **Mass-market crypto (gold `5A992.c`, model says `5A002`) is *mostly a benchmark-design
  issue.*** The discriminating fact — does the part qualify for the Cryptography Note's
  mass-market treatment — is a commercial/distribution fact (retail availability, crypto not
  user-modifiable, no substantial supplier support) that is **absent from a technical
  description by construction**. On the inputs given, `5A002` is the *defensible* answer; the
  prompt even steers away from the needed context ("classify on technical characteristics, not
  end use"). So we are partly penalizing missing data, not bad reasoning.
  - *But it is not purely design:* (a) mass-market status is partly inferable from item identity
    — the models got the ESP32 / Raspberry Pi / BLE-module `5A992.c` items right, so the signal
    isn't wholly absent; (b) several misses went straight to `5A002`/`3A991` without ever
    reaching the `5A992` branch, suggesting the model doesn't reliably know the decontrol pathway
    exists — a regulatory-knowledge gap an availability field wouldn't fix; (c) 2 of the 5 misses
    (`ti-am3358`, `adsp-sc589` → `3A991`) miss Category 5 entirely and are really category/recall
    errors mis-bucketed as "mass-market."

- **Ethernet PHY (gold EAR99, model says `5A991`) is *mostly a genuine model issue.*** Every
  fact needed is present in the inputs (the part is a 10/100 IEEE-802.3 PHY) plus the CCL tools;
  the correct answer is a self-contained derivation — read `5A991`, confirm no specific parameter
  is met, return EAR99. The failure signature is reasoning, not data: **tools made it worse** (the
  catch-all text lured it) and the error is **consistent across all three top models on the same
  parts**.
  - *But it is not purely the model's fault:* the gold rests on the manufacturer's EAR99
    self-classification, and **Digi-Key classified the same parts `5A991.b.1`** — professional
    classifiers split, so the model's answer matches a real human authority. Treating the
    manufacturer as authoritative is itself a design choice baked into the gold; and the lure is
    partly harness-induced (the agentic search surfaces the catch-all).

**Net:** mass-market = mostly design with a model tail; PHY = mostly model with a design tail.
Two cheap experiments separate the components rather than assuming:
1. *Mass-market:* run the availability-field A/B (§1, option B). If adding the field fixes the
   crypto misses → it was design. If the model still skips `5A992.c` → it didn't know to apply the
   Note (model gap). This decomposes the error instead of guessing.
2. *PHY:* check whether the model, when it picks `5A991`, cites a *specific parameter* it believes
   is met. It can't (none is) — asserting one anyway is a clean reasoning failure; hedging means
   the item is genuinely ambiguous and the label deserves a confidence tier (§8). Either way,
   record the Digi-Key/manufacturer split in the item's provenance so the gold isn't presented as
   uncontested.

---

## 2. Separate "recall" errors from "judgment" errors in scoring/reporting

This session's central qualitative finding: **tools decisively fix recall failures** (an entry
the model forgot exists, a threshold it half-remembered) but **barely move — and can worsen —
judgment calls** at controlled/uncontrolled boundaries (mass-market crypto; over-reading the
`5A991` telecom catch-all onto EAR99 Ethernet PHYs, where tools made `lan8720a`/`lan8742a`
*worse* than no-tools).

For v2, **tag each item with its expected failure mode / discriminating skill**, e.g.:
- `recall` (does the right entry/threshold even get found),
- `boundary-judgment` (controlled vs decontrolled / mass-market / catch-all precision),
- `over-classification-trap` (gold is EAR99 or a lighter entry; a controlled entry is a tempting lure).

Then report accuracy *per failure-mode bucket*. This turns "+111% exact with tools" into the more
useful claim "tools nearly close the recall gap but not the judgment gap," with numbers attached.

---

## 3. Report over- vs under-classification asymmetry (and consider a cost-weighted metric)

For the actual BIS use case the two error directions have very different real-world costs:
over-classification (controlling a benign item) creates needless trade friction; under-
classification (missing a control) is an enforcement/security miss. Current scoring (exact +
graded segment-match) is direction-agnostic.

Add to the report (and optionally the scorer):
- a signed **over/under-classification rate** per model/condition (we already observe over-
  classification dominates ~2:1 for Opus 4.8 + tools),
- optionally a **cost-weighted score** that penalizes under-classification of genuinely
  controlled items differently from over-classification of EAR99 items. Keep the raw graded
  score too; present cost-weighting as a separate lens, not a replacement.

---

## 4. Category diversity and balance

Current distribution is heavily skewed: Category 5 + EAR99 dominate (EAR99 alone is 10 of 34,
Categories **0/4/6/8 are empty**, and 1/2/9 are thin (1–3 items each). Per-category n is so small
that the per-category grade table is noisy and category-level claims are weak.

For v2:
- **Backfill 0/4/6/8.** Leads already scoped in CLAUDE.md: Cat 6 thermal cameras (DRS Infrared /
  Pelco, `6A003`) — note Cat 6 regressed to empty when the Thorlabs lasers were cut; Cat 4
  (Apple/HP/Oracle ECCN matrices); Cat 8 marine (few public sources); Cat 0 likely a BIS FAQ
  example or skip (essentially no commercial product with a visible ECCN).
- **Target a minimum n per category** (e.g. ≥5) so per-category numbers mean something.
- **Rebalance EAR99** so it doesn't dominate, while keeping enough EAR99 to test the
  over-classification failure mode.
- Maintain the provenance rules (human-visible ECCN on the source page; manufacturer over
  distributor; prefer BIS-listed self-classifiers) when sourcing new items.

---

## 5. Grow dataset size and report uncertainty

n = 23 verified (34 with candidates) is small; single-run point estimates carry wide intervals,
and we already see meaningful run-to-run noise (5 Opus 4.8 no-tool runs were pooled to damp it).

- Grow toward a larger verified set (target a few hundred if sourcing allows) for statistical
  power.
- Report **confidence intervals** on headline accuracy, and keep **pooling multiple runs** per
  (model, condition) as the default — already done in `aggregate_runs.py`; make CIs explicit in
  the site/findings.

---

## 6. Strengthen and systematize "trap" items

The single planted over-classification lure (Nachi super-precision bearing, gold EAR99, beaten
by tools) was high-signal. Expand this deliberately:
- A small, labeled suite of **over-classification traps** (benign items adjacent to a tempting
  controlled entry) and **decontrol-note traps** (controlled-looking items that decontrol via
  mass-market / note / threshold).
- These directly probe precision/judgment, the capability tools *don't* fix, and make the
  double-edged "more text can lower precision" finding measurable rather than anecdotal.

---

## 7. Provider equalization (prerequisite for any head-to-head ranking)

Already an open issue: each model currently runs in its strongest native config (reasoning on
for Claude/GPT-5.5/Qwen3; GPT-4o at temp 0; tools only some runs). Every cross-model table is
labeled "not equalized — capability snapshot." Before any *ranking* claim:
- decide an equalization protocol (matched reasoning/effort, matched tool access, matched
  decoding params where possible), or
- keep every head-to-head explicitly labeled as not-equalized. Pick one and state it once.

---

## 8. Per-item provenance confidence tiers (carry into v2 labeling)

Some gold labels are weaker than others and may themselves be wrong (which would mis-score the
model). Surface a confidence tier per item and consider dropping/re-sourcing the weakest:
- **Weakest current items:** the Piper Cat-9 pair (`9A991.d`, looks distributor-templated from a
  BIS aircraft-parts default), and the LND `1C232` He-3 detector (family-level FAQ hedge, per-part
  unconfirmed). Note these are also exactly where tools "disagreed" with gold — the disagreement
  may be a label problem, not a model problem.
- Decide drop-vs-keep before re-running, and keep a `provenance_confidence` field so analysis can
  exclude low-confidence items from headline numbers.

---

## 9. RAG stretch goal (still unbuilt)

Build the CCL retrieval index from the eCFR (15 CFR 774 Supp. No. 1; `rag/build_index.py` TODO)
and A/B retrieval-augmented vs no-retrieval, as a cheaper-than-agentic grounding condition.
Compare the three conditions: no-context / RAG / agentic-tools. Relevant given the central
finding that **grounding is the lever, not scale** — RAG may capture much of the agentic recall
lift at lower token cost.

---

## 10. Hold out items to keep testing tool/prompt overfitting

The agentic lift was checked against overfitting three ways (generalized to post-hoc categories,
survived a generic prompt, beat the planted trap). Institutionalize this: keep a **holdout slice**
never used while iterating on tools/prompts, so every reported tool-lift number has an
out-of-sample component by construction.

---

## Quick reference — the mass-market arithmetic that prompted this doc

Opus 4.8 + tools, agentic run over 34 items (`results/expanded_agentic__claude-opus-4-8.jsonl`):

| | Current | If all 8 gold-`5A992.c` correct | Δ |
|---|---|---|---|
| Exact accuracy | 0.559 | 0.706 | **+0.147** |
| Mean graded score | 0.641 | 0.753 | +0.112 |

5 of the 8 mass-market items are currently wrong (3 → `5A002` crypto over-control, scoring 0.40
each; 2 → `3A991`, missing Cat 5 entirely, scoring 0.00). The achievable uplift from *just* the
crypto-mass-market judgment (the 3 `5A002` items) is ~exact +0.088 / grade +0.053; the rest comes
from the two items that miss the category outright and wouldn't necessarily be fixed by a
mass-market cue alone.
