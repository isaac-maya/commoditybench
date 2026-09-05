# Design note — Minimal RAG retrieval condition for CommodityBench

**Status:** working implementation in
`src/commoditybench/rag/retriever_bm25.py` (pure Python + stdlib, zero new
dependencies; suite: 58 passed — 40 pre-existing + 18 new abstention/retriever/
guard tests). Ranking runs over each entry's full text; the prompt injection is a
capped excerpt anchored on the entry's List-of-Items-Controlled region (default
1400 chars), and the runner records retrieval quality (`hit_rate`) on every `--rag`
summary.
**Open gap this closes:** the RAG stretch goal — the repo's `rag/` scaffolding
(`build_index.py` still raises `NotImplementedError` for CCL acquisition; `retriever.py`
assumes a Chroma vector index that doesn't exist yet).

---

## 1. Scope: minimal but real

| Decision | Choice | Why |
|---|---|---|
| Corpus | the committed `data/ccl/ccl_index.json` (637 entries) | It is already parsed, committed, and the sanctioned source for the agentic condition. One chunk per ECCN (title + full controlling text incl. Items: thresholds). No eCFR download, no XML pipeline. |
| Retriever | Okapi BM25 (k1=1.5, b=0.75), pure Python | No embedding model, no Chroma, no network, no new dependency — reproducible anywhere the repo runs. |
| Rendering | rank on full text, inject a capped excerpt (default 1400 chars) anchored on the List-of-Items-Controlled region | Full entries run to ~16k chars; 6 of them would drown the question. The deciding thresholds live in the Items region, which starts past the license-requirements preamble — the excerpt starts there. |
| Interface | same `retrieve(question) -> str` contract as `CCLRetriever` | Drop-in for `run_eval`'s `--rag` path (`_evaluate_model` already calls `retriever.retrieve(q)`). |
| Output | citable excerpts headed by `[ECCN]` | The model can cite what it read — same principle as the agentic tools. |
| Quality gate | `hit_rate()` (gold entry in top-k) recorded on every `--rag` summary | A RAG claim must say whether the controlling entry even surfaced; retrieval that never finds it cannot help. |

Why BM25 is the right first condition, not a compromise: the repo's own agentic
condition already uses keyword search as one of its four tools (`ccl.py search`); BM25
is that same idea scored properly. It is the *lexical baseline* any vector condition
must beat, and it is testable with zero API spend — which matters for a benchmark whose
headline claims must be reproducible.

## 2. What exploration showed (retrieval diagnostics, 2026-09-01, no API calls)

Retrieval over the 9 backfill questions, top-3, full entry text. These diagnostics
describe RANKING behavior, which the render cap does not change. The shipped A/B
numbers (closed-book vs --rag over 12 items, run twice) are in the results/ files
whose run-ids begin `rag_`; every such summary carries its own hit_rate.

- **Thermal camera 60 Hz → 6A003 ranks #1.** The controlled entry surfaces with its
  full text, including Note 3. Good retrieval.
- **Thermal camera 9 Hz → 6A003 still outranks 6A993.a.** The decontrolled entry's own
  text ("Cameras that meet the criteria of Note 3 to 6A003.b.4") is terse and indirect;
  the *deciding text* (the 9 Hz decontrol) lives in Note 3 of the **controlled** entry.
  A model given the top-3 would read 6A003 (correct — it must be considered) but not
  necessarily 6A993.a. **Retrieval does not resolve the frame-rate trap; the entry
  text still must be read.** This is a real, reportable limitation, not a bug.
- **Rack server → 4A003 outranks 4A994.** The description's "Adjusted Peak
  Performance" token-matches 4A003's text (where the ≤70 WT NLR note lives); the
  catch-all 4A994's text says "not controlled by 4A001 or 4A003" — lexically weaker.
  This mirrors the repo's own agentic finding: **grounding can induce over-control**
  (tools made the 5A991-over-Ethernet-PHY error worse). Retrieval surfaces the right
  *text to reason over* (4A003's note is exactly what resolves the item to 4A994), but
  the top hit is the controlled entry.
- **8-GPU AI server → 4E091 (AI model parameters) surfaces, not 4A090.** 4A090's text
  anchors on "meets or exceeds the limits in 3A090.a" — the threshold entry (3A090) is
  present in the index but the query tokens (HBM, NVLink, 640 GB) don't lexically match
  the rule text. Semantic retrieval or a synonym-aware query would do better here; this
  is the honest evidence for why embeddings are the follow-up.

**Summary of the honest finding:** lexical retrieval is a *recall* aid, not a *judgment*
aid. It reliably surfaces the category family and the controlling entry text; it does
not resolve threshold traps (9 Hz, APP bands) and it inherits the repo's documented
over-control failure mode. Those are exactly the claims a BIS pilot needs measured.

### Shipped A/B (results/rag_ab_*, 2026-09-05, deepseek-v4-flash, temperature 0.0)

Closed-book vs `--rag` over the same 12 items (data/experiments/rag_ab_12.jsonl),
run twice per arm. Retrieval did not improve exact accuracy: 5/24 exact closed-book
vs 4/24 with retrieval (grades 0.30/0.38 vs 0.36/0.33). Two mechanism examples, both
in the repo's own error-mode vocabulary:

- **flir-boson-640-60hz (gold 6A003.b.4.b): exact closed-book, wrong with retrieval.**
  The gold entry ranks #1 (hit@6 works), but the injected 6A003 text — which carries
  the sibling subparagraphs and the 9 Hz decontrol Note — moved the model to 6A002.
  Grounding induced the error, the same direction as the repo's agentic finding
  (tools made the 5A991-over-Ethernet-PHY error worse).
- **microchip-lan8720a (gold EAR99): over-controlled to 5A991.b.4 closed-book,
  EAR99 with retrieval.** The reverse of the usual direction, n=1 — reported as an
  observation, not a claim.

Per-run summaries record `hit_rate`: the gold entry's CCL head landed in the top-6
for 4 of 9 scorable questions (0.444) — with the controlling entry missing from the
context half the time, a null result is what the retrieval quality predicts. This is
the honest baseline a vector condition must beat.

## 3. Implementation (what shipped)

`run_eval --rag` builds the retriever over the shipped index and injects the capped
excerpts into the prompt:

```python
# run_eval --rag, one line:
from commoditybench.rag.retriever_bm25 import BM25CCLRetriever
retriever = BM25CCLRetriever(index_path="data/ccl/ccl_index.json", top_k=6).build()
# then pass retriever=retriever into _evaluate_model (already supported)
```

Every `--rag` run summary records the retriever block: index (repo-relative), top_k,
render_cap, and `hit_rate` over the run's questions — how often the gold entry's
CCL head landed in the retrieved top-k. Compare on the same item set: closed-book vs
`--rag` vs `--agentic`, per model, reporting grade/exact per condition plus the
retrieval-quality diagnostic (`hit_rate` makes it measurable without any model calls).

## 4. Deliberate non-goals

- No embedding/vector store (Chromadb remains the declared follow-up; `rag/build_index.py`
  can then reuse the same chunking via `CCLChunk`).
- No re-ranking, no query expansion — the point is the cleanest baseline.
- No changes to the agentic condition; RAG is a *third* condition, not a replacement.

## 5. Validation

- `retriever_bm25.py` imports clean; suite `58 passed`, incl. `tests/test_retriever_bm25.py`
  (prompt-context bound, Items-anchored rendering, gold-rank/EAR99 handling,
  hit-rate self-consistency, determinism).
- Retrieval quality over the published 34 (top-6, no API calls): gold-head entry in
  the top-6 for 9 of 24 scorable questions (hit@6 = 0.375) — the honest number the
  RAG A/B results should be read against.
