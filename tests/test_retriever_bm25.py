"""BM25 retriever: bounded rendering and retrieval-quality reporting.

The RAG condition must (a) keep prompt context bounded regardless of entry size
and (b) say whether the controlling entry even surfaced — hit@k is the
self-reported quality gate for any RAG claim.
"""
from pathlib import Path

from commoditybench.dataset import load_questions
from commoditybench.rag.retriever_bm25 import (
    BM25CCLRetriever,
    DEFAULT_RENDER_CAP,
    RetrievedChunk,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "ccl" / "ccl_index.json"
Q34 = ROOT / "data" / "questions.jsonl"


def _retriever(**kw):
    return BM25CCLRetriever(index_path=INDEX, **kw).build()


def test_render_caps_the_prompt_context():
    rt = _retriever(top_k=6)
    qs = load_questions(Q34)
    for q in qs[:5]:
        ctx = rt.retrieve(q)
        # 6 excerpts of ~render_cap chars + heads; full entries are up to ~16k
        # chars each, so an uncapped render would blow far past this bound.
        assert len(ctx) <= 6 * (DEFAULT_RENDER_CAP + 500), (q.id, len(ctx))


def test_render_anchors_on_the_items_region():
    # Long entries carry license-requirements preamble before the controlling
    # items; the excerpt must start at the List of Items Controlled region.
    chunk = RetrievedChunk(
        eccn="5A002", title="InfoSec equipment", score=0.0,
        text="5A002 heading (see List of Items Controlled)\n\nLicense "
             "Requirements preamble that goes on for many lines.\n\nList of "
             "Items Controlled\n" + "a. Controlled text " * 500,
    )
    rendered = chunk.render()
    assert rendered.startswith("[5A002] InfoSec equipment\nList of Items Controlled")
    assert len(rendered) <= DEFAULT_RENDER_CAP + 200


def test_gold_rank_none_for_ear99_gold():
    rt = _retriever()
    qs = load_questions(Q34)
    ear99 = next(q for q in qs if q.gold_eccn == "EAR99")
    assert rt.gold_rank(ear99) is None


def test_hit_rate_is_self_consistent_on_the_34():
    rt = _retriever(top_k=6)
    qs = load_questions(Q34)
    hr = rt.hit_rate(qs, k=6)
    assert hr["k"] == 6
    assert hr["n_scorable"] > 0
    assert hr["gold_found"] <= hr["n_scorable"]
    assert hr["hit_at_k"] == round(hr["gold_found"] / hr["n_scorable"], 4)
    assert sum(hr["rank_counts"].values()) == hr["gold_found"]
    assert all(1 <= r <= 6 for r in hr["rank_counts"])


def test_retrieval_is_deterministic():
    rt = _retriever()
    qs = load_questions(Q34)
    q = qs[0]
    assert rt.retrieve(q) == rt.retrieve(q)
    assert rt.gold_rank(q) == rt.gold_rank(q)
