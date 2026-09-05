"""Minimal BM25 lexical retriever over the shipped CCL index (RAG condition).

Scope (honest)
--------------
The repo's stretch goal is a retrieval condition: inject relevant CCL excerpts into the
prompt and A/B model accuracy with vs. without retrieval. The existing scaffolding
(``rag/retriever.py``) assumes a Chroma vector index built from a downloaded eCFR XML
(``rag/build_index.py`` still raises ``NotImplementedError`` for the acquisition step).

This module takes the short road, deliberately:

- **Corpus**: the already-committed ``data/ccl/ccl_index.json`` (637 ECCN entries,
  one "chunk" per entry — title + full controlling text, including the Items:
  subparagraph thresholds). No eCFR download, no XML parsing, no network.
- **Retriever**: Okapi BM25 (k1=1.5, b=0.75), pure Python + stdlib. No embedding model,
  no vector store, no new dependency.
- **Rendering**: ranking runs over each entry's FULL text, but only a capped excerpt
  (``render_cap`` chars, default 1400) is injected into the prompt. Full entries are
  thousands of words and 6 of them would drown the question; the excerpt keeps the
  head + the controlling subparagraphs while bounding prompt cost. The retriever also
  reports retrieval quality itself (gold entry rank / hit@k) so a RAG run can say
  whether the right entry even surfaced — see ``hit_rate`` / ``gold_rank``.
- **Why BM25 is defensible here**: the CCL is a small, dense, terminology-heavy corpus
  (637 documents) and the repo's own agentic condition already uses keyword search as
  one of its four tools — ``ccl.py search``. BM25 is that same idea, scored properly.
  It is the *lexical* baseline a vector condition must beat, and it is exactly the kind
  of minimal-but-real condition the RAG stretch goal needs before spending on embeddings.
- **What this is not**: not a semantic/vector condition, not a guarantee of uplift.
  Retrieval here is an *input* change; whether it improves classification is the
  experiment, not the assumption (the repo's own finding: the agentic tool condition
  improved recall but could *induce* over-control, e.g. 5A991 on Ethernet PHYs).

Interface
---------
Implements the same ``retrieve(question) -> str`` contract as ``CCLRetriever``
(``rag/retriever.py``) so ``run_eval``'s ``--rag`` path can use it as a drop-in:

    from commoditybench.rag.retriever_bm25 import BM25CCLRetriever
    retriever = BM25CCLRetriever(index_path="data/ccl/ccl_index.json", top_k=6)
    context = retriever.retrieve(question)   # newline-joined "[ECCN] ..." blocks

Each returned chunk is headed by its ECCN so the model can cite what it read — the
same citable-excerpt principle as the agentic tools.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..dataset import Question

DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[3] / "data" / "ccl" / "ccl_index.json"

# BM25 hyperparameters (classic defaults; the corpus is small so k1 dominates).
_K1 = 1.5
_B = 0.75

#: Rendered excerpt cap per retrieved entry (chars). Ranking uses full text; only the
#: prompt injection is capped, so a RAG prompt stays bounded no matter the entry size.
DEFAULT_RENDER_CAP = 1400


@dataclass
class RetrievedChunk:
    eccn: str
    title: str
    text: str
    score: float
    render_cap: int = DEFAULT_RENDER_CAP

    def render(self) -> str:
        head = f"[{self.eccn}] {self.title}\n"
        # Anchor the excerpt on the List-of-Items-Controlled region: that is where
        # the deciding thresholds live, and in a full entry (up to ~16k chars) it
        # starts well past the license-requirements preamble. Entries without the
        # marker fall back to their head.
        body = self.text
        marker = body.rfind("List of Items Controlled")
        if marker != -1:
            body = body[marker:]
        if len(body) > self.render_cap:
            body = body[: self.render_cap].rstrip() + " […]"
        return head + body


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; ECCNs stay whole (e.g. '6a003', 'b.4.b')."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25CCLRetriever:
    """BM25 over the parsed CCL index. Build-once, retrieve-many."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        *,
        top_k: int = 6,
        k1: float = _K1,
        b: float = _B,
        render_cap: int = DEFAULT_RENDER_CAP,
    ):
        self.index_path = Path(index_path)
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.render_cap = render_cap
        self._docs: list[RetrievedChunk] = []
        self._doc_terms: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avgdl = 0.0
        self._idf: dict[str, float] = {}
        self._built = False

    # -- build ----------------------------------------------------------------

    def build(self) -> "BM25CCLRetriever":
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if not entries:
            raise ValueError(f"No entries in CCL index: {self.index_path}")

        self._docs = [
            RetrievedChunk(
                eccn=e["eccn"],
                title=e.get("title", ""),
                text=e.get("text", ""),
                score=0.0,
                render_cap=self.render_cap,
            )
            for e in entries
        ]
        self._doc_terms = [Counter(tokenize(f"{d.title}\n{d.text}")) for d in self._docs]
        self._doc_lens = [sum(c.values()) for c in self._doc_terms]
        n = len(self._docs)
        self._avgdl = sum(self._doc_lens) / n

        df: Counter[str] = Counter()
        for terms in self._doc_terms:
            df.update(terms.keys())
        # Classic BM25 idf with smoothing; zero-df query terms contribute nothing.
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._built = True
        return self

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    # -- scoring --------------------------------------------------------------

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        terms = self._doc_terms[doc_idx]
        dl = self._doc_lens[doc_idx]
        denom = 1 - self.b + self.b * dl / self._avgdl if self._avgdl else 1.0
        total = 0.0
        for qt in query_terms:
            idf = self._idf.get(qt)
            if idf is None:
                continue
            tf = terms.get(qt, 0)
            if tf == 0:
                continue
            total += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * denom)
        return total

    # -- interface ------------------------------------------------------------

    def retrieve(self, question: Question) -> str:
        """Return the top-k most relevant CCL excerpts as a single text block.

        Matches ``CCLRetriever.retrieve`` so this can back ``run_eval --rag``.
        """
        self._ensure_built()
        query_terms = tokenize(f"{question.item_name}. {question.description}")
        if not query_terms:
            return ""
        scored = sorted(
            range(len(self._docs)),
            key=lambda i: self._score(query_terms, i),
            reverse=True,
        )[: self.top_k]
        return "\n\n".join(self._docs[i].render() for i in scored)

    def top_chunks(self, question: Question, k: int | None = None) -> list[RetrievedChunk]:
        """Return scored chunks (for diagnostics/reporting), best first."""
        self._ensure_built()
        k = k or self.top_k
        query_terms = tokenize(f"{question.item_name}. {question.description}")
        scored = sorted(
            range(len(self._docs)),
            key=lambda i: self._score(query_terms, i),
            reverse=True,
        )[:k]
        chunks = []
        for i in scored:
            c = self._docs[i]
            chunks.append(
                RetrievedChunk(eccn=c.eccn, title=c.title, text=c.text[:200],
                               score=round(self._score(query_terms, i), 3),
                               render_cap=self.render_cap)
            )
        return chunks

    # -- retrieval quality (does the right entry surface?) ---------------------

    @staticmethod
    def _gold_head(question) -> str | None:
        """CGNNN head of the question's gold ECCN, or None when there is none.

        EAR99-gold and malformed golds have no CCL entry to retrieve.
        """
        gold = (getattr(question, "gold_eccn", "") or "").strip()
        if not gold or gold == "EAR99":
            return None
        m = re.match(r"([0-9][A-E][0-9]{3})", gold)
        return m.group(1) if m else None

    def gold_rank(self, question: Question, k: int | None = None) -> int | None:
        """Rank (1-based) of the question's gold-head entry among retrieved chunks.

        ``None`` when the gold has no CCL head (EAR99/unlisted) or the head does not
        appear in the top-k. Retrieval that never surfaces the controlling entry
        cannot help the model — this makes that failure visible per question.
        """
        self._ensure_built()
        head = self._gold_head(question)
        if head is None:
            return None
        k = k or self.top_k
        query_terms = tokenize(f"{question.item_name}. {question.description}")
        if not query_terms:
            return None
        order = sorted(
            range(len(self._docs)),
            key=lambda i: self._score(query_terms, i),
            reverse=True,
        )[:k]
        for rank, i in enumerate(order, start=1):
            if self._docs[i].eccn == head:
                return rank
        return None

    def hit_rate(self, questions, k: int | None = None) -> dict:
        """Coverage of the gold entry in the top-k, over a list of questions.

        ``hit_at_k`` is the fraction of scorable questions (gold has a CCL head —
        EAR99-gold rows have no entry to retrieve) whose gold-head entry appears in
        the retrieved top-k. Returns counts + rank distribution so a RAG run can
        state whether retrieval surfaced the controlling entry at all.
        """
        self._ensure_built()
        k = k or self.top_k
        scorable = 0
        found = 0
        ranks: dict[int, int] = {}
        for q in questions:
            head = self._gold_head(q)
            if head is None:
                continue
            scorable += 1
            rank = self.gold_rank(q, k=k)
            if rank is not None:
                found += 1
                ranks[rank] = ranks.get(rank, 0) + 1
        return {
            "n_scorable": scorable,
            "gold_found": found,
            "hit_at_k": round(found / scorable, 4) if scorable else None,
            "rank_counts": dict(sorted(ranks.items())),
            "k": k,
        }


def demo(questions_path: str, index_path: str = str(DEFAULT_INDEX_PATH), top_k: int = 3) -> None:
    """Retrieve for the first few questions and print the top hits (no API calls)."""
    from ..dataset import load_questions

    retriever = BM25CCLRetriever(index_path=index_path, top_k=top_k).build()
    questions = load_questions(questions_path)
    print(f"index: {len(retriever._docs)} ECCN entries | top_k={top_k}\n")
    for q in questions[:5]:
        print(f"Q: {q.item_name}")
        print(f"   gold: {q.gold_eccn}")
        for c in retriever.top_chunks(q, k=top_k):
            print(f"   - [{c.eccn}] ({c.score}) {c.title[:70]}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BM25 retrieval over the CCL (RAG condition).")
    ap.add_argument("--questions", required=True, help="Dataset JSONL (for demo).")
    ap.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="ccl_index.json path.")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args(argv)
    demo(args.questions, args.index, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
