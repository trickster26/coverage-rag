"""Retrieval: jurisdiction filter first, then hybrid keyword + vector search.

Local mode runs BM25 in process and, when embeddings are configured, cosine
search over cached chunk embeddings; the two ranked lists are merged with
reciprocal rank fusion. Azure mode (azure_search.py) does the same job inside
Azure AI Search.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .config import DATA
from .ingest import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
STOP = set("""a an and are as at be been but by can do does for from has have how i if in is it its
me my of on or our that the their them then there these this to was what when where which who
why will with you your am under any all no not get""".split())

EmbedFn = Callable[[list[str]], list[list[float]]]


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def rrf(result_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Uses ranks only, so BM25 and cosine scores never need comparing."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def term_coverage(query: str, text: str) -> float:
    """Share of the query's content words that appear in the text. Used to decide
    whether the best hit is evidence at all, independent of which backend ranked it."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    matched = len(q & set(tokenize(text)))
    # One shared word ("capital") is not evidence; ask for two when the query has two.
    if matched < min(2, len(q)):
        return 0.0
    return matched / len(q)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused rank score
    coverage: float  # term_coverage(query, chunk)


class Retriever(Protocol):
    def search(self, query: str, jurisdiction: str | None, top_k: int) -> list[Hit]: ...


class LocalHybridIndex:
    def __init__(self, chunks: list[Chunk], embed: EmbedFn | None = None, candidates: int = 50):
        self.chunks = chunks
        self.candidates = candidates
        self.embed = embed
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])
        self.vectors = self._load_vectors() if embed else None

    def _load_vectors(self) -> np.ndarray:
        """Embed every chunk once and cache to disk, keyed by the chunk texts."""
        texts = [c.embed_text for c in self.chunks]
        digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        cache = DATA / f"embeddings-{digest}.npy"
        if cache.exists():
            return np.load(cache)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 64):
            vectors += self.embed(texts[i:i + 64])
        arr = np.asarray(vectors, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        np.save(cache, arr)
        return arr

    def search(self, query: str, jurisdiction: str | None = None, top_k: int = 5) -> list[Hit]:
        allowed = np.array([jurisdiction is None or c.jurisdiction == jurisdiction for c in self.chunks])
        lists: list[list[str]] = []

        bm25 = np.where(allowed, self.bm25.get_scores(tokenize(query)), -np.inf)
        order = [i for i in np.argsort(-bm25)[: self.candidates] if np.isfinite(bm25[i]) and bm25[i] > 0]
        lists.append([self.chunks[i].id for i in order])

        if self.vectors is not None:
            q = np.asarray(self.embed([query])[0], dtype=np.float32)
            q /= np.linalg.norm(q)
            cos = np.where(allowed, self.vectors @ q, -np.inf)
            lists.append([self.chunks[i].id for i in np.argsort(-cos)[: self.candidates]])

        by_id = {c.id: c for c in self.chunks}
        fused = rrf(lists)[:top_k]
        return [Hit(by_id[i], s, term_coverage(query, by_id[i].embed_text)) for i, s in fused]
