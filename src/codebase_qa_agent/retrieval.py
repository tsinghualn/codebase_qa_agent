"""Hybrid retrieval: BM25 + vectors (RRF), optional cross-encoder rerank (M3)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from codebase_qa_agent.bm25_index import BM25Index
from codebase_qa_agent.langchain_rag import open_chroma_vectorstore, vector_search
from codebase_qa_agent.rerank import DEFAULT_RERANK_MODEL, rerank_candidates
from codebase_qa_agent.retrieval_metrics import hit_in_top_k as _hit_in_top_k
from codebase_qa_agent.schemas import Chunk, RetrievalHit
from codebase_qa_agent.settings import Settings, get_settings

RetrievalMode = Literal["vector", "hybrid"]

hit_in_top_k = _hit_in_top_k
RRF_K = 60
RERANK_PREFETCH_FACTOR = 4


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    *,
    k: int = RRF_K,
) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk.chunk_id] = chunk
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunks[cid], score) for cid, score in ordered]


class RetrievalEngine:
    def __init__(
        self,
        chroma_path: Path,
        embedding_model_name: str,
        *,
        mode: RetrievalMode = "hybrid",
        settings: Settings | None = None,
        use_rerank: bool | None = None,
        rerank_model: str | None = None,
    ) -> None:
        self._chroma_path = chroma_path
        self._mode: RetrievalMode = mode
        settings = settings or get_settings()
        self._store = open_chroma_vectorstore(chroma_path, settings)
        self._bm25 = BM25Index.load(chroma_path)
        self._use_rerank = settings.rerank_enabled if use_rerank is None else use_rerank
        self._rerank_model = rerank_model or settings.rerank_model
        self._cross_encoder = None

    def search(self, query: str, *, top_k: int = 5, chunk_type: str | None = None) -> list[RetrievalHit]:
        prefetch = max(top_k * RERANK_PREFETCH_FACTOR, 20)
        vector_hits = vector_search(self._store, query, top_k=prefetch, chunk_type=chunk_type)
        vector_ranked = [c for c, _ in vector_hits]

        if self._mode == "vector":
            fused = list(vector_hits)
            source = "vector"
        else:
            bm25_hits = self._bm25.search(query, top_k=prefetch)
            if chunk_type:
                bm25_hits = [(c, s) for c, s in bm25_hits if c.chunk_type == chunk_type]
            bm25_ranked = [c for c, _ in bm25_hits]
            fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked]) if bm25_ranked else list(
                vector_hits
            )
            source = "hybrid"

        if self._use_rerank and fused:
            fused, self._cross_encoder = rerank_candidates(
                query,
                fused,
                self._rerank_model,
                top_k=top_k,
                cross_encoder=self._cross_encoder,
            )
            source = f"{source}+rerank"

        out: list[RetrievalHit] = []
        for chunk, score in fused[:top_k]:
            out.append(RetrievalHit(chunk=chunk, score=score, source=source))
        return out