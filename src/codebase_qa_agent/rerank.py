"""Cross-encoder reranking over fused retrieval candidates (M3)."""

from __future__ import annotations

from codebase_qa_agent.schemas import Chunk

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_candidates(
    query: str,
    candidates: list[tuple[Chunk, float]],
    model_name: str,
    *,
    top_k: int,
    cross_encoder=None,
) -> tuple[list[tuple[Chunk, float]], object]:
    """Return top_k chunks reranked by cross-encoder relevance."""
    if not candidates:
        return [], cross_encoder
    if cross_encoder is None:
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(model_name)
    pairs = [(query, chunk.text[:2000]) for chunk, _ in candidates]
    scores = cross_encoder.predict(pairs)
    scored = sorted(
        zip(candidates, scores, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    out = [(chunk, float(score)) for (chunk, _), score in scored[:top_k]]
    return out, cross_encoder