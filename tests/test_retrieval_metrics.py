from codebase_qa_agent.retrieval_metrics import (
    context_precision_at_k,
    hit_in_top_k,
    path_matches_expected,
)
from codebase_qa_agent.schemas import Chunk, RetrievalHit


def _hit(path: str) -> RetrievalHit:
    c = Chunk("id", path, 1, 10, "sym", "code", "text")
    return RetrievalHit(c, 1.0, "hybrid")


def test_path_matches_prefix():
    assert path_matches_expected("libs/langgraph/foo.py", ("libs/langgraph/",))
    assert not path_matches_expected("docs/index.md", ("libs/langgraph/",))


def test_hit_in_top_k():
    hits = [_hit("libs/langgraph/checkpoint/base.py")]
    assert hit_in_top_k(hits, ("libs/langgraph/checkpoint/",))


def test_context_precision():
    hits = [
        _hit("libs/langgraph/a.py"),
        _hit("docs/b.md"),
    ]
    p = context_precision_at_k(hits, ("libs/langgraph/",))
    assert p == 0.5