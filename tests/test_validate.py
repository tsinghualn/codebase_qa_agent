from pathlib import Path

from codebase_qa_agent.answer_schemas import AnswerDraft, Citation, Claim
from codebase_qa_agent.corpus_read import resolve_corpus_file
from codebase_qa_agent.schemas import Chunk
from codebase_qa_agent.validate import citation_in_retrieved, validate_draft, validate_claim


def _chunk(path: str, start: int, end: int) -> Chunk:
    return Chunk("id", path, start, end, "sym", "code", "body")


def test_citation_must_overlap_retrieved_chunk():
    cite = Citation(file="libs/langgraph/langgraph/types.py", start_line=10, end_line=20)
    chunks = [_chunk("libs/langgraph/langgraph/types.py", 1, 50)]
    assert citation_in_retrieved(cite, chunks)
    assert not citation_in_retrieved(
        Citation(file="libs/langgraph/langgraph/types.py", start_line=100, end_line=110),
        chunks,
    )


def test_validate_rejects_fake_path(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    real = corpus / "libs/langgraph/foo.py"
    real.parent.mkdir(parents=True)
    real.write_text("line1\nline2\nline3\n", encoding="utf-8")
    chunks = [_chunk("libs/langgraph/foo.py", 1, 2)]
    good = Claim(
        text="Foo exists",
        citations=[Citation(file="libs/langgraph/foo.py", start_line=1, end_line=2)],
    )
    bad = Claim(
        text="Fake",
        citations=[Citation(file="libs/langgraph/missing.py", start_line=1, end_line=1)],
    )
    assert validate_claim(good, chunks, corpus)
    assert not validate_claim(bad, chunks, corpus)
    result = validate_draft(AnswerDraft(claims=[good, bad]), chunks, corpus)
    assert result.ok
    assert len(result.valid_claims) == 1
    assert len(result.dropped_claims) == 1


def test_resolve_corpus_rejects_traversal(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert resolve_corpus_file(corpus, "../etc/passwd") is None