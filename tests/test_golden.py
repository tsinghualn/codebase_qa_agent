from pathlib import Path

from codebase_qa_agent.evals.golden import golden_stats, load_golden


def test_golden_loads_minimum_count():
    cases = load_golden(Path(__file__).resolve().parents[1] / "evals" / "golden.jsonl")
    stats = golden_stats(cases)
    assert stats["total"] >= 15
    assert stats["unanswerable"] >= 2