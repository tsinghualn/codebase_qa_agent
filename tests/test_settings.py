import os

from codebase_qa_agent.settings import get_settings


def test_settings_never_expose_key_in_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-not-real")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    s = get_settings()
    assert s.has_llm
    assert "sk-test" not in repr(s)