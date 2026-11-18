.PHONY: help install test eval check-env ingest query

VENV := .venv
PY := $(VENV)/bin/python

help:
	@echo "Targets: install, ingest, query, test, eval, check-env"

install:
	uv venv $(VENV)
	uv pip install -e ".[dev,ml]"
	@test -f .env || cp .env.example .env
	@echo "Created .env from .env.example if missing — add your API keys locally."

check-env:
	@$(PY) -c "from codebase_qa_agent.settings import get_settings; s=get_settings(); print('LLM available:', s.has_llm, '| provider:', s.llm_provider, '| chroma:', s.chroma_path)"

ingest:
	$(PY) -m codebase_qa_agent.ingest --write-pin

query:
	@$(PY) -m codebase_qa_agent.query_cli "$(Q)"

test:
	$(PY) -m pytest -q

eval:
	$(PY) eval.py