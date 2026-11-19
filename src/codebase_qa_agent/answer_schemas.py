"""Structured LLM outputs for cited answers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    file: str = Field(description="Repo-relative path from the LangGraph corpus pin")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class Claim(BaseModel):
    text: str = Field(description="One factual statement supported by the citations")
    citations: list[Citation] = Field(min_length=1)


class AnswerDraft(BaseModel):
    claims: list[Claim] = Field(min_length=1)