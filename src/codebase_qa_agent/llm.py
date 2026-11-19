"""Chat model factory (secrets from settings only)."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from codebase_qa_agent.settings import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for LLM_PROVIDER=anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required for the agent (set in .env)")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)