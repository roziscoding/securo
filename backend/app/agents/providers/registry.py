from __future__ import annotations

from typing import Any, Optional

from app.agents.providers.anthropic import AnthropicProvider
from app.agents.providers.base import LLMProvider
from app.agents.providers.ollama import OllamaProvider
from app.agents.providers.openai import OpenAIProvider
from app.agents.providers.openai_chatgpt import OpenAIChatGPTProvider
from app.agents.providers.openai_compatible import OpenAICompatibleProvider


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "openai_chatgpt": OpenAIChatGPTProvider,
    "anthropic": AnthropicProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def list_providers() -> list[str]:
    return sorted(_PROVIDERS.keys())


def build_provider(
    name: str,
    *,
    api_key: str = "",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    access_token: Any = None,
) -> LLMProvider:
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {name}. Available: {list_providers()}")
    if name == "openai_chatgpt":
        if access_token is None or api_key or base_url:
            raise ValueError("openai_chatgpt requires an explicit OAuth lifecycle")
        return OpenAIChatGPTProvider(access_token=access_token, model=model)
    kwargs: dict = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url
    return cls(**{k: v for k, v in kwargs.items() if v is not None})
