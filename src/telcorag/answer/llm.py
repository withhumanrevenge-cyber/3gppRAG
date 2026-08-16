"""Chat-completion providers behind one interface.

``none`` is not a degraded mode bolted on for convenience -- it is the honest
floor of the system. With no model configured the pipeline answers extractively,
quoting retrieved clauses verbatim. That cannot hallucinate by construction, so
a reviewer with no API key still sees the retrieval and verification stack work
end to end, and the LLM path can be judged against a zero-fabrication baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1:8b",
}


class LLMError(RuntimeError):
    pass


@dataclass
class Message:
    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLM(Protocol):
    name: str
    available: bool

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str: ...


class NullLLM:
    name = "none"
    available = False

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        raise LLMError("no LLM provider configured")


class OpenAICompatible:
    """Groq, OpenAI, and anything else exposing /chat/completions."""

    def __init__(self, base_url: str, api_key: str, model: str, label: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = f"{label}:{model}"
        self.timeout = timeout
        self.available = bool(api_key)

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [m.as_dict() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LLMError(f"{self.name} HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"{self.name} malformed response: {exc}") from exc


class Ollama:
    def __init__(self, host: str, model: str, timeout: float = 180.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.timeout = timeout
        self.available = True

    def complete(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [m.as_dict() for m in messages],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LLMError(f"{self.name} HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()["message"]["content"].strip()


def resolve(settings) -> LLM:
    provider = (settings.llm_provider or "auto").lower()
    model = settings.llm_model

    def groq() -> LLM:
        return OpenAICompatible(
            "https://api.groq.com/openai/v1",
            settings.groq_api_key,
            model or DEFAULT_MODELS["groq"],
            "groq",
            settings.llm_timeout,
        )

    def openai() -> LLM:
        return OpenAICompatible(
            "https://api.openai.com/v1",
            settings.openai_api_key,
            model or DEFAULT_MODELS["openai"],
            "openai",
            settings.llm_timeout,
        )

    if provider == "groq":
        return groq()
    if provider == "openai":
        return openai()
    if provider == "ollama":
        return Ollama(settings.ollama_host, model or DEFAULT_MODELS["ollama"], settings.llm_timeout)
    if provider in ("none", "off", "extractive"):
        return NullLLM()

    if settings.groq_api_key:
        return groq()
    if settings.openai_api_key:
        return openai()
    return NullLLM()
