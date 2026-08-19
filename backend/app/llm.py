from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import urlparse

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from .auth import UserContext
from .config import Settings


@dataclass(slots=True)
class LLMResult:
    content: str
    requested_alias: str
    served_model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LiteLLMKeyResolver:
    """Resolve a LiteLLM virtual key without exposing it to the browser."""

    def __init__(self, settings: Settings) -> None:
        self.default_key = settings.litellm_default_api_key
        self.key_map = settings.litellm_key_map

    def resolve(self, user: UserContext) -> str:
        user_key = self.key_map.get(f"user:{user.user_id}")
        if user_key:
            return user_key
        if user.team_id:
            team_key = self.key_map.get(f"team:{user.team_id}")
            if team_key:
                return team_key
        return self.key_map.get("default", self.default_key)


class LLMGateway:
    """The only model client used by the application.

    Provider credentials never live here. Every request goes to the LiteLLM Proxy,
    which owns provider selection, budgets, retries, fallbacks, and spend logging.
    """

    _DIRECT_PROVIDER_HOSTS = {
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.mistral.ai",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key_resolver = LiteLLMKeyResolver(settings)
        parsed = urlparse(settings.litellm_base_url)
        if not settings.allow_direct_provider_url and parsed.hostname in self._DIRECT_PROVIDER_HOSTS:
            raise ValueError("LITELLM_BASE_URL points to a model provider instead of the LiteLLM Proxy")
        base_url = settings.litellm_base_url.rstrip("/")
        self.base_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        self._clients: dict[str, Any] = {}

    def _client(self, user: UserContext) -> Any:
        api_key = self.key_resolver.resolve(user)
        client = self._clients.get(api_key)
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("The openai package is required to use LiteLLM") from exc
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,  # retries and fallbacks belong in LiteLLM
            )
            self._clients[api_key] = client
        return client

    async def chat(
        self,
        *,
        user: UserContext,
        model_alias: str,
        messages: Sequence[dict[str, str]],
        run_id: str,
        workflow_id: str,
        stage: str,
        temperature: float = 0.2,
        max_tokens: int = 1800,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model_alias,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "user": user.user_id,
            "extra_body": {
                "metadata": {
                    "run_id": run_id,
                    "workflow_id": workflow_id,
                    "stage": stage,
                    "team_id": user.team_id or "",
                }
            },
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await self._client(user).chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResult(
            content=content.strip(),
            requested_alias=model_alias,
            served_model=str(getattr(response, "model", model_alias)),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    async def embed_texts(
        self,
        *,
        user: UserContext,
        texts: Sequence[str],
        run_id: str,
        stage: str,
    ) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client(user).embeddings.create(
            model=self.settings.embedding_model_alias,
            input=list(texts),
            user=user.user_id,
            extra_body={
                "metadata": {
                    "run_id": run_id,
                    "workflow_id": "document_indexing",
                    "stage": stage,
                    "team_id": user.team_id or "",
                }
            },
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        for vector in vectors:
            if len(vector) != self.settings.embedding_dimensions:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {self.settings.embedding_dimensions}, received {len(vector)}"
                )
        return vectors

    async def close(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.close()

    @staticmethod
    def extract_json_object(content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        return parsed
