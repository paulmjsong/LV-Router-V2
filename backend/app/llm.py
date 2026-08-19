from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from .auth import UserContext
from .config import Settings

logger = logging.getLogger("saegyeol.llm")


@dataclass(slots=True)
class LLMResult:
    content: str
    requested_alias: str
    served_model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(slots=True)
class LLMStreamChunk:
    content: str
    requested_alias: str
    served_model: str
    finish_reason: str | None = None


StreamSink = Callable[[LLMStreamChunk], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _StreamRegistration:
    sink: StreamSink
    stages: frozenset[str]


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
    """The application's only model client; every request goes through LiteLLM."""

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
        self._stream_registrations: dict[str, _StreamRegistration] = {}

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

    def _effective_max_tokens(self, model_alias: str, requested: int) -> int:
        if model_alias == self.settings.local_router_model_alias:
            return min(requested, self.settings.local_router_max_tokens)
        if model_alias == "local-fast":
            return min(requested, self.settings.local_fast_max_tokens)
        return requested

    @asynccontextmanager
    async def stream_run(
        self,
        *,
        run_id: str,
        sink: StreamSink,
        stages: set[str] | frozenset[str],
    ) -> AsyncIterator[None]:
        """Stream only selected user-visible stages while preserving graph results."""
        if run_id in self._stream_registrations:
            raise RuntimeError(f"A stream is already registered for run {run_id}")
        self._stream_registrations[run_id] = _StreamRegistration(
            sink=sink,
            stages=frozenset(stages),
        )
        try:
            yield
        finally:
            self._stream_registrations.pop(run_id, None)

    @staticmethod
    def _chat_kwargs(
        *,
        user: UserContext,
        model_alias: str,
        messages: Sequence[dict[str, str]],
        run_id: str,
        workflow_id: str,
        stage: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "metadata": {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "stage": stage,
                "team_id": user.team_id or "",
            }
        }
        # Ollama may expose a separate reasoning stream for thinking models.
        # Disable it for the local control and local answer aliases.
        if model_alias in {"local-router", "local-fast"}:
            extra_body["think"] = False
        kwargs: dict[str, Any] = {
            "model": model_alias,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "user": user.user_id,
            "stream": stream,
            "extra_body": extra_body,
        }
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    async def _chat_streamed(
        self,
        *,
        user: UserContext,
        model_alias: str,
        messages: Sequence[dict[str, str]],
        run_id: str,
        workflow_id: str,
        stage: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        sink: StreamSink,
    ) -> LLMResult:
        stream = await self._client(user).chat.completions.create(
            **self._chat_kwargs(
                user=user,
                model_alias=model_alias,
                messages=messages,
                run_id=run_id,
                workflow_id=workflow_id,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                stream=True,
            )
        )

        parts: list[str] = []
        served_model = model_alias
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        reasoning_notice_sent = False

        async for chunk in stream:
            served_model = str(getattr(chunk, "model", served_model) or served_model)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            text = content if isinstance(content, str) else ""
            reasoning = None
            if delta is not None:
                reasoning = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "thinking", None)
                )
            if text:
                parts.append(text)
            elif isinstance(reasoning, str) and reasoning and not reasoning_notice_sent:
                reasoning_notice_sent = True
                await sink(
                    LLMStreamChunk(
                        content="> **Model status:** hidden reasoning suppressed.\n\n",
                        requested_alias=model_alias,
                        served_model=served_model,
                    )
                )
            await sink(
                LLMStreamChunk(
                    content=text,
                    requested_alias=model_alias,
                    served_model=served_model,
                    finish_reason=getattr(choice, "finish_reason", None),
                )
            )

        return LLMResult(
            content="".join(parts).strip(),
            requested_alias=model_alias,
            served_model=served_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

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
        max_tokens = self._effective_max_tokens(model_alias, max_tokens)
        logger.info(
            "LLM call alias=%s workflow=%s stage=%s stream=%s max_tokens=%d",
            model_alias,
            workflow_id,
            stage,
            self._stream_registrations.get(run_id) is not None,
            max_tokens,
        )
        registration = self._stream_registrations.get(run_id)
        if registration is not None and stage in registration.stages:
            return await self._chat_streamed(
                user=user,
                model_alias=model_alias,
                messages=messages,
                run_id=run_id,
                workflow_id=workflow_id,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                sink=registration.sink,
            )

        response = await self._client(user).chat.completions.create(
            **self._chat_kwargs(
                user=user,
                model_alias=model_alias,
                messages=messages,
                run_id=run_id,
                workflow_id=workflow_id,
                stage=stage,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                stream=False,
            )
        )
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
