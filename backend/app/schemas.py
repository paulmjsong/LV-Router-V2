from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class WorkflowId(StrEnum):
    AUTO = "auto"
    DIRECT = "direct"
    REGULATIONS = "gist-regulations"
    WEB_SEARCH = "web-search"
    PAPER = "research-paper"
    GRANT = "grant"
    WEBSITE = "website"


class ModelTier(StrEnum):
    LOCAL_FAST = "local-fast"
    CLOUD_SMALL = "cloud-small"
    CLOUD_LARGE = "cloud-large"


class Quality(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


class WorkflowInfo(BaseModel):
    id: WorkflowId
    name: str
    description: str
    allowed_roles: list[str]
    enabled: bool = True
    placeholder: bool = False


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=50000)
    conversation_id: UUID | None = None
    workflow: WorkflowId = WorkflowId.AUTO
    quality: Quality = Quality.BALANCED

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class SourceCitation(BaseModel):
    source_type: Literal["document", "web"] = "document"
    chunk_id: int
    document_id: UUID
    title: str
    page: int | None = None
    score: float
    excerpt: str
    url: str | None = None
    published_at: str | None = None
    publisher: str | None = None


class ChatResponse(BaseModel):
    run_id: UUID
    conversation_id: UUID
    workflow: WorkflowId
    route_reason: str
    route_fallback: bool = False
    route_difficulty: str = ""
    answer: str
    model_tiers: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    status: str = "completed"
