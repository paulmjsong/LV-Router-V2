from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class WorkflowId(StrEnum):
    AUTO = "auto"
    DIRECT = "direct"
    DOMAIN_RAG = "domain_rag"
    PAPER = "paper"
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


class Visibility(StrEnum):
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


class WorkflowInfo(BaseModel):
    id: WorkflowId
    name: str
    description: str
    allowed_roles: list[str]
    mutating: bool = False


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=50000)
    conversation_id: UUID | None = None
    workflow: WorkflowId = WorkflowId.AUTO
    quality: Quality = Quality.BALANCED
    collection_ids: list[UUID] = Field(default_factory=list, max_length=20)
    use_documents: bool | None = None

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class SourceCitation(BaseModel):
    chunk_id: int
    document_id: UUID
    title: str
    page: int | None = None
    score: float
    excerpt: str


class PendingAction(BaseModel):
    action_type: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    run_id: UUID
    conversation_id: UUID
    workflow: WorkflowId
    route_reason: str
    answer: str
    model_tiers: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    status: str = "completed"
    pending_action: PendingAction | None = None


class RunDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    feedback: str | None = Field(default=None, max_length=4000)


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    visibility: Visibility = Visibility.TEAM


class CollectionInfo(BaseModel):
    id: UUID
    name: str
    description: str
    visibility: Visibility
    owner_user_id: str
    team_id: str | None
    created_at: datetime


class DocumentInfo(BaseModel):
    id: UUID
    collection_id: UUID
    filename: str
    mime_type: str
    status: str
    chunk_count: int = 0
    created_at: datetime


class UploadResponse(BaseModel):
    documents: list[DocumentInfo]
