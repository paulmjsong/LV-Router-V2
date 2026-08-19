from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "SaeGyeol Lab AI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql://saegyeol:saegyeol@postgres:5432/saegyeol"

    litellm_base_url: str = "http://litellm:4000"
    litellm_default_api_key: str = "sk-change-me"
    litellm_keys_json: str = "{}"
    local_router_model_alias: str = "local-router"
    local_router_max_tokens: int = Field(default=64, ge=32, le=256)
    local_router_min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Hard ceiling for local answer generation on CPU/dev deployments.
    local_fast_max_tokens: int = Field(default=512, ge=64, le=4096)
    embedding_model_alias: str = "embedding"
    embedding_dimensions: int = Field(default=1536, ge=128, le=8192)
    llm_timeout_seconds: float = Field(default=120.0, ge=5, le=900)
    allow_direct_provider_url: bool = False

    auth_mode: Literal["dev", "oidc", "cloudflare_access", "openwebui"] = "dev"
    auth_token_header: str = "Cf-Access-Jwt-Assertion"
    oidc_jwks_url: str | None = None
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_user_id_claim: str = "sub"
    oidc_email_claim: str = "email"
    oidc_name_claim: str = "name"
    oidc_roles_claim: str = "roles"
    oidc_team_claim: str = "team_id"

    openwebui_backend_key: str = "sk-openwebui-change-me"
    openwebui_identity_jwt_secret: str = "change-this-openwebui-identity-secret"
    openwebui_identity_jwt_header: str = "X-OpenWebUI-User-Jwt"
    openwebui_default_team_id: str = "lab"
    lab_admin_api_key: str | None = None

    dev_default_user_id: str = "dev-user"
    dev_default_team_id: str = "lab"
    dev_default_roles: str = "member,editor,admin"

    cors_origins: str = "http://localhost:3000,http://localhost:8080"
    conversation_history_messages: int = Field(default=16, ge=2, le=100)
    conversation_history_chars: int = Field(default=24000, ge=2000, le=200000)

    max_upload_mb: int = Field(default=30, ge=1, le=500)
    chunk_size: int = Field(default=1400, ge=300, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    rag_top_k: int = Field(default=8, ge=1, le=30)
    rag_context_chars: int = Field(default=18000, ge=2000, le=100000)
    rag_query_chars: int = Field(default=8000, ge=500, le=30000)

    object_store: Literal["local", "s3"] = "local"
    local_object_store_path: Path = Path("/data/documents")
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_bucket: str = "saegyeol-documents"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None

    github_token: str | None = None
    github_repository: str | None = None
    github_base_branch: str = "main"
    github_allowed_paths: str = "src/content,src/data,public"

    @field_validator("litellm_base_url")
    @classmethod
    def normalize_litellm_url(cls, value: str) -> str:
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_security_settings(self) -> Self:
        if self.auth_mode in {"oidc", "cloudflare_access"} and not self.oidc_jwks_url:
            raise ValueError("OIDC_JWKS_URL is required for oidc and cloudflare_access modes")
        if self.auth_mode == "openwebui":
            if not self.openwebui_backend_key or self.openwebui_backend_key == "sk-openwebui-change-me":
                raise ValueError("OPENWEBUI_BACKEND_KEY must be changed in openwebui mode")
            if (
                not self.openwebui_identity_jwt_secret
                or self.openwebui_identity_jwt_secret == "change-this-openwebui-identity-secret"
            ):
                raise ValueError("OPENWEBUI_IDENTITY_JWT_SECRET must be changed in openwebui mode")
        if self.environment == "production":
            if self.auth_mode == "dev":
                raise ValueError("AUTH_MODE=dev is forbidden in production")
            if self.litellm_default_api_key in {"", "sk-change-me"}:
                raise ValueError("A restricted BACKEND_LITELLM_KEY is required in production")
            if "*" in self.cors_origin_list:
                raise ValueError("Wildcard CORS origins are forbidden in production")
        return self

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url.startswith("postgresql+psycopg://"):
            return self.database_url
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        if self.database_url.startswith("postgres://"):
            return self.database_url.replace("postgres://", "postgresql+psycopg://", 1)
        return self.database_url

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def dev_role_set(self) -> set[str]:
        return {item.strip() for item in self.dev_default_roles.split(",") if item.strip()}

    @property
    def github_allowed_path_list(self) -> list[str]:
        return [item.strip().strip("/") for item in self.github_allowed_paths.split(",") if item.strip()]

    @property
    def litellm_key_map(self) -> dict[str, str]:
        try:
            parsed = json.loads(self.litellm_keys_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("LITELLM_KEYS_JSON must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("LITELLM_KEYS_JSON must be a JSON object")
        return {str(k): str(v) for k, v in parsed.items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
