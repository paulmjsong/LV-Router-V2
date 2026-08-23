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

    app_name: str = "Infonet AI Router"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql://infonet:infonet@postgres:5432/infonet"

    litellm_base_url: str = "http://litellm:4000"
    litellm_default_api_key: str = "sk-change-me"
    litellm_keys_json: str = "{}"
    local_router_model_alias: str = "local-router"
    local_router_max_tokens: int = Field(default=64, ge=32, le=256)
    local_router_min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    local_router_fallback_tier: Literal["cloud-small", "cloud-large"] = "cloud-small"
    local_fast_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    local_fast_max_tokens: int = Field(default=512, ge=64, le=4096)
    embedding_model_alias: str = "embedding"
    embedding_dimensions: int = Field(default=1536, ge=128, le=8192)
    llm_timeout_seconds: float = Field(default=240.0, ge=5, le=900)
    allow_direct_provider_url: bool = False

    gist_regulations_index_dir: Path = Path("/app/jireumgil_index")
    gist_regulations_top_k: int = Field(default=5, ge=1, le=20)
    gist_regulations_vector_dimensions: int = Field(default=1536, ge=128, le=8192)
    gist_regulations_context_chars: int = Field(default=16000, ge=2000, le=50000)

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
        if self.gist_regulations_vector_dimensions != self.embedding_dimensions:
            raise ValueError(
                "The supplied GIST FAISS index was built with the configured embedding dimension; "
                "GIST_REGULATIONS_VECTOR_DIMENSIONS must match EMBEDDING_DIMENSIONS"
            )
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
