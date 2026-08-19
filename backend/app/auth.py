from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from jwt import PyJWKClient
from pydantic import BaseModel, Field

from .config import Settings, get_settings


class UserContext(BaseModel):
    user_id: str
    email: str | None = None
    display_name: str | None = None
    team_id: str | None = None
    roles: set[str] = Field(default_factory=set)

    def has_any_role(self, allowed: set[str]) -> bool:
        return bool(self.roles.intersection(allowed))


@lru_cache(maxsize=8)
def _jwk_client(url: str) -> PyJWKClient:
    return PyJWKClient(url, cache_keys=True)


def _claim_as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.replace(" ", ",").split(",") if part.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(part).strip() for part in value if str(part).strip()}
    return {str(value)}


def _bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization", "")
    return authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else None


def _decode_token(token: str, settings: Settings) -> dict[str, Any]:
    if not settings.oidc_jwks_url:
        raise HTTPException(status_code=500, detail="OIDC_JWKS_URL is not configured")
    try:
        signing_key = _jwk_client(settings.oidc_jwks_url).get_signing_key_from_jwt(token)
        options = {"verify_aud": bool(settings.oidc_audience)}
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
            options=options,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from exc


def _openwebui_roles(role: str | None) -> set[str]:
    # Open WebUI currently exposes coarse `admin` and `user` roles. The backend keeps
    # its own finer-grained policy and maps conservatively.
    if (role or "").strip().lower() == "admin":
        return {"member", "editor", "admin"}
    return {"member"}


def _service_user(request: Request, settings: Settings) -> UserContext | None:
    if not settings.lab_admin_api_key:
        return None
    token = _bearer_token(request)
    if not token or not secrets.compare_digest(token, settings.lab_admin_api_key):
        return None
    user_id = request.headers.get("X-Lab-User-Id", "lab-admin-cli").strip()
    team_id = request.headers.get("X-Lab-Team-Id", settings.openwebui_default_team_id).strip() or None
    roles = _claim_as_set(request.headers.get("X-Lab-User-Roles")) or {"member", "editor", "admin"}
    return UserContext(user_id=user_id, team_id=team_id, roles=roles)


def _openwebui_user(request: Request, settings: Settings) -> UserContext:
    token = _bearer_token(request)
    if not token or not secrets.compare_digest(token, settings.openwebui_backend_key):
        raise HTTPException(status_code=401, detail="Invalid Open WebUI backend credential")

    identity_jwt = request.headers.get(settings.openwebui_identity_jwt_header)
    if not identity_jwt:
        raise HTTPException(status_code=401, detail="Missing signed Open WebUI identity header")
    try:
        claims = jwt.decode(
            identity_jwt,
            settings.openwebui_identity_jwt_secret,
            algorithms=["HS256"],
            issuer="open-webui",
            options={"verify_aud": False},
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid Open WebUI identity token") from exc

    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Open WebUI identity lacks a user ID")
    return UserContext(
        user_id=user_id,
        email=(str(claims.get("email")).strip() if claims.get("email") else None),
        display_name=(str(claims.get("name")).strip() if claims.get("name") else None),
        team_id=settings.openwebui_default_team_id or None,
        roles=_openwebui_roles(str(claims.get("role", "user"))),
    )


async def verify_openai_backend_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Authenticate OpenAI-compatible discovery endpoints.

    `/v1/models` may be fetched before Open WebUI has attached a user identity JWT, so it
    validates only the least-privilege backend connection key.
    """
    token = _bearer_token(request)
    if settings.auth_mode == "dev":
        return
    if settings.lab_admin_api_key and token and secrets.compare_digest(token, settings.lab_admin_api_key):
        return
    if settings.auth_mode == "openwebui" and token and secrets.compare_digest(
        token, settings.openwebui_backend_key
    ):
        return
    raise HTTPException(status_code=401, detail="Invalid backend credential")


async def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> UserContext:
    service_user = _service_user(request, settings)
    if service_user is not None:
        return service_user

    if settings.auth_mode == "dev":
        user_id = request.headers.get("X-User-Id", settings.dev_default_user_id).strip()
        team_id = request.headers.get("X-Team-Id", settings.dev_default_team_id).strip() or None
        roles = _claim_as_set(request.headers.get("X-User-Roles")) or settings.dev_role_set
        if not user_id:
            raise HTTPException(status_code=401, detail="Missing development user ID")
        return UserContext(user_id=user_id, team_id=team_id, roles=roles)

    if settings.auth_mode == "openwebui":
        return _openwebui_user(request, settings)

    if settings.auth_mode == "cloudflare_access":
        token = request.headers.get(settings.auth_token_header)
    else:
        token = _bearer_token(request)

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    claims = _decode_token(token, settings)
    user_id = str(claims.get(settings.oidc_user_id_claim, "")).strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Token does not contain a user identifier")

    return UserContext(
        user_id=user_id,
        email=claims.get(settings.oidc_email_claim),
        display_name=claims.get(settings.oidc_name_claim),
        team_id=(str(claims.get(settings.oidc_team_claim)).strip() if claims.get(settings.oidc_team_claim) else None),
        roles=_claim_as_set(claims.get(settings.oidc_roles_claim)),
    )
