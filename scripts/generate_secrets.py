from __future__ import annotations

import secrets


for name, prefix in [
    ("POSTGRES_PASSWORD", ""),
    ("MINIO_ROOT_PASSWORD", ""),
    ("OPENWEBUI_BACKEND_KEY", "sk-"),
    ("OPENWEBUI_IDENTITY_JWT_SECRET", ""),
    ("OPENWEBUI_SECRET_KEY", ""),
    ("LAB_ADMIN_API_KEY", "sk-"),
    ("LITELLM_MASTER_KEY", "sk-"),
    ("LITELLM_SALT_KEY", "sk-"),
]:
    print(f"{name}={prefix}{secrets.token_urlsafe(36)}")
