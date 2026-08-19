from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name, "true" if default else "false").lower()
    return value in {"1", "true", "yes", "on"}


def extra_json(name: str) -> dict[str, Any]:
    raw = env(name, "{}")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return parsed


def deployment(
    *,
    alias: str,
    model: str,
    key_env: str | None = None,
    api_base: str = "",
    extras: dict[str, Any] | None = None,
    mode: str = "chat",
    zero_cost: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {"model": model}
    if key_env:
        params["api_key"] = f"os.environ/{key_env}"
    if api_base:
        params["api_base"] = api_base
    params.update(extras or {})
    if zero_cost:
        params["input_cost_per_token"] = 0
        params["output_cost_per_token"] = 0
    return {"model_name": alias, "litellm_params": params, "model_info": {"mode": mode}}


def local_deployment(alias: str) -> dict[str, Any]:
    backend = env("LOCAL_BACKEND", "vllm").lower()
    extras = extra_json("LOCAL_MODEL_EXTRA_JSON")
    if backend == "vllm":
        model_id = env("LOCAL_MODEL_ID", "Qwen/Qwen3-8B")
        return deployment(
            alias=alias,
            model=f"hosted_vllm/{model_id}",
            key_env="VLLM_API_KEY",
            api_base=env("VLLM_API_BASE", "http://vllm:8000/v1"),
            extras=extras,
            zero_cost=True,
        )
    if backend == "ollama":
        if alias == "local-router":
            model_id = env("OLLAMA_ROUTER_MODEL_ID", "qwen3:0.6b")
        else:
            model_id = env("OLLAMA_MODEL_ID", "qwen3:4b")
        return deployment(
            alias=alias,
            model=f"ollama_chat/{model_id}",
            api_base=env("OLLAMA_API_BASE", "http://ollama:11434"),
            extras=extras,
            zero_cost=True,
        )
    raise ValueError("LOCAL_BACKEND must be either 'vllm' or 'ollama'")


small = deployment(
    alias="cloud-small",
    model=env("CLOUD_SMALL_MODEL", "openai/gpt-5-mini"),
    key_env="CLOUD_SMALL_API_KEY",
    api_base=env("CLOUD_SMALL_API_BASE"),
    extras=extra_json("CLOUD_SMALL_EXTRA_JSON"),
)
large = deployment(
    alias="cloud-large",
    model=env("CLOUD_LARGE_MODEL", "openai/gpt-5"),
    key_env="CLOUD_LARGE_API_KEY",
    api_base=env("CLOUD_LARGE_API_BASE"),
    extras=extra_json("CLOUD_LARGE_EXTRA_JSON"),
)
embedding = deployment(
    alias="embedding",
    model=env("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
    key_env="EMBEDDING_API_KEY",
    api_base=env("EMBEDDING_API_BASE"),
    extras=extra_json("EMBEDDING_EXTRA_JSON"),
    mode="embedding",
)

model_list: list[dict[str, Any]] = [
    local_deployment("local-router"),
    local_deployment("local-fast"),
    small,
    large,
    embedding,
]

fallbacks: list[dict[str, list[str]]] = [{"cloud-small": ["cloud-large"]}]
if env_bool("ALLOW_LOCAL_MODEL_FALLBACK", False):
    fallbacks.append({"local-fast": ["cloud-small"]})
if env_bool("ALLOW_REMOTE_ROUTER_FALLBACK", False):
    # Disabled by default: a failed local router should not silently create a paid routing call.
    fallbacks.append({"local-router": ["cloud-small"]})

router_settings: dict[str, Any] = {
    "routing_strategy": env("LITELLM_ROUTING_STRATEGY", "least-busy"),
    "num_retries": int(env("LITELLM_NUM_RETRIES", "0")),
    "timeout": int(env("LITELLM_REQUEST_TIMEOUT", "240")),
    "fallbacks": fallbacks,
}
if env("REDIS_HOST"):
    router_settings.update(
        {
            "redis_host": env("REDIS_HOST"),
            "redis_port": int(env("REDIS_PORT", "6379")),
            "redis_password": env("REDIS_PASSWORD"),
        }
    )

litellm_settings: dict[str, Any] = {"drop_params": True}
if env("LANGFUSE_PUBLIC_KEY") and env("LANGFUSE_SECRET_KEY"):
    litellm_settings["success_callback"] = ["langfuse"]
    litellm_settings["failure_callback"] = ["langfuse"]

config = {
    "model_list": model_list,
    "router_settings": router_settings,
    "litellm_settings": litellm_settings,
    "general_settings": {
        "master_key": "os.environ/LITELLM_MASTER_KEY",
        "database_url": "os.environ/DATABASE_URL",
        "store_model_in_db": False,
        "json_logs": True,
    },
}

output = Path(env("LITELLM_RENDERED_CONFIG", "/tmp/litellm-config.yaml"))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(output)
