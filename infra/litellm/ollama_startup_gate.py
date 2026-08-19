from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = os.getenv("OLLAMA_API_BASE", "http://ollama:11434").rstrip("/")
ROUTER_MODEL = os.getenv("OLLAMA_ROUTER_MODEL_ID", "qwen3:0.6b")
ANSWER_MODEL = os.getenv("OLLAMA_MODEL_ID", "qwen3:4b-instruct-2507-q4_K_M")
STARTUP_TIMEOUT = int(os.getenv("OLLAMA_STARTUP_TIMEOUT_SECONDS", "1800"))
PRELOAD_TIMEOUT = int(os.getenv("OLLAMA_PRELOAD_TIMEOUT_SECONDS", "300"))
POLL_SECONDS = 2


def request_json(path: str, *, payload: dict | None = None, timeout: int = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def available_models() -> set[str]:
    payload = request_json("/api/tags", timeout=10)
    names: set[str] = set()
    for item in payload.get("models", []):
        for key in ("name", "model"):
            value = item.get(key)
            if value:
                names.add(str(value))
    return names


def present(requested: str, names: set[str]) -> bool:
    if requested in names:
        return True
    # Ollama may canonicalize an implicit :latest tag.
    if ":" not in requested and f"{requested}:latest" in names:
        return True
    return False


def wait_for_models(models: list[str]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last_seen: set[str] = set()
    while time.monotonic() < deadline:
        try:
            last_seen = available_models()
            missing = [m for m in models if not present(m, last_seen)]
            if not missing:
                return
            print(f"Waiting for Ollama model download(s): {', '.join(missing)}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"Waiting for Ollama API: {exc}", flush=True)
        time.sleep(POLL_SECONDS)
    raise SystemExit(
        "ERROR: Ollama models did not become available before timeout. "
        f"Requested={models}; seen={sorted(last_seen)}"
    )


def preload(model: str) -> None:
    print(f"Preloading Ollama model without generation: {model}", flush=True)
    request_json(
        "/api/generate",
        payload={"model": model, "prompt": "", "stream": False, "keep_alive": -1},
        timeout=PRELOAD_TIMEOUT,
    )
    print(f"Ollama model resident: {model}", flush=True)


def main() -> None:
    models = list(dict.fromkeys([ROUTER_MODEL, ANSWER_MODEL]))
    wait_for_models(models)
    for model in models:
        preload(model)
    print("Ollama startup gate passed; LiteLLM may start.", flush=True)


if __name__ == "__main__":
    main()
