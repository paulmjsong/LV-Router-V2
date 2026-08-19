.PHONY: up-gpu up-ollama down logs test lint compile audit render-config secrets

up-gpu:
	LOCAL_BACKEND=vllm docker compose --profile gpu up --build

up-ollama:
	LOCAL_BACKEND=ollama docker compose --profile ollama up --build

down:
	docker compose --profile gpu --profile ollama down

logs:
	docker compose logs -f open-webui backend litellm

test:
	cd backend && python -m pytest

lint:
	cd backend && ruff check app tests

compile:
	python -m compileall backend/app backend/tests infra/litellm scripts

audit:
	python scripts/validate_static.py

render-config:
	python infra/litellm/render_config.py

secrets:
	python scripts/generate_secrets.py
