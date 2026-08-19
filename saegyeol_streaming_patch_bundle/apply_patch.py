from __future__ import annotations

import argparse
import compileall
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPLACEMENTS = [
    ".env.example",
    "backend/app/config.py",
    "backend/app/llm.py",
    "backend/app/openai_compat.py",
    "backend/app/routing.py",
    "backend/app/runtime.py",
    "backend/tests/test_routing.py",
    "docker-compose.yml",
    "infra/litellm/render_config.py",
]

ENV_UPDATES = {
    "LITELLM_NUM_RETRIES": "0",
    "LITELLM_REQUEST_TIMEOUT": "240",
    "LOCAL_ROUTER_MAX_TOKENS": "64",
    "LOCAL_ROUTER_MIN_CONFIDENCE": "0.0",
    "OLLAMA_ROUTER_MODEL_ID": "qwen3:0.6b",
    "LLM_TIMEOUT_SECONDS": "240",
}

PLACEHOLDER_KEY_MAPS = {
    '{"team:lab":"sk-team-key"}',
    "{'team:lab':'sk-team-key'}",
}


def parse_env(path: Path) -> tuple[list[str], dict[str, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    positions: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        positions[key] = i
    return lines, positions


def update_env(path: Path) -> list[str]:
    notes: list[str] = []
    if not path.exists():
        notes.append("No .env file was found; copy .env.example to .env before starting.")
        return notes

    lines, positions = parse_env(path)

    old_key = "LOCAL_ANSWER_MIN_CONFIDENCE"
    if old_key in positions:
        lines[positions[old_key]] = "# Removed by streaming patch: LOCAL_ANSWER_MIN_CONFIDENCE"

    for key, value in ENV_UPDATES.items():
        if key in positions:
            lines[positions[key]] = f"{key}={value}"
        else:
            lines.append(f"{key}={value}")

    if "OLLAMA_MODEL_ID" not in positions:
        lines.append("OLLAMA_MODEL_ID=qwen3:4b")

    if "LITELLM_KEYS_JSON" in positions:
        current = lines[positions["LITELLM_KEYS_JSON"]].split("=", 1)[1].strip()
        if current in PLACEHOLDER_KEY_MAPS:
            lines[positions["LITELLM_KEYS_JSON"]] = "LITELLM_KEYS_JSON={}"
            notes.append("Replaced the placeholder team LiteLLM key map with {}.")
        elif current not in {"", "{}"}:
            notes.append("Preserved your custom LITELLM_KEYS_JSON mapping.")
    else:
        lines.append("LITELLM_KEYS_JSON={}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the SaeGyeol low-latency routing and true-streaming patch."
    )
    parser.add_argument(
        "project_root",
        nargs="?",
        default=".",
        help="Path to the SaeGyeol project root (default: current directory).",
    )
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parent
    replacement_root = bundle_root / "replacements"
    project_root = Path(args.project_root).resolve()

    required_markers = [
        project_root / "backend" / "app" / "runtime.py",
        project_root / "infra" / "litellm" / "render_config.py",
        project_root / "docker-compose.yml",
    ]
    missing = [str(path) for path in required_markers if not path.exists()]
    if missing:
        print("ERROR: This does not look like the SaeGyeol project root.", file=sys.stderr)
        for item in missing:
            print(f"  missing: {item}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = project_root / f".patch-backup-{stamp}"
    backup_root.mkdir(parents=True, exist_ok=False)

    env_path = project_root / ".env"
    files_to_backup = [project_root / rel for rel in REPLACEMENTS]
    if env_path.exists():
        files_to_backup.append(env_path)

    for source in files_to_backup:
        if not source.exists():
            continue
        relative = source.relative_to(project_root)
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for rel in REPLACEMENTS:
        source = replacement_root / rel
        destination = project_root / rel
        if not source.exists():
            print(f"ERROR: patch bundle is missing {source}", file=sys.stderr)
            return 3
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    notes = update_env(env_path)

    compile_targets = [
        str(project_root / "backend" / "app"),
        str(project_root / "backend" / "tests"),
        str(project_root / "infra" / "litellm"),
    ]
    syntax_ok = all(compileall.compile_dir(path, quiet=1) for path in compile_targets)
    if not syntax_ok:
        print("ERROR: Python syntax validation failed. Restore from:", file=sys.stderr)
        print(f"  {backup_root}", file=sys.stderr)
        return 4

    docker_validation = "not run"
    try:
        completed = subprocess.run(
            ["docker", "compose", "--profile", "ollama", "config", "--quiet"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            docker_validation = "PASS"
        else:
            docker_validation = "FAILED"
            print(completed.stderr.strip(), file=sys.stderr)
    except FileNotFoundError:
        docker_validation = "skipped (docker command not found)"

    print("Patch applied.")
    print(f"Backup: {backup_root}")
    print("Python syntax validation: PASS")
    print(f"Docker Compose validation: {docker_validation}")
    for note in notes:
        print(f"Note: {note}")
    print()
    print("Next commands (PowerShell):")
    print("  docker compose --profile ollama down --remove-orphans")
    print("  docker compose --profile ollama up -d --build --force-recreate")
    print("  docker compose logs -f backend litellm ollama open-webui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
