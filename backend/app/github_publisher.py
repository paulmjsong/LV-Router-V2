from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import httpx

from .config import Settings


@dataclass(slots=True)
class PublishResult:
    published: bool
    message: str
    pull_request_url: str | None = None


class GitHubPublisher:
    """Create a branch and pull request after explicit human approval."""

    def __init__(self, settings: Settings) -> None:
        self.token = settings.github_token
        self.repository = settings.github_repository
        self.base_branch = settings.github_base_branch
        self.allowed_paths = settings.github_allowed_path_list

    @property
    def configured(self) -> bool:
        return bool(self.token and self.repository and "/" in self.repository)

    def validate_path(self, raw_path: str) -> str:
        path = str(PurePosixPath(raw_path.strip().lstrip("/")))
        if path.startswith("../") or path == "." or ".." in PurePosixPath(path).parts:
            raise ValueError("Unsafe repository path")
        if not any(path == prefix or path.startswith(f"{prefix}/") for prefix in self.allowed_paths):
            raise ValueError(f"Path is outside the allowed website directories: {path}")
        return path

    async def publish(self, proposal: dict[str, Any], run_id: str) -> PublishResult:
        if not self.configured:
            return PublishResult(
                published=False,
                message="The change was approved, but GitHub publishing is not configured.",
            )

        path = self.validate_path(str(proposal["path"]))
        content = str(proposal["content"])
        branch = f"saegyeol-ai/{run_id[:12]}"
        owner_repo = self.repository
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        api = f"https://api.github.com/repos/{owner_repo}"

        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            ref_response = await client.get(f"{api}/git/ref/heads/{self.base_branch}")
            ref_response.raise_for_status()
            base_sha = ref_response.json()["object"]["sha"]

            create_ref = await client.post(
                f"{api}/git/refs",
                json={"ref": f"refs/heads/{branch}", "sha": base_sha},
            )
            create_ref.raise_for_status()

            existing_sha: str | None = None
            existing = await client.get(f"{api}/contents/{path}", params={"ref": self.base_branch})
            if existing.status_code == 200:
                existing_sha = existing.json().get("sha")
            elif existing.status_code != 404:
                existing.raise_for_status()

            payload: dict[str, Any] = {
                "message": str(proposal.get("commit_message") or "Update website content via SaeGyeol AI"),
                "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                "branch": branch,
            }
            if existing_sha:
                payload["sha"] = existing_sha
            update = await client.put(f"{api}/contents/{path}", json=payload)
            update.raise_for_status()

            pull = await client.post(
                f"{api}/pulls",
                json={
                    "title": str(proposal.get("pr_title") or "Website update from SaeGyeol AI"),
                    "head": branch,
                    "base": self.base_branch,
                    "body": str(proposal.get("pr_body") or "Human-approved website change."),
                },
            )
            pull.raise_for_status()
            url = pull.json().get("html_url")
            return PublishResult(True, "Created a pull request for the approved change.", url)
