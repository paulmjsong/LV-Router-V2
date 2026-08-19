from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx


def _headers() -> dict[str, str]:
    key = os.getenv("LAB_ADMIN_API_KEY", "").strip()
    if not key:
        raise SystemExit("LAB_ADMIN_API_KEY is not configured in the backend container")
    return {
        "Authorization": f"Bearer {key}",
        "X-Lab-User-Id": os.getenv("LABCTL_USER_ID", "lab-admin-cli"),
        "X-Lab-Team-Id": os.getenv("LABCTL_TEAM_ID", "lab"),
        "X-Lab-User-Roles": os.getenv("LABCTL_ROLES", "member,editor,admin"),
    }


def _request(method: str, path: str, **kwargs: Any) -> Any:
    base_url = os.getenv("LABCTL_BASE_URL", "http://localhost:8000").rstrip("/")
    with httpx.Client(base_url=base_url, headers=_headers(), timeout=300) as client:
        response = client.request(method, path, **kwargs)
    if response.is_error:
        raise SystemExit(f"{response.status_code}: {response.text}")
    return response.json()


def list_collections(_: argparse.Namespace) -> None:
    print(json.dumps(_request("GET", "/api/collections"), indent=2, ensure_ascii=False))


def create_collection(args: argparse.Namespace) -> None:
    payload = {
        "name": args.name,
        "description": args.description,
        "visibility": args.visibility,
    }
    print(
        json.dumps(
            _request("POST", "/api/collections", json=payload),
            indent=2,
            ensure_ascii=False,
        )
    )


def upload(args: argparse.Namespace) -> None:
    opened = []
    files = []
    try:
        for raw_path in args.files:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise SystemExit(f"File not found: {path}")
            handle = path.open("rb")
            opened.append(handle)
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files.append(("files", (path.name, handle, mime)))
        result = _request(
            "POST",
            "/api/documents/upload",
            data={"collection_id": args.collection_id},
            files=files,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        for handle in opened:
            handle.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Administer SaeGyeol Lab AI document collections")
    sub = root.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-collections")
    list_parser.set_defaults(func=list_collections)

    create = sub.add_parser("create-collection")
    create.add_argument("--name", required=True)
    create.add_argument("--description", default="")
    create.add_argument("--visibility", choices=["private", "team", "public"], default="team")
    create.set_defaults(func=create_collection)

    uploader = sub.add_parser("upload")
    uploader.add_argument("--collection-id", required=True)
    uploader.add_argument("files", nargs="+")
    uploader.set_defaults(func=upload)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
