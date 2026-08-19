#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


def post(client: httpx.Client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=payload)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected LiteLLM response from {path}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a budgeted LiteLLM virtual key and print its backend mapping"
    )
    parser.add_argument("--proxy", default=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"))
    parser.add_argument("--master-key", default=os.getenv("LITELLM_MASTER_KEY"))
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--user-id", help="Existing LiteLLM user ID")
    identity.add_argument("--team-id", help="Existing LiteLLM team ID")
    identity.add_argument("--create-team", metavar="ALIAS", help="Create a new LiteLLM team first")
    parser.add_argument(
        "--map-as",
        help="Application user/team claim used in LITELLM_KEYS_JSON; defaults to the supplied identity",
    )
    parser.add_argument("--budget", type=float, required=True, help="USD budget")
    parser.add_argument("--duration", default="30d")
    parser.add_argument("--rpm-limit", type=int, default=60)
    parser.add_argument("--tpm-limit", type=int, default=200000)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["local-fast", "cloud-small", "cloud-large", "embedding"],
    )
    args = parser.parse_args()
    if not args.master_key:
        parser.error("--master-key or LITELLM_MASTER_KEY is required")
    if args.budget <= 0:
        parser.error("--budget must be positive")

    headers = {"Authorization": f"Bearer {args.master_key}"}
    with httpx.Client(base_url=args.proxy.rstrip("/"), headers=headers, timeout=30.0) as client:
        team_id = args.team_id
        mapping_label: str

        if args.create_team:
            team = post(
                client,
                "/team/new",
                {
                    "team_alias": args.create_team,
                    "models": args.models,
                    "max_budget": args.budget,
                    "budget_duration": args.duration,
                    "rpm_limit": args.rpm_limit,
                    "tpm_limit": args.tpm_limit,
                },
            )
            team_id = str(team["team_id"])
            mapping_label = f"team:{args.map_as or args.create_team}"
            print(f"Created LiteLLM team {args.create_team!r} with ID {team_id}", file=sys.stderr)
            key_payload: dict[str, Any] = {"team_id": team_id, "models": args.models}
        elif team_id:
            mapping_label = f"team:{args.map_as or team_id}"
            key_payload = {
                "team_id": team_id,
                "models": args.models,
                "max_budget": args.budget,
                "budget_duration": args.duration,
                "rpm_limit": args.rpm_limit,
                "tpm_limit": args.tpm_limit,
            }
        else:
            mapping_label = f"user:{args.map_as or args.user_id}"
            key_payload = {
                "user_id": args.user_id,
                "models": args.models,
                "max_budget": args.budget,
                "budget_duration": args.duration,
                "rpm_limit": args.rpm_limit,
                "tpm_limit": args.tpm_limit,
            }

        result = post(client, "/key/generate", key_payload)
        key = result.get("key")
        if not key:
            raise RuntimeError("LiteLLM did not return a virtual key")
        print(json.dumps({mapping_label: key}, indent=2))


if __name__ == "__main__":
    main()
