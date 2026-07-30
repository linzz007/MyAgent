#!/usr/bin/env python3
"""Healthcheck an OpenAI-compatible API profile without printing secrets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def model_ids(payload: Any) -> list[str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def fetch_models(api_base_url: str, api_key: str, timeout: float) -> list[str]:
    url = f"{api_base_url.rstrip('/')}/models"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return model_ids(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"missing API key env: {args.api_key_env}", file=sys.stderr)
        return 1

    try:
        models = fetch_models(args.api_base_url, api_key, args.timeout)
    except HTTPError as exc:
        print(f"API healthcheck failed: HTTP {exc.code} from {args.api_base_url.rstrip('/')}/models", file=sys.stderr)
        return 1
    except (OSError, URLError, json.JSONDecodeError) as exc:
        print(f"API healthcheck failed: {type(exc).__name__} from {args.api_base_url.rstrip('/')}/models", file=sys.stderr)
        return 1

    if args.model not in models:
        preview = ", ".join(models[:5]) if models else "none"
        print(f"model not listed by API: {args.model}; first listed models: {preview}", file=sys.stderr)
        return 1

    print(f"api healthcheck ok: {args.api_base_url.rstrip('/')} model={args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
