"""Shared experiment model registry for server-side screening scripts."""

from __future__ import annotations

import re
from pathlib import Path


KNOWN_TESTED_LOCAL_MODELS = {
    "Qwen3-32B",
    "Qwen3-14B-AWQ",
    "Qwen2.5-14B-Instruct-AWQ",
    "Qwen2.5-14B-AWQ",
    "Qwen2.5-3B-Instruct",
}

KNOWN_TESTED_LOCAL_MODEL_KEYS = {
    "qwen332b",
    "qwen314bawq",
    "qwen2514bawq",
    "qwen2514binstructawq",
    "qwen253binstruct",
}


def normalized_model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower()).removesuffix("local")


def known_tested_model_key(*values: str | Path | None) -> str | None:
    for value in values:
        if value is None:
            continue
        key = normalized_model_key(str(value))
        if key in KNOWN_TESTED_LOCAL_MODEL_KEYS:
            return key
        if isinstance(value, Path):
            name_key = normalized_model_key(value.name)
            if name_key in KNOWN_TESTED_LOCAL_MODEL_KEYS:
                return name_key
    return None
