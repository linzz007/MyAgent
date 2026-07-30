"""Shared API provider registry for server-side screening scripts."""

from __future__ import annotations

import re


API_KEY_NAMES = (
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "ANTHROPIC_API_KEY",
    "SILICONFLOW_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPU_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "ARK_API_KEY",
    "VOLC_API_KEY",
    "AZURE_OPENAI_API_KEY",
)

API_PROVIDER_DEFAULTS = {
    "openrouter": {
        "provider": "OpenRouter",
        "api_base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


def normalize_provider(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def api_provider_defaults(provider: str) -> dict[str, str]:
    return API_PROVIDER_DEFAULTS.get(normalize_provider(provider), {})


def provider_profiles_for_api_keys(api_keys: list[str]) -> dict[str, dict[str, object]]:
    present = set(api_keys)
    profiles: dict[str, dict[str, object]] = {}
    for defaults in API_PROVIDER_DEFAULTS.values():
        key_env = defaults["api_key_env"]
        if key_env not in present:
            continue
        provider = defaults["provider"]
        profiles[provider] = {
            "api_base_url": defaults["api_base_url"],
            "api_key_env": key_env,
            "prepare_model_gate_run_args": ["--backend", "api", "--api-provider", provider],
            "requires_model_name": True,
        }
    return dict(sorted(profiles.items()))
