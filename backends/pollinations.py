"""Pollinations backend (primary).

https://gen.pollinations.ai/v1 — OpenAI-compatible, officially supports
Hermes Agent via `polli harness`. Free entry: signup at
enter.pollinations.ai/keys, starter Pollen via Quests, no card/deposit.
Text models support tools/streaming/reasoning/vision per catalog.
Public catalog: GET /v1/models (no auth).
"""
from __future__ import annotations

from typing import Any

from backends.base import BackendAdapter, ModelInfo, _as_int

# Curated coding/agent fallback when catalog is unreachable.
# Verified 2026-10-06 from live /v1/models (tools=True).
# Order: models confirmed runnable on the free starter balance first.
CURATED_MODELS: tuple[str, ...] = (
    "inclusionai/ling-3.1-flash",
    "openai/gpt-5.4-nano",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "qwen/qwen3-coder-next",
    "moonshotai/kimi-k2.7-code",
    "openai/gpt-5.3-codex",
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.3",
    "xiaomi/mimo-v2.5",
)
DEFAULT_MODEL = "inclusionai/ling-3.1-flash"


class PollinationsAdapter(BackendAdapter):
    name = "pollinations"
    title = "Pollinations"
    base_url = "https://gen.pollinations.ai/v1"
    models_path = "/models"
    chat_path = "/chat/completions"
    env_vars = ("POLLINATIONS_API_KEY",)
    auth_required = True
    public_catalog = True

    def parse_models(self, raw: Any) -> list[ModelInfo]:
        items = raw.get("data") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return []
        out: list[ModelInfo] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            if not model_id:
                continue
            if str(entry.get("category") or "text") not in {"text", ""}:
                # Keep text-capable only; image/audio/embedding ids would
                # fail a chat request and look like a provider bug.
                # Community media models sometimes lack category; require
                # an explicit text endpoint instead.
                endpoints = entry.get("supported_endpoints") or []
                if "/v1/chat/completions" not in endpoints:
                    continue
            else:
                endpoints = entry.get("supported_endpoints") or ["/v1/chat/completions"]
            params = entry.get("supported_parameters") or []
            capabilities = entry.get("capabilities") or []
            tools = bool(entry.get("tools") is True
                         or "tools" in params
                         or "tool_calling" in capabilities)
            modalities = entry.get("input_modalities") or []
            out.append(ModelInfo(
                id=model_id,
                backend=self.name,
                tool_calling=tools,
                streaming=True,  # chat endpoint streams on all text models
                reasoning=bool(entry.get("reasoning") is True
                               or "reasoning" in capabilities),
                vision="image" in modalities,
                context=_as_int(entry.get("context_length"), 128000),
                max_output=None,
                endpoints=tuple(endpoints) if isinstance(endpoints, list) else ("/v1/chat/completions",),
                free=None,  # pollen-metered; starter Pollen via Quests
                note="pollen-metered; free starter via Quests",
                raw={k: entry.get(k) for k in ("pricing", "owned_by", "title")},
            ))
        return out
