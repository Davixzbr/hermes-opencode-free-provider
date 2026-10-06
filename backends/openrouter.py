"""OpenRouter backend (secondary fallback).

https://openrouter.ai/api/v1 — OpenAI-compatible. `:free` models are $0
with no card on file; free key at openrouter.ai/keys. Tight free limits
(~20 req/min, ~50/day) so it is a fallback, not the primary agent loop.
Public catalog: GET /api/v1/models (no auth).
"""
from __future__ import annotations

from typing import Any

from backends.base import BackendAdapter, ModelInfo, _as_int


class OpenRouterAdapter(BackendAdapter):
    name = "openrouter"
    title = "OpenRouter"
    base_url = "https://openrouter.ai/api/v1"
    models_path = "/models"
    chat_path = "/chat/completions"
    env_vars = ("OPENROUTER_API_KEY",)
    auth_required = True
    public_catalog = True

    def headers(self, api_key: str | None) -> dict[str, str]:
        headers = super().headers(api_key)
        # Attribution (optional, identifies the client politely).
        headers["HTTP-Referer"] = "https://github.com/hermes-opencode-free-provider"
        headers["X-Title"] = "Hermes opencode-free provider"
        return headers

    def parse_models(self, raw: Any) -> list[ModelInfo]:
        items = raw.get("data") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return []
        out: list[ModelInfo] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            if not model_id or not model_id.endswith(":free"):
                continue
            params = entry.get("supported_parameters") or []
            top = entry.get("top_provider") or {}
            context = _as_int(entry.get("context_length",
                                        top.get("context_length")), 128000)
            out.append(ModelInfo(
                id=model_id,
                backend=self.name,
                tool_calling="tools" in params,
                streaming=True,
                reasoning="reasoning" in params,
                vision="vision" in str(entry.get("description") or "").lower()
                        or "image" in params,
                context=context,
                max_output=_as_int(entry.get("max_completion_tokens"), 0) or None,
                endpoints=("/v1/chat/completions",),
                free=True,
                note="$0 :free model; ~50 req/day",
                raw={"pricing": entry.get("pricing")},
            ))
        return out
