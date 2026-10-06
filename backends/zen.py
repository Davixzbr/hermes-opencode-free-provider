"""Zen backend (legacy, opt-in only).

OpenCode Zen anonymous free tier is blocked upstream
(`FreeTierError: can only be used from within OpenCode`). This adapter
exists so an AUTHENTICATED Zen key still works, but anonymous use is
never attempted — the client reports the block with setup guidance
instead of burning retries against a 403.
"""
from __future__ import annotations

from typing import Any

from backends.base import BackendAdapter, ModelInfo

ZEN_CHAT_URL = "https://opencode.ai/zen/v1/chat/completions"
ZEN_MODELS_URL = "https://opencode.ai/zen/v1/models"


class ZenAdapter(BackendAdapter):
    name = "zen"
    title = "OpenCode Zen"
    base_url = "https://opencode.ai/zen/v1"
    models_path = "/models"
    chat_path = "/chat/completions"
    env_vars = ("OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY")
    auth_required = True
    public_catalog = False

    def is_configured(self, config: Any = None) -> bool:
        # Anonymous Zen is blocked upstream; only explicit keys qualify.
        key = self.api_key(config)
        return bool(key and key.strip().lower() != "public")

    def parse_models(self, raw: Any) -> list[ModelInfo]:
        items = raw.get("data") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return []
        out: list[ModelInfo] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            if model_id:
                out.append(ModelInfo(id=model_id, backend=self.name,
                                     tool_calling=True, streaming=True,
                                     context=128000, free=None,
                                     note="Zen (authenticated)"))
        return out
