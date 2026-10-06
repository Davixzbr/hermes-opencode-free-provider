"""Groq backend (tertiary fallback).

https://api.groq.com/openai/v1 — OpenAI-compatible, fast. Free key at
groq.com (no card). Free tier ~30 req/min and generous daily caps,
good for agent loops. Catalog requires auth.
"""
from __future__ import annotations

from typing import Any

from backends.base import BackendAdapter, ModelInfo

# Conservative known-good tool-capable instruct models (catalog entries
# are sparse; capabilities below are floor values, not ceilings).
KNOWN_CONTEXT: dict[str, int] = {
    "llama-3.3-70b-versatile": 128000,
    "llama-3.1-8b-instant": 128000,
    "qwen/qwen3-32b": 128000,
    "openai/gpt-oss-120b": 128000,
    "openai/gpt-oss-20b": 128000,
}


class GroqAdapter(BackendAdapter):
    name = "groq"
    title = "Groq"
    base_url = "https://api.groq.com/openai/v1"
    models_path = "/models"
    chat_path = "/chat/completions"
    env_vars = ("GROQ_API_KEY",)
    auth_required = True
    public_catalog = False

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
            flat = model_id.lower()
            if any(tag in flat for tag in ("whisper", "tts", "audio", "vision-guard",
                                           "guard", "moderation")):
                continue  # non-chat endpoints would fail agent turns
            out.append(ModelInfo(
                id=model_id,
                backend=self.name,
                tool_calling=True,  # Groq instruct models support tools
                streaming=True,
                reasoning="reasoning" in flat or "deepseek" in flat,
                vision="vision" in flat,
                context=KNOWN_CONTEXT.get(model_id, KNOWN_CONTEXT.get(flat, 128000)),
                max_output=None,
                endpoints=("/v1/chat/completions",),
                free=True,
                note="Groq free tier",
            ))
        return out
