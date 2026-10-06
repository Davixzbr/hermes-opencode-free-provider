"""Per-model capabilities: transport, tool calling, reasoning, context.

Centralises every `if model == ...` decision in one place so the client,
transports and discovery layers stay model-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CHAT_TRANSPORT = "chat"
RESPONSES_TRANSPORT = "responses"

# Prefixes that Zen serves on /v1/responses (per published Zen endpoint table).
# Everything else falls through to /v1/chat/completions.
# Kept as prefixes (not exact ids) so future free variants route correctly.
RESPONSES_PREFIXES = ("muse-spark", "gpt-", "grok-")
# Prefixes served on /v1/messages (Anthropic wire). Free tier currently has
# none, but the layer supports it so a future free Claude/Qwen routes sanely.
MESSAGES_PREFIXES = ("claude-", "qwen")

# Conservative context windows for known free families (tokens). Unknown
# models fall back to DEFAULT_CONTEXT. These are caps for truncation
# heuristics only — never sent as hard limits unless the caller asks.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "big-pickle": 128000,
    "mimo": 128000,
    "ling": 128000,
    "nemotron": 128000,
    "muse-spark": 200000,
    "space-bunny": 128000,
    "longcat": 128000,
    "fledge": 128000,
    "minimax": 128000,
    "glm": 128000,
    "kimi": 128000,
    "qwen": 128000,
    "deepseek": 128000,
}
DEFAULT_CONTEXT = 128000


@dataclass(frozen=True)
class ModelCapabilities:
    model: str
    transport: str = CHAT_TRANSPORT  # chat | responses
    tool_calling: bool = True
    reasoning: bool = False  # True when the wire may return reasoning blocks
    structured_output: bool = False
    max_context: int = DEFAULT_CONTEXT
    extra: dict[str, Any] = field(default_factory=dict)


def _flat(model: str) -> str:
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def detect_transport(model: str, *, configured: str = "auto") -> str:
    """Resolve the wire for *model*. `configured` may force chat/responses."""
    forced = (configured or "auto").strip().lower()
    if forced in {"chat", "chat_completions", "/chat/completions"}:
        return CHAT_TRANSPORT
    if forced in {"responses", "/responses"}:
        return RESPONSES_TRANSPORT
    flat = _flat(model)
    if flat.startswith(RESPONSES_PREFIXES):
        return RESPONSES_TRANSPORT
    # messages-wire models are served via chat-compatible gateway here;
    # the client keeps them on chat unless explicitly forced.
    return CHAT_TRANSPORT


def alternate_transport(transport: str) -> str:
    return RESPONSES_TRANSPORT if transport == CHAT_TRANSPORT else CHAT_TRANSPORT


def context_window_for(model: str) -> int:
    flat = _flat(model)
    for prefix, window in MODEL_CONTEXT_WINDOWS.items():
        if flat.startswith(prefix):
            return window
    return DEFAULT_CONTEXT


def capabilities_for(model: str, *, configured_transport: str = "auto") -> ModelCapabilities:
    flat = _flat(model)
    transport = detect_transport(model, configured=configured_transport)
    reasoning = flat.startswith(("muse-spark", "deepseek", "mimo", "qwen", "kimi", "nemotron", "ling"))
    return ModelCapabilities(
        model=flat,
        transport=transport,
        tool_calling=True,  # discovery already filters to tool-capable
        reasoning=reasoning,
        structured_output=False,
        max_context=context_window_for(model),
    )


def capabilities_for_info(info: Any, *, configured_transport: str = "auto") -> ModelCapabilities:
    """Capabilities from a backend ModelInfo (explicit, never faked)."""
    transport = detect_transport(getattr(info, "id", ""), configured=configured_transport)
    # Responses wire only where the backend advertises it; all current
    # backends serve chat, so force chat unless explicitly configured.
    if transport == RESPONSES_TRANSPORT and "/v1/responses" not in (getattr(info, "endpoints", ()) or ()):
        transport = CHAT_TRANSPORT
    return ModelCapabilities(
        model=str(getattr(info, "id", "")),
        transport=transport,
        tool_calling=bool(getattr(info, "tool_calling", False)),
        reasoning=bool(getattr(info, "reasoning", False)),
        structured_output=False,
        max_context=int(getattr(info, "context", 0) or DEFAULT_CONTEXT),
        extra={"backend": getattr(info, "backend", ""),
               "vision": bool(getattr(info, "vision", False)),
               "free": getattr(info, "free", None)},
    )
