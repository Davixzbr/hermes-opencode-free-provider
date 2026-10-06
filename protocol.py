"""Internal normalization layer: ProviderResponse / ToolCall / Usage / Reasoning.

Avoids logic scattered across the client. Both wires (/chat/completions and
/responses) normalize into these shapes before Hermes sees them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


VALID_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "error"}


def normalize_finish_reason(value: Any, *, has_tool_calls: bool = False) -> str:
    if has_tool_calls:
        return "tool_calls"
    text = str(value or "stop").strip().lower()
    mapping = {
        "stop": "stop", "end_turn": "stop", "completed": "stop",
        "length": "length", "max_tokens": "length",
        "tool_calls": "tool_calls", "function_call": "tool_calls",
        "content_filter": "content_filter",
    }
    return mapping.get(text, "stop")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON object string
    type: str = "function"

    def to_openai_dict(self, *, index: int = 0) -> dict[str, Any]:
        return {"index": index, "id": self.id, "type": self.type,
                "function": {"name": self.name, "arguments": self.arguments}}

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(id=self.id, call_id=self.id, type=self.type,
                               function=SimpleNamespace(name=self.name, arguments=self.arguments),
                               response_item_id=None)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    @classmethod
    def from_raw(cls, raw: Any) -> "Usage":
        if not isinstance(raw, dict):
            return cls()
        prompt = int(raw.get("prompt_tokens", raw.get("input_tokens", raw.get("input", 0))) or 0)
        completion = int(raw.get("completion_tokens", raw.get("output_tokens", raw.get("output", 0))) or 0)
        total = int(raw.get("total_tokens", prompt + completion) or prompt + completion)
        details = raw.get("prompt_tokens_details") or raw.get("prompt_cache") or {}
        cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
        return cls(prompt, completion, total, cached)

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(prompt_tokens=self.prompt_tokens,
                               completion_tokens=self.completion_tokens,
                               total_tokens=self.total_tokens,
                               prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached_tokens))


@dataclass
class Reasoning:
    text: str | None = None
    details: list[Any] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.text) or bool(self.details)


@dataclass
class ProviderResponse:
    model: str
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: Reasoning = field(default_factory=Reasoning)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    raw_id: Any = None

    def to_openai_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            id=self.raw_id, object="chat.completion", model=self.model,
            choices=[SimpleNamespace(
                index=0, finish_reason=self.finish_reason,
                message=SimpleNamespace(
                    role="assistant", content=self.content,
                    tool_calls=[t.to_namespace() for t in self.tool_calls] or None,
                    reasoning=self.reasoning.text, reasoning_content=self.reasoning.text,
                    reasoning_details=self.reasoning.details or None,
                ))],
            usage=self.usage.to_namespace(),
        )


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("text") is not None:
                    parts.append(str(part["text"]))
                elif part.get("content") is not None:
                    parts.append(message_text(part["content"]))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


def truncate_tool_result(text: str, *, max_chars: int = 40000) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head - 200
    return (text[:head]
            + f"\n\n...[truncated {len(text) - max_chars} chars by oc-free-provider]...\n\n"
            + text[len(text) - tail:])
