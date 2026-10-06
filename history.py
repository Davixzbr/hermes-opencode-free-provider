"""History / context handling for multi-turn agent loops.

Guarantees after `assistant -> tool_call / tool -> result` the next request
contains the exact history the wire needs:
  * tool_call IDs preserved byte-for-byte
  * tool results keep matching call IDs
  * no duplicated messages, no truncated arguments
  * large results truncated with an explicit marker (never silently)
"""
from __future__ import annotations

import json
from typing import Any

from protocol import message_text, truncate_tool_result
from tools_adapter import rewrite_history_messages


def sanitize_messages(messages: list[dict[str, Any]] | None, *,
                      mapped: dict[str, str] | None = None,
                      max_tool_result_chars: int = 40000) -> list[dict[str, Any]]:
    """Return wire-ready messages: history rewritten + results bounded.

    * Preserves Hermes system messages VERBATIM (no identity overwrite).
    * Rewrites recorded assistant tool_calls to alias vocabulary when mapped.
    * Normalises tool content to string, truncating oversized payloads.
    * Drops empty tool messages without IDs (they would 400); keeps
      everything else so the call-ID chain never breaks.
    """
    out: list[dict[str, Any]] = []
    source = list(messages or [])
    if mapped:
        source = rewrite_history_messages(source, mapped)
    for message in source:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        item = dict(message)
        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("call_id") or "")
            if not call_id:
                continue  # unmatched tool result would 400 on every wire
            item["tool_call_id"] = call_id
            content = message.get("content")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False) if content is not None else ""
                except (TypeError, ValueError):
                    content = message_text(content)
            item["content"] = truncate_tool_result(str(content or ""),
                                                   max_chars=max_tool_result_chars)
            # Strip Hermes-internal keys strict wires reject.
            for key in ("call_id", "response_item_id", "tool_name", "timestamp",
                        "platform_message_id", "api_content"):
                item.pop(key, None)
        elif role == "assistant":
            # Ensure tool_calls arguments are strings (some Hermes paths store dicts).
            calls = item.get("tool_calls")
            if isinstance(calls, list):
                fixed = []
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    call = dict(call)
                    fn = dict(call.get("function") or {})
                    args = fn.get("arguments", "{}")
                    if not isinstance(args, str):
                        try:
                            args = json.dumps(args, separators=(",", ":"), ensure_ascii=False)
                        except (TypeError, ValueError):
                            args = "{}"
                    fn["arguments"] = args
                    call["function"] = fn
                    if not call.get("id"):
                        continue  # ID-less calls break continuation
                    fixed.append(call)
                item["tool_calls"] = fixed or None
                if item["tool_calls"] is None:
                    item.pop("tool_calls", None)
        out.append(item)
    return out


def split_system(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split (system_messages, other_messages) preserving order and content."""
    system = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    rest = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]
    return system, rest


def system_text(system_messages: list[dict[str, Any]]) -> str:
    """Concatenate system contents verbatim (no added identity)."""
    parts = []
    for message in system_messages:
        text = message_text(message.get("content"))
        if text:
            parts.append(text)
    return "\n\n".join(parts)
