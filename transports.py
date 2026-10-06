"""Wire transports: SSE parsing, chat merging, Responses conversion.

Supports when appropriate:
  /chat/completions  (OpenAI-compatible SSE)
  /responses         (OpenAI Responses SSE -> normalized chat SSE)

Processes text / reasoning / tool calls / arguments / usage /
finish reason separately. Tool arguments accumulate until JSON-complete
before translation (never emits partial invalid calls).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from errors import OpenCodeError
from protocol import (ProviderResponse, Reasoning, ToolCall, Usage,
                      message_text, normalize_finish_reason)
from tools_adapter import ToolTranslationError, is_valid_tool_arguments, opencode_to_hermes


def parse_sse_lines(raw: Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """Parse text/event-stream bytes into JSON payload dicts."""
    for raw_line in raw:
        try:
            line = raw_line.decode("utf-8", "replace").strip()
        except Exception:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenCodeError(f"OpenCode returned invalid SSE JSON: {exc}") from exc
        if isinstance(data, dict):
            yield data


def _translate_parts(parts: dict[int, dict[str, Any]], mapped: dict[str, str]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(parts):
        part = parts[index]
        name = str(part.get("name") or "")
        raw_args = str(part.get("arguments") or "{}")
        # Never translate partial JSON: if incomplete, keep raw and let the
        # caller decide (merge validates; stream defers to finish event).
        if raw_args.strip() and not is_valid_tool_arguments(raw_args):
            # Still attempt: a truncated stream at finish is an error, not a call.
            raise OpenCodeError(
                f"OpenCode tool '{name or '?'}' returned incomplete JSON arguments; "
                "buffered stream ended before the object closed.")
        try:
            native_name, native_args = opencode_to_hermes(name, raw_args)
        except ToolTranslationError as exc:
            raise OpenCodeError(str(exc)) from exc
        # Fail-closed on compatibility aliases without a Hermes target is
        # handled by opencode_to_hermes raising; unknown native tools pass.
        calls.append(ToolCall(id=str(part.get("id") or f"call_{index}"),
                              name=native_name, arguments=native_args))
    return calls


def merge_chat_sse(events: Iterator[dict[str, Any]], *, model: str,
                   mapped: dict[str, str]) -> ProviderResponse:
    content: list[str] = []
    reasoning_text: list[str] = []
    reasoning_details: list[Any] = []
    parts: dict[int, dict[str, Any]] = {}
    finish = "stop"
    usage_raw: Any = None
    seen = False
    active_model = model
    for event in events:
        seen = True
        if event.get("error"):
            raise OpenCodeError(f"OpenCode inference error: {event['error']}")
        active_model = str(event.get("model") or active_model)
        if isinstance(event.get("usage"), dict):
            usage_raw = event["usage"]
        for choice in event.get("choices") or []:
            if choice.get("finish_reason"):
                finish = str(choice["finish_reason"])
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("content") is not None:
                content.append(str(delta["content"]))
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key) is not None:
                    reasoning_text.append(str(delta[key]))
                    break
            details = delta.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(details)
            elif details is not None:
                reasoning_details.append(details)
            for raw_call in delta.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                index = int(raw_call.get("index", 0) or 0)
                slot = parts.setdefault(index, {"id": "", "type": "function", "name": "", "arguments": ""})
                if raw_call.get("id"):
                    slot["id"] = str(raw_call["id"])
                slot["type"] = str(raw_call.get("type") or slot["type"])
                fn = raw_call.get("function") or {}
                if isinstance(fn, dict):
                    if fn.get("name"):
                        # A new name on an existing index means a new call.
                        if slot["name"] and slot["name"] != str(fn["name"]) and slot["arguments"]:
                            index = max(parts.keys(), default=-1) + 1
                            slot = parts.setdefault(index, {"id": str(raw_call.get("id") or f"call_{index}"),
                                                            "type": "function", "name": "", "arguments": ""})
                        slot["name"] = str(fn["name"])
                    slot["arguments"] += str(fn.get("arguments") or "")
    if not seen:
        raise OpenCodeError("OpenCode inference returned no events.")
    if finish == "content_filter":
        raise OpenCodeError("OpenCode inference was blocked by the content filter.")
    tool_calls = _translate_parts(parts, mapped) if parts else []
    text = "".join(content) or None
    thought = "".join(reasoning_text) or None
    return ProviderResponse(
        model=active_model, content=text, tool_calls=tool_calls,
        reasoning=Reasoning(text=thought, details=reasoning_details),
        usage=Usage.from_raw(usage_raw),
        finish_reason=normalize_finish_reason(finish, has_tool_calls=bool(tool_calls)),
    )


def convert_chat_bytes_to_events(raw: Iterator[bytes], *, model: str,
                                 mapped: dict[str, str]) -> Iterator[dict[str, Any]]:
    """Yield OpenAI-style chunk dicts with Hermes-native tool calls at finish.

    Buffers tool-call deltas per (choice,index); emits translated calls only
    on finish (or as a terminal synthetic chunk when the stream ends open).
    """
    buffered: dict[tuple[int, int], dict[str, Any]] = {}
    active_model = model
    seen = False
    for event in parse_sse_lines(raw):
        seen = True
        if event.get("error"):
            raise OpenCodeError(f"OpenCode inference error: {event['error']}")
        active_model = str(event.get("model") or active_model)
        event.setdefault("model", active_model)
        for choice in event.get("choices") or []:
            if choice.get("finish_reason") == "content_filter":
                raise OpenCodeError("OpenCode inference was blocked by the content filter.")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
                choice["delta"] = delta
            raw_calls = delta.pop("tool_calls", [])
            choice_index = int(choice.get("index", 0) or 0)
            for raw_call in raw_calls or []:
                if not isinstance(raw_call, dict):
                    continue
                index = int(raw_call.get("index", 0) or 0)
                key = (choice_index, index)
                slot = buffered.setdefault(key, {"id": "", "type": "function", "name": "", "arguments": ""})
                if raw_call.get("id"):
                    slot["id"] = str(raw_call["id"])
                fn = raw_call.get("function") or {}
                if isinstance(fn, dict):
                    if fn.get("name"):
                        slot["name"] = str(fn["name"])
                    slot["arguments"] += str(fn.get("arguments") or "")
            if choice.get("finish_reason") and any(k[0] == choice_index for k in buffered):
                # Translate this choice's buffered calls now (validates JSON).
                parts = {idx: slot for (ch, idx), slot in sorted(buffered.items()) if ch == choice_index}
                translated = _translate_parts(parts, mapped)
                delta["tool_calls"] = [t.to_openai_dict(index=i) for i, t in enumerate(translated)]
                for key in [k for k in buffered if k[0] == choice_index]:
                    del buffered[key]
        yield event
    if not seen:
        raise OpenCodeError("OpenCode inference returned no events.")
    if buffered:
        # Stream ended without finish_reason but with buffered calls (some
        # relays do this): emit a terminal tool_calls chunk.
        by_choice: dict[int, dict[int, dict[str, Any]]] = {}
        for (ch, idx), slot in buffered.items():
            by_choice.setdefault(ch, {})[idx] = slot
        yield {"model": active_model, "choices": [
            {"index": ch, "delta": {"tool_calls": [
                t.to_openai_dict(index=i) for i, t in enumerate(_translate_parts(slots, mapped))]},
             "finish_reason": "tool_calls"}
            for ch, slots in sorted(by_choice.items())]}


# ---------------------------------------------------------------------------
# Responses SSE -> chat SSE
# ---------------------------------------------------------------------------
def convert_responses_bytes_to_chat_events(raw: Iterator[bytes], *, model: str,
                                           mapped: dict[str, str]) -> Iterator[dict[str, Any]]:
    """Convert Responses stream into chat-completion chunks (Hermes-native calls)."""
    call_ids: dict[int, int] = {}
    call_slots: dict[int, dict[str, Any]] = {}
    active_model = model
    saw_text = False
    for event in parse_sse_lines(raw):
        kind = str(event.get("type") or "")
        response = event.get("response") or {}
        if isinstance(response, dict) and response.get("model"):
            active_model = str(response["model"])
        if kind == "error" or event.get("error"):
            raise OpenCodeError(f"OpenCode inference error: {event}")
        if kind == "response.output_text.delta":
            saw_text = True
            yield {"model": active_model, "choices": [
                {"index": 0, "delta": {"content": str(event.get("delta") or "")},
                 "finish_reason": None}]}
        elif kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            yield {"model": active_model, "choices": [
                {"index": 0, "delta": {"reasoning_content": str(event.get("delta") or "")},
                 "finish_reason": None}]}
        elif kind == "response.output_item.done":
            item = event.get("item") or {}
            if not isinstance(item, dict):
                continue
            if item.get("type") == "reasoning":
                yield {"model": active_model, "choices": [
                    {"index": 0, "delta": {"reasoning_details": [item]}, "finish_reason": None}]}
            elif item.get("type") == "function_call":
                output_index = int(event.get("output_index", 0) or 0)
                index = call_ids.setdefault(output_index, len(call_ids))
                slot = call_slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
                slot["id"] = str(item.get("call_id") or item.get("id") or f"call_{index}")
                slot["name"] = str(item.get("name") or "")
                slot["arguments"] = str(item.get("arguments") or "{}")
                # Validate now so malformed calls fail fast with context.
                if slot["arguments"] and not is_valid_tool_arguments(slot["arguments"]):
                    raise OpenCodeError(f"OpenCode tool '{slot['name']}' returned invalid JSON arguments.")
        elif kind == "response.completed":
            resp = event.get("response") or {}
            usage_raw = resp.get("usage") if isinstance(resp, dict) else None
            if call_slots:
                parts = {i: {"id": s["id"], "type": "function", "name": s["name"], "arguments": s["arguments"]}
                         for i, s in sorted(call_slots.items())}
                translated = _translate_parts(parts, mapped)
                yield {"model": active_model, "choices": [
                    {"index": 0, "delta": {"tool_calls": [
                        t.to_openai_dict(index=i) for i, t in enumerate(translated)]},
                     "finish_reason": "tool_calls"}],
                    "usage": usage_raw}
            else:
                status = str((resp.get("status") or "") if isinstance(resp, dict) else "")
                finish = "stop" if status in {"", "completed"} else "stop"
                yield {"model": active_model, "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": usage_raw}
            return
        elif kind == "response.failed":
            raise OpenCodeError(f"OpenCode Responses request failed: {event}")
    # Non-SSE fallback: some relays return a single JSON Responses object.
    _ = saw_text  # text-only streams without completed event still need a stop
