"""Optional debug/diagnostic logging with secret redaction.

Enable with PluginConfig(debug=True) or OC_FREE_DEBUG=1.
Never logs API keys, tokens, or Authorization headers.
"""
from __future__ import annotations

import json
import time
from typing import Any


SENSITIVE_KEYS = {"authorization", "api-key", "api_key", "token", "secret", "bearer"}
SENSITIVE_SUBSTRINGS = ("sk-", "zen-", "Bearer ", "secret")


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            lowered = str(key).lower()
            if lowered in SENSITIVE_KEYS or any(s in str(key) for s in ("auth", "key")) and "cache" not in lowered:
                # Be conservative: redact anything that looks like a credential,
                # except prompt_cache_key which is a non-secret routing hash.
                if lowered == "prompt_cache_key":
                    out[key] = value
                else:
                    out[key] = "***REDACTED***"
            else:
                out[key] = redact(value)
        return out
    if isinstance(obj, list):
        return [redact(item) for item in obj]
    if isinstance(obj, str):
        redacted = obj
        for marker in SENSITIVE_SUBSTRINGS:
            if marker in redacted:
                return "***REDACTED-STRING***"
        if len(redacted) > 4000:
            return redacted[:4000] + f"...[truncated {len(redacted) - 4000} chars]"
        return redacted
    return obj


class DebugLogger:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.events: list[dict[str, Any]] = []

    def log(self, kind: str, payload: Any = None, **fields: Any) -> None:
        """Record an event. kind is one of MODEL TRANSPORT REQUEST RESPONSE
        TOOL_CALL TOOL_RESULT TOKENS LATENCY RETRY ERROR."""
        if not self.enabled:
            return
        entry = {"t": time.time(), "kind": kind}
        if payload is not None:
            entry["payload"] = redact(payload)
        for key, value in fields.items():
            entry[key] = redact(value)
        self.events.append(entry)

    def summary(self) -> str:
        if not self.enabled:
            return "debug disabled"
        lines = []
        for event in self.events:
            try:
                lines.append(f"[{event.get('kind')}] {json.dumps(event, default=str)[:1000]}")
            except Exception:
                lines.append(f"[{event.get('kind')}] <unserializable>")
        return "\n".join(lines)

    def clear(self) -> None:
        self.events.clear()
