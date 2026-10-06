"""Error taxonomy, HTTP classification and retry policy.

Never mask an error as a valid response. Retry only transient failures
with exponential backoff + jitter. All decisions are pure functions so
they are easy to test on every OS.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass


class OpenCodeError(RuntimeError):
    """Fatal provider error. Never returned as assistant text."""

    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True)
class ErrorDecision:
    retryable: bool
    kind: str  # auth | not_found | rate_limited | timeout | server | bad_request | malformed | model_unavailable
    message: str


# Statuses that are always worth one more attempt (with backoff).
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# Statuses that mean "try the other transport" when fallback is enabled.
TRANSPORT_FALLBACK_STATUSES = frozenset({400, 404, 405, 422})


def classify_http_error(status: int, body: str, *, context: str = "inference") -> ErrorDecision:
    snippet = (body or "")[:800]
    # Explicit free-tier guardrail (OpenCode now rejects anonymous free-tier
    # use outside its own client). Surface actionable guidance, not a raw 403.
    if "FreeTierError" in (body or "") or "only be used from within OpenCode" in (body or ""):
        return ErrorDecision(False, "auth",
            f"OpenCode {context} refused anonymous free-tier use (HTTP {status}): "
            "OpenCode's free tier can only be used from within OpenCode. "
            "Set OPENCODE_ZEN_API_KEY (Zen API key) and retry — authenticated "
            "Zen requests work from other agents per Zen docs. "
            f"Upstream said: {snippet}")
    prefix = f"Backend {context} failed with HTTP {status}"
    full = f"{prefix}: {snippet}" if snippet else prefix
    if status == 401:
        return ErrorDecision(False, "auth",
            full + " — missing/invalid key. Set POLLINATIONS_API_KEY "
                   "(enter.pollinations.ai/keys), OPENROUTER_API_KEY, or GROQ_API_KEY. "
                   "Never hardcode keys; use env vars.")
    if status == 402:
        return ErrorDecision(False, "billing",
            full + " — budget exhausted on this backend (e.g. Pollen). "
                   "Top up, complete free Quests, or let backend fallback try the next key.")
    if status == 403:
        return ErrorDecision(False, "auth",
            full + " — forbidden (WAF, quota, or terms block). Do not retry blindly.")
    if status == 404:
        return ErrorDecision(False, "model_unavailable",
            full + " — endpoint or model not found. Try the other transport or another model.")
    if status == 429:
        return ErrorDecision(True, "rate_limited", full + " — rate limited, back off.")
    if status in RETRYABLE_STATUSES:
        return ErrorDecision(True, "server" if status >= 500 else "timeout", full)
    if status in (400, 409, 422):
        return ErrorDecision(False, "bad_request", full)
    return ErrorDecision(False, "server", full)


def should_fallback_transport(status: int | None, message: str) -> bool:
    """True when trying the alternate wire (/chat/completions <-> /responses) may help."""
    if status in TRANSPORT_FALLBACK_STATUSES:
        return True
    lowered = (message or "").lower()
    markers = (
        "not supported", "unsupported model", "no such model", "model not found",
        "responses", "chat/completions", "content_filter", "text is not set",
    )
    return any(m in lowered for m in markers)


def compute_backoff(attempt: int, *, base_s: float = 1.0, max_s: float = 20.0) -> float:
    """Exponential backoff with jitter. attempt is 1-based (first retry == 1)."""
    exp = base_s * (2.0 ** max(0, attempt - 1))
    capped = min(exp, max_s)
    return capped * (0.7 + 0.6 * random.random())


def sleep_backoff(attempt: int, *, base_s: float = 1.0, max_s: float = 20.0) -> None:
    time.sleep(compute_backoff(attempt, base_s=base_s, max_s=max_s))
