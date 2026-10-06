"""Errors, retry, history, protocol, debug redaction."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from debug import DebugLogger, redact  # noqa: E402
from errors import classify_http_error, compute_backoff, should_fallback_transport  # noqa: E402
from history import sanitize_messages  # noqa: E402
from protocol import Usage, message_text, normalize_finish_reason, truncate_tool_result  # noqa: E402


class TestErrors(unittest.TestCase):
    def test_429_retryable(self):
        decision = classify_http_error(429, "slow down")
        self.assertTrue(decision.retryable)
        self.assertEqual(decision.kind, "rate_limited")

    def test_401_not_retryable(self):
        self.assertFalse(classify_http_error(401, "no").retryable)

    def test_500_retryable(self):
        self.assertTrue(classify_http_error(500, "e").retryable)

    def test_400_not_retryable(self):
        self.assertFalse(classify_http_error(400, "bad").retryable)

    def test_fallback_statuses(self):
        self.assertTrue(should_fallback_transport(404, "not found"))
        self.assertTrue(should_fallback_transport(None, "model not supported, try responses"))
        self.assertFalse(should_fallback_transport(429, "rate limited"))

    def test_backoff_grows(self):
        first = compute_backoff(1, base_s=1.0, max_s=20.0)
        third = compute_backoff(3, base_s=1.0, max_s=20.0)
        self.assertGreaterEqual(third, first)
        self.assertLessEqual(first, 20.0)


class TestHistorySanitize(unittest.TestCase):
    def test_tool_result_truncated_with_marker(self):
        big = "x" * 50000
        out = sanitize_messages([{"role": "tool", "tool_call_id": "c1", "content": big}],
                                max_tool_result_chars=1000)
        self.assertLess(len(out[0]["content"]), 5000)
        self.assertIn("truncated", out[0]["content"])

    def test_drops_idless_tool_message(self):
        out = sanitize_messages([{"role": "tool", "content": "orphan"}])
        self.assertEqual(out, [])

    def test_preserves_system_verbatim(self):
        system = {"role": "system", "content": "You are Hermes, my agent."}
        out = sanitize_messages([system, {"role": "user", "content": "hi"}])
        self.assertEqual(out[0]["content"], "You are Hermes, my agent.")
        self.assertNotIn("opencode", out[0]["content"].lower())

    def test_long_result_chain(self):
        messages = [{"role": "user", "content": "go"}]
        for i in range(12):
            messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "terminal", "arguments": '{"command":"x"}'}}]})
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
        out = sanitize_messages(messages)
        tools = [m for m in out if m.get("role") == "tool"]
        self.assertEqual(len(tools), 12)
        self.assertEqual(tools[0]["tool_call_id"], "c0")


class TestProtocol(unittest.TestCase):
    def test_finish_reason_tool_calls_wins(self):
        self.assertEqual(normalize_finish_reason("stop", has_tool_calls=True), "tool_calls")

    def test_usage_from_raw(self):
        usage = Usage.from_raw({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
        self.assertEqual(usage.total_tokens, 7)

    def test_message_text(self):
        self.assertEqual(message_text("hi"), "hi")
        self.assertEqual(message_text([{"text": "a"}, {"text": "b"}]), "a\nb")

    def test_truncate(self):
        text = truncate_tool_result("x" * 100, max_chars=10)
        self.assertIn("truncated", text)


class TestDebug(unittest.TestCase):
    def test_redacts_auth(self):
        out = redact({"Authorization": "Bearer sk-123", "model": "m"})
        self.assertEqual(out["Authorization"], "***REDACTED***")
        self.assertEqual(out["model"], "m")

    def test_logger_disabled_noop(self):
        logger = DebugLogger(enabled=False)
        logger.log("MODEL", {"a": 1})
        self.assertEqual(logger.events, [])

    def test_logger_enabled_records(self):
        logger = DebugLogger(enabled=True)
        logger.log("MODEL", {"model": "m"})
        self.assertEqual(len(logger.events), 1)


if __name__ == "__main__":
    unittest.main()
