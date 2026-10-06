"""Mocked integration: full agent loop turns without network.

Simulates real OpenAI-compatible /chat/completions and /responses SSE
payloads across free backends, verifying multi-turn tool calling,
continuation, parallel calls, and error->retry behaviour through
OpenCodeClient with _post and catalog patched (hermetic, no network).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))
for extra in ("backends",):
    path = str(PLUGIN_DIR / extra)
    if path not in sys.path:
        sys.path.insert(0, str(PLUGIN_DIR))

from client import OpenCodeClient  # noqa: E402
from config import PluginConfig  # noqa: E402
from errors import OpenCodeError  # noqa: E402


def chat_sse_bytes(*payloads):
    chunks = []
    for payload in payloads:
        chunks.append(("data: " + json.dumps(payload) + "\n\n").encode())
    chunks.append(b"data: [DONE]\n\n")
    return iter(chunks)


def hermes_tools():
    return [
        {"type": "function", "function": {"name": "terminal", "description": "run",
                                          "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read_file", "description": "read",
                                          "parameters": {"type": "object"}}},
    ]


def tool_chunk(call_id, name, args, *, model="big-pickle", finish=None):
    return {"model": model, "choices": [{"index": 0, "delta": {
        "tool_calls": [{"index": 0, "id": call_id,
                        "function": {"name": name, "arguments": args}}]},
        "finish_reason": finish}]}


class TestChatIntegration(unittest.TestCase):
    def make_client(self, **overrides):
        cfg = PluginConfig.from_dict({"debug": False, "retry": {"enabled": False},
                                      **overrides})
        client = OpenCodeClient(config=cfg)
        # Hermetic: no live catalog fetch (would hit network).
        client.model_infos = lambda **kwargs: {}  # type: ignore[method-assign]
        return client

    def test_single_tool_call_turn(self):
        client = self.make_client()
        payload = chat_sse_bytes(
            tool_chunk("call_1", "bash", '{"command":"ls"}'),
            {"model": "big-pickle", "choices": [{"index": 0, "delta": {},
                                                 "finish_reason": "tool_calls"}]})
        with patch.object(OpenCodeClient, "_post", return_value=payload):
            resp = client._inference_sync("big-pickle",
                                          [{"role": "user", "content": "list files"}],
                                          hermes_tools())
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        call = resp.choices[0].message.tool_calls[0]
        self.assertEqual(call.function.name, "terminal")
        self.assertEqual(json.loads(call.function.arguments)["command"], "ls")
        client.close()

    def test_multi_turn_continuation(self):
        client = self.make_client()
        first = chat_sse_bytes(
            tool_chunk("c1", "read", '{"filePath":"a.txt"}'),
            {"model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        second = chat_sse_bytes(
            {"model": "m", "choices": [{"index": 0, "delta": {"content": "file says hi"},
                                        "finish_reason": None}]},
            {"model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        captured = {"n": 0}

        def fake_post(url, body, timeout, backend=None, **kwargs):
            captured["n"] += 1
            captured["body"] = body
            return first if captured["n"] == 1 else second

        with patch.object(OpenCodeClient, "_post", side_effect=fake_post):
            r1 = client._inference_sync("big-pickle", [{"role": "user", "content": "read a"}],
                                        hermes_tools())
            call_id = r1.choices[0].message.tool_calls[0].id
            history = [
                {"role": "user", "content": "read a"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": call_id, "type": "function",
                     "function": {"name": "read_file",
                                  "arguments": '{"path":"a.txt"}'}}]},
                {"role": "tool", "tool_call_id": call_id, "content": "hi"},
            ]
            r2 = client._inference_sync("big-pickle", history, hermes_tools())
        self.assertEqual(r2.choices[0].message.content, "file says hi")
        # Second request must contain the full chain with matching IDs.
        sent = captured["body"]["messages"]
        assistant = next(m for m in sent if m.get("role") == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], call_id)
        tool_msg = next(m for m in sent if m.get("role") == "tool")
        self.assertEqual(tool_msg["tool_call_id"], call_id)
        client.close()

    def test_parallel_tool_calls(self):
        client = self.make_client()
        payload = chat_sse_bytes(
            {"model": "m", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "c0",
                 "function": {"name": "bash", "arguments": '{"command":"a"}'}},
                {"index": 1, "id": "c1",
                 "function": {"name": "read", "arguments": '{"filePath":"f"}'}}]},
                "finish_reason": "tool_calls"}]})
        with patch.object(OpenCodeClient, "_post", return_value=payload):
            resp = client._inference_sync("big-pickle", [{"role": "user", "content": "go"}],
                                          hermes_tools())
        names = sorted(c.function.name for c in resp.choices[0].message.tool_calls)
        self.assertEqual(names, ["read_file", "terminal"])
        client.close()

    def test_system_prompt_preserved(self):
        client = self.make_client()
        captured = {}

        def fake_post(url, body, timeout, backend=None, **kwargs):
            captured["body"] = body
            return chat_sse_bytes({"model": "m", "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})

        system = "You are Hermes, a precise coding agent. Follow user rules."
        with patch.object(OpenCodeClient, "_post", side_effect=fake_post):
            client._inference_sync("big-pickle", [{"role": "system", "content": system},
                                                  {"role": "user", "content": "hi"}],
                                   hermes_tools())
        systems = [m for m in captured["body"]["messages"] if m.get("role") == "system"]
        self.assertEqual(len(systems), 1)
        self.assertEqual(systems[0]["content"], system)
        client.close()

    def test_retry_on_503_then_success(self):
        cfg = PluginConfig.from_dict({"retry": {"enabled": True, "max_attempts": 2}})
        client = OpenCodeClient(config=cfg)
        client.model_infos = lambda **kwargs: {}  # type: ignore[method-assign]
        good = chat_sse_bytes({"model": "m", "choices": [
            {"index": 0, "delta": {"content": "recovered"}, "finish_reason": "stop"}]})
        calls = {"n": 0}

        def fake_post(url, body, timeout, backend=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OpenCodeError("x", status_code=503, retryable=True)
            return good

        with patch.object(OpenCodeClient, "_post", side_effect=fake_post):
            with patch("client.time.sleep", return_value=None):
                chunks = list(client._request_once("big-pickle",
                                                   [{"role": "user", "content": "hi"}],
                                                   [], transport="chat", timeout=5))
        self.assertEqual(calls["n"], 2)
        self.assertTrue(chunks)
        client.close()

    def test_malformed_never_masked(self):
        client = self.make_client()
        with patch.object(OpenCodeClient, "_post",
                          return_value=iter([b"data: {broken\n\n", b"data: [DONE]\n\n"])):
            with self.assertRaises(OpenCodeError):
                client._inference_sync("big-pickle", [{"role": "user", "content": "hi"}], [])
        client.close()

    def test_responses_transport_tool_call(self):
        client = self.make_client(transport="responses")
        raw = iter([
            ("data: " + json.dumps({
                "type": "response.output_item.done", "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_r1",
                         "name": "bash", "arguments": '{"command":"ls"}'}}) + "\n\n").encode(),
            ("data: " + json.dumps({
                "type": "response.completed",
                "response": {"status": "completed", "model": "muse-spark-1.3-contributor-free"}}) + "\n\n").encode(),
        ])
        with patch.object(OpenCodeClient, "_post", return_value=raw):
            resp = client._inference_sync("muse-spark-1.3-contributor-free",
                                          [{"role": "user", "content": "ls"}],
                                          hermes_tools())
        self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "terminal")
        client.close()

    def test_no_backend_configured_error(self):
        client = self.make_client(backend_fallback=False)
        client._candidate_backends = lambda model: []  # type: ignore[method-assign]
        with self.assertRaises(OpenCodeError) as ctx:
            client._inference_sync("some-model", [{"role": "user", "content": "hi"}], [])
        self.assertIn("POLLINATIONS_API_KEY", str(ctx.exception))
        client.close()


if __name__ == "__main__":
    unittest.main()
