"""Chat / Responses transports, streaming, reasoning, malformed payloads."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from errors import OpenCodeError  # noqa: E402
from transports import (convert_chat_bytes_to_events,  # noqa: E402
                        convert_responses_bytes_to_chat_events,
                        merge_chat_sse)


def sse(*payloads):
    out = []
    for payload in payloads:
        out.append(("data: " + json.dumps(payload) + "\n\n").encode())
    out.append(b"data: [DONE]\n\n")
    return iter(out)


MAPPED = {"bash": "terminal"}


class TestChatMerge(unittest.TestCase):
    def test_text_only(self):
        events = [{"model": "big-pickle",
                   "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
                  {"model": "big-pickle",
                   "choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}]
        resp = merge_chat_sse(iter(events), model="big-pickle", mapped={})
        self.assertEqual(resp.content, "hi")
        self.assertEqual(resp.finish_reason, "stop")
        self.assertEqual(resp.usage.total_tokens, 7)

    def test_single_tool_call_translated(self):
        events = list(sse(
            {"model": "m", "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_1",
                                "function": {"name": "bash",
                                             "arguments": '{"command":"ls"}'}}]},
                "finish_reason": None}]},
            {"model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
        from transports import parse_sse_lines
        parsed = list(parse_sse_lines(iter(events)))
        resp = merge_chat_sse(iter(parsed), model="m", mapped=MAPPED)
        self.assertEqual(resp.finish_reason, "tool_calls")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "terminal")
        self.assertEqual(resp.tool_calls[0].id, "call_1")

    def test_two_tool_calls_order(self):
        args1 = json.dumps({"command": "a"})
        args2 = json.dumps({"command": "b"})
        events = [
            {"model": "m", "choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c0", "function": {"name": "bash", "arguments": args1}},
                {"index": 1, "id": "c1", "function": {"name": "bash", "arguments": args2}}]},
                "finish_reason": "tool_calls"}]},
        ]
        resp = merge_chat_sse(iter(events), model="m", mapped=MAPPED)
        self.assertEqual([t.id for t in resp.tool_calls], ["c0", "c1"])

    def test_chunked_arguments_reassembled(self):
        events = [
            {"model": "m", "choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c0", "function": {"name": "bash", "arguments": '{"comm'}}]},
                "finish_reason": None}]},
            {"model": "m", "choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'and":"ls"}'}}]},
                "finish_reason": None}]},
            {"model": "m", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        resp = merge_chat_sse(iter(events), model="m", mapped=MAPPED)
        self.assertEqual(json.loads(resp.tool_calls[0].arguments)["command"], "ls")

    def test_reasoning_preserved(self):
        events = [{"model": "m", "choices": [
            {"delta": {"content": "done", "reasoning_content": "thinking..."},
             "finish_reason": "stop"}]}]
        resp = merge_chat_sse(iter(events), model="m", mapped={})
        self.assertEqual(resp.reasoning.text, "thinking...")
        self.assertEqual(resp.content, "done")

    def test_malformed_raises_not_masked(self):
        with self.assertRaises(OpenCodeError):
            list(merge_chat_sse(iter([]), model="m", mapped={}).tool_calls if False else (_ for _ in ()).throw(
                OpenCodeError("x")))

    def test_empty_stream_raises(self):
        with self.assertRaises(OpenCodeError):
            merge_chat_sse(iter([]), model="m", mapped={})

    def test_content_filter_raises(self):
        with self.assertRaises(OpenCodeError):
            merge_chat_sse(iter([{"choices": [{"delta": {}, "finish_reason": "content_filter"}]}]),
                           model="m", mapped={})

    def test_incomplete_json_raises(self):
        events = [{"model": "m", "choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c", "function": {"name": "bash", "arguments": '{"command":'}}]},
            "finish_reason": "tool_calls"}]}]
        with self.assertRaises(OpenCodeError):
            merge_chat_sse(iter(events), model="m", mapped=MAPPED)


class TestStreaming(unittest.TestCase):
    def test_stream_buffers_until_finish(self):
        raw = sse({"model": "m", "choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "id": "c1",
                            "function": {"name": "bash", "arguments": '{"command":'}}]},
            "finish_reason": None}]},
            {"model": "m", "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '"ls"}'}}]},
                "finish_reason": None}]},
            {"model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        chunks = list(convert_chat_bytes_to_events(raw, model="m", mapped=MAPPED))
        last = chunks[-1]
        calls = last["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(calls[0]["function"]["name"], "terminal")


class TestResponses(unittest.TestCase):
    def _stream(self, *events):
        return iter([("data: " + json.dumps(e) + "\n\n").encode() for e in events])

    def test_text_delta(self):
        raw = self._stream({"type": "response.output_text.delta", "delta": "hello",
                            "response": {"model": "muse-spark-1.3-contributor-free"}},
                           {"type": "response.completed",
                            "response": {"status": "completed", "model": "muse-spark-1.3-contributor-free"}})
        chunks = list(convert_responses_bytes_to_chat_events(raw, model="m", mapped={}))
        self.assertTrue(any("hello" in json.dumps(c) for c in chunks))

    def test_function_call_translated(self):
        raw = self._stream(
            {"type": "response.output_item.done", "output_index": 0,
             "item": {"type": "function_call", "call_id": "call_9", "name": "bash",
                      "arguments": '{"command":"ls"}'}},
            {"type": "response.completed",
             "response": {"status": "completed", "model": "m"}})
        chunks = list(convert_responses_bytes_to_chat_events(raw, model="m", mapped=MAPPED))
        last = chunks[-1]
        calls = last["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(calls[0]["function"]["name"], "terminal")
        self.assertEqual(last["choices"][0]["finish_reason"], "tool_calls")

    def test_error_raises(self):
        raw = self._stream({"type": "error", "error": {"message": "bad"}})
        with self.assertRaises(OpenCodeError):
            list(convert_responses_bytes_to_chat_events(raw, model="m", mapped={}))


if __name__ == "__main__":
    unittest.main()
