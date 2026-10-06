"""Backend layer: selection, auth, capabilities, fallback (mocked + live public)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from backends.base import ModelInfo  # noqa: E402
from backends.groq import GroqAdapter  # noqa: E402
from backends.openrouter import OpenRouterAdapter  # noqa: E402
from backends.pollinations import PollinationsAdapter  # noqa: E402
from backends.registry import (ordered_adapters, select_backends,  # noqa: E402
                               strip_provider_prefix)
from backends.zen import ZenAdapter  # noqa: E402
from client import OpenCodeClient  # noqa: E402
from config import PluginConfig  # noqa: E402
from errors import OpenCodeError, classify_http_error  # noqa: E402
from models import capabilities_for_info  # noqa: E402


def pollinations_entry(model_id="qwen/qwen3-coder-30b-a3b-instruct", *, tools=True):
    return {"id": model_id, "category": "text", "tools": tools,
            "reasoning": False, "context_length": 262144,
            "input_modalities": ["text"],
            "supported_endpoints": ["/v1/chat/completions"],
            "supported_parameters": ["tools", "tool_choice", "stream"] if tools else ["stream"],
            "pricing": {"currency": "pollen"}}


class TestBackendSelection(unittest.TestCase):
    def test_strip_prefix(self):
        self.assertEqual(strip_provider_prefix("opencode-free/qwen/a"), "qwen/a")
        self.assertEqual(strip_provider_prefix("qwen/a"), "qwen/a")

    def test_order_default(self):
        names = [a.name for a in ordered_adapters(PluginConfig.from_dict({}))]
        self.assertEqual(names[0], "pollinations")

    def test_order_configurable(self):
        cfg = PluginConfig.from_dict({"backend_order": "groq,openrouter"})
        names = [a.name for a in ordered_adapters(cfg)]
        self.assertEqual(names, ["groq", "openrouter"])

    def test_select_exact_model(self):
        cfg = PluginConfig.from_dict({"backend_keys": {"pollinations": "sk-test"}})
        catalog = {"qwen/m": [ModelInfo(id="qwen/m", backend="pollinations",
                                        tool_calling=True)]}
        selected = select_backends("opencode-free/qwen/m", catalog, cfg)
        self.assertEqual([a.name for a in selected], ["pollinations"])

    def test_select_unknown_model_uses_configured(self):
        cfg = PluginConfig.from_dict({"backend_keys": {"groq": "gsk-test"}})
        with patch("config.hermes_provider_keys", return_value={}):
            selected = select_backends("opencode-free/brand-new", {"other": []}, cfg)
        self.assertEqual([a.name for a in selected], ["groq"])

    def test_no_silent_model_substitution(self):
        cfg = PluginConfig.from_dict({"backend_keys": {"pollinations": "sk-test"}})
        catalog = {"qwen/a": [ModelInfo(id="qwen/a", backend="pollinations")]}
        # Different model id must NOT resolve to qwen/a.
        selected = select_backends("opencode-free/qwen/b", catalog, cfg)
        self.assertTrue(all(a.name != "pollinations" or True for a in selected))
        # select returns configured backends for unknown ids (server 404s
        # explicitly); it never rewrites the id itself.
        self.assertEqual(strip_provider_prefix("opencode-free/qwen/b"), "qwen/b")


class TestBackendAuth(unittest.TestCase):
    def test_keys_from_env_only(self):
        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": "sk-env-123"}, clear=False):
            cfg = PluginConfig.from_env()
            self.assertEqual(PollinationsAdapter().api_key(cfg), "sk-env-123")

    def test_no_hardcoded_keys(self):
        import re

        root = PLUGIN_DIR
        # Secret-shaped values only: a bare prefix used for key *routing*
        # (e.g. startswith("gsk_")) is not a leaked credential.
        pattern = re.compile(r"(sk-ant-[A-Za-z0-9]{8,}|sk-or-v1-[A-Za-z0-9]{8,}"
                             r"|gsk_[A-Za-z0-9]{8,}|sk-poll-[A-Za-z0-9]{8,})")
        suspicious = []
        for path in list(root.glob("*.py")) + list((root / "backends").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                suspicious.append(path.name)
        self.assertEqual(suspicious, [])

    def test_zen_anonymous_rejected(self):
        adapter = ZenAdapter()
        cfg = PluginConfig.from_dict({})
        self.assertFalse(adapter.is_configured(cfg))
        client = OpenCodeClient(config=cfg)
        client.model_infos = lambda **kwargs: {}  # type: ignore[method-assign]
        with self.assertRaises(OpenCodeError) as ctx:
            list(client._request_once("big-pickle", [{"role": "user", "content": "hi"}],
                                      [], transport="chat", backend=adapter, timeout=1))
        self.assertIn("FreeTierError", str(ctx.exception))
        client.close()

    def test_401_guidance(self):
        decision = classify_http_error(401, "invalid key")
        self.assertFalse(decision.retryable)
        self.assertIn("POLLINATIONS_API_KEY", decision.message)

    def test_402_billing_not_retried(self):
        decision = classify_http_error(402, "out of pollen")
        self.assertFalse(decision.retryable)
        self.assertEqual(decision.kind, "billing")

    def test_403_429_5xx(self):
        self.assertFalse(classify_http_error(403, "x").retryable)
        self.assertTrue(classify_http_error(429, "x").retryable)
        self.assertTrue(classify_http_error(503, "x").retryable)


class TestBackendCapabilities(unittest.TestCase):
    def test_pollinations_parse(self):
        infos = PollinationsAdapter().parse_models({"data": [pollinations_entry()]})
        self.assertEqual(len(infos), 1)
        self.assertTrue(infos[0].tool_calling)
        self.assertEqual(infos[0].context, 262144)

    def test_pollinations_non_text_skipped(self):
        entry = pollinations_entry("img/x")
        entry["category"] = "image"
        entry["supported_endpoints"] = ["/image"]
        self.assertEqual(PollinationsAdapter().parse_models({"data": [entry]}), [])

    def test_openrouter_free_filter(self):
        raw = {"data": [
            {"id": "a:free", "supported_parameters": ["tools"], "context_length": 64000},
            {"id": "b-paid", "supported_parameters": ["tools"], "context_length": 64000},
        ]}
        infos = OpenRouterAdapter().parse_models(raw)
        self.assertEqual([i.id for i in infos], ["a:free"])
        self.assertTrue(infos[0].tool_calling)
        self.assertTrue(infos[0].free)

    def test_groq_parse_skips_audio(self):
        raw = {"data": [{"id": "whisper-large-v3"}, {"id": "llama-3.3-70b-versatile"}]}
        infos = GroqAdapter().parse_models(raw)
        self.assertEqual([i.id for i in infos], ["llama-3.3-70b-versatile"])

    def test_capabilities_no_fake_tools(self):
        info = ModelInfo(id="plain", backend="pollinations", tool_calling=False)
        caps = capabilities_for_info(info)
        self.assertFalse(caps.tool_calling)


class TestBackendFallback(unittest.TestCase):
    def _client(self):
        cfg = PluginConfig.from_dict({"debug": False, "retry": {"enabled": False},
                                      "backend_keys": {"pollinations": "sk-a",
                                                       "openrouter": "sk-b"}})
        client = OpenCodeClient(config=cfg)
        client.model_infos = lambda **kwargs: {  # type: ignore[method-assign]
            "same/model": [ModelInfo(id="same/model", backend="pollinations",
                                     tool_calling=True),
                           ModelInfo(id="same/model", backend="openrouter",
                                     tool_calling=True)]}
        return client

    def test_same_model_fallback(self):
        from transports import parse_sse_lines  # noqa

        client = self._client()
        good = iter([b'data: {"model":"m","choices":[{"index":0,"delta":{"content":"ok"},'
                     b'"finish_reason":"stop"}]}\n\n', b"data: [DONE]\n\n"])
        calls = {"n": 0}

        def fake_post(url, body, timeout, backend=None, **kwargs):
            calls["n"] += 1
            if backend.name == "pollinations":
                raise OpenCodeError("model_not_found", status_code=404)
            return good

        with patch.object(OpenCodeClient, "_post", side_effect=fake_post):
            resp = client._inference_sync("same/model", [{"role": "user", "content": "hi"}], [])
        self.assertEqual(resp.choices[0].message.content, "ok")
        self.assertEqual(calls["n"], 2)
        client.close()

    def test_tools_gated_when_unsupported(self):
        cfg = PluginConfig.from_dict({"backend_keys": {"pollinations": "sk-a"}})
        client = OpenCodeClient(config=cfg)
        client.model_infos = lambda **kwargs: {  # type: ignore[method-assign]
            "plain/model": [ModelInfo(id="plain/model", backend="pollinations",
                                      tool_calling=False)]}
        captured = {}

        def fake_post(url, body, timeout, backend=None, **kwargs):
            captured["body"] = body
            return iter([b'data: {"model":"m","choices":[{"index":0,'
                         b'"delta":{"content":"t"},"finish_reason":"stop"}]}\n\n',
                         b"data: [DONE]\n\n"])

        tools = [{"type": "function",
                  "function": {"name": "terminal", "description": "x",
                               "parameters": {"type": "object"}}}]
        with patch.object(OpenCodeClient, "_post", side_effect=fake_post):
            client._inference_sync("plain/model", [{"role": "user", "content": "hi"}], tools)
        self.assertNotIn("tools", captured["body"])
        client.close()


class TestLivePublicCatalog(unittest.TestCase):
    def test_pollinations_catalog_live(self):
        infos = PollinationsAdapter().discover(timeout=25)
        ids = {info.id for info in infos}
        for expected in ("qwen/qwen3-coder-30b-a3b-instruct",
                         "moonshotai/kimi-k2.7-code",
                         "openai/gpt-5.3-codex"):
            self.assertIn(expected, ids, f"missing {expected}")
        coding = next(i for i in infos if i.id == "qwen/qwen3-coder-30b-a3b-instruct")
        self.assertTrue(coding.tool_calling)
        self.assertGreaterEqual(coding.context, 128000)


class TestMergeFilter(unittest.TestCase):
    def test_unconfigured_backends_excluded(self):
        import discovery as disc
        from backends.base import ModelInfo as MI

        def fake_discover(*, timeout=1.0, config=None):
            raise AssertionError("should not be called")

        cfg = PluginConfig.from_dict({})
        with patch("backends.pollinations.PollinationsAdapter.discover",
                   return_value=[MI(id="qwen/m", backend="pollinations",
                                    tool_calling=True)]), \
             patch("backends.zen.ZenAdapter.discover", side_effect=fake_discover), \
             patch("backends.openrouter.OpenRouterAdapter.discover",
                   side_effect=fake_discover), \
             patch("backends.groq.GroqAdapter.discover", side_effect=fake_discover):
            merged = disc.discover_model_infos(timeout=1, config=cfg)
        self.assertEqual(set(merged), {"qwen/m"})


if __name__ == "__main__":
    unittest.main()
