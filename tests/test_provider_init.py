"""Provider init, config, capabilities, tool schemas."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from config import PluginConfig, RetryConfig  # noqa: E402
from models import (ModelCapabilities, alternate_transport,  # noqa: E402
                    capabilities_for, context_window_for, detect_transport)
from tools_adapter import (ALIAS_TO_NATIVE, CANONICAL_PARAMETERS,  # noqa: E402
                           build_wire_tools)


def hermes_tool(name, params=None):
    return {"type": "function",
            "function": {"name": name, "description": f"{name} tool",
                         "parameters": params or {"type": "object"}}}


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PluginConfig.from_dict({})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.provider_name, "opencode-free")
        self.assertTrue(cfg.auto_discover)
        self.assertTrue(cfg.cache_models)
        self.assertEqual(cfg.transport, "auto")
        self.assertTrue(cfg.fallback_transport)
        self.assertEqual(cfg.reasoning, "auto")
        self.assertTrue(cfg.streaming)
        self.assertTrue(cfg.tool_translation)
        self.assertEqual(cfg.request_timeout, 180)
        self.assertTrue(cfg.retry.enabled)
        self.assertEqual(cfg.retry.max_attempts, 3)

    def test_retry_parsing(self):
        cfg = PluginConfig.from_dict({"retry": {"enabled": False, "max_attempts": 5}})
        self.assertFalse(cfg.retry.enabled)
        self.assertEqual(cfg.retry.max_attempts, 5)


class TestCapabilities(unittest.TestCase):
    def test_chat_for_free_models(self):
        for model in ("big-pickle", "mimo-v2.5-free", "ling-3.0-flash-fin-free",
                      "nemotron-3-ultra-free"):
            self.assertEqual(detect_transport(model), "chat", model)

    def test_responses_for_muse_spark(self):
        self.assertEqual(detect_transport("muse-spark-1.3-contributor-free"), "responses")
        self.assertEqual(detect_transport("gpt-5.5"), "responses")
        self.assertEqual(detect_transport("grok-4.5"), "responses")

    def test_forced_transport(self):
        self.assertEqual(detect_transport("big-pickle", configured="responses"), "responses")
        self.assertEqual(detect_transport("muse-spark-1.3-contributor-free", configured="chat"), "chat")

    def test_alternate(self):
        self.assertEqual(alternate_transport("chat"), "responses")
        self.assertEqual(alternate_transport("responses"), "chat")

    def test_capabilities_shape(self):
        caps = capabilities_for("big-pickle")
        self.assertIsInstance(caps, ModelCapabilities)
        self.assertTrue(caps.tool_calling)
        self.assertGreater(caps.max_context, 0)

    def test_context_windows(self):
        self.assertGreater(context_window_for("big-pickle"), 0)
        self.assertGreater(context_window_for("unknown-model-xyz"), 0)


class TestToolSchemas(unittest.TestCase):
    def test_all_aliases_have_canonical_schema(self):
        for alias in ALIAS_TO_NATIVE:
            self.assertIn(alias, CANONICAL_PARAMETERS, alias)
            schema = CANONICAL_PARAMETERS[alias]
            self.assertEqual(schema.get("type"), "object")
            self.assertIn("properties", schema)
            self.assertIn("required", schema)

    def test_wire_omits_unavailable(self):
        # Only terminal available -> only bash exposed, no "Never call" markers.
        wire, mapped = build_wire_tools([hermes_tool("terminal")])
        names = {t["function"]["name"] for t in wire}
        self.assertIn("bash", names)
        self.assertNotIn("edit", names)
        self.assertNotIn("read", names)
        for tool in wire:
            self.assertNotIn("Never call", tool["function"].get("description", ""))
        self.assertEqual(mapped, {"bash": "terminal"})

    def test_wire_preserves_unmapped_verbatim(self):
        native = hermes_tool("my_custom_tool", {"type": "object",
                                                "properties": {"x": {"type": "string"}}})
        wire, mapped = build_wire_tools([hermes_tool("terminal"), native])
        names = {t["function"]["name"] for t in wire}
        self.assertIn("bash", names)
        self.assertIn("my_custom_tool", names)
        custom = next(t for t in wire if t["function"]["name"] == "my_custom_tool")
        self.assertEqual(custom["function"]["parameters"]["properties"]["x"], {"type": "string"})

    def test_wire_search_files_dual_alias(self):
        wire, mapped = build_wire_tools([hermes_tool("search_files")])
        names = {t["function"]["name"] for t in wire}
        self.assertIn("glob", names)
        self.assertIn("grep", names)
        self.assertEqual(mapped.get("glob"), "search_files")
        self.assertEqual(mapped.get("grep"), "search_files")

    def test_wire_empty(self):
        wire, mapped = build_wire_tools([])
        self.assertEqual(wire, [])
        self.assertEqual(mapped, {})


if __name__ == "__main__":
    unittest.main()
