"""Discovery + cache behaviour (mocked HTTP, temp HERMES_HOME)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import discovery  # noqa: E402


def _zen(models):
    return {"data": [{"id": m} for m in models]}


def _catalog(models):
    return {"opencode": {"models": {
        m: {"tool_call": True, "cost": {"input": 0, "output": 0, "cache": 0}}
        for m in models}}}


class TestFreeLiveModels(unittest.TestCase):
    def test_intersection(self):
        zen = _zen(["big-pickle", "paid-x"])
        catalog = _catalog(["big-pickle", "paid-x"])
        # paid-x has zero cost in this fixture so it passes; add non-zero case
        catalog["opencode"]["models"]["paid-x"]["cost"] = {"input": 1, "output": 1}
        self.assertEqual(discovery._free_live_models(zen, catalog), ["big-pickle"])

    def test_requires_tool_call(self):
        zen = _zen(["m"])
        catalog = {"opencode": {"models": {"m": {"tool_call": False,
                                                "cost": {"input": 0, "output": 0}}}}}
        self.assertEqual(discovery._free_live_models(zen, catalog), [])

    def test_deprecated_excluded(self):
        zen = _zen(["m"])
        catalog = {"opencode": {"models": {"m": {"tool_call": True, "status": "deprecated",
                                                "cost": {"input": 0, "output": 0}}}}}
        self.assertEqual(discovery._free_live_models(zen, catalog), [])

    def test_live_free_suffix_without_catalog(self):
        zen = _zen(["brand-new-free"])
        self.assertIn("brand-new-free", discovery._free_live_models(zen, {}))


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HERMES_HOME"] = self.tmp.name
        discovery.reset_snapshot()

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("HERMES_HOME", None)
        discovery.reset_snapshot()

    def test_save_load_roundtrip(self):
        discovery.save_cache(["b", "a"])
        models, _ = discovery.load_cache()
        self.assertEqual(models, ["a", "b"])

    def test_discover_uses_live_then_cache_then_fallback(self):
        from backends.base import ModelInfo

        infos = {"qwen/qwen3-coder-30b-a3b-instruct": [
            ModelInfo(id="qwen/qwen3-coder-30b-a3b-instruct", backend="pollinations",
                      tool_calling=True, context=262144)]}
        with patch.object(discovery, "discover_model_infos", return_value=infos):
            models = discovery.discover_models(timeout=1, use_cache=True, force_refresh=True)
        self.assertEqual(models, ["qwen/qwen3-coder-30b-a3b-instruct"])
        # Second call hits in-process snapshot, no HTTP.
        with patch.object(discovery, "discover_model_infos",
                          side_effect=AssertionError("should not fetch")):
            self.assertEqual(discovery.discover_models(),
                             ["qwen/qwen3-coder-30b-a3b-instruct"])

    def test_discover_falls_back_when_offline(self):
        discovery.reset_snapshot()
        with patch.object(discovery, "discover_model_infos", side_effect=OSError("down")):
            models = discovery.discover_models(timeout=1, use_cache=True, force_refresh=True)
        self.assertTrue(len(models) > 0)  # fallback bundle


if __name__ == "__main__":
    unittest.main()
