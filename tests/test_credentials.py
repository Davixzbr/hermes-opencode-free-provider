"""Credential resolution: auth_type, env, and the config.yaml bridge."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import config as config_mod  # noqa: E402
from config import PluginConfig, backend_for_key, hermes_provider_keys  # noqa: E402
import provider as provider_mod  # noqa: E402


SAMPLE = """\
model:
  default: inclusionai/ling-3.1-flash
providers:
  openrouter:
    api_key: sk-or-v1-from-config-file
  opencode-free:
    api_key: gsk_generic_from_config_file
toolsets:
  - hermes-cli
"""


class TestConfigBridge(unittest.TestCase):
    def setUp(self):
        config_mod._HERMES_KEYS_CACHE = None

    def tearDown(self):
        config_mod._HERMES_KEYS_CACHE = None

    def test_parses_providers_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(SAMPLE, encoding="utf-8")
            with patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                keys = hermes_provider_keys()
        self.assertEqual(keys.get("openrouter"), "sk-or-v1-from-config-file")
        self.assertEqual(keys.get("opencode-free"), "gsk_generic_from_config_file")
        self.assertNotIn("pollinations", keys)

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                self.assertEqual(hermes_provider_keys(), {})

    def test_key_for_prefers_explicit_then_env_then_config(self):
        cfg = PluginConfig.from_dict({"pollinations_api_key": "sk-explicit"})
        self.assertEqual(cfg.key_for("pollinations"), "sk-explicit")

        cfg = PluginConfig.from_dict({})
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_env_key_12345"}, clear=False):
            self.assertEqual(cfg.key_for("groq"), "gsk_env_key_12345")

        cfg = PluginConfig.from_dict({})
        with patch.object(config_mod, "hermes_provider_keys",
                          return_value={"openrouter": "sk-or-v1-config-file"}):
            self.assertEqual(cfg.key_for("openrouter"), "sk-or-v1-config-file")
            self.assertIsNone(cfg.key_for("pollinations"))

    def test_generic_provider_key_routed_by_shape(self):
        cfg = PluginConfig.from_dict({})
        with patch.object(config_mod, "hermes_provider_keys",
                          return_value={"opencode-free": "gsk_generic_abcdef"}):
            # A generic provider key must land on the backend that owns its shape.
            self.assertEqual(cfg.key_for("groq"), "gsk_generic_abcdef")
            self.assertIsNone(cfg.key_for("pollinations"))
            self.assertIsNone(cfg.key_for("openrouter"))

    def test_backend_for_key_shapes(self):
        self.assertEqual(backend_for_key("sk-or-v1-abc"), "openrouter")
        self.assertEqual(backend_for_key("gsk_abc"), "groq")
        self.assertEqual(backend_for_key("sk_something"), "pollinations")
        self.assertIsNone(backend_for_key(""))


class TestProfileAuth(unittest.TestCase):
    def test_auth_type_is_api_key(self):
        # external_process made Hermes treat 401s as a missing OAuth login and
        # refuse `hermes auth add`; api_key restores both paths.
        profile = provider_mod.build_profile()
        self.assertEqual(profile.auth_type, "api_key")
        self.assertIn("POLLINATIONS_API_KEY", profile.env_vars)
        self.assertTrue(profile.signup_url)

    def test_bare_api_key_routed_to_owning_backend(self):
        from client import OpenCodeClient

        cfg = PluginConfig.from_dict({})
        client = OpenCodeClient(config=cfg, api_key="sk-or-v1-routed-key")
        self.assertEqual(cfg.key_for("openrouter"), "sk-or-v1-routed-key")
        self.assertIsNone(cfg.key_for("pollinations"))
        client.close()

        cfg2 = PluginConfig.from_dict({})
        client2 = OpenCodeClient(config=cfg2, api_key="sk_pollinations_key")
        self.assertEqual(cfg2.key_for("pollinations"), "sk_pollinations_key")
        client2.close()


if __name__ == "__main__":
    unittest.main()
