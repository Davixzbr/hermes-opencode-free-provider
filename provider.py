"""Hermes ProviderProfile for opencode-free (free backends)."""
from __future__ import annotations

import sys
from typing import Any

from config import PluginConfig
from discovery import FALLBACK_MODELS, DEFAULT_MODEL


def _load_profile_base():  # type: ignore[no-untyped-def]
    try:
        from providers.base import ProviderProfile  # type: ignore[import-not-found]
        return ProviderProfile
    except Exception:
        # Standalone/test fallback: minimal dataclass mirroring required fields.
        from dataclasses import dataclass, field

        @dataclass
        class _FallbackProfile:
            name: str
            api_mode: str = "chat_completions"
            aliases: tuple = ()
            display_name: str = ""
            description: str = ""
            env_vars: tuple = ()
            base_url: str = ""
            models_url: str = ""
            auth_type: str = "external_process"
            process_command: str = ""
            process_command_env_vars: tuple = ()
            fallback_models: tuple = ()
            default_aux_model: str = ""
            model_capabilities: dict = field(default_factory=dict)
            default_headers: dict = field(default_factory=dict)

            def fetch_models(self, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
                return None

            def create_client(self, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
                return None

        return _FallbackProfile


_ProviderProfile = _load_profile_base()


class OpenCodeFreeProfile(_ProviderProfile):  # type: ignore[valid-type,misc]
    """Native Hermes provider profile. Hermes owns the agent loop."""

    def __init__(self, config: PluginConfig | None = None, **kwargs: Any):
        self._plugin_config = config or PluginConfig.from_env()
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def create_client(self, **client_kwargs: Any) -> Any:
        from client import OpenCodeClient

        cfg = self._plugin_config
        # Merge explicit dict config if passed via client kwargs.
        extra_cfg = client_kwargs.pop("plugin_config", None)
        if isinstance(extra_cfg, dict):
            from config import PluginConfig as _PC
            cfg = _PC.from_dict({**cfg.__dict__, **extra_cfg})
        return OpenCodeClient(config=cfg, **client_kwargs)

    def fetch_models(self, *, api_key: str | None = None,
                     base_url: str | None = None, timeout: float = 15.0) -> list[str] | None:
        from client import OpenCodeClient

        client = OpenCodeClient(config=self._plugin_config, api_key=api_key,
                                base_url=base_url or self.base_url)
        try:
            models = client.list_models(timeout=timeout)
            return models or None
        except Exception:
            return None
        finally:
            try:
                client.close()
            except Exception:
                pass

    # -- hooks: preserve Hermes behaviour, minimal intervention -------------
    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Pass-through: Hermes system prompt, identity, memory and rules
        # are preserved verbatim. No injected identity.
        return messages

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        return {}

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None,
                                **context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}


def build_profile(config: PluginConfig | None = None) -> Any:
    from client import LOGICAL_BASE_URL

    cfg = config or PluginConfig.from_env()
    return OpenCodeFreeProfile(
        config=cfg,
        name=cfg.provider_name,
        aliases=("oc-free", "ocfree"),
        display_name="OpenCode Free",
        description=("Free backends for Hermes agents (Pollinations primary, "
                     "OpenRouter/Groq fallback) with Hermes-local tools"),
        api_mode="chat_completions",
        auth_type="external_process",
        env_vars=("POLLINATIONS_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"),
        base_url=LOGICAL_BASE_URL,
        process_command=sys.executable,
        process_command_env_vars=(),
        fallback_models=tuple(FALLBACK_MODELS),
        default_aux_model=DEFAULT_MODEL,
        model_capabilities={m: {"supports_tools": True} for m in FALLBACK_MODELS},
    )
