"""Plugin configuration for hermes-opencode-free-provider.

Simple, explicit, cross-platform. Loaded from dict (config.yaml block),
environment variables, or defaults. No useless options.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
        return ivalue if ivalue > 0 else default
    except (TypeError, ValueError):
        return default


@dataclass
class RetryConfig:
    enabled: bool = True
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RetryConfig":
        data = data or {}
        return cls(
            enabled=_as_bool(data.get("enabled", True), True),
            max_attempts=max(1, _as_int(data.get("max_attempts", 3), 3)),
            base_delay_s=float(data.get("base_delay_s", 1.0) or 1.0),
            max_delay_s=float(data.get("max_delay_s", 20.0) or 20.0),
        )


@dataclass
class PluginConfig:
    """Runtime configuration. All fields have safe defaults."""

    enabled: bool = True
    provider_name: str = "opencode-free"
    auto_discover: bool = True
    cache_models: bool = True
    cache_ttl_s: int = 21600  # 6h
    transport: str = "auto"  # auto | chat | responses
    fallback_transport: bool = True
    reasoning: str = "auto"  # auto | include | omit
    streaming: bool = True
    tool_translation: bool = True
    request_timeout: float = 180.0
    debug: bool = False
    retry: RetryConfig = field(default_factory=RetryConfig)
    # Optional explicit API key for higher limits. Free tier works with "public".
    api_key: str | None = None
    # Max chars for a single tool result before truncation with marker.
    max_tool_result_chars: int = 40000
    # Multi-backend selection. Hermes sees one provider; backends fail over
    # internally trying the SAME model on the next backend.
    backend_order: tuple[str, ...] = ("pollinations", "openrouter", "groq", "zen")
    backend_fallback: bool = True
    pollinations_api_key: str | None = None
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    backend_keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PluginConfig":
        data = dict(data or {})
        retry = RetryConfig.from_dict(data.get("retry") if isinstance(data.get("retry"), dict) else None)
        order = data.get("backend_order", data.get("backends", ("pollinations", "openrouter", "groq", "zen")))
        if isinstance(order, str):
            order = tuple(part.strip().lower() for part in order.split(",") if part.strip())
        order = tuple(o for o in (order or ()) if isinstance(o, str)) or ("pollinations", "openrouter", "groq", "zen")
        backend_keys = data.get("backend_keys") if isinstance(data.get("backend_keys"), dict) else {}
        return cls(
            enabled=_as_bool(data.get("enabled", True), True),
            provider_name=str(data.get("provider_name", "opencode-free") or "opencode-free"),
            auto_discover=_as_bool(data.get("auto_discover", True), True),
            cache_models=_as_bool(data.get("cache_models", True), True),
            cache_ttl_s=_as_int(data.get("cache_ttl_s", 21600), 21600),
            transport=str(data.get("transport", "auto") or "auto").lower(),
            fallback_transport=_as_bool(data.get("fallback_transport", True), True),
            reasoning=str(data.get("reasoning", "auto") or "auto").lower(),
            streaming=_as_bool(data.get("streaming", True), True),
            tool_translation=_as_bool(data.get("tool_translation", True), True),
            request_timeout=float(data.get("request_timeout", 180) or 180),
            debug=_as_bool(data.get("debug", os.environ.get("OC_FREE_DEBUG", False)), False),
            retry=retry,
            api_key=data.get("api_key"),
            max_tool_result_chars=_as_int(data.get("max_tool_result_chars", 40000), 40000),
            backend_order=order,  # type: ignore[arg-type]
            backend_fallback=_as_bool(data.get("backend_fallback", data.get("fallback_backends", True)), True),
            pollinations_api_key=data.get("pollinations_api_key"),
            openrouter_api_key=data.get("openrouter_api_key"),
            groq_api_key=data.get("groq_api_key"),
            backend_keys=dict(backend_keys),
        )

    @classmethod
    def from_env(cls) -> "PluginConfig":
        """Build config from environment variables (useful for tests/CI)."""
        return cls.from_dict(
            {
                "enabled": os.environ.get("OC_FREE_ENABLED", True),
                "transport": os.environ.get("OC_FREE_TRANSPORT", "auto"),
                "fallback_transport": os.environ.get("OC_FREE_FALLBACK_TRANSPORT", True),
                "reasoning": os.environ.get("OC_FREE_REASONING", "auto"),
                "streaming": os.environ.get("OC_FREE_STREAMING", True),
                "tool_translation": os.environ.get("OC_FREE_TOOL_TRANSLATION", True),
                "request_timeout": os.environ.get("OC_FREE_TIMEOUT", 180),
                "debug": os.environ.get("OC_FREE_DEBUG", False),
                "api_key": os.environ.get("OPENCODE_ZEN_API_KEY") or os.environ.get("OPENCODE_API_KEY"),
                "backend_order": os.environ.get("OC_FREE_BACKENDS", "pollinations,openrouter,groq,zen"),
                "backend_fallback": os.environ.get("OC_FREE_BACKEND_FALLBACK", True),
                "pollinations_api_key": os.environ.get("POLLINATIONS_API_KEY"),
                "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
                "groq_api_key": os.environ.get("GROQ_API_KEY"),
                "retry": {
                    "enabled": os.environ.get("OC_FREE_RETRY", True),
                    "max_attempts": os.environ.get("OC_FREE_RETRY_ATTEMPTS", 3),
                },
            }
        )

    def key_for(self, backend: str) -> str | None:
        """Resolve a backend key: explicit field, backend_keys map, env, then
        the Hermes `providers.<name>.api_key` block in config.yaml."""
        import os as _os

        for candidate in (getattr(self, f"{backend}_api_key", None),
                          (self.backend_keys or {}).get(backend)):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        env_map = {"pollinations": ("POLLINATIONS_API_KEY",),
                   "openrouter": ("OPENROUTER_API_KEY",),
                   "groq": ("GROQ_API_KEY",),
                   "zen": ("OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY")}
        for var in env_map.get(backend, ()):
            value = _os.environ.get(var)
            if value and value.strip():
                return value.strip()
        hermes = hermes_provider_keys()
        direct = hermes.get(backend)
        if direct:
            return direct
        # `providers.opencode-free.api_key` holds whatever key the user added for
        # this provider: route it by key shape so an OpenRouter key never hits Pollinations.
        generic = hermes.get("opencode-free")
        if generic and backend_for_key(generic) == backend:
            return generic
        return None


# -- Hermes config.yaml credential bridge ------------------------------------
# The user already keeps keys in `$HERMES_HOME/config.yaml` under `providers:`
# (e.g. `providers.openrouter.api_key`). Reading them here means the plugin's
# fallback backends work with the credentials Hermes already manages, without
# asking for the same key twice. Secrets are never logged.
_KEY_ALIASES = {
    "pollinations": ("pollinations",),
    "openrouter": ("openrouter",),
    "groq": ("groq",),
    "zen": ("opencode", "opencode-zen", "zen"),
    # Our own provider entry: routed by key shape in key_for().
    "opencode-free": ("opencode-free", "oc-free"),
}
_HERMES_KEYS_CACHE: dict[str, str] | None = None


def hermes_provider_keys() -> dict[str, str]:
    """backend -> key from `$HERMES_HOME/config.yaml` `providers.<id>.api_key`.

    Minimal indentation-aware parse (stdlib only, no YAML dependency).
    Returns {} when the file is absent/unreadable. Cached per process.
    """
    global _HERMES_KEYS_CACHE
    if _HERMES_KEYS_CACHE is not None:
        return _HERMES_KEYS_CACHE
    out: dict[str, str] = {}
    try:
        import pathlib

        try:
            from hermes_constants import get_hermes_home  # type: ignore[import-not-found]

            path = pathlib.Path(get_hermes_home()) / "config.yaml"
        except Exception:
            base = os.environ.get("HERMES_HOME") or str(pathlib.Path.home() / ".hermes")
            path = pathlib.Path(base) / "config.yaml"
        in_providers = False
        current: str | None = None
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            text = line.strip()
            if indent == 0:
                in_providers = text.startswith("providers:")
                current = None
                continue
            if not in_providers:
                continue
            if indent <= 2:
                current = text.rstrip(":").strip().strip('"').strip("'") if text.endswith(":") else None
                continue
            if current and text.startswith(("api_key:", "api-key:")):
                value = text.split(":", 1)[1].strip().strip('"').strip("'")
                if value and value not in {"*", "null", "none"}:
                    for backend, names in _KEY_ALIASES.items():
                        if current.lower() in names:
                            out.setdefault(backend, value)
    except Exception:
        return {}
    _HERMES_KEYS_CACHE = out
    return out


def backend_for_key(key: str) -> str | None:
    """Route a bare credential to the backend that owns its key shape."""
    raw = (key or "").strip()
    if not raw:
        return None
    if raw.startswith("sk-or-"):
        return "openrouter"
    if raw.startswith("gsk_"):
        return "groq"
    return "pollinations"
