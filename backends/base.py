"""Backend adapter abstraction.

Provider -> BackendAdapter -> API -> Model.

Hermes always sees ONE provider (`opencode-free/<model>`). Each adapter
owns exactly one upstream: base URL, auth, catalog parsing and headers.
All current backends speak OpenAI-compatible `/chat/completions` + SSE,
so inference stays shared in `client.py`; adapters only differ in
discovery, auth and per-model capabilities.

Rules enforced here:
  * No hardcoded secrets. Keys come only from env/config at runtime.
  * No silent model substitution: fallback tries the SAME model string
    on the next backend; a different model is never picked quietly.
  * Capabilities are explicit: a model without tool support is reported
    as such so the client omits tools instead of faking support.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

HERMES_UA = "hermes-cli"


@dataclass
class ModelInfo:
    """One servable model on one backend."""

    id: str  # backend-native id, e.g. "qwen/qwen3-coder-30b-a3b-instruct"
    backend: str  # adapter name, e.g. "pollinations"
    tool_calling: bool = False
    streaming: bool = True
    reasoning: bool = False
    vision: bool = False
    context: int = 128000
    max_output: int | None = None
    endpoints: tuple[str, ...] = ("/v1/chat/completions",)
    free: bool | None = None  # True=$0, False=billed, None=metered/unknown
    note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendStatus:
    available: bool
    reason: str = ""
    models: int = 0


def http_get_json(url: str, timeout: float, *,
                  headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": HERMES_UA,
                 **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class BackendAdapter(ABC):
    """One upstream API. Implementations are stateless and thread-safe."""

    name: str = "base"
    title: str = "Base"
    base_url: str = ""
    models_path: str = "/models"
    chat_path: str = "/chat/completions"
    env_vars: tuple[str, ...] = ()
    auth_required: bool = True
    public_catalog: bool = False

    # -- auth ---------------------------------------------------------
    def api_key(self, config: Any = None) -> str | None:
        """Resolve key from explicit config, then env, then Hermes config.yaml.

        Never hardcoded. Delegates to PluginConfig.key_for() when available so
        every layer shares one resolution order.
        """
        import os

        key_for = getattr(config, "key_for", None)
        if callable(key_for):
            try:
                return key_for(self.name)
            except Exception:
                pass
        if config is not None:
            for attr in ("api_key", f"{self.name}_api_key", "apiKey"):
                value = getattr(config, attr, None)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            data = getattr(config, "backend_keys", None)
            if isinstance(data, dict):
                for key in (self.name, *self.env_vars):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        for var in self.env_vars:
            value = os.environ.get(var)
            if value and value.strip():
                return value.strip()
        return None

    def is_configured(self, config: Any = None) -> bool:
        if not self.auth_required:
            return True
        return bool(self.api_key(config))

    def headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json",
                   "Accept": "text/event-stream",
                   "User-Agent": HERMES_UA}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # -- urls ----------------------------------------------------------
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + self.models_path

    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + self.chat_path

    # -- catalog -------------------------------------------------------
    @abstractmethod
    def parse_models(self, raw: Any) -> list[ModelInfo]:
        """Convert raw catalog payload to ModelInfo list."""

    def discover(self, *, timeout: float = 15.0,
                 config: Any = None) -> list[ModelInfo]:
        headers = None
        key = self.api_key(config)
        if key:
            headers = {"Authorization": f"Bearer {key}"}
        raw = http_get_json(self.models_url(), timeout, headers=headers)
        return self.parse_models(raw)

    def status(self, config: Any = None) -> BackendStatus:
        if self.auth_required and not self.is_configured(config):
            return BackendStatus(False, f"missing { '/'.join(self.env_vars)}")
        return BackendStatus(True, "ok")


def _as_int(value: Any, default: int) -> int:
    try:
        number = int(value)
        return number if number > 0 else default
    except (TypeError, ValueError):
        return default
