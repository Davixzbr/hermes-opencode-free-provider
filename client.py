"""Free-backend inference client for Hermes.

Hermes stays the agent runtime: this client only handles discovery and
transport. No OpenCode CLI, no `opencode serve`, no ACP, no paywall bypass.

Provider -> BackendAdapter -> API -> Model. Hermes sees one provider
(`opencode-free/<model>`); the client resolves which backend serves that
exact model string and fails over across backends trying the SAME model.

Backends (ordered, configurable): pollinations -> openrouter -> groq,
plus authenticated-only zen. All speak OpenAI-compatible
`/chat/completions` + SSE, so tool translation, history, streaming and
reasoning stay shared and backend-agnostic.

Quality rules (unchanged from v4):
  * Hermes system prompt preserved verbatim (never "You are opencode").
  * Unavailable tools omitted, never "Never call" markers.
  * Tools sent only when the serving model reports tool support.
  * Streaming buffers tool JSON until valid before translating.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from config import PluginConfig
from debug import DebugLogger
from discovery import DEFAULT_MODEL, catalog_snapshot, discover_models
from errors import (OpenCodeError, classify_http_error, compute_backoff,
                    should_fallback_transport)
from history import sanitize_messages, split_system, system_text
from models import alternate_transport, capabilities_for, capabilities_for_info
from protocol import message_text
from tools_adapter import build_wire_tools, rewrite_tool_choice
from transports import (convert_chat_bytes_to_events,
                        convert_responses_bytes_to_chat_events,
                        merge_chat_sse)

# Primary backend base (display only; per-model backends resolved at runtime).
CHAT_URL = "https://gen.pollinations.ai/v1/chat/completions"
RESPONSES_URL = "https://gen.pollinations.ai/v1/responses"
LOGICAL_BASE_URL = "https://gen.pollinations.ai/v1"
USER_AGENT = "hermes-cli"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Any:  # type: ignore[override]
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _auth_headers(config: PluginConfig, backend: Any = None) -> dict[str, str]:
    """Legacy helper (kept for compat). Prefers backend adapter headers."""
    key = None
    name = getattr(backend, "name", None) if backend is not None else None
    if name and hasattr(config, "key_for"):
        try:
            key = config.key_for(name)
        except Exception:
            key = None
    if key is None:
        import os
        key = (getattr(config, "api_key", None)
               or os.environ.get("POLLINATIONS_API_KEY")
               or os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("GROQ_API_KEY")
               or os.environ.get("OPENCODE_ZEN_API_KEY"))
    headers = {"Content-Type": "application/json",
               "Accept": "text/event-stream", "User-Agent": USER_AGENT}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if name == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/hermes-opencode-free-provider"
        headers["X-Title"] = "Hermes opencode-free provider"
    return headers


def _effective_timeout(value: Any, default: float = 180.0) -> float:
    if isinstance(value, (int, float)) and value and value > 0:
        return float(value)
    for attr in ("read", "write", "connect", "pool", "timeout"):
        try:
            candidate = getattr(value, attr, None)
        except Exception:
            candidate = None
        if isinstance(candidate, (int, float)) and candidate and candidate > 0:
            return float(candidate)
    return float(default)


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


def _next_or_done(iterator: Iterator[Any]) -> tuple[bool, Any]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


class _LazyValue:
    def __init__(self, factory):  # type: ignore[no-untyped-def]
        self._factory = factory
        self._value = None
        self._ready = False
        self._lock = threading.Lock()

    def _resolve(self):  # type: ignore[no-untyped-def]
        if not self._ready:
            with self._lock:
                if not self._ready:
                    self._value = self._factory()
                    self._ready = True
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __await__(self):  # type: ignore[no-untyped-def]
        return asyncio.to_thread(self._resolve).__await__()


class _LazyStream:
    def __init__(self, factory):  # type: ignore[no-untyped-def]
        self._factory = factory
        self._iterator = None

    def __iter__(self):  # type: ignore[no-untyped-def]
        if self._iterator is None:
            self._iterator = iter(self._factory())
        return self._iterator

    def close(self) -> None:
        close = getattr(self._iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _self():  # type: ignore[no-untyped-def]
            return self
        return _self().__await__()

    def __aiter__(self):  # type: ignore[no-untyped-def]
        async def _iterate():  # type: ignore[no-untyped-def]
            iterator = iter(self)
            while True:
                ok, item = await asyncio.to_thread(_next_or_done, iterator)
                if not ok:
                    return
                yield item
        return _iterate()


def _is_model_not_found(exc: OpenCodeError) -> bool:
    if getattr(exc, "status_code", None) == 404:
        return True
    lowered = str(exc).lower()
    return any(marker in lowered for marker in (
        "model_not_found", "no endpoints found", "model not found",
        "unknown model", "not supported", "no such model"))


def _is_transport_fallbackable(exc: OpenCodeError) -> bool:
    """Transport fallback applies unless the model itself is unknown there."""
    if _is_model_not_found(exc):
        return False  # same model on next backend, not another wire here
    return should_fallback_transport(getattr(exc, "status_code", None), str(exc))


class OpenCodeClient:
    """OpenAI-compatible client over free backends, Hermes-owned loop."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 config: PluginConfig | dict[str, Any] | None = None, **_: Any):
        if isinstance(config, dict):
            config = PluginConfig.from_dict(config)
        self.config = config or PluginConfig.from_env()
        if api_key and api_key not in {"opencode-public", "public"}:
            # Back-compat: a bare key maps to the backend that owns its shape.
            from config import backend_for_key

            target = backend_for_key(api_key) or "pollinations"
            if not getattr(self.config, f"{target}_api_key", None):
                setattr(self.config, f"{target}_api_key", api_key)
        self.base_url = base_url or LOGICAL_BASE_URL
        self.debug = DebugLogger(enabled=self.config.debug)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.is_closed = False
        self._cancelled = threading.Event()

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        self.is_closed = True

    def cancel(self) -> None:
        """Interruptible-request hook (Hermes calls from interrupting thread)."""
        self._cancelled.set()

    def list_models(self, *, timeout: float = 15.0) -> list[str]:
        if not self.config.auto_discover:
            from discovery import FALLBACK_MODELS
            return list(FALLBACK_MODELS)
        return discover_models(timeout=timeout, cache_ttl_s=self.config.cache_ttl_s,
                               use_cache=self.config.cache_models,
                               config=self.config)

    def model_infos(self, *, timeout: float = 15.0) -> dict[str, list]:
        from discovery import catalog_snapshot, discover_model_infos
        infos = discover_model_infos(timeout=timeout, config=self.config)
        return infos or catalog_snapshot()

    # -- chat.completions.create ----------------------------------------
    def _create(self, *, model: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                stream: bool = False,
                tools: list[dict[str, Any]] | None = None,
                tool_choice: Any = None, timeout: Any = None,
                temperature: Any = None, max_tokens: Any = None,
                top_p: Any = None, stop: Any = None,
                extra_body: dict[str, Any] | None = None,
                **kwargs: Any) -> Any:
        seconds = _effective_timeout(timeout, default=self.config.request_timeout)
        selected = self._strip_prefix(str(model or DEFAULT_MODEL))
        use_stream = bool(self.config.streaming and stream)

        def factory() -> SimpleNamespace:
            return self._inference_sync(
                selected, messages or [], tools or [], tool_choice=tool_choice,
                timeout=seconds, temperature=temperature, max_tokens=max_tokens,
                top_p=top_p, stop=stop, extra_body=extra_body, extra=kwargs)

        if not use_stream and not stream:
            return _LazyValue(factory)

        def stream_factory() -> Iterator[SimpleNamespace]:
            for chunk in self._inference_stream(
                    selected, messages or [], tools or [], tool_choice=tool_choice,
                    timeout=seconds, temperature=temperature, max_tokens=max_tokens,
                    top_p=top_p, stop=stop, extra_body=extra_body, extra=kwargs):
                yield _namespace(chunk)

        if stream:
            return _LazyStream(stream_factory)
        return _LazyValue(factory)

    @staticmethod
    def _strip_prefix(model: str) -> str:
        from backends.registry import strip_provider_prefix
        return strip_provider_prefix(model)

    # -- inference -------------------------------------------------------
    def _inference_sync(self, model: str, messages: list[dict[str, Any]],
                        tools: list[dict[str, Any]], **opts: Any) -> SimpleNamespace:
        chunks = list(self._request_with_backends(model, messages, tools, **opts))
        events = [c for c in chunks if isinstance(c, dict)]
        merged = merge_chat_sse(iter(events), model=model, mapped=self._last_mapped)
        self.debug.log("TOKENS", {"model": model,
                                  "prompt": merged.usage.prompt_tokens,
                                  "completion": merged.usage.completion_tokens})
        return merged.to_openai_namespace()

    def _inference_stream(self, model: str, messages: list[dict[str, Any]],
                          tools: list[dict[str, Any]], **opts: Any) -> Iterator[dict[str, Any]]:
        yield from self._request_with_backends(model, messages, tools, **opts)

    _last_mapped: dict[str, str] = {}

    def _candidate_backends(self, model: str) -> list[Any]:
        from backends.registry import ordered_adapters, select_backends

        catalog = catalog_snapshot()
        if not catalog:
            try:
                catalog = self.model_infos(timeout=10.0)
            except Exception:
                catalog = {}
        self._last_catalog = catalog or {}
        # Ensure the catalog is at least attempted so unknown-model routing
        # still reaches configured backends.
        backends = select_backends(model, catalog or None, self.config)
        if backends:
            return backends
        # Test/local mode (no keys): still return primary adapter so mocked
        # _post tests exercise the full request path.
        adapters = ordered_adapters(self.config)
        if self.config.backend_fallback:
            return adapters
        return adapters[:1]

    def _request_with_backends(self, model: str, messages: list[dict[str, Any]],
                               tools: list[dict[str, Any]], **opts: Any) -> Iterator[dict[str, Any]]:
        backends = self._candidate_backends(model)
        if not backends:
            from backends.registry import configured_statuses
            raise OpenCodeError(
                "No free backend is configured. Set one of POLLINATIONS_API_KEY "
                "(enter.pollinations.ai/keys, free starter via Quests), "
                "OPENROUTER_API_KEY, or GROQ_API_KEY. "
                f"Status: {configured_statuses(self.config)}")
        attempted: list[str] = []
        last_error: Exception | None = None
        chain = backends if self.config.backend_fallback else backends[:1]
        for backend in chain:
            attempted.append(backend.name)
            try:
                yield from self._request_with_transports(model, messages, tools,
                                                         backend=backend, **opts)
                if backend.name != chain[0].name:
                    self.debug.log("RETRY", {"backend_fallback_to": backend.name,
                                             "model": model})
                return
            except OpenCodeError as exc:
                last_error = exc
                self.debug.log("ERROR", {"backend": backend.name,
                                         "error": str(exc)[:500],
                                         "status": getattr(exc, "status_code", None)})
                if _is_model_not_found(exc) and backend is not chain[-1]:
                    continue  # same model, next backend
                if getattr(exc, "status_code", None) == 402 and backend is not chain[-1]:
                    continue  # budget exhausted here; another key may serve
                raise
        if last_error is not None:
            tried = ", ".join(attempted)
            raise OpenCodeError(f"{last_error} [tried backends: {tried}]") from last_error
        raise OpenCodeError("No backend attempted the request.")

    def _request_with_transports(self, model: str, messages: list[dict[str, Any]],
                                 tools: list[dict[str, Any]], *,
                                 backend: Any, **opts: Any) -> Iterator[dict[str, Any]]:
        caps = self._capabilities_for(model, backend)
        primary = caps.transport
        transports = [primary]
        if self.config.fallback_transport:
            alt = alternate_transport(primary)
            if alt not in transports:
                transports.append(alt)
        last_error: Exception | None = None
        for attempt_transport in transports:
            try:
                yield from self._request_once(model, messages, tools,
                                              transport=attempt_transport,
                                              backend=backend, **opts)
                return
            except OpenCodeError as exc:
                last_error = exc
                self.debug.log("ERROR", {"backend": backend.name,
                                         "transport": attempt_transport,
                                         "error": str(exc)[:300],
                                         "status": getattr(exc, "status_code", None)})
                if (self.config.fallback_transport and attempt_transport == primary
                        and _is_transport_fallbackable(exc)):
                    self.debug.log("RETRY", {"fallback_to": transports[-1]})
                    continue
                raise
        if last_error is not None:
            raise last_error

    def _capabilities_for(self, model: str, backend: Any) -> Any:
        catalog = getattr(self, "_last_catalog", None) or catalog_snapshot()
        infos = catalog.get(model)
        if infos:
            for info in infos:
                if getattr(info, "backend", None) == backend.name:
                    return capabilities_for_info(info, configured_transport=self.config.transport)
            return capabilities_for_info(infos[0], configured_transport=self.config.transport)
        if backend.name == "zen":
            return capabilities_for(model, configured_transport=self.config.transport)
        # Unknown id: assume chat + tools (server validates; 404 is explicit).
        from models import ModelCapabilities, detect_transport
        return ModelCapabilities(model=model,
                                 transport=detect_transport(model, configured=self.config.transport),
                                 tool_calling=True, reasoning=False,
                                 max_context=128000,
                                 extra={"backend": backend.name})

    def _request_once(self, model: str, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]], *, transport: str,
                      backend: Any = None,
                      tool_choice: Any = None, timeout: float = 180.0,
                      temperature: Any = None, max_tokens: Any = None,
                      top_p: Any = None, stop: Any = None,
                      extra_body: dict[str, Any] | None = None,
                      extra: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        from backends.registry import ordered_adapters

        if self.is_closed:
            raise OpenCodeError("OpenCode client is closed.")
        if self._cancelled.is_set():
            raise OpenCodeError("OpenCode request was cancelled.")
        backend = backend or ordered_adapters(self.config)[0]
        if backend.name == "zen" and not backend.is_configured(self.config):
            raise OpenCodeError(
                "OpenCode Zen anonymous free tier is blocked upstream "
                "(FreeTierError: can only be used from within OpenCode). "
                "Set POLLINATIONS_API_KEY (primary, free starter via Quests), "
                "OPENROUTER_API_KEY, or GROQ_API_KEY — or OPENCODE_ZEN_API_KEY "
                "for authenticated Zen.")
        extra = extra or {}
        caps = self._capabilities_for(model, backend)
        tools = tools or []
        if tools and not caps.tool_calling:
            # Degrade cleanly: never advertise tools the model cannot use.
            self.debug.log("TRANSPORT", {"backend": backend.name, "model": model,
                                         "tools_omitted": len(tools),
                                         "reason": "model reports no tool support"})
            tools, tool_choice = [], None
        tool_translation = self.config.tool_translation
        wire_tools, mapped = build_wire_tools(tools) if tool_translation else (list(tools), {})
        self._last_mapped = mapped
        clean = sanitize_messages(messages, mapped=mapped if tool_translation else None,
                                  max_tool_result_chars=self.config.max_tool_result_chars)
        self.debug.log("MODEL", {"model": model, "backend": backend.name,
                                 "transport": transport,
                                 "tool_calling": caps.tool_calling})
        self.debug.log("TRANSPORT", {"transport": transport, "tools": len(wire_tools)})

        max_attempts = self.config.retry.max_attempts if self.config.retry.enabled else 1
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                if transport == "responses":
                    body = self._responses_body(model, clean, wire_tools, mapped, tool_choice,
                                                temperature, max_tokens, top_p, extra_body, extra)
                    raw_iter = self._post(backend.base_url.rstrip("/") + "/responses",
                                          body, timeout, backend=backend)
                    yield from convert_responses_bytes_to_chat_events(raw_iter, model=model, mapped=mapped)
                else:
                    body = self._chat_body(model, clean, wire_tools, mapped, tool_choice,
                                           temperature, max_tokens, top_p, stop, extra_body, extra)
                    raw_iter = self._post(backend.chat_url(), body, timeout, backend=backend)
                    yield from convert_chat_bytes_to_events(raw_iter, model=model, mapped=mapped)
                return
            except OpenCodeError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                retryable = bool(getattr(exc, "retryable", False))
                self.debug.log("RETRY" if retryable else "ERROR",
                               {"backend": backend.name, "attempt": attempt,
                                "max": max_attempts, "status": status,
                                "error": str(exc)[:500]})
                if retryable and attempt < max_attempts:
                    delay = compute_backoff(attempt, base_s=self.config.retry.base_delay_s,
                                            max_s=self.config.retry.max_delay_s)
                    self.debug.log("LATENCY", {"backoff_s": delay, "attempt": attempt})
                    time.sleep(delay)
                    continue
                raise
        if last_error is not None:
            raise last_error

    # -- bodies (verbatim Hermes messages; format-only adaptation) ------------
    def _chat_body(self, model: str, messages: list[dict[str, Any]],
                   wire_tools: list[dict[str, Any]], mapped: dict[str, str],
                   tool_choice: Any, temperature: Any, max_tokens: Any,
                   top_p: Any, stop: Any, extra_body: dict[str, Any] | None,
                   extra: dict[str, Any]) -> dict[str, Any]:
        # Preserve Hermes messages verbatim: NO injected identity.
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if wire_tools:
            body["tools"] = wire_tools
            if tool_choice is not None:
                body["tool_choice"] = rewrite_tool_choice(tool_choice, mapped)
            if extra.get("parallel_tool_calls") is not None:
                body["parallel_tool_calls"] = extra["parallel_tool_calls"]
        for key, value in (("temperature", temperature), ("max_tokens", max_tokens),
                           ("top_p", top_p), ("stop", stop)):
            if value is not None:
                body[key] = value
        if extra.get("response_format") is not None:
            body["response_format"] = extra["response_format"]
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if key not in {"model", "messages", "tools", "stream", "stream_options"}:
                    body[key] = value
        self.debug.log("REQUEST", {"model": model,
                                   "messages": len(messages), "tools": len(wire_tools)})
        return body

    def _responses_body(self, model: str, messages: list[dict[str, Any]],
                        wire_tools: list[dict[str, Any]], mapped: dict[str, str],
                        tool_choice: Any, temperature: Any, max_tokens: Any,
                        top_p: Any, extra_body: dict[str, Any] | None,
                        extra: dict[str, Any]) -> dict[str, Any]:
        system, rest = split_system(messages)
        instructions = system_text(system)  # verbatim, no prefix
        input_items = self._responses_input(rest, mapped)
        tools = []
        for tool in wire_tools:
            fn = tool.get("function") or {}
            tools.append({"type": "function", "name": str(fn.get("name") or ""),
                          "description": str(fn.get("description") or ""),
                          "parameters": fn.get("parameters") or {"type": "object"}})
        body: dict[str, Any] = {"model": model, "input": input_items, "stream": True, "store": False}
        if instructions:
            body["instructions"] = instructions
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                rewritten = rewrite_tool_choice(tool_choice, mapped)
                if isinstance(rewritten, dict) and rewritten.get("type") == "function":
                    fn = rewritten.get("function") or {}
                    body["tool_choice"] = {"type": "function", "name": str(fn.get("name") or "")}
                else:
                    body["tool_choice"] = rewritten
            if extra.get("parallel_tool_calls") is not None:
                body["parallel_tool_calls"] = extra["parallel_tool_calls"]
        for key, value in (("temperature", temperature), ("top_p", top_p)):
            if value is not None:
                body[key] = value
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if key not in {"model", "input", "instructions", "tools", "stream", "store"}:
                    body[key] = value
        self.debug.log("REQUEST", {"model": model,
                                   "input_items": len(input_items), "tools": len(tools)})
        return body

    def _responses_input(self, messages: list[dict[str, Any]],
                         mapped: dict[str, str]) -> list[dict[str, Any]]:
        from tools_adapter import hermes_to_opencode
        items: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "tool":
                call_id = str(message.get("tool_call_id") or message.get("call_id") or "")
                if not call_id:
                    continue
                items.append({"type": "function_call_output", "call_id": call_id,
                              "output": message_text(message.get("content"))})
                continue
            if role == "assistant":
                for detail in message.get("reasoning_details") or []:
                    if isinstance(detail, dict):
                        items.append(dict(detail))
                content = message.get("content")
                text = message_text(content)
                if text:
                    items.append({"role": "assistant",
                                  "content": [{"type": "output_text", "text": text}]})
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") or {}
                    raw_args = fn.get("arguments", "{}")
                    if not isinstance(raw_args, str):
                        raw_args = json.dumps(raw_args, separators=(",", ":"))
                    try:
                        name, args = hermes_to_opencode(str(fn.get("name") or ""), raw_args)
                    except Exception:
                        name, args = str(fn.get("name") or ""), raw_args if isinstance(raw_args, str) else "{}"
                    call_id = str(call.get("id") or call.get("call_id") or "")
                    if call_id and name:
                        items.append({"type": "function_call", "call_id": call_id,
                                      "name": name, "arguments": args})
                continue
            content = message.get("content")
            text = message_text(content)
            if text:
                items.append({"role": role if role in {"user", "assistant", "system", "developer"} else "user",
                              "content": [{"type": "input_text", "text": text}]})
        return items

    # -- HTTP -----------------------------------------------------------------
    def _post(self, url: str, body: dict[str, Any], timeout: float,
              backend: Any = None) -> Iterator[bytes]:
        from backends.registry import ordered_adapters

        backend = backend or ordered_adapters(self.config)[0]
        key = self.config.key_for(backend.name) if hasattr(self.config, "key_for") else None
        headers = backend.headers(key)
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST", headers=headers)
        started = time.time()
        try:
            response = _OPENER.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read().decode("utf-8", "replace")
            except Exception:
                payload = ""
            decision = classify_http_error(exc.code, payload)
            if exc.code == 402:
                decision = type(decision)(retryable=False, kind="billing",
                                          message=f"{decision.message} "
                                                  f"[{backend.name}: budget exhausted]")
            err = OpenCodeError(decision.message, status_code=exc.code, retryable=decision.retryable)
            raise err from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower() or "timeout" in reason.lower():
                raise OpenCodeError(f"{backend.name} request timed out after {timeout}s.",
                                    status_code=408, retryable=True) from exc
            raise OpenCodeError(f"{backend.name} request failed: {reason}",
                                status_code=None, retryable=True) from exc
        except TimeoutError as exc:
            raise OpenCodeError(f"{backend.name} request timed out after {timeout}s.",
                                status_code=408, retryable=True) from exc

        def _iter() -> Iterator[bytes]:
            try:
                with response as resp:
                    for chunk in resp:
                        if self._cancelled.is_set():
                            break
                        yield chunk
            finally:
                self.debug.log("LATENCY", {"backend": backend.name,
                                           "elapsed_s": round(time.time() - started, 3)})
        return _iter()
