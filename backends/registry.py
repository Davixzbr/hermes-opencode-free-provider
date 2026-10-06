"""Backend registry: ordered selection + fallback.

Order (configurable via `backend_order`):
  1. pollinations — primary coding/agent backend, official Hermes harness
  2. openrouter  — :free fallback
  3. groq        — fast free-tier fallback
  4. zen         — only when an authenticated key exists (never anonymous)

Fallback rule: the SAME model string is tried on the next backend that
serves that exact id. A different model is never substituted silently —
if no backend serves the id, the error lists what was tried and the
closest available ids.
"""
from __future__ import annotations

from typing import Any

from backends.base import BackendAdapter, ModelInfo
from backends.groq import GroqAdapter
from backends.openrouter import OpenRouterAdapter
from backends.pollinations import PollinationsAdapter
from backends.zen import ZenAdapter

DEFAULT_ORDER: tuple[str, ...] = ("pollinations", "openrouter", "groq", "zen")


def all_adapters() -> dict[str, BackendAdapter]:
    return {
        "pollinations": PollinationsAdapter(),
        "openrouter": OpenRouterAdapter(),
        "groq": GroqAdapter(),
        "zen": ZenAdapter(),
    }


def ordered_adapters(config: Any = None) -> list[BackendAdapter]:
    order: Any = getattr(config, "backend_order", None) if config is not None else None
    if isinstance(order, str):
        order = [part.strip().lower() for part in order.split(",") if part.strip()]
    if not isinstance(order, (list, tuple)) or not order:
        order = list(DEFAULT_ORDER)
    table = all_adapters()
    adapters = [table[name] for name in order if name in table]
    return adapters or [table["pollinations"]]


def strip_provider_prefix(model: str, provider_names: tuple[str, ...] = ("opencode-free", "opencode", "oc-free")) -> str:
    text = (model or "").strip()
    if "/" in text:
        head, _, rest = text.partition("/")
        if head.strip().lower() in {name.lower() for name in provider_names}:
            return rest.strip() or text
    return text


def select_backends(model: str, catalog: dict[str, list[ModelInfo]] | None,
                    config: Any = None) -> list[BackendAdapter]:
    """Backends serving *model* exactly, in priority order.

    Backends without credentials are skipped unless their catalog is
    public AND the model was seen there. Unknown models (not in catalog)
    fall back to all configured backends so a brand-new id still routes.
    """
    native = strip_provider_prefix(model)
    table = {adapter.name: adapter for adapter in ordered_adapters(config)}
    if not catalog:
        return [adapter for adapter in table.values() if adapter.is_configured(config)]
    order = [name for name in (getattr(config, "backend_order", None) or DEFAULT_ORDER)
             if isinstance(name, str)]
    seen: list[BackendAdapter] = []
    for backend_name in order:
        adapter = table.get(backend_name)
        if adapter is None or not adapter.is_configured(config):
            continue
        infos = catalog.get(native, [])
        if any(getattr(info, "backend", None) == backend_name for info in infos):
            seen.append(adapter)
    if seen:
        return seen
    # Model unknown: try configured backends in order (may 404 -> clear error).
    return [adapter for adapter in ordered_adapters(config) if adapter.is_configured(config)]


def configured_statuses(config: Any = None) -> dict[str, str]:
    return {adapter.name: adapter.status(config).reason or "ok"
            for adapter in ordered_adapters(config)}
