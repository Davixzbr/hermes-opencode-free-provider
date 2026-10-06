"""Model discovery across free backends + legacy Zen helpers.

Strategy (in order):
  1. Live catalog per backend (pollinations public, openrouter public,
     groq/zen authenticated) merged to ModelInfo entries.
  2. Last successful persisted cache (TTL-guarded; stale beats nothing).
  3. Bundled curated FALLBACK_MODELS (pollinations coding models).

Legacy Zen helpers (`_free_live_models`, `_http_get_json`, `discover_models`
returning `list[str]`) are preserved for backwards compatibility: the
string list is the unified `opencode-free/<native-id>` view.

Only stdlib HTTPS (urllib) so Windows/Linux/macOS share one code path.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ZEN_MODELS_URL = "https://opencode.ai/zen/v1/models"
MODELS_DEV_URL = "https://models.dev/api.json"
HERMES_UA = "hermes-cli"
OPENCODE_UA = "opencode/1.18.31"

# Curated coding/agent fallback (pollinations native ids, tools=True live).
# Order: models verified to run on the free starter balance first.
FALLBACK_MODELS: tuple[str, ...] = (
    "inclusionai/ling-3.1-flash",
    "openai/gpt-5.4-nano",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "qwen/qwen3-coder-next",
    "moonshotai/kimi-k2.7-code",
    "openai/gpt-5.3-codex",
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.3",
    "xiaomi/mimo-v2.5",
    # Legacy Zen ids (only with authenticated Zen key):
    "big-pickle",
    "muse-spark-1.3-contributor-free",
)
DEFAULT_MODEL = "inclusionai/ling-3.1-flash"

_SNAPSHOT: tuple[str, ...] | None = None
_SNAPSHOT_TS: float = 0.0
_INFO_SNAPSHOT: dict[str, list] | None = None
_LOCK = threading.Lock()


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home  # type: ignore[import-not-found]

        return Path(get_hermes_home())
    except Exception:
        import os

        base = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        return Path(base)


def cache_path() -> Path:
    return _hermes_home() / "cache" / "oc-free-provider" / "models.json"


def _http_get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": HERMES_UA},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _free_live_models(zen: Any, catalog: Any) -> list[str]:
    """Legacy Zen intersection (kept for compat + zen adapter tests)."""
    zen_ids: set[str] = set()
    if isinstance(zen, dict):
        data = zen.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    zen_ids.add(str(item["id"]))
    elif isinstance(zen, list):
        for item in zen:
            if isinstance(item, dict) and item.get("id"):
                zen_ids.add(str(item["id"]))
            elif isinstance(item, str):
                zen_ids.add(item)

    provider = catalog.get("opencode") if isinstance(catalog, dict) else None
    metadata = provider.get("models") if isinstance(provider, dict) else None
    if not isinstance(metadata, dict):
        free_like = sorted(m for m in zen_ids if m.endswith("-free") or m in FALLBACK_MODELS or m == "big-pickle")
        return free_like

    free: list[str] = []
    for model_id, details in metadata.items():
        if not isinstance(details, dict):
            continue
        if details.get("status") == "deprecated":
            continue
        if model_id not in zen_ids:
            continue
        if details.get("tool_call") is not True:
            continue
        costs = details.get("cost")
        if not isinstance(costs, dict) or not costs:
            continue
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0
            for v in costs.values()
        ):
            free.append(str(model_id))
    for live_id in zen_ids:
        if live_id not in free and (live_id.endswith("-free") or live_id == "big-pickle"):
            free.append(live_id)
    return sorted(set(free))


def load_cache() -> tuple[list[str], float]:
    """Return (models, mtime). Empty list when no usable cache."""
    path = cache_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], 0.0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if isinstance(raw, dict):
        if raw.get("v", 1) != 2:
            return [], 0.0  # v1 (Zen-only) cache superseded by backend merge
        models = raw.get("models", [])
        saved_at = float(raw.get("saved_at", mtime) or mtime)
    elif isinstance(raw, list):
        models, saved_at = raw, mtime
    else:
        return [], 0.0
    if not isinstance(models, list) or any(not isinstance(m, str) or not m for m in models):
        return [], 0.0
    return sorted(set(models)), saved_at


def save_cache(models: list[str]) -> None:
    from backends.registry import strip_provider_prefix  # noqa: E402  (local import, no cycle)

    path = cache_path()
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"v": 2, "models": sorted(set(models)), "saved_at": time.time()}
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    _ = strip_provider_prefix


def discover_model_infos(*, timeout: float = 15.0, config: Any = None) -> dict[str, list]:
    """Full backend-merged catalog: native-id -> [ModelInfo, ...]. Never raises.

    Only backends that can actually serve are merged: configured ones
    (key present) plus pollinations, whose chat endpoint answers
    keyless attempts on some models (server decides; 401 otherwise).
    Zen without a key is excluded — its anonymous tier is blocked
    upstream and listing it would offer models that cannot run.
    """
    from backends.registry import ordered_adapters

    merged: dict[str, list] = {}
    for adapter in ordered_adapters(config):
        if adapter.name != "pollinations" and not adapter.is_configured(config):
            continue
        try:
            infos = adapter.discover(timeout=timeout, config=config)
        except Exception:
            continue
        for info in infos:
            merged.setdefault(info.id, []).append(info)
    return merged


def discover_models(*, timeout: float = 15.0, cache_ttl_s: int = 21600,
                    use_cache: bool = True, force_refresh: bool = False,
                    config: Any = None) -> list[str]:
    """Unified string catalog (`opencode-free/<native-id>` display ids).

    Prefers live backend catalogs; falls back to cache, then curated bundle.
    Never raises, never empty.
    """
    global _SNAPSHOT, _SNAPSHOT_TS, _INFO_SNAPSHOT
    with _LOCK:
        now = time.time()
        if _SNAPSHOT is not None and not force_refresh and config is None:
            return list(_SNAPSHOT)
        cached, saved_at = load_cache() if use_cache else ([], 0.0)
        fresh_cache = bool(cached) and (now - saved_at) < max(60, cache_ttl_s)
        if fresh_cache and not force_refresh:
            _SNAPSHOT, _SNAPSHOT_TS = tuple(cached), now
            return list(_SNAPSHOT)
        live: list[str] = []
        try:
            merged = discover_model_infos(timeout=timeout, config=config)
            if merged:
                _INFO_SNAPSHOT = merged
                # Prefer tool-capable models first (agent quality), then rest.
                tool_ok = sorted(mid for mid, infos in merged.items()
                                 if any(getattr(info, "tool_calling", False) for info in infos))
                rest = sorted(mid for mid in merged if mid not in tool_ok)
                live = tool_ok + rest
        except Exception:
            live = []
        if live:
            if use_cache and config is None:
                save_cache(live)
            _SNAPSHOT, _SNAPSHOT_TS = tuple(live), now
            return list(live)
        if cached:
            _SNAPSHOT, _SNAPSHOT_TS = tuple(cached), now
            return list(cached)
        _SNAPSHOT, _SNAPSHOT_TS = tuple(FALLBACK_MODELS), now
        return list(FALLBACK_MODELS)


def catalog_snapshot() -> dict[str, list]:
    return dict(_INFO_SNAPSHOT or {})


def reset_snapshot() -> None:
    global _SNAPSHOT, _SNAPSHOT_TS, _INFO_SNAPSHOT
    with _LOCK:
        _SNAPSHOT = None
        _SNAPSHOT_TS = 0.0
        _INFO_SNAPSHOT = None
