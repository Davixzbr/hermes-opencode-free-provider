"""Hermes model-provider registration for OpenCode free models."""
# ruff: noqa: E402, I001 -- Hermes imports plugin directories without a package name.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PLUGIN_DIR = Path(__file__).parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

try:
    from provider import build_profile  # type: ignore[import-not-found]
    from providers import register_provider  # type: ignore[import-not-found]

    _profile = build_profile()
    register_provider(_profile)
    opencode_free = _profile
except Exception:
    # Import must never break Hermes discovery; failures surface in `hermes plugins doctor`.
    opencode_free = None  # type: ignore[assignment]
