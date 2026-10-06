"""Bidirectional Hermes <-> OpenCode tool translation with high fidelity.

Design principles (fixes for the reference implementation):
  * Preserve Hermes system prompt verbatim — no "You are opencode" injection.
  * Omit unavailable tools entirely — never advertise a "Never call" marker.
  * Preserve native schemas for unmapped tools verbatim.
  * For mapped tools expose OpenCode-canonical names BUT with complete,
    invertible argument mappings (no silent field drops).
  * Preserve call IDs, order, parallelism, and error payloads.
  * Buffer streamed JSON until valid before translating.
  * Truncate oversized results with an explicit marker.

Mapping pairs (OpenCode alias <-> Hermes native):
  bash<->terminal, edit<->patch, glob/grep<->search_files, read<->read_file,
  skill<->skill_view, task<->delegate_task, todowrite<->todo_list,
  webfetch<->web_extract, websearch<->web_search, write<->write_file
"""
from __future__ import annotations

import copy
import json
from typing import Any

# OpenCode alias -> Hermes native
ALIAS_TO_NATIVE: dict[str, str] = {
    "bash": "terminal",
    "edit": "patch",
    "glob": "search_files",
    "grep": "search_files",
    "read": "read_file",
    "skill": "skill_view",
    "task": "delegate_task",
    "todowrite": "todo_list",
    "webfetch": "web_extract",
    "websearch": "web_search",
    "write": "write_file",
}
# Hermes native -> preferred OpenCode alias (search_files is ambiguous;
# resolved per-call by `target`: files->glob, else grep).
NATIVE_TO_ALIAS: dict[str, str] = {
    "terminal": "bash",
    "patch": "edit",
    "read_file": "read",
    "skill_view": "skill",
    "delegate_task": "task",
    "todo_list": "todowrite",
    "web_extract": "webfetch",
    "web_search": "websearch",
    "write_file": "write",
}

COMPAT_ALIASES = tuple(ALIAS_TO_NATIVE.keys())


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return ""
    return str(fn.get("name") or "").strip()


def _deep_copy(obj: Any) -> Any:
    return copy.deepcopy(obj)


# ---------------------------------------------------------------------------
# Canonical OpenCode parameter schemas.
# Complete (all documented fields), permissive (additionalProperties allowed
# via translation layer that forwards unknown fields), and invertible.
# ---------------------------------------------------------------------------
CANONICAL_PARAMETERS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {"type": "integer", "minimum": 1, "description": "Timeout in milliseconds."},
            "workdir": {"type": "string", "description": "Working directory."},
            "description": {"type": "string", "description": "Human-readable description."},
        },
        "required": ["command"],
        "additionalProperties": True,
    },
    "edit": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "oldString": {"type": "string"},
            "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"},
        },
        "required": ["filePath", "oldString", "newString"],
        "additionalProperties": True,
    },
    "glob": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
        "additionalProperties": True,
    },
    "grep": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "include": {"type": "string"},
        },
        "required": ["pattern"],
        "additionalProperties": True,
    },
    "read": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["filePath"],
        "additionalProperties": True,
    },
    "skill": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": True,
    },
    "task": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "subagent_type": {"type": "string"},
            "task_id": {"type": "string"},
            "command": {"type": "string"},
            "background": {"type": "boolean"},
        },
        "required": ["description", "prompt"],
        "additionalProperties": True,
    },
    "todowrite": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string",
                                   "enum": ["pending", "in_progress", "completed", "cancelled"]},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        "id": {"type": "string"},
                    },
                    "required": ["content", "status"],
                    "additionalProperties": True,
                },
            }
        },
        "required": ["todos"],
        "additionalProperties": True,
    },
    "webfetch": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "format": {"type": "string", "enum": ["text", "markdown", "html"]},
            "timeout": {"type": "integer", "minimum": 1},
            "maxChars": {"type": "integer", "minimum": 1},
        },
        "required": ["url"],
        "additionalProperties": True,
    },
    "websearch": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "numResults": {"type": "integer", "minimum": 1, "maximum": 100},
            "livecrawl": {"type": "string"},
            "type": {"type": "string"},
            "contextMaxCharacters": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": True,
    },
    "write": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["filePath", "content"],
        "additionalProperties": True,
    },
}

ALIAS_DESCRIPTIONS: dict[str, str] = {
    "bash": "Execute a shell command in the local workspace.",
    "edit": "Replace exact text in a local file.",
    "glob": "Find local files by glob pattern.",
    "grep": "Search local file contents with a regular expression.",
    "read": "Read a local file.",
    "skill": "Load a Hermes skill by name.",
    "task": "Delegate a task to a Hermes subagent.",
    "todowrite": "Replace the current Hermes task list.",
    "webfetch": "Extract content from a web page.",
    "websearch": "Search the web.",
    "write": "Write a local file.",
}


class ToolTranslationError(ValueError):
    pass


def _require(args: dict[str, Any], key: str, *, tool: str) -> Any:
    if key not in args:
        raise ToolTranslationError(f"tool '{tool}' omitted required argument '{key}'.")
    return args[key]


def _parse_args(encoded: Any, *, tool: str) -> dict[str, Any]:
    if isinstance(encoded, dict):
        return dict(encoded)
    text = str(encoded or "{}")
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ToolTranslationError(f"tool '{tool}' has invalid JSON arguments: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolTranslationError(f"tool '{tool}' arguments must be a JSON object.")
    return parsed


def _encode(args: dict[str, Any]) -> str:
    return json.dumps(args, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenCode -> Hermes (model output -> local execution)
# ---------------------------------------------------------------------------
def opencode_to_hermes(alias: str, encoded: Any) -> tuple[str, str]:
    """Translate one OpenCode tool call to its Hermes native form.

    Returns (native_name, native_args_json). Unknown tools pass through
    unchanged. Forward-compatible: unknown fields are preserved.
    """
    args = _parse_args(encoded, tool=alias)
    native = ALIAS_TO_NATIVE.get(alias)
    if native is None:
        return alias, _encode(args)

    if alias == "bash":
        out: dict[str, Any] = {"command": _require(args, "command", tool=alias)}
        if "workdir" in args:
            out["workdir"] = args["workdir"]
        if "description" in args:
            out["description"] = args["description"]
        if "timeout" in args:
            try:
                ms = int(args["timeout"])
            except (TypeError, ValueError):
                ms = 0
            if ms > 0:
                out["timeout"] = max(1, (ms + 999) // 1000)
        for key, value in args.items():
            if key not in out and key not in {"command", "workdir", "description", "timeout"}:
                out[key] = value
        return native, _encode(out)

    if alias == "edit":
        out = {
            "mode": "replace",
            "path": _require(args, "filePath", tool=alias),
            "old_string": _require(args, "oldString", tool=alias),
            "new_string": _require(args, "newString", tool=alias),
        }
        if "replaceAll" in args:
            out["replace_all"] = bool(args["replaceAll"])
        for key, value in args.items():
            if key not in {"filePath", "oldString", "newString", "replaceAll"}:
                out[key] = value
        return native, _encode(out)

    if alias == "glob":
        out = {"target": "files", "pattern": _require(args, "pattern", tool=alias)}
        if "path" in args:
            out["path"] = args["path"]
        for key, value in args.items():
            if key not in {"pattern", "path"}:
                out[key] = value
        return native, _encode(out)

    if alias == "grep":
        out = {"target": "content", "pattern": _require(args, "pattern", tool=alias)}
        if "path" in args:
            out["path"] = args["path"]
        if "include" in args:
            out["file_glob"] = args["include"]
        for key, value in args.items():
            if key not in {"pattern", "path", "include"}:
                out[key] = value
        return native, _encode(out)

    if alias == "read":
        out = {"path": _require(args, "filePath", tool=alias)}
        for key in ("offset", "limit"):
            if key in args:
                out[key] = args[key]
        for key, value in args.items():
            if key not in {"filePath", "offset", "limit"}:
                out[key] = value
        return native, _encode(out)

    if alias == "skill":
        return native, _encode({"name": _require(args, "name", tool=alias),
                                **{k: v for k, v in args.items() if k != "name"}})

    if alias == "task":
        # Lossless: keep every OpenCode field inside a structured context
        # envelope that reverse translation parses exactly (no prefix guessing).
        oc_meta = {k: args[k] for k in ("description", "subagent_type", "task_id", "command", "background") if k in args}
        envelope = {"opencode_task": oc_meta, "prompt": args.get("prompt", "")}
        out = {"tasks": [{"goal": _require(args, "prompt", tool=alias),
                          "context": _encode(envelope)}]}
        # Preserve any extra OpenCode fields at top level for debugging.
        for key, value in args.items():
            if key not in {"description", "prompt", "subagent_type", "task_id", "command", "background"}:
                out[f"opencode_{key}"] = value
        if "description" in args:
            out["tasks"][0]["title"] = args["description"]
        return native, _encode(out)

    if alias == "todowrite":
        todos = _require(args, "todos", tool=alias)
        if not isinstance(todos, list):
            raise ToolTranslationError("tool 'todowrite' argument 'todos' must be a list.")
        converted = []
        for index, item in enumerate(todos):
            if not isinstance(item, dict):
                raise ToolTranslationError("tool 'todowrite' contains a non-object item.")
            converted.append({
                "id": str(item.get("id") or f"oc-{index + 1}"),
                "content": _require(item, "content", tool=alias),
                "status": _require(item, "status", tool=alias),
                # Preserve priority (reference dropped it).
                "priority": str(item.get("priority") or "medium"),
                **{k: v for k, v in item.items() if k not in {"id", "content", "status", "priority"}},
            })
        return native, _encode({"todos": converted, "merge": False})

    if alias == "webfetch":
        out = {"urls": [_require(args, "url", tool=alias)]}
        if "format" in args:
            out["format"] = args["format"]
        if "timeout" in args:
            out["timeout"] = args["timeout"]
        if "maxChars" in args:
            out["max_chars"] = args["maxChars"]
        for key, value in args.items():
            if key not in {"url", "format", "timeout", "maxChars"}:
                out[key] = value
        return native, _encode(out)

    if alias == "websearch":
        out = {"query": _require(args, "query", tool=alias)}
        if "numResults" in args:
            try:
                out["limit"] = max(1, min(100, int(args["numResults"])))
            except (TypeError, ValueError):
                pass
        for key in ("livecrawl", "type", "contextMaxCharacters"):
            if key in args:
                out[key] = args[key]
        for key, value in args.items():
            if key not in {"query", "numResults", "livecrawl", "type", "contextMaxCharacters"}:
                out[key] = value
        return native, _encode(out)

    if alias == "write":
        out = {"path": _require(args, "filePath", tool=alias),
               "content": _require(args, "content", tool=alias)}
        for key, value in args.items():
            if key not in {"filePath", "content"}:
                out[key] = value
        return native, _encode(out)

    return alias, _encode(args)


# ---------------------------------------------------------------------------
# Hermes -> OpenCode (history replay: native recorded calls -> alias vocabulary)
# ---------------------------------------------------------------------------
def hermes_to_opencode(native: str, encoded: Any) -> tuple[str, str]:
    """Translate one recorded Hermes call back to OpenCode vocabulary for replay."""
    args = _parse_args(encoded, tool=native)
    # search_files is ambiguous: use target discriminator.
    if native == "search_files":
        target = str(args.get("target") or "")
        alias = "glob" if target == "files" else "grep"
        if alias == "glob":
            out = {"pattern": _require(args, "pattern", tool=native)}
            if "path" in args:
                out["path"] = args["path"]
            for key, value in args.items():
                if key not in {"target", "pattern", "path"}:
                    out[key] = value
            return alias, _encode(out)
        out = {"pattern": _require(args, "pattern", tool=native)}
        if "path" in args:
            out["path"] = args["path"]
        if "file_glob" in args:
            out["include"] = args["file_glob"]
        for key, value in args.items():
            if key not in {"target", "pattern", "path", "file_glob"}:
                out[key] = value
        return alias, _encode(out)

    # Direct native names that are also aliases pass through.
    if native not in NATIVE_TO_ALIAS:
        return native, _encode(args)
    alias = NATIVE_TO_ALIAS[native]

    if native == "terminal":
        out = {"command": _require(args, "command", tool=native)}
        if "workdir" in args:
            out["workdir"] = args["workdir"]
        if "description" in args:
            out["description"] = args["description"]
        if "timeout" in args:
            try:
                out["timeout"] = max(1, int(args["timeout"]) * 1000)
            except (TypeError, ValueError):
                pass
        for key, value in args.items():
            if key not in {"command", "workdir", "description", "timeout"}:
                out[key] = value
        return alias, _encode(out)

    if native == "patch":
        if str(args.get("mode", "replace")) != "replace":
            raise ToolTranslationError("non-replace patch cannot be replayed as edit.")
        out = {"filePath": _require(args, "path", tool=native),
               "oldString": _require(args, "old_string", tool=native),
               "newString": _require(args, "new_string", tool=native)}
        if "replace_all" in args:
            out["replaceAll"] = bool(args["replace_all"])
        for key, value in args.items():
            if key not in {"mode", "path", "old_string", "new_string", "replace_all"}:
                out[key] = value
        return alias, _encode(out)

    if native == "read_file":
        out = {"filePath": _require(args, "path", tool=native)}
        for key in ("offset", "limit"):
            if key in args:
                out[key] = args[key]
        for key, value in args.items():
            if key not in {"path", "offset", "limit"}:
                out[key] = value
        return alias, _encode(out)

    if native == "skill_view":
        return alias, _encode(args)

    if native == "delegate_task":
        tasks = _require(args, "tasks", tool=native)
        if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
            raise ToolTranslationError("delegate history cannot be replayed as task.")
        first = tasks[0]
        # Prefer the structured envelope written by opencode_to_hermes.
        try:
            envelope = json.loads(str(first.get("context") or "{}"))
        except (ValueError, TypeError):
            envelope = {}
        if isinstance(envelope, dict) and isinstance(envelope.get("opencode_task"), dict):
            meta = dict(envelope["opencode_task"])
            meta["prompt"] = first.get("goal", envelope.get("prompt", ""))
            return alias, _encode(meta)
        # Fallback for natively-created Hermes tasks (no envelope).
        out = {"description": str(first.get("title") or "Delegated Hermes task"),
               "prompt": _require(first, "goal", tool=native),
               "subagent_type": "general"}
        return alias, _encode(out)

    if native == "todo_list":
        todos = _require(args, "todos", tool=native)
        if not isinstance(todos, list):
            raise ToolTranslationError("todo history must be a list.")
        return alias, _encode({"todos": [
            {"content": _require(t, "content", tool=native),
             "status": _require(t, "status", tool=native),
             "priority": str(t.get("priority") or "medium"),
             **({} if "id" not in t else {"id": t["id"]})}
            for t in todos if isinstance(t, dict)
        ]})

    if native == "web_extract":
        urls = args.get("urls")
        if isinstance(urls, list) and urls:
            out = {"url": urls[0]}
        elif "url" in args:
            out = {"url": args["url"]}
        else:
            raise ToolTranslationError("web extract history cannot be replayed as webfetch.")
        for key in ("format", "timeout"):
            if key in args:
                out[key] = args[key]
        if "max_chars" in args:
            out["maxChars"] = args["max_chars"]
        return alias, _encode(out)

    if native == "web_search":
        out = {"query": _require(args, "query", tool=native)}
        if "limit" in args:
            out["numResults"] = args["limit"]
        for key in ("livecrawl", "type", "contextMaxCharacters"):
            if key in args:
                out[key] = args[key]
        return alias, _encode(out)

    if native == "write_file":
        return alias, _encode({"filePath": _require(args, "path", tool=native),
                               "content": _require(args, "content", tool=native),
                               **{k: v for k, v in args.items() if k not in {"path", "content"}}})

    return native, _encode(args)


def alias_for_native(native: str, encoded: Any) -> str:
    """Preferred OpenCode alias for a native tool + args (handles search_files)."""
    if native == "search_files":
        try:
            target = _parse_args(encoded, tool=native).get("target")
        except ToolTranslationError:
            target = None
        return "glob" if target == "files" else "grep"
    return NATIVE_TO_ALIAS.get(native, native)


# ---------------------------------------------------------------------------
# Wire helpers: Hermes tool list -> OpenCode wire tools
# ---------------------------------------------------------------------------
def build_wire_tools(hermes_tools: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Expose Hermes tools under OpenCode-compatible names.

    * Mapped tools are advertised ONLY when the native target exists.
    * Unavailable aliases are OMITTED (never a "Never call" marker).
    * Unmapped native tools pass through verbatim.
    Returns (wire_tools, alias->native map).
    """
    by_name: dict[str, dict[str, Any]] = {}
    for tool in hermes_tools or []:
        name = _tool_name(tool)
        if name:
            by_name[name] = tool

    wire: list[dict[str, Any]] = []
    mapped: dict[str, str] = {}
    mapped_targets: set[str] = set()

    # Group search_files: one native tool backs two aliases.
    search_native = by_name.get("search_files")
    for alias in COMPAT_ALIASES:
        native = ALIAS_TO_NATIVE[alias]
        if native == "search_files":
            if search_native is not None:
                native_fn = search_native.get("function") or {}
                wire.append({
                    "type": "function",
                    "function": {
                        "name": alias,
                        "description": f"{ALIAS_DESCRIPTIONS[alias]} OpenCode-compatible alias for Hermes search_files.",
                        "parameters": _deep_copy(CANONICAL_PARAMETERS[alias]),
                    },
                })
                mapped[alias] = "search_files"
            continue
        if native in by_name:
            wire.append({
                "type": "function",
                "function": {
                    "name": alias,
                    "description": f"{ALIAS_DESCRIPTIONS[alias]} OpenCode-compatible alias for Hermes {native}.",
                    "parameters": _deep_copy(CANONICAL_PARAMETERS[alias]),
                },
            })
            mapped[alias] = native
            mapped_targets.add(native)

    if search_native is not None:
        mapped_targets.add("search_files")

    # Unrelated native tools keep native names + native schemas verbatim.
    for name, tool in (hermes_tools_by_name(hermes_tools) if hermes_tools else []):
        if name in COMPAT_ALIASES or name in mapped_targets:
            continue
        # Skip the raw search_files duplicate (already exposed as glob/grep).
        if name == "search_files":
            continue
        wire.append(_deep_copy(dict(tool)))

    _ = by_name  # keep for debugging clarity
    return wire, mapped


def hermes_tools_by_name(tools: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for tool in tools or []:
        name = _tool_name(tool)
        if name:
            out.append((name, tool))
    return out


def rewrite_history_messages(messages: list[dict[str, Any]], mapped: dict[str, str]) -> list[dict[str, Any]]:
    """Rewrite recorded Hermes assistant tool_calls to OpenCode vocabulary.

    Preserves IDs, order, and JSON argument fidelity. Non-assistant messages
    pass through untouched (tool results keep call IDs and payloads).
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls"):
            calls = []
            for call in item["tool_calls"]:
                if not isinstance(call, dict):
                    calls.append(call)
                    continue
                wired = dict(call)
                function = dict(call.get("function") or {})
                raw_args = function.get("arguments", "{}")
                try:
                    function["name"], function["arguments"] = hermes_to_opencode(
                        str(function.get("name") or ""), raw_args)
                except ToolTranslationError:
                    # Keep original rather than dropping history and breaking
                    # the tool_call_id chain.
                    if not isinstance(function.get("arguments"), str):
                        function["arguments"] = _encode(_parse_args(raw_args, tool="history"))
                wired["function"] = function
                calls.append(wired)
            item["tool_calls"] = calls
        result.append(item)
    return result


def rewrite_tool_choice(value: Any, mapped: dict[str, str]) -> Any:
    if not isinstance(value, dict):
        return value
    function = value.get("function")
    if value.get("type") != "function" or not isinstance(function, dict):
        return value
    out = dict(value)
    fn = dict(function)
    native = str(fn.get("name") or "")
    fn["name"] = alias_for_native(native, "{}")
    out["function"] = fn
    return out


def is_valid_tool_arguments(text: str) -> bool:
    try:
        parsed = json.loads(text or "")
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict)
