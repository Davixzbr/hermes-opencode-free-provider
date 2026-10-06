"""Tool translation fidelity: IDs, args, round-trips, errors, parallelism."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from tools_adapter import (ToolTranslationError, hermes_to_opencode,  # noqa: E402
                           opencode_to_hermes, rewrite_history_messages,
                           rewrite_tool_choice)


class TestOpenCodeToHermes(unittest.TestCase):
    def test_bash_ms_to_s(self):
        name, args = opencode_to_hermes("bash", json.dumps({"command": "ls", "timeout": 5000}))
        self.assertEqual(name, "terminal")
        self.assertEqual(json.loads(args), {"command": "ls", "timeout": 5})

    def test_bash_preserves_extras(self):
        name, args = opencode_to_hermes("bash", json.dumps({"command": "x", "description": "d"}))
        self.assertEqual(json.loads(args)["description"], "d")

    def test_edit(self):
        name, args = opencode_to_hermes("edit", json.dumps(
            {"filePath": "a.txt", "oldString": "x", "newString": "y"}))
        self.assertEqual(name, "patch")
        parsed = json.loads(args)
        self.assertEqual(parsed["path"], "a.txt")
        self.assertEqual(parsed["mode"], "replace")

    def test_glob_grep_target(self):
        n1, a1 = opencode_to_hermes("glob", json.dumps({"pattern": "**/*.py"}))
        self.assertEqual(json.loads(a1)["target"], "files")
        n2, a2 = opencode_to_hermes("grep", json.dumps({"pattern": "foo", "include": "*.py"}))
        parsed = json.loads(a2)
        self.assertEqual(parsed["target"], "content")
        self.assertEqual(parsed["file_glob"], "*.py")

    def test_todowrite_preserves_priority(self):
        payload = {"todos": [{"content": "a", "status": "pending", "priority": "high"}]}
        _, args = opencode_to_hermes("todowrite", json.dumps(payload))
        parsed = json.loads(args)
        self.assertEqual(parsed["todos"][0]["priority"], "high")
        self.assertTrue(parsed["todos"][0]["id"].startswith("oc-"))

    def test_task_lossless_roundtrip(self):
        oc = {"description": "d", "prompt": "p", "subagent_type": "general",
              "task_id": "t1", "background": True}
        native, encoded = opencode_to_hermes("task", json.dumps(oc))
        self.assertEqual(native, "delegate_task")
        alias, back = hermes_to_opencode(native, encoded)
        self.assertEqual(alias, "task")
        restored = json.loads(back)
        self.assertEqual(restored["description"], "d")
        self.assertEqual(restored["prompt"], "p")
        self.assertEqual(restored["subagent_type"], "general")
        self.assertEqual(restored["task_id"], "t1")
        self.assertTrue(restored["background"])

    def test_webfetch_websearch(self):
        _, a = opencode_to_hermes("webfetch", json.dumps({"url": "https://x"}))
        self.assertEqual(json.loads(a)["urls"], ["https://x"])
        _, b = opencode_to_hermes("websearch", json.dumps({"query": "q", "numResults": 5}))
        self.assertEqual(json.loads(b)["limit"], 5)

    def test_missing_required_raises(self):
        with self.assertRaises(ToolTranslationError):
            opencode_to_hermes("bash", json.dumps({}))

    def test_invalid_json_raises(self):
        with self.assertRaises(ToolTranslationError):
            opencode_to_hermes("bash", "{not json")

    def test_unknown_passthrough(self):
        name, args = opencode_to_hermes("mystery", json.dumps({"a": 1}))
        self.assertEqual(name, "mystery")


class TestHermesToOpenCode(unittest.TestCase):
    def test_terminal_s_to_ms(self):
        alias, args = hermes_to_opencode("terminal", json.dumps({"command": "ls", "timeout": 5}))
        self.assertEqual(alias, "bash")
        self.assertEqual(json.loads(args)["timeout"], 5000)

    def test_search_files_dispatch(self):
        a1, _ = hermes_to_opencode("search_files", json.dumps({"target": "files", "pattern": "x"}))
        a2, _ = hermes_to_opencode("search_files", json.dumps({"target": "content", "pattern": "x"}))
        self.assertEqual(a1, "glob")
        self.assertEqual(a2, "grep")

    def test_patch_replace_only(self):
        with self.assertRaises(ToolTranslationError):
            hermes_to_opencode("patch", json.dumps({"mode": "create", "path": "a"}))


class TestHistory(unittest.TestCase):
    def _history(self, n=2):
        calls = [{"id": f"call_{i}", "type": "function",
                  "function": {"name": "terminal",
                               "arguments": json.dumps({"command": f"echo {i}"})}} for i in range(n)]
        return [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            *[{"role": "tool", "tool_call_id": f"call_{i}",
                "content": f"ok {i}"} for i in range(n)],
        ]

    def test_ids_preserved_multi_turn(self):
        history = self._history(2)
        mapped = {"bash": "terminal"}
        # history is already native; rewrite maps native->alias for replay
        rewritten = rewrite_history_messages(history, mapped)
        assistant = next(m for m in rewritten if m.get("role") == "assistant")
        self.assertEqual(len(assistant["tool_calls"]), 2)
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_0")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "bash")
        tools = [m for m in rewritten if m.get("role") == "tool"]
        self.assertEqual(tools[0]["tool_call_id"], "call_0")

    def test_ten_parallel_calls(self):
        history = self._history(10)
        rewritten = rewrite_history_messages(history, {"bash": "terminal"})
        assistant = next(m for m in rewritten if m.get("role") == "assistant")
        self.assertEqual(len(assistant["tool_calls"]), 10)
        ids = [c["id"] for c in assistant["tool_calls"]]
        self.assertEqual(len(set(ids)), 10)

    def test_tool_choice_rewrite(self):
        out = rewrite_tool_choice({"type": "function",
                                   "function": {"name": "terminal"}}, {"bash": "terminal"})
        self.assertEqual(out["function"]["name"], "bash")

    def test_error_retry_chain_kept(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "terminal", "arguments": json.dumps({"command": "bad"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Error: not found"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "terminal", "arguments": json.dumps({"command": "good"})}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        ]
        rewritten = rewrite_history_messages(history, {"bash": "terminal"})
        self.assertEqual(rewritten[1]["tool_call_id"], "c1")
        self.assertIn("Error", rewritten[1]["content"])


if __name__ == "__main__":
    unittest.main()
