"""Contract tests for repository placement policy."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "repository_policy.py"
_SPEC = importlib.util.spec_from_file_location("repository_policy", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Failed to load repository_policy.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def evaluate(changes, files=None, existing=None):
    files = files or {"toolsets.py": "_HERMES_CORE_TOOLS = ['terminal']"}
    existing = existing or set()
    return _MODULE.evaluate(
        changes,
        base_has_path=lambda path: path in existing,
        read_text=lambda path: files.get(path),
    )


class RepositoryPolicyTests(unittest.TestCase):
    def test_rejects_new_generic_roots_and_in_tree_memory_providers(self):
        errors = evaluate(
            [
                ("A", "helpers/new.py", None),
                ("A", "plugins/memory/vendor/__init__.py", None),
            ]
        )
        self.assertEqual([error[0] for error in errors], ["HR001", "HR002"])

    def test_grandfathers_existing_generic_roots_and_memory_providers(self):
        self.assertEqual(
            evaluate(
                [("A", "helpers/new.py", None), ("M", "plugins/memory/honcho/client.py", None)],
                existing={"helpers", "plugins/memory/honcho"},
            ),
            [],
        )

    def test_allows_shared_memory_root_files(self):
        self.assertEqual(evaluate([("A", "plugins/memory/shared_helper.py", None)]), [])

    def test_rejects_new_legacy_provider_modules(self):
        self.assertEqual(evaluate([("A", "providers/new_vendor.py", None)])[0][0], "HR003")

    def test_requires_registered_tools_to_be_exposed(self):
        files = {
            "toolsets.py": "_HERMES_CORE_TOOLS = ['terminal']",
            "tools/example.py": "registry.register(name='example', handler=handler)",
        }
        self.assertEqual(evaluate([("A", "tools/example.py", None)], files=files)[0][0], "HR004")
        files["toolsets.py"] = "_HERMES_CORE_TOOLS = ['terminal', 'example']"
        self.assertEqual(evaluate([("A", "tools/example.py", None)], files=files)[0][0], "HR004")
        files["toolsets.py"] = "TOOLSETS = {'core': {'tools': ['terminal', 'example']}}"
        self.assertEqual(evaluate([("A", "tools/example.py", None)], files=files), [])

    def test_ignores_unrelated_toolset_strings(self):
        files = {
            "toolsets.py": "TOOLSETS = {'video': {'description': 'example', 'tools': []}}",
            "tools/example.py": "registry.register(name='video', handler=handler)",
        }
        self.assertEqual(evaluate([("A", "tools/example.py", None)], files=files)[0][0], "HR004")

    def test_parses_renames_without_losing_the_old_path(self):
        self.assertEqual(
            _MODULE.parse_name_status(b"R100\0providers/old.py\0providers/new.py\0"),
            [("R", "providers/new.py", "providers/old.py")],
        )
