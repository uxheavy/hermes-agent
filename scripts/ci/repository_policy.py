#!/usr/bin/env python3
"""Enforce mechanically provable repository placement rules."""

from __future__ import annotations

import argparse
import ast
import subprocess
from collections.abc import Callable
from pathlib import Path

GENERIC_ROOTS = {"common", "helpers", "shared", "utils"}


def parse_name_status(raw: bytes) -> list[tuple[str, str, str | None]]:
    fields = [part.decode() for part in raw.split(b"\0") if part]
    changes: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status.startswith(("R", "C")):
            old_path = fields[index]
            path = fields[index + 1]
            index += 2
            changes.append((status[0], path, old_path))
        else:
            changes.append((status[0], fields[index], None))
            index += 1
    return changes


def registered_tools(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "register":
            continue
        for keyword in node.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    names.add(keyword.value.value)
    return names


def exposed_tool_names(source: str) -> set[str]:
    return {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def evaluate(
    changes: list[tuple[str, str, str | None]],
    *,
    base_has_path: Callable[[str], bool],
    read_text: Callable[[str], str | None],
) -> list[tuple[str, str, str]]:
    errors: list[tuple[str, str, str]] = []
    toolsets = exposed_tool_names(read_text("toolsets.py") or "")

    for status, path, _old_path in changes:
        added = status in {"A", "R"}
        parts = Path(path).parts

        if added and parts and parts[0] in GENERIC_ROOTS and not base_has_path(parts[0]):
            errors.append(("HR001", path, f"new generic root {parts[0]} needs a concrete owner"))

        if added and len(parts) >= 3 and parts[:2] == ("plugins", "memory"):
            provider_root = "/".join(parts[:3])
            if not base_has_path(provider_root):
                errors.append(("HR002", path, "new memory providers belong in standalone plugin repositories"))

        if added and len(parts) == 2 and parts[0] == "providers" and path.endswith(".py"):
            if not base_has_path(path):
                errors.append(("HR003", path, "new providers belong in plugins/model-providers"))

        if status in {"A", "M", "R"} and len(parts) == 2 and parts[0] == "tools" and path.endswith(".py"):
            source = read_text(path)
            if source is None:
                continue
            for name in sorted(registered_tools(source) - toolsets):
                errors.append(("HR004", path, f"registered tool {name!r} is not exposed by toolsets.py"))

    return errors


def git(*args: str) -> bytes:
    return subprocess.check_output(("git", *args))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    changes = parse_name_status(git("diff", "--name-status", "-z", "--find-renames", f"{args.base}...HEAD"))

    def base_has_path(path: str) -> bool:
        return subprocess.run(
            ("git", "cat-file", "-e", f"{args.base}:{path}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0

    def read_text(path: str) -> str | None:
        candidate = Path(path)
        return candidate.read_text(encoding="utf-8") if candidate.is_file() else None

    errors = evaluate(changes, base_has_path=base_has_path, read_text=read_text)
    for rule, path, message in errors:
        print(f"{path}: {rule} {message}")
    if errors:
        return 1
    print(f"repository policy passed ({len(changes)} changed paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
