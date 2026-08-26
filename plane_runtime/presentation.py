"""Bounded, invocation-local model guidance for Plane runtime calls."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .g1_contract import G1RunSnapshot, MAX_PROMPT_BYTES


class PresentationBoundsError(ValueError):
    """The model guidance cannot fit the accepted prompt bound."""


_RULES = (
    "Call plane_operation directly with the listed exact input shape for every eager operation; eager operations are already disclosed and must not be rediscovered.",
    "For an operation not listed under eagerOperations, call catalog.search once and then operation:catalog.describe once before invocation.",
    "For an assigned work-item route, make operation:search_workspace the first Plane call and use its prepared typed references; do not begin with catalog.search or catalog.describe.",
    "Never guess input field names.",
    "Disclosure is not authorization.",
    "Ordinary final text is not publication.",
)

_PLANE_CODE_MODE_RULES = (
    "Use Plane:discover only when the current declarations do not contain a required method; describe the whole workflow.",
    "Plane:discover replaces the previous declaration slice and does not authorize execution.",
    "Use ordinary typed methods on plane; do not import, export, construct a client, or access credentials, network, filesystem, or process state.",
    "Return compact JSON for further reasoning, or call await plane.finish(...) exactly once.",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def build_model_guidance(snapshot: G1RunSnapshot) -> str:
    """Render the complete bounded model-facing invocation guidance."""

    assignment = {
        "workspaceRef": snapshot.workspace_ref,
        "targetRef": snapshot.target_ref,
        "objective": snapshot.objective,
        "acceptanceCriteria": list(snapshot.acceptance_criteria),
        "contextRefs": list(snapshot.context_refs),
        "eagerOperations": [
            {
                "operationRef": operation["operationRef"],
                "inputSchema": _plain(operation["inputSchema"]),
            }
            for operation in snapshot.eager_operations
        ],
    }
    compact_assignment = json.dumps(
        assignment,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    guidance = "\n".join(
        (
            snapshot.behavioral_prompt,
            "",
            "Plane invocation guidance:",
            compact_assignment,
            "Rules:",
            *(f"- {rule}" for rule in _RULES),
        )
    )
    if len(guidance.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise PresentationBoundsError(
            f"Plane invocation guidance exceeds {MAX_PROMPT_BYTES} UTF-8 bytes"
        )
    return guidance


def build_plane_task_kit(snapshot: G1RunSnapshot) -> dict[str, Any]:
    """Return only the immutable assignment values allowed in generated code."""

    return {
        "target": snapshot.target_ref,
        "objective": snapshot.objective,
        "acceptanceCriteria": list(snapshot.acceptance_criteria),
    }


def build_initial_declarations(snapshot: G1RunSnapshot) -> str:
    """Build the bounded base declarations without exposing gateway vocabulary."""

    methods: list[str] = []
    method_names = {
        "operation:work_item.read": ("workItems", "retrieve"),
        "operation:search_workspace": ("workItems", "search"),
        "operation:agent.assignment.delegate": ("assignments", "delegate"),
    }
    for operation in snapshot.eager_operations:
        namespace_method = method_names.get(str(operation["operationRef"]))
        if namespace_method is None:
            continue
        namespace, method = namespace_method
        declaration = f"  {method}(input: Readonly<Record<string, unknown>>): Promise<unknown>;"
        entry = next((item for item in methods if item.startswith(f"  {namespace}:")), None)
        if entry is None:
            methods.append(f"  {namespace}: {{\n{declaration}\n  }};")
        elif declaration not in entry:
            methods[methods.index(entry)] = entry[:-3] + f"\n{declaration}\n  }};"
    declarations = "\n".join(
        (
            "type JsonPrimitive = string | number | boolean | null;",
            "type JsonValue = JsonPrimitive | readonly JsonValue[] | { readonly [key: string]: JsonValue };",
            "type PlaneRef<Kind extends string> = string & { readonly __planeKind: Kind };",
            "declare const task: Readonly<{ target: PlaneRef<string>; objective: string; acceptanceCriteria: readonly string[] }>;",
            "type FinishInput = { kind: \"completed\"; summary: string; content?: string; artifacts?: readonly PlaneRef<\"artifact\">[]; evidence?: readonly PlaneRef<string>[] } | { kind: \"waiting_for_input\"; question: string; context?: JsonValue } | { kind: \"blocked\"; reason: string; evidence?: readonly PlaneRef<string>[] };",
            "declare const plane: Readonly<{",
            *methods,
            "  finish(input: FinishInput): Promise<never>;",
            "}>;",
        )
    )
    if len(declarations.encode("utf-8")) > 16 * 1024:
        raise PresentationBoundsError("Plane declaration slice exceeds 16384 UTF-8 bytes")
    return declarations


def build_plane_code_mode_guidance(snapshot: G1RunSnapshot) -> str:
    """Render the stable Plane Code Mode prompt and immutable initial task kit."""

    task_kit = json.dumps(
        build_plane_task_kit(snapshot),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    guidance = "\n".join(
        (
            snapshot.behavioral_prompt,
            "",
            "Plane Agent Code Mode:",
            "The following assignment values are immutable typed data:",
            task_kit,
            "Initial declarations:",
            build_initial_declarations(snapshot),
            "Rules:",
            *(f"- {rule}" for rule in _PLANE_CODE_MODE_RULES),
        )
    )
    if len(guidance.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise PresentationBoundsError(
            f"Plane Code Mode guidance exceeds {MAX_PROMPT_BYTES} UTF-8 bytes"
        )
    return guidance


__all__ = [
    "PresentationBoundsError",
    "build_initial_declarations",
    "build_model_guidance",
    "build_plane_code_mode_guidance",
    "build_plane_task_kit",
]
