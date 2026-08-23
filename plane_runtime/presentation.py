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
    "After search_workspace returns the assigned work item, copy the opaque assignmentWorkItemReadCall string into plane_operation input {\"preparedCallRef\":\"<the returned prepared-call:...>\"}; do not construct an action/operationRef/input envelope, nest an object under preparedCallRef, or use project_id/issue_id.",
    "Never guess input field names.",
    "Disclosure is not authorization.",
    "Ordinary final text is not publication.",
)

_CODE_MODE_RULES = (
    "Use plane_execute_typescript for the supplied Code Mode module; do not call plane_operation for this commission.",
    "Use plane_publish explicitly for the terminal product publication; ordinary final text is not publication.",
    "For an operation not listed under eagerOperations, call catalog.search once and then operation:catalog.describe once before invocation.",
    "Never guess input field names.",
    "Disclosure is not authorization.",
)

_PREPARED_READ_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["preparedCallRef"],
    "properties": {
        "preparedCallRef": {
            "type": "string",
            "minLength": len("prepared-call:"),
            "maxLength": 256,
        }
    },
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _model_input_schema(operation: Mapping[str, Any]) -> Any:
    """Project the opaque work-item handoff into the model-facing contract."""

    if operation.get("operationRef") == "operation:work_item.read":
        return _plain(_PREPARED_READ_INPUT_SCHEMA)
    return _plain(operation["inputSchema"])


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
                "inputSchema": _model_input_schema(operation),
            }
            for operation in snapshot.eager_operations
        ],
    }
    if snapshot.standard_route is not None:
        assignment["standardRoute"] = _plain(snapshot.standard_route)
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
            *(f"- {rule}" for rule in (_CODE_MODE_RULES if snapshot.model_toolset == "code_mode_only" else _RULES)),
        )
    )
    if len(guidance.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise PresentationBoundsError(
            f"Plane invocation guidance exceeds {MAX_PROMPT_BYTES} UTF-8 bytes"
        )
    return guidance


__all__ = ["PresentationBoundsError", "build_model_guidance"]
