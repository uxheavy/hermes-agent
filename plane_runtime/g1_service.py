"""One-shot G1 runtime service for a replaceable Hermes child process."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, TextIO

from .g1_contract import (
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    bind_snapshot_and_invocation,
    build_event,
    build_exit,
    validate_g1_frames,
)
from .hermes_adapter import (
    DeterministicKernelAdapter,
    HermesKernelAdapter,
    HermesKernelResult,
    HermesCredentialSource,
    NeverCancelled,
)
from .host_port import PlaneHostPort


_MODEL_USAGE_PROTOCOL = "plane.agent-runtime/internal-usage/v1"
_HOST_CALLBACK_PHASES = frozenset(
    {"before_host_call", "host_return", "model_observation_emit", "adapter_event"}
)


def _bounded_host_operation_diagnostic(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Project only the finite, digested host callback facts onto RuntimeExit."""

    if not isinstance(value, Mapping):
        return None
    allowed = {
        "callbackPhase",
        "operationRefDigest",
        "codeModeHostStatus",
        "codeModeFailureClass",
    }
    if set(value).difference(allowed):
        return None
    phase = value.get("callbackPhase")
    operation_ref_digest = value.get("operationRefDigest")
    if (
        phase not in _HOST_CALLBACK_PHASES
        or not isinstance(operation_ref_digest, str)
        or len(operation_ref_digest) != 64
        or any(char not in "0123456789abcdef" for char in operation_ref_digest)
    ):
        return None
    result: dict[str, Any] = {
        "callbackPhase": phase,
        "operationRefDigest": operation_ref_digest,
    }
    code_mode_fields = {"codeModeHostStatus", "codeModeFailureClass"}
    present = code_mode_fields.intersection(value)
    if present and present != code_mode_fields:
        return None
    if present:
        if value["codeModeHostStatus"] not in {
            "ok", "replayed", "denied", "conflict", "unavailable", "invalid"
        } or value["codeModeFailureClass"] not in {
            "code_mode", "callback", "transport", "contract", "unknown"
        }:
            return None
        result.update(
            {
                "codeModeHostStatus": value["codeModeHostStatus"],
                "codeModeFailureClass": value["codeModeFailureClass"],
            }
        )
    return result


def _write_model_usage(diagnostics: TextIO | None, model_calls: int | None) -> None:
    if diagnostics is None or isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
        return
    diagnostics.write(json.dumps(
        {"modelCalls": model_calls, "protocol": _MODEL_USAGE_PROTOCOL},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n")
    diagnostics.flush()


def _runtime_suffix(invocation_id: str) -> str:
    return hashlib.sha256(invocation_id.encode("utf-8")).hexdigest()[:16]


def _publication(kind: str, suffix: str) -> dict[str, str]:
    return {
        "action": "proposal",
        "productKind": kind,
        "productRef": f"{kind.replace('_', '-')}:runtime-{suffix}",
        "operationAttemptRef": f"operation-attempt:runtime-{suffix}",
    }


def _lease_is_alive(invocation: G1InvocationEnvelope) -> bool:
    expires_at = str(invocation.lease["expiresAt"])
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise G1ContractError("lease.expiresAt is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise G1ContractError("lease.expiresAt must include a timezone")
    return datetime.now(timezone.utc) < parsed.astimezone(timezone.utc)


def _failure_result(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    failure_cause: str | None = None,
) -> HermesKernelResult:
    return HermesKernelResult(
        kind="failed",
        failure_code=code,
        failure_message=message,
        failure_cause=failure_cause,
        retryable=retryable,
    )


def _terminal_failure(
    snapshot: G1RunSnapshot,
    invocation: G1InvocationEnvelope,
    result: HermesKernelResult,
    sequence: int,
) -> dict[str, Any]:
    kind = result.kind
    if kind == "cancelled":
        exit_kind = "cancelled"
        code = result.failure_code or "cancelled"
    elif kind == "waiting_for_input":
        raise G1ContractError("waiting_for_input must be represented by an input event")
    elif kind == "blocked":
        exit_kind = "blocked"
        code = result.failure_code or "runtime_error"
    else:
        exit_kind = "failed"
        code = result.failure_code or "runtime_error"
    failure = {
        "code": code,
        "message": result.failure_message or "Hermes runtime did not complete",
        "retryable": bool(result.retryable),
    }
    if result.failure_cause is not None:
        failure["cause"] = result.failure_cause
    diagnostic = _bounded_host_operation_diagnostic(result.host_operation_diagnostic)
    if diagnostic is not None:
        failure.update(diagnostic)
    return build_exit(
        snapshot=snapshot,
        invocation=invocation,
        final_sequence=sequence - 1 if sequence else 0,
        kind=exit_kind,
        failure=failure,
    )


def serve_once_g1(
    request_line: str,
    output: TextIO,
    *,
    production: bool = False,
    diagnostics: TextIO | None = None,
    model_call_allowance: int | None = None,
    host_port: PlaneHostPort | None = None,
    credential_source: HermesCredentialSource | None = None,
    http_client_factory: Callable[[], Any] | None = None,
    cancellation: Callable[[], bool] | None = None,
) -> int:
    """Consume exactly one G1 request and write direct event/exit frames.

    The child owns no Plane state.  It validates the immutable inputs, invokes
    one Hermes adapter, and returns observations plus runtime evidence only.
    ``http_client_factory`` is an optional trusted, invocation-scoped
    dependency and is passed through only when configured.
    """

    request = json.loads(request_line)
    if not isinstance(request, dict) or set(request) != {"run", "invocation"}:
        raise G1ContractError("G1 service request must contain run and invocation")
    snapshot = G1RunSnapshot.from_dict(request["run"])
    invocation = G1InvocationEnvelope.from_dict(request["invocation"])
    bind_snapshot_and_invocation(snapshot, invocation)

    frames: list[dict[str, Any]] = []

    def emit_body(body: Mapping[str, Any]) -> None:
        event = build_event(
            snapshot=snapshot,
            invocation=invocation,
            sequence=len(frames),
            body=body,
        )
        frames.append(event)

    try:
        if not _lease_is_alive(invocation):
            result = _failure_result("lease_expired", "invocation lease is no longer valid")
        elif any(value == 0 for value in invocation.remaining_budget.values()):
            result = _failure_result("budget_exhausted", "cumulative invocation budget is exhausted")
        elif production and (model_call_allowance is None or isinstance(model_call_allowance, bool) or not isinstance(model_call_allowance, int) or model_call_allowance < 0):
            result = _failure_result(
                "runtime_error",
                "trusted model-call allowance is required",
                failure_cause="static_configuration_failure",
            )
        elif production and credential_source is None:
            result = _failure_result(
                "runtime_error",
                "trusted bootstrap credential handoff is required",
                failure_cause="static_configuration_failure",
            )
        elif snapshot.adapter_name == "deterministic-test-adapter" and not production:
            result = DeterministicKernelAdapter().dispatch(snapshot, invocation, NeverCancelled(), emit_body)
        elif snapshot.adapter_name == "deterministic-test-adapter" and production:
            result = _failure_result(
                "runtime_error",
                "deterministic adapter is test-only",
                failure_cause="static_configuration_failure",
            )
        else:
            adapter_kwargs: dict[str, Any] = {
                "credential_source": credential_source,
                "host_port": host_port,
            }
            if http_client_factory is not None:
                adapter_kwargs["http_client_factory"] = http_client_factory
            result = HermesKernelAdapter(**adapter_kwargs).dispatch(
                snapshot,
                invocation,
                cancellation or NeverCancelled(),
                emit_body,
                model_call_allowance=model_call_allowance,
            )
    except Exception:
        result = _failure_result("runtime_error", "Hermes runtime execution failed", retryable=True)

    if result.kind == "completed":
        exit_frame = build_exit(
            snapshot=snapshot,
            invocation=invocation,
            final_sequence=len(frames) - 1,
            kind="completed",
        )
    elif result.kind == "waiting_for_input":
        if not result.question:
            raise G1ContractError("waiting_for_input result has no question")
        input_event = build_event(
            snapshot=snapshot,
            invocation=invocation,
            sequence=len(frames),
            body={
                "kind": "input_request_observed",
                "question": result.question,
                "publication": _publication("input_request", _runtime_suffix(invocation.invocation_id)),
            },
        )
        frames.append(input_event)
        exit_frame = build_exit(
            snapshot=snapshot,
            invocation=invocation,
            final_sequence=len(frames) - 1,
            kind="waiting_for_input",
            input_event_ref=input_event["eventId"],
        )
    else:
        exit_frame = _terminal_failure(snapshot, invocation, result, len(frames))

    all_frames = [*frames, exit_frame]
    validate_g1_frames(all_frames, snapshot.to_dict(), invocation.to_dict())
    for frame in all_frames:
        output.write(json.dumps(frame, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    output.flush()
    _write_model_usage(diagnostics, result.model_calls)
    return 0


__all__ = ["serve_once_g1"]
