"""Minimal JSON-lines host for a separately spawned runtime process.

This is deliberately a transport-shaped seam, not a queue or RPC decision.
One process accepts one invocation request, streams validated events, and
returns one exit.  A supervisor can replace the process with a new invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Callable, TextIO

from .adapter import (
    CanonicalLeaseAuthority,
    CanonicalLeaseBinding,
    CheckpointAuthority,
    CheckpointAttestation,
    EventCollector,
    ExecutionPhase,
    FixtureCanonicalLeaseAuthority,
    FixtureCheckpointAuthority,
    FixtureTerminalReconciliationPort,
    FakeKernel,
    FakeKernelPlan,
    KernelPort,
    RecordingHost,
    RuntimeHost,
    CancellationAuthority,
    CancellationSignal,
    TerminalReconciliationError,
    TerminalReconciliationPort,
    execute,
    reconcile_terminal_proposal,
)
from .contract import (
    AssignmentSnapshot,
    ContractDigests,
    ContractError,
    BindingError,
    InvocationEnvelope,
    MAX_NEW_CONTEXT_EVENT_REFS,
    OperationDescriptor,
    RuntimeFailure,
    RuntimeModelRoute,
    RuntimeBudgetPolicy,
    RunSnapshot,
    RuntimeBudget,
    TerminalProposal,
    ToolPresentation,
    VersionedContextRef,
)


GENERIC_RUNTIME_FAILURE = "runtime execution failed; Plane reconciliation is required"
InternalFailureHook = Callable[[Exception], None]


class _DemoTerminalPort(FixtureTerminalReconciliationPort):
    """Explicit demo-only atomic fixture, not production durability."""


def _handle_runtime_failure(
    *,
    exc: Exception,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    collector: EventCollector,
    terminal_port: TerminalReconciliationPort,
    output: TextIO,
    internal_failure_hook: InternalFailureHook | None,
) -> int:
    """Keep raw failure detail local and require a bounded terminal handoff."""

    if internal_failure_hook is not None:
        try:
            internal_failure_hook(exc)
        except Exception:
            pass
    proposal = TerminalProposal(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        kind="failed",
        final_sequence=collector.last_sequence,
        evidence_event_ids=tuple(
            event.event_id for event in collector.events[-MAX_NEW_CONTEXT_EVENT_REFS:]
        ),
        failure=RuntimeFailure(
            code="runtime_exception",
            message=GENERIC_RUNTIME_FAILURE,
            retryable=True,
        ),
    )
    try:
        receipt = reconcile_terminal_proposal(port=terminal_port, proposal=proposal)
    except Exception:
        receipt = None
    if receipt is not None and receipt.accepted and receipt.legal_transition:
        output.write(
            json.dumps(
                {"type": "reconciliation", "receipt": receipt.to_dict()},
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        return 0
    output.write(
        json.dumps(
            {
                "type": "reconciliation_request",
                "request": {
                    "kind": "failed",
                    "code": "runtime_exception",
                    "message": GENERIC_RUNTIME_FAILURE,
                    "runId": run.run_id,
                    "invocationId": invocation.invocation_id,
                    "finalSequence": collector.last_sequence,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    output.flush()
    return 1


def _handle_terminal_reconciliation_failure(
    *,
    exc: TerminalReconciliationError,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    collector: EventCollector,
    output: TextIO,
    internal_failure_hook: InternalFailureHook | None,
) -> int:
    """Leave rejected/unavailable terminal application for the supervisor."""

    if internal_failure_hook is not None:
        try:
            internal_failure_hook(exc)
        except Exception:
            pass
    output.write(
        json.dumps(
            {
                "type": "reconciliation_request",
                "request": {
                    "kind": "failed",
                    "code": "terminal_reconciliation_unavailable",
                    "message": "terminal reconciliation is unavailable; supervisor action is required",
                    "runId": run.run_id,
                    "invocationId": invocation.invocation_id,
                    "finalSequence": collector.last_sequence,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    output.flush()
    return 1


_DEMO_LEASE_BINDINGS = (
    CanonicalLeaseBinding(
        run_id="run:one",
        invocation_id="invocation:one",
        lease_id="lease:one",
        holder_ref="host:one",
        active=True,
        expires_at="2099-01-01T00:00:00Z",
    ),
    CanonicalLeaseBinding(
        run_id="run:one",
        invocation_id="invocation:replacement",
        lease_id="lease:one",
        holder_ref="host:one",
        active=True,
        expires_at="2099-01-01T00:00:00Z",
    ),
)


def _demo_snapshot() -> RunSnapshot:
    """Return the fixed host fixture used to attest the demo continuation."""

    return RunSnapshot(
        protocol="plane.agent-runtime/v1",
        run_id="run:one",
        assignment=AssignmentSnapshot(
            version="assignment:v1",
            target_ref="issue:one",
            objective="Produce the assigned result",
            acceptance_criteria=("The result is deterministic",),
        ),
        actor_ref="agent:one",
        profile_version="profile:v1",
        behavioral_prompt="Use the runtime port.",
        context=(VersionedContextRef("context:one", "sha256:context"),),
        tool_presentation=ToolPresentation(
            eager_operations=(OperationDescriptor("operation:read", "sha256:operation"),),
            catalog_digest="sha256:catalog",
        ),
        model=RuntimeModelRoute(model="fake-model", route_ref="route:fake"),
        total_budget_policy=RuntimeBudgetPolicy(total=RuntimeBudget(5, 100, 100)),
        contract_digests=ContractDigests("snapshot:v1", "invocation:v1", "event:v1", "exit:v1"),
    )


_DEMO_CHECKPOINT_ATTESTATION = CheckpointAttestation(
    checkpoint_ref="checkpoint:one",
    source_run_id="run:one",
    source_invocation_id="invocation:one",
    snapshot_digest=_demo_snapshot().digest(),
    actor_ref="agent:one",
    profile_version="profile:v1",
    continuation_event_ref="event:answer",
    continuation_trigger_kind="continuation",
    allowed_target_invocation_id="invocation:replacement",
)


def _demo_lease_binding(invocation: InvocationEnvelope) -> CanonicalLeaseBinding:
    """Select one explicit fixture binding; never copy fields from the envelope."""

    for binding in _DEMO_LEASE_BINDINGS:
        if (binding.run_id, binding.invocation_id) == (invocation.run_id, invocation.invocation_id):
            return binding
    raise ContractError("demo host has no canonical lease fixture for this invocation")


def _demo_checkpoint_attestation(invocation: InvocationEnvelope) -> CheckpointAttestation | None:
    if invocation.checkpoint_ref is None:
        return None
    if (
        invocation.run_id != "run:one"
        or invocation.invocation_id != "invocation:replacement"
        or invocation.checkpoint_ref != "checkpoint:one"
        or invocation.trigger.kind != "continuation"
        or invocation.trigger.event_ref != "event:answer"
    ):
        raise ContractError("demo host has no canonical checkpoint fixture for this invocation")
    return _DEMO_CHECKPOINT_ATTESTATION


def _fake_plan(raw: Any) -> FakeKernelPlan:
    if raw is None:
        return FakeKernelPlan()
    if not isinstance(raw, dict):
        raise ContractError("fakePlan must be an object")
    unknown = sorted(set(raw).difference({
        "transcript",
        "transcriptRef",
        "usage",
        "terminalKind",
        "inputRequest",
        "inputRequestRef",
        "outcomeSubmissionRequested",
        "publicationRequested",
        "holdAfterObservations",
    }))
    if unknown:
        raise ContractError(f"fakePlan has unknown field(s): {', '.join(unknown)}")
    defaults = FakeKernelPlan()
    usage = raw.get("usage")
    transcript = raw.get("transcript", defaults.transcript)
    transcript_ref = raw.get("transcriptRef", defaults.transcript_ref)
    terminal_kind = raw.get("terminalKind", defaults.terminal_kind)
    input_request = raw.get("inputRequest")
    input_request_ref = raw.get("inputRequestRef", defaults.input_request_ref)
    outcome_submission_requested = raw.get(
        "outcomeSubmissionRequested", defaults.outcome_submission_requested
    )
    publication_requested = raw.get("publicationRequested", False)
    hold_after_observations = raw.get("holdAfterObservations")
    if not isinstance(transcript, str) or not transcript:
        raise ContractError("fakePlan.transcript must be a non-empty string")
    if not isinstance(transcript_ref, str) or not transcript_ref:
        raise ContractError("fakePlan.transcriptRef must be a non-empty string")
    if not isinstance(terminal_kind, str):
        raise ContractError("fakePlan.terminalKind must be a string")
    if input_request is not None and (not isinstance(input_request, str) or not input_request):
        raise ContractError("fakePlan.inputRequest must be a non-empty string")
    if not isinstance(input_request_ref, str) or not input_request_ref:
        raise ContractError("fakePlan.inputRequestRef must be a non-empty string")
    if not isinstance(publication_requested, bool):
        raise ContractError("fakePlan.publicationRequested must be a boolean")
    if not isinstance(outcome_submission_requested, bool):
        raise ContractError("fakePlan.outcomeSubmissionRequested must be a boolean")
    if hold_after_observations is not None and (
        isinstance(hold_after_observations, bool)
        or not isinstance(hold_after_observations, int)
        or hold_after_observations < 1
    ):
        raise ContractError("fakePlan.holdAfterObservations must be an integer >= 1")
    return FakeKernelPlan(
        transcript=transcript,
        transcript_ref=transcript_ref,
        usage=RuntimeBudget.from_dict(
            usage
            if usage is not None
            else {"iterations": 1, "inputTokens": 0, "outputTokens": 4},
            "fakePlan.usage",
        ),
        terminal_kind=terminal_kind,
        input_request=input_request,
        input_request_ref=input_request_ref,
        outcome_submission_requested=outcome_submission_requested,
        publication_requested=publication_requested,
        hold_after_observations=hold_after_observations,
    )


def serve_once(
    request_line: str,
    output: TextIO,
    *,
    host: RuntimeHost | None = None,
    lease_authority: CanonicalLeaseAuthority | None = None,
    lease_binding: CanonicalLeaseBinding | None = None,
    checkpoint_authority: CheckpointAuthority | None = None,
    checkpoint_attestation: CheckpointAttestation | None = None,
    cancellation: CancellationSignal | None = None,
    cancellation_authority: CancellationAuthority | None = None,
    terminal_port: TerminalReconciliationPort | None = None,
    kernel: KernelPort | None = None,
    internal_failure_hook: InternalFailureHook | None = None,
) -> int:
    """Read one serialized invocation and write event/exit JSON lines."""

    run: RunSnapshot | None = None
    invocation: InvocationEnvelope | None = None
    collector: EventCollector | None = None
    try:
        request = json.loads(request_line)
        if not isinstance(request, dict):
            raise ContractError("service request must be an object")
        unknown = sorted(set(request).difference({"run", "invocation", "fakePlan"}))
        if unknown:
            raise ContractError(f"service request has unknown field(s): {', '.join(unknown)}")
        run = RunSnapshot.from_dict(request.get("run"))
        invocation = InvocationEnvelope.from_dict(request.get("invocation"))
        plan = _fake_plan(request.get("fakePlan"))
        if (
            host is None
            or lease_authority is None
            or lease_binding is None
            or terminal_port is None
        ):
            raise ContractError(
                "service requires separately injected host, canonical lease authority/binding, "
                "and terminal reconciliation port"
            )
    except (ContractError, TypeError, ValueError, json.JSONDecodeError):
        output.write(
            json.dumps(
                {"type": "error", "error": {"code": "invalid_request", "message": "invalid runtime request"}},
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        raise

    # Everything after parsing is execution.  In particular, ValueError and
    # TypeError from a kernel or host callback are runtime failures, not bad
    # request data.
    collector = EventCollector(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        expected_causation_ref=invocation.causation_ref,
    )
    phase = ExecutionPhase()

    def emit(event) -> None:
        collector.emit(event)
        output.write(json.dumps({"type": "event", "event": event.to_dict()}, sort_keys=True) + "\n")
        output.flush()

    try:
        exit_value = execute(
            run=run,
            invocation=invocation,
            host=host,
            emit=emit,
            cancellation=cancellation,
            cancellation_authority=cancellation_authority,
            kernel=kernel or FakeKernel(plan),
            lease_authority=lease_authority,
            lease_binding=lease_binding,
            checkpoint_authority=checkpoint_authority,
            checkpoint_attestation=checkpoint_attestation,
            terminal_port=terminal_port,
            execution_phase=phase,
        )
        output.write(json.dumps({"type": "exit", "exit": exit_value.to_dict()}, sort_keys=True) + "\n")
        output.flush()
        return 0
    except BindingError:
        output.write(
            json.dumps(
                {"type": "error", "error": {"code": "binding_rejected", "message": "runtime binding rejected"}},
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        return 1
    except TerminalReconciliationError as exc:
        return _handle_terminal_reconciliation_failure(
            exc=exc,
            run=run,
            invocation=invocation,
            collector=collector,
            output=output,
            internal_failure_hook=internal_failure_hook,
        )
    except (ContractError, TypeError, ValueError) as exc:
        if not phase.execution_started:
            output.write(
                json.dumps(
                    {
                        "type": "error",
                        "error": {"code": "binding_rejected", "message": "runtime binding rejected"},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            output.flush()
            return 1
        return _handle_runtime_failure(
            exc=exc,
            run=run,
            invocation=invocation,
            collector=collector,
            terminal_port=terminal_port,
            output=output,
            internal_failure_hook=internal_failure_hook,
        )
    except Exception as exc:
        return _handle_runtime_failure(
            exc=exc,
            run=run,
            invocation=invocation,
            collector=collector,
            terminal_port=terminal_port,
            output=output,
            internal_failure_hook=internal_failure_hook,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Plane Agent runtime invocation")
    parser.add_argument("--once", action="store_true", help="accept one JSON-lines invocation (the default)")
    args = parser.parse_args(argv)
    del args
    request_line = sys.stdin.readline()
    if not request_line:
        return 2
    try:
        request = json.loads(request_line)
        if not isinstance(request, dict):
            raise ContractError("service request must be an object")
        run = RunSnapshot.from_dict(request.get("run"))
        invocation = InvocationEnvelope.from_dict(request.get("invocation"))
        lease_binding = _demo_lease_binding(invocation)
        lease_authority = FixtureCanonicalLeaseAuthority(
            _DEMO_LEASE_BINDINGS, clock=lambda: datetime.now(timezone.utc)
        )
        checkpoint_attestation = _demo_checkpoint_attestation(invocation)
        checkpoint_authority = (
            FixtureCheckpointAuthority([checkpoint_attestation])
            if checkpoint_attestation is not None
            else None
        )
        return serve_once(
            request_line,
            sys.stdout,
            host=RecordingHost(),
            lease_authority=lease_authority,
            lease_binding=lease_binding,
            checkpoint_authority=checkpoint_authority,
            checkpoint_attestation=checkpoint_attestation,
            terminal_port=_DemoTerminalPort(),
        )
    except Exception:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
