"""Minimal JSON-lines host for a separately spawned runtime process.

This is deliberately a transport-shaped seam, not a queue or RPC decision.
One process accepts one invocation request, streams validated events, and
returns one exit.  A supervisor can replace the process with a new invocation.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import selectors
import sys
import time
from datetime import datetime, timezone
from typing import Any, BinaryIO, Callable, TextIO

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
    CancellationAuthority,
    CancellationSignal,
    LeaseError,
    TerminalReconciliationError,
    TerminalReconciliationPort,
    execute,
    reconcile_terminal_proposal,
)
from .contract import (
    AssignmentSnapshot,
    BoundsError,
    ContractDigests,
    ContractError,
    BindingError,
    InvocationEnvelope,
    MAX_INVOCATION_BYTES,
    MAX_REFERENCE_LENGTH,
    MAX_RUN_SNAPSHOT_BYTES,
    MAX_TEXT_LENGTH,
    MAX_NEW_CONTEXT_EVENT_REFS,
    OperationDescriptor,
    RuntimeFailure,
    RuntimeConfigurationError,
    RuntimeModelRoute,
    RuntimeBudgetPolicy,
    RunSnapshot,
    RuntimeBudget,
    TerminalProposal,
    ToolPresentation,
    VersionedContextRef,
    _check_raw_wire_size,
)


GENERIC_RUNTIME_FAILURE = "runtime execution failed; Plane reconciliation is required"
InternalFailureHook = Callable[[Exception], None]

# The request carries two independently bounded contracts plus the bounded
# fake-kernel configuration used only by this deterministic service fixture.
# The factor of three covers the largest UTF-8-to-JSON-escape expansion before
# the composite frame is decoded; the final allowance covers object framing.
MAX_FAKE_PLAN_BYTES = 2 * MAX_TEXT_LENGTH + 2 * MAX_REFERENCE_LENGTH + 8 * 1024
MAX_SERVICE_REQUEST_BYTES = (
    3 * MAX_RUN_SNAPSHOT_BYTES
    + 3 * MAX_INVOCATION_BYTES
    + MAX_FAKE_PLAN_BYTES
    + 16 * 1024
)
SERVICE_FRAME_TIMEOUT_SECONDS = 1.0
SERVICE_FRAME_READ_CHUNK_BYTES = 16 * 1024


class _ServiceFrameReader:
    """Acquire bounded JSON-lines frames from a real stream without blocking forever."""

    def __init__(
        self,
        stream: TextIO | BinaryIO,
        *,
        timeout_seconds: float = SERVICE_FRAME_TIMEOUT_SECONDS,
        chunk_bytes: int = SERVICE_FRAME_READ_CHUNK_BYTES,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        self._owner = stream
        self._source: BinaryIO | TextIO = getattr(stream, "buffer", stream)
        self._timeout_seconds = float(timeout_seconds)
        self._chunk_bytes = chunk_bytes
        self._carry = bytearray()
        self._selector: selectors.BaseSelector | None = None
        self._fd: int | None = None
        try:
            self._fd = self._source.fileno()  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            self._fd = None
        if self._fd is not None:
            selector: selectors.BaseSelector | None = None
            try:
                selector = selectors.DefaultSelector()
                selector.register(self._source, selectors.EVENT_READ)
                self._selector = selector
            except (OSError, ValueError):
                if selector is not None:
                    selector.close()
                self._selector = None
                self._fd = None

    def __enter__(self) -> "_ServiceFrameReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        self._carry.clear()

    def _read_chunk(self, deadline: float, max_bytes: int) -> bytes:
        read_size = min(self._chunk_bytes, max_bytes)
        if read_size <= 0:
            raise BoundsError(
                f"service request exceeds {MAX_SERVICE_REQUEST_BYTES} UTF-8 bytes"
            )
        if self._selector is not None and self._fd is not None:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BoundsError("service request frame timed out")
                if not self._selector.select(timeout=remaining):
                    raise BoundsError("service request frame timed out")
                try:
                    return os.read(self._fd, read_size)
                except InterruptedError:
                    continue
                except BlockingIOError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        raise ContractError("service input stream could not be read") from exc
                    continue
                except OSError as exc:
                    raise ContractError("service input stream could not be read") from exc

        try:
            # Text-only test streams count characters in read(size), so cap
            # the character chunk at the worst-case UTF-8 expansion too.
            raw = self._source.read(max(1, read_size // 4))  # type: ignore[union-attr]
        except Exception as exc:
            raise ContractError("service input stream could not be read") from exc
        if isinstance(raw, str):
            try:
                return raw.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ContractError("service request must be valid UTF-8") from exc
        if isinstance(raw, bytes):
            return raw
        if not raw:
            return b""
        raise ContractError("service input stream returned an invalid frame")

    def read_frame(self) -> str:
        """Read one complete frame, retaining bytes belonging to later frames."""

        deadline = time.monotonic() + self._timeout_seconds
        while True:
            newline = self._carry.find(b"\n")
            if newline >= 0:
                frame = bytes(self._carry[:newline])
                del self._carry[: newline + 1]
                if frame.endswith(b"\r"):
                    frame = frame[:-1]
                if len(frame) > MAX_SERVICE_REQUEST_BYTES:
                    raise BoundsError(
                        f"service request exceeds {MAX_SERVICE_REQUEST_BYTES} UTF-8 bytes"
                    )
                try:
                    return frame.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ContractError("service request must be valid UTF-8") from exc

            if len(self._carry) > MAX_SERVICE_REQUEST_BYTES:
                raise BoundsError(
                    f"service request exceeds {MAX_SERVICE_REQUEST_BYTES} UTF-8 bytes"
                )
            raw = self._read_chunk(
                deadline,
                MAX_SERVICE_REQUEST_BYTES + 1 - len(self._carry),
            )
            if not raw:
                if not self._carry:
                    return ""
                raise BoundsError("service request frame is unterminated")
            self._carry.extend(raw)


def _read_bounded_request_line(
    stream: TextIO | BinaryIO,
    *,
    timeout_seconds: float = SERVICE_FRAME_TIMEOUT_SECONDS,
) -> str:
    """Read one bounded UTF-8 JSON-lines frame from a service stream."""

    reader = getattr(stream, "_plane_runtime_frame_reader", None)
    if (
        not isinstance(reader, _ServiceFrameReader)
        or reader._owner is not stream
        or reader._timeout_seconds != float(timeout_seconds)
    ):
        if isinstance(reader, _ServiceFrameReader):
            reader.close()
        reader = _ServiceFrameReader(stream, timeout_seconds=timeout_seconds)
        try:
            setattr(stream, "_plane_runtime_frame_reader", reader)
        except (AttributeError, TypeError):
            with reader:
                return reader.read_frame()
    return reader.read_frame()


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
        actor_ref=run.actor_ref,
        workspace_ref=run.workspace_ref,
        snapshot_digest=run.digest(),
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
        workspace_ref="workspace:one",
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
    fake_plan_size = len(
        json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if fake_plan_size > MAX_FAKE_PLAN_BYTES:
        raise ContractError("fakePlan exceeds its bounded service request surface")
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
    host: object | None = None,
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
        if "\n" in request_line:
            if not request_line.endswith("\n") or request_line.count("\n") != 1:
                raise ContractError("service request must be one JSON line")
            request_line = request_line[:-1]
            if request_line.endswith("\r"):
                request_line = request_line[:-1]
        request_line = _check_raw_wire_size(
            request_line, "serviceRequest", MAX_SERVICE_REQUEST_BYTES
        )
        request = json.loads(request_line)
        if not isinstance(request, dict):
            raise ContractError("service request must be an object")
        unknown = sorted(set(request).difference({"run", "invocation", "fakePlan"}))
        if unknown:
            raise ContractError(f"service request has unknown field(s): {', '.join(unknown)}")
        run = RunSnapshot.from_dict(request.get("run"))
        invocation = InvocationEnvelope.from_dict(request.get("invocation"))
        plan = _fake_plan(request.get("fakePlan"))
        checkpoint_configuration_missing = invocation.checkpoint_ref is not None and (
            checkpoint_authority is None or checkpoint_attestation is None
        )
        if (
            lease_authority is None
            or lease_binding is None
            or terminal_port is None
            or checkpoint_configuration_missing
        ):
            output.write(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "code": "runtime_configuration",
                            "message": "runtime dependencies are not configured",
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            output.flush()
            return 1
    except (ContractError, TypeError, json.JSONDecodeError):
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
    except (BindingError, LeaseError):
        output.write(
            json.dumps(
                {"type": "error", "error": {"code": "binding_rejected", "message": "runtime binding rejected"}},
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        return 1
    except RuntimeConfigurationError:
        output.write(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "code": "runtime_configuration",
                        "message": "runtime dependencies are not configured",
                    },
                },
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
    try:
        with _ServiceFrameReader(sys.stdin) as reader:
            request_line = reader.read_frame()
    except (BoundsError, ContractError):
        return 2
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
