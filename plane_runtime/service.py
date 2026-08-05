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
import signal
import sys
import threading
import time
from typing import BinaryIO, Callable, Protocol, TextIO

from .adapter import (
    CanonicalLeaseAuthority,
    CanonicalLeaseBinding,
    CheckpointAuthority,
    CheckpointAttestation,
    EventCollector,
    ExecutionPhase,
    KernelPort,
    CancellationAuthority,
    CancellationSignal,
    LeaseError,
    TerminalReconciliationRejected,
    TerminalReconciliationError,
    TerminalReconciliationPort,
    execute,
    execute_proposal_only,
    reconcile_terminal_proposal,
)
from .contract import (
    BoundsError,
    ContractError,
    BindingError,
    InvocationEnvelope,
    MAX_INVOCATION_BYTES,
    MAX_RUN_SNAPSHOT_BYTES,
    MAX_NEW_CONTEXT_EVENT_REFS,
    RuntimeFailure,
    RuntimeConfigurationError,
    RuntimeExit,
    RunSnapshot,
    TerminalProposal,
    _check_raw_wire_size,
)
from .host_port import UnixSocketPlaneHostPort
from .g1_bootstrap_contract import read_g1_bootstrap_frames
from .g1_contract import G1RunSnapshot


GENERIC_RUNTIME_FAILURE = "runtime execution failed; Plane reconciliation is required"
InternalFailureHook = Callable[[Exception], None]

# The request carries two independently bounded contracts.  The factor of
# three covers the largest UTF-8-to-JSON-escape expansion before the composite
# frame is decoded; the final allowance covers object framing.
MAX_SERVICE_REQUEST_BYTES = (
    3 * MAX_RUN_SNAPSHOT_BYTES
    + 3 * MAX_INVOCATION_BYTES
    + 16 * 1024
)
SERVICE_FRAME_TIMEOUT_SECONDS = 1.0
SERVICE_FRAME_READ_CHUNK_BYTES = 16 * 1024
SERVICE_FRAME_TERMINATOR_ALLOWANCE = 2


class _BootstrapCancellation:
    """Trusted parent signal state for one production child invocation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._previous: Any = None
        self._installed = False

    def _handler(self, _signum: int, _frame: Any) -> None:
        self._event.set()

    def __enter__(self) -> Callable[[], bool]:
        if not hasattr(signal, "SIGUSR1"):
            raise RuntimeConfigurationError("trusted cancellation signal is unavailable")
        self._previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handler)
        self._installed = True
        return self._event.is_set

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._installed:
            signal.signal(signal.SIGUSR1, self._previous)


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

        frame_limit = MAX_SERVICE_REQUEST_BYTES + SERVICE_FRAME_TERMINATOR_ALLOWANCE
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

            if len(self._carry) >= frame_limit:
                raise BoundsError(
                    f"service request exceeds {MAX_SERVICE_REQUEST_BYTES} UTF-8 bytes"
                )
            raw = self._read_chunk(
                deadline,
                frame_limit - len(self._carry),
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


class _CapturingTerminalPort:
    """Capture the one accepted receipt for the supervised process seam."""

    def __init__(self, delegate: TerminalReconciliationPort) -> None:
        self._delegate = delegate
        self.receipt = None

    def reconcile_terminal(self, proposal: TerminalProposal):
        receipt = self._delegate.reconcile_terminal(proposal)
        self.receipt = receipt
        return receipt


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


def _handle_terminal_reconciliation_rejection(
    *,
    exc: TerminalReconciliationRejected,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    collector: EventCollector,
    output: TextIO,
    internal_failure_hook: InternalFailureHook | None,
) -> int:
    """Tell the supervisor about a validated legal rejection, not an outage."""

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
                    "code": "terminal_reconciliation_rejected",
                    "message": "terminal reconciliation was legally rejected; supervisor action is required",
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
    emit_terminal_receipt: bool = False,
) -> int:
    """Trusted-host convenience path that reconciles through ``terminal_port``.

    The fixed container command never calls this function.  The production
    child path is :func:`serve_once_proposal_only`, which has no terminal port.
    The kernel is always injected; the production ``main`` path never
    constructs a fixture kernel.
    """

    run: RunSnapshot | None = None
    invocation: InvocationEnvelope | None = None
    collector: EventCollector | None = None
    receipt_capture = (
        _CapturingTerminalPort(terminal_port) if emit_terminal_receipt and terminal_port is not None else None
    )
    execution_terminal_port = receipt_capture or terminal_port
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
        unknown = sorted(set(request).difference({"run", "invocation"}))
        if unknown:
            raise ContractError(f"service request has unknown field(s): {', '.join(unknown)}")
        run = RunSnapshot.from_dict(request.get("run"))
        invocation = InvocationEnvelope.from_dict(request.get("invocation"))
        checkpoint_configuration_missing = invocation.checkpoint_ref is not None and (
            checkpoint_authority is None or checkpoint_attestation is None
        )
        if (
            lease_authority is None
            or lease_binding is None
            or terminal_port is None
            or kernel is None
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
    terminal_rejections: list[TerminalReconciliationRejected] = []
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
            kernel=kernel,
            lease_authority=lease_authority,
            lease_binding=lease_binding,
            checkpoint_authority=checkpoint_authority,
            checkpoint_attestation=checkpoint_attestation,
            terminal_port=execution_terminal_port,
            execution_phase=phase,
            _terminal_rejection_sink=terminal_rejections.append,
        )
        if terminal_rejections:
            if len(terminal_rejections) != 1:
                raise TerminalReconciliationError(
                    message="terminal reconciliation produced multiple rejection results",
                )
            return _handle_terminal_reconciliation_rejection(
                exc=terminal_rejections[0],
                run=run,
                invocation=invocation,
                collector=collector,
                output=output,
                internal_failure_hook=internal_failure_hook,
            )
        if receipt_capture is not None and receipt_capture.receipt is not None:
            output.write(
                json.dumps(
                    {"type": "reconciliation", "receipt": receipt_capture.receipt.to_dict()},
                    sort_keys=True,
                )
                + "\n"
            )
            output.flush()
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
            terminal_port=execution_terminal_port,
            output=output,
            internal_failure_hook=internal_failure_hook,
        )
    except Exception as exc:
        return _handle_runtime_failure(
            exc=exc,
            run=run,
            invocation=invocation,
            collector=collector,
            terminal_port=execution_terminal_port,
            output=output,
            internal_failure_hook=internal_failure_hook,
        )


def serve_once_proposal_only(
    request_line: str,
    output: TextIO,
    *,
    lease_authority: CanonicalLeaseAuthority | None,
    lease_binding: CanonicalLeaseBinding | None,
    checkpoint_authority: CheckpointAuthority | None = None,
    checkpoint_attestation: CheckpointAttestation | None = None,
    cancellation: CancellationSignal | None = None,
    kernel: KernelPort | None,
    internal_failure_hook: InternalFailureHook | None = None,
) -> int:
    """Run one child invocation and emit observations, proposal, and exit only."""

    run: RunSnapshot | None = None
    invocation: InvocationEnvelope | None = None
    collector: EventCollector | None = None
    proposals: list[TerminalProposal] = []
    try:
        if "\n" in request_line:
            if not request_line.endswith("\n") or request_line.count("\n") != 1:
                raise ContractError("service request must be one JSON line")
            request_line = request_line[:-1]
            if request_line.endswith("\r"):
                request_line = request_line[:-1]
        request_line = _check_raw_wire_size(request_line, "serviceRequest", MAX_SERVICE_REQUEST_BYTES)
        request = json.loads(request_line)
        if not isinstance(request, dict):
            raise ContractError("service request must be an object")
        unknown = sorted(set(request).difference({"run", "invocation"}))
        if unknown:
            raise ContractError(f"service request has unknown field(s): {', '.join(unknown)}")
        run = RunSnapshot.from_dict(request.get("run"))
        invocation = InvocationEnvelope.from_dict(request.get("invocation"))
        if kernel is None or lease_authority is None or lease_binding is None:
            raise RuntimeConfigurationError("proposal-only execution has no injected runtime binding")
        if invocation.checkpoint_ref is not None and (
            checkpoint_authority is None or checkpoint_attestation is None
        ):
            raise RuntimeConfigurationError("proposal-only continuation has no checkpoint binding")
    except (ContractError, TypeError, json.JSONDecodeError):
        output.write(json.dumps({"type": "error", "error": {"code": "invalid_request", "message": "invalid runtime request"}}, sort_keys=True) + "\n")
        output.flush()
        raise

    collector = EventCollector(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        expected_causation_ref=invocation.causation_ref,
    )

    def emit(event) -> None:
        collector.emit(event)
        output.write(json.dumps({"type": "event", "event": event.to_dict()}, sort_keys=True) + "\n")
        output.flush()

    try:
        exit_value = execute_proposal_only(
            run=run,
            invocation=invocation,
            emit=emit,
            cancellation=cancellation,
            kernel=kernel,
            lease_authority=lease_authority,
            lease_binding=lease_binding,
            checkpoint_authority=checkpoint_authority,
            checkpoint_attestation=checkpoint_attestation,
            execution_phase=ExecutionPhase(),
            proposal_sink=proposals.append,
        )
    except (BindingError, LeaseError):
        output.write(json.dumps({"type": "error", "error": {"code": "binding_rejected", "message": "runtime binding rejected"}}, sort_keys=True) + "\n")
        output.flush()
        return 1
    except RuntimeConfigurationError:
        output.write(json.dumps({"type": "error", "error": {"code": "runtime_configuration", "message": "runtime dependencies are not configured"}}, sort_keys=True) + "\n")
        output.flush()
        return 1
    except Exception as exc:
        if internal_failure_hook is not None:
            try:
                internal_failure_hook(exc)
            except Exception:
                pass
        proposals.append(
            TerminalProposal(
                run_id=run.run_id,
                invocation_id=invocation.invocation_id,
                actor_ref=run.actor_ref,
                workspace_ref=run.workspace_ref,
                snapshot_digest=run.digest(),
                kind="failed",
                final_sequence=collector.last_sequence,
                evidence_event_ids=tuple(event.event_id for event in collector.events[-MAX_NEW_CONTEXT_EVENT_REFS:]),
                failure=RuntimeFailure(
                    code="runtime_exception",
                    message="runtime execution failed; host reconciliation is required",
                    retryable=True,
                ),
            )
        )
        exit_value = RuntimeExit(
            kind="failed",
            final_sequence=collector.last_sequence,
            failure=RuntimeFailure(
                code="runtime_exception",
                message="runtime execution failed; host reconciliation is required",
                retryable=True,
            ),
        )
    if len(proposals) != 1:
        output.write(json.dumps({"type": "error", "error": {"code": "proposal_missing", "message": "runtime did not emit exactly one terminal proposal"}}, sort_keys=True) + "\n")
        output.flush()
        return 1
    output.write(json.dumps({"type": "proposal", "proposal": proposals[0].to_dict()}, sort_keys=True) + "\n")
    output.write(json.dumps({"type": "exit", "exit": exit_value.to_dict()}, sort_keys=True) + "\n")
    output.flush()
    return 0


class _ProductionBinding(Protocol):
    """Future real-kernel binding; no production instance is configured here."""

    kernel: KernelPort
    lease_authority: CanonicalLeaseAuthority
    lease_binding: CanonicalLeaseBinding
    checkpoint_authority: CheckpointAuthority | None
    checkpoint_attestation: CheckpointAttestation | None


_PRODUCTION_BINDING: _ProductionBinding | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Plane Agent runtime invocation")
    parser.add_argument("--once", action="store_true", help="accept one JSON-lines invocation (the default)")
    parser.add_argument("--g1-test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--g1-production", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--g1-bootstrap-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-call-allowance", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plane-host-socket", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Production is authoritative only through the trusted bootstrap.  The
    # marker is private parent-to-child wiring, not a second public entrypoint.
    if args.g1_production != args.g1_bootstrap_child:
        return 2
    if args.g1_bootstrap_child:
        frames = None
        host_port = None
        try:
            from .g1_service import serve_once_g1
            from .hermes_adapter import InlineCredentialSource

            with _BootstrapCancellation() as cancellation:
                frames = read_g1_bootstrap_frames(sys.stdin)
                request_line = frames.request.decode("utf-8")
                request = json.loads(request_line)
                snapshot = G1RunSnapshot.from_dict(request["run"])
                host_port = (
                    UnixSocketPlaneHostPort(args.plane_host_socket)
                    if args.plane_host_socket is not None
                    else None
                )
                source = InlineCredentialSource(frames.credentials, snapshot.model_provider)
                return serve_once_g1(
                    request_line,
                    sys.stdout,
                    production=True,
                    diagnostics=sys.stderr,
                    model_call_allowance=frames.model_call_allowance,
                    host_port=host_port,
                    credential_source=source,
                    cancellation=cancellation,
                )
        except Exception:
            return 2
        finally:
            if host_port is not None:
                host_port.close()
            if frames is not None:
                frames.clear()
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
        if not isinstance(request, dict) or set(request) != {"run", "invocation"}:
            raise ContractError("production runtime request has unsupported fields")
        # The accepted G1 contract is a separate, direct-frame path.  Keep the
        # older proposal-only service available for its existing callers while
        # making the new child process consume exact G1 values.
        run_value = request.get("run")
        if isinstance(run_value, dict) and {"profile", "runtimePolicy"}.issubset(run_value):
            if args.g1_test_only and args.g1_production:
                raise RuntimeConfigurationError("G1 execution mode is ambiguous")
            if not args.g1_test_only and not args.g1_production:
                raise RuntimeConfigurationError("G1 execution requires an attested supervisor mode")
            from .g1_service import serve_once_g1

            host_port = (
                UnixSocketPlaneHostPort(args.plane_host_socket)
                if args.plane_host_socket is not None
                else None
            )
            try:
                return serve_once_g1(
                    request_line,
                    sys.stdout,
                    production=args.g1_production,
                    diagnostics=sys.stderr if args.g1_production else None,
                    model_call_allowance=args.model_call_allowance,
                    host_port=host_port,
                )
            finally:
                if host_port is not None:
                    host_port.close()
        # Parsing proves the fixed command accepts arbitrary valid envelopes,
        # but it does not authorize execution.  A real KernelPort binding must
        # be installed by the future runtime service; fail closed otherwise.
        RunSnapshot.from_dict(request.get("run"))
        InvocationEnvelope.from_dict(request.get("invocation"))
        binding = _PRODUCTION_BINDING
        if binding is None:
            sys.stdout.write(json.dumps({"type": "error", "error": {"code": "runtime_configuration", "message": "no real kernel binding is configured"}}, sort_keys=True) + "\n")
            sys.stdout.flush()
            return 2
        return serve_once_proposal_only(
            request_line,
            sys.stdout,
            lease_authority=binding.lease_authority,
            lease_binding=binding.lease_binding,
            checkpoint_authority=binding.checkpoint_authority,
            checkpoint_attestation=binding.checkpoint_attestation,
            kernel=binding.kernel,
        )
    except Exception:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
