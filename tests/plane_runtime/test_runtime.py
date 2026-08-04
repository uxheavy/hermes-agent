"""Focused contract, adapter, and process-boundary proof for plane_runtime."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch

from plane_runtime import (
    MAX_ACCEPTANCE_CRITERIA,
    MAX_CONTEXT_REFS,
    MAX_EAGER_OPERATIONS,
    MAX_INVOCATION_BYTES,
    MAX_NEW_CONTEXT_EVENT_REFS,
    MAX_RUN_SNAPSHOT_BYTES,
    MAX_EVENTS_PER_INVOCATION,
    MAX_EVENT_STREAM_BYTES,
    MAX_TRANSCRIPT_BYTES,
    MAX_TRANSCRIPT_OBSERVATIONS,
    MAX_OPTIONAL_EVENT_TAIL,
    MAX_ARTIFACT_PROPOSALS,
    MAX_INPUT_PROPOSALS,
    MAX_MESSAGE_PROPOSALS,
    MAX_TERMINAL_PROPOSAL_BYTES,
    MAX_TERMINAL_RECEIPT_BYTES,
    PROTOCOL,
    ArtifactObserved,
    ArtifactProposal,
    AssignmentSnapshot,
    BindingError,
    BoundsError,
    CancellationAuthorityReceipt,
    CanonicalLeaseBinding,
    CheckpointAttestation,
    ContractDigests,
    ContractError,
    EventCollector,
    FakeKernel,
    FakeKernelPlan,
    FixtureCanonicalLeaseAuthority,
    FixtureCancellationAuthority,
    FixtureCheckpointAuthority,
    FixtureTerminalReconciliationPort,
    InvocationEnvelope,
    InvocationTrigger,
    LeaseError,
    MutableCancellation,
    OperationDescriptor,
    OutcomeProposal,
    OutcomeSubmissionObserved,
    ProductReceipt,
    product_proof_identity,
    ProgressObserved,
    MessageProposal,
    MessageProposalObserved,
    RuntimeBudget,
    RuntimeBudgetPolicy,
    RuntimeConfigurationError,
    RuntimeEvent,
    RuntimeExit,
    RuntimeFailure,
    RuntimeLease,
    RuntimeModelRoute,
    RunSnapshot,
    SequenceError,
    TerminalProposal,
    TerminalProof,
    TerminalReconciliationError,
    TerminalReconciliationReceipt,
    ToolPresentation,
    TranscriptObserved,
    UsageObserved,
    VersionedContextRef,
    classify_process_death,
    execute,
    parse_utc_timestamp,
    reconcile_terminal_proposal,
    reconcile_process_death,
)
from plane_runtime.adapter import KernelObservation, KernelRequest, KernelResult
from plane_runtime.service import (
    MAX_SERVICE_REQUEST_BYTES,
    SERVICE_FRAME_READ_CHUNK_BYTES,
    _ServiceFrameReader,
    _read_bounded_request_line,
    main as service_main,
    serve_once,
)


TRUSTED_NOW = datetime(2026, 8, 4, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).parents[2]


def make_snapshot(*, total: RuntimeBudget | None = None) -> RunSnapshot:
    return RunSnapshot(
        protocol=PROTOCOL,
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
        total_budget_policy=RuntimeBudgetPolicy(total or RuntimeBudget(5, 100, 100)),
        contract_digests=ContractDigests("snapshot:v1", "invocation:v1", "event:v1", "exit:v1"),
    )


def make_invocation(
    snapshot: RunSnapshot,
    *,
    invocation_id: str = "invocation:one",
    trigger: InvocationTrigger | None = None,
    remaining: RuntimeBudget | None = None,
    checkpoint_ref: str | None = None,
    context_refs: tuple[str, ...] = (),
) -> InvocationEnvelope:
    return InvocationEnvelope(
        protocol=PROTOCOL,
        invocation_id=invocation_id,
        run_id=snapshot.run_id,
        run_snapshot_digest=snapshot.digest(),
        trigger=trigger or InvocationTrigger("initial"),
        new_context_event_refs=context_refs,
        checkpoint_ref=checkpoint_ref,
        remaining_budget=remaining or snapshot.total_budget_policy.total,
        lease=RuntimeLease("lease:one", "host:one", "2099-01-01T00:00:00Z"),
        causation_ref="cause:one",
        cancellation_ref="cancel:one",
    )


def product_event_resource(receipt: TerminalReconciliationReceipt) -> str:
    return next(
        item.resource_ref for item in receipt.proofs if item.proof_kind == "product_event"
    )


def typed_terminal_proofs(proposal: TerminalProposal) -> tuple[TerminalProof, ...]:
    prefixes = {
        "operation_attempt": "operation",
        "application": "application",
        "gateway": "gateway",
        "audit": "audit",
        "product_event": "product-event",
    }
    return tuple(
        TerminalProof(
            proof_kind=proof_kind,
            proof_ref=f"terminal-proof:{proof_kind}:{proposal.idempotency_key}",
            resource_ref=f"{prefix}:{proposal.idempotency_key}",
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            actor_ref=proposal.actor_ref,
            workspace_ref=proposal.workspace_ref,
            snapshot_digest=proposal.snapshot_digest,
            terminal_slot=proposal.idempotency_key,
            terminal_kind=proposal.kind,
            proposal_digest=proposal.digest(),
        )
        for proof_kind, prefix in prefixes.items()
    )


def cancellation_receipt(snapshot: RunSnapshot, invocation: InvocationEnvelope) -> CancellationAuthorityReceipt:
    idempotency_key = f"cancel:{snapshot.run_id}:{invocation.invocation_id}"
    return CancellationAuthorityReceipt(
        resource_ref=invocation.cancellation_ref,
        receipt_ref=f"cancel-receipt:{invocation.invocation_id}",
        run_id=snapshot.run_id,
        invocation_id=invocation.invocation_id,
        actor_ref=snapshot.actor_ref,
        workspace_ref=snapshot.workspace_ref,
        snapshot_digest=snapshot.digest(),
        idempotency_key=idempotency_key,
        gateway_receipt_ref=f"gateway:{idempotency_key}",
        audit_ref=f"audit:{idempotency_key}",
    )


def event(
    *,
    sequence: int,
    event_id: str = "event:one",
    idempotency_key: str | None = None,
    run_id: str = "run:one",
    correlation_ref: str = "cause:one",
) -> RuntimeEvent:
    return RuntimeEvent(
        protocol=PROTOCOL,
        run_id=run_id,
        invocation_id="invocation:one",
        sequence=sequence,
        event_id=event_id,
        correlation_ref=correlation_ref,
        idempotency_key=idempotency_key or event_id,
        body=ProgressObserved("observed"),
    )


def run_execute(*, snapshot: RunSnapshot, invocation: InvocationEnvelope, **kwargs) -> RuntimeExit:
    binding = kwargs.pop(
        "lease_binding",
        CanonicalLeaseBinding(
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            lease_id="lease:one",
            holder_ref="host:one",
            active=True,
            expires_at=invocation.lease.expires_at,
        ),
    )
    kwargs.setdefault(
        "lease_authority",
        FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
    )
    if invocation.checkpoint_ref is not None:
        attestation = kwargs.pop(
            "checkpoint_attestation",
            CheckpointAttestation(
                checkpoint_ref=invocation.checkpoint_ref,
                source_run_id=snapshot.run_id,
                source_invocation_id="invocation:one",
                snapshot_digest=snapshot.digest(),
                actor_ref=snapshot.actor_ref,
                profile_version=snapshot.profile_version,
                continuation_event_ref=invocation.trigger.event_ref or "event:unknown",
                continuation_trigger_kind=invocation.trigger.kind,
                allowed_target_invocation_id=invocation.invocation_id,
            ),
        )
        kwargs.setdefault("checkpoint_attestation", attestation)
        kwargs.setdefault("checkpoint_authority", FixtureCheckpointAuthority([attestation]))
    cancellation_receipt = CancellationAuthorityReceipt(
        resource_ref=invocation.cancellation_ref,
        receipt_ref=f"cancel-receipt:{invocation.invocation_id}",
        run_id=snapshot.run_id,
        invocation_id=invocation.invocation_id,
        actor_ref=snapshot.actor_ref,
        workspace_ref=snapshot.workspace_ref,
        snapshot_digest=snapshot.digest(),
        idempotency_key=f"cancel:{snapshot.run_id}:{invocation.invocation_id}",
        gateway_receipt_ref=f"gateway:cancel:{snapshot.run_id}:{invocation.invocation_id}",
        audit_ref=f"audit:cancel:{snapshot.run_id}:{invocation.invocation_id}",
    )
    kwargs.setdefault(
        "cancellation_authority",
        FixtureCancellationAuthority([cancellation_receipt]),
    )
    kwargs.setdefault("terminal_port", FixtureTerminalReconciliationPort())
    return execute(run=snapshot, invocation=invocation, lease_binding=binding, **kwargs)


def invoke_service(request: dict[str, object]) -> list[dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, "-m", "plane_runtime.service", "--once"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return [json.loads(line) for line in completed.stdout.splitlines()]


def serve_fixture(
    snapshot: RunSnapshot,
    invocation: InvocationEnvelope,
    *,
    kernel=None,
    host=None,
    terminal_port=None,
    cancellation=None,
    cancellation_authority=None,
) -> tuple[int, list[dict[str, object]], object]:
    binding = CanonicalLeaseBinding(
        snapshot.run_id,
        invocation.invocation_id,
        "lease:one",
        "host:one",
        True,
        invocation.lease.expires_at,
    )
    if cancellation_authority is None:
        cancellation_receipt = CancellationAuthorityReceipt(
            resource_ref=invocation.cancellation_ref,
            receipt_ref=f"cancel-receipt:{invocation.invocation_id}",
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            actor_ref=snapshot.actor_ref,
            workspace_ref=snapshot.workspace_ref,
            snapshot_digest=snapshot.digest(),
            idempotency_key=f"cancel:{snapshot.run_id}:{invocation.invocation_id}",
            gateway_receipt_ref=f"gateway:cancel:{snapshot.run_id}:{invocation.invocation_id}",
            audit_ref=f"audit:cancel:{snapshot.run_id}:{invocation.invocation_id}",
        )
        cancellation_authority = FixtureCancellationAuthority([cancellation_receipt])
    output = StringIO()
    status = serve_once(
        json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
        output,
        host=host,
        lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
        lease_binding=binding,
        cancellation=cancellation,
        cancellation_authority=cancellation_authority,
        terminal_port=terminal_port or FixtureTerminalReconciliationPort(),
        kernel=kernel,
    )
    return status, [json.loads(line) for line in output.getvalue().splitlines()], output


class ArtifactKernel:
    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, cancellation
        emit(
            KernelObservation(
                kind="artifact",
                artifact_ref="artifact:one",
                artifact_digest="sha256:artifact",
            )
        )
        emit(
            KernelObservation(
                kind="outcome_submission",
                submission_ref="submission:one",
                outcome_content="artifact outcome",
            )
        )
        return KernelResult("completed")


class ExplodingKernel:
    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, emit, cancellation
        raise RuntimeError("provider secret must stay internal")


class ValueErrorKernel:
    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, emit, cancellation
        raise ValueError("provider secret from ValueError kernel")


class TypeErrorKernel:
    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, emit, cancellation
        raise TypeError("provider secret from TypeError kernel")


class NoDispatchKernel:
    def __init__(self) -> None:
        self.dispatched = False

    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, emit, cancellation
        self.dispatched = True
        raise AssertionError("kernel dispatch was not allowed")


class CancelAfterProgressKernel:
    def __init__(self, cancellation: MutableCancellation) -> None:
        self.cancellation = cancellation
        self.dispatched = False

    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, cancellation
        self.dispatched = True
        emit(KernelObservation(kind="progress", message="before cancellation"))
        self.cancellation.cancel()
        return KernelResult("completed")


class ManyEvidenceKernel:
    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, cancellation
        emit(
            KernelObservation(
                kind="outcome_submission",
                submission_ref="submission:mandatory",
                outcome_content="mandatory outcome content",
            )
        )
        for index in range(160):
            emit(
                KernelObservation(
                    kind="artifact",
                    artifact_ref=f"artifact:{index}",
                    artifact_digest=f"sha256:{index}",
                )
            )
            emit(KernelObservation(kind="progress", message=f"optional progress {index}"))
        return KernelResult("completed")


class TenThousandProgressKernel:
    def __init__(self) -> None:
        self.attempted = 0

    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, cancellation
        for index in range(10_000):
            self.attempted += 1
            emit(KernelObservation(kind="progress", message=f"progress:{index}"))
        return KernelResult("completed")


class MultibyteTranscriptKernel:
    def __init__(self) -> None:
        self.attempted = 0

    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, cancellation
        for index in range(100):
            self.attempted += 1
            emit(
                KernelObservation(
                    kind="transcript",
                    transcript_ref=f"transcript:{index}",
                    text="é" * 6_000,
                )
            )
        return KernelResult("completed")


class RaisingLeaseAuthority:
    def validate_lease(self, *, run, invocation, binding) -> None:
        del run, invocation, binding
        raise ValueError("lease secret must stay internal")


class RaisingCheckpointAuthority:
    def claim_checkpoint(self, *, run, invocation, attestation) -> None:
        del run, invocation, attestation
        raise RuntimeError("checkpoint secret must stay internal")


class RaisingCancellationAuthority:
    def validate_cancellation(self, *, run, invocation) -> ProductReceipt:
        del run, invocation
        raise TypeError("cancellation secret must stay internal")


class RaisingCancellationSignal:
    def is_cancelled(self) -> bool:
        raise ValueError("cancellation signal secret must stay internal")


class RecordingCancellationSignal:
    def __init__(self, calls: list[str], cancelled: bool) -> None:
        self.calls = calls
        self.cancelled = cancelled

    def is_cancelled(self) -> bool:
        self.calls.append("signal")
        return self.cancelled


class RecordingLeaseAuthority:
    def __init__(self, calls: list[str], delegate) -> None:
        self.calls = calls
        self.delegate = delegate

    def validate_lease(self, *, run, invocation, binding) -> None:
        self.calls.append("lease")
        self.delegate.validate_lease(run=run, invocation=invocation, binding=binding)


class RecordingCheckpointAuthority:
    def __init__(self, calls: list[str], delegate) -> None:
        self.calls = calls
        self.delegate = delegate

    def claim_checkpoint(self, *, run, invocation, attestation) -> None:
        self.calls.append("checkpoint")
        self.delegate.claim_checkpoint(
            run=run, invocation=invocation, attestation=attestation
        )


class RecordingCancellationAuthority:
    def __init__(self, calls: list[str], delegate) -> None:
        self.calls = calls
        self.delegate = delegate

    def validate_cancellation(self, *, run, invocation):
        self.calls.append("cancellation-authority")
        return self.delegate.validate_cancellation(run=run, invocation=invocation)


class CountingLeaseAuthority:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0

    def validate_lease(self, *, run, invocation, binding) -> None:
        self.calls += 1
        self.delegate.validate_lease(run=run, invocation=invocation, binding=binding)


class CountingCheckpointAuthority:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0

    def claim_checkpoint(self, *, run, invocation, attestation) -> None:
        self.calls += 1
        self.delegate.claim_checkpoint(
            run=run, invocation=invocation, attestation=attestation
        )


class CountingCancellationAuthority:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls = 0

    def validate_cancellation(self, *, run, invocation):
        self.calls += 1
        return self.delegate.validate_cancellation(run=run, invocation=invocation)


class ForgedReceiptPort:
    """Return one forged accepted receipt without applying any product mutation."""

    def __init__(self, forge) -> None:
        self.forge = forge
        self.calls = 0
        self.product_events: list[str] = []

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        self.calls += 1
        receipt = FixtureTerminalReconciliationPort().reconcile_terminal(proposal)
        return self.forge(receipt)


class RejectingTerminalPort:
    def __init__(self) -> None:
        self.calls = 0
        self.proposals: list[TerminalProposal] = []
        self.product_events: list[str] = []

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        self.calls += 1
        self.proposals.append(proposal)
        return TerminalReconciliationReceipt(
            receipt_ref=f"rejected:{proposal.idempotency_key}",
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            kind=proposal.kind,
            idempotency_key=proposal.idempotency_key,
            accepted=False,
            legal_transition=False,
        )


class FailingTerminalPort:
    def __init__(self) -> None:
        self.calls = 0

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        del proposal
        self.calls += 1
        raise ValueError("terminal provider secret")


class SharedTerminalPort:
    """Thread/process-safe durable-like fake for terminal reconciliation tests."""

    def __init__(self, store=None, lock=None, accepted=None, path: Path | None = None) -> None:
        self.store = store if store is not None else {}
        self.lock = lock if lock is not None else threading.RLock()
        self.accepted = accepted if accepted is not None else []
        self.path = path
        self.product_events: list[str] = []
        self.product_receipts: list[ProductReceipt] = []

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        key = proposal.idempotency_key
        if self.path is not None:
            return self._reconcile_file(proposal)
        with self.lock:
            prior = self.store.get(key)
            if prior is not None:
                prior_proposal, prior_receipt = prior
                if prior_proposal != proposal:
                    return TerminalReconciliationReceipt(
                        receipt_ref=f"rejected:{key}",
                        run_id=proposal.run_id,
                        invocation_id=proposal.invocation_id,
                        kind=proposal.kind,
                        idempotency_key=key,
                        accepted=False,
                        legal_transition=False,
                    )
                return prior_receipt
            receipt = self._new_receipt(proposal)
            if not receipt.accepted:
                return receipt
            self.store[key] = (proposal, receipt)
            self.accepted.append(key)
            self.product_events.append(product_event_resource(receipt))
            self.product_receipts.extend(receipt.product_receipts)
            return receipt

    def _new_receipt(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        key = proposal.idempotency_key
        product_event_ref = f"product-event:{key}"
        if proposal.kind == "completed":
            assert proposal.outcome_proposal is not None
            required = [("outcome_submission", proposal.outcome_proposal.submission_ref)]
        elif proposal.kind == "waiting_for_input":
            assert proposal.input_request_proposal is not None
            required = [("input_request", proposal.input_request_proposal.request_ref)]
        else:
            required = [("terminal_event", product_event_ref)]
        required.extend(("artifact", item.artifact_ref) for item in proposal.artifact_proposals)
        required.extend(("message", item.message_ref) for item in proposal.message_proposals)
        product_receipts = tuple(
            self._product_receipt(proposal, kind=kind, resource_ref=resource_ref)
            for kind, resource_ref in required
        )
        return TerminalReconciliationReceipt(
            receipt_ref=f"receipt:{key}",
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            kind=proposal.kind,
            idempotency_key=key,
            accepted=True,
            legal_transition=True,
            proofs=typed_terminal_proofs(proposal),
            product_receipts=product_receipts,
        )

    @staticmethod
    def _product_receipt(
        proposal: TerminalProposal, *, kind: str, resource_ref: str
    ) -> ProductReceipt:
        receipt_ref, idempotency_key = product_proof_identity(
            proof_kind=kind,
            product_kind=kind,
            resource_ref=resource_ref,
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            actor_ref=proposal.actor_ref,
            workspace_ref=proposal.workspace_ref,
            snapshot_digest=proposal.snapshot_digest,
            terminal_slot=proposal.idempotency_key,
            terminal_kind=proposal.kind,
            proposal_digest=proposal.digest(),
        )
        return ProductReceipt(
            resource_ref=resource_ref,
            receipt_ref=receipt_ref,
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            idempotency_key=idempotency_key,
            kind=kind,
            actor_ref=proposal.actor_ref,
            workspace_ref=proposal.workspace_ref,
            snapshot_digest=proposal.snapshot_digest,
            terminal_slot=proposal.idempotency_key,
            proposal_digest=proposal.digest(),
            terminal_kind=proposal.kind,
        )

    def _reconcile_file(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        assert self.path is not None
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            content = handle.read().strip()
            state = json.loads(content) if content else {}
            key = proposal.idempotency_key
            prior = state.get(key)
            if prior is not None:
                if prior["proposal"] != proposal.to_dict():
                    return TerminalReconciliationReceipt(
                        receipt_ref=f"rejected:{key}",
                        run_id=proposal.run_id,
                        invocation_id=proposal.invocation_id,
                        kind=proposal.kind,
                        idempotency_key=key,
                        accepted=False,
                        legal_transition=False,
                    )
                return TerminalReconciliationReceipt.from_dict(prior["receipt"])
            receipt = self._new_receipt(proposal)
            if receipt.accepted:
                state[key] = {"proposal": proposal.to_dict(), "receipt": receipt.to_dict()}
                handle.seek(0)
                handle.truncate()
                json.dump(state, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                self.accepted.append(key)
                self.product_events.append(product_event_resource(receipt))
                self.product_receipts.extend(receipt.product_receipts)
            return receipt


class RuntimeContractTests(unittest.TestCase):
    def test_run_snapshot_is_deeply_immutable(self) -> None:
        criteria = ["one"]
        context = [VersionedContextRef("context:one", "digest")]
        base = make_snapshot()
        snapshot = replace(
            base,
            assignment=AssignmentSnapshot("v1", "target", "objective", criteria),
            context=context,
        )

        criteria.append("mutated outside")
        context.append(VersionedContextRef("context:two", "digest"))
        self.assertEqual(snapshot.assignment.acceptance_criteria, ("one",))
        self.assertEqual(snapshot.context, (VersionedContextRef("context:one", "digest"),))
        with self.assertRaises((AttributeError, TypeError)):
            snapshot.run_id = "mutated"  # type: ignore[misc]
        wire = snapshot.to_dict()
        wire["assignment"]["acceptanceCriteria"].append("mutated wire")
        self.assertEqual(snapshot.assignment.acceptance_criteria, ("one",))

    def test_contracts_round_trip_through_json(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transcript_event = RuntimeEvent(
            PROTOCOL,
            snapshot.run_id,
            invocation.invocation_id,
            1,
            "event:transcript",
            invocation.causation_ref,
            "event:transcript",
            TranscriptObserved("transcript:one", "final text is evidence"),
        )
        exit_value = RuntimeExit(
            "failed",
            1,
            RuntimeFailure("process_died", "the replaceable process exited", True),
        )

        self.assertEqual(RunSnapshot.from_json(snapshot.to_json()), snapshot)
        self.assertEqual(InvocationEnvelope.from_json(invocation.to_json()), invocation)
        self.assertEqual(RuntimeEvent.from_json(transcript_event.to_json()), transcript_event)
        self.assertEqual(RuntimeExit.from_json(exit_value.to_json()), exit_value)

        outcome = OutcomeProposal(
            "submission:one",
            "final outcome",
            "event:outcome",
            "receipt:outcome",
        )
        terminal = TerminalProposal(
            snapshot.run_id,
            invocation.invocation_id,
            snapshot.actor_ref,
            snapshot.workspace_ref,
            snapshot.digest(),
            "completed",
            2,
            evidence_event_ids=(outcome.event_id,),
            evidence_receipt_refs=(outcome.proposal_receipt_ref,),
            outcome_proposal=outcome,
        )
        outcome_receipt_ref, outcome_idempotency_key = product_proof_identity(
            proof_kind="outcome_submission",
            product_kind="outcome_submission",
            resource_ref="submission:one",
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            actor_ref=snapshot.actor_ref,
            workspace_ref=snapshot.workspace_ref,
            snapshot_digest=snapshot.digest(),
            terminal_slot=terminal.idempotency_key,
            terminal_kind="completed",
            proposal_digest=terminal.digest(),
        )
        receipt = TerminalReconciliationReceipt(
            receipt_ref="terminal-receipt:one",
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            kind="completed",
            idempotency_key=terminal.idempotency_key,
            accepted=True,
            legal_transition=True,
            proofs=tuple(
                TerminalProof(
                    proof_kind=proof_kind,
                    proof_ref=f"terminal-proof:{proof_kind}:{terminal.idempotency_key}",
                    resource_ref=f"{resource_prefix}:{terminal.idempotency_key}",
                    run_id=snapshot.run_id,
                    invocation_id=invocation.invocation_id,
                    actor_ref=snapshot.actor_ref,
                    workspace_ref=snapshot.workspace_ref,
                    snapshot_digest=snapshot.digest(),
                    terminal_slot=terminal.idempotency_key,
                    terminal_kind="completed",
                    proposal_digest=terminal.digest(),
                )
                for proof_kind, resource_prefix in (
                    ("operation_attempt", "operation"),
                    ("application", "application"),
                    ("gateway", "gateway"),
                    ("audit", "audit"),
                    ("product_event", "product-event"),
                )
            ),
            product_receipts=(
                ProductReceipt(
                    "submission:one",
                    outcome_receipt_ref,
                    snapshot.run_id,
                    invocation.invocation_id,
                    outcome_idempotency_key,
                    kind="outcome_submission",
                    actor_ref=snapshot.actor_ref,
                    workspace_ref=snapshot.workspace_ref,
                    snapshot_digest=snapshot.digest(),
                    terminal_slot=terminal.idempotency_key,
                    proposal_digest=terminal.digest(),
                    terminal_kind="completed",
                ),
            ),
        )
        self.assertEqual(TerminalProposal.from_json(terminal.to_json()), terminal)
        self.assertEqual(TerminalReconciliationReceipt.from_json(receipt.to_json()), receipt)

    def test_all_wire_parsers_reject_unknown_fields_in_nested_objects(self) -> None:
        snapshot = make_snapshot().to_dict()
        snapshot_cases = [
            ({**snapshot, "forged": True}, RunSnapshot),
            ({**snapshot, "assignment": {**snapshot["assignment"], "forged": True}}, RunSnapshot),
            ({**snapshot, "context": [{**snapshot["context"][0], "forged": True}]}, RunSnapshot),
            (
                {
                    **snapshot,
                    "toolPresentation": {
                        **snapshot["toolPresentation"],
                        "eagerOperations": [{**snapshot["toolPresentation"]["eagerOperations"][0], "forged": True}],
                    },
                },
                RunSnapshot,
            ),
            ({**snapshot, "model": {**snapshot["model"], "forged": True}}, RunSnapshot),
            (
                {
                    **snapshot,
                    "totalBudgetPolicy": {
                        "total": {**snapshot["totalBudgetPolicy"]["total"], "forged": True}
                    },
                },
                RunSnapshot,
            ),
            (
                {
                    **snapshot,
                    "contractDigests": {**snapshot["contractDigests"], "forged": True},
                },
                RunSnapshot,
            ),
        ]
        for payload, parser in snapshot_cases:
            with self.subTest(parser=parser.__name__, payload=payload):
                with self.assertRaises(ContractError):
                    parser.from_dict(payload)

        invocation = make_invocation(make_snapshot()).to_dict()
        invocation_cases = [
            ({**invocation, "forged": True}, InvocationEnvelope),
            ({**invocation, "trigger": {**invocation["trigger"], "forged": True}}, InvocationEnvelope),
            ({**invocation, "lease": {**invocation["lease"], "forged": True}}, InvocationEnvelope),
            (
                {
                    **invocation,
                    "remainingBudget": {**invocation["remainingBudget"], "forged": True},
                },
                InvocationEnvelope,
            ),
        ]
        for payload, parser in invocation_cases:
            with self.subTest(parser=parser.__name__, payload=payload):
                with self.assertRaises(ContractError):
                    parser.from_dict(payload)

        runtime_event = event(sequence=1).to_dict()
        event_cases = [
            {**runtime_event, "forged": True},
            {**runtime_event, "body": {**runtime_event["body"], "forged": True}},
        ]
        for payload in event_cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ContractError):
                    RuntimeEvent.from_dict(payload)

        with self.assertRaises(ContractError):
            RuntimeExit.from_dict({**RuntimeExit("completed", 0).to_dict(), "forged": True})
        with self.assertRaises(ContractError):
            RuntimeFailure.from_dict({"code": "x", "message": "x", "retryable": False, "forged": True})

    def test_event_bodies_require_product_receipts(self) -> None:
        input_body = {"kind": "input_request", "requestRef": "input:one", "prompt": "answer"}
        artifact_body = {"kind": "artifact", "artifactRef": "artifact:one", "digest": "sha256:x"}
        for body in (input_body, artifact_body):
            with self.subTest(body=body):
                with self.assertRaises(ContractError):
                    RuntimeEvent.from_dict(
                        {
                            **event(sequence=1).to_dict(),
                            "body": body,
                        }
                    )

    def test_later_input_is_an_event_reference_and_does_not_mutate_snapshot(self) -> None:
        snapshot = make_snapshot()
        before = snapshot.to_json()
        invocation = make_invocation(
            snapshot,
            trigger=InvocationTrigger("human_input", "event:answer"),
            context_refs=("event:answer",),
            remaining=RuntimeBudget(4, 90, 90),
        )
        kernel = FakeKernel()
        run_execute(snapshot=snapshot, invocation=invocation, kernel=kernel)

        self.assertEqual(snapshot.to_json(), before)
        self.assertEqual(invocation.trigger.event_ref, "event:answer")
        self.assertNotIn("answer content", invocation.to_json())
        self.assertEqual(kernel.requests[0].new_context_event_refs, ("event:answer",))

    def test_runtime_total_budget_cap_is_enforced_and_forwarded(self) -> None:
        snapshot = make_snapshot(total=RuntimeBudget(5, 100, 100))
        invocation = make_invocation(snapshot, remaining=RuntimeBudget(3, 90, 80))
        kernel = FakeKernel(FakeKernelPlan(usage=RuntimeBudget(3, 90, 80)))
        self.assertEqual(run_execute(snapshot=snapshot, invocation=invocation, kernel=kernel).kind, "completed")
        self.assertEqual(kernel.requests[0].remaining_budget, RuntimeBudget(3, 90, 80))
        with self.assertRaises(ContractError):
            run_execute(
                snapshot=snapshot,
                invocation=make_invocation(snapshot, remaining=RuntimeBudget(6, 100, 100)),
            )

    def test_runtime_does_not_claim_later_plane_cumulative_accounting(self) -> None:
        snapshot = make_snapshot(total=RuntimeBudget(5, 100, 100))
        first = make_invocation(snapshot, remaining=RuntimeBudget(5, 100, 100))
        second = make_invocation(
            snapshot,
            invocation_id="invocation:two",
            trigger=InvocationTrigger("continuation", "event:continuation"),
            remaining=RuntimeBudget(3, 90, 80),
        )
        first_kernel = FakeKernel(FakeKernelPlan(usage=RuntimeBudget(2, 10, 20)))
        second_kernel = FakeKernel(FakeKernelPlan(usage=RuntimeBudget(3, 90, 80)))
        self.assertEqual(run_execute(snapshot=snapshot, invocation=first, kernel=first_kernel).kind, "completed")
        self.assertEqual(run_execute(snapshot=snapshot, invocation=second, kernel=second_kernel).kind, "completed")
        self.assertEqual(second_kernel.requests[0].remaining_budget, RuntimeBudget(3, 90, 80))

    def test_event_stream_binds_correlation_and_idempotency(self) -> None:
        collector = EventCollector(
            run_id="run:one",
            invocation_id="invocation:one",
            expected_correlation_ref="cause:one",
        )
        first = event(sequence=1, idempotency_key="idem:one")
        self.assertTrue(collector.accept(first))
        self.assertFalse(collector.accept(first))
        with self.assertRaises(BindingError):
            collector.accept(event(sequence=2, event_id="event:two", correlation_ref="forged"))
        with self.assertRaises(SequenceError):
            collector.accept(event(sequence=2, event_id="event:two", idempotency_key="idem:one"))
        with self.assertRaises(SequenceError):
            collector.accept(event(sequence=3, event_id="event:three"))
        with self.assertRaises(SequenceError):
            collector.accept(event(sequence=2, event_id="event:one", idempotency_key="idem:changed"))

    def test_event_stream_accepts_maximum_count_then_stops_without_map_growth(self) -> None:
        collector = EventCollector(
            run_id="run:one",
            invocation_id="invocation:one",
            expected_correlation_ref="cause:one",
        )
        for sequence in range(1, MAX_EVENTS_PER_INVOCATION + 1):
            self.assertTrue(
                collector.accept(
                    event(
                        sequence=sequence,
                        event_id=f"event:{sequence}",
                        idempotency_key=f"idem:{sequence}",
                    )
                )
            )
        with self.assertRaises(BoundsError):
            collector.accept(
                event(
                    sequence=MAX_EVENTS_PER_INVOCATION + 1,
                    event_id="event:overflow",
                    idempotency_key="idem:overflow",
                )
            )
        self.assertEqual(collector.event_count, MAX_EVENTS_PER_INVOCATION)
        self.assertEqual(collector.retained_sequence_entries, MAX_EVENTS_PER_INVOCATION)
        self.assertEqual(collector.retained_idempotency_entries, MAX_EVENTS_PER_INVOCATION)
        self.assertLessEqual(collector.retained_event_count, MAX_OPTIONAL_EVENT_TAIL)
        self.assertLessEqual(collector.stream_bytes, MAX_EVENT_STREAM_BYTES)

    def test_hostile_ingestion_reconciles_once_and_retains_only_bounded_state(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:10k-events")
        collector = EventCollector(
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            expected_correlation_ref=invocation.causation_ref,
        )
        kernel = TenThousandProgressKernel()
        port = FixtureTerminalReconciliationPort()
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=invocation,
            kernel=kernel,
            emit=collector.emit,
            terminal_port=port,
        )
        self.assertEqual(exit_value.kind, "failed")
        self.assertEqual(exit_value.failure.code, "ingestion_bounds")  # type: ignore[union-attr]
        self.assertLess(kernel.attempted, 10_000)
        self.assertEqual(collector.event_count, MAX_EVENTS_PER_INVOCATION)
        self.assertLessEqual(collector.retained_event_count, MAX_OPTIONAL_EVENT_TAIL)
        self.assertLessEqual(collector.retained_sequence_entries, MAX_EVENTS_PER_INVOCATION)
        self.assertLessEqual(collector.retained_idempotency_entries, MAX_EVENTS_PER_INVOCATION)
        self.assertLessEqual(collector.stream_bytes, MAX_EVENT_STREAM_BYTES)
        self.assertEqual(port.product_events, [(port.accepted[0], "failed")])
        self.assertEqual(len(port.receipts), 1)
        self.assertEqual(
            port.receipts[0].product_receipts[0].resource_ref,
            product_event_resource(port.receipts[0]),
        )

    def test_multibyte_transcript_limit_fails_before_unbounded_retention(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:multibyte")
        collector = EventCollector(
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            expected_correlation_ref=invocation.causation_ref,
        )
        kernel = MultibyteTranscriptKernel()
        port = FixtureTerminalReconciliationPort()
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=invocation,
            kernel=kernel,
            emit=collector.emit,
            terminal_port=port,
        )
        self.assertEqual(exit_value.kind, "failed")
        self.assertEqual(exit_value.failure.code, "ingestion_bounds")  # type: ignore[union-attr]
        self.assertLess(kernel.attempted, 100)
        self.assertLessEqual(
            collector.category_counts.get("transcript", 0), MAX_TRANSCRIPT_OBSERVATIONS
        )
        self.assertLessEqual(
            collector.category_bytes.get("transcript", 0), MAX_TRANSCRIPT_BYTES
        )
        self.assertLessEqual(collector.stream_bytes, MAX_EVENT_STREAM_BYTES)
        self.assertEqual(port.product_events, [(port.accepted[0], "failed")])

    def test_adapter_rejects_binding_mismatch(self) -> None:
        snapshot = make_snapshot()
        wrong_run = replace(make_invocation(snapshot), run_id="run:other")
        with self.assertRaises(BindingError):
            run_execute(snapshot=snapshot, invocation=wrong_run)

        collector = EventCollector(run_id=snapshot.run_id, invocation_id="invocation:one")
        with self.assertRaises(BindingError):
            collector.accept(event(sequence=1, run_id="run:other"))

    def test_cancellation_is_invocation_scoped(self) -> None:
        cancellation = MutableCancellation()
        cancellation.cancel()
        snapshot = make_snapshot()
        events: list[RuntimeEvent] = []
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot),
            emit=events.append,
            cancellation=cancellation,
        )
        self.assertEqual(exit_value, RuntimeExit("cancelled", 0))
        self.assertEqual(events, [])

        cancellation = MutableCancellation()
        kernel = FakeKernel(on_step=cancellation.cancel)
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot, invocation_id="invocation:cancelled"),
            cancellation=cancellation,
            kernel=kernel,
        )
        self.assertEqual(exit_value.kind, "cancelled")
        self.assertEqual(exit_value.final_sequence, 1)

    def test_cancellation_authority_receipt_is_lossless_and_fully_bound(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:cancellation-roundtrip")
        authority_receipt = cancellation_receipt(snapshot, invocation)
        self.assertEqual(
            CancellationAuthorityReceipt.from_json(authority_receipt.to_json()),
            authority_receipt,
        )

        cancellation = MutableCancellation()
        cancellation.cancel()
        port = FixtureTerminalReconciliationPort()
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=invocation,
            cancellation=cancellation,
            cancellation_authority=FixtureCancellationAuthority([authority_receipt]),
            terminal_port=port,
        )
        self.assertEqual(exit_value.kind, "cancelled")
        proposal = port.proposals[0]
        self.assertEqual(
            TerminalProposal.from_json(proposal.to_json()),
            proposal,
        )
        self.assertEqual(proposal.cancellation_receipt, authority_receipt)
        self.assertEqual(len(port.product_events), 1)

        wrong_values = {
            "run_id": "run:forged",
            "invocation_id": "invocation:forged",
            "actor_ref": "agent:forged",
            "workspace_ref": "workspace:forged",
            "snapshot_digest": "digest:forged",
            "resource_ref": "cancel:forged",
            "idempotency_key": "cancel:forged",
            "gateway_receipt_ref": "gateway:forged",
            "audit_ref": "audit:forged",
        }
        for field_name, value in wrong_values.items():
            with self.subTest(field_name=field_name):
                forged = replace(authority_receipt, **{field_name: value})
                forged_port = FixtureTerminalReconciliationPort()
                with self.assertRaises(BindingError):
                    run_execute(
                        snapshot=snapshot,
                        invocation=invocation,
                        cancellation=cancellation,
                        cancellation_authority=FixtureCancellationAuthority([forged]),
                        terminal_port=forged_port,
                    )
                self.assertEqual(forged_port.product_events, [])

        wrong_kind = replace(authority_receipt)
        object.__setattr__(wrong_kind, "kind", "terminal_event")
        with self.assertRaises(BindingError):
            run_execute(
                snapshot=snapshot,
                invocation=invocation,
                cancellation=cancellation,
                cancellation_authority=FixtureCancellationAuthority([wrong_kind]),
                terminal_port=FixtureTerminalReconciliationPort(),
            )
        missing_kind = authority_receipt.to_dict()
        del missing_kind["kind"]
        with self.assertRaises(ContractError):
            CancellationAuthorityReceipt.from_dict(missing_kind)

    def test_canonical_lease_authority_binds_all_fields_atomically(self) -> None:
        self.assertEqual(parse_utc_timestamp("2026-08-04T00:00:00Z"), datetime(2026, 8, 4, tzinfo=timezone.utc))
        for timestamp in ("2026-08-04T00:00:00+00:00", "2026-08-04T00:00:00", "2026-08-04T01:00:00+01:00"):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ContractError):
                    parse_utc_timestamp(timestamp)

        snapshot = make_snapshot()
        expired = replace(
            make_invocation(snapshot),
            lease=RuntimeLease("lease:one", "host:one", "2026-08-04T00:00:00Z"),
        )
        with self.assertRaises(LeaseError):
            run_execute(snapshot=snapshot, invocation=expired)
        with self.assertRaises(LeaseError):
            execute(
                run=snapshot,
                invocation=make_invocation(snapshot),
                terminal_port=SharedTerminalPort(),
            )

        invocation = make_invocation(snapshot)
        binding = CanonicalLeaseBinding(
            "run:one", invocation.invocation_id, "lease:one", "host:one", True, invocation.lease.expires_at
        )
        authority = FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW)
        with self.assertRaises(BindingError):
            execute(
                run=snapshot,
                invocation=replace(invocation, lease=RuntimeLease("lease:forged", "host:one", invocation.lease.expires_at)),
                lease_authority=authority,
                lease_binding=binding,
                terminal_port=SharedTerminalPort(),
            )

    def test_initial_checkpoint_is_rejected_and_continuation_requires_supervisor_attestation(self) -> None:
        snapshot = make_snapshot()
        with self.assertRaises(ContractError):
            make_invocation(snapshot, checkpoint_ref="checkpoint:forged")
        continuation = make_invocation(
            snapshot,
            trigger=InvocationTrigger("recoverable_restart", "event:restart"),
            checkpoint_ref="checkpoint:one",
            context_refs=("event:restart",),
            remaining=RuntimeBudget(3, 80, 80),
        )
        with self.assertRaises(ContractError):
            binding = CanonicalLeaseBinding(
                "run:one", continuation.invocation_id, "lease:one", "host:one", True, continuation.lease.expires_at
            )
            execute(
                run=snapshot,
                invocation=continuation,
                lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
                lease_binding=binding,
                terminal_port=SharedTerminalPort(),
            )
        kernel = FakeKernel()
        run_execute(
            snapshot=snapshot,
            invocation=continuation,
            kernel=kernel,
        )
        self.assertEqual(kernel.requests[0].checkpoint_ref, "checkpoint:one")
        with self.assertRaises(BindingError):
            valid_attestation = CheckpointAttestation(
                "checkpoint:one",
                "run:one",
                "invocation:one",
                snapshot.digest(),
                snapshot.actor_ref,
                snapshot.profile_version,
                "event:restart",
                "recoverable_restart",
                continuation.invocation_id,
            )
            run_execute(
                snapshot=snapshot,
                invocation=replace(continuation, checkpoint_ref="checkpoint:forged"),
                checkpoint_authority=FixtureCheckpointAuthority([valid_attestation]),
                checkpoint_attestation=valid_attestation,
            )

    def test_checkpoint_attestation_cannot_replay_across_runs_or_targets(self) -> None:
        snapshot = make_snapshot()
        continuation = make_invocation(
            snapshot,
            invocation_id="invocation:target-one",
            trigger=InvocationTrigger("continuation", "event:answer"),
            checkpoint_ref="checkpoint:one",
        )
        attestation = CheckpointAttestation(
            checkpoint_ref="checkpoint:one",
            source_run_id=snapshot.run_id,
            source_invocation_id="invocation:source",
            snapshot_digest=snapshot.digest(),
            actor_ref=snapshot.actor_ref,
            profile_version=snapshot.profile_version,
            continuation_event_ref="event:answer",
            continuation_trigger_kind="continuation",
            allowed_target_invocation_id=continuation.invocation_id,
        )
        authority = FixtureCheckpointAuthority([attestation])
        other_snapshot = replace(snapshot, run_id="run:two")
        other_invocation = replace(
            continuation,
            run_id=other_snapshot.run_id,
            run_snapshot_digest=other_snapshot.digest(),
        )
        with self.assertRaises(BindingError):
            run_execute(
                snapshot=other_snapshot,
                invocation=other_invocation,
                checkpoint_authority=authority,
                checkpoint_attestation=attestation,
            )
        other_target = replace(continuation, invocation_id="invocation:target-two")
        with self.assertRaises(BindingError):
            run_execute(
                snapshot=snapshot,
                invocation=other_target,
                checkpoint_authority=authority,
                checkpoint_attestation=attestation,
            )

    def test_checkpoint_claim_is_single_use_and_every_attestation_field_is_bound(self) -> None:
        snapshot = make_snapshot()
        continuation = make_invocation(
            snapshot,
            invocation_id="invocation:target-one",
            trigger=InvocationTrigger("continuation", "event:answer"),
            checkpoint_ref="checkpoint:one",
        )
        attestation = CheckpointAttestation(
            checkpoint_ref="checkpoint:one",
            source_run_id=snapshot.run_id,
            source_invocation_id="invocation:source",
            snapshot_digest=snapshot.digest(),
            actor_ref=snapshot.actor_ref,
            profile_version=snapshot.profile_version,
            continuation_event_ref="event:answer",
            continuation_trigger_kind="continuation",
            allowed_target_invocation_id=continuation.invocation_id,
        )
        mutations = {
            "checkpoint_ref": "checkpoint:two",
            "source_run_id": "run:two",
            "source_invocation_id": "invocation:other-source",
            "snapshot_digest": "digest:forged",
            "actor_ref": "agent:forged",
            "profile_version": "profile:forged",
            "continuation_event_ref": "event:forged",
            "continuation_trigger_kind": "recoverable_restart",
            "allowed_target_invocation_id": "invocation:target-two",
        }
        for field_name, value in mutations.items():
            with self.subTest(field_name=field_name):
                authority = FixtureCheckpointAuthority([attestation])
                with self.assertRaises(BindingError):
                    run_execute(
                        snapshot=snapshot,
                        invocation=continuation,
                        checkpoint_authority=authority,
                        checkpoint_attestation=replace(attestation, **{field_name: value}),
                    )

        authority = FixtureCheckpointAuthority([attestation])
        kernel = FakeKernel()
        run_execute(
            snapshot=snapshot,
            invocation=continuation,
            checkpoint_authority=authority,
            checkpoint_attestation=attestation,
            kernel=kernel,
        )
        with self.assertRaises(SequenceError):
            run_execute(
                snapshot=snapshot,
                invocation=continuation,
                checkpoint_authority=authority,
                checkpoint_attestation=attestation,
                kernel=kernel,
            )
        self.assertEqual(len(kernel.requests), 1)

    def test_signalled_cancellation_preflights_authority_before_checkpoint_claim(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(
            snapshot,
            invocation_id="invocation:cancel-checkpoint",
            trigger=InvocationTrigger("continuation", "event:answer"),
            checkpoint_ref="checkpoint:retry",
        )
        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            invocation.invocation_id,
            "lease:one",
            "host:one",
            True,
            invocation.lease.expires_at,
        )
        calls: list[str] = []
        lease_authority = RecordingLeaseAuthority(
            calls,
            FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
        )
        attestation = CheckpointAttestation(
            checkpoint_ref=invocation.checkpoint_ref,
            source_run_id=snapshot.run_id,
            source_invocation_id="invocation:source",
            snapshot_digest=snapshot.digest(),
            actor_ref=snapshot.actor_ref,
            profile_version=snapshot.profile_version,
            continuation_event_ref="event:answer",
            continuation_trigger_kind="continuation",
            allowed_target_invocation_id=invocation.invocation_id,
        )
        checkpoint_authority = RecordingCheckpointAuthority(
            calls, FixtureCheckpointAuthority([attestation])
        )
        cancellation_signal = RecordingCancellationSignal(calls, True)
        cancellation_authority = RecordingCancellationAuthority(
            calls,
            FixtureCancellationAuthority([cancellation_receipt(snapshot, invocation)]),
        )

        with self.assertRaises(RuntimeConfigurationError):
            execute(
                run=snapshot,
                invocation=invocation,
                cancellation=cancellation_signal,
                lease_authority=lease_authority,
                lease_binding=binding,
                checkpoint_authority=checkpoint_authority,
                checkpoint_attestation=attestation,
                terminal_port=FixtureTerminalReconciliationPort(),
            )
        self.assertEqual(calls, ["signal"])

        service_calls: list[str] = []
        service_signal = RecordingCancellationSignal(service_calls, True)
        output = StringIO()
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            cancellation=service_signal,
            lease_authority=RecordingLeaseAuthority(
                service_calls,
                FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
            ),
            lease_binding=binding,
            checkpoint_authority=RecordingCheckpointAuthority(
                service_calls, FixtureCheckpointAuthority([attestation])
            ),
            checkpoint_attestation=attestation,
            terminal_port=FixtureTerminalReconciliationPort(),
        )
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "runtime_configuration")
        self.assertEqual(service_calls, ["signal"])

        exit_value = execute(
            run=snapshot,
            invocation=invocation,
            cancellation=cancellation_signal,
            cancellation_authority=cancellation_authority,
            lease_authority=lease_authority,
            lease_binding=binding,
            checkpoint_authority=checkpoint_authority,
            checkpoint_attestation=attestation,
            terminal_port=FixtureTerminalReconciliationPort(),
        )
        self.assertEqual(exit_value.kind, "cancelled")
        self.assertEqual(calls, ["signal", "signal", "lease", "checkpoint", "cancellation-authority"])

    def test_unsignalled_cancellation_does_not_require_or_call_authority(self) -> None:
        snapshot = make_snapshot()
        calls: list[str] = []
        cancellation = RecordingCancellationSignal(calls, False)
        exit_value = execute(
            run=snapshot,
            invocation=make_invocation(snapshot, invocation_id="invocation:no-cancel-authority"),
            cancellation=cancellation,
            lease_authority=RecordingLeaseAuthority(
                calls,
                FixtureCanonicalLeaseAuthority(
                    [CanonicalLeaseBinding(
                        snapshot.run_id,
                        "invocation:no-cancel-authority",
                        "lease:one",
                        "host:one",
                        True,
                        "2099-01-01T00:00:00Z",
                    )],
                    clock=lambda: TRUSTED_NOW,
                ),
            ),
            lease_binding=CanonicalLeaseBinding(
                snapshot.run_id,
                "invocation:no-cancel-authority",
                "lease:one",
                "host:one",
                True,
                "2099-01-01T00:00:00Z",
            ),
            terminal_port=FixtureTerminalReconciliationPort(),
        )
        self.assertEqual(exit_value.kind, "completed")
        self.assertEqual(calls[:3], ["signal", "lease", "signal"])
        self.assertGreaterEqual(calls.count("signal"), 3)
        self.assertNotIn("cancellation-authority", calls)

    def test_cancellation_signal_exception_is_evaluated_before_authority_mutation(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(
            snapshot,
            invocation_id="invocation:raising-signal-checkpoint",
            trigger=InvocationTrigger("continuation", "event:answer"),
            checkpoint_ref="checkpoint:signal",
        )
        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            invocation.invocation_id,
            "lease:one",
            "host:one",
            True,
            invocation.lease.expires_at,
        )
        attestation = CheckpointAttestation(
            checkpoint_ref=invocation.checkpoint_ref,
            source_run_id=snapshot.run_id,
            source_invocation_id="invocation:source",
            snapshot_digest=snapshot.digest(),
            actor_ref=snapshot.actor_ref,
            profile_version=snapshot.profile_version,
            continuation_event_ref="event:answer",
            continuation_trigger_kind="continuation",
            allowed_target_invocation_id=invocation.invocation_id,
        )
        calls: list[str] = []
        with self.assertRaises(ContractError):
            execute(
                run=snapshot,
                invocation=invocation,
                cancellation=RaisingCancellationSignal(),
                cancellation_authority=RecordingCancellationAuthority(
                    calls,
                    FixtureCancellationAuthority([cancellation_receipt(snapshot, invocation)]),
                ),
                lease_authority=RecordingLeaseAuthority(
                    calls,
                    FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
                ),
                lease_binding=binding,
                checkpoint_authority=RecordingCheckpointAuthority(
                    calls, FixtureCheckpointAuthority([attestation])
                ),
                checkpoint_attestation=attestation,
                terminal_port=FixtureTerminalReconciliationPort(),
            )
        self.assertEqual(calls, [])

    def test_runtime_configuration_is_checked_before_authority_and_retry_keeps_checkpoint(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(
            snapshot,
            invocation_id="invocation:configured-retry",
            trigger=InvocationTrigger("continuation", "event:answer"),
            checkpoint_ref="checkpoint:retry",
        )
        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            invocation.invocation_id,
            "lease:one",
            "host:one",
            True,
            invocation.lease.expires_at,
        )
        lease_authority = CountingLeaseAuthority(
            FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW)
        )
        attestation = CheckpointAttestation(
            checkpoint_ref=invocation.checkpoint_ref,
            source_run_id=snapshot.run_id,
            source_invocation_id="invocation:source",
            snapshot_digest=snapshot.digest(),
            actor_ref=snapshot.actor_ref,
            profile_version=snapshot.profile_version,
            continuation_event_ref="event:answer",
            continuation_trigger_kind="continuation",
            allowed_target_invocation_id=invocation.invocation_id,
        )
        checkpoint_authority = CountingCheckpointAuthority(
            FixtureCheckpointAuthority([attestation])
        )
        cancellation = cancellation_receipt(snapshot, invocation)
        cancellation_authority = CountingCancellationAuthority(
            FixtureCancellationAuthority([cancellation])
        )

        with self.assertRaises(ContractError):
            execute(
                run=snapshot,
                invocation=invocation,
                lease_authority=lease_authority,
                lease_binding=binding,
                checkpoint_authority=checkpoint_authority,
                checkpoint_attestation=attestation,
                cancellation_authority=cancellation_authority,
                terminal_port=None,
            )
        self.assertEqual(lease_authority.calls, 0)
        self.assertEqual(checkpoint_authority.calls, 0)
        self.assertEqual(cancellation_authority.calls, 0)

        exit_value = execute(
            run=snapshot,
            invocation=invocation,
            lease_authority=lease_authority,
            lease_binding=binding,
            checkpoint_authority=checkpoint_authority,
            checkpoint_attestation=attestation,
            cancellation_authority=cancellation_authority,
            terminal_port=FixtureTerminalReconciliationPort(),
        )
        self.assertEqual(exit_value.kind, "completed")
        self.assertEqual(lease_authority.calls, 1)
        self.assertEqual(checkpoint_authority.calls, 1)
        self.assertEqual(cancellation_authority.calls, 0)

    def test_input_requests_are_authorized_receipt_correlated_and_terminally_consistent(self) -> None:
        snapshot = make_snapshot()
        port = FixtureTerminalReconciliationPort()
        events: list[RuntimeEvent] = []
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot),
            emit=events.append,
            terminal_port=port,
            kernel=FakeKernel(FakeKernelPlan(terminal_kind="waiting_for_input", input_request="Need one answer")),
        )
        self.assertEqual(exit_value.kind, "waiting_for_input")
        input_events = [item for item in events if item.body.to_dict()["kind"] == "input_request"]
        self.assertEqual(len(input_events), 1)
        self.assertTrue(input_events[0].body.to_dict()["receiptRef"].startswith("proposal:"))
        self.assertEqual(port.product_events, [("terminal:run:one:invocation:one", "waiting_for_input")])
        self.assertEqual(port.receipts[0].product_receipts[0].resource_ref, "input:fake")
        self.assertEqual(port.product_event_payloads[0]["kind"], "InputRequest")

        with self.assertRaises(ContractError):
            run_execute(
                snapshot=snapshot,
                invocation=make_invocation(snapshot, invocation_id="invocation:completed-input"),
                kernel=FakeKernel(FakeKernelPlan(terminal_kind="completed", input_request="Need one answer")),
            )
        with self.assertRaises(ContractError):
            run_execute(
                snapshot=snapshot,
                invocation=make_invocation(snapshot, invocation_id="invocation:no-input"),
                kernel=FakeKernel(FakeKernelPlan(terminal_kind="waiting_for_input")),
            )

    def test_artifacts_are_authorized_receipt_correlated_and_not_a_second_mutation_path(self) -> None:
        snapshot = make_snapshot()
        port = FixtureTerminalReconciliationPort()
        events: list[RuntimeEvent] = []
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot),
            emit=events.append,
            terminal_port=port,
            kernel=ArtifactKernel(),
        )
        self.assertEqual(exit_value.kind, "completed")
        artifact = [item for item in events if isinstance(item.body, ArtifactObserved)]
        self.assertEqual(len(artifact), 1)
        self.assertTrue(artifact[0].body.receipt_ref.startswith("proposal:"))
        self.assertEqual(port.product_events, [("terminal:run:one:invocation:one", "completed")])
        receipt = port.receipts[0]
        self.assertEqual(
            [item.resource_ref for item in receipt.product_receipts],
            ["submission:one", "artifact:one"],
        )
        self.assertEqual(port.product_event_payloads[0]["kind"], "OutcomeSubmission")
        self.assertEqual(port.product_event_payloads[0]["content"], "artifact outcome")

    def test_transcript_and_message_proposals_are_non_product_evidence(self) -> None:
        snapshot = make_snapshot()
        events: list[RuntimeEvent] = []
        port = FixtureTerminalReconciliationPort()
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot),
            emit=events.append,
            kernel=FakeKernel(FakeKernelPlan(publication_requested=False)),
            terminal_port=port,
        )
        self.assertEqual(exit_value.kind, "completed")
        self.assertTrue(any(isinstance(item.body, TranscriptObserved) for item in events))
        self.assertFalse(any(item.body.to_dict()["kind"] == "conversation_publication" for item in events))

        events = []
        port = RejectingTerminalPort()
        with self.assertRaises(ContractError):
            run_execute(
                snapshot=snapshot,
                invocation=make_invocation(snapshot, invocation_id="invocation:published"),
                emit=events.append,
                kernel=FakeKernel(FakeKernelPlan(publication_requested=True)),
                terminal_port=port,
            )
        publication = [item for item in events if item.body.to_dict()["kind"] == "message_proposal"]
        self.assertEqual(len(publication), 1)
        self.assertTrue(publication[0].body.to_dict()["proposalReceiptRef"].startswith("proposal:"))
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(port.product_events, [])

        port = FixtureTerminalReconciliationPort()
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot, invocation_id="invocation:terminal-message"),
            kernel=FakeKernel(FakeKernelPlan(publication_requested=True)),
            terminal_port=port,
        )
        self.assertEqual(exit_value.kind, "completed")
        proposal = port.proposals[0]
        self.assertEqual(len(proposal.message_proposals), 1)
        self.assertEqual(
            [item.resource_ref for item in port.receipts[0].product_receipts],
            ["submission:fake", "transcript:fake"],
        )
        self.assertEqual(port.product_event_payloads[0]["messageRefs"], ["transcript:fake"])

    def test_product_receipt_identity_rejects_hostile_substitutions_without_retry(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:product-proof")
        canonical_port = FixtureTerminalReconciliationPort()
        run_execute(
            snapshot=snapshot,
            invocation=invocation,
            kernel=ArtifactKernel(),
            terminal_port=canonical_port,
        )
        canonical_receipt = canonical_port.receipts[0]
        self.assertEqual(len(canonical_receipt.product_receipts), 2)
        proposal = canonical_port.proposals[0]
        for field_name, value in {
            "receipt_ref": "product-receipt:forged",
            "idempotency_key": "product-idempotency:forged",
            "run_id": "run:forged",
            "invocation_id": "invocation:forged",
            "kind": "message",
            "resource_ref": "resource:forged",
            "actor_ref": "agent:forged",
            "workspace_ref": "workspace:forged",
            "snapshot_digest": "snapshot:forged",
            "terminal_slot": "terminal:forged",
            "proposal_digest": "proposal:forged",
            "terminal_kind": "failed",
        }.items():
            with self.subTest(field_name=field_name):
                forged_port = ForgedReceiptPort(
                    lambda receipt, field_name=field_name, value=value: replace(
                        receipt,
                        product_receipts=(
                            replace(receipt.product_receipts[0], **{field_name: value}),
                            *receipt.product_receipts[1:],
                        ),
                    )
                )
                with self.assertRaises(TerminalReconciliationError):
                    reconcile_terminal_proposal(
                        port=forged_port,
                        proposal=proposal,
                    )
                self.assertEqual(forged_port.calls, 1)
                self.assertEqual(forged_port.product_events, [])

        swapped_port = ForgedReceiptPort(
            lambda receipt: replace(
                receipt,
                product_receipts=(
                    receipt.product_receipts[1],
                    receipt.product_receipts[0],
                ),
            )
        )
        with self.assertRaises(TerminalReconciliationError):
            reconcile_terminal_proposal(port=swapped_port, proposal=proposal)
        self.assertEqual(swapped_port.calls, 1)

        shared_identity_port = ForgedReceiptPort(
            lambda receipt: replace(
                receipt,
                product_receipts=(
                    receipt.product_receipts[0],
                    replace(
                        receipt.product_receipts[1],
                        idempotency_key=receipt.product_receipts[0].idempotency_key,
                    ),
                ),
            )
        )
        with self.assertRaises(TerminalReconciliationError):
            reconcile_terminal_proposal(port=shared_identity_port, proposal=proposal)
        self.assertEqual(shared_identity_port.calls, 1)

    def test_completed_requires_outcome_receipt_not_progress_usage_or_transcript_evidence(self) -> None:
        with self.assertRaises(ContractError):
            run_execute(
                snapshot=make_snapshot(),
                invocation=make_invocation(make_snapshot(), invocation_id="invocation:evidence-only"),
                kernel=FakeKernel(FakeKernelPlan(outcome_submission_requested=False)),
            )

    def test_terminal_slot_reconciles_runtime_and_supervisor_competition_once(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:terminal-slot")
        port = SharedTerminalPort()
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=invocation,
            terminal_port=port,
        )
        self.assertEqual(exit_value.kind, "completed")
        process_death = reconcile_process_death(
            port=port,
            run=snapshot,
            invocation_id=invocation.invocation_id,
            final_sequence=exit_value.final_sequence,
        )
        replay = reconcile_process_death(
            port=port,
            run=snapshot,
            invocation_id=invocation.invocation_id,
            final_sequence=exit_value.final_sequence,
        )
        self.assertFalse(process_death.accepted)
        self.assertFalse(process_death.legal_transition)
        self.assertEqual(process_death, replay)
        self.assertEqual(len(port.accepted), 1)
        self.assertEqual(
            TerminalProposal(
                snapshot.run_id,
                invocation.invocation_id,
                snapshot.actor_ref,
                snapshot.workspace_ref,
                snapshot.digest(),
                "failed",
                exit_value.final_sequence,
                failure=RuntimeFailure("process_died", "runtime process exited before returning an exit", True),
                source="supervisor",
            ).idempotency_key,
            process_death.idempotency_key,
        )

    def test_supervisor_first_rejects_late_runtime_product_mutation(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:supervisor-first")
        port = FixtureTerminalReconciliationPort()
        supervisor_receipt = reconcile_process_death(
            port=port,
            run=snapshot,
            invocation_id=invocation.invocation_id,
            final_sequence=0,
        )
        self.assertTrue(supervisor_receipt.accepted)
        with self.assertRaises(ContractError):
            run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port)
        self.assertEqual(port.accepted, ["terminal:run:one:invocation:supervisor-first"])
        self.assertEqual(port.product_events, [(port.accepted[0], "failed")])
        self.assertEqual(len(port.receipts), 1)
        self.assertEqual(
            port.receipts[0].product_receipts[0].resource_ref,
            product_event_resource(port.receipts[0]),
        )

    def test_synchronized_runtime_and_supervisor_race_has_one_product_event(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:synchronized")
        port = FixtureTerminalReconciliationPort()
        start = threading.Barrier(2)
        results: list[tuple[str, bool]] = []
        errors: list[Exception] = []

        def runtime() -> None:
            try:
                start.wait(timeout=5)
                run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port)
                results.append(("runtime", True))
            except ContractError:
                results.append(("runtime", False))
            except Exception as exc:  # pragma: no cover - makes a race failure visible
                errors.append(exc)

        def supervisor() -> None:
            try:
                start.wait(timeout=5)
                receipt = reconcile_process_death(
                    port=port,
                    run=snapshot,
                    invocation_id=invocation.invocation_id,
                    final_sequence=0,
                )
                results.append(("supervisor", receipt.accepted))
            except Exception as exc:  # pragma: no cover - makes a race failure visible
                errors.append(exc)

        threads = [threading.Thread(target=runtime), threading.Thread(target=supervisor)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertEqual({name for name, _ in results}, {"runtime", "supervisor"})
        self.assertEqual(sum(accepted for _, accepted in results), 1)
        self.assertEqual(len(port.accepted), 1)
        self.assertEqual(len(port.product_events), 1)
        self.assertEqual(
            port.receipts[0].product_receipts[0].resource_ref,
            product_event_resource(port.receipts[0]),
        )

    def test_exact_runtime_replay_returns_the_same_atomic_receipt(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:replay")
        port = FixtureTerminalReconciliationPort()
        first = run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port)
        second = run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port)
        self.assertEqual(first, second)
        self.assertEqual(len(port.accepted), 1)
        self.assertEqual(len(port.product_events), 1)
        self.assertEqual(port.reconcile_terminal(port.proposals[0]), port.receipts[0])

    def test_accepted_receipt_proof_is_complete_and_exact(self) -> None:
        def missing_application(receipt: TerminalReconciliationReceipt):
            proofs = tuple(
                item for item in receipt.proofs if item.proof_kind != "application"
            )
            object.__setattr__(receipt, "proofs", proofs)
            return receipt

        def wrong_product_kind(receipt: TerminalReconciliationReceipt):
            product = receipt.product_receipts[0]
            object.__setattr__(product, "kind", "forged_kind")
            return receipt

        def extra_product_receipt(receipt: TerminalReconciliationReceipt):
            object.__setattr__(receipt, "product_receipts", receipt.product_receipts * 2)
            return receipt

        def wrong_binding(receipt: TerminalReconciliationReceipt):
            proof = next(item for item in receipt.proofs if item.proof_kind == "audit")
            object.__setattr__(proof, "actor_ref", "agent:forged")
            return receipt

        def forged_proposal_digest(receipt: TerminalReconciliationReceipt):
            proof = next(item for item in receipt.proofs if item.proof_kind == "product_event")
            object.__setattr__(proof, "proposal_digest", "digest:forged")
            return receipt

        for index, forge in enumerate(
            (
                missing_application,
                wrong_product_kind,
                extra_product_receipt,
                wrong_binding,
                forged_proposal_digest,
            )
        ):
            with self.subTest(forge=forge.__name__):
                snapshot = make_snapshot()
                invocation = make_invocation(snapshot, invocation_id=f"invocation:forged:{index}")
                port = ForgedReceiptPort(forge)
                status, lines, _ = serve_fixture(
                    snapshot,
                    invocation,
                    terminal_port=port,
                )
                self.assertEqual(status, 1)
                self.assertEqual(port.calls, 1)
                self.assertEqual(port.product_events, [])
                self.assertEqual(lines[-1]["type"], "reconciliation_request")

        def replace_proof(receipt: TerminalReconciliationReceipt, proof_kind: str, **changes):
            proofs = tuple(
                replace(item, **changes) if item.proof_kind == proof_kind else item
                for item in receipt.proofs
            )
            object.__setattr__(receipt, "proofs", proofs)
            return receipt

        for proof_kind in (
            "operation_attempt",
            "application",
            "gateway",
            "audit",
            "product_event",
        ):
            with self.subTest(proof_kind=proof_kind):
                snapshot = make_snapshot()
                invocation = make_invocation(snapshot, invocation_id=f"invocation:replace:{proof_kind}")
                port = ForgedReceiptPort(
                    lambda receipt, proof_kind=proof_kind: replace_proof(
                        receipt, proof_kind, resource_ref="resource:forged"
                    )
                )
                status, lines, _ = serve_fixture(snapshot, invocation, terminal_port=port)
                self.assertEqual(status, 1)
                self.assertEqual(port.calls, 1)
                self.assertEqual(port.product_events, [])
                self.assertEqual(lines[-1]["type"], "reconciliation_request")

        for field_name, forged_value in {
            "run_id": "run:forged",
            "invocation_id": "invocation:forged",
            "actor_ref": "agent:forged",
            "workspace_ref": "workspace:forged",
            "snapshot_digest": "snapshot:forged",
            "terminal_slot": "terminal:forged",
            "terminal_kind": "failed",
            "proposal_digest": "digest:forged",
        }.items():
            with self.subTest(proof_field=field_name):
                snapshot = make_snapshot()
                invocation = make_invocation(snapshot, invocation_id=f"invocation:field:{field_name}")

                def forge_field(receipt, field_name=field_name, forged_value=forged_value):
                    proof = next(
                        item for item in receipt.proofs if item.proof_kind == "audit"
                    )
                    object.__setattr__(proof, field_name, forged_value)
                    return receipt

                port = ForgedReceiptPort(forge_field)
                status, lines, _ = serve_fixture(snapshot, invocation, terminal_port=port)
                self.assertEqual(status, 1)
                self.assertEqual(port.calls, 1)
                self.assertEqual(port.product_events, [])
                self.assertEqual(lines[-1]["type"], "reconciliation_request")

        for forge_name, forge in (
            (
                "duplicate_shared_resource",
                lambda receipt: replace_proof(
                    receipt,
                    "application",
                    resource_ref=next(
                        item.resource_ref
                        for item in receipt.proofs
                        if item.proof_kind == "operation_attempt"
                    ),
                ),
            ),
            (
                "wrong_terminal_kind",
                lambda receipt: replace_proof(
                    receipt, "audit", terminal_kind="failed"
                ),
            ),
            (
                "wrong_proof_kind",
                lambda receipt: replace_proof(
                    receipt, "audit", proof_kind="gateway"
                ),
            ),
        ):
            with self.subTest(forge_name=forge_name):
                snapshot = make_snapshot()
                invocation = make_invocation(snapshot, invocation_id=f"invocation:{forge_name}")
                port = ForgedReceiptPort(forge)
                status, lines, _ = serve_fixture(snapshot, invocation, terminal_port=port)
                self.assertEqual(status, 1)
                self.assertEqual(port.calls, 1)
                self.assertEqual(port.product_events, [])
                self.assertEqual(lines[-1]["type"], "reconciliation_request")

    def test_finalized_terminal_proposal_preserves_required_evidence_under_overflow(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:overflow")
        port = FixtureTerminalReconciliationPort()
        collector = EventCollector(
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            expected_correlation_ref=invocation.causation_ref,
        )
        self.assertEqual(
            run_execute(
                snapshot=snapshot,
                invocation=invocation,
                terminal_port=port,
                kernel=ManyEvidenceKernel(),
                emit=collector.emit,
            ).kind,
            "failed",
        )
        self.assertTrue(
            any(isinstance(item.body, OutcomeSubmissionObserved) for item in collector.events)
        )
        self.assertLessEqual(
            collector.retained_event_count,
            MAX_OPTIONAL_EVENT_TAIL + 1 + MAX_ARTIFACT_PROPOSALS,
        )
        proposal = port.proposals[0]
        self.assertLessEqual(len(proposal.evidence_event_ids), 128)
        self.assertLessEqual(len(proposal.evidence_receipt_refs), 128)
        self.assertLessEqual(len(proposal.artifact_proposals), 128)
        self.assertEqual(proposal.failure.code, "ingestion_bounds")  # type: ignore[union-attr]
        self.assertEqual(len(port.product_event_payloads), 1)
        self.assertEqual(port.product_event_payloads[0]["kind"], "TerminalFailure")
        self.assertLessEqual(len(port.receipts[0].product_receipts), 128)
        self.assertEqual(
            port.receipts[0].product_receipts[0].resource_ref,
            product_event_resource(port.receipts[0]),
        )

    def test_terminal_proposal_rejects_wrong_kind_and_forged_required_evidence(self) -> None:
        outcome = OutcomeProposal("submission:one", "outcome", "event:outcome", "receipt:outcome")
        with self.assertRaises(ContractError):
            TerminalProposal(
                run_id="run:one",
                invocation_id="invocation:one",
                actor_ref="agent:one",
                workspace_ref="workspace:one",
                snapshot_digest="snapshot:one",
                kind="failed",
                final_sequence=1,
                failure=RuntimeFailure("failed", "failure", True),
                outcome_proposal=outcome,
            )
        with self.assertRaises(ContractError):
            TerminalProposal(
                run_id="run:one",
                invocation_id="invocation:one",
                actor_ref="agent:one",
                workspace_ref="workspace:one",
                snapshot_digest="snapshot:one",
                kind="completed",
                final_sequence=1,
                evidence_event_ids=("event:other",),
                evidence_receipt_refs=("receipt:other",),
                outcome_proposal=outcome,
            )
        forged_cancellation = CancellationAuthorityReceipt(
            resource_ref="cancel:one",
            receipt_ref="receipt:cancel",
            run_id="run:other",
            invocation_id="invocation:one",
            actor_ref="agent:one",
            workspace_ref="workspace:one",
            snapshot_digest="snapshot:one",
            idempotency_key="cancel:run:other:invocation:one",
            gateway_receipt_ref="gateway:cancel:run:other:invocation:one",
            audit_ref="audit:cancel:run:other:invocation:one",
        )
        with self.assertRaises(BindingError):
            TerminalProposal(
                run_id="run:one",
                invocation_id="invocation:one",
                actor_ref="agent:one",
                workspace_ref="workspace:one",
                snapshot_digest="snapshot:one",
                kind="cancelled",
                final_sequence=1,
                evidence_receipt_refs=("receipt:cancel",),
                cancellation_receipt=forged_cancellation,
            )

    def test_payloads_are_bounded_at_event_and_contract_boundaries(self) -> None:
        with self.assertRaises(BoundsError):
            RuntimeEvent(
                PROTOCOL,
                "run:one",
                "invocation:one",
                1,
                "event:large",
                "cause:one",
                "event:large",
                TranscriptObserved("transcript:large", "x" * 40_000),
            )

        criteria = tuple("criterion" for _ in range(MAX_ACCEPTANCE_CRITERIA))
        replace(make_snapshot(), assignment=replace(make_snapshot().assignment, acceptance_criteria=criteria))
        with self.assertRaises(BoundsError):
            AssignmentSnapshot("v1", "target", "objective", criteria + ("overflow",))

        context = tuple(VersionedContextRef(f"context:{i}", "digest") for i in range(MAX_CONTEXT_REFS))
        replace(make_snapshot(), context=context)
        with self.assertRaises(BoundsError):
            replace(make_snapshot(), context=context + (VersionedContextRef("context:overflow", "digest"),))

        operations = tuple(OperationDescriptor(f"operation:{i}", "digest") for i in range(MAX_EAGER_OPERATIONS))
        replace(make_snapshot(), tool_presentation=ToolPresentation(operations, "catalog"))
        with self.assertRaises(BoundsError):
            ToolPresentation(operations + (OperationDescriptor("operation:overflow", "digest"),), "catalog")

        refs = tuple(f"event:{i}" for i in range(MAX_NEW_CONTEXT_EVENT_REFS))
        replace(make_invocation(make_snapshot()), new_context_event_refs=refs)
        with self.assertRaises(BoundsError):
            replace(make_invocation(make_snapshot()), new_context_event_refs=refs + ("event:overflow",))

        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:wire-boundary")
        outcome = OutcomeProposal(
            "submission:wire",
            "bounded outcome",
            "event:outcome",
            "receipt:outcome",
        )
        terminal = TerminalProposal(
            snapshot.run_id,
            invocation.invocation_id,
            snapshot.actor_ref,
            snapshot.workspace_ref,
            snapshot.digest(),
            "completed",
            1,
            evidence_event_ids=(outcome.event_id,),
            evidence_receipt_refs=(outcome.proposal_receipt_ref,),
            outcome_proposal=outcome,
        )
        receipt = FixtureTerminalReconciliationPort().reconcile_terminal(terminal)
        self.assertLessEqual(
            len(terminal.to_json().encode("utf-8")), MAX_TERMINAL_PROPOSAL_BYTES
        )
        self.assertLessEqual(
            len(receipt.to_json().encode("utf-8")), MAX_TERMINAL_RECEIPT_BYTES
        )
        with self.assertRaises(BoundsError):
            TerminalProposal.from_json("{" + ("é" * MAX_TERMINAL_PROPOSAL_BYTES))
        with self.assertRaises(BoundsError):
            TerminalReconciliationReceipt.from_json("{" + ("é" * MAX_TERMINAL_RECEIPT_BYTES))
        oversized_artifacts = tuple(
            ArtifactProposal(
                artifact_ref=f"artifact:{index}:{'x' * 240}",
                digest="d" * 256,
                event_id=f"event:{index}:{'x' * 240}",
            )
            for index in range(128)
        )
        with self.assertRaises(BoundsError):
            TerminalProposal(
                snapshot.run_id,
                invocation.invocation_id,
                snapshot.actor_ref,
                snapshot.workspace_ref,
                snapshot.digest(),
                "failed",
                1,
                evidence_event_ids=tuple(f"event-evidence:{index}:{'x' * 240}" for index in range(128)),
                evidence_receipt_refs=tuple(
                    f"receipt-evidence:{index}:{'x' * 240}" for index in range(128)
                ),
                failure=RuntimeFailure("oversized", "bounded failure", False),
                artifact_proposals=oversized_artifacts,
            )

    def test_snapshot_and_invocation_accept_exact_byte_boundary_and_reject_one_byte_over(self) -> None:
        base = make_snapshot()
        prefix = ("x" * 32_000,) * 4
        def snapshot_for_length(length: int) -> RunSnapshot:
            return replace(
                base,
                assignment=replace(base.assignment, acceptance_criteria=prefix + ("x" * length,)),
            )

        low, high = 1, 32_000
        while low < high:
            middle = (low + high + 1) // 2
            try:
                snapshot_for_length(middle)
            except BoundsError:
                high = middle - 1
            else:
                low = middle
        exact_snapshot = snapshot_for_length(low)
        if len(exact_snapshot.to_json().encode("utf-8")) != MAX_RUN_SNAPSHOT_BYTES:
            exact_snapshot = None
        self.assertIsNotNone(exact_snapshot, "the snapshot byte boundary must be reachable")
        assert exact_snapshot is not None
        self.assertEqual(len(exact_snapshot.to_json().encode("utf-8")), MAX_RUN_SNAPSHOT_BYTES)
        self.assertEqual(RunSnapshot.from_json(exact_snapshot.to_json()), exact_snapshot)
        with self.assertRaises(BoundsError):
            RunSnapshot.from_json(exact_snapshot.to_json() + " ")
        last = exact_snapshot.assignment.acceptance_criteria[-1]
        with self.assertRaises(BoundsError):
            replace(
                exact_snapshot,
                assignment=replace(
                    exact_snapshot.assignment,
                    acceptance_criteria=exact_snapshot.assignment.acceptance_criteria[:-1] + (last + "x",),
                ),
            )

        invocation_base = make_invocation(base)
        ref_prefix = tuple(f"e{i}:" + ("x" * 252) for i in range(61))
        def invocation_for_length(length: int) -> InvocationEnvelope:
            return replace(
                invocation_base,
                new_context_event_refs=ref_prefix + ("last:" + "x" * length,),
            )

        low, high = 1, 251
        while low < high:
            middle = (low + high + 1) // 2
            try:
                invocation_for_length(middle)
            except BoundsError:
                high = middle - 1
            else:
                low = middle
        exact_invocation = invocation_for_length(low)
        if len(exact_invocation.to_json().encode("utf-8")) != MAX_INVOCATION_BYTES:
            exact_invocation = None
        self.assertIsNotNone(exact_invocation, "the invocation byte boundary must be reachable")
        assert exact_invocation is not None
        self.assertEqual(len(exact_invocation.to_json().encode("utf-8")), MAX_INVOCATION_BYTES)
        self.assertEqual(InvocationEnvelope.from_json(exact_invocation.to_json()), exact_invocation)
        with self.assertRaises(BoundsError):
            InvocationEnvelope.from_json(exact_invocation.to_json() + " ")
        last = exact_invocation.new_context_event_refs[-1]
        with self.assertRaises(BoundsError):
            replace(exact_invocation, new_context_event_refs=exact_invocation.new_context_event_refs[:-1] + (last + "x",))

    def test_replaced_processes_round_trip_serialized_invocations(self) -> None:
        snapshot = make_snapshot()
        first = make_invocation(snapshot, remaining=RuntimeBudget(5, 100, 100))
        first_lines = invoke_service(
            {
                "run": snapshot.to_dict(),
                "invocation": first.to_dict(),
                "fakePlan": {"terminalKind": "waiting_for_input", "inputRequest": "Need one answer"},
            }
        )
        self.assertEqual([line["type"] for line in first_lines].count("exit"), 1)
        self.assertEqual(first_lines[-1]["exit"]["kind"], "waiting_for_input")  # type: ignore[index]
        first_events = [RuntimeEvent.from_dict(line["event"]) for line in first_lines if line["type"] == "event"]  # type: ignore[arg-type]
        first_stream = EventCollector(
            run_id=snapshot.run_id,
            invocation_id=first.invocation_id,
            expected_causation_ref=first.causation_ref,
        )
        for item in first_events:
            first_stream.accept(item)
        self.assertEqual(RuntimeExit.from_dict(first_lines[-1]["exit"]).kind, "waiting_for_input")  # type: ignore[arg-type,index]

        second = make_invocation(
            snapshot,
            invocation_id="invocation:replacement",
            trigger=InvocationTrigger("continuation", "event:answer"),
            context_refs=("event:answer",),
            checkpoint_ref="checkpoint:one",
            remaining=RuntimeBudget(4, 90, 90),
        )
        second_lines = invoke_service(
            {
                "run": snapshot.to_dict(),
                "invocation": second.to_dict(),
                "fakePlan": {"terminalKind": "completed", "transcript": "continued"},
            }
        )
        self.assertEqual([line["type"] for line in second_lines].count("exit"), 1)
        self.assertEqual(second_lines[-1]["exit"]["kind"], "completed")  # type: ignore[index]
        self.assertGreaterEqual(sum(line["type"] == "event" for line in second_lines), 3)

    def test_service_frame_reader_bounds_bytes_and_stops_at_one_line(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        valid = json.dumps(
            {"run": snapshot.to_dict(), "invocation": invocation.to_dict()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        stream = StringIO(valid + "\n" + ("x" * (MAX_SERVICE_REQUEST_BYTES + 1)) + "\n")
        self.assertEqual(_read_bounded_request_line(stream), valid)
        self.assertLess(stream.tell(), len(valid) + MAX_SERVICE_REQUEST_BYTES)

        exact_frame = "x" * MAX_SERVICE_REQUEST_BYTES
        self.assertEqual(
            _read_bounded_request_line(StringIO(exact_frame + "\n")), exact_frame
        )
        with self.assertRaises(BoundsError):
            _read_bounded_request_line(StringIO("a" * (MAX_SERVICE_REQUEST_BYTES + 1) + "\n"))
        with self.assertRaises(BoundsError):
            _read_bounded_request_line(
                StringIO("é" * ((MAX_SERVICE_REQUEST_BYTES // 2) + 1) + "\n")
            )
        with self.assertRaises(BoundsError):
            _read_bounded_request_line(StringIO(valid))

    def test_service_frame_reader_preserves_multiple_and_partial_chunks(self) -> None:
        class ChunkedStream:
            def __init__(self, *chunks: bytes) -> None:
                self.chunks = list(chunks)

            def read(self, size: int) -> bytes:
                del size
                return self.chunks.pop(0) if self.chunks else b""

        reader = _ServiceFrameReader(
            ChunkedStream(b"first\nse", b"cond\nthird\n"),
            chunk_bytes=SERVICE_FRAME_READ_CHUNK_BYTES,
        )
        self.assertEqual(reader.read_frame(), "first")
        self.assertEqual(reader.read_frame(), "second")
        self.assertEqual(reader.read_frame(), "third")
        self.assertEqual(reader.read_frame(), "")
        reader.close()

        cached_stream = StringIO("cached-one\ncached-two\n")
        self.assertEqual(_read_bounded_request_line(cached_stream), "cached-one")
        self.assertEqual(_read_bounded_request_line(cached_stream), "cached-two")

        repeated = _ServiceFrameReader(BytesIO(b"same\n" * 128))
        self.assertEqual([repeated.read_frame() for _ in range(128)], ["same"] * 128)
        self.assertEqual(repeated.read_frame(), "")
        repeated.close()

    def test_service_frame_reader_rejects_eof_and_one_mib_without_retaining_unbounded_data(self) -> None:
        read_fd, write_fd = os.pipe()
        pipe = os.fdopen(read_fd, "rb")
        pipe_reader = _ServiceFrameReader(pipe, timeout_seconds=0.1)
        try:
            os.write(write_fd, b"partial-open")
            started = time.monotonic()
            with self.assertRaises(BoundsError):
                pipe_reader.read_frame()
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            pipe_reader.close()
            pipe.close()
            os.close(write_fd)

        reader = _ServiceFrameReader(StringIO("partial"))
        with self.assertRaises(BoundsError):
            reader.read_frame()
        reader.close()

        reader = _ServiceFrameReader(StringIO("x" * (1_048_576 + 1) + "\n"))
        with self.assertRaises(BoundsError):
            reader.read_frame()
        self.assertLessEqual(
            len(reader._carry), MAX_SERVICE_REQUEST_BYTES + 4  # type: ignore[attr-defined]
        )
        reader.close()

    def test_service_rejects_oversized_composite_and_unterminated_frames_before_decode(self) -> None:
        oversized = "{" + ("a" * (MAX_SERVICE_REQUEST_BYTES + 1)) + "}\n"
        with patch("sys.stdin", StringIO(oversized)), patch("sys.stdout", StringIO()):
            self.assertEqual(service_main(["--once"]), 2)

        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        request = json.dumps(
            {"run": snapshot.to_dict(), "invocation": invocation.to_dict()},
            separators=(",", ":"),
        )
        with patch("sys.stdin", StringIO(request)), patch("sys.stdout", StringIO()):
            self.assertEqual(service_main(["--once"]), 2)

    def test_snapshot_and_invocation_raw_multibyte_and_mib_bounds_fail_before_decode(self) -> None:
        for parser in (RunSnapshot.from_json, InvocationEnvelope.from_json):
            with self.subTest(parser=parser):
                with self.assertRaises(BoundsError):
                    parser("é" * (1_048_576 // 2 + 1))
                with self.assertRaises(BoundsError):
                    parser("a" * (1_048_576 + 1))

    def test_service_request_rejects_unknown_fields(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        output = StringIO()
        with self.assertRaises(ContractError):
            serve_once(
                json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict(), "forged": True}),
                output,
            )

    def test_true_json_lines_streaming_and_idempotent_process_death_reconciliation(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        process = subprocess.Popen(
            [sys.executable, "-m", "plane_runtime.service", "--once"],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(
            json.dumps(
                {
                    "run": snapshot.to_dict(),
                    "invocation": invocation.to_dict(),
                    "fakePlan": {"holdAfterObservations": 1},
                }
            )
            + "\n"
        )
        process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        self.assertTrue(selector.select(timeout=5), "the first event must be flushed before process exit")
        first_line = process.stdout.readline()
        self.assertTrue(first_line)
        first = json.loads(first_line)
        self.assertEqual(first["type"], "event")
        streamed_event = RuntimeEvent.from_dict(first["event"])
        process.kill()
        process.wait(timeout=5)
        selector.close()
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

        stream = EventCollector(
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            expected_causation_ref=invocation.causation_ref,
        )
        self.assertTrue(stream.accept(streamed_event))
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "terminal-reconciliation.json"
            port = SharedTerminalPort(path=store_path)
            first_receipt = reconcile_process_death(
                port=port,
                run=snapshot,
                invocation_id=invocation.invocation_id,
                final_sequence=stream.last_sequence,
            )
            restarted_port = SharedTerminalPort(path=store_path)
            second_receipt = reconcile_process_death(
                port=restarted_port,
                run=snapshot,
                invocation_id=invocation.invocation_id,
                final_sequence=stream.last_sequence,
            )
            self.assertEqual(first_receipt, second_receipt)
            state = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual(list(state), [first_receipt.idempotency_key])

    def test_real_service_rejects_open_partial_frame_by_finite_deadline(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-m", "plane_runtime.service", "--once"],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            process.stdin.write(b'{"partial":"secret-that-must-not-echo"')
            process.stdin.flush()
            started = time.monotonic()
            returncode = process.wait(timeout=4.0)
            elapsed = time.monotonic() - started
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            self.assertEqual(returncode, 2)
            self.assertLess(elapsed, 3.5)
            self.assertNotIn(b"secret-that-must-not-echo", stdout + stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2.0)
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    def test_supervisor_classifies_process_death_as_one_terminal_exit(self) -> None:
        exit_value = classify_process_death(final_sequence=2)
        self.assertEqual(exit_value.kind, "failed")
        self.assertIsNotNone(exit_value.failure)
        self.assertEqual(exit_value.failure.code, "process_died")  # type: ignore[union-attr]
        self.assertEqual(RuntimeExit.from_json(exit_value.to_json()), exit_value)

    def test_terminal_reconciliation_is_required_and_exactly_once_under_concurrency(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:terminal")
        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            invocation.invocation_id,
            "lease:one",
            "host:one",
            True,
            invocation.lease.expires_at,
        )
        authority = FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW)
        with self.assertRaises(ContractError):
            execute(
                run=snapshot,
                invocation=invocation,
                lease_authority=authority,
                lease_binding=binding,
            )

        store: dict[str, object] = {}
        lock = threading.RLock()
        accepted: list[str] = []
        port = SharedTerminalPort(store, lock, accepted)
        results: list[RuntimeExit] = []

        def invoke() -> None:
            results.append(run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port))

        threads = [threading.Thread(target=invoke) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(item == results[0] for item in results))
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            port.product_events,
            [f"product-event:terminal:{invocation.run_id}:{invocation.invocation_id}"],
        )
        self.assertEqual(
            [item.resource_ref for item in port.product_receipts],
            ["submission:fake"],
        )

    def test_process_boundary_sanitizes_unexpected_exception_and_keeps_detail_internal(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:exploding")
        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            invocation.invocation_id,
            "lease:one",
            "host:one",
            True,
            invocation.lease.expires_at,
        )
        output = StringIO()
        captured: list[Exception] = []
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
            lease_binding=binding,
            terminal_port=SharedTerminalPort(),
            kernel=ExplodingKernel(),
            internal_failure_hook=captured.append,
        )
        wire = output.getvalue()
        self.assertNotIn("provider secret", wire)
        self.assertNotIn("Traceback", wire)
        self.assertEqual(status, 0)
        self.assertEqual(len(captured), 1)
        self.assertIn("provider secret", str(captured[0]))
        reconciliation = json.loads(wire)
        self.assertEqual(reconciliation["type"], "reconciliation")
        self.assertTrue(reconciliation["receipt"]["accepted"])

    def test_process_boundary_returns_nonzero_when_exception_reconciliation_is_not_accepted(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:unreconciled")
        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            invocation.invocation_id,
            "lease:one",
            "host:one",
            True,
            invocation.lease.expires_at,
        )
        output = StringIO()
        port = RejectingTerminalPort()
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
            lease_binding=binding,
            terminal_port=port,
            kernel=ExplodingKernel(),
        )
        self.assertEqual(status, 1)
        line = json.loads(output.getvalue())
        self.assertEqual(line["type"], "reconciliation_request")
        self.assertEqual(
            line["request"]["message"],
            "runtime execution failed; Plane reconciliation is required",
        )
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(port.product_events, [])

    def test_service_terminal_port_failure_is_nonzero_without_a_second_mutation_attempt(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:port-failure")
        port = FailingTerminalPort()
        status, lines, _ = serve_fixture(
            snapshot,
            invocation,
            terminal_port=port,
        )
        self.assertEqual(status, 1)
        self.assertEqual(port.calls, 1)
        self.assertEqual(lines[-1]["type"], "reconciliation_request")
        self.assertEqual(
            lines[-1]["request"]["message"],
            "terminal reconciliation is unavailable; supervisor action is required",
        )
        self.assertNotIn("terminal provider secret", json.dumps(lines))

    def test_service_legal_terminal_rejection_is_bounded_and_single_attempt(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:legal-rejection")
        port = RejectingTerminalPort()
        status, lines, _ = serve_fixture(
            snapshot,
            invocation,
            terminal_port=port,
        )
        self.assertEqual(status, 1)
        self.assertEqual(port.calls, 1)
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(port.product_events, [])
        self.assertEqual(lines[-1]["type"], "reconciliation_request")
        self.assertEqual(
            lines[-1]["request"]["code"], "terminal_reconciliation_unavailable"
        )
        self.assertNotIn("audit_ref", json.dumps(lines))

    def test_service_kernel_valueerror_and_typeerror_use_generic_failure_phase(self) -> None:
        snapshot = make_snapshot()
        for kernel in (ValueErrorKernel(), TypeErrorKernel()):
            with self.subTest(kernel=type(kernel).__name__):
                status, lines, _ = serve_fixture(
                    snapshot,
                    make_invocation(snapshot, invocation_id=f"invocation:{type(kernel).__name__}"),
                    kernel=kernel,
                )
                self.assertEqual(status, 0)
                self.assertEqual(lines[-1]["type"], "reconciliation")
                wire = json.dumps(lines)
                self.assertNotIn("provider secret", wire)

    def test_service_dependency_failures_are_execution_failures_not_invalid_requests(self) -> None:
        snapshot = make_snapshot()

        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            "invocation:raising-lease",
            "lease:one",
            "host:one",
            True,
            "2099-01-01T00:00:00Z",
        )
        invocation = make_invocation(snapshot, invocation_id=binding.invocation_id)
        output = StringIO()
        captured: list[Exception] = []
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            lease_authority=RaisingLeaseAuthority(),
            lease_binding=binding,
            terminal_port=FixtureTerminalReconciliationPort(),
            internal_failure_hook=captured.append,
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["type"], "reconciliation")
        self.assertEqual(len(captured), 1)
        self.assertIn("lease secret", str(captured[0]))
        self.assertNotIn("lease secret", output.getvalue())

        continuation = make_invocation(
            snapshot,
            invocation_id="invocation:raising-checkpoint",
            trigger=InvocationTrigger("continuation", "event:answer"),
            checkpoint_ref="checkpoint:one",
        )
        continuation_binding = CanonicalLeaseBinding(
            snapshot.run_id,
            continuation.invocation_id,
            "lease:one",
            "host:one",
            True,
            continuation.lease.expires_at,
        )
        attestation = CheckpointAttestation(
            checkpoint_ref="checkpoint:one",
            source_run_id=snapshot.run_id,
            source_invocation_id="invocation:source",
            snapshot_digest=snapshot.digest(),
            actor_ref=snapshot.actor_ref,
            profile_version=snapshot.profile_version,
            continuation_event_ref="event:answer",
            continuation_trigger_kind="continuation",
            allowed_target_invocation_id=continuation.invocation_id,
        )
        output = StringIO()
        captured = []
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": continuation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([continuation_binding], clock=lambda: TRUSTED_NOW),
            lease_binding=continuation_binding,
            checkpoint_authority=RaisingCheckpointAuthority(),
            checkpoint_attestation=attestation,
            terminal_port=FixtureTerminalReconciliationPort(),
            internal_failure_hook=captured.append,
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["type"], "reconciliation")
        self.assertEqual(len(captured), 1)
        self.assertIn("checkpoint secret", str(captured[0]))
        self.assertNotIn("checkpoint secret", output.getvalue())

        output = StringIO()
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": continuation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([continuation_binding], clock=lambda: TRUSTED_NOW),
            lease_binding=continuation_binding,
            terminal_port=FixtureTerminalReconciliationPort(),
        )
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "runtime_configuration")

        output = StringIO()
        captured = []
        cancellation = MutableCancellation()
        cancellation.cancel()
        status = serve_once(
            json.dumps(
                {
                    "run": snapshot.to_dict(),
                    "invocation": make_invocation(
                        snapshot, invocation_id="invocation:raising-cancellation-authority"
                    ).to_dict(),
                }
            ),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([CanonicalLeaseBinding(
                snapshot.run_id,
                "invocation:raising-cancellation-authority",
                "lease:one",
                "host:one",
                True,
                "2099-01-01T00:00:00Z",
            )], clock=lambda: TRUSTED_NOW),
            lease_binding=CanonicalLeaseBinding(
                snapshot.run_id,
                "invocation:raising-cancellation-authority",
                "lease:one",
                "host:one",
                True,
                "2099-01-01T00:00:00Z",
            ),
            cancellation=cancellation,
            cancellation_authority=RaisingCancellationAuthority(),
            terminal_port=FixtureTerminalReconciliationPort(),
            internal_failure_hook=captured.append,
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["type"], "reconciliation")
        self.assertEqual(len(captured), 1)
        self.assertIn("cancellation secret", str(captured[0]))
        self.assertNotIn("cancellation secret", output.getvalue())

        signal_invocation = make_invocation(snapshot, invocation_id="invocation:raising-signal")
        signal_binding = CanonicalLeaseBinding(
            snapshot.run_id,
            signal_invocation.invocation_id,
            "lease:one",
            "host:one",
            True,
            signal_invocation.lease.expires_at,
        )
        output = StringIO()
        captured = []
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": signal_invocation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([signal_binding], clock=lambda: TRUSTED_NOW),
            lease_binding=signal_binding,
            cancellation=RaisingCancellationSignal(),
            terminal_port=FixtureTerminalReconciliationPort(),
            internal_failure_hook=captured.append,
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["type"], "reconciliation")
        self.assertEqual(len(captured), 1)
        self.assertIn("cancellation signal", str(captured[0]))
        self.assertNotIn("cancellation signal", output.getvalue())

        output = StringIO()
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": make_invocation(snapshot).to_dict()}),
            output,
        )
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "runtime_configuration")
        self.assertNotEqual(json.loads(output.getvalue())["error"]["code"], "invalid_request")

    def test_service_pre_dispatch_cancellation_uses_signal_and_independent_authority(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:pre-cancel")
        cancellation = MutableCancellation()
        cancellation.cancel()
        kernel = NoDispatchKernel()
        port = FixtureTerminalReconciliationPort()
        status, lines, _ = serve_fixture(
            snapshot,
            invocation,
            kernel=kernel,
            terminal_port=port,
            cancellation=cancellation,
        )
        self.assertEqual(status, 0)
        self.assertFalse(kernel.dispatched)
        self.assertEqual(lines[-1]["type"], "exit")
        self.assertEqual(lines[-1]["exit"]["kind"], "cancelled")
        self.assertEqual(
            port.product_events,
            [("terminal:run:one:invocation:pre-cancel", "cancelled")],
        )

    def test_service_mid_run_cancellation_wins_over_late_completion(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:mid-cancel")
        cancellation = MutableCancellation()
        port = FixtureTerminalReconciliationPort()
        kernel = CancelAfterProgressKernel(cancellation)
        status, lines, _ = serve_fixture(
            snapshot,
            invocation,
            kernel=kernel,
            terminal_port=port,
            cancellation=cancellation,
        )
        self.assertEqual(status, 0)
        self.assertEqual(lines[-1]["type"], "exit")
        self.assertEqual(lines[-1]["exit"]["kind"], "cancelled")
        self.assertEqual(port.product_events, [("terminal:run:one:invocation:mid-cancel", "cancelled")])
        self.assertFalse(any(item[1] == "completed" for item in port.product_events))

    def test_service_json_lines_exact_replay_reuses_one_terminal_receipt(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:service-replay")
        port = FixtureTerminalReconciliationPort()
        first_status, first_lines, _ = serve_fixture(
            snapshot,
            invocation,
            terminal_port=port,
        )
        second_status, second_lines, _ = serve_fixture(
            snapshot,
            invocation,
            terminal_port=port,
        )
        self.assertEqual(first_status, 0)
        self.assertEqual(second_status, 0)
        self.assertEqual(first_lines[-1], second_lines[-1])
        self.assertEqual(len(port.accepted), 1)
        self.assertEqual(len(port.product_events), 1)
        self.assertEqual(len(port.receipts), 1)

    def test_service_rejects_wrong_invocation_and_cancellation_bindings_without_dispatch(self) -> None:
        snapshot = make_snapshot()
        wrong_invocation = replace(
            make_invocation(snapshot, invocation_id="invocation:wrong-run"), run_id="run:forged"
        )
        kernel = NoDispatchKernel()
        status, lines, _ = serve_fixture(snapshot, wrong_invocation, kernel=kernel)
        self.assertEqual(status, 1)
        self.assertFalse(kernel.dispatched)
        self.assertEqual(lines[-1]["error"]["code"], "binding_rejected")

        invocation = make_invocation(snapshot, invocation_id="invocation:wrong-cancel-ref")
        cancellation = MutableCancellation()
        cancellation.cancel()
        forged_receipt = CancellationAuthorityReceipt(
            resource_ref="cancel:forged",
            receipt_ref=f"cancel-receipt:{invocation.invocation_id}",
            run_id=snapshot.run_id,
            invocation_id=invocation.invocation_id,
            actor_ref=snapshot.actor_ref,
            workspace_ref=snapshot.workspace_ref,
            snapshot_digest=snapshot.digest(),
            idempotency_key=f"cancel:{snapshot.run_id}:{invocation.invocation_id}",
            gateway_receipt_ref=f"gateway:cancel:{snapshot.run_id}:{invocation.invocation_id}",
            audit_ref=f"audit:cancel:{snapshot.run_id}:{invocation.invocation_id}",
        )
        port = FixtureTerminalReconciliationPort()
        status, lines, _ = serve_fixture(
            snapshot,
            invocation,
            kernel=NoDispatchKernel(),
            terminal_port=port,
            cancellation=cancellation,
            cancellation_authority=FixtureCancellationAuthority([forged_receipt]),
        )
        self.assertEqual(status, 1)
        self.assertEqual(lines[-1]["error"]["code"], "binding_rejected")
        self.assertEqual(port.product_events, [])

        continuation = make_invocation(
            snapshot,
            invocation_id="invocation:wrong-actor",
            trigger=InvocationTrigger("continuation", "event:answer"),
            checkpoint_ref="checkpoint:one",
        )
        forged_attestation = CheckpointAttestation(
            checkpoint_ref="checkpoint:one",
            source_run_id=snapshot.run_id,
            source_invocation_id="invocation:source",
            snapshot_digest=snapshot.digest(),
            actor_ref="agent:forged",
            profile_version=snapshot.profile_version,
            continuation_event_ref="event:answer",
            continuation_trigger_kind="continuation",
            allowed_target_invocation_id=continuation.invocation_id,
        )
        binding = CanonicalLeaseBinding(
            snapshot.run_id,
            continuation.invocation_id,
            "lease:one",
            "host:one",
            True,
            continuation.lease.expires_at,
        )
        output = StringIO()
        status = serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": continuation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
            lease_binding=binding,
            checkpoint_authority=FixtureCheckpointAuthority([forged_attestation]),
            checkpoint_attestation=forged_attestation,
            terminal_port=FixtureTerminalReconciliationPort(),
            kernel=NoDispatchKernel(),
        )
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "binding_rejected")

    def test_parse_valueerror_control_and_main_exit_statuses(self) -> None:
        output = StringIO()
        with self.assertRaises(ValueError):
            serve_once("not-json", output)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "invalid_request")

        with patch("sys.stdin", StringIO("not-json\n")), patch("sys.stdout", StringIO()):
            self.assertEqual(service_main(["--once"]), 2)

        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        with patch("sys.stdin", StringIO(json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}) + "\n")), patch("sys.stdout", StringIO()):
            self.assertEqual(service_main(["--once"]), 0)

    def test_runtime_import_graph_uses_fresh_complete_sys_modules_delta(self) -> None:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, json, sys; before=set(sys.modules); [importlib.import_module(name) for name in ('plane_runtime', 'plane_runtime.contract', 'plane_runtime.adapter', 'plane_runtime.service')]; print(json.dumps(sorted(set(sys.modules)-before)))",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        delta = json.loads(probe.stdout)
        forbidden_roots = (
            "plane", "plane_api", "plane_server", "plane_product", "buzz",
            "run_agent", "model_tools", "hermes_state", "agent", "cron", "gateway",
            "memory", "delegation", "requests", "httpx", "aiohttp", "urllib3",
            "openai", "anthropic", "boto3", "botocore", "slack_sdk", "telegram",
            "discord", "websocket", "websockets",
        )
        forbidden = [
            name
            for name in delta
            if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
        ]
        self.assertEqual(forbidden, [], f"forbidden fresh import(s): {forbidden}; delta={delta}")
        for name in ("agentic", "memoryview", "plane_apiary", "websocket_client_extra"):
            with self.subTest(name=name):
                self.assertFalse(
                    any(name == root or name.startswith(root + ".") for root in forbidden_roots)
                )


if __name__ == "__main__":
    unittest.main()
