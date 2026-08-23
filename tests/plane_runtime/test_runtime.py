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

from plane_runtime.adapter import (
    MAX_EVENTS_PER_INVOCATION,
    MAX_EVENT_STREAM_BYTES,
    MAX_TRANSCRIPT_BYTES,
    MAX_TRANSCRIPT_OBSERVATIONS,
    MAX_OPTIONAL_EVENT_TAIL,
    MAX_ARTIFACT_PROPOSALS,
    MAX_INPUT_PROPOSALS,
    MAX_MESSAGE_PROPOSALS,
    PROTOCOL,
    ArtifactObserved,
    ArtifactProposal,
    BindingError,
    BoundsError,
    CancellationAuthorityReceipt,
    CanonicalLeaseBinding,
    CheckpointAttestation,
    ContractError,
    EventCollector,
    FakeKernel,
    FakeKernelPlan,
    FixtureCanonicalLeaseAuthority,
    FixtureCancellationAuthority,
    FixtureCheckpointAuthority,
    FixtureTerminalReconciliationPort,
    InvocationEnvelope,
    LeaseError,
    MutableCancellation,
    OutcomeProposal,
    OutcomeSubmissionObserved,
    ProductReceipt,
    product_proof_identity,
    ProgressObserved,
    MessageProposal,
    MessageProposalObserved,
    RuntimeBudget,
    RuntimeConfigurationError,
    RuntimeEvent,
    RuntimeExit,
    RuntimeFailure,
    RuntimeLease,
    RunSnapshot,
    SequenceError,
    TerminalProposal,
    TerminalProof,
    TerminalReconciliationError,
    TerminalReconciliationRejected,
    TerminalReconciliationReceipt,
    TranscriptObserved,
    UsageObserved,
    classify_process_death,
    execute,
    parse_utc_timestamp,
    reconcile_terminal_proposal,
    reconcile_process_death,
)

from plane_runtime.contract import (
    MAX_ACCEPTANCE_CRITERIA,
    MAX_CONTEXT_REFS,
    MAX_EAGER_OPERATIONS,
    MAX_INVOCATION_BYTES,
    MAX_NEW_CONTEXT_EVENT_REFS,
    MAX_RUN_SNAPSHOT_BYTES,
    MAX_TERMINAL_PROPOSAL_BYTES,
    MAX_TERMINAL_RECEIPT_BYTES,
    AssignmentSnapshot,
    ContractDigests,
    InvocationTrigger,
    OperationDescriptor,
    RuntimeBudgetPolicy,
    RuntimeModelRoute,
    ToolPresentation,
    VersionedContextRef,
)

from plane_runtime.adapter import KernelObservation, KernelRequest, KernelResult


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


def proofless_rejection(proposal: TerminalProposal) -> TerminalReconciliationReceipt:
    """Build the only valid denial shape shared by terminal fakes."""

    return TerminalReconciliationReceipt(
        receipt_ref=f"rejected:{proposal.idempotency_key}",
        run_id=proposal.run_id,
        invocation_id=proposal.invocation_id,
        kind=proposal.kind,
        idempotency_key=proposal.idempotency_key,
        accepted=False,
        legal_transition=False,
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


class DirectTerminalRejectionKernel:
    def __init__(self, receipt: TerminalReconciliationReceipt) -> None:
        self.receipt = receipt

    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, emit, cancellation
        raise TerminalReconciliationRejected(self.receipt)


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
        return proofless_rejection(proposal)


class FailingTerminalPort:
    def __init__(self) -> None:
        self.calls = 0

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        del proposal
        self.calls += 1
        raise ValueError("terminal provider secret")


class AttachedReceiptFailurePort:
    """Raise a transport failure that happens to carry a misleading receipt."""

    def __init__(self) -> None:
        self.calls = 0
        self.product_events: list[str] = []

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        self.calls += 1
        raise TerminalReconciliationError(proofless_rejection(proposal))


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
                    return proofless_rejection(proposal)
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
                    return proofless_rejection(proposal)
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
        with self.assertRaises(TerminalReconciliationRejected):
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

    def test_validated_rejection_is_structurally_distinct_from_bad_reconciliation(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:rejection-type")
        port = RejectingTerminalPort()
        with self.assertRaises(TerminalReconciliationRejected) as raised:
            run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port)
        self.assertNotIsInstance(raised.exception, TerminalReconciliationError)
        self.assertEqual(port.calls, 1)
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(port.product_events, [])
        receipt = raised.exception.receipt
        self.assertFalse(receipt.accepted)
        self.assertFalse(receipt.legal_transition)
        self.assertEqual(receipt.proofs, ())
        self.assertEqual(receipt.product_receipts, ())

        malformed = ForgedReceiptPort(lambda receipt: object())
        with self.assertRaises(TerminalReconciliationError) as malformed_error:
            reconcile_terminal_proposal(port=malformed, proposal=port.proposals[0])
        self.assertIs(type(malformed_error.exception), TerminalReconciliationError)

    def test_completed_requires_outcome_receipt_not_progress_usage_or_transcript_evidence(self) -> None:
        with self.assertRaises(ContractError):
            run_execute(
                snapshot=make_snapshot(),
                invocation=make_invocation(make_snapshot(), invocation_id="invocation:evidence-only"),
                kernel=FakeKernel(FakeKernelPlan(outcome_submission_requested=False)),
            )

    def test_runtime_first_rejects_late_supervisor_mutation_with_one_slot(self) -> None:
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
        self.assertEqual(process_death.proofs, ())
        self.assertEqual(process_death.product_receipts, ())
        self.assertEqual(process_death, replay)
        self.assertEqual(len(port.accepted), 1)
        self.assertEqual(len(port.product_events), 1)
        self.assertEqual(len(port.product_receipts), 1)
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
        port = SharedTerminalPort()
        supervisor_receipt = reconcile_process_death(
            port=port,
            run=snapshot,
            invocation_id=invocation.invocation_id,
            final_sequence=0,
        )
        self.assertTrue(supervisor_receipt.accepted)
        with self.assertRaises(TerminalReconciliationRejected) as raised:
            run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port)
        self.assertFalse(raised.exception.receipt.accepted)
        self.assertFalse(raised.exception.receipt.legal_transition)
        self.assertEqual(raised.exception.receipt.proofs, ())
        self.assertEqual(raised.exception.receipt.product_receipts, ())
        self.assertEqual(port.accepted, ["terminal:run:one:invocation:supervisor-first"])
        self.assertEqual(port.product_events, [f"product-event:{port.accepted[0]}"])
        self.assertEqual(len(port.product_receipts), 1)
        self.assertEqual(
            port.product_receipts[0].resource_ref,
            f"product-event:{port.accepted[0]}",
        )

    def test_synchronized_runtime_and_supervisor_race_has_one_product_event(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot, invocation_id="invocation:synchronized")
        port = SharedTerminalPort()
        start = threading.Barrier(2)
        results: list[tuple[str, bool]] = []
        errors: list[Exception] = []

        def runtime() -> None:
            try:
                start.wait(timeout=5)
                run_execute(snapshot=snapshot, invocation=invocation, terminal_port=port)
                results.append(("runtime", True))
            except TerminalReconciliationRejected:
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
        self.assertEqual(len(port.product_receipts), 1)
        self.assertEqual(port.product_receipts[0].resource_ref, port.product_events[0])

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






    def test_snapshot_and_invocation_raw_multibyte_and_mib_bounds_fail_before_decode(self) -> None:
        for parser in (RunSnapshot.from_json, InvocationEnvelope.from_json):
            with self.subTest(parser=parser):
                with self.assertRaises(BoundsError):
                    parser("é" * (1_048_576 // 2 + 1))
                with self.assertRaises(BoundsError):
                    parser("a" * (1_048_576 + 1))




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
