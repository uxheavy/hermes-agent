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
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from plane_runtime import (
    MAX_ACCEPTANCE_CRITERIA,
    MAX_CONTEXT_REFS,
    MAX_EAGER_OPERATIONS,
    MAX_INVOCATION_BYTES,
    MAX_NEW_CONTEXT_EVENT_REFS,
    MAX_RUN_SNAPSHOT_BYTES,
    PROTOCOL,
    ArtifactObserved,
    AssignmentSnapshot,
    BindingError,
    BoundsError,
    CanonicalLeaseBinding,
    CheckpointAttestation,
    ContractDigests,
    ContractError,
    EventCollector,
    FakeKernel,
    FakeKernelPlan,
    FixtureCanonicalLeaseAuthority,
    FixtureCheckpointAuthority,
    InvocationEnvelope,
    InvocationTrigger,
    LeaseError,
    MutableCancellation,
    OperationDescriptor,
    ProductReceipt,
    ProgressObserved,
    PublicationReceipt,
    RecordingHost,
    RuntimeBudget,
    RuntimeBudgetPolicy,
    RuntimeEvent,
    RuntimeExit,
    RuntimeFailure,
    RuntimeLease,
    RuntimeModelRoute,
    RunSnapshot,
    SequenceError,
    TerminalProposal,
    TerminalReconciliationReceipt,
    ToolPresentation,
    TranscriptObserved,
    UsageObserved,
    VersionedContextRef,
    classify_process_death,
    execute,
    parse_utc_timestamp,
    reconcile_process_death,
)
from plane_runtime.adapter import KernelObservation, KernelRequest, KernelResult
from plane_runtime.service import serve_once


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
    kwargs.setdefault("host", RecordingHost())
    kwargs.setdefault("terminal_port", SharedTerminalPort())
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
        return KernelResult("completed")


class ExplodingKernel:
    def dispatch(self, request: KernelRequest, emit, cancellation) -> KernelResult:
        del request, emit, cancellation
        raise RuntimeError("provider secret must stay internal")


class BadInputReceiptHost(RecordingHost):
    def request_input(self, **kwargs) -> ProductReceipt:
        receipt = super().request_input(**kwargs)
        return replace(receipt, resource_ref="input:forged")


class BadPublicationReceiptHost(RecordingHost):
    def publish_transcript(self, **kwargs) -> PublicationReceipt:
        receipt = super().publish_transcript(**kwargs)
        return replace(receipt, invocation_id="invocation:forged")


class SharedTerminalPort:
    """Thread/process-safe durable-like fake for terminal reconciliation tests."""

    def __init__(self, store=None, lock=None, accepted=None, path: Path | None = None) -> None:
        self.store = store if store is not None else {}
        self.lock = lock if lock is not None else threading.RLock()
        self.accepted = accepted if accepted is not None else []
        self.path = path

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        key = proposal.idempotency_key
        if self.path is not None:
            return self._reconcile_file(proposal)
        with self.lock:
            prior = self.store.get(key)
            if prior is not None:
                prior_proposal, prior_receipt = prior
                if prior_proposal != proposal:
                    raise SequenceError("terminal idempotency key was reused with different content")
                return prior_receipt
            receipt = self._new_receipt(proposal)
            if not receipt.accepted:
                return receipt
            self.store[key] = (proposal, receipt)
            self.accepted.append(key)
            return receipt

    def _new_receipt(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        accepted = not (
            (proposal.kind == "completed" and not proposal.evidence_event_ids)
            or (proposal.kind == "waiting_for_input" and not proposal.evidence_receipt_refs)
        )
        return TerminalReconciliationReceipt(
            receipt_ref=f"{'receipt' if accepted else 'rejected'}:{proposal.idempotency_key}",
            audit_ref=f"audit:{proposal.idempotency_key}",
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            kind=proposal.kind,
            idempotency_key=proposal.idempotency_key,
            accepted=accepted,
            legal_transition=accepted,
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
                if prior["proposal"] != repr(proposal):
                    raise SequenceError("terminal idempotency key was reused with different content")
                return TerminalReconciliationReceipt(**prior["receipt"])
            receipt = self._new_receipt(proposal)
            if receipt.accepted:
                state[key] = {"proposal": repr(proposal), "receipt": receipt.__dict__}
                handle.seek(0)
                handle.truncate()
                json.dump(state, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                self.accepted.append(key)
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
                host=RecordingHost(),
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
                host=RecordingHost(),
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
                host=RecordingHost(),
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

    def test_input_requests_are_authorized_receipt_correlated_and_terminally_consistent(self) -> None:
        snapshot = make_snapshot()
        host = RecordingHost()
        events: list[RuntimeEvent] = []
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot),
            host=host,
            emit=events.append,
            kernel=FakeKernel(FakeKernelPlan(terminal_kind="waiting_for_input", input_request="Need one answer")),
        )
        self.assertEqual(exit_value.kind, "waiting_for_input")
        input_events = [item for item in events if item.body.to_dict()["kind"] == "input_request"]
        self.assertEqual(len(input_events), 1)
        self.assertEqual(input_events[0].body.to_dict()["receiptRef"], "receipt:invocation:one:input:input:fake")
        self.assertEqual(len(host.input_requests), 1)

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
        with self.assertRaises(BindingError):
            run_execute(
                snapshot=snapshot,
                invocation=make_invocation(snapshot, invocation_id="invocation:forged-input"),
                host=BadInputReceiptHost(),
                kernel=FakeKernel(FakeKernelPlan(terminal_kind="waiting_for_input", input_request="Need one answer")),
            )

    def test_artifacts_are_authorized_receipt_correlated_and_not_a_second_mutation_path(self) -> None:
        snapshot = make_snapshot()
        host = RecordingHost()
        events: list[RuntimeEvent] = []
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot),
            host=host,
            emit=events.append,
            kernel=ArtifactKernel(),
        )
        self.assertEqual(exit_value.kind, "completed")
        artifact = [item for item in events if isinstance(item.body, ArtifactObserved)]
        self.assertEqual(len(artifact), 1)
        self.assertEqual(artifact[0].body.receipt_ref, "receipt:invocation:one:artifact:artifact:one")
        self.assertEqual(host.artifacts, [("run:one", "artifact:one", "invocation:one:artifact:artifact:one")])

    def test_transcript_is_evidence_and_explicit_publication_requires_correlated_receipt(self) -> None:
        host = RecordingHost()
        snapshot = make_snapshot()
        events: list[RuntimeEvent] = []
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot),
            host=host,
            emit=events.append,
            kernel=FakeKernel(FakeKernelPlan(publication_requested=False)),
        )
        self.assertEqual(exit_value.kind, "completed")
        self.assertTrue(any(isinstance(item.body, TranscriptObserved) for item in events))
        self.assertFalse(any(item.body.to_dict()["kind"] == "conversation_publication" for item in events))
        self.assertEqual(host.publications, [])

        events = []
        exit_value = run_execute(
            snapshot=snapshot,
            invocation=make_invocation(snapshot, invocation_id="invocation:published"),
            host=host,
            emit=events.append,
            kernel=FakeKernel(FakeKernelPlan(publication_requested=True)),
        )
        self.assertEqual(exit_value.kind, "completed")
        publication = [item for item in events if item.body.to_dict()["kind"] == "conversation_publication"]
        self.assertEqual(len(publication), 1)
        self.assertTrue(publication[0].body.to_dict()["receiptRef"].startswith("receipt:"))
        with self.assertRaises(BindingError):
            run_execute(
                snapshot=snapshot,
                invocation=make_invocation(snapshot, invocation_id="invocation:bad-publication"),
                host=BadPublicationReceiptHost(),
                kernel=FakeKernel(FakeKernelPlan(publication_requested=True)),
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
                run_id=snapshot.run_id,
                invocation_id=invocation.invocation_id,
                final_sequence=stream.last_sequence,
            )
            restarted_port = SharedTerminalPort(path=store_path)
            second_receipt = reconcile_process_death(
                port=restarted_port,
                run_id=snapshot.run_id,
                invocation_id=invocation.invocation_id,
                final_sequence=stream.last_sequence,
            )
            self.assertEqual(first_receipt, second_receipt)
            state = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual(list(state), [first_receipt.idempotency_key])

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
                host=RecordingHost(),
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
        serve_once(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            host=RecordingHost(),
            lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: TRUSTED_NOW),
            lease_binding=binding,
            terminal_port=SharedTerminalPort(),
            kernel=ExplodingKernel(),
            internal_failure_hook=captured.append,
        )
        wire = output.getvalue()
        self.assertNotIn("provider secret", wire)
        self.assertNotIn("Traceback", wire)
        self.assertEqual(len(captured), 1)
        self.assertIn("provider secret", str(captured[0]))
        reconciliation = json.loads(wire)
        self.assertEqual(reconciliation["type"], "reconciliation_request")
        self.assertEqual(reconciliation["request"]["message"], "runtime execution failed; Plane reconciliation is required")

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
        forbidden_prefixes = (
            "plane.", "plane_api", "plane_server", "plane_product", "buzz",
            "run_agent", "model_tools", "hermes_state", "agent", "cron", "gateway",
            "memory", "delegation", "requests", "httpx", "aiohttp", "urllib3",
            "openai", "anthropic", "boto3", "botocore", "slack_sdk", "telegram",
            "discord", "websocket", "websockets",
        )
        forbidden = [
            name
            for name in delta
            if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in forbidden_prefixes)
        ]
        self.assertEqual(forbidden, [], f"forbidden fresh import(s): {forbidden}; delta={delta}")


if __name__ == "__main__":
    unittest.main()
