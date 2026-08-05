"""Evidence for the host-authoritative invocation supervisor boundary."""

from __future__ import annotations

import hashlib
import gc
import io
import inspect
import json
import math
import os
import subprocess
import sys
import threading
import time
import unittest
import weakref
from datetime import datetime, timezone
from unittest.mock import patch

import plane_runtime.invocation_supervisor as supervisor_module

from plane_runtime import (
    AssignmentSnapshot,
    CanonicalCancellationAuthority,
    CanonicalCancellationBinding,
    CanonicalLeaseBinding,
    ContractDigests,
    EventCollector,
    FakeKernel,
    FixtureCanonicalLeaseAuthority,
    FixtureTerminalReconciliationPort,
    InvocationEnvelope,
    InvocationTrigger,
    MutableCancellation,
    OperationDescriptor,
    RuntimeBudget,
    RuntimeBudgetPolicy,
    CancellationAuthorityReceipt,
    RuntimeEvent,
    RuntimeExit,
    RuntimeLease,
    RuntimeModelRoute,
    RunSnapshot,
    TerminalProposal,
    TerminalReconciliationReceipt,
    ToolPresentation,
    VersionedContextRef,
)
from plane_runtime.contract import OutcomeProposal, OutcomeSubmissionObserved, PROTOCOL
from plane_runtime.invocation_supervisor import (
    CleanupReport,
    EnforcementAttestation,
    InvocationPolicy,
    InvocationSupervisor,
    ProcessCapture,
    SubprocessDockerRunner,
    _SubprocessDockerProcess,
    _MAX_RETAINED_INVOCATIONS,
    _parse_child_output,
    build_invocation_argv,
    build_invocation_env,
    invocation_container_name,
)
from plane_runtime.service import serve_once_proposal_only


IMAGE = "registry.example/plane/runtime@sha256:" + "a" * 64
OTHER_IMAGE = "registry.example/plane/runtime@sha256:" + "b" * 64
NOW = datetime(2026, 8, 4, tzinfo=timezone.utc)


def make_snapshot(run_id: str = "run:supervisor") -> RunSnapshot:
    return RunSnapshot(
        protocol=PROTOCOL,
        run_id=run_id,
        assignment=AssignmentSnapshot(
            version="assignment:v1",
            target_ref="issue:one",
            objective="Produce one bounded result",
            acceptance_criteria=("The result is deterministic",),
        ),
        actor_ref="agent:one",
        workspace_ref="workspace:one",
        profile_version="profile:v1",
        behavioral_prompt="Use the bounded runtime.",
        context=(VersionedContextRef("context:one", "sha256:context"),),
        tool_presentation=ToolPresentation(
            eager_operations=(OperationDescriptor("operation:read", "sha256:operation"),),
            catalog_digest="sha256:catalog",
        ),
        model=RuntimeModelRoute("fake-model", "route:fake"),
        total_budget_policy=RuntimeBudgetPolicy(RuntimeBudget(5, 100, 100)),
        contract_digests=ContractDigests("snapshot:v1", "invocation:v1", "event:v1", "exit:v1"),
    )


def make_invocation(snapshot: RunSnapshot, invocation_id: str = "invocation:one") -> InvocationEnvelope:
    return InvocationEnvelope(
        protocol=PROTOCOL,
        invocation_id=invocation_id,
        run_id=snapshot.run_id,
        run_snapshot_digest=snapshot.digest(),
        trigger=InvocationTrigger("initial"),
        new_context_event_refs=(),
        checkpoint_ref=None,
        remaining_budget=snapshot.total_budget_policy.total,
        lease=RuntimeLease("lease:one", "host:one", "2099-01-01T00:00:00Z"),
        causation_ref="cause:one",
        cancellation_ref="cancel:one",
    )


def make_binding(snapshot: RunSnapshot, invocation: InvocationEnvelope) -> CanonicalLeaseBinding:
    return CanonicalLeaseBinding(
        snapshot.run_id,
        invocation.invocation_id,
        "lease:one",
        "host:one",
        True,
        invocation.lease.expires_at,
    )


def make_cancellation_binding(
    snapshot: RunSnapshot,
    invocation: InvocationEnvelope,
) -> CanonicalCancellationBinding:
    idempotency_key = f"cancel:{snapshot.run_id}:{invocation.invocation_id}"
    return CanonicalCancellationBinding(
        run_id=snapshot.run_id,
        invocation_id=invocation.invocation_id,
        actor_ref=snapshot.actor_ref,
        workspace_ref=snapshot.workspace_ref,
        snapshot_digest=snapshot.digest(),
        lease_id=invocation.lease.lease_id,
        lease_holder_ref=invocation.lease.holder_ref,
        lease_expires_at=invocation.lease.expires_at,
        cancellation_ref=invocation.cancellation_ref,
        idempotency_key=idempotency_key,
        gateway_receipt_ref=f"gateway:{idempotency_key}",
        audit_ref=f"audit:{idempotency_key}",
        receipt_ref=f"cancel-receipt:{invocation.invocation_id}",
    )


def make_proposal(snapshot: RunSnapshot, invocation: InvocationEnvelope) -> tuple[RuntimeEvent, TerminalProposal]:
    event = RuntimeEvent(
        protocol=PROTOCOL,
        run_id=snapshot.run_id,
        invocation_id=invocation.invocation_id,
        sequence=1,
        event_id="event:outcome",
        correlation_ref=invocation.causation_ref,
        idempotency_key="idempotency:outcome",
        body=OutcomeSubmissionObserved(
            submission_ref="submission:one",
            receipt_ref="proposal:outcome",
            content="bounded outcome",
        ),
    )
    proposal = TerminalProposal(
        run_id=snapshot.run_id,
        invocation_id=invocation.invocation_id,
        actor_ref=snapshot.actor_ref,
        workspace_ref=snapshot.workspace_ref,
        snapshot_digest=snapshot.digest(),
        kind="completed",
        final_sequence=1,
        evidence_event_ids=(event.event_id,),
        evidence_receipt_refs=(event.body.receipt_ref,),
        outcome_proposal=OutcomeProposal(
            submission_ref="submission:one",
            content="bounded outcome",
            event_id=event.event_id,
            proposal_receipt_ref=event.body.receipt_ref,
        ),
    )
    return event, proposal


def frame(kind: str, value: object) -> bytes:
    return (json.dumps({"type": kind, kind: value}, sort_keys=True, separators=(",", ":")) + "\n").encode()


def proposal_output(snapshot: RunSnapshot, invocation: InvocationEnvelope) -> bytes:
    event, proposal = make_proposal(snapshot, invocation)
    return b"".join(
        (
            frame("event", event.to_dict()),
            frame("proposal", proposal.to_dict()),
            frame("exit", RuntimeExit("completed", 1).to_dict()),
        )
    )


class FakeProcess:
    def __init__(self, capture: ProcessCapture, error: Exception | None = None) -> None:
        self.capture = capture
        self.error = error

    def collect(self, **kwargs) -> ProcessCapture:
        if self.error is not None:
            raise self.error
        return self.capture


class ExplodingKernel:
    def dispatch(self, request, emit, cancellation):
        del request, emit, cancellation
        raise RuntimeError("provider secret")


class RejectingPort:
    def __init__(self, response=None) -> None:
        self.proposals = []
        self.response = response

    def reconcile_terminal(self, proposal):
        self.proposals.append(proposal)
        if self.response is not None:
            return self.response
        return TerminalReconciliationReceipt(
            receipt_ref="rejected:" + proposal.idempotency_key,
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            kind=proposal.kind,
            idempotency_key=proposal.idempotency_key,
            accepted=False,
            legal_transition=False,
        )


class RaisingPort:
    def reconcile_terminal(self, proposal):
        del proposal
        raise RuntimeError("host reconciliation unavailable")


class FakeRunner:
    """Explicit test seam; its attestation is never production enforcement."""

    def __init__(self, process: FakeProcess | None = None, *, launch_error: Exception | None = None) -> None:
        self.process = process
        self.launch_error = launch_error
        self.argv: tuple[str, ...] | None = None
        self.launch_calls = 0
        self.cleanup_calls: list[str] = []
        self.attest_calls = 0
        self.cleanup_report = None

    def attest_invocation(self, argv, *, client_env):
        self.attest_calls += 1
        digest = hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()
        name = argv[argv.index("--name") + 1]
        return EnforcementAttestation("test", digest, name, ("injected-test-runner",))

    def launch(self, argv, *, client_env, input_bytes):
        self.launch_calls += 1
        self.argv = tuple(argv)
        if self.launch_error is not None:
            raise self.launch_error
        assert self.process is not None
        return self.process

    def cleanup(self, container_name, **kwargs):
        self.cleanup_calls.append(container_name)
        if self.cleanup_report is not None:
            return self.cleanup_report
        return CleanupReport(container_name, True, False, True, post_cleanup_absent=True)


class SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = make_snapshot()
        self.invocation = make_invocation(self.snapshot)
        self.policy = InvocationPolicy(IMAGE)

    def make_supervisor(
        self,
        output: bytes,
        *,
        runner: FakeRunner | object | None = None,
        port=None,
        cancellation_authority=None,
    ):
        docker = runner or FakeRunner(FakeProcess(ProcessCapture(0, stdout=output)))
        lease_binding = make_binding(self.snapshot, self.invocation)
        return (
            InvocationSupervisor(
                policy=self.policy,
                runner=docker,
                terminal_port=port or FixtureTerminalReconciliationPort(),
                lease_authority=FixtureCanonicalLeaseAuthority([lease_binding], clock=lambda: NOW),
                lease_binding=lease_binding,
                cancellation_authority=cancellation_authority,
            ),
            docker,
        )

    def test_argv_has_fixed_safe_controls_and_bounded_storage_logging_environment(self) -> None:
        argv = build_invocation_argv(self.snapshot, self.invocation, self.policy)
        self.assertEqual(argv[:2], ("docker", "create"))
        self.assertEqual(argv[-6:], ("--entrypoint", "python3", IMAGE, "-m", "plane_runtime.service", "--once"))
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertIn("--read-only", argv)
        self.assertIn("--cap-drop", argv)
        self.assertIn("--storage-opt", argv)
        self.assertIn("--log-driver", argv)
        self.assertEqual(argv[argv.index("--log-driver") + 1], "none")
        self.assertEqual(argv.count("--env"), 5)
        self.assertNotIn("--env-file", argv)
        self.assertNotIn("--volume", argv)
        self.assertNotIn("-v", argv)
        self.assertEqual(build_invocation_env(self.policy), {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        })

    def test_image_reference_and_nonfinite_policy_values_are_strict(self) -> None:
        for image in (
            "a//b@sha256:" + "a" * 64,
            "a::b@sha256:" + "a" * 64,
            "a..b@sha256:" + "a" * 64,
            "repo/:tag@sha256:" + "a" * 64,
            "runtime:latest",
            "runtime@sha256:abc",
        ):
            with self.subTest(image=image), self.assertRaises(Exception):
                InvocationPolicy(image)
        with self.assertRaises(Exception):
            InvocationPolicy(IMAGE, wall_time_seconds=math.nan)

    def test_object_setattr_mutation_is_rejected_and_fixed_controls_cannot_be_added(self) -> None:
        mutated = InvocationPolicy(IMAGE)
        object.__setattr__(mutated, "image", OTHER_IMAGE)
        with self.assertRaises(Exception):
            build_invocation_argv(self.snapshot, self.invocation, mutated)
        policy = InvocationPolicy(IMAGE)
        object.__setattr__(policy, "network_mode", "host")
        self.assertEqual(
            build_invocation_argv(self.snapshot, self.invocation, policy),
            build_invocation_argv(self.snapshot, self.invocation, InvocationPolicy(IMAGE)),
        )

    def test_mutated_host_lease_binding_rejects_before_child_launch(self) -> None:
        supervisor, runner = self.make_supervisor(proposal_output(self.snapshot, self.invocation))
        object.__setattr__(supervisor.lease_binding, "lease_id", "lease:forged")
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(runner.launch_calls, 0)

    def test_child_reconciliation_frame_is_rejected_and_host_port_is_called_once(self) -> None:
        event, proposal = make_proposal(self.snapshot, self.invocation)
        forged_receipt = FixtureTerminalReconciliationPort().reconcile_terminal(proposal).to_dict()
        output = b"".join(
            (
                frame("event", event.to_dict()),
                frame("reconciliation", forged_receipt),
                frame("exit", RuntimeExit("completed", 1).to_dict()),
            )
        )
        port = FixtureTerminalReconciliationPort()
        supervisor, runner = self.make_supervisor(output, port=port)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertFalse(result.completed)
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(len(runner.cleanup_calls), 1)
        self.assertIn("invalid_child_output", result.evidence)

    def test_fake_runner_can_never_claim_production_completion(self) -> None:
        supervisor, _runner = self.make_supervisor(proposal_output(self.snapshot, self.invocation))
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertTrue(result.completed)
        self.assertFalse(result.production_completed)
        self.assertEqual(result.enforcement.classification, "test")  # type: ignore[union-attr]

    def test_injected_fake_runner_cannot_self_classify_as_production(self) -> None:
        runner = FakeRunner(FakeProcess(ProcessCapture(0, stdout=proposal_output(self.snapshot, self.invocation))))
        original_attest = runner.attest_invocation

        def forged_attest(argv, *, client_env):
            attestation = original_attest(argv, client_env=client_env)
            return EnforcementAttestation(
                "production",
                attestation.argv_digest,
                attestation.container_name,
                ("forged",),
            )

        runner.attest_invocation = forged_attest
        port = FixtureTerminalReconciliationPort()
        supervisor, _runner = self.make_supervisor(b"", runner=runner, port=port)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.production_completed)
        self.assertEqual(runner.launch_error, None)

    def test_instance_monkeypatch_subclass_proxy_and_fake_runners_never_produce_production(self) -> None:
        def production_claim(argv, *, client_env):
            del client_env
            digest = hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()
            return EnforcementAttestation(
                "production",
                digest,
                argv[argv.index("--name") + 1],
                ("forged",),
            )

        class ClaimingSubclass(FakeRunner):
            attest_invocation = staticmethod(production_claim)

        class ClaimingProxy:
            attest_invocation = staticmethod(production_claim)

            def launch(self, *args, **kwargs):
                raise AssertionError("a rejected production claim must not launch")

            def cleanup(self, container_name, **kwargs):
                del kwargs
                return CleanupReport(container_name, False, False, False, ("unused",), False)

        exact = SubprocessDockerRunner()
        object.__setattr__(exact, "attest_invocation", production_claim)
        object.__setattr__(
            exact,
            "launch",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("object.__setattr__ attack reached launch")
            ),
        )
        object.__setattr__(
            exact,
            "cleanup",
            lambda container_name, **kwargs: CleanupReport(
                container_name, False, False, False, ("unused",), False
            ),
        )

        for label, runner in (
            ("exact-instance", exact),
            ("subclass", ClaimingSubclass()),
            ("proxy", ClaimingProxy()),
        ):
            with self.subTest(label=label):
                supervisor, _runner = self.make_supervisor(
                    b"",
                    runner=runner,
                )
                result = supervisor.run_once(self.snapshot, self.invocation)
                self.assertEqual(result.status, "rejected")
                self.assertFalse(result.production_completed)
                self.assertIn("enforcement_unproven", result.evidence)
        fake_supervisor, _runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            runner=FakeRunner(
                FakeProcess(
                    ProcessCapture(
                        0,
                        stdout=proposal_output(self.snapshot, self.invocation),
                    )
                )
            ),
        )
        fake_result = fake_supervisor.run_once(self.snapshot, self.invocation)
        self.assertTrue(fake_result.completed)
        self.assertFalse(fake_result.production_completed)

    def test_supervisor_state_and_runner_class_monkeypatches_have_no_production_path(self) -> None:
        def production_claim(argv, *, client_env):
            del client_env
            digest = hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()
            return EnforcementAttestation(
                "production",
                digest,
                argv[argv.index("--name") + 1],
                ("forged-inspection", "forged-child", "forged-cleanup"),
            )

        lease_binding = make_binding(self.snapshot, self.invocation)
        port = FixtureTerminalReconciliationPort()
        unconfigured = InvocationSupervisor(
            policy=self.policy,
            runner=None,
            terminal_port=port,
            lease_authority=FixtureCanonicalLeaseAuthority([lease_binding], clock=lambda: NOW),
            lease_binding=lease_binding,
        )
        # These were the rejected design's production switches. Adding them
        # back to the instance must not create a path that the supervisor uses.
        object.__setattr__(unconfigured, "_production_runner", object())
        object.__setattr__(unconfigured, "_runner_is_injected", False)
        result = unconfigured.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.production_completed)
        self.assertIn("enforcement_unproven", result.evidence)
        self.assertEqual(port.proposals, [])

        launch_calls = 0

        def forbidden_launch(*args, **kwargs):
            nonlocal launch_calls
            del args, kwargs
            launch_calls += 1
            raise AssertionError("a rejected production claim must not launch")

        with (
            patch.object(SubprocessDockerRunner, "attest_invocation", staticmethod(production_claim)),
            patch.object(SubprocessDockerRunner, "launch", forbidden_launch),
        ):
            patched_runner = SubprocessDockerRunner()
            patched_supervisor, _runner = self.make_supervisor(
                proposal_output(self.snapshot, self.invocation),
                runner=patched_runner,
                port=port,
            )
            patched_result = patched_supervisor.run_once(self.snapshot, self.invocation)

        self.assertEqual(patched_result.status, "rejected")
        self.assertFalse(patched_result.production_completed)
        self.assertEqual(launch_calls, 0)
        self.assertEqual(port.proposals, [])

    def test_object_setattr_results_replacement_and_forged_subclass_are_ignored(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, _runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=port,
        )

        class ForgedResult(supervisor_module._EXACT_INVOCATION_RESULT):
            @property
            def production_completed(self):
                return True

        name = invocation_container_name(self.snapshot, self.invocation)
        forged = ForgedResult(
            status="completed",
            container_name=name,
            enforcement=EnforcementAttestation(
                "production",
                "0" * 64,
                name,
                ("forged",),
            ),
        )
        object.__setattr__(supervisor, "_results", {
            (self.snapshot.run_id, self.invocation.invocation_id): forged,
        })
        first = supervisor.run_once(self.snapshot, self.invocation)
        object.__setattr__(supervisor, "_results", {
            (self.snapshot.run_id, self.invocation.invocation_id): forged,
        })
        second = supervisor.run_once(self.snapshot, self.invocation)
        self.assertNotIsInstance(second, ForgedResult)
        self.assertFalse(second.production_completed)
        self.assertEqual(second, first)
        self.assertEqual(len(port.proposals), 1)

    def test_module_result_symbol_replacement_before_and_after_construction_is_ignored(self) -> None:
        class PatchedResult(supervisor_module._EXACT_INVOCATION_RESULT):
            @property
            def production_completed(self):
                return True

        before_port = FixtureTerminalReconciliationPort()
        before_supervisor, _runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=before_port,
        )
        with patch.object(supervisor_module, "InvocationResult", PatchedResult):
            after = before_supervisor.run_once(self.snapshot, self.invocation)
        with patch.object(supervisor_module, "InvocationResult", PatchedResult):
            after_read = before_supervisor.run_once(self.snapshot, self.invocation)
        self.assertNotIsInstance(after, PatchedResult)
        self.assertNotIsInstance(after_read, PatchedResult)
        self.assertFalse(after.production_completed)
        self.assertEqual(after_read, after)
        self.assertEqual(len(before_port.proposals), 1)

        inside_port = FixtureTerminalReconciliationPort()
        with patch.object(supervisor_module, "InvocationResult", PatchedResult):
            inside_supervisor, _runner = self.make_supervisor(
                proposal_output(self.snapshot, self.invocation),
                port=inside_port,
            )
            before = inside_supervisor.run_once(self.snapshot, self.invocation)
        self.assertNotIsInstance(before, PatchedResult)
        self.assertFalse(before.production_completed)
        self.assertEqual(len(inside_port.proposals), 1)

    def test_legacy_ledger_and_constructor_capture_surfaces_are_not_runtime_authority(self) -> None:
        self.assertFalse(hasattr(supervisor_module, "_SUPERVISOR_STATES"))
        self.assertFalse(hasattr(supervisor_module, "_new_invocation_result"))
        self.assertIsNone(supervisor_module._draft_result.__defaults__)
        self.assertIsNone(supervisor_module._draft_result.__kwdefaults__)

        original_defaults = supervisor_module._draft_result.__kwdefaults__
        supervisor_module._draft_result.__kwdefaults__ = {"status": "completed"}
        try:
            with patch.object(supervisor_module.InvocationResult, "__dataclass_fields__", {}):
                supervisor, _runner = self.make_supervisor(
                    proposal_output(self.snapshot, self.invocation)
                )
                result = supervisor.run_once(self.snapshot, self.invocation)
        finally:
            supervisor_module._draft_result.__kwdefaults__ = original_defaults
        self.assertTrue(result.completed)
        self.assertFalse(result.production_completed)

    def test_returned_value_class_metadata_is_not_shared_with_canonical_readback(self) -> None:
        supervisor, _runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation)
        )
        first = supervisor.run_once(self.snapshot, self.invocation)
        type(first).production_completed = property(lambda _result: True)
        second = supervisor.run_once(self.snapshot, self.invocation)
        self.assertTrue(first.production_completed)
        self.assertFalse(second.production_completed)
        self.assertTrue(second.completed)
        self.assertIsNot(type(first), type(second))

    def test_reachable_sqlite_row_and_digest_replacement_cannot_forge_readback(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, _runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=port,
        )
        first = supervisor.run_once(self.snapshot, self.invocation)
        key = (self.snapshot.run_id, self.invocation.invocation_id)
        authority = supervisor._retention_authority
        self.assertFalse(hasattr(supervisor, "_retention_anchor"))
        self.assertFalse(hasattr(supervisor, "_retention_uri"))
        self.assertFalse(hasattr(supervisor, "_results"))
        canonical = dict(authority.read("result", key)["value"])
        canonical["evidence"] = [*canonical["evidence"], "forged-but-well-shaped"]
        self.assertEqual(authority.create("result", key, canonical)["status"], "conflict")
        canonical["payloadDigest"] = "0" * 64
        self.assertEqual(authority.create("result", key, canonical)["status"], "conflict")
        second = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(second, first)
        self.assertEqual(second.evidence, ("host_reconciliation", "enforcement_attested"))
        self.assertFalse(second.production_completed)
        self.assertEqual(len(port.proposals), 1)
        self.assertTrue(first.completed)

    def test_parent_authority_replacement_before_terminalization_fails_closed(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=port,
        )
        authority = supervisor._retention_authority
        original_process = authority._process
        object.__setattr__(supervisor, "_retention_authority", object())
        result = supervisor.run_once(self.snapshot, self.invocation)
        supervisor.close()
        self.assertEqual(result.status, "supervisor_action_required")
        self.assertIn("retained_result_invalid", result.evidence)
        self.assertEqual(runner.launch_calls, 0)
        self.assertEqual(port.proposals, [])
        self.assertEqual(original_process.returncode, 0)

    def test_parent_authority_handle_replacements_fail_closed_and_reap_original(self) -> None:
        before_port = FixtureTerminalReconciliationPort()
        before, before_runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=before_port,
        )
        before_authority = before._retention_authority
        before_process = before_authority._process
        object.__setattr__(before_authority, "_process", object())
        before_result = before.run_once(self.snapshot, self.invocation)
        before.close()
        self.assertEqual(before_result.status, "supervisor_action_required")
        self.assertEqual(before_runner.launch_calls, 0)
        self.assertIsNotNone(before_process.poll())

        after_port = FixtureTerminalReconciliationPort()
        after, after_runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=after_port,
        )
        self.assertEqual(after.run_once(self.snapshot, self.invocation).status, "completed")
        after_authority = after._retention_authority
        after_process = after_authority._process
        object.__setattr__(after_authority, "_stdout", object())
        after_result = after.run_once(self.snapshot, self.invocation)
        after.close()
        self.assertEqual(after_result.status, "supervisor_action_required")
        self.assertEqual(after_runner.launch_calls, 1)
        self.assertEqual(len(after_port.proposals), 1)
        self.assertIsNotNone(after_process.poll())

    def test_all_reachable_authority_handles_replace_fail_closed_and_reap_original(self) -> None:
        supervisor, runner = self.make_supervisor(b"")
        authority = supervisor._retention_authority
        original_process = authority._process
        replacements = {
            "_process": object(),
            "_process_identity": object(),
            "_stdin": object(),
            "_stdin_identity": object(),
            "_stdout": object(),
            "_stdout_identity": object(),
            "_stdin_fd": -1,
            "_stdout_fd": -1,
            "_stdin_type": object,
            "_stdout_type": object,
            "_lock": object(),
            "_closed": False,
        }
        for name, value in replacements.items():
            object.__setattr__(authority, name, value)
        result = supervisor.run_once(self.snapshot, self.invocation)
        supervisor.close()
        self.assertEqual(result.status, "supervisor_action_required")
        self.assertEqual(runner.launch_calls, 0)
        self.assertIsNotNone(original_process.poll())

        after, after_runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
        )
        self.assertEqual(after.run_once(self.snapshot, self.invocation).status, "completed")
        after_authority = after._retention_authority
        after_process = after_authority._process
        for name, value in replacements.items():
            object.__setattr__(after_authority, name, value)
        after_result = after.run_once(self.snapshot, self.invocation)
        after.close()
        self.assertEqual(after_result.status, "supervisor_action_required")
        self.assertEqual(after_runner.launch_calls, 1)
        self.assertIsNotNone(after_process.poll())

    def test_post_construction_transport_helper_replacement_is_ignored_by_endpoint(self) -> None:
        authority = supervisor_module._RetentionAuthority()
        try:
            with patch.object(
                supervisor_module,
                "_retention_read_frame",
                side_effect=AssertionError("replaceable reader was called"),
            ):
                self.assertEqual(authority.count_results(), 0)
        finally:
            authority.close()

    def test_authority_death_after_terminalization_fails_closed(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=port,
        )
        first = supervisor.run_once(self.snapshot, self.invocation)
        authority_process = supervisor._retention_authority._process
        authority_process.kill()
        authority_process.wait(timeout=2)
        second = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "supervisor_action_required")
        self.assertFalse(second.production_completed)
        self.assertEqual(runner.launch_calls, 1)
        self.assertEqual(len(port.proposals), 1)

    def test_explicit_close_reaps_authority_and_is_idempotent(self) -> None:
        supervisor, _runner = self.make_supervisor(b"")
        authority_process = supervisor._retention_authority._process
        started = time.monotonic()
        supervisor.close()
        elapsed = time.monotonic() - started
        supervisor.close()
        self.assertIsNotNone(authority_process.poll())
        self.assertEqual(authority_process.returncode, 0)
        self.assertLess(elapsed, 1.0)

    def test_deadline_reader_rejects_stall_oversize_eof_and_wrong_id(self) -> None:
        def read_from_peer(payload: bytes, *, limit: int, delay: float = 0.0) -> None:
            read_fd, write_fd = os.pipe()
            os.set_blocking(read_fd, False)

            def writer() -> None:
                try:
                    os.write(write_fd, payload)
                    if delay:
                        time.sleep(delay)
                finally:
                    os.close(write_fd)

            thread = threading.Thread(target=writer)
            thread.start()
            try:
                with self.assertRaises(Exception):
                    supervisor_module._retention_read_frame(
                        read_fd,
                        time.monotonic() + 0.05,
                        limit,
                    )
            finally:
                os.close(read_fd)
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())

        read_from_peer(b"{", limit=128, delay=0.2)
        read_from_peer(b"x" * 129, limit=128)
        read_from_peer(b"{", limit=128)

        key = b"retention-test-key"
        response_without_mac = {
            "protocol": supervisor_module._RETENTION_PROTOCOL,
            "status": "missing",
            "record": "result",
            "requestId": 1,
            "runId": "run:one",
            "invocationId": "invocation:one",
        }
        response = {
            **response_without_mac,
            "mac": supervisor_module._retention_message_mac(key, response_without_mac),
        }
        with self.assertRaises(supervisor_module.RuntimeConfigurationError):
            supervisor_module._retention_decode_response(
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode() + b"\n",
                request={
                    "op": "read",
                    "record": "result",
                    "runId": "run:one",
                    "invocationId": "invocation:one",
                },
                request_id=2,
                session_key=key,
            )

    def test_spawn_seam_replacement_fails_closed_before_ordinary_process_launch(self) -> None:
        launched = False

        def substituted_popen(*args, **kwargs):
            nonlocal launched
            launched = True
            return supervisor_module._RETENTION_POPEN(*args, **kwargs)

        with patch.object(supervisor_module.subprocess, "Popen", substituted_popen):
            with self.assertRaises(supervisor_module.RuntimeConfigurationError):
                supervisor_module._RetentionAuthority()
        self.assertFalse(launched)

    def test_coordinated_spawn_replacement_cannot_authenticate_ordinary_peer(self) -> None:
        original_popen = supervisor_module._RETENTION_POPEN
        source_digest = supervisor_module._RETENTION_SOURCE_DIGEST
        source_path = supervisor_module._RETENTION_SOURCE_PATH
        peer = r'''
import hashlib
import hmac
import json
import sys

source_digest, source_path = sys.argv[1:]
source_key = hmac.new(
    b"plane-agent-retention-session/v1",
    source_digest.encode("ascii"),
    hashlib.sha256,
).digest()

def mac(key, message):
    return hmac.new(key, json.dumps(message, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()

auth_request = json.loads(sys.stdin.buffer.readline())
session_key = hmac.new(source_key, b"session:" + bytes.fromhex(auth_request["nonce"]), hashlib.sha256).digest()
auth_response = {
    "protocol": auth_request["protocol"],
    "status": "ready",
    "record": "auth",
    "requestId": auth_request["requestId"],
    "runId": "",
    "invocationId": "",
    "proof": hmac.new(source_key, bytes.fromhex(auth_request["nonce"]), hashlib.sha256).hexdigest(),
    "sourceDigest": source_digest,
    "sourcePath": source_path,
}
auth_response["mac"] = mac(session_key, auth_response)
sys.stdout.write(json.dumps(auth_response, sort_keys=True, separators=(",", ":")) + "\n")
sys.stdout.flush()
count_request = json.loads(sys.stdin.buffer.readline())
count_response = {
    "protocol": count_request["protocol"],
    "status": "ok",
    "record": "result_count",
    "requestId": count_request["requestId"],
    "runId": "",
    "invocationId": "",
    "value": 424242,
}
count_response["mac"] = mac(session_key, count_response)
sys.stdout.write(json.dumps(count_response, sort_keys=True, separators=(",", ":")) + "\n")
sys.stdout.flush()
'''
        launched: list[subprocess.Popen[bytes]] = []

        class CoordinatedPopen(original_popen):
            def __init__(self, _command, **kwargs):
                launched.append(self)
                original_popen.__init__(
                    self,
                    (sys.executable, "-c", peer, source_digest, source_path),
                    **kwargs,
                )

        try:
            with patch.object(supervisor_module.subprocess, "Popen", CoordinatedPopen), patch.object(
                supervisor_module, "_RETENTION_POPEN", CoordinatedPopen
            ):
                with self.assertRaises(supervisor_module.RuntimeConfigurationError):
                    supervisor_module._RetentionAuthority()
            self.assertEqual(launched, [])
        finally:
            for process in launched:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=2)

    def test_post_construction_registry_and_helper_replacement_cannot_block_close(self) -> None:
        authority = supervisor_module._RetentionAuthority()
        process = authority._process
        try:
            supervisor_module._RETENTION_FINALIZERS = weakref.WeakKeyDictionary()
            with patch.object(
                supervisor_module,
                "_close_retention_process",
                side_effect=AssertionError("replaceable cleanup helper was called"),
                create=True,
            ):
                authority.close()
            self.assertEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            supervisor_module.__dict__.pop("_RETENTION_FINALIZERS", None)

    def test_authority_constructor_has_no_dependency_injection_surface(self) -> None:
        self.assertEqual(tuple(inspect.signature(supervisor_module._RetentionAuthority).parameters), ())
        self.assertEqual(
            tuple(inspect.signature(supervisor_module._RetentionAuthority.__new__).parameters),
            ("cls",),
        )
        former_dependency_names = (
            "_trusted_popen",
            "_trusted_executable",
            "_trusted_module_name",
            "_trusted_source_path",
            "_trusted_source_digest",
            "_trusted_validator",
            "_trusted_make_endpoint",
            "_trusted_write_frame",
            "_trusted_close_process",
            "_trusted_token_bytes",
            "_trusted_getpid",
            "_trusted_set_blocking",
            "_trusted_pipe",
            "_trusted_devnull",
            "_trusted_monotonic",
            "_trusted_finalize",
        )
        launched: list[object] = []

        def fail_if_launched(*args, **kwargs):
            launched.append((args, kwargs))
            raise AssertionError("dependency-injection attempt launched a process")

        with patch.object(supervisor_module.subprocess, "Popen", fail_if_launched):
            for name in former_dependency_names:
                supplied_calls: list[object] = []

                def supplied(*args, **kwargs):
                    supplied_calls.append((args, kwargs))
                    raise AssertionError("caller-supplied dependency was invoked")

                with self.subTest(name=name), self.assertRaises(TypeError):
                    supervisor_module._RetentionAuthority(**{name: supplied})
                self.assertEqual(supplied_calls, [])
            with self.assertRaises(TypeError):
                supervisor_module._RetentionAuthority(object())
        self.assertEqual(launched, [])

    def test_supervisor_uses_import_time_authority_constructor_not_replaced_globals(self) -> None:
        original_authority = supervisor_module._RetentionAuthority
        original_test_factory = supervisor_module._make_retention_authority_for_test
        replacement_calls: list[str] = []
        supplied_endpoint_calls: list[object] = []
        replaced_test_factory_calls: list[object] = []

        def supplied_endpoint(*args, **kwargs):
            supplied_endpoint_calls.append((args, kwargs))
            raise AssertionError("supplied test endpoint was invoked")

        class ReplacementAuthority(original_authority):
            def __new__(cls):
                replacement_calls.append("constructor")
                return original_test_factory(make_endpoint=supplied_endpoint)

        def replaced_test_factory(*args, **kwargs):
            replaced_test_factory_calls.append((args, kwargs))
            raise AssertionError("replaced test factory was invoked")

        original_constructor = supervisor_module._RETENTION_AUTHORITY_PRODUCTION_CONSTRUCTOR
        try:
            supervisor_module._RETENTION_AUTHORITY_PRODUCTION_CONSTRUCTOR = replaced_test_factory
            with (
                patch.object(supervisor_module, "_RetentionAuthority", ReplacementAuthority),
                patch.object(supervisor_module, "_make_retention_authority_for_test", replaced_test_factory),
                patch.object(supervisor_module, "_retention_authority_test_harness", replaced_test_factory),
                patch.object(
                    supervisor_module,
                    "_make_production_retention_authority_constructor",
                    replaced_test_factory,
                ),
            ):
                supervisor, _runner = self.make_supervisor(b"")
                authority_process = supervisor._retention_authority._process
                self.assertIsInstance(supervisor._retention_authority, original_authority)
                self.assertNotIsInstance(supervisor._retention_authority, ReplacementAuthority)
                supervisor.close()
                self.assertIsNotNone(authority_process.poll())
        finally:
            supervisor_module._RETENTION_AUTHORITY_PRODUCTION_CONSTRUCTOR = original_constructor

        self.assertEqual(replacement_calls, [])
        self.assertEqual(supplied_endpoint_calls, [])
        self.assertEqual(replaced_test_factory_calls, [])

    def test_authority_construction_failure_reaps_bootstrapped_child(self) -> None:
        captured: list[subprocess.Popen[bytes]] = []

        def fail_after_launch(*args, **kwargs):
            captured.append(args[1])
            raise RuntimeError("endpoint construction failed")

        with self.assertRaises(supervisor_module.RuntimeConfigurationError):
            supervisor_module._make_retention_authority_for_test(make_endpoint=fail_after_launch)
        self.assertEqual(len(captured), 1)
        self.assertIsNotNone(captured[0].poll())

    def test_source_executable_and_command_replacements_fail_closed_before_launch(self) -> None:
        replacements = (
            (supervisor_module.sys, "executable", "/tmp/ordinary-python"),
            (supervisor_module, "_RETENTION_MODULE_NAME", "ordinary_module"),
            (supervisor_module, "_RETENTION_SOURCE_PATH", "/tmp/ordinary-source.py"),
            (supervisor_module, "_RETENTION_SOURCE_DIGEST", "0" * 64),
        )
        for target, name, value in replacements:
            with self.subTest(name=name), patch.object(target, name, value):
                with self.assertRaises(supervisor_module.RuntimeConfigurationError):
                    supervisor_module._RetentionAuthority()

    def test_structural_authority_finalizer_reaps_after_handle_replacement(self) -> None:
        authority = supervisor_module._RetentionAuthority()
        process = authority._process
        object.__setattr__(authority, "_process", object())
        object.__setattr__(authority, "_stdin", object())
        object.__setattr__(authority, "_stdout", object())
        del authority
        for _ in range(100):
            gc.collect()
            if process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertEqual(process.returncode, 0)

    def test_concurrent_close_and_requests_reap_one_original_authority(self) -> None:
        supervisor, _runner = self.make_supervisor(b"")
        original_process = supervisor._retention_authority._process
        barrier = threading.Barrier(20)
        errors: list[Exception] = []
        threads: list[threading.Thread] = []

        def request() -> None:
            try:
                barrier.wait(timeout=2)
                supervisor.run_once(self.snapshot, self.invocation)
            except Exception as exc:
                errors.append(exc)

        def close() -> None:
            try:
                barrier.wait(timeout=2)
                supervisor.close()
            except Exception as exc:
                errors.append(exc)

        for index in range(10):
            threads.extend((threading.Thread(target=request), threading.Thread(target=close)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        supervisor.close()
        self.assertIsNotNone(original_process.poll())

    def test_returned_result_mutation_cannot_change_retained_nonproduction_truth(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, _runner = self.make_supervisor(
            proposal_output(self.snapshot, self.invocation),
            port=port,
        )
        first = supervisor.run_once(self.snapshot, self.invocation)
        with self.assertRaises(AttributeError):
            object.__setattr__(first, "enforcement", EnforcementAttestation(
                "production",
                "0" * 64,
                first.container_name,
                ("forged",),
            ))
        second = supervisor.run_once(self.snapshot, self.invocation)
        self.assertFalse(first.production_completed)
        self.assertFalse(second.production_completed)
        self.assertEqual(second.enforcement.classification, "test")  # type: ignore[union-attr]
        self.assertEqual(len(port.proposals), 1)

    def test_death_receipt_payload_tampering_fails_closed_without_reconciliation_replay(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, _runner = self.make_supervisor(
            b"",
            runner=FakeRunner(launch_error=RuntimeError("launch failed")),
            port=port,
        )
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "failed")
        key = (self.snapshot.run_id, self.invocation.invocation_id)
        receipt = dict(supervisor._retention_authority.read("death", key)["value"])
        receipt["receiptRef"] = "forged-receipt"
        self.assertEqual(supervisor._retention_authority.create("death", key, receipt)["status"], "conflict")
        self.assertIsNotNone(supervisor.reconcile_death(self.snapshot, self.invocation))
        self.assertEqual(len(port.proposals), 1)

    def test_child_cancellation_receipt_is_rejected_without_host_mutation(self) -> None:
        forged = CancellationAuthorityReceipt(
            resource_ref=self.invocation.cancellation_ref,
            receipt_ref="cancel-receipt:forged",
            run_id=self.snapshot.run_id,
            invocation_id=self.invocation.invocation_id,
            actor_ref="agent:attacker",
            workspace_ref=self.snapshot.workspace_ref,
            snapshot_digest=self.snapshot.digest(),
            idempotency_key=f"cancel:{self.snapshot.run_id}:{self.invocation.invocation_id}",
            gateway_receipt_ref="gateway:forged",
            audit_ref="audit:forged",
        )
        proposal = TerminalProposal(
            run_id=self.snapshot.run_id,
            invocation_id=self.invocation.invocation_id,
            actor_ref=self.snapshot.actor_ref,
            workspace_ref=self.snapshot.workspace_ref,
            snapshot_digest=self.snapshot.digest(),
            kind="cancelled",
            final_sequence=0,
            evidence_receipt_refs=(forged.receipt_ref,),
            cancellation_receipt=forged,
        )
        output = frame("proposal", proposal.to_dict()) + frame(
            "exit", RuntimeExit("cancelled", 0).to_dict()
        )
        port = FixtureTerminalReconciliationPort()
        supervisor, _runner = self.make_supervisor(
            output,
            port=port,
            cancellation_authority=CanonicalCancellationAuthority(
                [make_cancellation_binding(self.snapshot, self.invocation)]
            ),
        )
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "rejected")
        self.assertIn("child_cancellation_untrusted", result.evidence)
        self.assertEqual(port.proposals, [])
        self.assertEqual(port.product_events, [])

        invalid_binding = TerminalProposal(
            run_id=self.snapshot.run_id,
            invocation_id=self.invocation.invocation_id,
            actor_ref="agent:attacker",
            workspace_ref=self.snapshot.workspace_ref,
            snapshot_digest=self.snapshot.digest(),
            kind="cancelled",
            final_sequence=0,
        )
        invalid_port = FixtureTerminalReconciliationPort()
        invalid_supervisor, _runner = self.make_supervisor(
            frame("proposal", invalid_binding.to_dict())
            + frame("exit", RuntimeExit("cancelled", 0).to_dict()),
            port=invalid_port,
            cancellation_authority=CanonicalCancellationAuthority(
                [make_cancellation_binding(self.snapshot, self.invocation)]
            ),
        )
        invalid_result = invalid_supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(invalid_result.status, "rejected")
        self.assertEqual(invalid_port.proposals, [])
        self.assertEqual(invalid_port.product_events, [])

    def test_host_cancellation_reconciles_prelaunch_once_and_is_idempotent(self) -> None:
        cancellation = MutableCancellation()
        cancellation.cancel()
        port = FixtureTerminalReconciliationPort()
        authority = CanonicalCancellationAuthority(
            [make_cancellation_binding(self.snapshot, self.invocation)]
        )
        supervisor, runner = self.make_supervisor(
            b"",
            port=port,
            cancellation_authority=authority,
        )
        first = supervisor.run_once(self.snapshot, self.invocation, cancellation=cancellation)
        second = supervisor.run_once(self.snapshot, self.invocation, cancellation=cancellation)
        self.assertEqual(first.status, "cancelled")
        # Public results are fresh views; idempotency is the single host event.
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(runner.launch_calls, 0)
        self.assertEqual(port.product_events, [("terminal:run:supervisor:invocation:one", "cancelled")])
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(
            first.proposal.cancellation_receipt.gateway_receipt_ref,  # type: ignore[union-attr]
            "gateway:cancel:run:supervisor:invocation:one",
        )

    def test_host_cancellation_reconciles_active_stop_before_child_output(self) -> None:
        cancellation = MutableCancellation()
        cancellation.cancel()
        port = FixtureTerminalReconciliationPort()
        authority = CanonicalCancellationAuthority(
            [make_cancellation_binding(self.snapshot, self.invocation)]
        )
        runner = FakeRunner(FakeProcess(ProcessCapture(None, cancelled=True)))
        supervisor, _runner = self.make_supervisor(
            b"",
            runner=runner,
            port=port,
            cancellation_authority=authority,
        )
        result = supervisor.run_once(self.snapshot, self.invocation, cancellation=cancellation)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(port.product_events, [("terminal:run:supervisor:invocation:one", "cancelled")])

    def test_result_cache_never_exceeds_bound_and_overflow_is_not_inserted(self) -> None:
        supervisor, _runner = self.make_supervisor(b"")
        retained = []
        for index in range(_MAX_RETAINED_INVOCATIONS):
            invocation = make_invocation(self.snapshot, f"invocation:hostile-{index}")
            retained.append(supervisor.run_once(self.snapshot, invocation))
        self.assertEqual(supervisor._retention_authority.count_results(), _MAX_RETAINED_INVOCATIONS)
        overflow_invocation = make_invocation(self.snapshot, "invocation:hostile-overflow")
        overflow = supervisor.run_once(self.snapshot, overflow_invocation)
        self.assertEqual(overflow.status, "supervisor_action_required")
        self.assertIn("result_not_retained", overflow.evidence)
        self.assertEqual(
            supervisor._retention_authority.read(
                "result", (self.snapshot.run_id, overflow_invocation.invocation_id)
            )["status"],
            "missing",
        )
        self.assertEqual(supervisor._retention_authority.count_results(), _MAX_RETAINED_INVOCATIONS)
        self.assertEqual(
            supervisor.run_once(self.snapshot, make_invocation(self.snapshot, "invocation:hostile-0")),
            retained[0],
        )

    def test_attach_timeout_terminates_and_reaps_sleeping_child(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            capture = _SubprocessDockerProcess(process).collect(
                input_bytes=b"",
                deadline=time.monotonic() + 0.05,
                stdout_limit_bytes=1024,
                stderr_limit_bytes=1024,
                is_cancelled=lambda: False,
            )
            self.assertTrue(capture.timed_out)
            self.assertTrue(capture.reaped)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)

    def test_host_rejection_is_non_success_after_one_validated_proposal(self) -> None:
        port = RejectingPort()
        supervisor, _runner = self.make_supervisor(proposal_output(self.snapshot, self.invocation), port=port)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "supervisor_action_required")
        self.assertFalse(result.completed)
        self.assertEqual(len(port.proposals), 1)

    def test_host_exception_or_invalid_receipt_is_non_success(self) -> None:
        for port in (RaisingPort(), RejectingPort(response=None)):
            with self.subTest(port=type(port).__name__):
                if isinstance(port, RejectingPort):
                    port.response = object()
                supervisor, _runner = self.make_supervisor(proposal_output(self.snapshot, self.invocation), port=port)
                result = supervisor.run_once(self.snapshot, self.invocation)
                self.assertEqual(result.status, "supervisor_action_required")
                self.assertFalse(result.completed)

    def test_launch_exception_always_reconciles_failure_and_cleans_deterministic_name(self) -> None:
        runner = FakeRunner(launch_error=RuntimeError("create failed"))
        supervisor, runner = self.make_supervisor(b"", runner=runner)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "failed")
        self.assertEqual(runner.cleanup_calls, [invocation_container_name(self.snapshot, self.invocation)])
        self.assertTrue(result.cleanup.succeeded)  # type: ignore[union-attr]

    def test_cleanup_failure_never_completes_or_hides_supervisor_action(self) -> None:
        runner = FakeRunner(FakeProcess(ProcessCapture(0, stdout=proposal_output(self.snapshot, self.invocation))))
        name = invocation_container_name(self.snapshot, self.invocation)
        runner.cleanup_report = CleanupReport(name, True, True, True, ("remove_failed",), False)
        supervisor, _ = self.make_supervisor(b"", runner=runner)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "supervisor_action_required")
        self.assertFalse(result.completed)

    def test_hostile_process_outcomes_all_reconcile_once_and_never_complete(self) -> None:
        captures = (
            ("timeout", ProcessCapture(None, timed_out=True), None),
            ("cancelled", ProcessCapture(None, cancelled=True), None),
            ("output", ProcessCapture(None, output_exceeded=True), None),
            ("nonzero", ProcessCapture(17), None),
            ("reader", ProcessCapture(None), RuntimeError("reader failed")),
        )
        for label, capture, error in captures:
            with self.subTest(label=label):
                port = FixtureTerminalReconciliationPort()
                runner = FakeRunner(FakeProcess(capture, error))
                supervisor, _runner = self.make_supervisor(b"", runner=runner, port=port)
                result = supervisor.run_once(self.snapshot, self.invocation)
                self.assertFalse(result.completed)
                self.assertEqual(result.status, "failed")
                self.assertEqual(len(port.proposals), 1)
                self.assertTrue(result.cleanup.succeeded)  # type: ignore[union-attr]

    def test_already_stopped_cleanup_is_idempotent_for_test_runner(self) -> None:
        runner = FakeRunner(FakeProcess(ProcessCapture(0, stdout=b"bad\n")))
        name = invocation_container_name(self.snapshot, self.invocation)
        runner.cleanup_report = CleanupReport(name, False, False, True, post_cleanup_absent=True)
        supervisor, _ = self.make_supervisor(b"", runner=runner)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.cleanup.succeeded)  # type: ignore[union-attr]

    def test_frame_state_machine_rejects_reordered_exit_duplicate_and_trailing_frames(self) -> None:
        event, proposal = make_proposal(self.snapshot, self.invocation)
        exit_frame = frame("exit", RuntimeExit("completed", 1).to_dict())
        proposal_frame = frame("proposal", proposal.to_dict())
        event_frame = frame("event", event.to_dict())
        for output in (
            exit_frame + proposal_frame + event_frame,
            event_frame + proposal_frame + exit_frame + event_frame,
            event_frame + proposal_frame + exit_frame + frame("reconciliation", {}),
        ):
            with self.subTest(output=output):
                with self.assertRaises(Exception):
                    _parse_child_output(output, run=self.snapshot, invocation=self.invocation, policy=self.policy)

    def test_twenty_concurrent_calls_submit_one_host_reconciliation(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, _runner = self.make_supervisor(proposal_output(self.snapshot, self.invocation), port=port)
        results = []
        threads = [threading.Thread(target=lambda: results.append(supervisor.run_once(self.snapshot, self.invocation))) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(results), 20)
        self.assertTrue(all(item == results[0] for item in results))
        self.assertTrue(all(item is not results[0] for item in results[1:]))
        self.assertEqual(len(port.proposals), 1)

    def test_proposal_only_service_accepts_arbitrary_ids_and_emits_no_receipt(self) -> None:
        snapshot = make_snapshot("run:arbitrary-782")
        invocation = make_invocation(snapshot, "invocation:arbitrary-991")
        binding = make_binding(snapshot, invocation)
        output = io.StringIO()
        status = serve_once_proposal_only(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: NOW),
            lease_binding=binding,
            kernel=FakeKernel(),
        )
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(status, 0)
        self.assertEqual([line["type"] for line in lines][-2:], ["proposal", "exit"])
        self.assertNotIn("reconciliation", {line["type"] for line in lines})
        self.assertEqual(lines[-2]["proposal"]["runId"], snapshot.run_id)

    def test_proposal_only_cancellation_has_no_child_authority_receipt(self) -> None:
        snapshot = make_snapshot("run:proposal-cancel")
        invocation = make_invocation(snapshot, "invocation:proposal-cancel")
        binding = make_binding(snapshot, invocation)
        cancellation = MutableCancellation()
        cancellation.cancel()
        output = io.StringIO()
        status = serve_once_proposal_only(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: NOW),
            lease_binding=binding,
            cancellation=cancellation,
            kernel=FakeKernel(),
        )
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(status, 0)
        self.assertEqual([line["type"] for line in lines][-2:], ["proposal", "exit"])
        self.assertEqual(lines[-2]["proposal"]["kind"], "cancelled")
        self.assertIsNone(lines[-2]["proposal"]["cancellationReceipt"])
        self.assertEqual(lines[-1]["exit"]["kind"], "cancelled")
        self.assertNotIn("reconciliation", {line["type"] for line in lines})

    def test_proposal_only_service_turns_kernel_exception_into_untrusted_failure_proposal(self) -> None:
        snapshot = make_snapshot("run:arbitrary-failure")
        invocation = make_invocation(snapshot, "invocation:arbitrary-failure")
        binding = make_binding(snapshot, invocation)
        output = io.StringIO()
        status = serve_once_proposal_only(
            json.dumps({"run": snapshot.to_dict(), "invocation": invocation.to_dict()}),
            output,
            lease_authority=FixtureCanonicalLeaseAuthority([binding], clock=lambda: NOW),
            lease_binding=binding,
            kernel=ExplodingKernel(),
        )
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(status, 0)
        self.assertEqual(lines[-2]["type"], "proposal")
        self.assertEqual(lines[-2]["proposal"]["kind"], "failed")
        self.assertEqual(lines[-1]["type"], "exit")
        self.assertEqual(lines[-1]["exit"]["kind"], "failed")
        self.assertNotIn("provider secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
