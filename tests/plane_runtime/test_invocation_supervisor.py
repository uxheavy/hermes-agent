"""Evidence for the host-authoritative invocation supervisor boundary."""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from plane_runtime import (
    AssignmentSnapshot,
    CanonicalLeaseBinding,
    ContractDigests,
    EventCollector,
    FakeKernel,
    FixtureCanonicalLeaseAuthority,
    FixtureTerminalReconciliationPort,
    InvocationEnvelope,
    InvocationTrigger,
    OperationDescriptor,
    RuntimeBudget,
    RuntimeBudgetPolicy,
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

    def make_supervisor(self, output: bytes, *, runner: FakeRunner | None = None, port=None):
        docker = runner or FakeRunner(FakeProcess(ProcessCapture(0, stdout=output)))
        lease_binding = make_binding(self.snapshot, self.invocation)
        return (
            InvocationSupervisor(
                policy=self.policy,
                runner=docker,
                terminal_port=port or FixtureTerminalReconciliationPort(),
                lease_authority=FixtureCanonicalLeaseAuthority([lease_binding], clock=lambda: NOW),
                lease_binding=lease_binding,
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
        self.assertTrue(all(item is results[0] for item in results))
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
