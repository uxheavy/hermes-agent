"""Fail-closed invocation supervisor tests using injected process runners."""

from __future__ import annotations

import json
import os
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

from plane_runtime import (
    AssignmentSnapshot,
    ContractDigests,
    FixtureTerminalReconciliationPort,
    InvocationEnvelope,
    InvocationPolicy,
    InvocationSupervisor,
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
    build_invocation_argv,
    build_invocation_env,
    classify_process_death,
    reconcile_process_death,
)
from plane_runtime.contract import (
    PROTOCOL,
    OutcomeProposal,
    OutcomeSubmissionObserved,
)
from plane_runtime.invocation_supervisor import (
    DockerRunnerCapabilities,
    ProcessCapture,
)


IMAGE = "registry.example/plane/runtime@sha256:" + "a" * 64
NOW = datetime(2026, 8, 4, tzinfo=timezone.utc)


def make_snapshot() -> RunSnapshot:
    return RunSnapshot(
        protocol=PROTOCOL,
        run_id="run:supervisor",
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
        contract_digests=ContractDigests(
            "snapshot:v1", "invocation:v1", "event:v1", "exit:v1"
        ),
    )


def make_invocation(snapshot: RunSnapshot) -> InvocationEnvelope:
    return InvocationEnvelope(
        protocol=PROTOCOL,
        invocation_id="invocation:one",
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


def completed_output(
    snapshot: RunSnapshot,
    invocation: InvocationEnvelope,
) -> tuple[bytes, TerminalReconciliationReceipt]:
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
    port = FixtureTerminalReconciliationPort()
    receipt = port.reconcile_terminal(proposal)
    output = (
        json.dumps({"type": "event", "event": event.to_dict()}, sort_keys=True, separators=(",", ":"))
        + "\n"
        + json.dumps({"type": "reconciliation", "receipt": receipt.to_dict()}, sort_keys=True, separators=(",", ":"))
        + "\n"
        + json.dumps(
            {"type": "exit", "exit": RuntimeExit("completed", 1).to_dict()},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return output.encode(), receipt


class FakeProcess:
    def __init__(self, capture: ProcessCapture) -> None:
        self.capture = capture
        self.calls: list[dict[str, object]] = []

    def collect(self, **kwargs) -> ProcessCapture:
        self.calls.append(kwargs)
        return self.capture


class FakeRunner:
    def __init__(self, process: FakeProcess, *, capabilities=None) -> None:
        self.process = process
        self.capabilities = capabilities or DockerRunnerCapabilities.fully_supported()
        self.argv: tuple[str, ...] | None = None
        self.client_env: dict[str, str] | None = None
        self.input_bytes: bytes | None = None
        self.launch_calls = 0
        self.cleanup_calls: list[tuple[str, str, float]] = []
        self.fail_actions: set[str] = set()

    def launch(self, argv, *, client_env, input_bytes):
        self.launch_calls += 1
        self.argv = tuple(argv)
        self.client_env = dict(client_env)
        self.input_bytes = input_bytes
        return self.process

    def _cleanup(self, action: str, name: str, timeout_seconds: float) -> None:
        self.cleanup_calls.append((action, name, timeout_seconds))
        if action in self.fail_actions:
            raise RuntimeError("injected cleanup failure")

    def stop(self, container_name, *, timeout_seconds):
        self._cleanup("stop", container_name, timeout_seconds)

    def kill(self, container_name, *, timeout_seconds):
        self._cleanup("kill", container_name, timeout_seconds)

    def remove(self, container_name, *, timeout_seconds):
        self._cleanup("remove", container_name, timeout_seconds)


class SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = make_snapshot()
        self.invocation = make_invocation(self.snapshot)
        self.policy = InvocationPolicy(IMAGE)

    def supervisor(self, process: FakeProcess, *, port=None, runner=None) -> tuple[InvocationSupervisor, FakeRunner]:
        docker = runner or FakeRunner(process)
        return (
            InvocationSupervisor(
                policy=self.policy,
                runner=docker,
                terminal_port=port or FixtureTerminalReconciliationPort(),
            ),
            docker,
        )

    def test_argv_is_exactly_bounded_and_has_no_escape_hatches(self) -> None:
        argv = build_invocation_argv(self.snapshot, self.invocation, self.policy)
        self.assertEqual(argv[0:2], ("docker", "run"))
        self.assertEqual(argv[-6:], ("--entrypoint", "python3", IMAGE, "-m", "plane_runtime.service", "--once"))
        self.assertEqual(argv[-1], "--once")
        forbidden = {
            "--rm", "--env-file", "--volume", "-v", "--mount", "--privileged",
            "--device", "--pid=host", "--ipc=host", "--network=host", "--workdir",
            "--cwd", "--userns=host", "--add-host", "--dns", "--cap-add",
        }
        self.assertFalse(forbidden.intersection(argv))
        self.assertIn("--network", argv)
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertEqual(argv.count("--name"), 1)
        self.assertEqual(argv.count("--entrypoint"), 1)
        self.assertEqual(argv.count("--pull"), 1)
        self.assertNotIn(self.invocation.invocation_id, argv)
        self.assertNotIn(self.snapshot.run_id, argv)

    def test_argv_binds_one_opaque_name_and_label_to_invocation(self) -> None:
        argv = build_invocation_argv(self.snapshot, self.invocation, self.policy)
        name = argv[argv.index("--name") + 1]
        labels = [argv[index + 1] for index, value in enumerate(argv) if value == "--label"]
        self.assertRegex(name, r"^plane-invocation-[a-f0-9]{32}$")
        self.assertIn("plane.agent-runtime/protocol=plane.agent-runtime/v1", labels)
        self.assertEqual(sum(label.startswith("plane.agent-runtime/invocation-binding=") for label in labels), 1)

    def test_digest_pinning_and_fixed_command_reject_mutation(self) -> None:
        for image in ("registry.example/plane/runtime:latest", "runtime", "runtime@sha256:abc", IMAGE.upper()):
            with self.subTest(image=image):
                with self.assertRaises(ValueError):
                    InvocationPolicy(image)
        for changes in (
            {"entrypoint": "sh"},
            {"service_module": "evil"},
            {"service_args": ("--once", "--shell")},
            {"network_mode": "host"},
            {"read_only_rootfs": False},
            {"no_new_privileges": False},
            {"drop_all_capabilities": False},
            {"user": "0:0"},
            {"pull_policy": "always"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(self.policy, **changes)

    def test_environment_is_literal_allowlist_even_with_hostile_ambient_state(self) -> None:
        hostile = {
            "HERMES_HOME": "/secret/hermes",
            "HOME": "/secret/home",
            "OPENAI_API_KEY": "provider-secret",
            "PLANE_AGENT_TOKEN": "plane-secret",
            "DATABASE_URL": "postgres://secret",
            "HTTP_PROXY": "http://secret-proxy",
            "AWS_SECRET_ACCESS_KEY": "cloud-secret",
        }
        with patch.dict(os.environ, hostile, clear=False):
            env = build_invocation_env(self.policy)
        self.assertEqual(set(env), {"LANG", "LC_ALL", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"})
        self.assertNotIn("HOME", env)
        self.assertNotIn("HERMES_HOME", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("PLANE_AGENT_TOKEN", env)
        self.assertNotIn("DATABASE_URL", env)
        self.assertNotIn("HTTP_PROXY", env)

    def test_missing_enforcement_rejects_before_launch(self) -> None:
        process = FakeProcess(ProcessCapture(0))
        missing = DockerRunnerCapabilities.fully_supported()
        missing = replace(missing, memory_limit=False)
        runner = FakeRunner(process, capabilities=missing)
        supervisor, _ = self.supervisor(process, runner=runner)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.completed)
        self.assertEqual(runner.launch_calls, 0)
        self.assertIn("runner_capability_missing", result.evidence)

    def test_clean_completion_requires_exit_receipt_and_cleanup(self) -> None:
        stdout, receipt = completed_output(self.snapshot, self.invocation)
        process = FakeProcess(ProcessCapture(0, stdout=stdout))
        supervisor, runner = self.supervisor(process)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertTrue(result.completed)
        self.assertEqual(result.exit.kind, "completed")  # type: ignore[union-attr]
        self.assertEqual(result.receipt, receipt)
        self.assertEqual([action for action, _, _ in runner.cleanup_calls], ["stop", "kill", "remove"])
        self.assertEqual(runner.process.calls[0]["input_bytes"], runner.input_bytes)

    def test_missing_terminal_receipt_reconciles_process_death_once(self) -> None:
        stdout = json.dumps({"type": "exit", "exit": RuntimeExit("completed", 0).to_dict()}).encode() + b"\n"
        process = FakeProcess(ProcessCapture(0, stdout=stdout))
        port = FixtureTerminalReconciliationPort()
        supervisor, runner = self.supervisor(process, port=port)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.completed)
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(port.proposals[0].failure.code, "process_died")  # type: ignore[union-attr]
        self.assertEqual([action for action, _, _ in runner.cleanup_calls], ["stop", "kill", "remove"])

    def test_timeout_and_cancellation_force_exact_container_cleanup(self) -> None:
        for capture, expected in (
            (ProcessCapture(None, timed_out=True), "timeout"),
            (ProcessCapture(None, cancelled=True), "cancellation_signal"),
        ):
            with self.subTest(expected=expected):
                process = FakeProcess(capture)
                supervisor, runner = self.supervisor(process)
                result = supervisor.run_once(self.snapshot, self.invocation)
                self.assertFalse(result.completed)
                self.assertIn(expected, result.evidence)
                self.assertEqual(len(runner.cleanup_calls), 3)
                self.assertEqual(len({name for _, name, _ in runner.cleanup_calls}), 1)

    def test_malformed_oversized_and_nonzero_child_output_never_completes(self) -> None:
        cases = (
            ProcessCapture(0, stdout=b"not-json\n"),
            ProcessCapture(0, stdout=b"x" * (self.policy.stdout_limit_bytes + 1)),
            ProcessCapture(1, stdout=b""),
            ProcessCapture(0, stderr=b"x" * (self.policy.stderr_limit_bytes + 1)),
        )
        for capture in cases:
            with self.subTest(capture=capture):
                process = FakeProcess(capture)
                supervisor, _ = self.supervisor(process)
                result = supervisor.run_once(self.snapshot, self.invocation)
                self.assertFalse(result.completed)
                self.assertNotEqual(result.status, "completed")

    def test_cleanup_failure_is_explicit_and_cannot_hide_completed_child(self) -> None:
        stdout, receipt = completed_output(self.snapshot, self.invocation)
        process = FakeProcess(ProcessCapture(0, stdout=stdout))
        runner = FakeRunner(process)
        runner.fail_actions.add("remove")
        supervisor, _ = self.supervisor(process, runner=runner)
        result = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(result.status, "supervisor_action_required")
        self.assertFalse(result.completed)
        self.assertEqual(result.receipt, receipt)
        self.assertIn("cleanup_failed", result.evidence)
        self.assertIn("supervisor_action_required", result.evidence)

    def test_repeated_and_concurrent_death_reconciliation_returns_one_receipt(self) -> None:
        port = FixtureTerminalReconciliationPort()
        supervisor, _ = self.supervisor(FakeProcess(ProcessCapture(0)), port=port)
        results: list[TerminalReconciliationReceipt | None] = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    supervisor.reconcile_death(self.snapshot, self.invocation)
                )
            )
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(port.proposals), 1)
        self.assertEqual(len({receipt.receipt_ref for receipt in results if receipt}), 1)
        self.assertEqual(supervisor.reconcile_death(self.snapshot, self.invocation), results[0])

    def test_repeated_run_once_does_not_reuse_or_launch_a_second_container(self) -> None:
        process = FakeProcess(ProcessCapture(0, stdout=b"broken\n"))
        supervisor, runner = self.supervisor(process)
        first = supervisor.run_once(self.snapshot, self.invocation)
        second = supervisor.run_once(self.snapshot, self.invocation)
        self.assertEqual(first, second)
        self.assertEqual(runner.launch_calls, 1)
        self.assertEqual(len(runner.cleanup_calls), 3)

    def test_supervisor_never_serializes_credentials_in_argv_or_input(self) -> None:
        stdout, _ = completed_output(self.snapshot, self.invocation)
        process = FakeProcess(ProcessCapture(0, stdout=stdout))
        supervisor, runner = self.supervisor(process)
        supervisor.run_once(self.snapshot, self.invocation)
        wire = json.dumps({"argv": runner.argv, "input": runner.input_bytes.decode()})
        self.assertNotIn("HERMES_HOME", wire)
        self.assertNotIn("OPENAI_API_KEY", wire)
        self.assertNotIn("PLANE_AGENT_TOKEN", wire)
        self.assertNotIn("DATABASE_URL", wire)
        self.assertNotIn("HTTP_PROXY", wire)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", wire)


if __name__ == "__main__":
    unittest.main()
