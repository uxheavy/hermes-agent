"""G1 contract and real subprocess boundary tests for ``plane_runtime``."""

from __future__ import annotations

import json
import io
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from plane_runtime.g1_contract import (
    G1_CONTRACT_DIGESTS,
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    build_event,
    build_exit,
    validate_g1_frames,
)
from plane_runtime.g1_ledger import G1RuntimeLedger
from plane_runtime.hermes_adapter import HermesKernelAdapter, HermesKernelResult, UnixSocketCredentialSource
from plane_runtime.invocation_supervisor import (
    G1InvocationSupervisor,
    G1LocalTestRunner,
    HostCredentialBroker,
    InvocationPolicy,
    ProductionG1RuntimeRunner,
    build_g1_docker_argv,
    build_invocation_env,
)
from plane_runtime.service import main as service_main
from plane_runtime.subprocess_transport import RuntimeTransportError, SubprocessRuntimeTransport


def _digest(prefix: str, value: object) -> str:
    import hashlib

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def make_snapshot() -> dict[str, object]:
    content: dict[str, object] = {
        "protocol": "plane.agent-runtime/v1",
        "workspaceRef": "workspace:test",
        "runId": "run:test",
        "assignment": {
            "assignmentRef": "assignment:test",
            "revision": "revision:one",
            "targetRef": "target:test",
            "objective": "Return a deterministic runtime outcome.",
            "acceptanceCriteria": ["The outcome is bounded."],
        },
        "actorRef": "actor:test",
        "profile": {
            "profileRef": "profile-version:test",
            "revision": "revision:one",
            "role": "worker",
            "behavioralPrompt": "Use only the supplied runtime contract.",
        },
        "context": [],
        "toolCatalog": {
            "catalogDigest": "content:" + "c" * 64,
            "eagerOperations": [],
        },
        "runtimePolicy": {
            "model": {"provider": "test-provider", "model": "test-model"},
            "adapter": "deterministic-test-adapter",
            "isolation": "single-invocation",
            "maxEventPayloadBytes": 4096,
            "maxArtifactBytes": 65536,
            "maxReceiptBytes": 4096,
        },
        "totalBudget": {"inputTokens": 100, "outputTokens": 100, "durationMs": 10000},
        "contractDigests": dict(G1_CONTRACT_DIGESTS),
    }
    content["contentDigest"] = _digest("snapshot", content)
    return content


def make_invocation(snapshot: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": "plane.agent-runtime/v1",
        "workspaceRef": snapshot["workspaceRef"],
        "actorRef": snapshot["actorRef"],
        "runId": snapshot["runId"],
        "invocationId": "invocation:test",
        "runSnapshotDigest": snapshot["contentDigest"],
        "trigger": {"kind": "initial"},
        "newContextEventRefs": [],
        "remainingBudget": {"inputTokens": 100, "outputTokens": 100, "durationMs": 10000},
        "lease": {
            "leaseId": "lease:test",
            "expiresAt": "2099-01-01T00:00:00Z",
            "renewAfterMs": 100,
        },
        "cancellationRef": "cancellation:test",
        "causationRef": "causation:test",
        "correlationId": "correlation:test",
        "idempotencyKey": "idempotency:test",
    }


def make_plane_accepted_snapshot() -> dict[str, object]:
    """Mirror the accepted Plane G1 ``tests/fixtures.ts`` snapshot values."""

    snapshot = make_snapshot()
    snapshot.update({"workspaceRef": "workspace:workspace-1", "runId": "run:run-1", "actorRef": "actor:actor-1"})
    snapshot["assignment"] = {
        "assignmentRef": "assignment:assignment-1",
        "revision": "revision-1",
        "targetRef": "target:issue-1",
        "objective": "Produce the requested result.",
        "acceptanceCriteria": ["The result is reviewable."],
    }
    snapshot["profile"] = {
        "profileRef": "profile-version:profile-1",
        "revision": "revision-1",
        "role": "worker",
        "behavioralPrompt": "Complete the assignment within the supplied Plane contract.",
    }
    snapshot["context"] = [
        {
            "contextRef": "context:context-1",
            "revision": "revision-1",
            "contentDigest": "content:" + "b" * 64,
        }
    ]
    snapshot["toolCatalog"] = {
        "catalogDigest": "content:" + "c" * 64,
        "eagerOperations": [
            {
                "operationRef": "operation:search_workspace",
                "schemaDigest": "content:" + "d" * 64,
                "disclosure": "eager",
            }
        ],
    }
    snapshot["totalBudget"] = {"inputTokens": 1000, "outputTokens": 500, "durationMs": 60000}
    snapshot["contentDigest"] = _digest(
        "snapshot", {key: value for key, value in snapshot.items() if key != "contentDigest"}
    )
    return snapshot


class G1RuntimeProcessTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("docker") or os.path.exists("/opt/homebrew/bin/docker"), "Docker CLI is unavailable")
    def test_real_existing_image_proves_production_docker_attestation(self) -> None:
        image = "alpine@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
        docker = shutil.which("docker") or "/opt/homebrew/bin/docker"
        available = subprocess.run((docker, "image", "inspect", image), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not available:
            self.skipTest("the existing digest-pinned integration image is unavailable")
        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "integration-secret"}
        runner = ProductionG1RuntimeRunner(image=image, credential_source=Credentials(), max_output_bytes=65536)
        fingerprint = "a" * 64
        try:
            attestation = runner.attest_invocation(runner.command, request_fingerprint=fingerprint)
            self.assertEqual(attestation.classification, "production")
            self.assertIn("production_docker_attested", attestation.evidence)
            self.assertNotIn("integration-secret", repr(attestation))
        finally:
            runner.cleanup(attestation.container_name if "attestation" in locals() else f"plane-invocation-{fingerprint[:32]}")
            runner.close()

    def test_accepted_plane_g1_fixture_snapshot_conforms(self) -> None:
        snapshot = G1RunSnapshot.from_dict(make_plane_accepted_snapshot())
        envelope_raw = make_invocation(snapshot.to_dict())
        envelope_raw.update(
            {
                "invocationId": "invocation:invocation-1",
                "trigger": {"kind": "initial"},
                "remainingBudget": {"inputTokens": 1000, "outputTokens": 500, "durationMs": 60000},
                "lease": {
                    "leaseId": "lease:lease-1",
                    "expiresAt": "2026-08-04T10:00:00Z",
                    "renewAfterMs": 10000,
                },
                "cancellationRef": "cancellation:cancellation-1",
                "causationRef": "causation:causation-1",
                "correlationId": "correlation:correlation-1",
                "idempotencyKey": "idempotency:idempotency-1",
            }
        )
        envelope = G1InvocationEnvelope.from_dict(envelope_raw)
        self.assertEqual(snapshot.raw["assignment"]["targetRef"], "target:issue-1")  # type: ignore[index]
        self.assertEqual(snapshot.raw["toolCatalog"]["eagerOperations"][0]["operationRef"], "operation:search_workspace")  # type: ignore[index]
        self.assertEqual(envelope.run_snapshot_digest, snapshot.digest)

    def test_service_rejects_direct_g1_execution_without_test_boundary(self) -> None:
        snapshot = make_snapshot()
        request = json.dumps({"run": snapshot, "invocation": make_invocation(snapshot)}) + "\n"
        with mock.patch("sys.stdin", io.StringIO(request)), mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ):
            self.assertEqual(service_main(["--once"]), 2)

    def test_production_g1_service_routes_to_hermes_and_host_broker(self) -> None:
        snapshot = make_snapshot()
        snapshot["runtimePolicy"] = dict(snapshot["runtimePolicy"])  # type: ignore[arg-type]
        snapshot["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in snapshot.items() if key != "contentDigest"}
        )
        invocation = make_invocation(snapshot)
        request = json.dumps({"run": snapshot, "invocation": invocation}) + "\n"
        with mock.patch("plane_runtime.g1_service.HermesKernelAdapter") as adapter:
            def dispatch(_snapshot, _invocation, _cancellation, emit_body):
                emit_body(
                    {
                        "kind": "progress_observed",
                        "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "started"},
                        "publication": {"action": "observation_only"},
                    }
                )
                return HermesKernelResult(kind="completed")

            adapter.return_value.dispatch.side_effect = dispatch
            output = io.StringIO()
            with mock.patch("sys.stdin", io.StringIO(request)), mock.patch("sys.stdout", output):
                self.assertEqual(service_main(["--once", "--g1-production"]), 0)
            adapter.assert_called_once()
            source = adapter.call_args.kwargs["credential_source"]
            self.assertIsInstance(source, UnixSocketCredentialSource)
        frames = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(validate_g1_frames(frames, snapshot, invocation)[-1]["kind"], "completed")

    def test_exact_g1_snapshot_and_envelope_are_immutable_and_bound(self) -> None:
        snapshot = G1RunSnapshot.from_dict(make_snapshot())
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        self.assertEqual(snapshot.digest, snapshot.to_dict()["contentDigest"])
        self.assertEqual(invocation.run_snapshot_digest, snapshot.digest)
        with self.assertRaises(TypeError):
            snapshot.raw["runId"] = "run:changed"  # type: ignore[index]

    def test_real_forked_service_consumes_g1_and_emits_ordered_idempotent_frames(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport(test_only=True)

        frames = transport.dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True)
        parsed = validate_g1_frames([json.loads(frame) for frame in frames], snapshot, invocation)

        self.assertEqual(parsed[-1]["kind"], "completed")
        self.assertEqual(parsed[-1]["finalSequence"], len(parsed) - 2)
        self.assertTrue(any(frame.get("body", {}).get("kind") == "transcript_evidence_observed" for frame in parsed))

        replay = transport.dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True)
        self.assertEqual(frames, replay)

    def test_malformed_binding_and_changed_replay_fail_before_spawn(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport(test_only=True)

        changed = dict(invocation)
        changed["runId"] = "run:other"
        with self.assertRaises(G1ContractError):
            transport.dispatch(json.dumps(snapshot), json.dumps(changed), lease_valid=lambda: True)

        changed_snapshot = make_snapshot()
        changed_snapshot["assignment"] = dict(changed_snapshot["assignment"])  # type: ignore[arg-type]
        changed_snapshot["assignment"]["objective"] = "changed"  # type: ignore[index]
        changed_snapshot["contentDigest"] = _digest("snapshot", {k: v for k, v in changed_snapshot.items() if k != "contentDigest"})
        with self.assertRaises(G1ContractError):
            transport.dispatch(json.dumps(changed_snapshot), json.dumps(invocation), lease_valid=lambda: True)

    def test_ordering_and_post_exit_frames_are_rejected(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport(test_only=True)
        frames = [json.loads(frame) for frame in transport.dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True)]

        event = next(frame for frame in frames if "trust" in frame)
        frames[frames.index(event)] = dict(event, sequence=event["sequence"] + 1)
        with self.assertRaises(G1ContractError):
            validate_g1_frames(frames, snapshot, invocation)

        frames = [
            json.loads(frame)
            for frame in transport.dispatch(
                json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True
            )
        ]
        exit_frame = frames[-1]
        frames.append(dict(event, sequence=exit_frame["finalSequence"] + 1))
        with self.assertRaises(G1ContractError):
            validate_g1_frames(frames, snapshot, invocation)

    def test_target_namespace_and_independent_event_identities_are_strict(self) -> None:
        invalid_target = make_snapshot()
        invalid_target["assignment"] = dict(invalid_target["assignment"])  # type: ignore[arg-type]
        invalid_target["assignment"]["targetRef"] = "issue:wrong"  # type: ignore[index]
        invalid_target["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in invalid_target.items() if key != "contentDigest"}
        )
        with self.assertRaises(G1ContractError):
            G1RunSnapshot.from_dict(invalid_target)

        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        frames = [
            json.loads(frame)
            for frame in SubprocessRuntimeTransport(test_only=True).dispatch(
                json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True
            )
        ]
        events = [frame for frame in frames if "trust" in frame]
        duplicate_event_id = list(frames)
        duplicate_event_id[1] = dict(events[0], sequence=1, eventId=events[0]["eventId"])
        with self.assertRaises(G1ContractError):
            validate_g1_frames(duplicate_event_id, snapshot, invocation)
        duplicate_idempotency = list(frames)
        duplicate_idempotency[1] = dict(events[0], sequence=1, idempotencyKey=events[0]["idempotencyKey"])
        with self.assertRaises(G1ContractError):
            validate_g1_frames(duplicate_idempotency, snapshot, invocation)

    def test_one_byte_event_policy_fails_with_bounded_terminal_exit(self) -> None:
        snapshot = make_snapshot()
        snapshot["runId"] = "run:one-byte"
        snapshot["runtimePolicy"] = dict(snapshot["runtimePolicy"])  # type: ignore[arg-type]
        snapshot["runtimePolicy"]["maxEventPayloadBytes"] = 1  # type: ignore[index]
        snapshot["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in snapshot.items() if key != "contentDigest"}
        )
        invocation = make_invocation(snapshot)
        frames = SubprocessRuntimeTransport(test_only=True).dispatch(
            json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True
        )
        parsed = validate_g1_frames([json.loads(frame) for frame in frames], snapshot, invocation)
        self.assertEqual(parsed[-1]["kind"], "failed")
        self.assertEqual(parsed[-1]["failure"]["code"], "runtime_error")

    def test_selected_artifact_and_payload_bounds_are_enforced(self) -> None:
        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[arg-type]
        snapshot_raw["runtimePolicy"]["maxArtifactBytes"] = 1  # type: ignore[index]
        snapshot_raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in snapshot_raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        with self.assertRaises(G1ContractError):
            build_event(
                snapshot=snapshot,
                invocation=invocation,
                sequence=0,
                body={
                    "kind": "artifact_observed",
                    "artifact": {
                        "artifactRef": "artifact:runtime-test",
                        "contentDigest": "content:" + "d" * 64,
                        "mediaType": "text/plain",
                        "sizeBytes": 2,
                    },
                    "publication": {
                        "action": "proposal",
                        "productKind": "artifact",
                        "productRef": "artifact:runtime-test",
                        "operationAttemptRef": "operation-attempt:runtime-test",
                    },
                },
            )
        payload_ref_event = build_event(
            snapshot=snapshot,
            invocation=invocation,
            sequence=0,
            body={
                "kind": "progress_observed",
                "payload": {
                    "kind": "payload_ref",
                    "payloadRef": "payload:runtime-test",
                    "contentType": "text/plain",
                    "contentDigest": "content:" + "d" * 64,
                    "sizeBytes": 2,
                },
                "publication": {"action": "observation_only"},
            },
        )
        self.assertEqual(payload_ref_event["body"]["payload"]["sizeBytes"], 2)

    def test_plane_byte_policy_scopes_artifact_receipt_and_total_stream_independently(self) -> None:
        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[arg-type]
        snapshot_raw["runtimePolicy"].update({"maxEventPayloadBytes": 4096, "maxArtifactBytes": 1, "maxReceiptBytes": 1})  # type: ignore[index]
        snapshot_raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in snapshot_raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        body = {
            "kind": "progress_observed",
            "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "x" * 1500},
            "publication": {"action": "observation_only"},
        }
        frames = [
            build_event(snapshot=snapshot, invocation=invocation, sequence=0, body=body),
            build_event(snapshot=snapshot, invocation=invocation, sequence=1, body=body),
            build_exit(snapshot=snapshot, invocation=invocation, final_sequence=1, kind="completed"),
        ]
        self.assertGreater(sum(len(json.dumps(frame, separators=(",", ":")).encode()) for frame in frames), 4096)
        self.assertEqual(
            len(validate_g1_frames(frames, snapshot.to_dict(), invocation.to_dict(), max_stream_bytes=8192)),
            3,
        )
        with self.assertRaises(G1ContractError):
            validate_g1_frames(frames, snapshot.to_dict(), invocation.to_dict(), max_stream_bytes=4096)
        with self.assertRaises(G1ContractError):
            build_event(
                snapshot=snapshot,
                invocation=invocation,
                sequence=0,
                body={
                    "kind": "artifact_observed",
                    "artifact": {
                        "artifactRef": "artifact:runtime-test",
                        "contentDigest": "content:" + "d" * 64,
                        "mediaType": "text/plain",
                        "sizeBytes": 2,
                    },
                    "publication": {"action": "proposal", "productKind": "artifact", "productRef": "artifact:runtime-test", "operationAttemptRef": "operation-attempt:runtime-test"},
                },
            )
        # A receipt policy is for publication receipt metadata, not RuntimeExit.
        failure_snapshot_raw = make_snapshot()
        failure_snapshot_raw["runtimePolicy"] = dict(failure_snapshot_raw["runtimePolicy"])  # type: ignore[arg-type]
        failure_snapshot_raw["runtimePolicy"]["maxReceiptBytes"] = 1  # type: ignore[index]
        failure_snapshot_raw["contentDigest"] = _digest("snapshot", {key: value for key, value in failure_snapshot_raw.items() if key != "contentDigest"})
        failure_snapshot = G1RunSnapshot.from_dict(failure_snapshot_raw)
        failure_invocation = G1InvocationEnvelope.from_dict(make_invocation(failure_snapshot.to_dict()))
        exit_frame = build_exit(
            snapshot=failure_snapshot,
            invocation=failure_invocation,
            final_sequence=0,
            kind="failed",
            failure={"code": "runtime_error", "message": "bounded failure evidence", "retryable": False},
        )
        validate_g1_frames([exit_frame], failure_snapshot.to_dict(), failure_invocation.to_dict())

    def test_restart_stable_ledger_replays_exact_bytes_and_marks_unknown(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "g1.sqlite3")
            first = SubprocessRuntimeTransport(test_only=True, ledger_path=path)
            expected = first.dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True)
            first.close()
            replacement = SubprocessRuntimeTransport(test_only=True, ledger_path=path)
            self.assertEqual(expected, replacement.dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True))
            replacement.close()
            ledger = G1RuntimeLedger(path)
            changed = dict(invocation)
            changed["invocationId"] = "invocation:crashed"
            changed["remainingBudget"] = {"inputTokens": 90, "outputTokens": 90, "durationMs": 9000}
            fingerprint = __import__("hashlib").sha256(json.dumps({"run": snapshot, "invocation": changed}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            claim = ledger.claim(run_id=snapshot["runId"], invocation_id=changed["invocationId"], request_fingerprint=fingerprint, snapshot_digest=snapshot["contentDigest"], total_budget=snapshot["totalBudget"], remaining_budget=changed["remainingBudget"])
            self.assertTrue(claim.owns_dispatch)
            ledger.mark_outcome_unknown(run_id=snapshot["runId"], invocation_id=changed["invocationId"], request_fingerprint=fingerprint)
            self.assertEqual(ledger.claim(run_id=snapshot["runId"], invocation_id=changed["invocationId"], request_fingerprint=fingerprint, snapshot_digest=snapshot["contentDigest"], total_budget=snapshot["totalBudget"], remaining_budget=changed["remainingBudget"]).state, "outcome_unknown")

    def test_production_runner_requires_fixed_broker_mount_and_production_command(self) -> None:
        image = "alpine@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "broker.sock")
            argv = build_g1_docker_argv("a" * 64, InvocationPolicy(image), credential_broker_source=source)
            self.assertIn("--network", argv)
            self.assertEqual(argv[argv.index("--network") + 1], "none")
            self.assertIn("--mount", argv)
            self.assertIn("--g1-production", argv)
        with self.assertRaises(Exception):
            ProductionG1RuntimeRunner(image=image, credential_source=object())

    def test_host_credential_broker_is_narrow_and_non_ambient(self) -> None:
        secret = "host-only-broker-secret"

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                if provider != "test-provider":
                    raise RuntimeError("provider denied")
                return {"api_key": secret, "base_url": "https://example.invalid"}

        broker = HostCredentialBroker(Credentials())
        try:
            self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(broker.path)).st_mode), 0o711)
            source = UnixSocketCredentialSource(path=broker.path)
            self.assertEqual(source.resolve("test-provider")["api_key"], secret)
            self.assertEqual(source.resolve("other-provider"), {})
            self.assertNotIn(secret, build_invocation_env().values())
        finally:
            broker.close()
        self.assertFalse(os.path.exists(broker.path))

    def test_changed_replay_conflicts_before_a_second_child(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport(test_only=True)
        first = transport.dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True)
        changed = make_snapshot()
        changed["assignment"] = dict(changed["assignment"])  # type: ignore[arg-type]
        changed["assignment"]["objective"] = "changed replay"  # type: ignore[index]
        changed["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in changed.items() if key != "contentDigest"}
        )
        changed_invocation = make_invocation(changed)
        with self.assertRaises(G1ContractError):
            transport.dispatch(json.dumps(changed), json.dumps(changed_invocation), lease_valid=lambda: True)
        self.assertEqual(first, transport.dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True))

    def test_twenty_exact_replays_spawn_one_child_and_use_fixed_environment(self) -> None:
        class CountingRunner(G1LocalTestRunner):
            def __init__(self) -> None:
                super().__init__()
                self.launch_count = 0
                self.environments: list[dict[str, str]] = []
                self.attestations = []
                self._count_lock = threading.Lock()

            def attest_invocation(self, command, *, request_fingerprint):
                attestation = super().attest_invocation(command, request_fingerprint=request_fingerprint)
                self.attestations.append(attestation)
                return attestation

            def launch(self, command, *, input_bytes, environment, max_output_bytes):
                with self._count_lock:
                    self.launch_count += 1
                    self.environments.append(dict(environment))
                return super().launch(
                    command,
                    input_bytes=input_bytes,
                    environment=environment,
                    max_output_bytes=max_output_bytes,
                )

        runner = CountingRunner()
        supervisor = G1InvocationSupervisor(runner=runner)
        transport = SubprocessRuntimeTransport(
            env={"HERMES_RUNTIME_API_KEY": "must-not-cross", "UNRELATED_SECRET": "must-not-cross"},
            supervisor=supervisor,
        )
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        request = (json.dumps(snapshot), json.dumps(invocation))
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(
                pool.map(
                    lambda _index: transport.dispatch(*request, lease_valid=lambda: True),
                    range(20),
                )
            )
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(runner.launch_count, 1)
        self.assertEqual(runner.environments, [build_invocation_env()])
        self.assertEqual(runner.attestations[0].classification, "test")
        self.assertIn("fixed_environment", runner.attestations[0].evidence)
        self.assertIn("test_runner_only", runner.attestations[0].evidence)
        self.assertIn("os_isolation_not_claimed", runner.attestations[0].evidence)
        self.assertNotIn("HERMES_RUNTIME_API_KEY", runner.environments[0])
        self.assertNotIn("UNRELATED_SECRET", runner.environments[0])
        self.assertNotIn("must-not-cross", json.dumps(results))
        transport.close()

    def test_lease_callback_is_mandatory_at_the_transport_boundary(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        with self.assertRaises(RuntimeTransportError):
            SubprocessRuntimeTransport(test_only=True).dispatch(json.dumps(snapshot), json.dumps(invocation))

    def test_cancellation_and_lease_death_emit_bounded_terminal_evidence(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport(test_only=True)

        cancelled = threading.Event()
        cancelled.set()
        cancellation_frames = transport.dispatch(
            json.dumps(snapshot), json.dumps(invocation), cancellation=cancelled, lease_valid=lambda: True
        )
        self.assertEqual(validate_g1_frames([json.loads(frame) for frame in cancellation_frames], snapshot, invocation)[-1]["kind"], "cancelled")

        lease_alive = False
        lease_invocation = dict(invocation)
        lease_invocation["invocationId"] = "invocation:lease-death"
        lease_frames = transport.dispatch(
            json.dumps(snapshot), json.dumps(lease_invocation), lease_valid=lambda: lease_alive
        )
        lease_exit = validate_g1_frames([json.loads(frame) for frame in lease_frames], snapshot, lease_invocation)[-1]
        self.assertEqual(lease_exit["kind"], "failed")
        self.assertEqual(lease_exit["failure"]["code"], "lease_expired")

    def test_credential_values_never_cross_the_g1_wire(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        secret = "runtime-secret-should-not-appear"
        old = os.environ.get("HERMES_RUNTIME_API_KEY")
        os.environ["HERMES_RUNTIME_API_KEY"] = secret
        try:
            frames = SubprocessRuntimeTransport(test_only=True).dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True)
        finally:
            if old is None:
                os.environ.pop("HERMES_RUNTIME_API_KEY", None)
            else:
                os.environ["HERMES_RUNTIME_API_KEY"] = old
        self.assertNotIn(secret, json.dumps(frames))

    def test_hermes_adapter_uses_existing_agent_loop_and_redacts_host_credentials(self) -> None:
        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[arg-type]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in snapshot_raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        captured: dict[str, object] = {}

        class FakeAgent:
            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                captured["message"] = message
                captured["system_message"] = system_message
                return {"final_response": "api_key=top-secret-value" + "x" * 10000}

        def factory(**kwargs: object) -> FakeAgent:
            captured["kwargs"] = kwargs
            return FakeAgent()

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                self.provider = provider
                return {"api_key": "top-secret-value", "api_mode": "chat_completions"}

        bodies: list[dict[str, object]] = []
        result = HermesKernelAdapter(
            agent_factory=factory,
            credential_source=Credentials(),
            enabled_toolsets=("safe",),
        ).dispatch(snapshot, invocation, lambda: False, bodies.append)

        self.assertEqual(result.kind, "completed")
        self.assertNotIn("top-secret-value", result.output_text)
        self.assertLessEqual(len(result.output_text.encode("utf-8")), 4096)
        self.assertEqual(captured["kwargs"]["enabled_toolsets"], ["safe"])  # type: ignore[index]
        self.assertEqual(captured["kwargs"]["api_key"], "top-secret-value")  # type: ignore[index]
        self.assertEqual(captured["kwargs"]["max_tokens"], 100)  # type: ignore[index]
        self.assertEqual(captured["kwargs"]["max_iterations"], 90)  # type: ignore[index]
        self.assertNotIn("top-secret-value", json.dumps(bodies))

    def test_hermes_adapter_fails_closed_when_measured_usage_exceeds_budget(self) -> None:
        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[arg-type]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in snapshot_raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation_raw = make_invocation(snapshot.to_dict())
        invocation_raw["remainingBudget"] = {"inputTokens": 1, "outputTokens": 1, "durationMs": 10000}
        invocation = G1InvocationEnvelope.from_dict(invocation_raw)

        class FakeAgent:
            session_input_tokens = 2
            session_output_tokens = 2

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                del message, system_message
                return {"final_response": "bounded"}

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "trusted-host-secret", "api_mode": "chat_completions"}

        result = HermesKernelAdapter(
            agent_factory=lambda **kwargs: FakeAgent(),
            credential_source=Credentials(),
        ).dispatch(snapshot, invocation, lambda: False, lambda body: None)

        self.assertEqual(result.failure_code, "budget_exhausted")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage["inputTokens"], 2)  # type: ignore[index]
        self.assertEqual(result.usage["outputTokens"], 2)  # type: ignore[index]

    def test_cumulative_budget_cannot_increase_across_invocations(self) -> None:
        snapshot = make_snapshot()
        first = make_invocation(snapshot)
        first["invocationId"] = "invocation:budget-first"
        first["remainingBudget"] = {"inputTokens": 90, "outputTokens": 90, "durationMs": 9000}
        second = make_invocation(snapshot)
        second["invocationId"] = "invocation:budget-second"
        second["remainingBudget"] = {"inputTokens": 91, "outputTokens": 90, "durationMs": 9000}
        transport = SubprocessRuntimeTransport(test_only=True)
        transport.dispatch(json.dumps(snapshot), json.dumps(first), lease_valid=lambda: True)
        with self.assertRaises(G1ContractError):
            transport.dispatch(json.dumps(snapshot), json.dumps(second), lease_valid=lambda: True)

    def test_cumulative_budget_exhaustion_does_not_start_a_child(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        invocation["remainingBudget"] = {"inputTokens": 0, "outputTokens": 100, "durationMs": 10000}
        frames = SubprocessRuntimeTransport(test_only=True).dispatch(json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True)
        exit_frame = validate_g1_frames([json.loads(frame) for frame in frames], snapshot, invocation)[-1]
        self.assertEqual(exit_frame["failure"]["code"], "budget_exhausted")

    def test_timeout_terminates_and_reaps_the_invocation_child(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        frames = SubprocessRuntimeTransport(timeout_seconds=0.001, test_only=True).dispatch(
            json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: True
        )
        exit_frame = validate_g1_frames([json.loads(frame) for frame in frames], snapshot, invocation)[-1]
        self.assertEqual(exit_frame["kind"], "failed")
        self.assertEqual(exit_frame["failure"]["code"], "budget_exhausted")

    def test_checkpoint_reference_requires_a_trusted_host_reconstruction_source(self) -> None:
        snapshot = G1RunSnapshot.from_dict(make_snapshot())
        invocation_raw = make_invocation(snapshot.to_dict())
        invocation_raw["checkpointRef"] = "checkpoint:approved"
        invocation = G1InvocationEnvelope.from_dict(invocation_raw)
        called = False

        def factory(**kwargs: object) -> object:
            nonlocal called
            called = True
            return object()

        result = HermesKernelAdapter(agent_factory=factory).dispatch(
            snapshot, invocation, lambda: False, lambda body: None
        )
        self.assertEqual(result.failure_code, "invalid_continuation")
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
