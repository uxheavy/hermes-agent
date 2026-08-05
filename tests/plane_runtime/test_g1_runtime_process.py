"""G1 contract and real subprocess boundary tests for ``plane_runtime``."""

from __future__ import annotations

import json
import os
import threading
import unittest

from plane_runtime.g1_contract import (
    G1_CONTRACT_DIGESTS,
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    validate_g1_frames,
)
from plane_runtime.hermes_adapter import HermesKernelAdapter
from plane_runtime.subprocess_transport import SubprocessRuntimeTransport


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


class G1RuntimeProcessTests(unittest.TestCase):
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
        transport = SubprocessRuntimeTransport()

        frames = transport.dispatch(json.dumps(snapshot), json.dumps(invocation))
        parsed = validate_g1_frames([json.loads(frame) for frame in frames], snapshot, invocation)

        self.assertEqual(parsed[-1]["kind"], "completed")
        self.assertEqual(parsed[-1]["finalSequence"], len(parsed) - 2)
        self.assertTrue(any(frame.get("body", {}).get("kind") == "transcript_evidence_observed" for frame in parsed))

        replay = transport.dispatch(json.dumps(snapshot), json.dumps(invocation))
        self.assertEqual(frames, replay)

    def test_malformed_binding_and_changed_replay_fail_before_spawn(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport()

        changed = dict(invocation)
        changed["runId"] = "run:other"
        with self.assertRaises(G1ContractError):
            transport.dispatch(json.dumps(snapshot), json.dumps(changed))

        changed_snapshot = make_snapshot()
        changed_snapshot["assignment"] = dict(changed_snapshot["assignment"])  # type: ignore[arg-type]
        changed_snapshot["assignment"]["objective"] = "changed"  # type: ignore[index]
        changed_snapshot["contentDigest"] = _digest("snapshot", {k: v for k, v in changed_snapshot.items() if k != "contentDigest"})
        with self.assertRaises(G1ContractError):
            transport.dispatch(json.dumps(changed_snapshot), json.dumps(invocation))

    def test_ordering_and_post_exit_frames_are_rejected(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport()
        frames = [json.loads(frame) for frame in transport.dispatch(json.dumps(snapshot), json.dumps(invocation))]

        event = next(frame for frame in frames if "trust" in frame)
        frames[frames.index(event)] = dict(event, sequence=event["sequence"] + 1)
        with self.assertRaises(G1ContractError):
            validate_g1_frames(frames, snapshot, invocation)

        exit_frame = frames[-1]
        frames.append(dict(event, sequence=exit_frame["finalSequence"] + 1))
        with self.assertRaises(G1ContractError):
            validate_g1_frames(frames, snapshot, invocation)

    def test_cancellation_and_lease_death_emit_bounded_terminal_evidence(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        transport = SubprocessRuntimeTransport()

        cancelled = threading.Event()
        cancelled.set()
        cancellation_frames = transport.dispatch(
            json.dumps(snapshot), json.dumps(invocation), cancellation=cancelled
        )
        self.assertEqual(validate_g1_frames([json.loads(frame) for frame in cancellation_frames], snapshot, invocation)[-1]["kind"], "cancelled")

        lease_alive = False
        lease_frames = transport.dispatch(
            json.dumps(snapshot), json.dumps(invocation), lease_valid=lambda: lease_alive
        )
        lease_exit = validate_g1_frames([json.loads(frame) for frame in lease_frames], snapshot, invocation)[-1]
        self.assertEqual(lease_exit["kind"], "failed")
        self.assertEqual(lease_exit["failure"]["code"], "lease_expired")

    def test_credential_values_never_cross_the_g1_wire(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        secret = "runtime-secret-should-not-appear"
        old = os.environ.get("HERMES_RUNTIME_API_KEY")
        os.environ["HERMES_RUNTIME_API_KEY"] = secret
        try:
            frames = SubprocessRuntimeTransport().dispatch(json.dumps(snapshot), json.dumps(invocation))
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
        self.assertNotIn("top-secret-value", json.dumps(bodies))

    def test_cumulative_budget_exhaustion_does_not_start_a_child(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        invocation["remainingBudget"] = {"inputTokens": 0, "outputTokens": 100, "durationMs": 10000}
        frames = SubprocessRuntimeTransport().dispatch(json.dumps(snapshot), json.dumps(invocation))
        exit_frame = validate_g1_frames([json.loads(frame) for frame in frames], snapshot, invocation)[-1]
        self.assertEqual(exit_frame["failure"]["code"], "budget_exhausted")

    def test_timeout_terminates_and_reaps_the_invocation_child(self) -> None:
        snapshot = make_snapshot()
        invocation = make_invocation(snapshot)
        frames = SubprocessRuntimeTransport(timeout_seconds=0.001).dispatch(
            json.dumps(snapshot), json.dumps(invocation)
        )
        exit_frame = validate_g1_frames([json.loads(frame) for frame in frames], snapshot, invocation)[-1]
        self.assertEqual(exit_frame["kind"], "failed")
        self.assertEqual(exit_frame["failure"]["code"], "runtime_error")

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
