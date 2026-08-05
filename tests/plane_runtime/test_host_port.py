"""Focused tests for the credential-free Hermes Plane host seam."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PlaneHostBinding,
    PlaneHostBoundsError,
    PlaneHostCancelled,
    PlaneHostUnavailable,
    bind_plane_host,
    install_plane_tools,
    plane_code_mode,
)
from tools.registry import registry


# Hermes' logging monitor may still emit after an AIAgent returns. Keep this
# hermetic home alive for the process instead of deleting it mid-test.
_TEST_HERMES_HOME = tempfile.TemporaryDirectory(prefix="hermes-plane-host-")


def _result(request: dict, *, status: str = "ok", output: object = None, **extra: object) -> dict:
    return {
        "protocol": "plane.agent-runtime/v1",
        "requestRef": request["requestRef"],
        "correlationId": request["correlationId"],
        "idempotencyKey": request["idempotencyKey"],
        "status": status,
        "replayed": status == "replayed",
        "output": output if output is not None else {"accepted": True},
        **extra,
    }


class HostPortTests(unittest.TestCase):
    def test_request_identity_is_stable_and_credential_free(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"value": "read"})

        port = CallablePlaneHostPort(rpc)
        first = PlaneHostBinding(
            port=port,
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        ).call(
            action="read",
            operation_ref="operation:work-item-get@1",
            input={"workItemRef": "work-item:test"},
            source="model",
        )
        second = PlaneHostBinding(
            port=port,
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        ).call(
            action="read",
            operation_ref="operation:work-item-get@1",
            input={"workItemRef": "work-item:test"},
            source="model",
        )

        self.assertEqual(requests[0]["requestRef"], requests[1]["requestRef"])
        self.assertEqual(requests[0]["idempotencyKey"], requests[1]["idempotencyKey"])
        self.assertEqual(first.model_payload(), second.model_payload())
        self.assertNotIn("credential", json.dumps(requests))
        self.assertNotIn("api_key", json.dumps(requests))

    def test_changed_replay_binding_and_noncanonical_response_fail_closed(self) -> None:
        def changed_response(request: dict) -> dict:
            response = _result(request)
            response["requestRef"] = "host-request:changed"
            return response

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(changed_response),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostUnavailable):
            binding.call(
                action="mutate",
                operation_ref="operation:work-item-update@1",
                input={"workItemRef": "work-item:test", "title": "new"},
                source="model",
            )
        self.assertIsNotNone(binding.fatal_error)

        def noncanonical_response(request: dict) -> str:
            return json.dumps(_result(request), indent=2)

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(noncanonical_response),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostUnavailable):
            binding.call(
                action="read",
                operation_ref="operation:work-item-get@1",
                input={},
                source="model",
            )

    def test_cancellation_and_call_bounds_are_invocation_scoped(self) -> None:
        cancelled = True
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: _result(request)),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: cancelled,
        )
        with self.assertRaises(PlaneHostCancelled):
            binding.call(
                action="read",
                operation_ref="operation:read@1",
                input={},
                source="model",
            )

        calls = []
        bounded = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: calls.append(request) or _result(request)),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            max_calls=1,
        )
        bounded.call(
            action="read",
            operation_ref="operation:read@1",
            input={},
            source="model",
        )
        with self.assertRaises(ValueError):
            bounded.call(
                action="read",
                operation_ref="operation:read@1",
                input={},
                source="model",
            )
        self.assertEqual(len(calls), 1)

        oversized = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: _result(request)),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostBoundsError):
            oversized.call(
                action="read",
                operation_ref="operation:read@1",
                input={"value": "x" * 9000},
                source="model",
            )
        self.assertIsNotNone(oversized.fatal_error)

    def test_explicit_publication_requires_a_gateway_publication_shape(self) -> None:
        bodies: list[dict] = []

        def rpc(request: dict) -> dict:
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "conversation",
                    "productRef": "conversation:conversation-1",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": "operation:conversation-publish@1",
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )
        binding.publish(
            kind="conversation",
            operation_ref="operation:conversation-publish@1",
            resource_ref="conversation:conversation-1",
            content="explicit product action",
        )
        self.assertEqual(
            [body["kind"] for body in bodies],
            ["progress_observed", "conversation_publication_observed"],
        )
        self.assertEqual(binding.publication_count, 1)

    def test_tools_are_registry_dispatchable_only_inside_a_bound_context(self) -> None:
        install_plane_tools()
        self.assertIsNotNone(registry.get_entry("plane_operation"))
        self.assertIsNotNone(registry.get_entry("plane_publish"))
        unbound = registry.dispatch(
            "plane_operation",
            {"action": "read", "operationRef": "operation:read@1", "input": {}},
        )
        self.assertIn("error", unbound)

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"read": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with bind_plane_host(binding):
            rejected_code = registry.dispatch(
                "plane_operation",
                {"action": "code", "operationRef": "operation:compose@1", "input": {}},
            )
            self.assertIn("restricted to execute_code", rejected_code)
            self.assertEqual(len(requests), 0)
            with plane_code_mode():
                code_result = registry.dispatch(
                    "plane_operation",
                    {"action": "code", "operationRef": "operation:compose@1", "input": {}},
                )
        self.assertIn('"read":true', code_result)
        self.assertEqual(len(requests), 1)

    def test_real_aiagent_loop_reaches_read_mutation_code_and_explicit_publication(self) -> None:
        """Use a deterministic provider boundary, not a fake Hermes agent."""

        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class DeterministicCompletions:
            def __init__(self) -> None:
                self.calls = 0

            @staticmethod
            def tool_call(name: str, arguments: dict[str, object], call_id: str):
                return SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                    extra_content=None,
                )

            def create(self, **_: object):
                self.calls += 1
                if self.calls == 1:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_operation",
                                "arguments": {"action": "discover", "input": {}},
                            },
                            "call-discover",
                        )
                    ]
                elif self.calls == 2:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_operation",
                                "arguments": {
                                    "action": "read",
                                    "operationRef": "operation:work-item-get@1",
                                    "input": {"workItemRef": "work-item:test"},
                                },
                            },
                            "call-read",
                        )
                    ]
                elif self.calls == 3:
                    tool_calls = [
                        self.tool_call(
                            "execute_code",
                            {
                                "code": (
                                    "from hermes_tools import plane_operation\n"
                                    "print(plane_operation('code', "
                                    "'operation:compose@1', {'workItemRef': 'work-item:test'}))"
                                )
                            },
                            "call-code",
                        )
                    ]
                elif self.calls == 4:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_operation",
                                "arguments": {
                                    "action": "mutate",
                                    "operationRef": "operation:work-item-update@1",
                                    "input": {
                                        "workItemRef": "work-item:test",
                                        "title": "updated",
                                    },
                                },
                            },
                            "call-mutate",
                        )
                    ]
                elif self.calls == 5:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_publish",
                                "arguments": {
                                    "kind": "conversation",
                                    "operationRef": "operation:conversation-publish@1",
                                    "resourceRef": "conversation:test",
                                    "content": "publish only through this explicit action",
                                },
                            },
                            "call-publish",
                        )
                    ]
                else:
                    tool_calls = []
                if tool_calls:
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=tool_calls,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    choice = SimpleNamespace(finish_reason="tool_calls", message=message)
                else:
                    message = SimpleNamespace(
                        content="ordinary final evidence",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    choice = SimpleNamespace(finish_reason="stop", message=message)
                return SimpleNamespace(
                    choices=[choice],
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                    ),
                )

        class DeterministicClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=DeterministicCompletions())

        class ModelCredentials:
            def resolve(self, provider: str) -> dict[str, str]:
                self.provider = provider
                return {
                    "api_key": "model-only-secret",
                    "base_url": "http://127.0.0.1",
                    "api_mode": "chat_completions",
                }

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            response: dict[str, object] = _result(
                request,
                output={"operationRef": request["operationRef"], "accepted": True},
            )
            if request["action"] == "publish":
                response["publication"] = {
                    "action": "applied",
                    "productKind": "conversation",
                    "productRef": "conversation:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": "operation:conversation-publish@1",
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                }
            return response

        from run_agent import AIAgent

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            client = DeterministicClient()
            # This replaces only the model/provider boundary. The real Hermes
            # AIAgent constructor, tool loop, registry, and execute_code path
            # remain active.
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=ModelCredentials(),
                enabled_toolsets=("code_execution",),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=8,
            )

        self.assertEqual(result.kind, "completed")
        self.assertEqual(
            [(request["action"], request["source"]) for request in requests],
            [
                ("discover", "model"),
                ("read", "model"),
                ("code", "code"),
                ("mutate", "model"),
                ("publish", "model"),
            ],
        )
        self.assertEqual(len({request["correlationId"] for request in requests}), 1)
        self.assertNotIn("model-only-secret", json.dumps(requests + bodies))
        self.assertNotIn("plane-secret", json.dumps(requests + bodies))
        self.assertEqual(
            [body["kind"] for body in bodies if "publication" in body and body["publication"].get("action") != "observation_only"],
            ["conversation_publication_observed"],
        )
        self.assertEqual(
            [body["payload"]["text"] for body in bodies if body["kind"] == "transcript_evidence_observed"],
            ["ordinary final evidence"],
        )

    def test_ordinary_final_text_never_publishes_and_host_failure_fails_closed(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from run_agent import AIAgent

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class FinalClient:
            class Completions:
                def create(self, **_: object):
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                message=SimpleNamespace(
                                    content="ordinary final only",
                                    tool_calls=None,
                                    reasoning=None,
                                    reasoning_content=None,
                                    refusal=None,
                                ),
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=1,
                            completion_tokens=1,
                            total_tokens=2,
                        ),
                    )

            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=self.Completions())

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                return {
                    "api_key": "model-only-secret",
                    "base_url": "http://127.0.0.1",
                    "api_mode": "chat_completions",
                }

        def factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            client = FinalClient()
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(
                    lambda request: (_ for _ in ()).throw(RuntimeError("host unavailable"))
                ),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=2,
            )
        self.assertEqual(result.kind, "completed")
        self.assertFalse(any(body["kind"] == "conversation_publication_observed" for body in bodies))
        self.assertTrue(any(body["kind"] == "transcript_evidence_observed" for body in bodies))

        class FailingClient:
            class Completions:
                def __init__(self) -> None:
                    self.calls = 0

                def create(self, **_: object):
                    self.calls += 1
                    if self.calls == 1:
                        tool_call = SimpleNamespace(
                            id="call-failure",
                            function=SimpleNamespace(
                                name="tool_call",
                                arguments=json.dumps(
                                    {
                                        "name": "plane_operation",
                                        "arguments": {
                                            "action": "read",
                                            "operationRef": "operation:read@1",
                                            "input": {},
                                        },
                                    }
                                ),
                            ),
                            extra_content=None,
                        )
                        message = SimpleNamespace(
                            content=None,
                            tool_calls=[tool_call],
                            reasoning=None,
                            reasoning_content=None,
                            refusal=None,
                        )
                        finish_reason = "tool_calls"
                    else:
                        message = SimpleNamespace(
                            content="should not become a successful runtime result",
                            tool_calls=None,
                            reasoning=None,
                            reasoning_content=None,
                            refusal=None,
                        )
                        finish_reason = "stop"
                    return SimpleNamespace(
                        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
                        usage=SimpleNamespace(
                            prompt_tokens=1,
                            completion_tokens=1,
                            total_tokens=2,
                        ),
                    )

            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=self.Completions())

        def failing_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            client = FailingClient()
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        failed_bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            failed = HermesKernelAdapter(
                agent_factory=failing_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(
                    lambda request: (_ for _ in ()).throw(RuntimeError("host unavailable"))
                ),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                failed_bodies.append,
                model_call_allowance=2,
            )
        self.assertEqual(failed.kind, "failed")
        self.assertEqual(failed.failure_code, "runtime_error")


if __name__ == "__main__":
    unittest.main()
