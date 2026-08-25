"""Provider-free adapter proof for one canonical Plane work-item lifecycle."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from unittest import mock

from plane_runtime.g1_contract import (
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    _bounded_payload,
)
from plane_runtime.hermes_adapter import (
    HermesKernelAdapter,
    ProviderOutcomeUnknownError,
    _emit_plane_runtime_diagnostics,
    _record_plane_runtime_host_callback,
    _classify_runtime_exception,
)
from plane_runtime.host_port import CallablePlaneHostPort, current_plane_host
from plane_runtime.g1_service import _terminal_failure
from tests.plane_runtime.test_g1_runtime_process import _digest, make_invocation, make_snapshot
from tools.registry import registry


def _runtime_diagnostics(bodies: list[dict[str, object]]) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for body in bodies:
        payload = body.get("payload")
        if not isinstance(payload, dict) or payload.get("kind") != "inline_text":
            continue
        if payload.get("contentType") != "text/plain":
            continue
        text = payload.get("text")
        if not isinstance(text, str):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("kind") != "runtime_diagnostics":
            continue
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if text != canonical:
            raise AssertionError("runtime diagnostics are not canonical JSON")
        values.append(value)
    return values


class AdapterPresentationTests(unittest.TestCase):
    def test_host_callback_projection_preserves_only_finite_runtime_subreason(self) -> None:
        class Agent:
            _plane_runtime_diagnostics = {"hostCallbacks": []}

        agent = Agent()
        _record_plane_runtime_host_callback(
            agent,
            {
                "callbackPhase": "host_return",
                "operationRefDigest": "a" * 64,
                "codeModeRuntimeSubreason": "catalog_operation_unavailable",
            },
        )
        self.assertEqual(
            agent._plane_runtime_diagnostics["hostCallbacks"][0][
                "codeModeRuntimeSubreason"
            ],
            "catalog_operation_unavailable",
        )
        _record_plane_runtime_host_callback(
            agent,
            {
                "callbackPhase": "host_return",
                "operationRefDigest": "b" * 64,
                "codeModeRuntimeSubreason": "raw-message",
            },
        )
        self.assertEqual(len(agent._plane_runtime_diagnostics["hostCallbacks"]), 1)

    def test_runtime_diagnostics_use_bounded_canonical_inline_text(self) -> None:
        class Agent:
            _plane_runtime_diagnostics = {
                "requests": [
                    {
                        "sequence": 1,
                        "toolChoice": "required",
                        "visibleToolset": "execute_only",
                        "visibleToolCount": 1,
                        "serialized": True,
                    }
                ],
                "responses": [
                    {"sequence": 1, "responseClass": "tool_call", "toolCall": "execute"}
                ],
                "hostCallbacks": [
                    {"sequence": 1, "phase": "before_host_call", "operationRefDigest": "a" * 64}
                ],
            }

        bodies: list[dict[str, object]] = []
        _emit_plane_runtime_diagnostics(Agent(), bodies.append)

        self.assertEqual(len(bodies), 1)
        payload = bodies[0]["payload"]
        self.assertEqual(_bounded_payload(payload), payload)
        self.assertEqual(payload["kind"], "inline_text")  # type: ignore[index]
        self.assertEqual(payload["contentType"], "text/plain")  # type: ignore[index]
        text = payload["text"]  # type: ignore[index]
        self.assertIsInstance(text, str)
        parsed = json.loads(text)
        self.assertEqual(parsed["kind"], "runtime_diagnostics")
        self.assertEqual(json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")), text)
        self.assertNotIn("prompt", text)
        self.assertNotIn("provider", text)
        self.assertNotIn("provider-test-secret", text)

    def test_real_agent_closed_client_recreation_is_bounded_provider_client_failure(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["runtimePolicy"] = dict(raw["runtimePolicy"])  # type: ignore[arg-type]
        raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        raw["runtimePolicy"]["model"] = {"provider": "xai", "model": "test-model"}  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class OpenAI:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

            def is_closed(self) -> bool:
                return True

        class Factory:
            calls = 0

            def __call__(self) -> object:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("relay factory failure")
                return object()

        factory = Factory()

        def agent_factory(**kwargs: object) -> object:
            from run_agent import AIAgent

            agent = AIAgent(**kwargs)
            agent._api_max_retries = 1
            return agent

        with tempfile.TemporaryDirectory() as hermes_home:
            import run_agent

            with mock.patch.dict(os.environ, {"HERMES_HOME": hermes_home}), mock.patch.object(
                run_agent, "OpenAI", OpenAI
            ), mock.patch.object(run_agent, "get_tool_definitions", return_value=[]), mock.patch.object(
                run_agent, "check_toolset_requirements", return_value={}
            ):
                # The initial client is intentionally closed. The first turn
                # must attempt exactly one invocation-scoped recreation before
                # any provider request; the factory then fails at that seam.
                result = HermesKernelAdapter(
                    agent_factory=agent_factory,
                    credential_source=type(
                        "Credentials",
                        (),
                        {"resolve": lambda _self, _provider: {
                            "api_key": "provider-free-test-secret",
                            "base_url": "http://provider.invalid/v1",
                            "api_mode": "chat_completions",
                        }},
                    )(),
                    http_client_factory=factory,
                ).dispatch(
                    snapshot,
                    invocation,
                    lambda: False,
                    lambda _body: None,
                    model_call_allowance=1,
                )

        self.assertEqual(factory.calls, 2)
        self.assertEqual(result.kind, "failed")
        self.assertEqual(
            _classify_runtime_exception(RuntimeError("Failed to recreate closed OpenAI client")),
            "provider_client_failure",
        )
        self.assertEqual(result.runtime_phase, "conversation")
        self.assertEqual(result.model_calls, 0)

    def test_standard_plane_snapshot_emits_request_response_and_host_callback_shape(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        captured: dict[str, object] = {}

        def rpc(request: dict[str, object]) -> dict[str, object]:
            return {
                "protocol": "plane.agent-runtime/v1",
                "requestRef": request["requestRef"],
                "correlationId": request["correlationId"],
                "idempotencyKey": request["idempotencyKey"],
                "status": "ok",
                "replayed": False,
                "output": {"ok": True, "result": {"work_item": {"title": "redacted"}}},
            }

        class StandardAgent:
            session_api_calls = 1

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                del message, system_message
                diagnostics = getattr(self, "_plane_runtime_diagnostics", None)
                captured["diagnostics_initialized"] = isinstance(diagnostics, dict)
                if not isinstance(diagnostics, dict):
                    return {"final_response": "ordinary final text"}
                diagnostics["requests"].append(
                    {
                        "sequence": 1,
                        "toolChoice": "auto",
                        "visibleToolset": "other",
                        "visibleToolCount": 2,
                        "serialized": True,
                    }
                )
                diagnostics["responses"].append(
                    {"sequence": 1, "responseClass": "tool_call", "toolCall": "other"}
                )
                host = current_plane_host()
                assert host is not None
                host.call(
                    action="read",
                    operation_ref="operation:work_item.read",
                    input={"project_id": "redacted", "issue_id": "redacted"},
                    source="model",
                )
                return {"final_response": "ordinary final text"}

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "provider-free-test-secret"}

        bodies: list[dict[str, object]] = []
        result = HermesKernelAdapter(
            agent_factory=lambda **kwargs: StandardAgent(),
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(rpc),
        ).dispatch(snapshot, invocation, lambda: False, bodies.append, model_call_allowance=1)

        self.assertEqual(result.kind, "completed")
        self.assertTrue(captured["diagnostics_initialized"])
        diagnostics = _runtime_diagnostics(bodies)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["requests"][0]["visibleToolset"], "other")  # type: ignore[index]
        self.assertEqual(diagnostics[0]["responses"][0]["responseClass"], "tool_call")  # type: ignore[index]
        callback = diagnostics[0]["hostCallbacks"][0]  # type: ignore[index]
        self.assertEqual(callback["phase"], "before_host_call")  # type: ignore[index]
        self.assertRegex(callback["operationRefDigest"], r"^[0-9a-f]{64}$")  # type: ignore[index]
        self.assertNotIn("operation:work_item.read", json.dumps(diagnostics))

    def test_transport_outcome_unknown_preserves_completed_code_mode_diagnostics(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["toolCatalog"]["modelToolset"] = "code_mode_only"  # type: ignore[index]
        raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        raw["runtimePolicy"].update(  # type: ignore[union-attr]
            {
                "maxCodeModeInputBytes": 65_536,
                "maxCodeModeOutputBytes": 65_536,
                "maxCodeModeCalls": 4,
            }
        )
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class TransportClosingAgent:
            session_api_calls = 6

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                del message, system_message
                diagnostics = self._plane_runtime_diagnostics
                for sequence in range(1, 6):
                    diagnostics["requests"].append(
                        {
                            "sequence": sequence,
                            "toolChoice": "required",
                            "visibleToolset": "execute_only",
                            "visibleToolCount": 1,
                            "serialized": True,
                        }
                    )
                    diagnostics["responses"].append(
                        {
                            "sequence": sequence,
                            "responseClass": "tool_call",
                            "toolCall": "execute",
                        }
                    )
                raise ProviderOutcomeUnknownError(status_code=200)

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "provider-free-test-secret"}

        bodies: list[dict[str, object]] = []
        result = HermesKernelAdapter(
            agent_factory=lambda **kwargs: TransportClosingAgent(),
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(lambda request: {}),
        ).dispatch(snapshot, invocation, lambda: False, bodies.append, model_call_allowance=8)

        self.assertEqual(result.failure_code, "outcome_unknown")
        self.assertFalse(result.retryable)
        diagnostics = _runtime_diagnostics(bodies)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(len(diagnostics[0]["requests"]), 5)  # type: ignore[index]
        self.assertEqual(diagnostics[0]["requests"][0]["toolChoice"], "required")  # type: ignore[index]
        self.assertEqual(diagnostics[0]["responses"][-1]["toolCall"], "execute")  # type: ignore[index]

    def test_generic_conversation_failure_preserves_bounded_diagnostics(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["runtimePolicy"] = dict(raw["runtimePolicy"])  # type: ignore[arg-type]
        raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class ExplodingAgent:
            session_api_calls = 0

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                del message, system_message
                self._plane_runtime_diagnostics["requests"].append(
                    {
                        "sequence": 1,
                        "toolChoice": "auto",
                        "visibleToolset": "other",
                        "visibleToolCount": 1,
                        "serialized": True,
                    }
                )
                self._plane_runtime_diagnostics["responses"].append(
                    {"sequence": 1, "responseClass": "text_response", "toolCall": "none"}
                )
                raise RuntimeError("private conversation detail")

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "provider-free-test-secret"}

        bodies: list[dict[str, object]] = []
        result = HermesKernelAdapter(
            agent_factory=lambda **kwargs: ExplodingAgent(),
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(lambda request: {}),
        ).dispatch(snapshot, invocation, lambda: False, bodies.append, model_call_allowance=1)

        self.assertEqual(result.failure_cause, "runtime_unknown_failure")
        self.assertEqual(result.runtime_phase, "conversation")
        self.assertEqual(result.exception_class, "RuntimeError")
        self.assertEqual(
            result.child_diagnostic,
            {
                "exceptionModule": "builtins",
                "exceptionClass": "RuntimeError",
                "runtimePhase": "conversation",
                "originToken": "run_conversation",
            },
        )
        exit_frame = _terminal_failure(snapshot, invocation, result, 0)
        self.assertEqual(
            exit_frame["failure"]["childDiagnostic"],
            result.child_diagnostic,
        )
        self.assertNotIn("private conversation detail", json.dumps(exit_frame))
        diagnostics = _runtime_diagnostics(bodies)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["requests"][0]["visibleToolset"], "other")  # type: ignore[index]
        self.assertEqual(diagnostics[0]["responses"][0]["responseClass"], "text_response")  # type: ignore[index]
        self.assertNotIn("private conversation detail", json.dumps(bodies))

    def test_serialized_code_mode_snapshot_installs_first_tool_guard_before_final_text(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["toolCatalog"]["modelToolset"] = "code_mode_only"  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        serialized = json.loads(json.dumps(raw, sort_keys=True, separators=(",", ":")))
        snapshot = G1RunSnapshot.from_dict(serialized)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        captured: dict[str, object] = {}

        class FinalTextAgent:
            session_api_calls = 1

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                del message, system_message
                captured["first_required_tool"] = getattr(self, "_plane_first_required_tool", None)
                return {"final_response": "ordinary final text"}

        def factory(**kwargs: object) -> FinalTextAgent:
            captured["enabled_toolsets"] = kwargs["enabled_toolsets"]
            return FinalTextAgent()

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "provider-free-test-secret"}

        missing = copy.deepcopy(serialized)
        del missing["toolCatalog"]["modelToolset"]
        with self.assertRaisesRegex(G1ContractError, "modelToolset"):
            G1RunSnapshot.from_dict(missing)

        result = HermesKernelAdapter(
            agent_factory=factory,
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(lambda request: {}),
        ).dispatch(snapshot, invocation, lambda: False, lambda body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "completed")
        self.assertEqual(captured["first_required_tool"], "plane_execute_typescript")
        self.assertNotIn("plane_operation", captured["enabled_toolsets"])

    def test_code_mode_snapshot_reduces_model_catalog_to_execute_and_publish(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["toolCatalog"]["modelToolset"] = "code_mode_only"  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        code = {"type": "function", "function": {"name": "plane_execute_typescript"}}
        publish = {"type": "function", "function": {"name": "plane_publish"}}
        operation = {"type": "function", "function": {"name": "plane_operation"}}
        captured: dict[str, object] = {}

        class OverexposedAgent:
            tools = [code, publish, operation]
            valid_tool_names = {
                "plane_execute_typescript",
                "plane_publish",
                "plane_operation",
            }
            session_api_calls = 1

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                del message, system_message
                captured["tools"] = self.tools
                captured["valid_tool_names"] = self.valid_tool_names
                return {"final_response": "ordinary final text"}

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "provider-free-test-secret"}

        result = HermesKernelAdapter(
            agent_factory=lambda **kwargs: OverexposedAgent(),
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(lambda request: {}),
        ).dispatch(snapshot, invocation, lambda: False, lambda body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "completed")
        self.assertEqual(
            [tool["function"]["name"] for tool in captured["tools"]],
            ["plane_execute_typescript", "plane_publish"],
        )
        self.assertEqual(
            captured["valid_tool_names"],
            {"plane_execute_typescript", "plane_publish"},
        )

    def test_manager_route_uses_search_then_canonical_work_item_read_input(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["assignment"] = {
            "assignmentRef": "assignment:manager",
            "revision": "revision:one",
            "targetRef": "target:assigned-work-item",
            "objective": "Coordinate a bounded Manager objective for the assigned work item.",
            "acceptanceCriteria": ["Publish one reviewable result."],
        }
        raw["profile"]["role"] = "delegator"  # type: ignore[index]
        raw["toolCatalog"] = {
            "catalogDigest": "content:" + "c" * 64,
            "modelToolset": "standard",
            "eagerOperations": [
                {
                    "operationRef": "operation:catalog.search",
                    "schemaDigest": "content:" + "b" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                    "disclosure": "eager",
                },
                {
                    "operationRef": "operation:catalog.describe",
                    "schemaDigest": "content:" + "a" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["operation_id"],
                        "properties": {"operation_id": {"type": "string"}},
                    },
                    "disclosure": "eager",
                },
                {
                    "operationRef": "operation:search_workspace",
                    "schemaDigest": "content:" + "d" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                    "disclosure": "eager",
                },
                {
                    "operationRef": "operation:work_item.read",
                    "schemaDigest": "content:" + "e" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["project_id", "issue_id"],
                        "properties": {
                            "project_id": {"type": "string"},
                            "issue_id": {"type": "string"},
                        },
                    },
                    "disclosure": "eager",
                },
            ],
        }
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == "operation:search_workspace":
                return {
                    "protocol": "plane.agent-runtime/v1",
                    "requestRef": request["requestRef"],
                    "correlationId": request["correlationId"],
                    "idempotencyKey": request["idempotencyKey"],
                    "status": "ok",
                    "replayed": False,
                    "output": {
                        "ok": True,
                        "result": {
                            "results": [
                                {
                                    "objectType": "work_item",
                                    "workItemReadCall": {
                                        "action": "read",
                                        "operationRef": "operation:work_item.read",
                                        "input": {"preparedCallRef": "prepared-call:opaque"},
                                    },
                                }
                            ]
                        },
                    },
                }
            if request["operationRef"] == "operation:work_item.read":
                self.assertEqual(
                    request["input"],
                    {"preparedCallRef": "prepared-call:opaque"},
                )
                return {
                    "protocol": "plane.agent-runtime/v1",
                    "requestRef": request["requestRef"],
                    "correlationId": request["correlationId"],
                    "idempotencyKey": request["idempotencyKey"],
                    "status": "ok",
                    "replayed": False,
                    "output": {"ok": True, "result": {"work_item": {"title": "assigned"}}},
                }
            raise AssertionError(f"unexpected operation {request['operationRef']}")

        captured: dict[str, object] = {}

        class FakeAgent:
            session_input_tokens = 1
            session_output_tokens = 1
            session_api_calls = 1

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                captured["message"] = message
                captured["system_message"] = system_message
                if "operation:search_workspace the first Plane call" not in system_message:
                    raise AssertionError("Manager guidance did not prioritize search_workspace")
                binding = current_plane_host()
                self.assert_binding(binding)
                search = json.loads(
                    registry.dispatch(
                        "plane_operation",
                        {
                            "action": "read",
                            "operationRef": "operation:search_workspace",
                            "input": {"query": "assigned work item"},
                        },
                    )
                )
                return {"final_response": search}

            @staticmethod
            def assert_binding(binding: object) -> None:
                if binding is None:
                    raise AssertionError("Plane host was not bound for the invocation")

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "provider-free-test-secret"}

        result = HermesKernelAdapter(
            agent_factory=lambda **kwargs: FakeAgent(),
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(rpc),
        ).dispatch(snapshot, invocation, lambda: False, lambda body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "completed")
        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["operation:search_workspace", "operation:work_item.read"],
        )
        self.assertNotIn("operation:catalog.describe", [request["operationRef"] for request in requests])
        self.assertIn('"required":["preparedCallRef"]', str(captured["system_message"]))
        self.assertNotIn('"project_id"', str(captured["system_message"]))
        self.assertNotIn('"issue_id"', str(captured["system_message"]))


if __name__ == "__main__":
    unittest.main()
