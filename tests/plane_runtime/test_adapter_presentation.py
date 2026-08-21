"""Provider-free adapter proof for one canonical Plane work-item lifecycle."""

from __future__ import annotations

import copy
import json
import unittest

from plane_runtime.g1_contract import G1ContractError, G1InvocationEnvelope, G1RunSnapshot
from plane_runtime.hermes_adapter import HermesKernelAdapter, ProviderOutcomeUnknownError
from plane_runtime.host_port import CallablePlaneHostPort, current_plane_host
from tests.plane_runtime.test_g1_runtime_process import _digest, make_invocation, make_snapshot
from tools.registry import registry


class AdapterPresentationTests(unittest.TestCase):
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
        diagnostics = [
            body["payload"]
            for body in bodies
            if body.get("kind") == "progress_observed"
            and isinstance(body.get("payload"), dict)
            and body["payload"].get("kind") == "runtime_diagnostics"
        ]
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
        diagnostics = [
            body["payload"]
            for body in bodies
            if body.get("kind") == "progress_observed"
            and isinstance(body.get("payload"), dict)
            and body["payload"].get("kind") == "runtime_diagnostics"
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(len(diagnostics[0]["requests"]), 5)  # type: ignore[index]
        self.assertEqual(diagnostics[0]["requests"][0]["toolChoice"], "required")  # type: ignore[index]
        self.assertEqual(diagnostics[0]["responses"][-1]["toolCall"], "execute")  # type: ignore[index]

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
        self.assertIn('"project_id"', str(captured["system_message"]))
        self.assertIn('"issue_id"', str(captured["system_message"]))


if __name__ == "__main__":
    unittest.main()
