"""Provider-free proof for the ADR-0011 Hermes model boundary."""

from __future__ import annotations

import json
import copy
import unittest

from plane_runtime.g1_contract import G1InvocationEnvelope, G1RunSnapshot
from plane_runtime.hermes_adapter import HermesKernelAdapter
from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PlaneHostBinding,
    PLANE_AGENT_TOOLSET,
    PLANE_DISCOVER_TOOL,
    PLANE_DISCOVERY_OPERATION,
    PLANE_EXECUTE_TOOL,
    PLANE_CODE_MODE_EXECUTE_OPERATION,
    PLANE_RUNTIME_TOOLSET,
    install_plane_tools,
    current_plane_host,
)
from tools.registry import registry
from tests.plane_runtime.test_g1_runtime_process import _digest, make_invocation, make_snapshot


def _result(request: dict, *, output: object = None, status: str = "ok", **extra: object) -> dict:
    return {
        "protocol": "plane.agent-runtime/v1",
        "requestRef": request["requestRef"],
        "correlationId": request["correlationId"],
        "idempotencyKey": request["idempotencyKey"],
        "status": status,
        "replayed": status == "replayed",
        "output": output,
        **extra,
    }


def _plane_snapshot() -> dict[str, object]:
    raw = copy.deepcopy(make_snapshot())
    raw["toolCatalog"] = {
        "catalogDigest": "content:" + "a" * 64,
        "server": "Plane",
        "tools": [
            {
                "name": "discover",
                "description": "The complete intended workflow.",
                "inputSchema": {"type": "object", "required": ["query"]},
            },
            {
                "name": "execute",
                "description": "TypeScript statements executed as an async function body.",
                "inputSchema": {"type": "object", "required": ["code"]},
            },
        ],
        "taskKit": {
            "task": {
                "target": "target:test",
                "objective": "Persisted objective.",
                "acceptanceCriteria": ["Persisted criterion."],
            },
            "declarations": "persisted declarations",
            "example": "persisted example",
        },
    }
    raw["contentDigest"] = _digest(
        "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
    )
    return raw


class Adr0011CodeModeTests(unittest.TestCase):
    def setUp(self) -> None:
        install_plane_tools()

    def test_plane_route_exposes_exactly_two_new_tools(self) -> None:
        definitions = registry.get_definitions(
            set(registry.get_tool_names_for_toolset(PLANE_AGENT_TOOLSET))
        )
        names = {item["function"]["name"] for item in definitions}
        self.assertEqual(names, {PLANE_DISCOVER_TOOL, PLANE_EXECUTE_TOOL})
        self.assertNotIn("plane_operation", names)
        self.assertNotIn("plane_execute_typescript", names)
        self.assertNotIn("plane_publish", names)

        discover = next(item["function"] for item in definitions if item["function"]["name"] == PLANE_DISCOVER_TOOL)
        execute = next(item["function"] for item in definitions if item["function"]["name"] == PLANE_EXECUTE_TOOL)
        self.assertEqual(discover["parameters"]["required"], ["query"])
        self.assertEqual(discover["parameters"]["properties"]["query"]["maxLength"], 500)
        self.assertEqual(
            discover["parameters"]["properties"]["query"]["description"],
            "The complete intended workflow, for example: list urgent unassigned work items, assign one member, then finish.",
        )
        self.assertEqual(execute["parameters"]["required"], ["code"])
        self.assertEqual(execute["parameters"]["properties"]["code"]["maxLength"], 8192)
        self.assertEqual(
            execute["parameters"]["properties"]["code"]["description"],
            "TypeScript statements executed as an async function body with ambient plane and task objects. Imports and exports are forbidden.",
        )
        for description in (discover["description"], execute["description"]):
            self.assertNotIn("operationRef", description)
            self.assertNotIn("prepared-call", description)
            self.assertNotIn("publish", description)

    def test_discover_routes_bounded_query_and_replaces_declaration_slot(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"declarations": "declare namespace plane { const workItems: unknown; }"})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            task_kit='{"target":"target:test","objective":"do it","acceptanceCriteria":["done"]}',
            declaration_slice="initial",
        )
        from plane_runtime.host_port import bind_plane_host

        with bind_plane_host(binding):
            result = json.loads(registry.dispatch(PLANE_DISCOVER_TOOL, {"query": "list work items, then finish"}))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(requests[0]["action"], "discover")
        self.assertEqual(requests[0]["operationRef"], PLANE_DISCOVERY_OPERATION)
        self.assertEqual(requests[0]["input"], {"query": "list work items, then finish"})
        self.assertEqual(binding.declaration_slice, result["declarations"])

    def test_model_bounds_count_characters(self) -> None:
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: _result(request)),
            run_id="run:test",
            invocation_id="invocation:bounds",
            correlation_id="correlation:bounds",
            cancellation=lambda: False,
        )
        from plane_runtime.host_port import bind_plane_host

        with bind_plane_host(binding):
            discover_error = json.loads(registry.dispatch(PLANE_DISCOVER_TOOL, {"query": "x" * 501}))
            execute_error = json.loads(registry.dispatch(PLANE_EXECUTE_TOOL, {"code": "x" * 8193}))
        self.assertEqual(discover_error["error"]["code"], "DISCOVER_FAILED")
        self.assertEqual(execute_error["error"]["code"], "SOURCE_TOO_LARGE")

    def test_run_snapshot_uses_plane_authored_task_kit_and_declarations(self) -> None:
        snapshot = G1RunSnapshot.from_dict(_plane_snapshot())
        self.assertTrue(snapshot.is_plane_code_mode)
        self.assertEqual(snapshot.plane_task["objective"], "Persisted objective.")
        self.assertEqual(snapshot.plane_initial_declarations, "persisted declarations")
        self.assertEqual(snapshot.plane_example, "persisted example")

    def test_execute_routes_only_code_and_projects_terminal_results(self) -> None:
        for terminal in ("completed", "waiting_for_input", "blocked"):
            requests: list[dict] = []

            def rpc(request: dict, terminal: str = terminal) -> dict:
                requests.append(request)
                return _result(request, output={"result": {"terminal": {"kind": terminal}}})

            binding = PlaneHostBinding(
                port=CallablePlaneHostPort(rpc),
                run_id="run:test",
                invocation_id=f"invocation:{terminal}",
                correlation_id="correlation:test",
                cancellation=lambda: False,
                task_kit='{"target":"target:test"}',
                declaration_slice="declare const plane: unknown;",
            )
            from plane_runtime.host_port import bind_plane_host

            with bind_plane_host(binding):
                result = json.loads(registry.dispatch(PLANE_EXECUTE_TOOL, {"code": "await plane.finish({ kind: \"" + terminal + "\" });"}))

            self.assertEqual(result, {"status": terminal})
            self.assertEqual(requests[0]["action"], "code")
            self.assertEqual(requests[0]["operationRef"], PLANE_CODE_MODE_EXECUTE_OPERATION)
            self.assertEqual(requests[0]["input"], {"code": "await plane.finish({ kind: \"" + terminal + "\" });"})

    def test_execute_returns_compact_value_and_explicit_missing_finish_error(self) -> None:
        outputs = iter(({"status": "returned", "value": {"ok": True}}, None))

        def rpc(request: dict) -> dict:
            return _result(request, output=next(outputs))

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        from plane_runtime.host_port import bind_plane_host

        with bind_plane_host(binding):
            returned = json.loads(registry.dispatch(PLANE_EXECUTE_TOOL, {"code": "return { ok: true };"}))
            missing = json.loads(registry.dispatch(PLANE_EXECUTE_TOOL, {"code": "const x = 1;"}))

        self.assertEqual(returned, {"status": "returned", "value": {"ok": True}})
        self.assertEqual(missing["error"]["code"], "MISSING_TERMINAL_PUBLICATION")

    def test_legacy_hermes_python_tool_remains_registered_for_non_plane_routes(self) -> None:
        import tools.code_execution_tool  # noqa: F401

        self.assertIsNotNone(registry.get_entry("execute_code"))
        self.assertIsNotNone(registry.get_entry("plane_operation"))

    def test_adapter_binds_task_kit_and_uses_only_plane_route_tools(self) -> None:
        snapshot = G1RunSnapshot.from_dict(_plane_snapshot())
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        captured: dict[str, object] = {}

        class FakeAgent:
            session_input_tokens = 0
            session_output_tokens = 0
            session_api_calls = 1

            def run_conversation(self, _message: str, *, system_message: str) -> dict[str, str]:
                captured["prompt"] = system_message
                captured["toolsets"] = self.enabled_toolsets
                binding = current_plane_host()
                assert binding is not None
                captured["task_kit"] = binding.task_kit
                return {"final_response": "technical transcript"}

        def factory(**kwargs: object) -> FakeAgent:
            agent = FakeAgent()
            agent.enabled_toolsets = kwargs["enabled_toolsets"]  # type: ignore[attr-defined]
            return agent

        class Credentials:
            def resolve(self, _provider: str) -> dict[str, str]:
                return {"api_key": "provider-free-test-secret"}

        result = HermesKernelAdapter(
            agent_factory=factory,
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(lambda request: _result(request)),
        ).dispatch(snapshot, invocation, lambda: False, lambda _body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "failed")
        self.assertIn("MISSING_TERMINAL_PUBLICATION", result.failure_message or "")
        self.assertEqual(captured["toolsets"], [PLANE_AGENT_TOOLSET])
        prompt = str(captured["prompt"])
        self.assertIn('"target":"target:test"', prompt)
        self.assertIn("Plane:discover", prompt)
        self.assertNotIn("plane_operation", prompt)
        self.assertNotIn("plane_publish", prompt)
        self.assertNotIn("prepared-call", prompt)
        self.assertIn('"objective":"Persisted objective."', str(captured["task_kit"]))
        self.assertIn("persisted declarations", prompt)
        self.assertIn("persisted example", prompt)

    def test_non_plane_adapter_keeps_existing_toolset_path(self) -> None:
        snapshot = G1RunSnapshot.from_dict(make_snapshot())
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        captured: dict[str, object] = {}

        class FakeAgent:
            session_input_tokens = 0
            session_output_tokens = 0
            session_api_calls = 1

            def run_conversation(self, _message: str, *, system_message: str) -> dict[str, str]:
                captured["prompt"] = system_message
                return {"final_response": "ordinary Hermes result"}

        def factory(**kwargs: object) -> FakeAgent:
            captured["toolsets"] = kwargs["enabled_toolsets"]
            return FakeAgent()

        class Credentials:
            def resolve(self, _provider: str) -> dict[str, str]:
                return {"api_key": "provider-free-test-secret"}

        result = HermesKernelAdapter(
            agent_factory=factory,
            credential_source=Credentials(),
            enabled_toolsets=["code_execution"],
        ).dispatch(snapshot, invocation, lambda: False, lambda _body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "completed")
        self.assertEqual(captured["toolsets"], [])
        self.assertNotIn(PLANE_AGENT_TOOLSET, captured["toolsets"])

    def test_legacy_snapshot_with_host_port_keeps_legacy_route(self) -> None:
        snapshot = G1RunSnapshot.from_dict(make_snapshot())
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        captured: dict[str, object] = {}

        class FakeAgent:
            session_input_tokens = 0
            session_output_tokens = 0
            session_api_calls = 1

            def run_conversation(self, _message: str, *, system_message: str) -> dict[str, str]:
                del system_message
                binding = current_plane_host()
                assert binding is not None
                captured["plane_agent_route"] = binding.plane_agent_route
                captured["task_kit"] = binding.task_kit
                captured["declarations"] = binding.declaration_slice
                return {"final_response": "legacy transcript"}

        def factory(**kwargs: object) -> FakeAgent:
            captured["toolsets"] = kwargs["enabled_toolsets"]
            return FakeAgent()

        class Credentials:
            def resolve(self, _provider: str) -> dict[str, str]:
                return {"api_key": "provider-free-test-secret"}

        result = HermesKernelAdapter(
            agent_factory=factory,
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(lambda request: _result(request)),
        ).dispatch(snapshot, invocation, lambda: False, lambda _body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "completed")
        self.assertEqual(captured["toolsets"], [PLANE_RUNTIME_TOOLSET])
        self.assertFalse(captured["plane_agent_route"])
        self.assertEqual(captured["task_kit"], "{}")
        self.assertEqual(captured["declarations"], "")


if __name__ == "__main__":
    unittest.main()
