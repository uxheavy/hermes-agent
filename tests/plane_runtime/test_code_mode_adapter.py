"""Provider-free contract tests for the Plane TypeScript Code Mode adapter."""

from __future__ import annotations

import json
import unittest

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    MAX_CODE_MODE_SOURCE_BYTES,
    PlaneHostBinding,
    PLANE_CODE_MODE_EXECUTE_OPERATION,
    PLANE_CODE_MODE_SCHEMA_VERSION,
    PLANE_CODE_MODE_TOOLSET,
    bind_plane_host,
    install_plane_tools,
)
from tools.registry import registry


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


def _binding(rpc, *, cancellation=lambda: False) -> PlaneHostBinding:
    return PlaneHostBinding(
        port=CallablePlaneHostPort(rpc),
        run_id="run:test",
        invocation_id="invocation:test",
        correlation_id="correlation:test",
        cancellation=cancellation,
    )


class CodeModeAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        install_plane_tools()

    def test_registration_precedes_python_and_retains_typescript_schema(self) -> None:
        entry = registry.get_entry("execute_code")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.toolset, PLANE_CODE_MODE_TOOLSET)
        self.assertEqual(entry.handler.__name__, "_handle_plane_code_mode")
        definitions = registry.get_definitions({"execute_code"})
        schema = definitions[0]["function"]
        self.assertEqual(schema["parameters"]["properties"], {
            "code": {
                "type": "string",
                "description": "Bounded TypeScript source exporting a default function.",
            }
        })
        self.assertIn("TypeScript", schema["description"])

    def test_typescript_dispatch_uses_exact_bound_four_field_capsule(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"value": "from-plane-isolate"})

        source = "export default ({ input }) => ({ ok: true, input });"
        with bind_plane_host(_binding(rpc)):
            result = registry.dispatch("execute_code", {"code": source})

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request["action"], "code")
        self.assertEqual(request["operationRef"], PLANE_CODE_MODE_EXECUTE_OPERATION)
        self.assertEqual(request["source"], "code")
        self.assertEqual(request["input"], {
            "schemaVersion": PLANE_CODE_MODE_SCHEMA_VERSION,
            "entrypoint": "default",
            "source": source,
            "input": {},
        })
        self.assertEqual(json.loads(result)["output"], {"value": "from-plane-isolate"})

    def test_python_source_is_opaque_and_never_falls_back_to_python(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"accepted": False, "error": "TypeScript required"})

        python_source = "from hermes_tools import plane_operation\nprint('must not execute')"
        with bind_plane_host(_binding(rpc)):
            result = registry.dispatch("execute_code", {"code": python_source})

        self.assertEqual(requests[0]["input"]["source"], python_source)
        self.assertEqual(json.loads(result)["output"]["accepted"], False)

    def test_malformed_and_oversized_source_fail_before_host(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request)

        with bind_plane_host(_binding(rpc)):
            empty = registry.dispatch("execute_code", {"code": "   "})
            oversized = registry.dispatch(
                "execute_code", {"code": "x" * (MAX_CODE_MODE_SOURCE_BYTES + 1)}
            )
            unknown = registry.dispatch("execute_code", {"code": "ok", "input": {}})

        self.assertEqual(requests, [])
        for result in (empty, oversized, unknown):
            self.assertEqual(json.loads(result)["status"], "error")

    def test_denial_and_host_error_are_bounded_model_results(self) -> None:
        def denied(request: dict) -> dict:
            return _result(
                request,
                status="denied",
                output=None,
                errorCode="NOT_AUTHORIZED",
                errorMessage="host denied the Code Mode action",
            )

        denied_binding = _binding(denied)
        with bind_plane_host(denied_binding):
            denied_result = registry.dispatch("execute_code", {"code": "export default () => 1;"})
        denied_payload = json.loads(denied_result)
        self.assertEqual(denied_payload["status"], "denied")
        self.assertEqual(denied_payload["errorCode"], "NOT_AUTHORIZED")
        self.assertIsNone(denied_binding.fatal_error)

        def unavailable(request: dict) -> dict:
            return _result(
                request,
                status="unavailable",
                errorCode="OPERATION_UNAVAILABLE",
                errorMessage="Code Mode host unavailable",
            )

        unavailable_binding = _binding(unavailable)
        with bind_plane_host(unavailable_binding):
            unavailable_result = registry.dispatch("execute_code", {"code": "export default () => 1;"})
        unavailable_payload = json.loads(unavailable_result)
        self.assertEqual(unavailable_payload["status"], "unavailable")
        self.assertEqual(unavailable_payload["errorCode"], "OPERATION_UNAVAILABLE")
        self.assertEqual(unavailable_binding.fatal_error, "Code Mode host unavailable")

    def test_cancellation_and_oversized_host_output_fail_closed(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output="x" * 13_000)

        cancelled_binding = _binding(rpc, cancellation=lambda: True)
        with bind_plane_host(cancelled_binding):
            cancelled = registry.dispatch("execute_code", {"code": "export default () => 1;"})
        self.assertEqual(json.loads(cancelled)["error"]["code"], "cancelled")
        self.assertEqual(requests, [])

        output_binding = _binding(rpc)
        with bind_plane_host(output_binding):
            oversized = registry.dispatch("execute_code", {"code": "export default () => 1;"})
        self.assertEqual(json.loads(oversized)["status"], "error")
        self.assertIn("host.modelResult", json.loads(oversized)["error"]["message"])


if __name__ == "__main__":
    unittest.main()
