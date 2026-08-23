from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import run_agent

from plane_runtime.g1_contract import G1InvocationEnvelope, G1RunSnapshot
from plane_runtime.hermes_adapter import HermesKernelAdapter
from plane_runtime.host_port import CallablePlaneHostPort
from tests.plane_runtime.test_g1_runtime_process import _digest, make_invocation, make_snapshot


class _Completions:
    def __init__(self, standard_route: bool) -> None:
        self.calls = 0
        self.standard_route = standard_route

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        tools = kwargs.get("tools")
        exposed_tools = {
            tool.get("function", {}).get("name")
            for tool in tools or []
            if isinstance(tool, dict)
        }
        direct_standard_route = self.standard_route and (
            {"plane_operation", "plane_publish"}.issubset(exposed_tools)
            or exposed_tools == {"plane_operation"}
        ) and exposed_tools <= {
            "plane_operation", "plane_publish", "plane_execute_typescript"
        }
        if self.calls == 1 and direct_standard_route:
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call-search",
                        type="function",
                        function=SimpleNamespace(
                            name="plane_operation",
                            arguments='{"action":"read","operationRef":"operation:search_workspace","input":{"query":"assigned"}}',
                        ),
                    )
                ],
            )
            finish_reason = "tool_calls"
        else:
            message = SimpleNamespace(content="bounded", tool_calls=[])
            finish_reason = "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


class _OpenAI:
    def __init__(self, standard_route: bool, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_Completions(standard_route))
        self.responses = SimpleNamespace()


def _snapshot(role: str) -> G1RunSnapshot:
    raw = copy.deepcopy(make_snapshot())
    raw["profile"]["role"] = role  # type: ignore[index]
    raw["runtimePolicy"] = dict(raw["runtimePolicy"])  # type: ignore[arg-type]
    raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
    raw["runtimePolicy"]["model"] = {"provider": "test-provider", "model": "test-model"}  # type: ignore[index]
    raw["toolCatalog"]["eagerOperations"].append(  # type: ignore[index]
        {
            "operationRef": "operation:search_workspace",
            "schemaDigest": "content:" + "f" * 64,
            "inputSchema": {"type": "object"},
            "disclosure": "eager",
        }
    )
    if role == "worker":
        raw["toolCatalog"]["standardRoute"] = {  # type: ignore[index]
            "schemaVersion": "plane.standard-route/v1",
            "steps": [{"operationRef": "operation:search_workspace"}],
        }
    raw["contentDigest"] = _digest(
        "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
    )
    return G1RunSnapshot.from_dict(raw)


class WorkerOperatorReproTests(TestCase):
    def test_real_adapter_worker_and_manager_style_routes_reach_fake_provider(self) -> None:
        with tempfile.TemporaryDirectory() as hermes_home, mock.patch.dict(
            os.environ, {"HERMES_HOME": hermes_home}
        ), mock.patch.object(run_agent, "_hermes_home", Path(hermes_home)):
            for role in ("worker", "delegator"):
                fake_provider = lambda **kwargs: _OpenAI(role == "worker", **kwargs)
                snapshot = _snapshot(role)
                invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
                bodies: list[object] = []
                def host_rpc(request: dict[str, object]) -> dict[str, object]:
                    return {
                        "protocol": "plane.agent-runtime/v1",
                        "requestRef": request["requestRef"],
                        "correlationId": request["correlationId"],
                        "idempotencyKey": request["idempotencyKey"],
                        "status": "ok",
                        "replayed": False,
                        "output": {"result": {"results": []}},
                    }

                with mock.patch.object(run_agent, "OpenAI", fake_provider):
                    result = HermesKernelAdapter(
                        credential_source=type(
                            "Credentials",
                            (),
                            {"resolve": lambda _self, _provider: {
                                "api_key": "provider-free-test-secret",
                                "base_url": "http://provider.invalid/v1",
                                "api_mode": "chat_completions",
                            }},
                        )(),
                        host_port=CallablePlaneHostPort(host_rpc),
                    ).dispatch(snapshot, invocation, lambda: False, bodies.append, model_call_allowance=2)
                self.assertEqual(result.kind, "completed", role)
                self.assertEqual(result.model_calls, 2 if role == "worker" else 1, role)
