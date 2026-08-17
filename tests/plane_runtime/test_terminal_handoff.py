"""Provider-free regressions for the Hermes terminal handoff seam."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PLANE_CODE_MODE_EXECUTE_OPERATION,
    PLANE_OUTCOME_PUBLISH_OPERATION,
)


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


def _tool_call(
    name: str,
    arguments: dict[str, object],
    call_id: str,
) -> SimpleNamespace:
    if name == "execute_code":
        function_name = name
        function_arguments = arguments
    else:
        function_name = "tool_call"
        function_arguments = {"name": name, "arguments": arguments}
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=function_name,
            arguments=json.dumps(function_arguments),
        ),
        extra_content=None,
    )


def test_real_hermes_long_route_preserves_applied_publication_after_observation_handoff_failure() -> None:
    """An applied product publication owns terminalization after its observation is attempted."""

    from plane_runtime.hermes_adapter import HermesKernelAdapter
    from run_agent import AIAgent
    from tests.plane_runtime.test_g1_runtime_process import (
        G1InvocationEnvelope,
        G1RunSnapshot,
        _digest,
        make_invocation,
        make_snapshot,
    )

    snapshot_raw = make_snapshot()
    snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
    snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
    snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
        "provider": "test-provider",
        "model": "deterministic-local",
    }
    snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
        "snapshot",
        {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
    )
    snapshot = G1RunSnapshot.from_dict(snapshot_raw)
    invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

    plan = [
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:catalog.search",
                "input": {"query": "work item", "limit": 8},
            },
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:catalog.describe",
                "input": {"operation_id": "work-item.read"},
            },
        ),
        (
            "execute_code",
            {"code": "export default ({ input }) => ({ ok: true, input });"},
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:work-item.read",
                "input": {"workItemRef": "work-item:test"},
            },
        ),
        (
            "plane_operation",
            {
                "action": "mutate",
                "operationRef": "operation:work-item-update",
                "input": {"workItemRef": "work-item:test", "title": "updated"},
            },
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:work-item.read",
                "input": {"workItemRef": "work-item:test", "forbidden": True},
            },
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:catalog.describe",
                "input": {"operation_id": "agent.outcome.submit"},
            },
        ),
        (
            "plane_operation",
            {
                "action": "mutate",
                "operationRef": "operation:agent.outcome.submit",
                "input": {
                    "run_ref": "run:test",
                    "summary": "bounded outcome",
                    "artifacts": [],
                    "evidence": [],
                },
            },
        ),
        (
            "plane_publish",
            {
                "kind": "outcome",
                "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                "resourceRef": "outcome-submission:test",
                "content": "explicit outcome publication",
            },
        ),
    ]

    class Completions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: object):
            if self.calls >= len(plan):
                raise AssertionError("Hermes made a provider call after terminal publication")
            name, arguments = plan[self.calls]
            self.calls += 1
            content = (
                "ordinary final transcript evidence"
                if self.calls == len(plan)
                else None
            )
            message = SimpleNamespace(
                content=content,
                tool_calls=[_tool_call(name, arguments, f"call-{self.calls}")],
                reasoning=None,
                reasoning_content=None,
                refusal=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    class Credentials:
        def resolve(self, provider: str) -> dict[str, str]:
            del provider
            return {
                "api_key": "provider-test-marker",
                "base_url": "http://127.0.0.1",
                "api_mode": "chat_completions",
            }

    client = Client()

    def agent_factory(**kwargs: object) -> AIAgent:
        agent = AIAgent(**kwargs)
        agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
        return agent

    requests: list[dict] = []
    outcome_ref = "outcome-submission:test"

    def rpc(request: dict) -> dict:
        requests.append(request)
        operation_ref = request["operationRef"]
        if operation_ref == "operation:catalog.describe":
            operation_id = request["input"]["operation_id"]
            return _result(
                request,
                output={
                    "operation": {
                        "operationId": operation_id,
                        "operationRef": f"operation:{operation_id}",
                        "inputSchema": {"type": "object"},
                    }
                },
            )
        if request["action"] == "code":
            assert operation_ref == PLANE_CODE_MODE_EXECUTE_OPERATION
            return _result(request, output={"accepted": True})
        if (
            operation_ref == "operation:work-item.read"
            and request["input"].get("forbidden") is True
        ):
            return _result(
                request,
                status="denied",
                errorCode="NOT_AUTHORIZED",
                errorMessage="policy denied this read",
            )
        if operation_ref == "operation:agent.outcome.submit":
            return _result(request, output={"outcomeRef": outcome_ref})
        if request["action"] == "publish":
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": outcome_ref,
                    "operationAttemptRef": "operation-attempt:test",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:agent-lifecycle",
                    "gatewayReceiptRef": "gateway-receipt:test",
                    "receiptRef": "receipt:test",
                    "auditReceiptRef": "audit-receipt:test",
                    "productEventRef": "product-event:test",
                },
            )
        return _result(request, output={"accepted": True})

    bodies: list[dict] = []

    def emit_body(body: dict) -> None:
        bodies.append(body)
        if body["kind"] == "outcome_submission_observed":
            raise RuntimeError("terminal observation handoff failed")

    with tempfile.TemporaryDirectory(prefix="hermes-plane-terminal-handoff-") as hermes_home_raw:
        hermes_home = Path(hermes_home_raw)
        with mock.patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            import run_agent

            run_agent._hermes_home = hermes_home
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                emit_body,
                model_call_allowance=12,
            )

    assert result.kind == "completed"
    assert result.output_text == "ordinary final transcript evidence"
    assert client.chat.completions.calls == len(plan)
    assert [request["action"] for request in requests].count("publish") == 1
    assert [request["operationRef"] for request in requests].count(
        PLANE_OUTCOME_PUBLISH_OPERATION
    ) == 1
    assert sum(body["kind"] == "outcome_submission_observed" for body in bodies) == 1
    assert sum(body["kind"] == "transcript_evidence_observed" for body in bodies) == 1
    assert not any(
        body.get("publication", {}).get("action") == "terminal"
        for body in bodies
    )
    assert "provider-test-marker" not in json.dumps(requests + bodies)
