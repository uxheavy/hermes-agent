"""Provider-free ADR-0011 Codex Responses tool-choice contract."""

from types import SimpleNamespace

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PlaneHostBinding,
    PLANE_DISCOVER_TOOL,
    PLANE_EXECUTE_TOOL,
    bind_plane_host,
    install_plane_tools,
)
from agent.chat_completion_helpers import _plane_codex_request_overrides
from agent.transports import get_transport
from tools.registry import registry


_TOOLS = [
    {"type": "function", "name": PLANE_DISCOVER_TOOL, "parameters": {}},
    {"type": "function", "name": PLANE_EXECUTE_TOOL, "parameters": {}},
]


def _request(agent, tools=_TOOLS):
    return get_transport("codex_responses").build_kwargs(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "complete the assignment"}],
        tools=tools,
        request_overrides=_plane_codex_request_overrides(agent, tools),
        is_codex_backend=True,
    )


def _binding(declarations="declare const plane: unknown;"):
    return PlaneHostBinding(
        port=CallablePlaneHostPort(lambda request: request),
        run_id="run:test",
        invocation_id="invocation:test",
        correlation_id="correlation:test",
        cancellation=lambda: False,
        declaration_slice=declarations,
        plane_agent_route=True,
    )


def test_frozen_declarations_select_execute_and_missing_declarations_select_discover():
    execute = _binding()
    discover = _binding("")

    execute_agent = SimpleNamespace(
        request_overrides={},
        _plane_runtime_code_mode_tool_choice=execute.code_mode_tool_choice,
    )
    discover_agent = SimpleNamespace(
        request_overrides={},
        _plane_runtime_code_mode_tool_choice=discover.code_mode_tool_choice,
    )
    assert _request(execute_agent)["tool_choice"] == {
        "type": "function",
        "name": PLANE_EXECUTE_TOOL,
    }
    assert _request(discover_agent)["tool_choice"] == {
        "type": "function",
        "name": PLANE_DISCOVER_TOOL,
    }


def test_capability_miss_arms_discovery_then_discovery_returns_to_execute():
    binding = _binding()
    agent = SimpleNamespace(
        request_overrides={},
        _plane_runtime_code_mode_tool_choice=binding.code_mode_tool_choice,
    )

    assert _request(agent)["tool_choice"]["name"] == PLANE_EXECUTE_TOOL
    binding.observe_code_mode_result(
        {"status": "returned", "value": {"error": {"code": "CAPABILITY_NOT_FOUND"}}}
    )
    assert _request(agent)["tool_choice"]["name"] == PLANE_DISCOVER_TOOL
    binding.set_declaration_slice("declare const plane: { workItems: unknown };")
    assert _request(agent)["tool_choice"]["name"] == PLANE_EXECUTE_TOOL


def test_execute_handler_advances_the_host_owned_phase_on_capability_miss():
    install_plane_tools()
    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(
            lambda request: {
                "protocol": "plane.agent-runtime/v1",
                "requestRef": request["requestRef"],
                "correlationId": request["correlationId"],
                "idempotencyKey": request["idempotencyKey"],
                "status": "ok",
                "replayed": False,
                "output": {"status": "returned", "value": {"error": {"code": "CAPABILITY_NOT_FOUND"}}},
            }
        ),
        run_id="run:test",
        invocation_id="invocation:test",
        correlation_id="correlation:test",
        cancellation=lambda: False,
        declaration_slice="declare const plane: unknown;",
        plane_agent_route=True,
    )
    with bind_plane_host(binding):
        registry.dispatch(PLANE_EXECUTE_TOOL, {"code": "return 1;"})

    assert binding.code_mode_tool_choice() == PLANE_DISCOVER_TOOL


def test_standard_legacy_and_no_tools_keep_auto():
    agent = SimpleNamespace(request_overrides={})
    assert _plane_codex_request_overrides(agent, []) == {}
    assert "tools" not in _request(agent, [])
    assert _request(
        agent,
        [{"type": "function", "name": "legacy", "parameters": {}}],
    ).get("tool_choice", "auto") == "auto"
