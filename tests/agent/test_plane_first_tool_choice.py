"""Provider-free contract for Plane Code Mode's first tool request."""

from types import SimpleNamespace

from agent.chat_completion_helpers import (
    _plane_codex_request_overrides,
    _plane_first_tool_tools,
)
from agent.conversation_loop import (
    _record_plane_runtime_request,
    _record_plane_runtime_response,
)
from agent.transports import get_transport


_PLANE_TOOL = {
    "type": "function",
    "function": {
        "name": "plane_execute_typescript",
        "description": "Execute the commissioned TypeScript module.",
        "parameters": {"type": "object", "properties": {}},
    },
}
_OTHER_TOOL = {
    "type": "function",
    "function": {
        "name": "other_tool",
        "description": "Another tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}
_PUBLISH_TOOL = {
    "type": "function",
    "function": {
        "name": "plane_publish",
        "description": "Publish the commissioned outcome.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _request(agent, tools):
    tools = _plane_first_tool_tools(agent, tools)
    overrides = _plane_codex_request_overrides(agent, tools)
    return get_transport("codex_responses").build_kwargs(
        model="gpt-5.6",
        messages=[{"role": "user", "content": "run the commissioned action"}],
        tools=tools,
        request_overrides=overrides,
        is_codex_backend=True,
    )


def test_code_mode_first_request_filters_to_execute_then_restores_publish():
    agent = SimpleNamespace(
        request_overrides={},
        _plane_first_required_tool="plane_execute_typescript",
    )

    first = _request(agent, [_PLANE_TOOL, _PUBLISH_TOOL])
    assert first["tool_choice"] == "required"
    assert [tool["name"] for tool in first["tools"]] == ["plane_execute_typescript"]

    # The conversation loop clears the finite hint after the first matching
    # tool invocation. The same transport path then uses its normal auto mode.
    agent._plane_first_required_tool = None
    second = _request(agent, [_PLANE_TOOL, _PUBLISH_TOOL])
    assert second["tool_choice"] == "auto"
    assert [tool["name"] for tool in second["tools"]] == [
        "plane_execute_typescript",
        "plane_publish",
    ]


def test_invalid_or_absent_required_tool_hint_never_forces_a_choice():
    invalid = SimpleNamespace(
        request_overrides={},
        _plane_first_required_tool="plane_operation",
    )
    invalid_request = _request(invalid, [_PLANE_TOOL, _PUBLISH_TOOL])
    assert invalid_request["tool_choice"] == "auto"
    assert [tool["name"] for tool in invalid_request["tools"]] == [
        "plane_execute_typescript",
        "plane_publish",
    ]

    absent = SimpleNamespace(
        request_overrides={},
        _plane_first_required_tool="plane_execute_typescript",
    )
    absent_request = _request(absent, [_OTHER_TOOL, _PUBLISH_TOOL])
    assert absent_request["tool_choice"] == "auto"
    assert [tool["name"] for tool in absent_request["tools"]] == ["other_tool", "plane_publish"]

    standard = SimpleNamespace(request_overrides={}, _plane_first_required_tool=None)
    standard_request = _request(standard, [_PLANE_TOOL, _PUBLISH_TOOL])
    assert standard_request["tool_choice"] == "auto"
    assert [tool["name"] for tool in standard_request["tools"]] == [
        "plane_execute_typescript",
        "plane_publish",
    ]


def test_code_mode_diagnostic_records_serialized_choice_and_response_class_only():
    agent = SimpleNamespace(
        _plane_runtime_diagnostics={"requests": [], "responses": []},
    )

    _record_plane_runtime_request(
        agent,
        {
            "tool_choice": "required",
            "tools": [_PLANE_TOOL],
        },
    )
    _record_plane_runtime_response(agent, SimpleNamespace(tool_calls=[], content="redacted"))

    assert agent._plane_runtime_diagnostics == {
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
            {
                "sequence": 1,
                "responseClass": "text_response",
                "toolCall": "none",
            }
        ],
    }
