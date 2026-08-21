from types import SimpleNamespace

from agent.chat_completion_helpers import (
    _plane_codex_request_overrides,
    _plane_first_tool_tools,
)
from agent.transports import get_transport


_EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "plane_execute_typescript",
        "description": "Run the commissioned TypeScript module.",
        "parameters": {"type": "object", "properties": {}},
    },
}
_PUBLISH_TOOL = {
    "type": "function",
    "function": {
        "name": "plane_publish",
        "description": "Publish the commissioned result.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _request(agent, tools):
    return get_transport("codex_responses").build_kwargs(
        model=agent.model,
        messages=[{"role": "user", "content": "continue"}],
        tools=tools,
        request_overrides=_plane_codex_request_overrides(agent, tools),
        is_codex_backend=True,
    )


def test_post_search_phase_selects_named_code_mode_tool():
    consumed = []
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_hint=lambda: "post_search",
        _plane_runtime_code_mode_phase_consume=lambda: consumed.append(True),
    )

    request = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])

    assert request["tool_choice"] == {
        "type": "function",
        "name": "plane_execute_typescript",
    }
    assert consumed == [True]


def test_post_search_phase_consumer_is_one_shot():
    phase = ["post_search"]
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_hint=lambda: phase[0],
        _plane_runtime_code_mode_phase_consume=lambda: phase.__setitem__(0, None),
    )

    first = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])
    second = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])

    assert first["tool_choice"] == {
        "type": "function",
        "name": "plane_execute_typescript",
    }
    assert second["tool_choice"] == "auto"


def test_first_tool_precedence_is_preserved_over_post_search_hint():
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_first_required_tool="plane_execute_typescript",
        _plane_runtime_code_mode_phase_hint=lambda: "post_search",
    )

    request = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])

    assert request["tool_choice"] == "required"


def test_successful_submit_requires_explicit_publish_tool():
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_outcome_submission_pending_check=lambda: True,
    )

    request = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])

    assert request["tool_choice"] == "required"
    assert agent._plane_first_required_tool == "plane_publish"
    assert _plane_first_tool_tools(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL]) == [_PUBLISH_TOOL]


def test_publish_requirement_is_not_created_for_ordinary_plane_or_non_plane_turns():
    for agent in (
        SimpleNamespace(
            provider="openai-codex",
            model="gpt-5.6-codex",
            request_overrides={},
            _plane_runtime_outcome_submission_pending_check=lambda: False,
        ),
        SimpleNamespace(
            provider="other-provider",
            model="other-model",
            request_overrides={},
        ),
    ):
        request = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])
        assert request["tool_choice"] == "auto"
        assert getattr(agent, "_plane_first_required_tool", None) is None


def test_pre_search_invalid_and_out_of_policy_hints_fail_closed_to_auto():
    for hint, provider, model in (
        (None, "openai-codex", "gpt-5.6-codex"),
        ("invalid", "openai-codex", "gpt-5.6-codex"),
        ("post_search", "other-provider", "gpt-5.6-codex"),
        ("post_search", "openai-codex", "gpt-5.5-codex"),
    ):
        agent = SimpleNamespace(
            provider=provider,
            model=model,
            request_overrides={},
            _plane_runtime_code_mode_phase_hint=lambda hint=hint: hint,
        )
        request = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])
        assert request["tool_choice"] == "auto"
