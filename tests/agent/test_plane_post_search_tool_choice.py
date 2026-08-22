from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import (
    PlaneCodeModeContinuationError,
    _plane_codex_request_overrides,
    _plane_first_tool_tools,
    _plane_standard_request_overrides,
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

_PLANE_OPERATION_TOOL = {
    "type": "function",
    "function": {
        "name": "plane_operation",
        "description": "Call one Plane operation.",
        "parameters": {"type": "object"},
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


def test_standard_route_pending_prepared_read_requires_existing_plane_operation_tool():
    agent = SimpleNamespace(
        request_overrides={"temperature": 0},
        _plane_runtime_prepared_read_pending_check=lambda: True,
    )

    overrides = _plane_standard_request_overrides(
        agent, [_PLANE_OPERATION_TOOL, _PUBLISH_TOOL]
    )

    assert overrides == {
        "temperature": 0,
        "tool_choice": {
            "type": "function",
            "function": {"name": "plane_operation"},
        },
    }


def test_standard_route_pending_prepared_read_fails_closed_for_invalid_signals():
    for pending in (
        False,
        None,
        {"preparedCallRef": "prepared-call:tampered", "extra": True},
        "x" * 257,
    ):
        agent = SimpleNamespace(
            request_overrides={"temperature": 0},
            _plane_runtime_prepared_read_pending_check=lambda pending=pending: pending,
        )

        assert _plane_standard_request_overrides(
            agent, [_PLANE_OPERATION_TOOL]
        ) == {"temperature": 0}


def test_standard_route_pending_prepared_read_does_not_create_a_tool_path():
    agent = SimpleNamespace(
        request_overrides={},
        _plane_runtime_prepared_read_pending_check=lambda: True,
    )

    assert _plane_standard_request_overrides(agent, [_PUBLISH_TOOL]) == {}


def test_standard_route_required_tool_callback_reuses_first_tool_redirect():
    agent = SimpleNamespace(
        request_overrides={},
        _plane_runtime_required_tool_check=lambda: "plane_operation",
    )

    overrides = _plane_standard_request_overrides(agent, [_PLANE_OPERATION_TOOL])

    assert overrides["tool_choice"] == {
        "type": "function",
        "function": {"name": "plane_operation"},
    }
    assert agent._plane_first_required_tool == "plane_operation"


def test_standard_route_unavailable_tool_clears_required_latch():
    agent = SimpleNamespace(
        request_overrides={},
        _plane_runtime_required_tool_check=lambda: "plane_operation",
    )

    assert _plane_standard_request_overrides(agent, [_PUBLISH_TOOL]) == {}
    assert agent._plane_first_required_tool is None


def test_post_search_phase_selects_named_code_mode_tool():
    consumed = []
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_consume=lambda **_kwargs: (
            consumed.append(True) or "post_search"
        ),
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
        _plane_runtime_code_mode_phase_consume=lambda **_kwargs: (
            phase.pop(0) if phase else None
        ),
    )

    first = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])
    second = _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])

    assert first["tool_choice"] == {
        "type": "function",
        "name": "plane_execute_typescript",
    }
    assert second["tool_choice"] == "auto"


def test_post_search_phase_requires_atomic_consumer():
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_hint=lambda: "post_search",
    )

    with pytest.raises(PlaneCodeModeContinuationError) as error:
        _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])

    assert error.value.reason == "phase_consumer_unavailable"


def test_post_search_phase_consumer_must_return_trusted_phase():
    consumed = []
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_hint=lambda: "post_search",
        _plane_runtime_code_mode_phase_consume=lambda **_kwargs: consumed.append(True),
    )

    with pytest.raises(PlaneCodeModeContinuationError) as error:
        _request(agent, [_EXECUTE_TOOL, _PUBLISH_TOOL])

    assert error.value.reason == "phase_consumer_invalid"
    assert consumed == [True]


def test_post_search_phase_is_not_consumed_without_execute_tool():
    phase = ["post_search"]
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_hint=lambda: "post_search",
        _plane_runtime_code_mode_phase_consume=lambda **_kwargs: phase.pop(0),
    )

    with pytest.raises(PlaneCodeModeContinuationError) as error:
        _request(agent, [_PUBLISH_TOOL])

    assert error.value.reason == "execute_tool_unavailable"
    assert phase == ["post_search"]


def test_post_search_phase_consumer_failure_fails_closed_without_provider_request():
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_consume=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("broken phase state")
        ),
    )

    with pytest.raises(PlaneCodeModeContinuationError) as error:
        _plane_codex_request_overrides(agent, [_EXECUTE_TOOL])

    assert error.value.reason == "phase_consume_failed"


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


def test_code_mode_ref_recovery_requires_one_named_execute_continuation():
    agent = SimpleNamespace(
        request_overrides={},
        _plane_runtime_code_mode_continuation_required_check=lambda: True,
    )

    overrides = _plane_standard_request_overrides(
        agent, [_EXECUTE_TOOL, _PUBLISH_TOOL]
    )

    assert overrides["tool_choice"] == {
        "type": "function",
        "function": {"name": "plane_execute_typescript"},
    }
    assert agent._plane_first_required_tool == "plane_execute_typescript"


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


def test_invalid_plane_phase_hint_fails_closed():
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-codex",
        request_overrides={},
        _plane_runtime_code_mode_phase_hint=lambda: "invalid",
    )

    with pytest.raises(PlaneCodeModeContinuationError) as error:
        _plane_codex_request_overrides(agent, [_EXECUTE_TOOL])

    assert error.value.reason == "phase_hint_invalid"
