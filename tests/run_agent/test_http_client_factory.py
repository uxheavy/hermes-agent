"""Behavioral coverage for the primary model HTTP-client injection seam."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.agent_runtime_helpers import create_openai_client
from agent.chat_completion_helpers import try_activate_fallback
from run_agent import AIAgent


class _FakeHttpClient:
    def __init__(self, serial: int):
        self.serial = serial
        self.is_closed = False

    def close(self):
        self.is_closed = True


class _FakeOpenAI:
    def __init__(self, constructed, **kwargs):
        self.kwargs = kwargs
        self._client = kwargs.get("http_client")
        self.is_closed = False
        constructed.append(self)

    def close(self):
        self.is_closed = True
        if self._client is not None:
            self._client.close()


def _make_http_client_factory(created):
    def factory():
        client = _FakeHttpClient(len(created))
        created.append(client)
        return client

    return factory


def test_invalid_http_client_factory_is_rejected_before_agent_initialization():
    with patch("agent.agent_init._install_safe_stdio") as install_safe_stdio:
        with pytest.raises(TypeError, match="callable or None"):
            AIAgent(http_client_factory=object())

    install_safe_stdio.assert_not_called()


def test_http_client_factory_none_result_fails_before_sdk_construction():
    agent = SimpleNamespace(
        provider="xai",
        _http_client_factory=lambda: None,
    )

    with patch("run_agent.OpenAI") as openai:
        with pytest.raises(TypeError, match="non-None HTTP client"):
            create_openai_client(
                agent,
                {"api_key": "xai-test-key", "base_url": "https://api.x.ai/v1"},
                reason="test_none_result",
                shared=True,
            )

    openai.assert_not_called()


def test_http_client_factory_exception_is_not_replaced_by_default_transport():
    expected = RuntimeError("relay construction failed")

    def factory():
        raise expected

    agent = SimpleNamespace(
        provider="xai",
        _http_client_factory=factory,
    )

    with patch("run_agent.OpenAI") as openai:
        with pytest.raises(RuntimeError) as raised:
            create_openai_client(
                agent,
                {"api_key": "xai-test-key", "base_url": "https://api.x.ai/v1"},
                reason="test_factory_exception",
                shared=True,
            )

    assert raised.value is expected
    openai.assert_not_called()


def _contains_identity(value, target, seen=None):
    if value is target:
        return True
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(
            _contains_identity(key, target, seen)
            or _contains_identity(item, target, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_identity(item, target, seen) for item in value)
    return False


def test_injected_http_client_factory_is_used_for_init_and_rebuild():
    created_http_clients = []
    constructed_clients = []
    factory = _make_http_client_factory(created_http_clients)
    fake_openai = lambda **kwargs: _FakeOpenAI(constructed_clients, **kwargs)

    with patch("run_agent.OpenAI", fake_openai):
        agent = AIAgent(
            api_key="xai-test-key",
            base_url="https://api.x.ai/v1",
            provider="xai",
            model="grok-4.5",
            http_client_factory=factory,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._retire_shared_openai_client = lambda client, reason: None

        assert agent._replace_primary_openai_client(reason="test_rebuild")

    assert len(created_http_clients) == 2
    assert len(constructed_clients) == 2
    assert constructed_clients[0].kwargs["http_client"] is created_http_clients[0]
    assert constructed_clients[1].kwargs["http_client"] is created_http_clients[1]
    assert created_http_clients[0] is not created_http_clients[1]
    assert not created_http_clients[1].is_closed

    # The trusted dependency belongs to the live agent only. It must not enter
    # the provider kwargs or the serialized primary-runtime snapshot.
    assert "http_client_factory" not in agent._client_kwargs
    assert "http_client_factory" not in agent._primary_runtime
    assert not _contains_identity(agent._client_kwargs, factory)
    assert not _contains_identity(agent._primary_runtime, factory)
    assert not _contains_identity(getattr(agent, "messages", []), factory)


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_injected_http_client_factory_survives_model_switch(mock_context_length):
    created_http_clients = []
    constructed_clients = []
    factory = _make_http_client_factory(created_http_clients)
    fake_openai = lambda **kwargs: _FakeOpenAI(constructed_clients, **kwargs)

    agent = AIAgent.__new__(AIAgent)
    agent.model = "grok-4.5"
    agent.provider = "xai"
    agent.requested_provider = "xai"
    agent.base_url = "https://api.x.ai/v1"
    agent.api_key = "xai-test-key"
    agent.api_mode = "chat_completions"
    agent.client = SimpleNamespace(is_closed=False)
    agent.quiet_mode = True
    agent._config_context_length = None
    agent._client_kwargs = {
        "api_key": "xai-test-key",
        "base_url": "https://api.x.ai/v1",
    }
    agent._http_client_factory = factory
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._transport_cache = {}
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = ""
    agent._is_anthropic_oauth = False
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.reasoning_config = None
    agent._cached_system_prompt = None
    agent._rate_limited_until = 0
    agent._retire_shared_openai_client = lambda client, reason: None
    agent._apply_client_headers_for_base_url = lambda base_url: None
    agent._anthropic_prompt_cache_policy = lambda **kwargs: (False, False)
    agent._ensure_lmstudio_runtime_loaded = lambda *args, **kwargs: None
    agent._lmstudio_load_was_unverified = lambda result: False
    agent._effective_lmstudio_context_length = lambda configured, runtime: configured
    agent.context_compressor = ContextCompressor(
        model=agent.model,
        threshold_percent=0.5,
        base_url=agent.base_url,
        api_key=agent.api_key,
        provider=agent.provider,
        quiet_mode=True,
        config_context_length=None,
    )

    with (
        patch("run_agent.OpenAI", fake_openai),
        patch("agent.credential_pool.load_pool", return_value=None),
    ):
        agent.client = agent._create_openai_client(
            agent._client_kwargs, reason="seed", shared=True
        )
        agent.switch_model(
            "gpt-4.1-mini",
            "custom",
            api_key="custom-test-key",
            base_url="https://model.example/v1",
        )

    assert len(created_http_clients) == 2
    assert len(constructed_clients) == 2
    assert constructed_clients[1].kwargs["http_client"] is created_http_clients[1]
    assert agent.provider == "custom"
    assert agent.model == "gpt-4.1-mini"
    assert not created_http_clients[1].is_closed
    assert not _contains_identity(agent._client_kwargs, factory)


def test_injected_http_client_factory_covers_openai_compatible_fallback():
    created_http_clients = []
    constructed_clients = []
    factory = _make_http_client_factory(created_http_clients)
    fake_openai = lambda **kwargs: _FakeOpenAI(constructed_clients, **kwargs)

    agent = MagicMock()
    agent.provider = "custom"
    agent.model = "primary-model"
    agent.base_url = "https://primary.example/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "primary-key"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = [{"provider": "openai-codex", "model": "gpt-5.5"}]
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._client_kwargs = {
        "api_key": "primary-key",
        "base_url": "https://primary.example/v1",
    }
    agent._transport_cache = {}
    agent._buffer_status = MagicMock()
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._provider_model_requires_responses_api.return_value = False
    agent._anthropic_prompt_cache_policy.return_value = (False, False)
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    agent.context_compressor = None
    agent._replace_primary_openai_client = MagicMock()
    agent._client_log_context.return_value = ""
    agent.__dict__["_http_client_factory"] = factory
    agent._create_openai_client.side_effect = (
        lambda kwargs, *, reason, shared: create_openai_client(
            agent, kwargs, reason=reason, shared=shared
        )
    )

    resolved_client = SimpleNamespace(
        api_key="fallback-key",
        base_url="https://chatgpt.com/backend-api/codex",
        _custom_headers={},
        close=MagicMock(),
    )
    fallback_pool = MagicMock()
    fallback_pool.provider = "openai-codex"
    fallback_pool.has_credentials.return_value = True

    with (
        patch("run_agent.OpenAI", fake_openai),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(resolved_client, "gpt-5.5"),
        ),
        patch("agent.credential_pool.load_pool", return_value=fallback_pool),
        patch(
            "agent.chat_completion_helpers.get_provider_request_timeout",
            return_value=None,
        ),
    ):
        assert try_activate_fallback(agent)

    assert len(created_http_clients) == 1
    assert agent.client is constructed_clients[0]
    assert agent.client.kwargs["http_client"] is created_http_clients[0]
    resolved_client.close.assert_called_once()
