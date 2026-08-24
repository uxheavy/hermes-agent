from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from agent import relay_llm, relay_runtime


class _HeldRelayStream:
    def __init__(self, closed: threading.Event) -> None:
        self._closed = closed

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise AssertionError("the held stream must be closed before consumption")

    async def aclose(self) -> None:
        self._closed.set()


class _FakeRelay:
    class LLMRequest:
        def __init__(self, headers, content) -> None:
            self.headers = headers
            self.content = content

    def __init__(self, streams):
        self.streams = streams
        self.llm = SimpleNamespace(stream_execute=self.stream_execute)

    async def stream_execute(
        self,
        _name,
        request,
        _provider_stream,
        _observe_chunk,
        _finalizer,
        **_kwargs,
    ):
        del request
        return self.streams.pop(0)


class _FakeRuntime:
    def __init__(self, relay):
        self.relay = relay

    def managed_execution_enabled(self) -> bool:
        return True

    async def run_in_session_async(self, _session, callback, *args, **kwargs):
        return await callback(*args, **kwargs)


def test_managed_stream_holds_session_until_relay_iterator_closes(monkeypatch):
    first_closed = threading.Event()
    second_closed = threading.Event()
    relay = _FakeRelay(
        [
            _HeldRelayStream(first_closed),
            _HeldRelayStream(second_closed),
        ]
    )
    runtime = _FakeRuntime(relay)
    session = SimpleNamespace(
        session_id="session-1",
        llm_lock=threading.Lock(),
    )
    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (runtime, session, None),
    )

    request = {"model": "test-model", "messages": []}
    first = relay_llm.stream(
        request,
        lambda _request: iter(()),
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=dict,
    )

    second_ready = threading.Event()
    second_holder = {}

    def start_second_call() -> None:
        second_holder["stream"] = relay_llm.stream(
            request,
            lambda _request: iter(()),
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            finalizer=dict,
        )
        second_ready.set()

    thread = threading.Thread(target=start_second_call)
    thread.start()
    time.sleep(0.05)
    assert second_ready.is_set() is False

    first.close()

    assert first_closed.is_set()
    assert second_ready.wait(timeout=1)
    second_holder["stream"].close()
    thread.join(timeout=1)
    assert thread.is_alive() is False
    assert second_closed.is_set()
