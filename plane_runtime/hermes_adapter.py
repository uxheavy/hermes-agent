"""Hermes-only execution adapter behind the G1 runtime seam."""

from __future__ import annotations

import re
import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .g1_contract import (
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    MAX_PROMPT_BYTES,
    MAX_TEXT_BYTES,
)


class HermesCredentialSource(Protocol):
    """Trusted host source; credentials never come from a runtime envelope."""

    def resolve(self, provider: str) -> Mapping[str, str]:
        ...


class HermesCheckpointSource(Protocol):
    """Trusted host source for a previously approved continuation checkpoint."""

    def load(self, checkpoint_ref: str) -> Sequence[Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class EnvironmentCredentialSource:
    """Fail closed; ambient environment is never a credential channel."""

    def resolve(self, provider: str) -> Mapping[str, str]:
        del provider
        return {}


@dataclass(frozen=True)
class UnixSocketCredentialSource:
    """Read one provider credential from the host-only broker socket.

    The socket path is fixed and non-secret.  The credential value is returned
    only to this adapter and is never part of the G1 request or child
    environment.  A missing, malformed, or over-sized broker response fails
    closed.
    """

    path: str = "/run/plane-agent-credential-broker/broker.sock"
    timeout_seconds: float = 2.0

    def resolve(self, provider: str) -> Mapping[str, str]:
        request = json.dumps({"protocol": "plane.agent-runtime/credentials/v1", "provider": provider}, separators=(",", ":")).encode("utf-8") + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(self.timeout_seconds)
            channel.connect(self.path)
            channel.sendall(request)
            response = bytearray()
            while len(response) <= 16 * 1024:
                chunk = channel.recv(min(4096, 16 * 1024 + 1 - len(response)))
                if not chunk:
                    break
                response.extend(chunk)
                if response.endswith(b"\n"):
                    break
        if not response.endswith(b"\n"):
            raise G1ContractError("credential broker response is incomplete")
        value = json.loads(response[:-1])
        if not isinstance(value, dict) or value.get("protocol") != "plane.agent-runtime/credentials/v1":
            raise G1ContractError("credential broker response is invalid")
        credentials = value.get("credentials")
        if not isinstance(credentials, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in credentials.items()):
            raise G1ContractError("credential broker credentials are invalid")
        return credentials


class HermesAuthStoreCredentialSource:
    """Resolve the active Hermes credential in the trusted host process."""

    def resolve(self, provider: str) -> Mapping[str, str]:
        from agent.credential_pool import load_pool

        pool = load_pool(provider)
        entry = pool.current() or next(iter(pool.entries()), None)
        if entry is None or not entry.runtime_api_key:
            return {}
        credentials: dict[str, str] = {"api_key": entry.runtime_api_key}
        if entry.runtime_base_url:
            credentials["base_url"] = entry.runtime_base_url
        return credentials


@dataclass(frozen=True)
class HermesKernelResult:
    kind: str
    output_text: str = ""
    question: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    retryable: bool = True
    usage: Mapping[str, int] | None = None


def redact_runtime_text(value: str, secrets: Sequence[str]) -> str:
    """Remove host credential values from all runtime-visible text."""

    redacted = str(value)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    # Provider error messages occasionally include bearer-style material even
    # when the credential source did not expose the exact original string.
    return re.sub(r"(?i)\b(?:sk|api|token|bearer)[_-][A-Za-z0-9._-]{12,}\b", "[redacted]", redacted)


def bound_runtime_text(value: str, maximum_bytes: int) -> str:
    """Keep model and error text within the inline payload bound."""

    maximum_bytes = max(1, min(MAX_TEXT_BYTES, int(maximum_bytes)))
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return text
    marker = "\n[truncated]"
    available = max(0, maximum_bytes - len(marker.encode("utf-8")))
    return encoded[:available].decode("utf-8", errors="ignore") + marker


class HermesKernelAdapter:
    """Translate one G1 invocation to one existing ``AIAgent`` turn.

    This is deliberately the only Plane-runtime module that imports Hermes'
    agent class.  The factory and host sources are dependency seams for local
    verification; production uses the lazy default factory and trusted host
    credential source.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[..., Any] | None = None,
        credential_source: HermesCredentialSource | None = None,
        checkpoint_source: HermesCheckpointSource | None = None,
        enabled_toolsets: Sequence[str] = (),
    ) -> None:
        self._agent_factory = agent_factory or self._default_agent_factory
        self._credential_source = credential_source or EnvironmentCredentialSource()
        self._checkpoint_source = checkpoint_source
        self._enabled_toolsets = tuple(enabled_toolsets)

    @staticmethod
    def _default_agent_factory(**kwargs: Any) -> Any:
        # Keep the replaceable kernel import lazy and local to this adapter.
        from run_agent import AIAgent

        return AIAgent(**kwargs)

    def dispatch(
        self,
        snapshot: G1RunSnapshot,
        invocation: G1InvocationEnvelope,
        cancellation: Callable[[], bool],
        emit_body: Callable[[Mapping[str, Any]], None],
    ) -> HermesKernelResult:
        if cancellation():
            return HermesKernelResult(
                kind="cancelled",
                failure_code="cancelled",
                failure_message="runtime cancellation was requested",
                retryable=False,
            )
        checkpoint_ref = invocation.raw.get("checkpointRef")
        prefill_messages: Sequence[Mapping[str, Any]] | None = None
        if checkpoint_ref is not None:
            if self._checkpoint_source is None:
                return HermesKernelResult(
                    kind="failed",
                    failure_code="invalid_continuation",
                    failure_message="continuation checkpoint is not available to the trusted host",
                    retryable=False,
                )
            try:
                prefill_messages = self._checkpoint_source.load(str(checkpoint_ref))
            except Exception:
                return HermesKernelResult(
                    kind="failed",
                    failure_code="invalid_continuation",
                    failure_message="continuation checkpoint could not be reconstructed",
                    retryable=False,
                )

        event_limit = int(snapshot.raw["runtimePolicy"]["maxEventPayloadBytes"])
        try:
            credentials = dict(self._credential_source.resolve(snapshot.model_provider))
        except Exception:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted host credential resolution failed",
                retryable=True,
            )
        if not credentials:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted host credential source is required",
                retryable=False,
            )
        credential_values = tuple(credentials.values())
        emit_body(
            {
                "kind": "progress_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "Hermes invocation started."},
                "publication": {"action": "observation_only"},
            }
        )
        streamed: list[str] = []
        streamed_bytes = 0

        def on_delta(delta: Any) -> None:
            nonlocal streamed_bytes
            if delta is not None:
                text = redact_runtime_text(str(delta), credential_values)
                if streamed_bytes < MAX_PROMPT_BYTES:
                    bounded = bound_runtime_text(text, MAX_PROMPT_BYTES - streamed_bytes)
                    streamed.append(bounded)
                    streamed_bytes += len(bounded.encode("utf-8"))

        def on_step(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            if not cancellation():
                emit_body(
                    {
                        "kind": "progress_observed",
                        "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "Hermes tool loop advanced."},
                        "publication": {"action": "observation_only"},
                    }
                )

        agent_kwargs: dict[str, Any] = {
            "provider": snapshot.model_provider,
            "model": snapshot.model_name,
            # This is the model-dispatch output ceiling.  IterationBudget is
            # only a loop safeguard; it is not the token-budget authority.
            "max_tokens": max(1, int(invocation.remaining_budget["outputTokens"])),
            "max_iterations": min(90, max(1, int(invocation.remaining_budget["outputTokens"]))),
            "session_id": invocation.invocation_id,
            "enabled_toolsets": list(self._enabled_toolsets),
            "quiet_mode": True,
            "skip_context_files": True,
            "skip_memory": True,
            "save_trajectories": False,
            "checkpoints_enabled": False,
            "stream_delta_callback": on_delta,
            "step_callback": on_step,
        }
        try:
            from agent.iteration_budget import IterationBudget

            agent_kwargs["iteration_budget"] = IterationBudget(
                max(1, min(90, int(invocation.remaining_budget["outputTokens"])))
            )
        except Exception:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="Hermes iteration budget could not be constructed",
                retryable=False,
            )
        if credentials.get("api_key"):
            agent_kwargs["api_key"] = credentials["api_key"]
        if credentials.get("base_url"):
            agent_kwargs["base_url"] = credentials["base_url"]
        if credentials.get("api_mode"):
            agent_kwargs["api_mode"] = credentials["api_mode"]
        if prefill_messages is not None:
            agent_kwargs["prefill_messages"] = [dict(message) for message in prefill_messages]

        started_at = time.monotonic()
        try:
            agent = self._agent_factory(**agent_kwargs)
            prompt = (
                f"{snapshot.behavioral_prompt}\n\n"
                f"Assignment objective: {snapshot.objective}\n"
                "Context references are Plane-owned and immutable: "
                + ", ".join(str(item["contextRef"]) for item in snapshot.raw["context"])
            )
            result = agent.run_conversation(snapshot.objective, system_message=prompt)
        except Exception as exc:
            message = bound_runtime_text(redact_runtime_text(str(exc), credential_values), event_limit)
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message=message or "Hermes invocation failed",
                retryable=True,
            )

        if cancellation():
            return HermesKernelResult(
                kind="cancelled",
                failure_code="cancelled",
                failure_message="runtime cancellation was requested",
                retryable=False,
            )
        if not isinstance(result, Mapping):
            raise G1ContractError("Hermes adapter returned a non-object result")
        usage = {
            "inputTokens": max(0, int(getattr(agent, "session_input_tokens", 0) or 0)),
            "outputTokens": max(0, int(getattr(agent, "session_output_tokens", 0) or 0)),
            "durationMs": max(0, int((time.monotonic() - started_at) * 1000)),
        }
        if any(usage[name] > invocation.remaining_budget[name] for name in usage):
            return HermesKernelResult(
                kind="failed",
                failure_code="budget_exhausted",
                failure_message="Hermes usage exceeded the cumulative invocation budget",
                retryable=False,
                usage=usage,
            )
        if result.get("interrupted") is True:
            return HermesKernelResult(
                kind="cancelled",
                failure_code="cancelled",
                failure_message="Hermes invocation was interrupted",
                retryable=False,
            )
        if result.get("failed") is True:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message=bound_runtime_text(
                    redact_runtime_text(str(result.get("error") or "Hermes invocation failed"), credential_values),
                    event_limit,
                ),
                retryable=True,
                usage=usage,
            )
        question = result.get("input_request") or result.get("waiting_for_input")
        if isinstance(question, str) and question:
            return HermesKernelResult(
                kind="waiting_for_input",
                question=bound_runtime_text(redact_runtime_text(question, credential_values), event_limit),
                usage=usage,
            )
        output = result.get("final_response")
        if output is None:
            output = "".join(streamed)
        output_text = bound_runtime_text(
            redact_runtime_text(str(output or "Hermes invocation completed."), credential_values),
            event_limit,
        )
        emit_body(
            {
                "kind": "usage_observed",
                "usage": usage,
                "publication": {"action": "observation_only"},
            }
        )
        emit_body(
            {
                "kind": "transcript_evidence_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": output_text or "Hermes invocation completed."},
                "publication": {"action": "observation_only"},
            }
        )
        return HermesKernelResult(kind="completed", output_text=output_text, usage=usage)


class DeterministicKernelAdapter:
    """No-network adapter used by G1 subprocess fixtures."""

    def dispatch(
        self,
        snapshot: G1RunSnapshot,
        invocation: G1InvocationEnvelope,
        cancellation: Callable[[], bool],
        emit_body: Callable[[Mapping[str, Any]], None],
    ) -> HermesKernelResult:
        if cancellation():
            return HermesKernelResult("cancelled", failure_code="cancelled", failure_message="runtime cancellation was requested", retryable=False)
        emit_body(
            {
                "kind": "progress_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "Deterministic Hermes adapter started."},
                "publication": {"action": "observation_only"},
            }
        )
        emit_body(
            {
                "kind": "usage_observed",
                "usage": {"inputTokens": 0, "outputTokens": 0, "durationMs": 0},
                "publication": {"action": "observation_only"},
            }
        )
        output = bound_runtime_text(
            f"Deterministic outcome for {snapshot.objective}",
            int(snapshot.raw["runtimePolicy"]["maxEventPayloadBytes"]),
        )
        emit_body(
            {
                "kind": "transcript_evidence_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": output},
                "publication": {"action": "observation_only"},
            }
        )
        return HermesKernelResult("completed", output_text=output)


__all__ = [
    "DeterministicKernelAdapter",
    "EnvironmentCredentialSource",
    "HermesAuthStoreCredentialSource",
    "UnixSocketCredentialSource",
    "HermesCheckpointSource",
    "HermesCredentialSource",
    "HermesKernelAdapter",
    "HermesKernelResult",
    "bound_runtime_text",
    "redact_runtime_text",
]
