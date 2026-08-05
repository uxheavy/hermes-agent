"""Hermes-only execution adapter behind the G1 runtime seam."""

from __future__ import annotations

import os
import re
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
    """Read only explicitly runtime-scoped environment credentials."""

    def resolve(self, provider: str) -> Mapping[str, str]:
        del provider
        values = {
            "api_key": os.environ.get("HERMES_RUNTIME_API_KEY", ""),
            "base_url": os.environ.get("HERMES_RUNTIME_BASE_URL", ""),
            "api_mode": os.environ.get("HERMES_RUNTIME_API_MODE", ""),
        }
        return {key: value for key, value in values.items() if value}


@dataclass(frozen=True)
class HermesKernelResult:
    kind: str
    output_text: str = ""
    question: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    retryable: bool = True


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
        if credentials.get("api_key"):
            agent_kwargs["api_key"] = credentials["api_key"]
        if credentials.get("base_url"):
            agent_kwargs["base_url"] = credentials["base_url"]
        if credentials.get("api_mode"):
            agent_kwargs["api_mode"] = credentials["api_mode"]
        if prefill_messages is not None:
            agent_kwargs["prefill_messages"] = [dict(message) for message in prefill_messages]

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
            )
        question = result.get("input_request") or result.get("waiting_for_input")
        if isinstance(question, str) and question:
            return HermesKernelResult(
                kind="waiting_for_input",
                question=bound_runtime_text(redact_runtime_text(question, credential_values), event_limit),
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
                "kind": "transcript_evidence_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": output_text or "Hermes invocation completed."},
                "publication": {"action": "observation_only"},
            }
        )
        return HermesKernelResult(kind="completed", output_text=output_text)


class DeterministicKernelAdapter:
    """No-network adapter used by G1 subprocess fixtures."""

    def dispatch(
        self,
        snapshot: G1RunSnapshot,
        invocation: G1InvocationEnvelope,
        cancellation: Callable[[], bool],
        emit_body: Callable[[Mapping[str, Any]], None],
    ) -> HermesKernelResult:
        del invocation
        if cancellation():
            return HermesKernelResult("cancelled", failure_code="cancelled", failure_message="runtime cancellation was requested", retryable=False)
        emit_body(
            {
                "kind": "progress_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "Deterministic Hermes adapter started."},
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
    "HermesCheckpointSource",
    "HermesCredentialSource",
    "HermesKernelAdapter",
    "HermesKernelResult",
    "bound_runtime_text",
    "redact_runtime_text",
]
