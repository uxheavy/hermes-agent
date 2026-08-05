"""Typed G1 transport facade over the host invocation supervisor."""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Mapping

from .g1_contract import (
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    bind_snapshot_and_invocation,
)
from .invocation_supervisor import G1InvocationSupervisor, G1LocalTestRunner


class RuntimeTransportError(G1ContractError):
    """The supervisor or its bounded G1 result was unusable."""


class SubprocessRuntimeTransport:
    """Accept exact G1 JSON and delegate launch, retention, and watchdogs.

    ``python_executable``, ``cwd``, and ``env`` remain accepted for source
    compatibility only.  They are intentionally ignored: child controls come
    from the supervisor's fixed test runner environment, never ambient state.
    A production runner must be installed at the supervisor seam; the local
    runner is explicitly test-classified and claims no OS isolation.
    """

    def __init__(
        self,
        *,
        python_executable: str = sys.executable,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 512 * 1024,
        supervisor: G1InvocationSupervisor | None = None,
    ) -> None:
        del python_executable, cwd, env
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._supervisor = supervisor or G1InvocationSupervisor(
            runner=G1LocalTestRunner(max_output_bytes=max_output_bytes),
            hard_timeout_seconds=timeout_seconds,
        )

    def dispatch(
        self,
        snapshot_json: str,
        envelope_json: str,
        *,
        cancellation: object | None = None,
        lease_valid: Callable[[], bool] | None = None,
    ) -> tuple[str, ...]:
        try:
            snapshot = G1RunSnapshot.from_dict(json.loads(snapshot_json))
            invocation = G1InvocationEnvelope.from_dict(json.loads(envelope_json))
            bind_snapshot_and_invocation(snapshot, invocation)
        except json.JSONDecodeError as exc:
            raise RuntimeTransportError("runtime request is not valid JSON") from exc
        if not callable(lease_valid):
            raise RuntimeTransportError("runtime lease validity callback is mandatory")
        return self._supervisor.run_once(
            snapshot,
            invocation,
            cancellation=cancellation,
            lease_valid=lease_valid,
        )

    def close(self) -> None:
        self._supervisor.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["RuntimeTransportError", "SubprocessRuntimeTransport"]
