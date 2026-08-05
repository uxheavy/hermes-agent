"""Minimal bounded subprocess transport for one G1 Hermes invocation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping

from .g1_contract import (
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    MAX_EVENT_BYTES,
    bind_snapshot_and_invocation,
    build_exit,
    validate_g1_frames,
)


class RuntimeTransportError(G1ContractError):
    """The child process or its bounded wire response was unusable."""


def _is_cancelled(value: object | None) -> bool:
    if value is None:
        return False
    method = getattr(value, "is_set", None) or getattr(value, "is_cancelled", None)
    return bool(method()) if callable(method) else bool(value)


def _failed_frame(
    snapshot: G1RunSnapshot,
    invocation: G1InvocationEnvelope,
    *,
    kind: str,
    code: str,
    message: str,
    retryable: bool = False,
) -> str:
    frame = build_exit(
        snapshot=snapshot,
        invocation=invocation,
        final_sequence=0,
        kind=kind,
        failure={"code": code, "message": message, "retryable": retryable},
    )
    return json.dumps(frame, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SubprocessRuntimeTransport:
    """Spawn a fresh Hermes runtime process for each invocation.

    The transport deliberately returns serialized direct frames.  Callers
    validate them at their authority boundary; this class validates as well so
    malformed child output never escapes the transport unchecked.
    """

    def __init__(
        self,
        *,
        python_executable: str = sys.executable,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 512 * 1024,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_bytes < MAX_EVENT_BYTES:
            raise ValueError("max_output_bytes is too small for one G1 frame")
        self._python_executable = python_executable
        self._cwd = str(cwd or Path(__file__).resolve().parents[1])
        self._env = dict(env) if env is not None else None
        self._timeout_seconds = float(timeout_seconds)
        self._max_output_bytes = int(max_output_bytes)

    def _child_env(self) -> dict[str, str]:
        # Do not inherit the host environment wholesale: credentials and
        # unrelated process configuration must not cross the service boundary.
        if self._env is not None:
            env = dict(self._env)
        else:
            env = {"PATH": os.environ.get("PATH", "")}
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        source_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = source_root if not existing_pythonpath else source_root + os.pathsep + existing_pythonpath
        return env

    def dispatch(
        self,
        snapshot_json: str,
        envelope_json: str,
        *,
        cancellation: object | None = None,
        lease_valid: Callable[[], bool] | None = None,
    ) -> tuple[str, ...]:
        try:
            snapshot_raw = json.loads(snapshot_json)
            invocation_raw = json.loads(envelope_json)
            snapshot = G1RunSnapshot.from_dict(snapshot_raw)
            invocation = G1InvocationEnvelope.from_dict(invocation_raw)
            bind_snapshot_and_invocation(snapshot, invocation)
        except (json.JSONDecodeError, TypeError, G1ContractError) as exc:
            if isinstance(exc, G1ContractError):
                raise
            raise G1ContractError("runtime request is not valid JSON") from exc

        if _is_cancelled(cancellation):
            return (_failed_frame(snapshot, invocation, kind="cancelled", code="cancelled", message="runtime cancellation was requested"),)
        if lease_valid is not None and not lease_valid():
            return (_failed_frame(snapshot, invocation, kind="failed", code="lease_expired", message="invocation lease is no longer valid"),)

        request_line = json.dumps(
            {"run": snapshot.to_dict(), "invocation": invocation.to_dict()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        if len(request_line.encode("utf-8")) > 3 * 128 * 1024:
            raise RuntimeTransportError("runtime request exceeds transport bound")

        try:
            process = subprocess.Popen(
                [self._python_executable, "-m", "plane_runtime.service", "--once"],
                cwd=self._cwd,
                env=self._child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeTransportError("runtime child could not be started") from exc

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            process.stdin.write(request_line.encode("utf-8"))
            process.stdin.close()
        except OSError as exc:
            self._terminate(process)
            raise RuntimeTransportError("runtime request could not be written") from exc

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        def read_stream(stream, chunks: list[bytes], limit: int, label: str) -> None:
            total = 0
            while True:
                chunk = stream.read(16 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise RuntimeTransportError(f"runtime {label} exceeded its output bound")
                chunks.append(chunk)

        stdout_error: list[BaseException] = []
        stderr_error: list[BaseException] = []

        def guarded(stream, chunks, errors, label):
            try:
                read_stream(stream, chunks, self._max_output_bytes, label)
            except BaseException as exc:  # thread boundary; re-raise in owner
                errors.append(exc)

        out_thread = threading.Thread(target=guarded, args=(process.stdout, stdout_chunks, stdout_error, "stdout"), daemon=True)
        err_thread = threading.Thread(target=guarded, args=(process.stderr, stderr_chunks, stderr_error, "stderr"), daemon=True)
        out_thread.start()
        err_thread.start()

        deadline = time.monotonic() + self._timeout_seconds
        interrupted: tuple[str, str] | None = None
        while process.poll() is None:
            if _is_cancelled(cancellation):
                interrupted = ("cancelled", "runtime cancellation was requested")
                break
            if lease_valid is not None and not lease_valid():
                interrupted = ("failed", "invocation lease is no longer valid")
                break
            if time.monotonic() >= deadline:
                interrupted = ("failed", "runtime child exceeded its invocation deadline")
                break
            time.sleep(0.01)

        if interrupted is not None:
            self._terminate(process)
        else:
            process.wait()
        out_thread.join(timeout=1.0)
        err_thread.join(timeout=1.0)
        if out_thread.is_alive() or err_thread.is_alive():
            self._terminate(process)
            out_thread.join(timeout=1.0)
            err_thread.join(timeout=1.0)
        process.stdout.close()
        process.stderr.close()
        if stdout_error:
            raise stdout_error[0]
        if stderr_error:
            raise stderr_error[0]
        if interrupted is not None:
            kind, message = interrupted
            return (_failed_frame(snapshot, invocation, kind=kind, code="cancelled" if kind == "cancelled" else "lease_expired" if "lease" in message else "runtime_error", message=message),)

        raw_output = b"".join(stdout_chunks)
        frames: list[dict[str, object]] = []
        for line in raw_output.splitlines():
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeTransportError("runtime child emitted malformed JSON") from exc
            if not isinstance(frame, dict):
                raise RuntimeTransportError("runtime child emitted a non-object frame")
            frames.append(frame)
        if process.returncode != 0:
            raise RuntimeTransportError("runtime child exited before producing valid runtime evidence")
        try:
            validate_g1_frames(frames, snapshot.to_dict(), invocation.to_dict())
        except G1ContractError as exc:
            raise RuntimeTransportError("runtime child emitted an invalid G1 frame sequence") from exc
        return tuple(json.dumps(frame, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for frame in frames)

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)


__all__ = ["RuntimeTransportError", "SubprocessRuntimeTransport"]
