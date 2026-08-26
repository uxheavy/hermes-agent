"""Trusted G1 container-host bootstrap.

Two private, canonical control frames are consumed before the ordinary G1
request: the model-call allowance and host credential broker material.  The
frames are never passed to Hermes or retained as evidence.  The bootstrap is
the trusted invocation host; generated code can reach credentials only via
the narrow Unix-socket adapter callback.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
from typing import Any

from ..g1_bootstrap_contract import G1BootstrapFrames, read_g1_bootstrap_frames
from ..hermes_adapter import validate_absolute_unix_socket_path

_MODEL_USAGE_PROTOCOL = "plane.agent-runtime/internal-usage/v1"
_MAX_DIAGNOSTIC_BYTES = 16 * 1024
_MAX_UNIX_SOCKET_PATH_BYTES = 103
_DIAGNOSTIC_RE = re.compile(r"event=agent\.runtime\.service status=failed exceptionClass=([A-Za-z_][A-Za-z0-9_]*) module=([A-Za-z_][A-Za-z0-9_.]*)\n")


def _write_failure_diagnostic(error: BaseException, source: str) -> None:
    exception_class = type(error).__name__
    module = type(error).__module__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", exception_class):
        exception_class = "Unknown"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
        module = "unknown"
    sys.stderr.write(
        f"event=agent.runtime.{source} status=failed exceptionClass={exception_class} module={module}\n"
    )


def _forward_child_diagnostic(raw: bytes, overflow: bool) -> None:
    if overflow:
        return
    try:
        match = _DIAGNOSTIC_RE.fullmatch(raw.decode("ascii"))
    except UnicodeDecodeError:
        return
    if match is not None:
        sys.stderr.write(raw.decode("ascii"))

def _plane_host_socket(value: object) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
        or len(value.encode("utf-8")) > _MAX_UNIX_SOCKET_PATH_BYTES
    ):
        raise ValueError("Plane host socket configuration is invalid")
    return value


def _provider_relay_socket(value: object) -> str | None:
    return validate_absolute_unix_socket_path(value)


def _read_child_diagnostics(stream: Any, result: bytearray, overflow: list[bool]) -> None:
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            if len(result) < _MAX_DIAGNOSTIC_BYTES:
                result.extend(chunk[: _MAX_DIAGNOSTIC_BYTES - len(result)])
            if len(result) >= _MAX_DIAGNOSTIC_BYTES or len(chunk) > _MAX_DIAGNOSTIC_BYTES:
                overflow[0] = True
    except Exception:
        overflow[0] = True


def _forward_valid_model_usage(raw: bytes, overflow: bool) -> None:
    if overflow or not raw.endswith(b"\n"):
        return
    try:
        value = json.loads(raw[:-1].decode("utf-8"))
        if not isinstance(value, dict) or set(value) != {"modelCalls", "protocol"}:
            return
        if json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") != raw[:-1]:
            return
        if value["protocol"] != _MODEL_USAGE_PROTOCOL or not isinstance(value["modelCalls"], int) or isinstance(value["modelCalls"], bool) or value["modelCalls"] < 0:
            return
        target = getattr(sys.stderr, "buffer", sys.stderr)
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        target.flush()
    except Exception:
        return


def _run(
    frames: G1BootstrapFrames,
    plane_host_socket: str | None = None,
    provider_relay_socket: str | None = None,
) -> int:
    plane_host_socket = _plane_host_socket(plane_host_socket)
    provider_relay_socket = _provider_relay_socket(provider_relay_socket)
    child: subprocess.Popen[bytes] | None = None
    diagnostics = bytearray()
    overflow = [False]
    stderr_thread: threading.Thread | None = None
    cancelled = threading.Event()
    previous_handler: Any = None
    signal_installed = False

    def forward_cancellation(_signum: int, _frame: Any) -> None:
        cancelled.set()
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signal.SIGUSR1)
            except OSError:
                pass

    try:
        previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, forward_cancellation)
        signal_installed = True
        child_command = [
            "python3",
            "-m",
            "plane_runtime.service",
            "--once",
            "--g1-production",
            "--g1-bootstrap-child",
        ]
        if plane_host_socket is not None:
            child_command.extend(("--plane-host-socket", plane_host_socket))
        if provider_relay_socket is not None:
            child_command.extend(("--provider-relay-socket", provider_relay_socket))
        child = subprocess.Popen(
            tuple(child_command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        if cancelled.is_set():
            forward_cancellation(signal.SIGUSR1, None)
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        stderr_thread = threading.Thread(target=_read_child_diagnostics, args=(child.stderr, diagnostics, overflow), daemon=True)
        stderr_thread.start()
        child_input = frames.child_bytes()
        try:
            child.stdin.write(bytes(child_input))
        finally:
            for index in range(len(child_input)):
                child_input[index] = 0
        child.stdin.close()
        shutil.copyfileobj(child.stdout, sys.stdout.buffer, length=16 * 1024)
        child.stdout.close()
        returncode = int(child.wait())
        stderr_thread.join(timeout=1.0)
        _forward_child_diagnostic(bytes(diagnostics), overflow[0])
        _forward_valid_model_usage(bytes(diagnostics), overflow[0])
        return returncode
    finally:
        if signal_installed:
            signal.signal(signal.SIGUSR1, previous_handler)
        frames.clear()
        for index in range(len(diagnostics)):
            diagnostics[index] = 0
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--g1-production", action="store_true")
    parser.add_argument("--plane-host-socket", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--provider-relay-socket", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.once or not args.g1_production:
        return 2
    frames: G1BootstrapFrames | None = None
    try:
        frames = read_g1_bootstrap_frames(sys.stdin)
        return _run(
            frames,
            _plane_host_socket(args.plane_host_socket),
            _provider_relay_socket(args.provider_relay_socket),
        )
    except Exception as error:
        if frames is not None:
            frames.clear()
        _write_failure_diagnostic(error, "bootstrap")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
