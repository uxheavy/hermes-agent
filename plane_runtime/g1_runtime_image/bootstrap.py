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
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
from typing import Any, Mapping

_CONTROL_PROTOCOL = "plane.agent-runtime/credential-control/v1"
_DISPATCH_PROTOCOL = "plane.agent-runtime/dispatch-control/v1"
_BROKER_PROTOCOL = "plane.agent-runtime/credentials/v1"
_MODEL_USAGE_PROTOCOL = "plane.agent-runtime/internal-usage/v1"
_BROKER_DIRECTORY = "/tmp/plane-agent-credential-broker"
_BROKER_PATH = f"{_BROKER_DIRECTORY}/broker.sock"
_MAX_CONTROL_BYTES = 16 * 1024
_MAX_DISPATCH_BYTES = 4096
_MAX_BROKER_BYTES = 4096
_MAX_DIAGNOSTIC_BYTES = 16 * 1024


def _read_line(stream: Any, limit: int) -> bytes:
    raw = stream.readline(limit + 1)
    if not raw or len(raw) > limit or not raw.endswith(b"\n"):
        raise ValueError("bootstrap frame is malformed")
    return raw[:-1]


def _strict_object(raw: bytes, expected: set[str], name: str, *, limit: int) -> dict[str, Any]:
    if len(raw) > limit or raw != raw.strip():
        raise ValueError(f"{name} is not canonical")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is malformed") from exc
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has an invalid key set")
    if json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") != raw:
        raise ValueError(f"{name} is not canonical")
    return value


def _bounded_credentials(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > 16:
        raise ValueError("credential bootstrap payload is invalid")
    result: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, str)
            or not key
            or not item
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in key + item)
            or len(key.encode("utf-8")) > 128
            or len(item.encode("utf-8")) > 16 * 1024
        ):
            raise ValueError("credential bootstrap payload is invalid")
        result[key] = item
    return result


def _dispatch_allowance(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4096:
        raise ValueError("model-call allowance is invalid")
    return value


class _CredentialBroker:
    """One-request broker reachable only by the spawned Hermes service PID."""

    def __init__(self, credentials: Mapping[str, str], expected_provider: str) -> None:
        self._credentials = dict(credentials)
        self._expected_provider = expected_provider
        self._expected_pid: int | None = None
        self._served = False
        self._lock = threading.Lock()
        os.makedirs(_BROKER_DIRECTORY, mode=0o700, exist_ok=True)
        os.chmod(_BROKER_DIRECTORY, 0o700)
        try:
            os.unlink(_BROKER_PATH)
        except FileNotFoundError:
            pass
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(_BROKER_PATH)
        os.chmod(_BROKER_PATH, 0o600)
        self._server.listen(1)
        self._server.settimeout(0.2)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="g1-credential-broker", daemon=True)

    def start(self, expected_pid: int) -> None:
        self._expected_pid = expected_pid
        self._thread.start()

    def _peer_pid(self, channel: socket.socket) -> int | None:
        try:
            raw = channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            return int(struct.unpack("3i", raw)[0])
        except (AttributeError, OSError, struct.error):
            return None

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                channel, _address = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with channel:
                channel.settimeout(2.0)
                peer_pid = self._peer_pid(channel)
                raw = bytearray()
                try:
                    while len(raw) <= _MAX_BROKER_BYTES and not raw.endswith(b"\n"):
                        chunk = channel.recv(min(1024, _MAX_BROKER_BYTES + 1 - len(raw)))
                        if not chunk:
                            break
                        raw.extend(chunk)
                except (OSError, socket.timeout):
                    raw.clear()
                response: dict[str, object] = {"credentials": {}, "protocol": _BROKER_PROTOCOL}
                try:
                    if len(raw) > _MAX_BROKER_BYTES or not raw.endswith(b"\n"):
                        raise ValueError("broker request is oversized")
                    request = _strict_object(bytes(raw[:-1]), {"provider", "protocol"}, "broker request", limit=_MAX_BROKER_BYTES)
                    with self._lock:
                        if (
                            self._served
                            or peer_pid != self._expected_pid
                            or request["protocol"] != _BROKER_PROTOCOL
                            or not isinstance(request["provider"], str)
                            or request["provider"] != self._expected_provider
                        ):
                            raise ValueError("credential broker request denied")
                        self._served = True
                        response["credentials"] = dict(self._credentials)
                except Exception:
                    pass
                try:
                    channel.sendall(json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
                except OSError:
                    pass
                finally:
                    for index in range(len(raw)):
                        raw[index] = 0

    def close(self) -> None:
        self._closed.set()
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)
        self._credentials.clear()
        try:
            os.unlink(_BROKER_PATH)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(_BROKER_DIRECTORY)
        except OSError:
            pass


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
        value = _strict_object(raw[:-1], {"modelCalls", "protocol"}, "internal usage", limit=_MAX_DIAGNOSTIC_BYTES)
        if value["protocol"] != _MODEL_USAGE_PROTOCOL or not isinstance(value["modelCalls"], int) or isinstance(value["modelCalls"], bool) or value["modelCalls"] < 0:
            return
        target = getattr(sys.stderr, "buffer", sys.stderr)
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        target.flush()
    except Exception:
        return


def _run(request: bytes, credentials: Mapping[str, str], model_call_allowance: int) -> int:
    request_value = _strict_object(request.rstrip(b"\n"), {"invocation", "run"}, "G1 request", limit=512 * 1024)
    provider = request_value["run"]["runtimePolicy"]["model"]["provider"]
    if not isinstance(provider, str):
        raise ValueError("G1 provider is invalid")
    broker = _CredentialBroker(credentials, provider)
    child: subprocess.Popen[bytes] | None = None
    diagnostics = bytearray()
    overflow = [False]
    stderr_thread: threading.Thread | None = None
    try:
        child = subprocess.Popen(
            ("python3", "-m", "plane_runtime.service", "--once", "--g1-production", "--model-call-allowance", str(model_call_allowance)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        broker.start(child.pid)
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        stderr_thread = threading.Thread(target=_read_child_diagnostics, args=(child.stderr, diagnostics, overflow), daemon=True)
        stderr_thread.start()
        child.stdin.write(request)
        child.stdin.close()
        shutil.copyfileobj(child.stdout, sys.stdout.buffer, length=16 * 1024)
        child.stdout.close()
        returncode = int(child.wait())
        stderr_thread.join(timeout=1.0)
        _forward_valid_model_usage(bytes(diagnostics), overflow[0])
        return returncode
    finally:
        broker.close()
        for index in range(len(diagnostics)):
            diagnostics[index] = 0
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--g1-production", action="store_true")
    args = parser.parse_args(argv)
    if not args.once or not args.g1_production:
        return 2
    try:
        dispatch_raw = bytearray(_read_line(sys.stdin.buffer, _MAX_DISPATCH_BYTES))
        try:
            dispatch = _strict_object(bytes(dispatch_raw), {"modelCallAllowance", "protocol"}, "dispatch control", limit=_MAX_DISPATCH_BYTES)
            if dispatch["protocol"] != _DISPATCH_PROTOCOL:
                return 2
            allowance = _dispatch_allowance(dispatch["modelCallAllowance"])
        finally:
            for index in range(len(dispatch_raw)):
                dispatch_raw[index] = 0
        control_raw = bytearray(_read_line(sys.stdin.buffer, _MAX_CONTROL_BYTES))
        try:
            control = _strict_object(bytes(control_raw), {"credentials", "protocol"}, "credential control", limit=_MAX_CONTROL_BYTES)
            if control["protocol"] != _CONTROL_PROTOCOL:
                return 2
            credentials = _bounded_credentials(control["credentials"])
        finally:
            for index in range(len(control_raw)):
                control_raw[index] = 0
        request = _read_line(sys.stdin.buffer, 512 * 1024)
        # The bootstrap validates the frame envelope and exact canonical form
        # before a child exists.  Full schema validation remains in service.
        _strict_object(request, {"invocation", "run"}, "G1 request", limit=512 * 1024)
        if sys.stdin.buffer.read(1):
            return 2
        try:
            return _run(request + b"\n", credentials, allowance)
        finally:
            credentials.clear()
    except Exception:
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
