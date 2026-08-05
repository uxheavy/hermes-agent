"""Trusted G1 container host bootstrap.

The host supervisor sends exactly one private credential-control frame followed
by the ordinary immutable G1 request.  This trusted bootstrap consumes the
control frame before starting the Hermes service, keeps credentials only in a
tmpfs-backed, mode-0600 broker, and authorizes exactly the spawned service PID
to read them.  It never forwards the control frame to Hermes or emits it.
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
_BROKER_PROTOCOL = "plane.agent-runtime/credentials/v1"
_BROKER_DIRECTORY = "/tmp/plane-agent-credential-broker"
_BROKER_PATH = f"{_BROKER_DIRECTORY}/broker.sock"
_MAX_CONTROL_BYTES = 16 * 1024


def _read_line(stream: Any, limit: int) -> bytes:
    raw = stream.readline(limit + 1)
    if not raw or len(raw) > limit or not raw.endswith(b"\n"):
        raise ValueError("credential bootstrap frame is malformed")
    return raw[:-1]


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
            or any(ord(char) < 0x20 or char in "\x7f" for char in key + item)
            or len(key) > 128
            or len(item.encode("utf-8")) > 16 * 1024
        ):
            raise ValueError("credential bootstrap payload is invalid")
        result[key] = item
    return result


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
                while len(raw) <= 4096 and not raw.endswith(b"\n"):
                    chunk = channel.recv(min(1024, 4097 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                response: dict[str, object] = {"protocol": _BROKER_PROTOCOL, "credentials": {}}
                try:
                    request = json.loads(bytes(raw).rstrip(b"\n"))
                    with self._lock:
                        if (
                            self._served
                            or peer_pid != self._expected_pid
                            or not isinstance(request, dict)
                            or request.get("protocol") != _BROKER_PROTOCOL
                            or request.get("provider") != self._expected_provider
                        ):
                            raise ValueError("credential broker request denied")
                        self._served = True
                        response["credentials"] = dict(self._credentials)
                except Exception:
                    pass
                try:
                    channel.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
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


def _run(request: bytes, credentials: Mapping[str, str]) -> int:
    provider = str(json.loads(request)["run"]["runtimePolicy"]["model"]["provider"])
    broker = _CredentialBroker(credentials, provider)
    child: subprocess.Popen[bytes] | None = None
    try:
        child = subprocess.Popen(
            ("python3", "-m", "plane_runtime.service", "--once", "--g1-production"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        broker.start(child.pid)
        assert child.stdin is not None and child.stdout is not None
        child.stdin.write(request)
        child.stdin.close()
        shutil.copyfileobj(child.stdout, sys.stdout.buffer, length=16 * 1024)
        child.stdout.close()
        return int(child.wait())
    finally:
        broker.close()
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
        control_raw = bytearray(_read_line(sys.stdin.buffer, _MAX_CONTROL_BYTES))
        try:
            control = json.loads(bytes(control_raw))
            if not isinstance(control, dict) or control.get("protocol") != _CONTROL_PROTOCOL:
                return 2
            credentials = _bounded_credentials(control.get("credentials"))
        finally:
            for index in range(len(control_raw)):
                control_raw[index] = 0
        request = _read_line(sys.stdin.buffer, 512 * 1024)
        if not request.startswith(b"{\"invocation\"") and not request.startswith(b'{"run"'):
            return 2
        # The host closes stdin after the exact two frames.  Any third frame,
        # including a duplicate/bootstrap frame after the request, is denied
        # before the Hermes service starts and never reaches its ledger.
        if sys.stdin.buffer.read(1):
            return 2
        try:
            return _run(request + b"\n", credentials)
        finally:
            credentials.clear()
    except Exception:
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
