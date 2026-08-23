"""Run one trusted Plane Agent G1 bootstrap-child invocation."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from typing import Any, Callable

from .host_port import UnixSocketPlaneHostPort
from .g1_bootstrap_contract import read_g1_bootstrap_frames
from .g1_contract import G1RunSnapshot


class _BootstrapCancellation:
    """Trusted parent signal state for one production child invocation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._previous: Any = None
        self._installed = False

    def _handler(self, _signum: int, _frame: Any) -> None:
        self._event.set()

    def __enter__(self) -> Callable[[], bool]:
        if not hasattr(signal, "SIGUSR1"):
            raise RuntimeError("trusted cancellation signal is unavailable")
        self._previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handler)
        self._installed = True
        return self._event.is_set

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._installed:
            signal.signal(signal.SIGUSR1, self._previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Plane Agent runtime invocation")
    parser.add_argument("--once", action="store_true", help="accept one JSON-lines invocation (the default)")
    parser.add_argument("--g1-production", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--g1-bootstrap-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-call-allowance", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plane-host-socket", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--provider-relay-socket", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Production is authoritative only through the trusted bootstrap.  The
    # marker is private parent-to-child wiring, not a second public entrypoint.
    if args.g1_production != args.g1_bootstrap_child:
        return 2
    if args.g1_bootstrap_child:
        frames = None
        host_port = None
        try:
            from .g1_service import serve_once_g1
            from .hermes_adapter import (
                InlineCredentialSource,
                prepare_provider_relay_credentials,
                validate_absolute_unix_socket_path,
            )

            with _BootstrapCancellation() as cancellation:
                frames = read_g1_bootstrap_frames(sys.stdin)
                request_line = frames.request.decode("utf-8")
                request = json.loads(request_line)
                snapshot = G1RunSnapshot.from_dict(request["run"])
                host_port = (
                    UnixSocketPlaneHostPort(args.plane_host_socket)
                    if args.plane_host_socket is not None
                    else None
                )
                provider_relay_socket = validate_absolute_unix_socket_path(
                    args.provider_relay_socket
                )
                source_credentials, http_client_factory = prepare_provider_relay_credentials(
                    frames.credentials,
                    expected_provider=snapshot.model_provider,
                    provider_relay_socket=provider_relay_socket,
                )
                service_kwargs: dict[str, object] = {
                    "production": True,
                    "diagnostics": sys.stderr,
                    "model_call_allowance": frames.model_call_allowance,
                    "host_port": host_port,
                    "credential_source": InlineCredentialSource(
                        source_credentials, snapshot.model_provider
                    ),
                    "cancellation": cancellation,
                }
                if http_client_factory is not None:
                    service_kwargs["http_client_factory"] = http_client_factory
                return serve_once_g1(
                    request_line,
                    sys.stdout,
                    **service_kwargs,
                )
        except Exception:
            return 2
        finally:
            if host_port is not None:
                host_port.close()
            if frames is not None:
                frames.clear()
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
