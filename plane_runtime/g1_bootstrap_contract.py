"""Private parent-to-child contract for the production G1 bootstrap.

The public runtime wire contains only ``plane.agent-runtime/v1`` request and
event frames.  This separate, trusted channel carries the host-only dispatch
allowance and credentials to the one-shot child.  It is deliberately small,
strictly ordered, and never handed to Hermes as prompt or tool data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Mapping


CONTROL_PROTOCOL = "plane.agent-runtime/credential-control/v1"
DISPATCH_PROTOCOL = "plane.agent-runtime/dispatch-control/v1"
MAX_CONTROL_BYTES = 16 * 1024
MAX_DISPATCH_BYTES = 4096
MAX_REQUEST_BYTES = 512 * 1024
MAX_CREDENTIALS = 16
MAX_CREDENTIAL_KEY_BYTES = 128
MAX_CREDENTIAL_VALUE_BYTES = 16 * 1024


def _canonical_object(raw: bytes, expected: set[str], name: str, *, limit: int) -> dict[str, Any]:
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


def _read_line(stream: BinaryIO, limit: int) -> bytes:
    raw = stream.readline(limit + 1)
    if not isinstance(raw, bytes) or not raw or len(raw) > limit + 1 or not raw.endswith(b"\n"):
        raise ValueError("bootstrap frame is malformed")
    frame = raw[:-1]
    if len(frame) > limit:
        raise ValueError("bootstrap frame is oversized")
    return frame


def _credentials(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > MAX_CREDENTIALS:
        raise ValueError("credential bootstrap payload is invalid")
    result: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, str)
            or not key
            or not item
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in key + item)
            or len(key.encode("utf-8")) > MAX_CREDENTIAL_KEY_BYTES
            or len(item.encode("utf-8")) > MAX_CREDENTIAL_VALUE_BYTES
        ):
            raise ValueError("credential bootstrap payload is invalid")
        result[key] = item
    return result


def _allowance(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4096:
        raise ValueError("model-call allowance is invalid")
    return value


def _frame(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


@dataclass
class G1BootstrapFrames:
    """Validated private controls and the immutable request bytes.

    ``credentials`` is mutable by design so the caller can clear it after the
    child has received the frame.  No object in this dataclass is model input.
    """

    model_call_allowance: int
    credentials: dict[str, str] = field(repr=False)
    request: bytes

    def child_bytes(self) -> bytearray:
        """Serialize the exact three-frame child input in contract order."""

        control = bytearray(
            _frame(
                {
                    "modelCallAllowance": self.model_call_allowance,
                    "protocol": DISPATCH_PROTOCOL,
                }
            )
        )
        control.extend(
            _frame(
                {
                    "credentials": self.credentials,
                    "protocol": CONTROL_PROTOCOL,
                }
            )
        )
        control.extend(self.request)
        control.extend(b"\n")
        return control

    def clear(self) -> None:
        self.credentials.clear()


def read_g1_bootstrap_frames(stream: Any) -> G1BootstrapFrames:
    """Read dispatch, credential, request, then require EOF.

    The caller must pass a binary stream or a text stream exposing ``buffer``.
    No error includes parsed credential values.
    """

    source = getattr(stream, "buffer", stream)
    dispatch = _canonical_object(
        _read_line(source, MAX_DISPATCH_BYTES),
        {"modelCallAllowance", "protocol"},
        "dispatch control",
        limit=MAX_DISPATCH_BYTES,
    )
    if dispatch["protocol"] != DISPATCH_PROTOCOL:
        raise ValueError("dispatch control protocol is invalid")
    allowance = _allowance(dispatch["modelCallAllowance"])

    control = _canonical_object(
        _read_line(source, MAX_CONTROL_BYTES),
        {"credentials", "protocol"},
        "credential control",
        limit=MAX_CONTROL_BYTES,
    )
    if control["protocol"] != CONTROL_PROTOCOL:
        raise ValueError("credential control protocol is invalid")
    credentials = _credentials(control["credentials"])

    request = _read_line(source, MAX_REQUEST_BYTES)
    _canonical_object(request, {"invocation", "run"}, "G1 request", limit=MAX_REQUEST_BYTES)
    if source.read(1):
        raise ValueError("bootstrap input contains trailing frames")
    return G1BootstrapFrames(allowance, credentials, request)


__all__ = [
    "CONTROL_PROTOCOL",
    "DISPATCH_PROTOCOL",
    "G1BootstrapFrames",
    "MAX_CONTROL_BYTES",
    "MAX_DISPATCH_BYTES",
    "MAX_REQUEST_BYTES",
    "read_g1_bootstrap_frames",
]
