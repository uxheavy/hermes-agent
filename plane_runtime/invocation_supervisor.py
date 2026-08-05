"""Fail-closed supervision for one disposable Plane runtime invocation.

The child process is an evidence producer, never a Plane authority.  It emits
bounded runtime events, one terminal proposal, and one exit.  This module runs
in the trusted host and is the only place that may submit the proposal through
the injected :class:`TerminalReconciliationPort`.

The Docker implementation is deliberately strict and conservative.  It uses
only fixed argv controls and performs bounded Docker JSON inspection before
and after launch.  This package does not yet contain a trusted production
entrypoint: Docker evidence and injected runners are explicitly test-only
until a real kernel/service binding is installed.  A production-shaped
attestation is rejected rather than treated as proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import secrets
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from .adapter import (
    CanonicalLeaseAuthority,
    CanonicalLeaseBinding,
    CancellationAuthority,
    ChildCancellationProposalRejected,
    CancellationSignal,
    EventCollector,
    TerminalReconciliationPort,
    build_host_cancellation_proposal,
    classify_process_death,
    reconcile_process_death,
    reconcile_terminal_proposal,
    validate_terminal_proposal,
)
from .contract import (
    BindingError,
    BoundsError,
    ContractError,
    InvocationEnvelope,
    MAX_EVENT_BYTES,
    MAX_INVOCATION_BYTES,
    MAX_REFERENCE_LENGTH,
    MAX_RUN_SNAPSHOT_BYTES,
    MAX_TERMINAL_PROPOSAL_BYTES,
    RuntimeExit,
    RuntimeConfigurationError,
    RunSnapshot,
    TerminalProposal,
    TerminalReconciliationReceipt,
    canonical_json_bytes,
)


# These aliases are used only while validating a freshly produced candidate or
# a death receipt before it crosses the retention seam.  Canonical readback
# never reconstructs contract objects through these replaceable module names;
# it returns immutable value views from serialized bytes instead.
_EXACT_RUNTIME_EXIT = RuntimeExit
_EXACT_TERMINAL_PROPOSAL = TerminalProposal
_EXACT_TERMINAL_RECEIPT = TerminalReconciliationReceipt


# Docker policy is intentionally module-private.  InvocationPolicy contains
# only bounded scalars and the immutable image reference; callers cannot pass
# network, namespace, entrypoint, mount, device, logging, or persistence
# controls into the command builder.
_FIXED_ENTRYPOINT = "python3"
_FIXED_SERVICE_MODULE = "plane_runtime.service"
_FIXED_SERVICE_ARGS = ("--once",)
_FIXED_NETWORK = "none"
_FIXED_USER = "65532:65532"
_FIXED_TMPFS_TARGET = "/tmp"
_FIXED_TMPFS_OPTIONS = "rw,noexec,nosuid,nodev"
_FIXED_PULL_POLICY = "never"
_FIXED_LOG_DRIVER = "none"
_FIXED_CONTAINER_PREFIX = "plane-invocation"
_FIXED_PROTOCOL_LABEL = "plane.agent-runtime/protocol=plane.agent-runtime/v1"
_FIXED_ALLOWED_ENV: tuple[tuple[str, str], ...] = (
    ("LANG", "C.UTF-8"),
    ("LC_ALL", "C.UTF-8"),
    ("PATH", "/usr/local/bin:/usr/bin:/bin"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("PYTHONUNBUFFERED", "1"),
)
_DOCKER_CLIENT_ENV: Mapping[str, str] = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
}
_ALLOWED_IMAGE_ENV_KEYS = frozenset(key for key, _ in _FIXED_ALLOWED_ENV)
_SUPPORTED_STORAGE_DRIVERS = frozenset({"overlay2", "btrfs", "zfs", "devicemapper"})
_IMAGE_DIGEST = re.compile(r"[a-z0-9][a-z0-9._:/-]*[a-z0-9]@sha256:[0-9a-f]{64}")
_REGISTRY_PORT = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?:[0-9]{1,5}")
_IMAGE_COMPONENT = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CONTAINER_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_HEX_ID = re.compile(r"[0-9a-f]{12,64}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_DEATH_REASON = "runtime process exited before terminal evidence"
_SAFE_DEATH_REASONS = frozenset(
    {
        _SAFE_DEATH_REASON,
        "runtime output was malformed or lacked terminal evidence",
        "runtime output exceeded supervisor bounds",
        "runtime invocation exceeded its wall-time bound",
        "runtime invocation was stopped by a cancellation signal",
        "runtime process died before terminal evidence",
        "runtime launch failed before terminal evidence",
    }
)
_MAX_RETAINED_INVOCATIONS = 1024
_MAX_DOCKER_INSPECTION_BYTES = 128 * 1024
_ATTACH_TERMINATE_TIMEOUT_SECONDS = 0.5
_ATTACH_KILL_TIMEOUT_SECONDS = 0.5
_POLICY_FINGERPRINTS: dict[int, tuple[object, ...]] = {}


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ContractError(f"{name} must be a positive finite number")
    return number


def _validate_image_reference(image: object) -> str:
    if not isinstance(image, str) or not image or _CONTROL.search(image) or any(
        character.isspace() for character in image
    ):
        raise ContractError("invocation image must be a canonical digest reference")
    if len(image) > MAX_REFERENCE_LENGTH:
        raise ContractError("invocation image reference is too long")
    if image.startswith("-") or image.count("@") != 1 or "//" in image or "::" in image:
        raise ContractError("invocation image must be a canonical digest reference")
    name, digest = image.rsplit("@", 1)
    if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
        raise ContractError("invocation image must use a lowercase sha256 digest")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ContractError("invocation image must use a lowercase sha256 digest")
    if not _IMAGE_DIGEST.fullmatch(image) or ".." in name:
        raise ContractError("invocation image must be a canonical digest reference")
    components = name.split("/")
    if any(not component for component in components):
        raise ContractError("invocation image contains an empty name component")
    for index, component in enumerate(components):
        if ":" in component:
            if index != 0 or not _REGISTRY_PORT.fullmatch(component):
                raise ContractError("invocation image contains an invalid registry component")
        elif not _IMAGE_COMPONENT.fullmatch(component):
            raise ContractError("invocation image contains an invalid name component")
    return image


def _policy_fingerprint(policy: "InvocationPolicy") -> tuple[object, ...]:
    return (
        policy.image,
        policy.cpu_millicores,
        policy.memory_bytes,
        policy.pids_limit,
        policy.wall_time_seconds,
        policy.stdout_limit_bytes,
        policy.stderr_limit_bytes,
        policy.frame_limit_bytes,
        policy.request_limit_bytes,
        policy.max_output_frames,
        policy.tmpfs_bytes,
        policy.storage_limit_bytes,
        policy.stop_timeout_seconds,
        policy.kill_timeout_seconds,
        policy.remove_timeout_seconds,
    )


def _validate_policy_values(policy: "InvocationPolicy") -> tuple[object, ...]:
    _validate_image_reference(policy.image)
    cpu = _positive_int(policy.cpu_millicores, "cpu_millicores")
    if cpu > 4000:
        raise ContractError("invocation CPU bound is outside the permitted range")
    memory = _positive_int(policy.memory_bytes, "memory_bytes", minimum=16 * 1024 * 1024)
    if memory > 4 * 1024 * 1024 * 1024:
        raise ContractError("invocation memory bound is outside the permitted range")
    pids = _positive_int(policy.pids_limit, "pids_limit", minimum=4)
    if pids > 4096:
        raise ContractError("invocation PID bound is outside the permitted range")
    wall = _positive_float(policy.wall_time_seconds, "wall_time_seconds")
    if wall > 3600:
        raise ContractError("invocation wall-time bound is outside the permitted range")
    stdout_limit = _positive_int(policy.stdout_limit_bytes, "stdout_limit_bytes")
    stderr_limit = _positive_int(policy.stderr_limit_bytes, "stderr_limit_bytes")
    if stdout_limit > 16 * 1024 * 1024 or stderr_limit > 16 * 1024 * 1024:
        raise ContractError("invocation output bound is outside the permitted range")
    frame_limit = _positive_int(policy.frame_limit_bytes, "frame_limit_bytes")
    if frame_limit < MAX_EVENT_BYTES or frame_limit > MAX_TERMINAL_PROPOSAL_BYTES:
        raise ContractError("frame_limit_bytes is outside the permitted range")
    request_limit = _positive_int(policy.request_limit_bytes, "request_limit_bytes")
    if request_limit < MAX_RUN_SNAPSHOT_BYTES + MAX_INVOCATION_BYTES or request_limit > 512 * 1024:
        raise ContractError("request_limit_bytes is outside the permitted range")
    frames = _positive_int(policy.max_output_frames, "max_output_frames", minimum=2)
    if frames > 4096:
        raise ContractError("invocation frame-count bound is outside the permitted range")
    tmpfs = _positive_int(policy.tmpfs_bytes, "tmpfs_bytes", minimum=1024 * 1024)
    storage = _positive_int(policy.storage_limit_bytes, "storage_limit_bytes", minimum=16 * 1024 * 1024)
    if tmpfs > 1024 * 1024 * 1024 or storage > 16 * 1024 * 1024 * 1024:
        raise ContractError("invocation storage bound is outside the permitted range")
    cleanup_values: list[float] = []
    for name, value in (
        ("stop_timeout_seconds", policy.stop_timeout_seconds),
        ("kill_timeout_seconds", policy.kill_timeout_seconds),
        ("remove_timeout_seconds", policy.remove_timeout_seconds),
    ):
        timeout = _positive_float(value, name)
        if timeout > 60:
            raise ContractError("invocation cleanup deadline is outside the permitted range")
        cleanup_values.append(timeout)
    return (
        policy.image,
        cpu,
        memory,
        pids,
        wall,
        stdout_limit,
        stderr_limit,
        frame_limit,
        request_limit,
        frames,
        tmpfs,
        storage,
        *cleanup_values,
    )


def _validate_policy(policy: "InvocationPolicy") -> tuple[object, ...]:
    if not isinstance(policy, InvocationPolicy):
        raise ContractError("invocation policy is invalid")
    values = _validate_policy_values(policy)
    if _POLICY_FINGERPRINTS.get(id(policy)) != _policy_fingerprint(policy):
        raise ContractError("invocation policy was mutated after construction")
    return values


@dataclass(frozen=True)
class InvocationPolicy:
    """Validated image and resource bounds for one launch.

    Isolation controls are intentionally absent from this type.  They are
    fixed module-private policy and are copied into argv by this module.
    """

    image: str
    cpu_millicores: int = 500
    memory_bytes: int = 256 * 1024 * 1024
    pids_limit: int = 64
    wall_time_seconds: float = 120.0
    stdout_limit_bytes: int = 256 * 1024
    stderr_limit_bytes: int = 64 * 1024
    frame_limit_bytes: int = MAX_TERMINAL_PROPOSAL_BYTES
    request_limit_bytes: int = 192 * 1024
    max_output_frames: int = 514
    tmpfs_bytes: int = 16 * 1024 * 1024
    storage_limit_bytes: int = 512 * 1024 * 1024
    stop_timeout_seconds: float = 2.0
    kill_timeout_seconds: float = 2.0
    remove_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        _validate_policy_values(self)
        _POLICY_FINGERPRINTS[id(self)] = _policy_fingerprint(self)


@dataclass(frozen=True)
class DockerRunnerCapabilities:
    """Deprecated test metadata; never used as enforcement authority."""

    immutable_digest: bool = False
    no_pull: bool = False
    network_none: bool = False
    read_only_rootfs: bool = False
    no_new_privileges: bool = False
    cap_drop_all: bool = False
    non_root_user: bool = False
    bounded_tmpfs: bool = False
    cpu_limit: bool = False
    memory_limit: bool = False
    pid_limit: bool = False
    bounded_output: bool = False
    bounded_wall_time: bool = False
    stop: bool = False
    kill: bool = False
    remove: bool = False

    @classmethod
    def fully_supported(cls) -> "DockerRunnerCapabilities":
        return cls(*(True for _ in cls.__dataclass_fields__))

    def missing(self) -> tuple[str, ...]:
        values = (
            ("immutable_digest", self.immutable_digest),
            ("no_pull", self.no_pull),
            ("network_none", self.network_none),
            ("read_only_rootfs", self.read_only_rootfs),
            ("no_new_privileges", self.no_new_privileges),
            ("cap_drop_all", self.cap_drop_all),
            ("non_root_user", self.non_root_user),
            ("bounded_tmpfs", self.bounded_tmpfs),
            ("cpu_limit", self.cpu_limit),
            ("memory_limit", self.memory_limit),
            ("pid_limit", self.pid_limit),
            ("bounded_output", self.bounded_output),
            ("bounded_wall_time", self.bounded_wall_time),
            ("stop", self.stop),
            ("kill", self.kill),
            ("remove", self.remove),
        )
        return tuple(
            name
            for name, supported in values
            if not supported
        )


@dataclass(frozen=True)
class EnforcementAttestation:
    """Bounded runner evidence; the current supervisor accepts test evidence only.

    ``production`` remains a parseable legacy value so hostile or stale
    adapters fail at the supervisor seam instead of failing during decoding.
    It is never an accepted classification for a current invocation.
    """

    classification: str
    argv_digest: str
    container_name: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.classification not in {"production", "test"}:
            raise ContractError("invalid runner trust classification")
        if not re.fullmatch(r"[0-9a-f]{64}", self.argv_digest):
            raise ContractError("runner attestation has an invalid argv digest")
        if not _CONTAINER_NAME.fullmatch(self.container_name):
            raise ContractError("runner attestation has an invalid container name")
        if len(self.evidence) > 16 or not self.evidence or any(
            not isinstance(item, str)
            or not item
            or len(item) > MAX_REFERENCE_LENGTH
            or _CONTROL.search(item)
            for item in self.evidence
        ):
            raise ContractError("runner attestation requires bounded evidence")

    @property
    def is_production(self) -> bool:
        return self.classification == "production"


_EXACT_ENFORCEMENT_ATTESTATION = EnforcementAttestation


@dataclass(frozen=True)
class ProcessCapture:
    """Bounded process output returned by a runner."""

    returncode: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    cancelled: bool = False
    output_exceeded: bool = False
    reaped: bool = True


class InvocationProcess(Protocol):
    def collect(
        self,
        *,
        input_bytes: bytes,
        deadline: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        is_cancelled: Callable[[], bool],
    ) -> ProcessCapture:
        ...


class DockerRunner(Protocol):
    def attest_invocation(
        self,
        argv: Sequence[str],
        *,
        client_env: Mapping[str, str],
    ) -> EnforcementAttestation:
        ...

    def launch(
        self,
        argv: Sequence[str],
        *,
        client_env: Mapping[str, str],
        input_bytes: bytes,
    ) -> InvocationProcess:
        ...

    def cleanup(
        self,
        container_name: str,
        *,
        stop_timeout_seconds: float,
        kill_timeout_seconds: float,
        remove_timeout_seconds: float,
    ) -> "CleanupReport":
        ...


@dataclass(frozen=True)
class CleanupReport:
    """Bounded cleanup evidence; absence must be proven after removal."""

    container_name: str
    stop_attempted: bool
    kill_attempted: bool
    remove_attempted: bool
    failures: tuple[str, ...] = ()
    post_cleanup_absent: bool = False

    @property
    def succeeded(self) -> bool:
        return self.post_cleanup_absent and not self.failures


_EXACT_CLEANUP_REPORT = CleanupReport


@dataclass(frozen=True)
class InvocationResult:
    """Supervisor result; child exit and proposal are never authority."""

    status: str
    container_name: str
    exit: RuntimeExit | None = None
    proposal: TerminalProposal | None = None
    receipt: TerminalReconciliationReceipt | None = None
    cleanup: CleanupReport | None = None
    enforcement: EnforcementAttestation | None = None
    evidence: tuple[str, ...] = ()

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def production_completed(self) -> bool:
        """Whether this result is a trusted production completion.

        The L9 foundation has no trusted production entrypoint yet.  Keeping
        this compatibility property permanently false prevents a caller-owned
        result or attestation from becoming a product completion claim.
        """

        return False


_EXACT_INVOCATION_RESULT = InvocationResult


_RESULT_STATUSES = frozenset(
    {
        "completed",
        "waiting_for_input",
        "failed",
        "blocked",
        "cancelled",
        "rejected",
        "supervisor_action_required",
    }
)
_TERMINAL_RESULT_STATUSES = frozenset(
    {"completed", "waiting_for_input", "failed", "blocked", "cancelled"}
)


def _draft_result(*args: object, **kwargs: object) -> tuple[object, ...]:
    """Build a transient tuple; no public result constructor is consulted.

    This value exists only until :func:`_canonicalize_result` serializes it.
    It deliberately has no constructor/type/default capture and is never
    retained or returned to a caller.
    """

    if len(args) > 2:
        raise TypeError("result draft accepts at most status and container name positionally")
    names = (
        "status",
        "container_name",
        "exit",
        "proposal",
        "receipt",
        "cleanup",
        "enforcement",
        "evidence",
    )
    values: list[object] = [None] * len(names)
    values[0] = args[0] if args else kwargs.pop("status")
    values[1] = args[1] if len(args) == 2 else kwargs.pop("container_name")
    for index, name in enumerate(names[2:], start=2):
        if name in kwargs:
            values[index] = kwargs.pop(name)
    if kwargs:
        raise TypeError(f"unknown result draft fields: {tuple(kwargs)}")
    return tuple(values)


def _canonical_contract_payload(
    value: object,
    *,
    expected_type: type,
    label: str,
) -> bytes:
    if type(value) is not expected_type:
        raise ContractError(f"{label} is not an exact contract value")
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise ContractError(f"{label} cannot be canonicalized")
    try:
        return canonical_json_bytes(to_dict())
    except Exception as exc:
        raise ContractError(f"{label} is malformed") from exc


def _parse_canonical_contract_payload(
    payload: bytes,
    *,
    expected_type: type,
    label: str,
) -> object:
    if type(payload) is not bytes:
        raise ContractError(f"{label} canonical payload has an invalid type")
    try:
        raw = json.loads(payload)
        value = expected_type.from_dict(raw)
    except Exception as exc:
        raise ContractError(f"{label} canonical payload is malformed") from exc
    if type(value) is not expected_type:
        raise ContractError(f"{label} canonical payload did not reconstruct exactly")
    if canonical_json_bytes(value.to_dict()) != payload:
        raise ContractError(f"{label} canonical payload is not stable")
    return value


def _canonicalize_result(
    key: tuple[str, str],
    result: object,
) -> bytes:
    """Validate a transient tuple and serialize the only retained form."""

    if type(result) is not tuple or len(result) != 8:
        raise ContractError("retained result is not an exact transient tuple")
    run_id, invocation_id = key
    status, container_name, exit_value, proposal_value, receipt_value, cleanup_value, enforcement_value, evidence = result
    if (
        type(run_id) is not str
        or type(invocation_id) is not str
        or type(status) is not str
        or status not in _RESULT_STATUSES
        or type(container_name) is not str
        or not _CONTAINER_NAME.fullmatch(container_name)
    ):
        raise ContractError("retained result has an invalid status or binding")

    exit_payload = (
        None
        if exit_value is None
        else _canonical_contract_payload(
            exit_value,
            expected_type=_EXACT_RUNTIME_EXIT,
            label="retained result exit",
        )
    )
    proposal_payload = (
        None
        if proposal_value is None
        else _canonical_contract_payload(
            proposal_value,
            expected_type=_EXACT_TERMINAL_PROPOSAL,
            label="retained result proposal",
        )
    )
    if proposal_value is not None and (
        proposal_value.run_id != run_id or proposal_value.invocation_id != invocation_id
    ):
        raise BindingError("retained proposal is bound to a different invocation")
    receipt_payload = (
        None
        if receipt_value is None
        else _canonical_contract_payload(
            receipt_value,
            expected_type=_EXACT_TERMINAL_RECEIPT,
            label="retained result receipt",
        )
    )
    if receipt_value is not None and (
        receipt_value.run_id != run_id or receipt_value.invocation_id != invocation_id
    ):
        raise BindingError("retained receipt is bound to a different invocation")

    cleanup = None
    if cleanup_value is not None:
        if type(cleanup_value) is not _EXACT_CLEANUP_REPORT:
            raise ContractError("retained cleanup is not an exact cleanup report")
        failures = cleanup_value.failures
        if (
            type(cleanup_value.container_name) is not str
            or cleanup_value.container_name != container_name
            or type(cleanup_value.stop_attempted) is not bool
            or type(cleanup_value.kill_attempted) is not bool
            or type(cleanup_value.remove_attempted) is not bool
            or type(failures) is not tuple
            or any(type(item) is not str for item in failures)
            or type(cleanup_value.post_cleanup_absent) is not bool
        ):
            raise BindingError("retained cleanup has an invalid binding")
        cleanup = (
            container_name,
            cleanup_value.stop_attempted,
            cleanup_value.kill_attempted,
            cleanup_value.remove_attempted,
            tuple(failures),
            cleanup_value.post_cleanup_absent,
        )

    enforcement = None
    if enforcement_value is not None:
        if type(enforcement_value) is not _EXACT_ENFORCEMENT_ATTESTATION:
            raise ContractError("retained enforcement is not an exact attestation")
        if (
            enforcement_value.classification != "test"
            or enforcement_value.container_name != container_name
            or type(enforcement_value.evidence) is not tuple
        ):
            raise ContractError("retained enforcement is not test-classified and bound")
        enforcement = (
            "test",
            enforcement_value.argv_digest,
            enforcement_value.container_name,
            tuple(enforcement_value.evidence),
        )

    if status in _TERMINAL_RESULT_STATUSES:
        if (
            receipt_value is None
            or not receipt_value.accepted
            or not receipt_value.legal_transition
            or receipt_value.kind != status
        ):
            raise ContractError("retained terminal result lacks an accepted matching receipt")
        if status in {"completed", "waiting_for_input", "failed", "blocked"} and enforcement is None:
            raise ContractError("retained terminal result lacks test enforcement evidence")

    if (
        type(evidence) is not tuple
        or len(evidence) > 32
        or any(type(item) is not str or not item or len(item) > MAX_REFERENCE_LENGTH for item in evidence)
    ):
        raise ContractError("retained result evidence is malformed")
    def decoded(payload: bytes | None) -> object:
        return None if payload is None else json.loads(payload)

    return canonical_json_bytes(
        {
            "run_id": run_id,
            "invocation_id": invocation_id,
            "status": status,
            "container_name": container_name,
            "exit": decoded(exit_payload),
            "proposal": decoded(proposal_payload),
            "receipt": decoded(receipt_payload),
            "cleanup": cleanup,
            "enforcement": enforcement,
            "evidence": list(evidence),
        }
    )


def _public_result_from_canonical(
    key: tuple[str, str],
    payload: bytes,
) -> object:
    """Validate bytes and return a fresh immutable value view.

    The view classes are created per read.  Mutating a returned value or its
    class metadata can therefore affect only that discarded view, never the
    serialized authority or a later read.  No module result symbol or captured
    constructor participates in this path.
    """

    canonical = _validated_canonical_payload(payload)
    run_id, invocation_id = key
    if canonical["run_id"] != run_id or canonical["invocation_id"] != invocation_id:
        raise BindingError("retained canonical result has an invalid invocation binding")
    return _result_view(canonical)


def _record_view(raw: dict[str, object]) -> object:
    """Create an immutable, per-read attribute view over canonical values."""

    field_names = tuple(_snake_case(name) for name in raw)
    values = tuple(_view_value(value) for value in raw.values())

    class RecordView(tuple):
        __slots__ = ()

        def __getattr__(self, name: str) -> object:
            try:
                return self[field_names.index(name)]
            except ValueError as exc:
                raise AttributeError(name) from exc

    return RecordView(values)


def _view_value(value: object) -> object:
    if type(value) is dict:
        return _record_view(value)
    if type(value) is list:
        return tuple(_view_value(item) for item in value)
    return value


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _result_view(canonical: dict[str, object]) -> object:
    cleanup = canonical["cleanup"]
    if cleanup is not None:
        cleanup = {
            "containerName": cleanup[0],
            "stopAttempted": cleanup[1],
            "killAttempted": cleanup[2],
            "removeAttempted": cleanup[3],
            "failures": cleanup[4],
            "postCleanupAbsent": cleanup[5],
        }
    enforcement = canonical["enforcement"]
    if enforcement is not None:
        enforcement = {
            "classification": enforcement[0],
            "argvDigest": enforcement[1],
            "containerName": enforcement[2],
            "evidence": enforcement[3],
        }
    raw = {
        "status": canonical["status"],
        "containerName": canonical["container_name"],
        "exit": canonical["exit"],
        "proposal": canonical["proposal"],
        "receipt": canonical["receipt"],
        "cleanup": cleanup,
        "enforcement": enforcement,
        "evidence": canonical["evidence"],
    }
    fields = tuple(_snake_case(name) for name in raw)
    values = list(_view_value(value) for value in raw.values())

    class ResultView(tuple):
        __slots__ = ()

        def __getattr__(self, name: str) -> object:
            try:
                return self[fields.index(name)]
            except ValueError as exc:
                raise AttributeError(name) from exc

        @property
        def completed(self) -> bool:
            return self[0] == "completed"

        @property
        def production_completed(self) -> bool:
            return False

    if cleanup is not None:
        values[5] = _cleanup_view(cleanup)
    if enforcement is not None:
        values[6] = _enforcement_view(enforcement)
    return ResultView(tuple(values))


def _special_view(raw: dict[str, object], *, succeeded: bool = False, production: bool = False) -> object:
    field_names = tuple(_snake_case(name) for name in raw)
    values = tuple(_view_value(value) for value in raw.values())

    class SpecialView(tuple):
        __slots__ = ()

        def __getattr__(self, name: str) -> object:
            try:
                return self[field_names.index(name)]
            except ValueError as exc:
                raise AttributeError(name) from exc

        @property
        def succeeded(self) -> bool:
            if not succeeded:
                raise AttributeError("succeeded")
            return self[4] == () and self[5] is True

        @property
        def is_production(self) -> bool:
            if not production:
                raise AttributeError("is_production")
            return self[0] == "production"

    return SpecialView(values)


def _cleanup_view(raw: dict[str, object]) -> object:
    return _special_view(raw, succeeded=True)


def _enforcement_view(raw: dict[str, object]) -> object:
    return _special_view(raw, production=True)


def _validated_canonical_payload(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise ContractError("retained canonical payload has an invalid type")
    try:
        raw = json.loads(payload)
    except Exception as exc:
        raise ContractError("retained canonical payload is malformed") from exc
    if type(raw) is not dict or canonical_json_bytes(raw) != payload:
        raise ContractError("retained canonical payload is not stable")
    required = {
        "run_id", "invocation_id", "status", "container_name", "exit", "proposal",
        "receipt", "cleanup", "enforcement", "evidence",
    }
    if set(raw) != required:
        raise ContractError("retained canonical payload has an invalid shape")
    if (
        type(raw["run_id"]) is not str
        or type(raw["invocation_id"]) is not str
        or type(raw["status"]) is not str
        or raw["status"] not in _RESULT_STATUSES
        or type(raw["container_name"]) is not str
        or not _CONTAINER_NAME.fullmatch(raw["container_name"])
        or (raw["exit"] is not None and type(raw["exit"]) is not dict)
        or (raw["proposal"] is not None and type(raw["proposal"]) is not dict)
        or (raw["receipt"] is not None and type(raw["receipt"]) is not dict)
        or (raw["cleanup"] is not None and type(raw["cleanup"]) is not list)
        or (raw["enforcement"] is not None and type(raw["enforcement"]) is not list)
        or type(raw["evidence"]) is not list
        or len(raw["evidence"]) > 32
        or any(type(item) is not str or not item or len(item) > MAX_REFERENCE_LENGTH for item in raw["evidence"])
    ):
        raise ContractError("retained canonical payload has invalid fields")
    if raw["proposal"] is not None and (
        raw["proposal"].get("runId") != raw["run_id"]
        or raw["proposal"].get("invocationId") != raw["invocation_id"]
    ):
        raise BindingError("retained proposal is bound to a different invocation")
    if raw["receipt"] is not None and (
        raw["receipt"].get("runId") != raw["run_id"]
        or raw["receipt"].get("invocationId") != raw["invocation_id"]
    ):
        raise BindingError("retained receipt is bound to a different invocation")
    if raw["cleanup"] is not None:
        cleanup = raw["cleanup"]
        if len(cleanup) != 6 or cleanup[0] != raw["container_name"] or type(cleanup[0]) is not str:
            raise BindingError("retained cleanup record has an invalid binding")
        if (
            any(type(item) is not bool for item in cleanup[1:4])
            or type(cleanup[4]) is not list
            or any(type(item) is not str for item in cleanup[4])
            or type(cleanup[5]) is not bool
        ):
            raise ContractError("retained cleanup record is malformed")
    if raw["enforcement"] is not None:
        enforcement = raw["enforcement"]
        if (
            len(enforcement) != 4
            or enforcement[0] != "test"
            or type(enforcement[1]) is not str
            or type(enforcement[2]) is not str
            or enforcement[2] != raw["container_name"]
            or type(enforcement[3]) is not list
            or any(type(item) is not str for item in enforcement[3])
        ):
            raise ContractError("retained enforcement record is not test-only")
    if raw["status"] in _TERMINAL_RESULT_STATUSES:
        receipt = raw["receipt"]
        if (
            receipt is None
            or receipt.get("accepted") is not True
            or receipt.get("legalTransition") is not True
            or receipt.get("kind") != raw["status"]
        ):
            raise ContractError("retained terminal result lacks an accepted matching receipt")
        if raw["status"] in {"completed", "waiting_for_input", "failed", "blocked"} and raw["enforcement"] is None:
            raise ContractError("retained terminal result lacks test enforcement evidence")
    return raw


def _argv_digest(argv: Sequence[str]) -> str:
    try:
        encoded = b"\0".join(item.encode("utf-8") for item in argv)
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ContractError("Docker argv must contain UTF-8 strings") from exc
    return hashlib.sha256(encoded).hexdigest()


def build_invocation_env(policy: InvocationPolicy | None = None) -> dict[str, str]:
    """Return the literal child environment; ambient process state is ignored."""

    if policy is not None:
        _validate_policy(policy)
    return dict(_FIXED_ALLOWED_ENV)


def _validate_binding(run: RunSnapshot, invocation: InvocationEnvelope) -> None:
    if not isinstance(run, RunSnapshot) or not isinstance(invocation, InvocationEnvelope):
        raise ContractError("invocation binding requires runtime contract values")
    if run.protocol != "plane.agent-runtime/v1" or invocation.protocol != run.protocol:
        raise ContractError("invocation protocol is invalid")
    if invocation.run_id != run.run_id or invocation.run_snapshot_digest != run.digest():
        raise BindingError("invocation is not bound to the supplied run snapshot")


def _binding_digest(run: RunSnapshot, invocation: InvocationEnvelope) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "invocationId": invocation.invocation_id,
                "runId": run.run_id,
                "snapshotDigest": run.digest(),
            }
        )
    ).hexdigest()


def invocation_container_name(run: RunSnapshot, invocation: InvocationEnvelope) -> str:
    """Derive one opaque, invocation-bound Docker name without user text."""

    _validate_binding(run, invocation)
    name = f"{_FIXED_CONTAINER_PREFIX}-{_binding_digest(run, invocation)[:32]}"
    if not _CONTAINER_NAME.fullmatch(name):  # pragma: no cover - fixed construction
        raise ContractError("derived container name is invalid")
    return name


def _size(value: int) -> str:
    if value % (1024 * 1024) == 0:
        return f"{value // (1024 * 1024)}m"
    if value % 1024 == 0:
        return f"{value // 1024}k"
    return f"{value}b"


def build_invocation_argv(
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    policy: InvocationPolicy,
) -> tuple[str, ...]:
    """Build the complete fixed Docker create argv for one invocation."""

    values = _validate_policy(policy)
    (
        image,
        cpu,
        memory,
        pids,
        _wall,
        _stdout,
        _stderr,
        _frame,
        _request,
        _frames,
        tmpfs,
        storage,
        stop_timeout,
        _kill_timeout,
        _remove_timeout,
    ) = values
    _validate_binding(run, invocation)
    name = invocation_container_name(run, invocation)
    binding = _binding_digest(run, invocation)
    argv: list[str] = [
        "docker",
        "create",
        "--name",
        name,
        "--label",
        _FIXED_PROTOCOL_LABEL,
        "--label",
        f"plane.agent-runtime/invocation-binding=sha256:{binding}",
        "--pull",
        _FIXED_PULL_POLICY,
        "--network",
        _FIXED_NETWORK,
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--user",
        _FIXED_USER,
        "--cpus",
        f"{int(cpu) / 1000:.3f}",
        "--memory",
        _size(int(memory)),
        "--pids-limit",
        str(int(pids)),
        "--stop-timeout",
        str(max(1, int(float(stop_timeout)))),
        "--tmpfs",
        f"{_FIXED_TMPFS_TARGET}:{_FIXED_TMPFS_OPTIONS},size={_size(int(tmpfs))}",
        "--storage-opt",
        f"size={_size(int(storage))}",
        "--log-driver",
        _FIXED_LOG_DRIVER,
    ]
    for key, value in _FIXED_ALLOWED_ENV:
        argv.extend(("--env", f"{key}={value}"))
    argv.extend(("--entrypoint", _FIXED_ENTRYPOINT, str(image), "-m", _FIXED_SERVICE_MODULE, *_FIXED_SERVICE_ARGS))
    return tuple(argv)


def _request_bytes(run: RunSnapshot, invocation: InvocationEnvelope, policy: InvocationPolicy) -> bytes:
    _validate_policy(policy)
    _validate_binding(run, invocation)
    payload = json.dumps(
        {"invocation": invocation.to_dict(), "run": run.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(payload) + 1 > int(policy.request_limit_bytes):
        raise BoundsError("invocation request exceeds the supervisor request bound")
    return payload + b"\n"


class _NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


def _safe_cancel_check(signal: CancellationSignal) -> tuple[bool, bool]:
    try:
        value = signal.is_cancelled()
    except Exception:
        return True, True
    if not isinstance(value, bool):
        return True, True
    return value, False


@dataclass(frozen=True)
class _ParsedOutput:
    exit: RuntimeExit
    proposal: TerminalProposal
    final_sequence: int


def _parse_child_output(
    stdout: bytes,
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    policy: InvocationPolicy,
) -> _ParsedOutput:
    _validate_policy(policy)
    if not stdout or len(stdout) > int(policy.stdout_limit_bytes):
        raise BoundsError("child stdout exceeded its supervisor bound")
    if not stdout.endswith(b"\n"):
        raise ContractError("child stdout ended with a truncated frame")
    collector = EventCollector(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        expected_causation_ref=invocation.causation_ref,
    )
    proposal: TerminalProposal | None = None
    exit_value: RuntimeExit | None = None
    frames = 0
    lines = stdout.split(b"\n")
    if lines[-1] != b"":
        raise ContractError("child stdout has an invalid line terminator")
    for raw_line in lines[:-1]:
        frames += 1
        if frames > int(policy.max_output_frames):
            raise BoundsError("child frame count exceeded its supervisor bound")
        if not raw_line or len(raw_line) > int(policy.frame_limit_bytes) or raw_line.endswith(b"\r"):
            raise BoundsError("child frame exceeded its supervisor bound")
        try:
            frame = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("child emitted malformed JSON output") from exc
        if not isinstance(frame, dict):
            raise ContractError("child emitted a non-object frame")
        kind = frame.get("type")
        if exit_value is not None:
            raise ContractError("child emitted bytes after terminal exit")
        if kind == "event":
            if proposal is not None:
                raise ContractError("child emitted an event after its terminal proposal")
            from .contract import RuntimeEvent

            collector.accept(RuntimeEvent.from_dict(frame.get("event")))
        elif kind == "proposal":
            if proposal is not None:
                raise ContractError("child emitted duplicate terminal proposals")
            raw_proposal = frame.get("proposal")
            raw_kind = raw_proposal.get("kind") if isinstance(raw_proposal, dict) else None
            try:
                proposal = TerminalProposal.from_dict(raw_proposal)
                validate_terminal_proposal(
                    run=run,
                    invocation=invocation,
                    kind=proposal.kind,
                    proposal=proposal,
                    stream=collector,
                )
            except ChildCancellationProposalRejected:
                raise
            except Exception as exc:
                if raw_kind == "cancelled":
                    raise ChildCancellationProposalRejected(
                        "child cancellation proposal failed host-only validation"
                    ) from exc
                raise
        elif kind == "exit":
            if proposal is None:
                raise ContractError("child emitted exit before its terminal proposal")
            exit_value = RuntimeExit.from_dict(frame.get("exit"))
        else:
            raise ContractError("child emitted an unsupported service frame")
    if proposal is None or exit_value is None:
        raise ContractError("child did not return exactly one proposal and one exit")
    if exit_value.final_sequence != collector.last_sequence:
        raise ContractError("child exit sequence does not match bounded event evidence")
    if proposal.final_sequence != collector.last_sequence or proposal.kind != exit_value.kind:
        raise BindingError("child proposal is not bound to the terminal exit")
    if proposal.source != "runtime":
        raise BindingError("child proposal has an invalid authority source")
    return _ParsedOutput(exit_value, proposal, collector.last_sequence)


class _SubprocessDockerProcess:
    """Selector-driven bounded I/O for the Docker attach process."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    def _close_streams(self) -> None:
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _terminate_and_reap(self) -> bool:
        """End the attach client and reap it under fixed, bounded deadlines."""

        if self._process.poll() is None:
            try:
                self._process.terminate()
            except OSError:
                pass
            try:
                self._process.wait(timeout=_ATTACH_TERMINATE_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                if self._process.poll() is None:
                    try:
                        self._process.kill()
                    except OSError:
                        pass
                    try:
                        self._process.wait(timeout=_ATTACH_KILL_TIMEOUT_SECONDS)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
        else:
            try:
                self._process.wait(timeout=_ATTACH_KILL_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return self._process.poll() is not None

    def collect(
        self,
        *,
        input_bytes: bytes,
        deadline: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        is_cancelled: Callable[[], bool],
    ) -> ProcessCapture:
        selector = selectors.DefaultSelector()
        stdout = bytearray()
        stderr = bytearray()
        pending = memoryview(input_bytes)
        capture: ProcessCapture | None = None

        def finish(**kwargs: object) -> ProcessCapture:
            nonlocal capture
            capture = ProcessCapture(**kwargs)  # type: ignore[arg-type]
            return capture

        try:
            if self._process.stdin is not None:
                selector.register(self._process.stdin, selectors.EVENT_WRITE, "stdin")
            if self._process.stdout is not None:
                selector.register(self._process.stdout, selectors.EVENT_READ, "stdout")
            if self._process.stderr is not None:
                selector.register(self._process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                try:
                    cancelled = is_cancelled()
                except Exception:
                    cancelled = True
                if cancelled:
                    return finish(
                        returncode=self._process.poll(),
                        stdout=bytes(stdout),
                        stderr=bytes(stderr),
                        cancelled=True,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return finish(
                        returncode=self._process.poll(),
                        stdout=bytes(stdout),
                        stderr=bytes(stderr),
                        timed_out=True,
                    )
                for key, mask in selector.select(min(remaining, 0.1)):
                    if key.data == "stdin" and mask & selectors.EVENT_WRITE:
                        if not pending:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        try:
                            written = os.write(key.fileobj.fileno(), pending[:16 * 1024])
                        except (BrokenPipeError, OSError):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        else:
                            pending = pending[written:]
                            if not pending:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                    elif mask & selectors.EVENT_READ:
                        try:
                            chunk = os.read(key.fileobj.fileno(), 16 * 1024)
                        except OSError:
                            chunk = b""
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        target = stdout if key.data == "stdout" else stderr
                        limit = stdout_limit_bytes if key.data == "stdout" else stderr_limit_bytes
                        if len(target) + len(chunk) > limit:
                            return finish(
                                returncode=self._process.poll(),
                                stdout=bytes(stdout),
                                stderr=bytes(stderr),
                                output_exceeded=True,
                            )
                        target.extend(chunk)
                if self._process.poll() is not None and "stdin" in {
                    item.data for item in selector.get_map().values()
                }:
                    key = next(item for item in selector.get_map().values() if item.data == "stdin")
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            return finish(
                returncode=self._process.poll(),
                stdout=bytes(stdout),
                stderr=bytes(stderr),
            )
        finally:
            selector.close()
            self._close_streams()
            reaped = self._terminate_and_reap()
            if capture is not None:
                object.__setattr__(capture, "reaped", reaped)


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    if not raw or len(raw) > _MAX_DOCKER_INSPECTION_BYTES or not raw.endswith(b"\n"):
        raise RuntimeConfigurationError(f"Docker {label} inspection was unbounded or incomplete")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigurationError(f"Docker {label} inspection was not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeConfigurationError(f"Docker {label} inspection was not an object")
    return value


class SubprocessDockerRunner:
    """Docker CLI adapter with evidence-based preflight and cleanup."""

    def __init__(self, capabilities: DockerRunnerCapabilities | None = None) -> None:
        # Kept only for source compatibility with old test adapters.  The
        # supervisor never reads this caller assertion.
        self.capabilities = capabilities
        self._attestation: EnforcementAttestation | None = None

    def _run_json(self, command: Sequence[str], *, timeout_seconds: float, label: str) -> dict[str, object]:
        try:
            completed = subprocess.run(
                tuple(command),
                env=dict(_DOCKER_CLIENT_ENV),
                cwd="/",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(0.1, timeout_seconds),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeConfigurationError(f"Docker {label} inspection failed") from exc
        if completed.returncode != 0:
            raise RuntimeConfigurationError(f"Docker {label} inspection was rejected")
        return _json_object(completed.stdout, label)

    @staticmethod
    def _image_env(config: dict[str, object]) -> None:
        raw_env = config.get("Env")
        if raw_env in (None, []):
            return
        if not isinstance(raw_env, list):
            raise RuntimeConfigurationError("image environment inspection is ambiguous")
        for item in raw_env:
            if not isinstance(item, str) or "=" not in item or _CONTROL.search(item):
                raise RuntimeConfigurationError("image environment contains an invalid entry")
            key, _value = item.split("=", 1)
            if key not in _ALLOWED_IMAGE_ENV_KEYS:
                raise RuntimeConfigurationError("image contains a non-allowlisted environment key")

    @staticmethod
    def _image_command(config: dict[str, object]) -> None:
        for field in ("Entrypoint", "Cmd"):
            value = config.get(field)
            if value is None:
                continue
            if not isinstance(value, list) or any(not isinstance(item, str) or _CONTROL.search(item) for item in value):
                raise RuntimeConfigurationError(f"image {field} inspection is ambiguous")
            if any(
                item in {"--privileged", "--network=host", "--pid=host", "--ipc=host"}
                or item.startswith(("--device", "--cap-add", "--mount", "--volume", "-v"))
                for item in value
            ):
                raise RuntimeConfigurationError(f"image {field} contains an unsafe surprise")

    @staticmethod
    def _size_bytes(value: str) -> int:
        match = re.fullmatch(r"([0-9]+)([kKmMgG]?)", value)
        if match is None:
            raise ContractError("Docker size value is invalid")
        number = int(match.group(1))
        multiplier = {"": 1, "k": 1024, "K": 1024, "m": 1024**2, "M": 1024**2, "g": 1024**3, "G": 1024**3}[match.group(2)]
        return number * multiplier

    @classmethod
    def _policy_from_argv(cls, argv: Sequence[str]) -> InvocationPolicy:
        image = cls._image_from_argv(argv)
        cpus = cls._flag_value(argv, "--cpus")
        cpu_number = float(cpus)
        if not math.isfinite(cpu_number) or cpu_number <= 0:
            raise ContractError("Docker CPU value is invalid")
        cpu_millicores = int(round(cpu_number * 1000))
        if f"{cpu_millicores / 1000:.3f}" != cpus:
            raise ContractError("Docker CPU value is not canonical")
        memory_bytes = cls._size_bytes(cls._flag_value(argv, "--memory"))
        if _size(memory_bytes) != cls._flag_value(argv, "--memory"):
            raise ContractError("Docker memory value is not canonical")
        pids_limit = int(cls._flag_value(argv, "--pids-limit"))
        if str(pids_limit) != cls._flag_value(argv, "--pids-limit"):
            raise ContractError("Docker PID value is not canonical")
        stop_timeout = cls._flag_value(argv, "--stop-timeout")
        if not stop_timeout.isdigit() or int(stop_timeout) < 1 or str(int(stop_timeout)) != stop_timeout:
            raise ContractError("Docker stop timeout is invalid")
        tmpfs = cls._flag_value(argv, "--tmpfs")
        target, options = tmpfs.split(":", 1)
        if target != _FIXED_TMPFS_TARGET:
            raise ContractError("Docker tmpfs target is invalid")
        option_values = options.split(",")
        fixed_options = _FIXED_TMPFS_OPTIONS.split(",")
        if set(option_values[: len(fixed_options)]) != set(fixed_options):
            raise ContractError("Docker tmpfs options are invalid")
        size_values = [item[5:] for item in option_values[len(fixed_options) :] if item.startswith("size=")]
        if len(size_values) != 1 or len(option_values) != len(fixed_options) + 1:
            raise ContractError("Docker tmpfs size is invalid")
        tmpfs_bytes = cls._size_bytes(size_values[0])
        if _size(tmpfs_bytes) != size_values[0]:
            raise ContractError("Docker tmpfs size is not canonical")
        storage = cls._flag_value(argv, "--storage-opt")
        if not storage.startswith("size=") or storage.count("=") != 1:
            raise ContractError("Docker storage size is invalid")
        storage_bytes = cls._size_bytes(storage[5:])
        if _size(storage_bytes) != storage[5:]:
            raise ContractError("Docker storage size is not canonical")
        return InvocationPolicy(
            image,
            cpu_millicores=cpu_millicores,
            memory_bytes=memory_bytes,
            pids_limit=pids_limit,
            tmpfs_bytes=tmpfs_bytes,
            storage_limit_bytes=storage_bytes,
        )

    @classmethod
    def _assert_fixed_argv_shape(cls, argv: Sequence[str]) -> None:
        if tuple(argv[:2]) != ("docker", "create"):
            raise ContractError("Docker runner accepts only fixed docker create argv")
        allowed_flags = {
            "--name", "--label", "--pull", "--network", "--read-only",
            "--security-opt", "--cap-drop", "--user", "--cpus", "--memory",
            "--pids-limit", "--stop-timeout", "--tmpfs", "--storage-opt",
            "--log-driver", "--env", "--entrypoint", "--once", "-m",
        }
        if any(item.startswith("-") and item not in allowed_flags for item in argv[2:]):
            raise ContractError("Docker argv contains an unsupported flag")
        for flag, expected in (
            ("--pull", _FIXED_PULL_POLICY),
            ("--network", _FIXED_NETWORK),
            ("--security-opt", "no-new-privileges"),
            ("--cap-drop", "ALL"),
            ("--user", _FIXED_USER),
            ("--log-driver", _FIXED_LOG_DRIVER),
            ("--entrypoint", _FIXED_ENTRYPOINT),
        ):
            if cls._flag_value(argv, flag) != expected:
                raise ContractError(f"Docker argv {flag} is not fixed")
        for flag in ("--read-only",):
            if list(argv).count(flag) != 1:
                raise ContractError(f"Docker argv must contain exactly one {flag}")
        if list(argv).count("--name") != 1 or list(argv).count("--label") != 2 or list(argv).count("--env") != len(_FIXED_ALLOWED_ENV):
            raise ContractError("Docker argv labels/environment are not fixed")
        labels = [argv[index + 1] for index, value in enumerate(argv) if value == "--label"]
        if _FIXED_PROTOCOL_LABEL not in labels or sum(
            bool(re.fullmatch(r"plane\.agent-runtime/invocation-binding=sha256:[0-9a-f]{64}", item))
            for item in labels
        ) != 1:
            raise ContractError("Docker argv binding labels are not fixed")
        expected_env = [f"{key}={value}" for key, value in _FIXED_ALLOWED_ENV]
        actual_env = [argv[index + 1] for index, value in enumerate(argv) if value == "--env"]
        if actual_env != expected_env:
            raise ContractError("Docker argv environment is not the fixed allowlist")
        cls._policy_from_argv(argv)

    def _preflight(self, argv: Sequence[str], *, timeout_seconds: float) -> None:
        if tuple(argv[:2]) != ("docker", "create"):
            raise ContractError("Docker runner accepts only fixed docker create argv")
        info = self._run_json(("docker", "info", "--format", "{{json .}}"), timeout_seconds=timeout_seconds, label="daemon")
        if info.get("OSType") != "linux" or not isinstance(info.get("Driver"), str):
            raise RuntimeConfigurationError("Docker daemon platform or storage driver is ambiguous")
        if info["Driver"] not in _SUPPORTED_STORAGE_DRIVERS:
            raise RuntimeConfigurationError("Docker storage driver cannot prove a bounded layer")
        cgroup_version = info.get("CgroupVersion")
        if cgroup_version not in {"1", "2"}:
            raise RuntimeConfigurationError("Docker cgroup version is ambiguous")
        security_options = info.get("SecurityOptions")
        if info.get("Rootless") is not False or not isinstance(security_options, list) or any("rootless" in str(item).lower() for item in security_options):
            raise RuntimeConfigurationError("Docker rootless/security mode is ambiguous")
        image = self._image_from_argv(argv)
        image_info = self._run_json(
            ("docker", "image", "inspect", "--format", "{{json .}}", image),
            timeout_seconds=timeout_seconds,
            label="image",
        )
        if image_info.get("Os") not in (None, "linux"):
            raise RuntimeConfigurationError("Docker image platform is not linux")
        repo_digests = image_info.get("RepoDigests")
        if not isinstance(repo_digests, list) or image not in repo_digests:
            raise RuntimeConfigurationError("exact digest-pinned image is not locally verified")
        config = image_info.get("Config")
        if not isinstance(config, dict):
            raise RuntimeConfigurationError("Docker image config is ambiguous")
        self._image_env(config)
        self._image_command(config)
        volumes = config.get("Volumes")
        if volumes not in (None, {}):
            raise RuntimeConfigurationError("image-declared volumes are not permitted")

    @staticmethod
    def _flag_value(argv: Sequence[str], flag: str) -> str:
        values = [index for index, value in enumerate(argv) if value == flag]
        if len(values) != 1 or values[0] + 1 >= len(argv):
            raise ContractError(f"Docker argv must contain exactly one {flag} flag")
        value = argv[values[0] + 1]
        if not isinstance(value, str) or _CONTROL.search(value):
            raise ContractError(f"Docker argv {flag} value is invalid")
        return value

    @classmethod
    def _image_from_argv(cls, argv: Sequence[str]) -> str:
        entrypoint_index = [index for index, value in enumerate(argv) if value == "--entrypoint"]
        if len(entrypoint_index) != 1:
            raise ContractError("Docker argv must contain exactly one entrypoint")
        image_index = entrypoint_index[0] + 2
        if image_index >= len(argv):
            raise ContractError("Docker argv has no image")
        image = argv[image_index]
        _validate_image_reference(image)
        if tuple(argv[image_index + 1 :]) != ("-m", _FIXED_SERVICE_MODULE, *_FIXED_SERVICE_ARGS):
            raise ContractError("Docker argv has an unexpected child command")
        return image

    def attest_invocation(
        self,
        argv: Sequence[str],
        *,
        client_env: Mapping[str, str],
    ) -> EnforcementAttestation:
        if dict(client_env) != dict(_DOCKER_CLIENT_ENV):
            raise ContractError("Docker client environment is not the fixed allowlist")
        name = self._flag_value(argv, "--name")
        if not _CONTAINER_NAME.fullmatch(name):
            raise ContractError("Docker argv has an invalid deterministic container name")
        self._assert_fixed_argv_shape(argv)
        self._preflight(argv, timeout_seconds=5.0)
        attestation = EnforcementAttestation(
            classification="test",
            argv_digest=_argv_digest(argv),
            container_name=name,
            evidence=(
                "daemon_inspected",
                "image_digest_inspected",
                "image_config_inspected",
                "fixed_argv",
                "production_path_unavailable",
            ),
        )
        self._attestation = attestation
        return attestation

    def _inspect_container(self, name: str, *, timeout_seconds: float, allow_absent: bool = False) -> dict[str, object] | None:
        try:
            completed = subprocess.run(
                ("docker", "inspect", "--format", "{{json .}}", name),
                env=dict(_DOCKER_CLIENT_ENV),
                cwd="/",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(0.1, timeout_seconds),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeConfigurationError("Docker container inspection failed") from exc
        if completed.returncode != 0:
            if allow_absent and b"no such object" in completed.stderr.lower():
                return None
            raise RuntimeConfigurationError("Docker container inspection was ambiguous")
        return _json_object(completed.stdout, "container")

    @staticmethod
    def _assert_post_launch(container: dict[str, object], argv: Sequence[str], policy: InvocationPolicy) -> None:
        host = container.get("HostConfig")
        config = container.get("Config")
        if not isinstance(host, dict) or not isinstance(config, dict):
            raise RuntimeConfigurationError("Docker post-launch config is ambiguous")
        if host.get("NetworkMode") != _FIXED_NETWORK or host.get("ReadonlyRootfs") is not True:
            raise RuntimeConfigurationError("Docker network/rootfs enforcement did not match")
        if config.get("User") != _FIXED_USER:
            raise RuntimeConfigurationError("Docker non-root user enforcement did not match")
        security = host.get("SecurityOpt")
        if not isinstance(security, list) or "no-new-privileges:true" not in security:
            raise RuntimeConfigurationError("Docker no-new-privileges enforcement did not match")
        cap_drop = host.get("CapDrop")
        if not isinstance(cap_drop, list) or "ALL" not in cap_drop:
            raise RuntimeConfigurationError("Docker capability drop enforcement did not match")
        if host.get("Memory") != policy.memory_bytes or host.get("PidsLimit") != policy.pids_limit:
            raise RuntimeConfigurationError("Docker memory/PID enforcement did not match")
        nano_cpus = host.get("NanoCpus")
        if nano_cpus != int(policy.cpu_millicores * 1_000_000):
            raise RuntimeConfigurationError("Docker CPU enforcement did not match")
        tmpfs = host.get("Tmpfs")
        expected_tmpfs = f"{_FIXED_TMPFS_OPTIONS},size={_size(policy.tmpfs_bytes)}"
        if not isinstance(tmpfs, dict) or tmpfs.get(_FIXED_TMPFS_TARGET) != expected_tmpfs:
            raise RuntimeConfigurationError("Docker tmpfs enforcement did not match")
        storage_opt = host.get("StorageOpt")
        if not isinstance(storage_opt, dict) or storage_opt.get("size") != _size(policy.storage_limit_bytes):
            raise RuntimeConfigurationError("Docker storage enforcement did not match")
        log_config = container.get("LogConfig")
        if not isinstance(log_config, dict) or log_config.get("Type") != _FIXED_LOG_DRIVER:
            raise RuntimeConfigurationError("Docker logging enforcement did not match")
        if (
            container.get("Mounts") not in ([], None)
            or host.get("Binds") not in ([], None)
            or host.get("VolumesFrom") not in ([], None)
            or host.get("Devices") not in ([], None)
        ):
            raise RuntimeConfigurationError("Docker mounts/devices are not isolated")
        for key in ("PidMode", "IpcMode", "UTSMode", "UsernsMode"):
            if host.get(key) in {"host", "/host"}:
                raise RuntimeConfigurationError("Docker host namespace is not isolated")
        expected_env = [f"{key}={value}" for key, value in _FIXED_ALLOWED_ENV]
        if config.get("Env") != expected_env:
            raise RuntimeConfigurationError("Docker environment did not clear image state")
        if config.get("Entrypoint") != [_FIXED_ENTRYPOINT] or config.get("Cmd") != ["-m", _FIXED_SERVICE_MODULE, *_FIXED_SERVICE_ARGS]:
            raise RuntimeConfigurationError("Docker child command did not match the fixed command")
        if container.get("Name") not in (None, f"/{SubprocessDockerRunner._flag_value(argv, '--name')}"):
            raise RuntimeConfigurationError("Docker container identity did not match")

    def launch(
        self,
        argv: Sequence[str],
        *,
        client_env: Mapping[str, str],
        input_bytes: bytes,
    ) -> InvocationProcess:
        if dict(client_env) != dict(_DOCKER_CLIENT_ENV):
            raise ContractError("Docker client environment is not the fixed allowlist")
        name = self._flag_value(argv, "--name")
        expected = self._attestation
        if expected is None or expected.argv_digest != _argv_digest(argv) or expected.container_name != name:
            raise RuntimeConfigurationError("Docker launch lacks an exact preflight attestation")
        del input_bytes
        try:
            created = subprocess.run(
                tuple(argv),
                env=dict(_DOCKER_CLIENT_ENV),
                cwd="/",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeConfigurationError("Docker create failed") from exc
        if created.returncode != 0 or not _HEX_ID.fullmatch(created.stdout.decode("ascii", "ignore").strip()):
            raise RuntimeConfigurationError("Docker create did not return a bounded container identity")
        container = self._inspect_container(name, timeout_seconds=5.0)
        if container is None:
            raise RuntimeConfigurationError("Docker create identity cannot be inspected")
        # Reconstruct only the post-launch scalar expectations from the exact
        # argv that was attested before create.  Dangerous controls remain
        # module-private and are checked by _assert_fixed_argv_shape.
        raise_policy = self._policy_from_argv(argv)
        self._assert_post_launch(container, argv, raise_policy)
        try:
            started = subprocess.run(
                ("docker", "start", name),
                env=dict(_DOCKER_CLIENT_ENV),
                cwd="/",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeConfigurationError("Docker start failed") from exc
        if started.returncode != 0:
            raise RuntimeConfigurationError("Docker start was rejected")
        running = self._inspect_container(name, timeout_seconds=5.0)
        if running is None:
            raise RuntimeConfigurationError("Docker container disappeared during start")
        self._assert_post_launch(running, argv, raise_policy)
        attached = subprocess.Popen(
            ("docker", "attach", "--sig-proxy=false", name),
            env=dict(_DOCKER_CLIENT_ENV),
            cwd="/",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        return _SubprocessDockerProcess(attached)

    def cleanup(
        self,
        container_name: str,
        *,
        stop_timeout_seconds: float,
        kill_timeout_seconds: float,
        remove_timeout_seconds: float,
    ) -> CleanupReport:
        if not _CONTAINER_NAME.fullmatch(container_name):
            return CleanupReport(container_name, False, False, False, ("invalid_name",), False)
        failures: list[str] = []
        stop_attempted = kill_attempted = remove_attempted = False
        try:
            state = self._inspect_container(container_name, timeout_seconds=stop_timeout_seconds, allow_absent=True)
        except Exception:
            return CleanupReport(container_name, False, False, False, ("inspect_failed",), False)
        if state is not None:
            state_data = state.get("State")
            running = isinstance(state_data, dict) and state_data.get("Running") is True
            if running:
                stop_attempted = True
                try:
                    self._control(("docker", "stop", "--time", "1", container_name), stop_timeout_seconds)
                except Exception:
                    try:
                        post_stop = self._inspect_container(container_name, timeout_seconds=stop_timeout_seconds, allow_absent=True)
                    except Exception:
                        post_stop = state
                    still_running = post_stop is not None and isinstance(post_stop.get("State"), dict) and post_stop["State"].get("Running") is True
                    if still_running:
                        kill_attempted = True
                        try:
                            self._control(("docker", "kill", container_name), kill_timeout_seconds)
                        except Exception:
                            failures.append("kill_failed")
            remove_attempted = True
            try:
                self._control(("docker", "rm", "--force", "--volumes", container_name), remove_timeout_seconds)
            except Exception:
                try:
                    if self._inspect_container(container_name, timeout_seconds=remove_timeout_seconds, allow_absent=True) is not None:
                        failures.append("remove_failed")
                except Exception:
                    failures.append("remove_failed")
        try:
            absent = self._inspect_container(container_name, timeout_seconds=remove_timeout_seconds, allow_absent=True) is None
        except Exception:
            absent = False
            failures.append("post_cleanup_inspect_failed")
        return CleanupReport(
            container_name,
            stop_attempted,
            kill_attempted,
            remove_attempted,
            tuple(dict.fromkeys(failures)),
            absent,
        )

    @staticmethod
    def _control(command: Sequence[str], timeout_seconds: float) -> None:
        completed = subprocess.run(
            tuple(command),
            env=dict(_DOCKER_CLIENT_ENV),
            cwd="/",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(0.1, timeout_seconds),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeConfigurationError("Docker cleanup command failed")


_RETENTION_SCHEMA = (
    "CREATE TABLE supervisor_results ("
    "run_id TEXT NOT NULL, invocation_id TEXT NOT NULL, payload BLOB NOT NULL,"
    "payload_digest BLOB NOT NULL,"
    "PRIMARY KEY (run_id, invocation_id))",
    "CREATE TABLE supervisor_death_receipts ("
    "run_id TEXT NOT NULL, invocation_id TEXT NOT NULL, payload BLOB, payload_digest BLOB,"
    "has_receipt INTEGER NOT NULL, PRIMARY KEY (run_id, invocation_id))",
)


def _new_retention_anchor() -> tuple[str, sqlite3.Connection]:
    """Create an opaque per-supervisor serialized store.

    SQLite owns transaction ordering in its C implementation.  No Python
    module-global dictionary, weak-reference ledger, live result object, or
    mutable lock is authoritative.  The URI is an in-memory SQLite name; the
    anchor connection only keeps that private database alive.
    """

    uri = f"file:hermes-retention-{secrets.token_hex(32)}?mode=memory&cache=shared"
    anchor = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
    for statement in _RETENTION_SCHEMA:
        anchor.execute(statement)
    return uri, anchor


def _retention_connection(owner: object) -> sqlite3.Connection:
    try:
        uri = object.__getattribute__(owner, "_retention_uri")
        anchor = object.__getattribute__(owner, "_retention_anchor")
    except AttributeError as exc:
        raise RuntimeConfigurationError("invocation retention store is unavailable") from exc
    if (
        type(uri) is not str
        or not uri.startswith("file:hermes-retention-")
        or type(anchor) is not sqlite3.Connection
    ):
        raise RuntimeConfigurationError("invocation retention store was replaced")
    try:
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
    except Exception as exc:
        raise RuntimeConfigurationError("invocation retention store is invalid") from exc
    return connection


def _retention_transaction(owner: object) -> sqlite3.Connection:
    """Acquire SQLite's serialized write transaction with bounded retry."""

    deadline = time.monotonic() + 5.0
    while True:
        connection = _retention_connection(owner)
        try:
            connection.execute("BEGIN IMMEDIATE")
            return connection
        except sqlite3.OperationalError as exc:
            connection.close()
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(0.005)


def _retention_snapshot(owner: object) -> dict[tuple[str, str], bytes]:
    connection = _retention_connection(owner)
    try:
        rows = connection.execute(
            "SELECT run_id, invocation_id, payload FROM supervisor_results"
        ).fetchall()
        return {(run_id, invocation_id): bytes(payload) for run_id, invocation_id, payload in rows}
    finally:
        connection.close()


def _retention_count(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM supervisor_results").fetchone()
    if row is None or type(row[0]) is not int:
        raise RuntimeConfigurationError("invocation retention count is invalid")
    return row[0]


def _verify_integrity(payload: bytes, expected_digest: object) -> None:
    if type(expected_digest) is not bytes or expected_digest != hashlib.sha256(payload).digest():
        raise ContractError("retained canonical payload failed its integrity check")


class InvocationSupervisor:
    """Run, bound, reconcile, and dispose of exactly one invocation."""

    def __init__(
        self,
        *,
        policy: InvocationPolicy,
        runner: DockerRunner | None = None,
        terminal_port: TerminalReconciliationPort,
        lease_authority: CanonicalLeaseAuthority,
        lease_binding: CanonicalLeaseBinding,
        cancellation_authority: CancellationAuthority | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_policy(policy)
        if terminal_port is None:
            raise RuntimeConfigurationError("invocation supervisor requires terminal reconciliation")
        if lease_authority is None or lease_binding is None:
            raise RuntimeConfigurationError("invocation supervisor requires host lease authority")
        self.policy = policy
        self.runner = runner
        self.terminal_port = terminal_port
        self.lease_authority = lease_authority
        self.lease_binding = lease_binding
        self.cancellation_authority = cancellation_authority
        self._lease_fingerprint = (
            lease_binding.run_id,
            lease_binding.invocation_id,
            lease_binding.lease_id,
            lease_binding.holder_ref,
            lease_binding.active,
            lease_binding.expires_at,
        )
        self._clock = clock
        self._retention_uri, self._retention_anchor = _new_retention_anchor()

    @property
    def _results(self) -> Mapping[tuple[str, str], bytes]:
        """Fresh diagnostic bytes; returned mappings are never authoritative."""

        return MappingProxyType(_retention_snapshot(self))

    @_results.setter
    def _results(self, replacement: object) -> None:
        """Discard caller-supplied cache replacement attempts."""

        del replacement

    def _attest(self, argv: Sequence[str]) -> EnforcementAttestation:
        runner = self.runner
        attest = getattr(runner, "attest_invocation", None)
        if not callable(attest):
            raise RuntimeConfigurationError(
                "production enforcement path is unavailable; runner evidence is not configured"
            )
        result = attest(argv, client_env=dict(_DOCKER_CLIENT_ENV))
        if type(result) is not _EXACT_ENFORCEMENT_ATTESTATION:
            raise RuntimeConfigurationError("runner returned an invalid enforcement attestation")
        expected_name = self._container_name_from_argv(argv)
        if result.argv_digest != _argv_digest(argv) or result.container_name != expected_name:
            raise BindingError("runner enforcement attestation is not bound to exact argv")
        if result.classification != "test":
            raise RuntimeConfigurationError(
                "production enforcement path is unavailable; runner attestations are test-only"
            )
        return _EXACT_ENFORCEMENT_ATTESTATION(
            classification="test",
            argv_digest=_argv_digest(argv),
            container_name=expected_name,
            evidence=tuple(dict.fromkeys((*result.evidence, "production_path_unavailable"))),
        )

    def _active_runner(self) -> DockerRunner:
        runner = self.runner
        if runner is None:  # pragma: no cover - constructor establishes this invariant
            raise RuntimeConfigurationError(
                "production enforcement path is unavailable; runner is not configured"
            )
        return runner

    @staticmethod
    def _container_name_from_argv(argv: Sequence[str]) -> str:
        indexes = [index for index, value in enumerate(argv) if value == "--name"]
        if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
            raise ContractError("Docker argv has no deterministic container name")
        name = argv[indexes[0] + 1]
        if not isinstance(name, str) or not _CONTAINER_NAME.fullmatch(name):
            raise ContractError("Docker argv has an invalid deterministic container name")
        return name

    def _cleanup(self, name: str) -> CleanupReport:
        _validate_policy(self.policy)
        cleanup = getattr(self._active_runner(), "cleanup", None)
        if not callable(cleanup):
            return CleanupReport(name, False, False, False, ("cleanup_not_supported",), False)
        try:
            report = cleanup(
                name,
                stop_timeout_seconds=float(self.policy.stop_timeout_seconds),
                kill_timeout_seconds=float(self.policy.kill_timeout_seconds),
                remove_timeout_seconds=float(self.policy.remove_timeout_seconds),
            )
        except Exception:
            return CleanupReport(name, False, False, False, ("cleanup_failed",), False)
        if type(report) is not _EXACT_CLEANUP_REPORT or report.container_name != name:
            return CleanupReport(name, False, False, False, ("invalid_cleanup_report",), False)
        return report

    @staticmethod
    def _invalid_retained_result(name: str) -> object:
        return _result_view(
            {
                "run_id": "invalid",
                "invocation_id": "invalid",
                "status": "supervisor_action_required",
                "container_name": name,
                "exit": None,
                "proposal": None,
                "receipt": None,
                "cleanup": None,
                "enforcement": None,
                "evidence": ["retained_result_invalid", "supervisor_action_required"],
            }
        )

    def _retain_result(
        self,
        connection: sqlite3.Connection,
        key: tuple[str, str],
        result: object,
        name: str,
    ) -> object:
        """Store only canonical bytes and reconstruct a fresh readback view."""

        try:
            canonical = _canonicalize_result(key, result)
            public = _public_result_from_canonical(key, canonical)
        except Exception:
            return self._invalid_retained_result(name)
        payload_digest = hashlib.sha256(canonical).digest()
        connection.execute(
            "INSERT INTO supervisor_results "
            "(run_id, invocation_id, payload, payload_digest) VALUES (?, ?, ?, ?)",
            (*key, canonical, payload_digest),
        )
        return public

    def _cached_result(
        self,
        connection: sqlite3.Connection,
        key: tuple[str, str],
        name: str,
        expected_argv_digest: str,
    ) -> object:
        """Validate a retained record; malformed state remains fail-closed."""

        row = connection.execute(
            "SELECT payload, payload_digest FROM supervisor_results "
            "WHERE run_id = ? AND invocation_id = ?",
            key,
        ).fetchone()
        if row is None:
            raise KeyError(key)
        try:
            payload = bytes(row[0])
            _verify_integrity(payload, bytes(row[1]))
            canonical = _validated_canonical_payload(payload)
            enforcement = canonical["enforcement"]
            if enforcement is not None and enforcement[1] != expected_argv_digest:
                raise BindingError("retained enforcement is bound to a different argv")
            return _public_result_from_canonical(key, payload)
        except Exception:
            return self._invalid_retained_result(name)

    def _result_for_host_cancellation(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        name: str,
        final_sequence: int,
        cleanup: CleanupReport | None,
        enforcement: EnforcementAttestation | None,
        evidence: tuple[str, ...],
    ) -> InvocationResult:
        """Reconcile cancellation from host state, never child evidence."""

        if self.cancellation_authority is None:
            return _draft_result(
                status="supervisor_action_required",
                container_name=name,
                exit=RuntimeExit("cancelled", final_sequence),
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=tuple(dict.fromkeys((*evidence, "cancellation_authority_missing", "supervisor_action_required"))),
            )
        try:
            proposal = build_host_cancellation_proposal(
                run=run,
                invocation=invocation,
                cancellation_authority=self.cancellation_authority,
                final_sequence=final_sequence,
            )
            receipt = reconcile_terminal_proposal(port=self.terminal_port, proposal=proposal)
        except Exception:
            return _draft_result(
                status="supervisor_action_required",
                container_name=name,
                exit=RuntimeExit("cancelled", final_sequence),
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=tuple(dict.fromkeys((*evidence, "host_cancellation_reconciliation_failed", "supervisor_action_required"))),
            )
        if not receipt.accepted or not receipt.legal_transition:
            return _draft_result(
                status="supervisor_action_required",
                container_name=name,
                exit=RuntimeExit("cancelled", final_sequence),
                proposal=proposal,
                receipt=receipt,
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=tuple(dict.fromkeys((*evidence, "host_cancellation_rejected", "supervisor_action_required"))),
            )
        if cleanup is not None and not cleanup.succeeded:
            return _draft_result(
                status="supervisor_action_required",
                container_name=name,
                exit=RuntimeExit("cancelled", final_sequence),
                proposal=proposal,
                receipt=receipt,
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=tuple(dict.fromkeys((*evidence, "host_cancellation", "cleanup_failed", "supervisor_action_required"))),
            )
        return _draft_result(
            status="cancelled",
            container_name=name,
            exit=RuntimeExit("cancelled", final_sequence),
            proposal=proposal,
            receipt=receipt,
            cleanup=cleanup,
            enforcement=enforcement,
            evidence=tuple(dict.fromkeys((*evidence, "host_cancellation", "host_reconciliation"))),
        )

    def _validate_host_lease(self, run: RunSnapshot, invocation: InvocationEnvelope) -> None:
        binding = self.lease_binding
        if (
            self._lease_fingerprint
            != (
                binding.run_id,
                binding.invocation_id,
                binding.lease_id,
                binding.holder_ref,
                binding.active,
                binding.expires_at,
            )
        ):
            raise BindingError("host lease binding was mutated after supervisor construction")
        if not binding.matches(invocation):
            raise BindingError("invocation lease is not the canonical host lease")
        self.lease_authority.validate_lease(run=run, invocation=invocation, binding=binding)

    def reconcile_death(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        final_sequence: int = 0,
        reason: str = _SAFE_DEATH_REASON,
    ) -> object:
        """Reconcile one process death once; repeated races reuse its receipt."""

        _validate_binding(run, invocation)
        if reason not in _SAFE_DEATH_REASONS:
            reason = _SAFE_DEATH_REASON
        key = (run.run_id, invocation.invocation_id)
        connection = _retention_transaction(self)
        try:
            self._reconcile_death_locked(
                connection,
                run,
                invocation,
                final_sequence=final_sequence,
                reason=reason,
            )
            connection.commit()
            row = connection.execute(
                "SELECT has_receipt, payload, payload_digest FROM supervisor_death_receipts "
                "WHERE run_id = ? AND invocation_id = ?",
                key,
            ).fetchone()
            if row is None or row[0] == 0 or row[1] is None or row[2] is None:
                return None
            payload = bytes(row[1])
            _verify_integrity(payload, bytes(row[2]))
            raw = json.loads(payload)
            if (
                type(raw) is not dict
                or canonical_json_bytes(raw) != payload
                or raw.get("runId") != key[0]
                or raw.get("invocationId") != key[1]
            ):
                raise ContractError("death receipt canonical payload is invalid")
            return _record_view(raw)
        except Exception:
            connection.rollback()
            return None
        finally:
            connection.close()

    def _reconcile_death_locked(
        self,
        connection: sqlite3.Connection,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        final_sequence: int,
        reason: str,
    ) -> TerminalReconciliationReceipt | None:
        key = (run.run_id, invocation.invocation_id)
        row = connection.execute(
            "SELECT has_receipt, payload, payload_digest FROM supervisor_death_receipts "
            "WHERE run_id = ? AND invocation_id = ?",
            key,
        ).fetchone()
        if row is not None:
            if row[0] == 0 or row[1] is None or row[2] is None:
                return None
            payload = bytes(row[1])
            _verify_integrity(payload, bytes(row[2]))
            return _parse_canonical_contract_payload(
                payload,
                expected_type=_EXACT_TERMINAL_RECEIPT,
                label="death receipt",
            )
        count = connection.execute("SELECT COUNT(*) FROM supervisor_death_receipts").fetchone()
        if count is None or count[0] >= _MAX_RETAINED_INVOCATIONS:
            return None
        try:
            receipt = reconcile_process_death(
                port=self.terminal_port,
                run=run,
                invocation_id=invocation.invocation_id,
                final_sequence=final_sequence,
                reason=reason,
            )
        except Exception:
            receipt = None
        if receipt is None:
            connection.execute(
                "INSERT INTO supervisor_death_receipts "
                "(run_id, invocation_id, payload, payload_digest, has_receipt) "
                "VALUES (?, ?, NULL, NULL, 0)",
                key,
            )
            return None
        payload = _canonical_contract_payload(
            receipt,
            expected_type=_EXACT_TERMINAL_RECEIPT,
            label="death receipt",
        )
        payload_digest = hashlib.sha256(payload).digest()
        connection.execute(
            "INSERT INTO supervisor_death_receipts "
            "(run_id, invocation_id, payload, payload_digest, has_receipt) "
            "VALUES (?, ?, ?, ?, 1)",
            (*key, payload, payload_digest),
        )
        return receipt

    def _result_for_death(
        self,
        connection: sqlite3.Connection,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        name: str,
        cleanup: CleanupReport,
        final_sequence: int,
        reason: str,
        evidence: tuple[str, ...],
        enforcement: EnforcementAttestation | None,
    ) -> tuple[object, ...]:
        receipt = self._reconcile_death_locked(
            connection,
            run,
            invocation,
            final_sequence=final_sequence,
            reason=reason,
        )
        status = "failed" if receipt is not None and receipt.accepted and cleanup.succeeded else "supervisor_action_required"
        codes = list(evidence)
        if receipt is None:
            codes.append("host_reconciliation_missing")
        if not cleanup.succeeded:
            codes.extend(("cleanup_failed", "supervisor_action_required"))
        return _draft_result(
            status=status,
            container_name=name,
            exit=classify_process_death(final_sequence=final_sequence, reason=reason),
            receipt=receipt,
            cleanup=cleanup,
            enforcement=enforcement,
            evidence=tuple(dict.fromkeys(codes)),
        )

    def _result_from_proposal(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        name: str,
        parsed: _ParsedOutput,
        cleanup: CleanupReport,
        enforcement: EnforcementAttestation,
    ) -> InvocationResult:
        if parsed.proposal.kind == "cancelled":
            return _draft_result(
                status="rejected",
                container_name=name,
                exit=parsed.exit,
                proposal=parsed.proposal,
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=("child_cancellation_untrusted", "supervisor_action_required"),
            )
        try:
            receipt = reconcile_terminal_proposal(port=self.terminal_port, proposal=parsed.proposal)
        except Exception:
            return _draft_result(
                status="supervisor_action_required",
                container_name=name,
                exit=parsed.exit,
                proposal=parsed.proposal,
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=("host_reconciliation_failed", "supervisor_action_required"),
            )
        if not receipt.accepted or not receipt.legal_transition:
            return _draft_result(
                status="supervisor_action_required",
                container_name=name,
                exit=parsed.exit,
                proposal=parsed.proposal,
                receipt=receipt,
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=("host_reconciliation_rejected", "supervisor_action_required"),
            )
        if not cleanup.succeeded:
            return _draft_result(
                status="supervisor_action_required",
                container_name=name,
                exit=parsed.exit,
                proposal=parsed.proposal,
                receipt=receipt,
                cleanup=cleanup,
                enforcement=enforcement,
                evidence=("cleanup_failed", "supervisor_action_required"),
            )
        return _draft_result(
            status=receipt.kind,
            container_name=name,
            exit=parsed.exit,
            proposal=parsed.proposal,
            receipt=receipt,
            cleanup=cleanup,
            enforcement=enforcement,
            evidence=("host_reconciliation", "enforcement_attested"),
        )

    def run_once(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> object:
        """Launch, bound, reconcile, and dispose of one invocation."""

        _validate_binding(run, invocation)
        _validate_policy(self.policy)
        key = (run.run_id, invocation.invocation_id)
        name = invocation_container_name(run, invocation)
        connection = _retention_transaction(self)
        try:
            def finish(result: object) -> object:
                public = self._retain_result(connection, key, result, name)
                connection.commit()
                return public

            row = connection.execute(
                "SELECT payload FROM supervisor_results WHERE run_id = ? AND invocation_id = ?",
                key,
            ).fetchone()
            if row is not None:
                expected_argv_digest = _argv_digest(build_invocation_argv(run, invocation, self.policy))
                public = self._cached_result(connection, key, name, expected_argv_digest)
                connection.commit()
                return public
            if _retention_count(connection) >= _MAX_RETAINED_INVOCATIONS:
                connection.commit()
                return _result_view(
                    {
                        "run_id": run.run_id,
                        "invocation_id": invocation.invocation_id,
                        "status": "supervisor_action_required",
                        "container_name": name,
                        "exit": None,
                        "proposal": None,
                        "receipt": None,
                        "cleanup": None,
                        "enforcement": None,
                        "evidence": ["result_not_retained", "supervisor_action_required"],
                    }
                )
            try:
                self._validate_host_lease(run, invocation)
            except (BindingError, ContractError, RuntimeConfigurationError):
                return finish(
                    _draft_result(
                        "rejected", name, evidence=("lease_rejected", "supervisor_action_required")
                    )
                )
            expected_argv_digest = _argv_digest(build_invocation_argv(run, invocation, self.policy))
            signal = cancellation or _NeverCancelled()
            cancelled, signal_error = _safe_cancel_check(signal)
            if cancelled:
                return finish(
                    self._result_for_host_cancellation(
                        run,
                        invocation,
                        name=name,
                        final_sequence=0,
                        cleanup=None,
                        enforcement=None,
                        evidence=("cancellation_signal_unavailable",) if signal_error else ("cancellation_signal",),
                    )
                )
            try:
                argv = build_invocation_argv(run, invocation, self.policy)
                request = _request_bytes(run, invocation, self.policy)
                enforcement = self._attest(argv)
            except (ContractError, BoundsError, RuntimeConfigurationError, BindingError):
                return finish(
                    _draft_result("rejected", name, evidence=("policy_rejected", "enforcement_unproven"))
                )
            try:
                process = self._active_runner().launch(
                    argv,
                    client_env=dict(_DOCKER_CLIENT_ENV),
                    input_bytes=request,
                )
            except Exception:
                cleanup = self._cleanup(name)
                return finish(
                    self._result_for_death(
                        connection,
                        run,
                        invocation,
                        name=name,
                        cleanup=cleanup,
                        final_sequence=0,
                        reason="runtime launch failed before terminal evidence",
                        evidence=("launch_failed",),
                        enforcement=enforcement,
                    )
                )

            deadline = self._clock() + float(self.policy.wall_time_seconds)
            collection_failed = False
            try:
                capture = process.collect(
                    input_bytes=request,
                    deadline=deadline,
                    stdout_limit_bytes=int(self.policy.stdout_limit_bytes),
                    stderr_limit_bytes=int(self.policy.stderr_limit_bytes),
                    is_cancelled=lambda: _safe_cancel_check(signal)[0],
                )
            except Exception:
                capture = ProcessCapture(None, reaped=False)
                collection_failed = True
            cleanup = self._cleanup(name)
            final_sequence = 0
            post_cancelled, post_cancel_error = _safe_cancel_check(signal)
            active_cancellation = not collection_failed and post_cancelled and not post_cancel_error
            if active_cancellation and self.cancellation_authority is not None:
                return finish(
                    self._result_for_host_cancellation(
                        run,
                        invocation,
                        name=name,
                        final_sequence=0,
                        cleanup=cleanup,
                        enforcement=enforcement,
                        evidence=("cancellation_signal",),
                    )
                )
            if collection_failed:
                reason = "runtime output was malformed or lacked terminal evidence"
                evidence = ("collection_failed",)
            elif active_cancellation:
                reason = "runtime invocation was stopped by a cancellation signal"
                evidence = ("cancellation_signal", "cancellation_authority_missing")
            elif not capture.reaped:
                reason = "runtime output was malformed or lacked terminal evidence"
                evidence = ("attach_process_not_reaped", "supervisor_action_required")
            elif (
                capture.output_exceeded
                or len(capture.stdout) > int(self.policy.stdout_limit_bytes)
                or len(capture.stderr) > int(self.policy.stderr_limit_bytes)
            ):
                reason = "runtime output exceeded supervisor bounds"
                evidence = ("output_limit_exceeded",)
            elif capture.timed_out:
                reason = "runtime invocation exceeded its wall-time bound"
                evidence = ("timeout",)
            elif capture.cancelled:
                reason = "runtime invocation was stopped by a cancellation signal"
                evidence = ("cancellation_signal",)
            elif capture.returncode not in (0, None):
                reason = _SAFE_DEATH_REASON
                evidence = ("child_nonzero_exit",)
            else:
                try:
                    parsed = _parse_child_output(capture.stdout, run=run, invocation=invocation, policy=self.policy)
                except ChildCancellationProposalRejected:
                    return finish(
                        _draft_result(
                            status="rejected",
                            container_name=name,
                            cleanup=cleanup,
                            enforcement=enforcement,
                            evidence=("child_cancellation_untrusted", "supervisor_action_required"),
                        )
                    )
                except Exception:
                    reason = "runtime output was malformed or lacked terminal evidence"
                    evidence = ("invalid_child_output",)
                else:
                    final_sequence = parsed.final_sequence
                    if capture.returncode is None:
                        reason = "runtime process died before terminal evidence"
                        evidence = ("child_process_died",)
                    else:
                        return finish(
                            self._result_from_proposal(
                                run,
                                invocation,
                                name=name,
                                parsed=parsed,
                                cleanup=cleanup,
                                enforcement=enforcement,
                            )
                        )
            return finish(
                self._result_for_death(
                    connection,
                    run,
                    invocation,
                    name=name,
                    cleanup=cleanup,
                    final_sequence=final_sequence,
                    reason=reason,
                    evidence=evidence,
                    enforcement=enforcement,
                )
            )
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()


__all__ = [
    "CleanupReport",
    "DockerRunnerCapabilities",
    "EnforcementAttestation",
    "InvocationPolicy",
    "InvocationResult",
    "InvocationSupervisor",
    "ProcessCapture",
    "SubprocessDockerRunner",
    "build_invocation_argv",
    "build_invocation_env",
    "invocation_container_name",
]
