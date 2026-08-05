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
import hmac
import json
import math
import os
import re
import secrets
import selectors
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
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


# The retention authority is deliberately launched through one exact, local
# source contract.  These values are captured at module import so a later
# replacement of the executable, module name, source, or Popen seam fails
# closed before a child is launched.
_RETENTION_MODULE_NAME = __name__
_RETENTION_SOURCE_PATH = os.path.realpath(__file__)
try:
    with open(_RETENTION_SOURCE_PATH, "rb") as _retention_source_file:
        _RETENTION_SOURCE_DIGEST = hashlib.sha256(_retention_source_file.read()).hexdigest()
except OSError as exc:  # pragma: no cover - import cannot safely continue
    raise RuntimeConfigurationError("retention authority source is unavailable") from exc
_RETENTION_EXECUTABLE = os.path.realpath(sys.executable)
_RETENTION_POPEN = subprocess.Popen


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


_RETENTION_PROTOCOL = "plane.agent-runtime/retention/v1"
_RETENTION_REQUEST_LIMIT = 512 * 1024
_RETENTION_RESPONSE_LIMIT = 512 * 1024
_RETENTION_REQUEST_TIMEOUT_SECONDS = 5.0
_RETENTION_CLOSE_TIMEOUT_SECONDS = 2.0
_RETENTION_SECRET_LENGTH = 32
_RETENTION_OPERATIONS = frozenset({"create", "read", "close"})
_RETENTION_RECORDS = frozenset({"result", "death"})
_RETENTION_READ_RECORDS = frozenset({"result", "death", "result_count", "death_count"})
_RETENTION_STATUSES = frozenset({"created", "replayed", "conflict", "missing", "ok", "closed"})
_RETENTION_AUTHENTICATION_LABEL = b"plane-agent-retention-session/v2"


@dataclass(frozen=True)
class _RetentionDependencyBundle:
    """Immutable import-time dependencies for the parent authority path."""

    module: object
    subprocess_module: object
    sys_module: object
    module_name: str
    source_path: str
    source_digest: str
    executable: str
    popen: type
    pipe: object
    devnull: object
    protocol: str
    request_limit: int
    response_limit: int
    request_timeout: float
    close_timeout: float
    secret_length: int
    operations: frozenset[str]
    records: frozenset[str]
    read_records: frozenset[str]
    statuses: frozenset[str]
    authentication_label: bytes
    max_retained_invocations: int
    max_reference_length: int
    runtime_configuration_error: type[Exception]
    contract_error: type[Exception]
    bounds_error: type[Exception]
    os_error: type[Exception]
    value_error: type[Exception]
    type_error: type[Exception]
    attribute_error: type[Exception]
    canonical_json: Callable[..., bytes]
    hmac_new: Callable[..., object]
    sha256: Callable[..., object]
    compare_digest: Callable[..., bool]
    json_loads: Callable[..., object]
    fullmatch: Callable[..., object]
    bytes_fromhex: Callable[[str], bytes]
    realpath: Callable[[str], str]
    open_file: Callable[..., object]
    token_bytes: Callable[[int], bytes]
    getpid: Callable[[], int]
    set_blocking: Callable[[int, bool], None]
    monotonic: Callable[[], float]
    selector_factory: Callable[[], object]
    os_read: Callable[..., bytes]
    os_write: Callable[..., int]
    finalize: Callable[..., object]
    weakref_ref: Callable[..., weakref.ReferenceType[object]]
    make_lock: Callable[[], threading.RLock]
    make_type: Callable[..., type]
    object_new: Callable[..., object]
    type_fn: Callable[[object], type]
    getattr_fn: Callable[..., object]
    getattribute: Callable[[object, str], object]
    session_key: Callable[..., bytes]
    message_mac: Callable[..., str]
    record_key: Callable[..., tuple[str, str, str]]
    write_frame: Callable[..., None]
    read_frame: Callable[..., bytes]
    decode_response: Callable[..., dict[str, object]]
    close_process: Callable[..., None]
    validate_spawn_contract: Callable[[], None]
    endpoint_factory: Callable[..., Callable[[str, object | None], object]] | None


def _retention_session_key(
    secret: bytes,
    nonce: bytes,
    source_digest: str,
    parent_pid: int,
    authority_pid: int,
    _hmac_new=hmac.new,
    _hash=hashlib.sha256,
) -> bytes:
    context = (
        _RETENTION_AUTHENTICATION_LABEL
        + b"\x00"
        + _RETENTION_PROTOCOL.encode("ascii")
        + b"\x00"
        + nonce
        + b"\x00"
        + source_digest.encode("ascii")
        + parent_pid.to_bytes(8, "big", signed=False)
        + authority_pid.to_bytes(8, "big", signed=False)
    )
    return _hmac_new(
        secret,
        context,
        _hash,
    ).digest()


def _retention_message_mac(
    key: bytes,
    message: Mapping[str, object],
    _hmac_new=hmac.new,
    _hash=hashlib.sha256,
    _canonical_json_bytes=canonical_json_bytes,
) -> str:
    return _hmac_new(key, _canonical_json_bytes(message), _hash).hexdigest()


def _retention_record_key(record: str, run_id: str, invocation_id: str) -> tuple[str, str, str]:
    if record not in _RETENTION_RECORDS:
        raise ContractError("retention record kind is invalid")
    if (
        type(run_id) is not str
        or not run_id
        or len(run_id) > MAX_REFERENCE_LENGTH
        or type(invocation_id) is not str
        or not invocation_id
        or len(invocation_id) > MAX_REFERENCE_LENGTH
    ):
        raise ContractError("retention record binding is invalid")
    return record, run_id, invocation_id


def _retention_authority_response(
    *,
    status: str,
    record: str,
    request_id: int,
    run_id: str = "",
    invocation_id: str = "",
    value: object | None = None,
) -> dict[str, object]:
    response: dict[str, object] = {
        "protocol": _RETENTION_PROTOCOL,
        "status": status,
        "record": record,
        "requestId": request_id,
        "runId": run_id,
        "invocationId": invocation_id,
    }
    if value is not None:
        response["value"] = value
    return response


def _retention_authority_main() -> None:
    """Own canonical retention in a process the supervisor cannot inspect."""

    secret_buffer = bytearray()
    while len(secret_buffer) < _RETENTION_SECRET_LENGTH:
        chunk = sys.stdin.buffer.read(_RETENTION_SECRET_LENGTH - len(secret_buffer))
        if not chunk:
            return
        secret_buffer.extend(chunk)
    secret = bytes(secret_buffer)
    records: dict[tuple[str, str, str], tuple[bytes, bytes]] = {}
    authenticated = False
    session_key: bytes | None = None

    def integrity(payload: bytes) -> bytes:
        return hmac.new(secret, payload, hashlib.sha256).digest()

    def emit(response: dict[str, object], key: bytes | None) -> None:
        if key is not None:
            response = dict(response)
            response["mac"] = _retention_message_mac(key, response)
        raw = canonical_json_bytes(response) + b"\n"
        if len(raw) > _RETENTION_RESPONSE_LIMIT:
            raise RuntimeConfigurationError("retention authority response exceeded its bound")
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()

    while True:
        raw_line = sys.stdin.buffer.readline(_RETENTION_REQUEST_LIMIT + 1)
        if not raw_line:
            return
        if not raw_line or len(raw_line) > _RETENTION_REQUEST_LIMIT or not raw_line.endswith(b"\n"):
            return
        request_id: object = 0
        record: object = None
        run_id: object = ""
        invocation_id: object = ""
        try:
            request = json.loads(raw_line[:-1])
            if type(request) is not dict or request.get("protocol") != _RETENTION_PROTOCOL:
                raise ContractError("retention request protocol is invalid")
            request_id = request.get("requestId")
            if type(request_id) is not int or isinstance(request_id, bool) or request_id < 1:
                raise ContractError("retention request id is invalid")
            if not authenticated:
                if set(request) != {
                    "protocol", "op", "requestId", "nonce", "sourceDigest", "parentPid", "authorityPid"
                }:
                    raise ContractError("retention authentication request has unexpected fields")
                nonce_hex = request.get("nonce")
                source_digest = request.get("sourceDigest")
                parent_pid = request.get("parentPid")
                authority_pid = request.get("authorityPid")
                if (
                    type(nonce_hex) is not str
                    or len(nonce_hex) != 64
                    or re.fullmatch(r"[0-9a-f]{64}", nonce_hex) is None
                    or source_digest != _RETENTION_SOURCE_DIGEST
                    or type(parent_pid) is not int
                    or isinstance(parent_pid, bool)
                    or type(authority_pid) is not int
                    or isinstance(authority_pid, bool)
                    or parent_pid != os.getppid()
                    or authority_pid != os.getpid()
                ):
                    return
                nonce = bytes.fromhex(nonce_hex)
                session_key = _retention_session_key(
                    secret,
                    nonce,
                    source_digest,
                    parent_pid,
                    authority_pid,
                )
                emit(
                    {
                        **_retention_authority_response(
                            status="ready",
                            record="auth",
                            request_id=request_id,
                        ),
                        "proof": hmac.new(session_key, b"proof:" + nonce, hashlib.sha256).hexdigest(),
                        "sourceDigest": _RETENTION_SOURCE_DIGEST,
                        "sourcePath": _RETENTION_SOURCE_PATH,
                        "parentPid": parent_pid,
                        "authorityPid": authority_pid,
                    },
                    session_key,
                )
                authenticated = True
                continue
            if session_key is None:
                return
            auth = request.get("auth")
            unsigned_request = dict(request)
            unsigned_request.pop("auth", None)
            if (
                type(auth) is not str
                or not hmac.compare_digest(auth, _retention_message_mac(session_key, unsigned_request))
            ):
                return
            operation = request.get("op")
            record = request.get("record")
            run_id = request.get("runId", "")
            invocation_id = request.get("invocationId", "")
            if operation not in _RETENTION_OPERATIONS:
                raise ContractError("retention request operation is invalid")
            if operation == "close":
                if set(request) != {"protocol", "op", "requestId", "auth"}:
                    raise ContractError("retention close request has unexpected fields")
                emit(
                    _retention_authority_response(
                        status="closed",
                        record="close",
                        request_id=request_id,
                    ),
                    session_key,
                )
                return
            if type(record) is not str:
                raise ContractError("retention request record is invalid")
            if operation == "read":
                if record in {"result_count", "death_count"}:
                    if set(request) != {"protocol", "op", "record", "requestId", "auth"}:
                        raise ContractError("retention count request has unexpected fields")
                    emit(
                        _retention_authority_response(
                            status="ok",
                            record=record,
                            request_id=request_id,
                            value=sum(key[0] == record[:-6] for key in records),
                        ),
                        session_key,
                    )
                    continue
                key = _retention_record_key(record, run_id, invocation_id)
                if set(request) != {
                    "protocol", "op", "record", "requestId", "runId", "invocationId", "auth"
                }:
                    raise ContractError("retention read request has unexpected fields")
                stored = records.get(key)
                if stored is None:
                    emit(
                        _retention_authority_response(
                            status="missing",
                            record=record,
                            request_id=request_id,
                            run_id=run_id,
                            invocation_id=invocation_id,
                        ),
                        session_key,
                    )
                    continue
                payload, expected_digest = stored
                if not hmac.compare_digest(integrity(payload), expected_digest):
                    return
                emit(
                    _retention_authority_response(
                        status="ok",
                        record=record,
                        request_id=request_id,
                        run_id=run_id,
                        invocation_id=invocation_id,
                        value=json.loads(payload),
                    ),
                    session_key,
                )
                continue
            key = _retention_record_key(record, run_id, invocation_id)
            if set(request) != {
                "protocol", "op", "record", "requestId", "runId", "invocationId", "value", "auth"
            }:
                raise ContractError("retention create request has unexpected fields")
            value = request["value"]
            if type(value) is not dict:
                raise ContractError("retention create value is invalid")
            payload = canonical_json_bytes(value)
            if record == "result":
                canonical = _validated_canonical_payload(payload)
                if canonical["run_id"] != run_id or canonical["invocation_id"] != invocation_id:
                    raise BindingError("retention result is bound to a different invocation")
            else:
                receipt = _parse_canonical_contract_payload(
                    payload,
                    expected_type=_EXACT_TERMINAL_RECEIPT,
                    label="retention death receipt",
                )
                if receipt.run_id != run_id or receipt.invocation_id != invocation_id:
                    raise BindingError("retention death receipt is bound to a different invocation")
            existing = records.get(key)
            if existing is not None:
                existing_payload, expected_digest = existing
                if not hmac.compare_digest(integrity(existing_payload), expected_digest):
                    return
                if existing_payload != payload:
                    emit(
                        _retention_authority_response(
                            status="conflict",
                            record=record,
                            request_id=request_id,
                            run_id=run_id,
                            invocation_id=invocation_id,
                        ),
                        session_key,
                    )
                else:
                    emit(
                        _retention_authority_response(
                            status="replayed",
                            record=record,
                            request_id=request_id,
                            run_id=run_id,
                            invocation_id=invocation_id,
                            value=json.loads(existing_payload),
                        ),
                        session_key,
                    )
                continue
            if record == "result" and sum(key[0] == "result" for key in records) >= _MAX_RETAINED_INVOCATIONS:
                emit(
                    _retention_authority_response(
                        status="conflict",
                        record=record,
                        request_id=request_id,
                        run_id=run_id,
                        invocation_id=invocation_id,
                    ),
                    session_key,
                )
                continue
            records[key] = (payload, integrity(payload))
            emit(
                _retention_authority_response(
                    status="created",
                    record=record,
                    request_id=request_id,
                    run_id=run_id,
                    invocation_id=invocation_id,
                    value=json.loads(payload),
                ),
                session_key,
            )
        except Exception:
            try:
                response_record = record if record in _RETENTION_RECORDS else "result"
                response_run_id = run_id if type(run_id) is str else ""
                response_invocation_id = invocation_id if type(invocation_id) is str else ""
                if authenticated and session_key is not None and isinstance(request_id, int):
                    emit(
                        _retention_authority_response(
                            status="conflict",
                            record=response_record,
                            request_id=request_id,
                            run_id=response_run_id,
                            invocation_id=response_invocation_id,
                        ),
                        session_key,
                    )
            except Exception:
                return


def _retention_write_frame(
    fd: int,
    payload: bytes,
    deadline: float,
    _selector_factory=selectors.DefaultSelector,
    _write=os.write,
    _monotonic=time.monotonic,
) -> None:
    if not payload or len(payload) > _RETENTION_REQUEST_LIMIT:
        raise BoundsError("canonical retention request exceeds its bound")
    pending = memoryview(payload)
    selector = _selector_factory()
    try:
        selector.register(fd, selectors.EVENT_WRITE)
        while pending:
            remaining = deadline - _monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise RuntimeConfigurationError("canonical retention authority timed out")
            try:
                written = _write(fd, pending[:16 * 1024])
            except BlockingIOError:
                continue
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeConfigurationError("canonical retention authority channel failed") from exc
            if written <= 0:
                raise RuntimeConfigurationError("canonical retention authority channel failed")
            pending = pending[written:]
    finally:
        selector.close()


def _retention_read_frame(
    fd: int,
    deadline: float,
    limit: int,
    _selector_factory=selectors.DefaultSelector,
    _read=os.read,
    _monotonic=time.monotonic,
) -> bytes:
    if limit < 1:
        raise BoundsError("canonical retention response bound is invalid")
    frame = bytearray()
    selector = _selector_factory()
    try:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise RuntimeConfigurationError("canonical retention authority timed out")
            try:
                chunk = _read(fd, min(16 * 1024, limit - len(frame) + 1))
            except BlockingIOError:
                continue
            except OSError as exc:
                raise RuntimeConfigurationError("canonical retention authority channel failed") from exc
            if not chunk:
                raise RuntimeConfigurationError("canonical retention authority returned EOF")
            newline = chunk.find(b"\n")
            if newline >= 0:
                if newline != len(chunk) - 1:
                    raise RuntimeConfigurationError("canonical retention authority returned multiple frames")
                frame.extend(chunk[:newline + 1])
                if len(frame) > limit:
                    raise BoundsError("canonical retention response exceeds its bound")
                return bytes(frame)
            frame.extend(chunk)
            if len(frame) > limit:
                raise BoundsError("canonical retention response exceeds its bound")

    finally:
        selector.close()


def _retention_decode_response(
    raw: bytes,
    *,
    request: Mapping[str, object],
    request_id: int,
    session_key: bytes,
    _message_mac=_retention_message_mac,
    _compare_digest=hmac.compare_digest,
    _json_loads=json.loads,
) -> dict[str, object]:
    if not raw or len(raw) > _RETENTION_RESPONSE_LIMIT or not raw.endswith(b"\n"):
        raise RuntimeConfigurationError("canonical retention authority returned an invalid response")
    if raw[-2:-1] == b"\r":
        raise ContractError("canonical retention authority returned an invalid line terminator")
    try:
        response = _json_loads(raw[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigurationError("canonical retention authority returned malformed JSON") from exc
    if type(response) is not dict:
        raise RuntimeConfigurationError("canonical retention authority returned a non-object response")
    mac = response.get("mac")
    unsigned = dict(response)
    unsigned.pop("mac", None)
    if type(mac) is not str or not _compare_digest(mac, _message_mac(session_key, unsigned)):
        raise RuntimeConfigurationError("canonical retention authority response authentication failed")
    operation = request.get("op")
    if operation == "authenticate":
        expected_fields = {
            "protocol", "status", "record", "requestId", "runId", "invocationId",
            "proof", "sourceDigest", "sourcePath", "parentPid", "authorityPid", "mac",
        }
        if set(response) != expected_fields or response.get("status") != "ready" or response.get("record") != "auth":
            raise RuntimeConfigurationError("canonical retention authority authentication response is invalid")
        if (
            response.get("parentPid") != request.get("parentPid")
            or response.get("authorityPid") != request.get("authorityPid")
        ):
            raise RuntimeConfigurationError("canonical retention authority identity is not request-bound")
    else:
        allowed_fields = {
            "protocol", "status", "record", "requestId", "runId", "invocationId", "mac", "value",
        }
        if set(response) - allowed_fields:
            raise RuntimeConfigurationError("canonical retention authority response has unexpected fields")
        if response.get("status") not in _RETENTION_STATUSES:
            raise RuntimeConfigurationError("canonical retention authority response has an invalid status")
        record = response.get("record")
        expected_record = request.get("record") if operation != "close" else "close"
        allowed_records = (*_RETENTION_RECORDS, "result_count", "death_count", "close")
        if record not in allowed_records or record != expected_record:
            raise RuntimeConfigurationError("canonical retention authority response is not request-bound")
    if response.get("protocol") != _RETENTION_PROTOCOL or response.get("requestId") != request_id:
        raise RuntimeConfigurationError("canonical retention authority response is not request-bound")
    if response.get("runId") != request.get("runId", "") or response.get("invocationId") != request.get("invocationId", ""):
        raise RuntimeConfigurationError("canonical retention authority response is not request-bound")
    return response


def _close_retention_process(process: object, *original_streams: object) -> None:
    """Best-effort bounded cleanup using only the original process object."""

    try:
        poll = getattr(process, "poll")
        wait = getattr(process, "wait")
        if poll() is None:
            try:
                getattr(process, "terminate")()
            except Exception:
                pass
            try:
                wait(timeout=_RETENTION_CLOSE_TIMEOUT_SECONDS)
            except Exception:
                try:
                    getattr(process, "kill")()
                except Exception:
                    pass
                try:
                    wait(timeout=_RETENTION_CLOSE_TIMEOUT_SECONDS)
                except Exception:
                    pass
        else:
            try:
                wait(timeout=_RETENTION_CLOSE_TIMEOUT_SECONDS)
            except Exception:
                pass
    except Exception:
        pass
    streams = original_streams
    if not streams:
        streams = tuple(
            getattr(process, stream_name, None)
            for stream_name in ("stdin", "stdout")
        )
    for stream in streams:
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass


def _make_retention_endpoint(
    authority_ref: weakref.ReferenceType[object],
    process: subprocess.Popen[bytes],
    stdin: object,
    stdout: object,
    stdin_fd: int,
    stdout_fd: int,
    stdin_type: type,
    stdout_type: type,
    lock: threading.RLock,
    secret: bytes,
    source_digest: str,
    source_path: str,
    parent_pid: int,
    authority_pid: int,
    *,
    trusted_session_key: Callable[..., bytes],
    trusted_popen_type: type,
    trusted_write_frame: Callable[..., None],
    trusted_read_frame: Callable[..., bytes],
    trusted_decode_response: Callable[..., dict[str, object]],
    trusted_message_mac: Callable[..., str],
    trusted_canonical_json_bytes: Callable[..., bytes],
    trusted_monotonic: Callable[[], float],
    trusted_close_process: Callable[..., None],
    trusted_hmac_new: Callable[..., object],
    trusted_sha256: Callable[..., object],
    trusted_fullmatch: Callable[..., object],
    trusted_bytes_fromhex: Callable[[str], bytes],
    trusted_type: Callable[[object], type],
    trusted_getattribute: Callable[[object, str], object],
    trusted_runtime_error: type[Exception],
    trusted_contract_error: type[Exception],
    trusted_attribute_error: type[Exception],
    protocol: str,
    response_limit: int,
    request_timeout: float,
    close_timeout: float,
) -> Callable[[str, object | None], object]:
    """Capture original transport resources outside replaceable attributes."""

    process_poll = process.poll
    process_wait = process.wait
    process_terminate = process.terminate
    process_kill = process.kill
    session_key: bytes | None = None
    authenticated = False
    closed = False
    next_request_id = 1

    def handles_are_intact() -> None:
        owner = authority_ref()
        if owner is None:
            raise trusted_runtime_error("canonical retention authority is unavailable")
        expected = {
            "_process": process,
            "_process_identity": process,
            "_stdin": stdin,
            "_stdin_identity": stdin,
            "_stdout": stdout,
            "_stdout_identity": stdout,
            "_stdin_fd": stdin_fd,
            "_stdout_fd": stdout_fd,
            "_stdin_type": stdin_type,
            "_stdout_type": stdout_type,
            "_lock": lock,
        }
        try:
            for name, value in expected.items():
                if trusted_getattribute(owner, name) is not value:
                    raise trusted_runtime_error("canonical retention authority handle was replaced")
            if trusted_getattribute(owner, "_closed"):
                raise trusted_runtime_error("canonical retention authority is closed")
            if (
                trusted_type(process) is not trusted_popen_type
                or process.poll != process_poll
                or process_poll() is not None
            ):
                raise trusted_runtime_error("canonical retention authority is unavailable")
            if (
                process.stdin is not stdin
                or process.stdout is not stdout
                or trusted_type(stdin) is not stdin_type
                or trusted_type(stdout) is not stdout_type
                or stdin.fileno() != stdin_fd
                or stdout.fileno() != stdout_fd
            ):
                raise trusted_runtime_error("canonical retention authority channel was replaced")
        except trusted_attribute_error as exc:
            raise trusted_runtime_error("canonical retention authority handle is unavailable") from exc

    def invalidate() -> None:
        nonlocal closed
        closed = True
        try:
            stdin.close()
        except Exception:
            pass
        trusted_close_process(process, stdin, stdout)
        try:
            stdout.close()
        except Exception:
            pass

    def send(request: Mapping[str, object], *, timeout: float, check_handles: bool) -> dict[str, object]:
        nonlocal next_request_id
        if closed:
            raise trusted_runtime_error("canonical retention authority is closed")
        if check_handles:
            handles_are_intact()
        request_id = next_request_id
        next_request_id += 1
        outbound = dict(request)
        if "requestId" in outbound:
            raise trusted_contract_error("retention request id is parent-owned")
        outbound["requestId"] = request_id
        if authenticated:
            if session_key is None:
                raise trusted_runtime_error("canonical retention authority is unauthenticated")
            outbound["auth"] = trusted_message_mac(session_key, outbound)
        raw = trusted_canonical_json_bytes(outbound) + b"\n"
        deadline = trusted_monotonic() + timeout
        try:
            trusted_write_frame(stdin_fd, raw, deadline)
            response_raw = trusted_read_frame(stdout_fd, deadline, response_limit)
            if session_key is None:
                raise trusted_runtime_error("canonical retention authority is unauthenticated")
            return trusted_decode_response(
                response_raw,
                request=outbound,
                request_id=request_id,
                session_key=session_key,
            )
        except Exception:
            invalidate()
            raise

    def close() -> None:
        nonlocal closed
        with lock:
            if closed:
                return
            closed = True
            try:
                if process_poll() is None and authenticated and session_key is not None:
                    closed = False
                    response = send(
                        {"protocol": protocol, "op": "close"},
                        timeout=close_timeout,
                        check_handles=False,
                    )
                    if response.get("status") != "closed":
                        raise trusted_runtime_error("retention close was not acknowledged")
                    closed = True
                    process_wait(timeout=close_timeout)
            except Exception:
                closed = True
            finally:
                try:
                    stdin.close()
                except Exception:
                    pass
                try:
                    if process_poll() is None:
                        process_wait(timeout=close_timeout)
                except Exception:
                    pass
                if process_poll() is None:
                    try:
                        process_terminate()
                    except Exception:
                        pass
                    try:
                        process_wait(timeout=close_timeout)
                    except Exception:
                        try:
                            process_kill()
                        except Exception:
                            pass
                        try:
                            process_wait(timeout=close_timeout)
                        except Exception:
                            pass
                try:
                    stdout.close()
                except Exception:
                    pass

    def endpoint(operation: str, argument: object | None = None) -> object:
        nonlocal authenticated, session_key
        if operation == "close":
            close()
            return None
        with lock:
            if closed:
                raise trusted_runtime_error("canonical retention authority is closed")
            if operation == "exclusive":
                handles_are_intact()
                if not callable(argument):
                    raise trusted_contract_error("retention exclusive operation requires a callback")
                return argument()
            if operation != "request" or trusted_type(argument) is not dict:
                raise trusted_contract_error("retention authority operation is invalid")
            request = dict(argument)
            if request.get("op") == "authenticate":
                if authenticated or set(request) != {"protocol", "op", "nonce", "sourceDigest"}:
                    raise trusted_contract_error("retention authentication request is invalid")
                nonce_hex = request.get("nonce")
                if trusted_type(nonce_hex) is not str or trusted_fullmatch(r"[0-9a-f]{64}", nonce_hex) is None:
                    raise trusted_contract_error("retention authentication nonce is invalid")
                nonce = trusted_bytes_fromhex(nonce_hex)
                response_key = trusted_session_key(
                    secret,
                    nonce,
                    source_digest,
                    parent_pid,
                    authority_pid,
                )
                request_key = session_key
                session_key = response_key
                try:
                    request["parentPid"] = parent_pid
                    request["authorityPid"] = authority_pid
                    response = send(request, timeout=request_timeout, check_handles=True)
                    expected_proof = trusted_hmac_new(response_key, b"proof:" + nonce, trusted_sha256).hexdigest()
                    if (
                        response.get("proof") != expected_proof
                        or response.get("sourceDigest") != source_digest
                        or response.get("sourcePath") != source_path
                        or response.get("parentPid") != parent_pid
                        or response.get("authorityPid") != authority_pid
                    ):
                        raise trusted_runtime_error("canonical retention authority identity is invalid")
                except Exception:
                    session_key = request_key
                    raise
                authenticated = True
                return response
            if not authenticated or session_key is None:
                raise trusted_runtime_error("canonical retention authority is unauthenticated")
            return send(request, timeout=request_timeout, check_handles=True)

    return endpoint


def _make_retention_dependency_bundle() -> _RetentionDependencyBundle:
    """Capture the complete parent authority dependency graph once at import."""

    trusted_module = sys.modules[__name__]
    trusted_subprocess_module = subprocess
    trusted_sys_module = sys
    trusted_popen = trusted_subprocess_module.Popen
    trusted_event_read = selectors.EVENT_READ
    trusted_event_write = selectors.EVENT_WRITE
    module_name = _RETENTION_MODULE_NAME
    source_path = _RETENTION_SOURCE_PATH
    source_digest = _RETENTION_SOURCE_DIGEST
    executable = _RETENTION_EXECUTABLE
    protocol = _RETENTION_PROTOCOL
    request_limit = _RETENTION_REQUEST_LIMIT
    response_limit = _RETENTION_RESPONSE_LIMIT
    request_timeout = _RETENTION_REQUEST_TIMEOUT_SECONDS
    close_timeout = _RETENTION_CLOSE_TIMEOUT_SECONDS
    secret_length = _RETENTION_SECRET_LENGTH
    operations = _RETENTION_OPERATIONS
    records = _RETENTION_RECORDS
    read_records = _RETENTION_READ_RECORDS
    statuses = _RETENTION_STATUSES
    authentication_label = _RETENTION_AUTHENTICATION_LABEL
    max_retained_invocations = _MAX_RETAINED_INVOCATIONS
    max_reference_length = MAX_REFERENCE_LENGTH

    trusted_runtime_error = RuntimeConfigurationError
    trusted_contract_error = ContractError
    trusted_bounds_error = BoundsError
    trusted_os_error = OSError
    trusted_value_error = ValueError
    trusted_type_error = TypeError
    trusted_attribute_error = AttributeError
    trusted_canonical_json = canonical_json_bytes
    trusted_hmac_new = hmac.new
    trusted_sha256 = hashlib.sha256
    trusted_compare_digest = hmac.compare_digest
    trusted_json_loads = json.loads
    trusted_json_decode_error = json.JSONDecodeError
    trusted_fullmatch = re.fullmatch
    trusted_bytes_fromhex = bytes.fromhex
    trusted_realpath = os.path.realpath
    trusted_open_file = open
    trusted_token_bytes = secrets.token_bytes
    trusted_getpid = os.getpid
    trusted_set_blocking = os.set_blocking
    trusted_monotonic = time.monotonic
    trusted_selector_factory = selectors.DefaultSelector
    trusted_os_read = os.read
    trusted_os_write = os.write
    trusted_finalize = weakref.finalize
    trusted_weakref_ref = weakref.ref
    trusted_make_lock = threading.RLock
    trusted_make_type = type
    trusted_object_new = object.__new__
    trusted_type = type
    trusted_getattr = getattr
    trusted_getattribute = object.__getattribute__
    trusted_make_endpoint = _make_retention_endpoint

    def session_key(
        secret: bytes,
        nonce: bytes,
        authority_source_digest: str,
        parent_pid: int,
        authority_pid: int,
    ) -> bytes:
        context = (
            authentication_label
            + b"\x00"
            + protocol.encode("ascii")
            + b"\x00"
            + nonce
            + b"\x00"
            + authority_source_digest.encode("ascii")
            + parent_pid.to_bytes(8, "big", signed=False)
            + authority_pid.to_bytes(8, "big", signed=False)
        )
        return trusted_hmac_new(secret, context, trusted_sha256).digest()

    def message_mac(key: bytes, message: Mapping[str, object]) -> str:
        return trusted_hmac_new(key, trusted_canonical_json(message), trusted_sha256).hexdigest()

    def record_key(record: str, run_id: str, invocation_id: str) -> tuple[str, str, str]:
        if record not in records:
            raise trusted_contract_error("retention record kind is invalid")
        if (
            trusted_type(run_id) is not str
            or not run_id
            or len(run_id) > max_reference_length
            or trusted_type(invocation_id) is not str
            or not invocation_id
            or len(invocation_id) > max_reference_length
        ):
            raise trusted_contract_error("retention record binding is invalid")
        return record, run_id, invocation_id

    def write_frame(fd: int, payload: bytes, deadline: float) -> None:
        if not payload or len(payload) > request_limit:
            raise trusted_bounds_error("canonical retention request exceeds its bound")
        pending = memoryview(payload)
        selector = trusted_selector_factory()
        try:
            selector.register(fd, trusted_event_write)
            while pending:
                remaining = deadline - trusted_monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise trusted_runtime_error("canonical retention authority timed out")
                try:
                    written = trusted_os_write(fd, pending[:16 * 1024])
                except BlockingIOError:
                    continue
                except (BrokenPipeError, trusted_os_error) as exc:
                    raise trusted_runtime_error("canonical retention authority channel failed") from exc
                if written <= 0:
                    raise trusted_runtime_error("canonical retention authority channel failed")
                pending = pending[written:]
        finally:
            selector.close()

    def read_frame(fd: int, deadline: float, limit: int) -> bytes:
        if limit < 1:
            raise trusted_bounds_error("canonical retention response bound is invalid")
        frame = bytearray()
        selector = trusted_selector_factory()
        try:
            selector.register(fd, trusted_event_read)
            while True:
                remaining = deadline - trusted_monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise trusted_runtime_error("canonical retention authority timed out")
                try:
                    chunk = trusted_os_read(fd, min(16 * 1024, limit - len(frame) + 1))
                except BlockingIOError:
                    continue
                except trusted_os_error as exc:
                    raise trusted_runtime_error("canonical retention authority channel failed") from exc
                if not chunk:
                    raise trusted_runtime_error("canonical retention authority returned EOF")
                newline = chunk.find(b"\n")
                if newline >= 0:
                    if newline != len(chunk) - 1:
                        raise trusted_runtime_error("canonical retention authority returned multiple frames")
                    frame.extend(chunk[:newline + 1])
                    if len(frame) > limit:
                        raise trusted_bounds_error("canonical retention response exceeds its bound")
                    return bytes(frame)
                frame.extend(chunk)
                if len(frame) > limit:
                    raise trusted_bounds_error("canonical retention response exceeds its bound")
        finally:
            selector.close()

    def decode_response(
        raw: bytes,
        *,
        request: Mapping[str, object],
        request_id: int,
        session_key: bytes,
    ) -> dict[str, object]:
        if not raw or len(raw) > response_limit or not raw.endswith(b"\n"):
            raise trusted_runtime_error("canonical retention authority returned an invalid response")
        if raw[-2:-1] == b"\r":
            raise trusted_contract_error("canonical retention authority returned an invalid line terminator")
        try:
            response = trusted_json_loads(raw[:-1])
        except (UnicodeDecodeError, trusted_json_decode_error) as exc:
            raise trusted_runtime_error("canonical retention authority returned malformed JSON") from exc
        if trusted_type(response) is not dict:
            raise trusted_runtime_error("canonical retention authority returned a non-object response")
        mac = response.get("mac")
        unsigned = dict(response)
        unsigned.pop("mac", None)
        if trusted_type(mac) is not str or not trusted_compare_digest(mac, message_mac(session_key, unsigned)):
            raise trusted_runtime_error("canonical retention authority response authentication failed")
        operation = request.get("op")
        if operation == "authenticate":
            expected_fields = {
                "protocol", "status", "record", "requestId", "runId", "invocationId",
                "proof", "sourceDigest", "sourcePath", "parentPid", "authorityPid", "mac",
            }
            if (
                set(response) != expected_fields
                or response.get("status") != "ready"
                or response.get("record") != "auth"
            ):
                raise trusted_runtime_error("canonical retention authority authentication response is invalid")
            if (
                response.get("parentPid") != request.get("parentPid")
                or response.get("authorityPid") != request.get("authorityPid")
            ):
                raise trusted_runtime_error("canonical retention authority identity is not request-bound")
        else:
            allowed_fields = {
                "protocol", "status", "record", "requestId", "runId", "invocationId", "mac", "value",
            }
            if set(response) - allowed_fields:
                raise trusted_runtime_error("canonical retention authority response has unexpected fields")
            if response.get("status") not in statuses:
                raise trusted_runtime_error("canonical retention authority response has an invalid status")
            record = response.get("record")
            expected_record = request.get("record") if operation != "close" else "close"
            allowed_records = records | {"result_count", "death_count", "close"}
            if record not in allowed_records or record != expected_record:
                raise trusted_runtime_error("canonical retention authority response is not request-bound")
        if response.get("protocol") != protocol or response.get("requestId") != request_id:
            raise trusted_runtime_error("canonical retention authority response is not request-bound")
        if (
            response.get("runId") != request.get("runId", "")
            or response.get("invocationId") != request.get("invocationId", "")
        ):
            raise trusted_runtime_error("canonical retention authority response is not request-bound")
        return response

    def close_process(process: object, *original_streams: object) -> None:
        """Best-effort bounded cleanup using only original process resources."""

        try:
            poll = trusted_getattr(process, "poll")
            wait = trusted_getattr(process, "wait")
            if poll() is None:
                try:
                    trusted_getattr(process, "terminate")()
                except Exception:
                    pass
                try:
                    wait(timeout=close_timeout)
                except Exception:
                    try:
                        trusted_getattr(process, "kill")()
                    except Exception:
                        pass
                    try:
                        wait(timeout=close_timeout)
                    except Exception:
                        pass
            else:
                try:
                    wait(timeout=close_timeout)
                except Exception:
                    pass
        except Exception:
            pass
        streams = original_streams
        if not streams:
            streams = tuple(
                trusted_getattr(process, stream_name, None)
                for stream_name in ("stdin", "stdout")
            )
        for stream in streams:
            try:
                if stream is not None:
                    trusted_getattr(stream, "close")()
            except Exception:
                pass

    def validate_spawn_contract() -> None:
        try:
            if (
                trusted_module.__name__ != module_name
                or trusted_sys_module.modules.get(module_name) is not trusted_module
            ):
                raise trusted_runtime_error("retention authority module identity was replaced")
            if trusted_realpath(trusted_module.__file__) != source_path:
                raise trusted_runtime_error("retention authority source path was replaced")
            with trusted_open_file(source_path, "rb") as source_file:
                current_digest = trusted_sha256(source_file.read()).hexdigest()
            if current_digest != source_digest:
                raise trusted_runtime_error("retention authority source changed after import")
            if trusted_realpath(trusted_sys_module.executable) != executable:
                raise trusted_runtime_error("retention authority executable was replaced")
            if trusted_subprocess_module.Popen is not trusted_popen:
                raise trusted_runtime_error("retention authority spawn helper was replaced")
        except (trusted_os_error, trusted_type_error, trusted_attribute_error) as exc:
            raise trusted_runtime_error("retention authority spawn contract is unavailable") from exc

    bundle: _RetentionDependencyBundle

    def make_endpoint(*args: object, **kwargs: object) -> Callable[[str, object | None], object]:
        return trusted_make_endpoint(
            *args,
            trusted_session_key=session_key,
            trusted_popen_type=trusted_popen,
            trusted_write_frame=write_frame,
            trusted_read_frame=read_frame,
            trusted_decode_response=decode_response,
            trusted_message_mac=message_mac,
            trusted_canonical_json_bytes=trusted_canonical_json,
            trusted_monotonic=trusted_monotonic,
            trusted_close_process=close_process,
            trusted_hmac_new=trusted_hmac_new,
            trusted_sha256=trusted_sha256,
            trusted_fullmatch=trusted_fullmatch,
            trusted_bytes_fromhex=trusted_bytes_fromhex,
            trusted_type=trusted_type,
            trusted_getattribute=trusted_getattribute,
            trusted_runtime_error=trusted_runtime_error,
            trusted_contract_error=trusted_contract_error,
            trusted_attribute_error=trusted_attribute_error,
            protocol=protocol,
            response_limit=response_limit,
            request_timeout=request_timeout,
            close_timeout=close_timeout,
            **kwargs,
        )

    bundle = _RetentionDependencyBundle(
        module=trusted_module,
        subprocess_module=trusted_subprocess_module,
        sys_module=trusted_sys_module,
        module_name=module_name,
        source_path=source_path,
        source_digest=source_digest,
        executable=executable,
        popen=trusted_popen,
        pipe=subprocess.PIPE,
        devnull=subprocess.DEVNULL,
        protocol=protocol,
        request_limit=request_limit,
        response_limit=response_limit,
        request_timeout=request_timeout,
        close_timeout=close_timeout,
        secret_length=secret_length,
        operations=operations,
        records=records,
        read_records=read_records,
        statuses=statuses,
        authentication_label=authentication_label,
        max_retained_invocations=max_retained_invocations,
        max_reference_length=max_reference_length,
        runtime_configuration_error=trusted_runtime_error,
        contract_error=trusted_contract_error,
        bounds_error=trusted_bounds_error,
        os_error=trusted_os_error,
        value_error=trusted_value_error,
        type_error=trusted_type_error,
        attribute_error=trusted_attribute_error,
        canonical_json=trusted_canonical_json,
        hmac_new=trusted_hmac_new,
        sha256=trusted_sha256,
        compare_digest=trusted_compare_digest,
        json_loads=trusted_json_loads,
        fullmatch=trusted_fullmatch,
        bytes_fromhex=trusted_bytes_fromhex,
        realpath=trusted_realpath,
        open_file=trusted_open_file,
        token_bytes=trusted_token_bytes,
        getpid=trusted_getpid,
        set_blocking=trusted_set_blocking,
        monotonic=trusted_monotonic,
        selector_factory=trusted_selector_factory,
        os_read=trusted_os_read,
        os_write=trusted_os_write,
        finalize=trusted_finalize,
        weakref_ref=trusted_weakref_ref,
        make_lock=trusted_make_lock,
        make_type=trusted_make_type,
        object_new=trusted_object_new,
        type_fn=trusted_type,
        getattr_fn=trusted_getattr,
        getattribute=trusted_getattribute,
        session_key=session_key,
        message_mac=message_mac,
        record_key=record_key,
        write_frame=write_frame,
        read_frame=read_frame,
        decode_response=decode_response,
        close_process=close_process,
        validate_spawn_contract=validate_spawn_contract,
        endpoint_factory=make_endpoint,
    )
    return bundle


def _validate_retention_spawn_contract(
    *,
    trusted_popen: type,
    trusted_executable: str,
    trusted_module_name: str,
    trusted_source_path: str,
    trusted_source_digest: str,
) -> None:
    try:
        if __name__ != trusted_module_name or _RETENTION_MODULE_NAME != trusted_module_name:
            raise RuntimeConfigurationError("retention authority module identity was replaced")
        if (
            os.path.realpath(__file__) != trusted_source_path
            or _RETENTION_SOURCE_PATH != trusted_source_path
        ):
            raise RuntimeConfigurationError("retention authority source path was replaced")
        with open(trusted_source_path, "rb") as source_file:
            source_digest = hashlib.sha256(source_file.read()).hexdigest()
        if source_digest != trusted_source_digest or _RETENTION_SOURCE_DIGEST != trusted_source_digest:
            raise RuntimeConfigurationError("retention authority source changed after import")
        if (
            os.path.realpath(sys.executable) != trusted_executable
            or _RETENTION_EXECUTABLE != trusted_executable
        ):
            raise RuntimeConfigurationError("retention authority executable was replaced")
        if subprocess.Popen is not trusted_popen or _RETENTION_POPEN is not trusted_popen:
            raise RuntimeConfigurationError("retention authority spawn helper was replaced")
    except (OSError, TypeError, AttributeError) as exc:
        raise RuntimeConfigurationError("retention authority spawn contract is unavailable") from exc


_RETENTION_DEPENDENCIES = _make_retention_dependency_bundle()


def _make_retention_authority_constructor(
    dependencies: _RetentionDependencyBundle,
) -> tuple[Callable[[type], object], Callable[[type, Callable[..., object]], object]]:
    """Build separate production and test constructors over one immutable bundle."""

    trusted_popen = dependencies.popen
    trusted_executable = dependencies.executable
    trusted_module_name = dependencies.module_name
    trusted_source_path = dependencies.source_path
    trusted_source_digest = dependencies.source_digest
    trusted_validator = dependencies.validate_spawn_contract
    trusted_make_endpoint = dependencies.endpoint_factory
    trusted_token_bytes = dependencies.token_bytes
    trusted_getpid = dependencies.getpid
    trusted_set_blocking = dependencies.set_blocking
    trusted_pipe = dependencies.pipe
    trusted_devnull = dependencies.devnull
    trusted_monotonic = dependencies.monotonic
    trusted_write_frame = dependencies.write_frame
    trusted_close_process = dependencies.close_process
    trusted_finalize = dependencies.finalize
    trusted_weakref_ref = dependencies.weakref_ref
    trusted_make_lock = dependencies.make_lock
    trusted_make_type = dependencies.make_type
    trusted_object_new = dependencies.object_new
    trusted_type = dependencies.type_fn
    trusted_record_key = dependencies.record_key
    trusted_runtime_error = dependencies.runtime_configuration_error
    trusted_os_error = dependencies.os_error
    trusted_value_error = dependencies.value_error
    trusted_secret_length = dependencies.secret_length
    trusted_request_timeout = dependencies.request_timeout
    trusted_protocol = dependencies.protocol

    def construct(cls: type, make_endpoint: Callable[..., object]) -> object:
        trusted_validator()
        command = (
            trusted_executable,
            "-m",
            trusted_module_name,
            "--retention-authority",
        )
        secret = trusted_token_bytes(trusted_secret_length)
        if trusted_type(secret) is not bytes or len(secret) != trusted_secret_length:
            raise trusted_runtime_error("canonical retention authority secret is unavailable")
        try:
            process = trusted_popen(
                command,
                stdin=trusted_pipe,
                stdout=trusted_pipe,
                stderr=trusted_devnull,
                close_fds=True,
            )
        except (trusted_os_error, trusted_value_error) as exc:
            raise trusted_runtime_error("canonical retention authority could not start") from exc
        if trusted_type(process) is not trusted_popen or process.stdin is None or process.stdout is None:
            trusted_close_process(process)
            raise trusted_runtime_error("canonical retention authority has no trusted typed channel")
        try:
            stdin_fd = process.stdin.fileno()
            stdout_fd = process.stdout.fileno()
            stdin_type = trusted_type(process.stdin)
            stdout_type = trusted_type(process.stdout)
            trusted_set_blocking(stdin_fd, False)
            trusted_set_blocking(stdout_fd, False)
        except (trusted_os_error, trusted_value_error) as exc:
            trusted_close_process(process)
            raise trusted_runtime_error("canonical retention authority channel is unavailable") from exc
        try:
            trusted_write_frame(
                stdin_fd,
                secret,
                trusted_monotonic() + trusted_request_timeout,
            )
        except Exception as exc:
            trusted_close_process(process, process.stdin, process.stdout)
            raise trusted_runtime_error("canonical retention authority bootstrap failed") from exc
        authority_pid = process.pid
        parent_pid = trusted_getpid()
        cleanup_cell: list[Callable[[str, object | None], object] | None] = [None]

        def bound_request(self: _RetentionAuthority, request: dict[str, object]) -> dict[str, object]:
            endpoint = cleanup_cell[0]
            if endpoint is None:
                raise trusted_runtime_error("canonical retention authority is unavailable")
            return endpoint("request", request)  # type: ignore[return-value]

        def bound_exclusive(self: _RetentionAuthority, callback: Callable[[], object]) -> object:
            endpoint = cleanup_cell[0]
            if endpoint is None:
                raise trusted_runtime_error("canonical retention authority is unavailable")
            return endpoint("exclusive", callback)

        def bound_close(self: _RetentionAuthority) -> None:
            endpoint = cleanup_cell[0]
            if endpoint is not None:
                try:
                    endpoint("close")
                except Exception:
                    pass

        def bound_create(
            self: _RetentionAuthority,
            record: str,
            key: tuple[str, str],
            value: dict[str, object],
        ) -> dict[str, object]:
            run_id, invocation_id = trusted_record_key(record, *key)[1:]
            return self._request({
                "protocol": trusted_protocol,
                "op": "create",
                "record": record,
                "runId": run_id,
                "invocationId": invocation_id,
                "value": value,
            })

        def bound_read(
            self: _RetentionAuthority,
            record: str,
            key: tuple[str, str],
        ) -> dict[str, object]:
            run_id, invocation_id = trusted_record_key(record, *key)[1:]
            return self._request({
                "protocol": trusted_protocol,
                "op": "read",
                "record": record,
                "runId": run_id,
                "invocationId": invocation_id,
            })

        def bound_count_results(self: _RetentionAuthority) -> int:
            response = self._request({
                "protocol": trusted_protocol,
                "op": "read",
                "record": "result_count",
            })
            value = response.get("value")
            if response.get("status") != "ok" or trusted_type(value) is not int:
                raise trusted_runtime_error("canonical retention count is invalid")
            return value

        def bound_count_death_receipts(self: _RetentionAuthority) -> int:
            response = self._request({
                "protocol": trusted_protocol,
                "op": "read",
                "record": "death_count",
            })
            value = response.get("value")
            if response.get("status") != "ok" or trusted_type(value) is not int:
                raise trusted_runtime_error("canonical death-receipt count is invalid")
            return value

        bound_type = trusted_make_type(
            "_BoundRetentionAuthority",
            (cls,),
            {
                "__slots__": (),
                "_request": bound_request,
                "exclusive": bound_exclusive,
                "close": bound_close,
                "create": bound_create,
                "read": bound_read,
                "count_results": bound_count_results,
                "count_death_receipts": bound_count_death_receipts,
                "__del__": bound_close,
            },
        )
        authority = trusted_object_new(bound_type)
        authority._process = process
        authority._process_identity = process
        authority._stdin = process.stdin
        authority._stdin_identity = process.stdin
        authority._stdout = process.stdout
        authority._stdout_identity = process.stdout
        authority._stdin_fd = stdin_fd
        authority._stdout_fd = stdout_fd
        authority._stdin_type = stdin_type
        authority._stdout_type = stdout_type
        authority._lock = trusted_make_lock()
        authority._closed = False
        try:
            endpoint = make_endpoint(
                trusted_weakref_ref(authority),
                process,
                process.stdin,
                process.stdout,
                stdin_fd,
                stdout_fd,
                stdin_type,
                stdout_type,
                authority._lock,
                secret,
                trusted_source_digest,
                trusted_source_path,
                parent_pid,
                authority_pid,
            )
        except Exception as exc:
            trusted_close_process(process, process.stdin, process.stdout)
            raise trusted_runtime_error("canonical retention authority endpoint is unavailable") from exc
        cleanup_cell[0] = endpoint
        trusted_finalize(authority, endpoint, "close")
        try:
            endpoint(
                "request",
                {
                    "protocol": trusted_protocol,
                    "op": "authenticate",
                    "nonce": trusted_token_bytes(trusted_secret_length).hex(),
                    "sourceDigest": trusted_source_digest,
                },
            )
        except Exception:
            bound_close(authority)
            raise

        return authority

    def production_new(cls: type) -> object:
        if trusted_make_endpoint is None:
            raise trusted_runtime_error("canonical retention authority endpoint is unavailable")
        return construct(cls, trusted_make_endpoint)

    def test_construct(cls: type, make_endpoint: Callable[..., object]) -> object:
        return construct(cls, make_endpoint)

    return production_new, test_construct


_retention_authority_new, _retention_authority_test_constructor = _make_retention_authority_constructor(
    _RETENTION_DEPENDENCIES,
)


class _RetentionAuthority:
    """Typed parent client for the isolated canonical-retention process."""

    __slots__ = (
        "_process",
        "_process_identity",
        "_stdin",
        "_stdin_identity",
        "_stdout",
        "_stdout_identity",
        "_stdin_fd",
        "_stdout_fd",
        "_stdin_type",
        "_stdout_type",
        "_lock",
        "_closed",
        "__weakref__",
    )

    __new__ = _retention_authority_new

    def __init__(self) -> None:
        pass

    def exclusive(self, callback: Callable[[], object]) -> object:
        raise RuntimeConfigurationError("canonical retention authority is unavailable")

    def create(self, record: str, key: tuple[str, str], value: dict[str, object]) -> dict[str, object]:
        run_id, invocation_id = _retention_record_key(record, *key)[1:]
        return self._request({
            "protocol": _RETENTION_PROTOCOL,
            "op": "create",
            "record": record,
            "runId": run_id,
            "invocationId": invocation_id,
            "value": value,
        })

    def read(self, record: str, key: tuple[str, str]) -> dict[str, object]:
        run_id, invocation_id = _retention_record_key(record, *key)[1:]
        return self._request({
            "protocol": _RETENTION_PROTOCOL,
            "op": "read",
            "record": record,
            "runId": run_id,
            "invocationId": invocation_id,
        })

    def count_results(self) -> int:
        response = self._request({
            "protocol": _RETENTION_PROTOCOL,
            "op": "read",
            "record": "result_count",
        })
        value = response.get("value")
        if response.get("status") != "ok" or type(value) is not int:
            raise RuntimeConfigurationError("canonical retention count is invalid")
        return value

    def count_death_receipts(self) -> int:
        response = self._request({
            "protocol": _RETENTION_PROTOCOL,
            "op": "read",
            "record": "death_count",
        })
        value = response.get("value")
        if response.get("status") != "ok" or type(value) is not int:
            raise RuntimeConfigurationError("canonical death-receipt count is invalid")
        return value

    def close(self) -> None:
        raise RuntimeConfigurationError("canonical retention authority is unavailable")


_EXACT_RETENTION_AUTHORITY = _RetentionAuthority


def _make_retention_authority_test_harness(
    authority_type: type,
    test_constructor: Callable[[type, Callable[..., object]], _RetentionAuthority],
) -> Callable[..., _RetentionAuthority]:
    """Keep the injectable endpoint factory behind a test-only closure."""

    trusted_authority_type = authority_type
    trusted_test_constructor = test_constructor

    def make_for_test(*, make_endpoint: Callable[..., object]) -> _RetentionAuthority:
        return trusted_test_constructor(trusted_authority_type, make_endpoint)

    return make_for_test


_retention_authority_test_harness = _make_retention_authority_test_harness(
    _RetentionAuthority,
    _retention_authority_test_constructor,
)


def _make_retention_authority_for_test(*, make_endpoint):
    """Test-only construction seam; production calls cannot select it."""

    return _retention_authority_test_harness(make_endpoint=make_endpoint)


def _make_production_retention_authority_constructor(
    authority_type: type,
    authority_new: Callable[[type], _RetentionAuthority],
) -> Callable[[], _RetentionAuthority]:
    """Capture the production authority type and constructor in one closure."""

    trusted_authority_type = authority_type
    trusted_authority_new = authority_new

    def construct() -> _RetentionAuthority:
        return trusted_authority_new(trusted_authority_type)

    return construct


_RETENTION_AUTHORITY_PRODUCTION_CONSTRUCTOR = _make_production_retention_authority_constructor(
    _RetentionAuthority,
    _retention_authority_new,
)


def _make_invocation_supervisor_initializer(
    production_authority_constructor: Callable[[], _RetentionAuthority],
) -> Callable[..., None]:
    """Bind supervisor construction to the import-time production constructor."""

    trusted_authority_constructor = production_authority_constructor
    trusted_validate_policy = _validate_policy
    trusted_finalize = weakref.finalize
    trusted_clock = time.monotonic

    def initialize(
        self,
        *,
        policy: InvocationPolicy,
        runner: DockerRunner | None = None,
        terminal_port: TerminalReconciliationPort,
        lease_authority: CanonicalLeaseAuthority,
        lease_binding: CanonicalLeaseBinding,
        cancellation_authority: CancellationAuthority | None = None,
        clock: Callable[[], float] = trusted_clock,
    ) -> None:
        trusted_validate_policy(policy)
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
        authority = trusted_authority_constructor()
        self._retention_authority = authority
        # Capture the authority's bound cleanup capability directly.  This is
        # deliberately independent of the public authority handle and every
        # module-level registry/helper, including during GC finalization.
        self._retention_finalizer = trusted_finalize(self, authority.close)

    initialize.__name__ = "__init__"
    initialize.__qualname__ = "InvocationSupervisor.__init__"
    return initialize


class InvocationSupervisor:
    """Run, bound, reconcile, and dispose of exactly one invocation."""

    __init__ = _make_invocation_supervisor_initializer(
        _RETENTION_AUTHORITY_PRODUCTION_CONSTRUCTOR,
    )

    def close(self) -> None:
        """Close the invocation-scoped retention authority process."""

        finalizer = object.__getattribute__(self, "_retention_finalizer")
        finalizer()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _authority(self, _authority_type=_EXACT_RETENTION_AUTHORITY) -> _RetentionAuthority:
        try:
            authority = object.__getattribute__(self, "_retention_authority")
        except AttributeError as exc:
            raise RuntimeConfigurationError("canonical retention authority is unavailable") from exc
        if not isinstance(authority, _authority_type):
            raise RuntimeConfigurationError("canonical retention authority was replaced")
        return authority

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
        key: tuple[str, str],
        result: object,
        name: str,
        expected_argv_digest: str,
    ) -> object:
        """Create one canonical record in the isolated authority."""

        try:
            canonical = _canonicalize_result(key, result)
            value = json.loads(canonical)
            if type(value) is not dict:
                raise ContractError("retained result canonical value is invalid")
            response = self._authority().create("result", key, value)
            if response.get("status") not in {"created", "replayed"}:
                raise ContractError("retained result create was rejected")
            return self._cached_result(key, name, expected_argv_digest)
        except Exception:
            return self._invalid_retained_result(name)

    def _cached_result(
        self,
        key: tuple[str, str],
        name: str,
        expected_argv_digest: str,
    ) -> object:
        """Read and validate a fresh record from the isolated authority."""

        try:
            response = self._authority().read("result", key)
            if response.get("status") != "ok" or type(response.get("value")) is not dict:
                raise KeyError(key)
            payload = canonical_json_bytes(response["value"])
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
        try:
            return self._authority().exclusive(lambda: self._reconcile_death_locked(
                run,
                invocation,
                final_sequence=final_sequence,
                reason=reason,
                public=True,
            ))
        except Exception:
            return None

    def _reconcile_death_locked(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        final_sequence: int,
        reason: str,
        public: bool = False,
    ) -> TerminalReconciliationReceipt | object | None:
        key = (run.run_id, invocation.invocation_id)
        existing = self._authority().read("death", key)
        if existing.get("status") == "ok" and type(existing.get("value")) is dict:
            payload = canonical_json_bytes(existing["value"])
            receipt = _parse_canonical_contract_payload(
                payload,
                expected_type=_EXACT_TERMINAL_RECEIPT,
                label="death receipt",
            )
            return _record_view(existing["value"]) if public else receipt
        if existing.get("status") != "missing":
            return None
        if self._authority().count_death_receipts() >= _MAX_RETAINED_INVOCATIONS:
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
            return None
        payload = _canonical_contract_payload(
            receipt,
            expected_type=_EXACT_TERMINAL_RECEIPT,
            label="death receipt",
        )
        response = self._authority().create("death", key, json.loads(payload))
        if response.get("status") not in {"created", "replayed"} or type(response.get("value")) is not dict:
            return None
        if public:
            return _record_view(response["value"])
        return _parse_canonical_contract_payload(
            canonical_json_bytes(response["value"]),
            expected_type=_EXACT_TERMINAL_RECEIPT,
            label="death receipt",
        )

    def _result_for_death(
        self,
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
        """Run one invocation through the isolated retention authority."""

        try:
            name = invocation_container_name(run, invocation)
            authority = self._authority()
            return authority.exclusive(lambda: self._run_once_locked(
                run,
                invocation,
                cancellation=cancellation,
            ))
        except Exception:
            try:
                return self._invalid_retained_result(invocation_container_name(run, invocation))
            except Exception:
                return self._invalid_retained_result("invalid")

    def _run_once_locked(
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
        def finish(result: object) -> object:
            return self._retain_result(key, result, name, expected_argv_digest)

        authority = self._authority()
        cached = authority.read("result", key)
        if cached.get("status") == "ok":
            expected_argv_digest = _argv_digest(build_invocation_argv(run, invocation, self.policy))
            return self._cached_result(key, name, expected_argv_digest)
        if cached.get("status") != "missing":
            return self._invalid_retained_result(name)
        if authority.count_results() >= _MAX_RETAINED_INVOCATIONS:
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
            expected_argv_digest = _argv_digest(build_invocation_argv(run, invocation, self.policy))
            return finish(_draft_result("rejected", name, evidence=("lease_rejected", "supervisor_action_required")))
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
            return finish(_draft_result("rejected", name, evidence=("policy_rejected", "enforcement_unproven")))
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


if __name__ == "__main__" and sys.argv[1:] == ["--retention-authority"]:
    _retention_authority_main()
