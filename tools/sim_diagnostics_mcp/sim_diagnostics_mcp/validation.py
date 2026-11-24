"""Strict input validation and path containment helpers for read-only diagnostics."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import MAX_TIMEOUT_SEC, MIN_TIMEOUT_SEC

_ROS_NAME_RE = re.compile(r"^/[A-Za-z0-9_][A-Za-z0-9_/]{0,254}$")
_NAMESPACE_RE = re.compile(r"^/?[A-Za-z][A-Za-z0-9_-]{0,63}$")
_GZ_WORLD_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_GZ_PARTITION_RE = re.compile(r"^[A-Za-z0-9_.-]{0,64}$")
_GZ_TOPIC_RE = re.compile(r"^/[A-Za-z0-9_][A-Za-z0-9_./-]{0,254}$")

MAX_INSTANCE_ID = 63
MAX_TAIL_LINES = 500
MAX_LIST_LIMIT = 200
MAX_DRONE_FILTER_COUNT = 64


class ValidationError(ValueError):
    """Raised when a tool argument fails strict safety validation."""


def validate_timeout(
    timeout_sec: float,
    *,
    minimum: float = MIN_TIMEOUT_SEC,
    maximum: float = MAX_TIMEOUT_SEC,
) -> float:
    """Validate and return a bounded finite timeout in seconds."""
    if isinstance(timeout_sec, bool):
        raise ValidationError("timeout_sec must be a number, not a boolean.")
    try:
        value = float(timeout_sec)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"timeout_sec must be a finite number, got {timeout_sec!r}.") from exc
    if not math.isfinite(value):
        raise ValidationError(f"timeout_sec must be finite, got {timeout_sec!r}.")
    if value < minimum or value > maximum:
        raise ValidationError(
            f"timeout_sec must be within [{minimum}, {maximum}] seconds, got {value}."
        )
    return value


def validate_positive_float(
    value: float,
    name: str,
    *,
    minimum: float = 0.05,
    maximum: float = 60.0,
) -> float:
    """Validate a bounded positive float parameter."""
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a number, got {value!r}.") from exc
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValidationError(
            f"{name} must be within [{minimum}, {maximum}], got {parsed}."
        )
    return parsed


def validate_bounded_int(
    value: int,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Validate an integer parameter within [minimum, maximum]."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {value!r}.")
    if value < minimum or value > maximum:
        raise ValidationError(f"{name} must be within [{minimum}, {maximum}], got {value}.")
    return value


def validate_ros_name(name: str, kind: str = "ROS 2 name") -> str:
    """Validate a ROS 2 topic or node name (must start with '/', no double slashes or shell chars)."""
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"{kind} must be a non-empty string.")
    cleaned = name.strip()
    if "//" in cleaned or ".." in cleaned or not _ROS_NAME_RE.match(cleaned):
        raise ValidationError(
            f"Invalid {kind} {name!r}. Expected an absolute ROS 2 name matching "
            f"'/<segment>[/<segment>...]' with only alphanumeric/underscore characters."
        )
    return cleaned


def validate_drone_id(drone_id: str) -> str:
    """Validate a drone namespace identifier (e.g. 'drone_0', 'drone_9', 'uav_3')."""
    if not isinstance(drone_id, str) or not drone_id.strip():
        raise ValidationError("drone_id must be a non-empty string.")
    cleaned = drone_id.strip()
    if not _NAMESPACE_RE.match(cleaned):
        raise ValidationError(
            f"Invalid drone identifier {drone_id!r}. Use letters, digits, '_' or '-' only."
        )
    return cleaned.lstrip("/")


def validate_drone_ids(drone_ids: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Validate an optional list of drone IDs, preserving order and deduplicating."""
    if drone_ids is None:
        return None
    if isinstance(drone_ids, str):
        raise ValidationError("drone_ids must be a list of drone identifier strings, not a bare string.")
    items = list(drone_ids)
    if len(items) > MAX_DRONE_FILTER_COUNT:
        raise ValidationError(
            f"drone_ids cannot exceed {MAX_DRONE_FILTER_COUNT} entries, got {len(items)}."
        )
    validated: List[str] = []
    seen = set()
    for item in items:
        cid = validate_drone_id(item)
        if cid not in seen:
            seen.add(cid)
            validated.append(cid)
    return validated


def validate_gz_world(world: Optional[str]) -> Optional[str]:
    """Validate a Gazebo world name."""
    if world is None:
        return None
    if not isinstance(world, str) or not world.strip():
        raise ValidationError("world must be a non-empty string when provided.")
    cleaned = world.strip()
    if cleaned in (".", "..") or ".." in cleaned or not _GZ_WORLD_RE.match(cleaned):
        raise ValidationError(
            f"Invalid Gazebo world name {world!r}. Allowed characters: letters, digits, '_', '.', '-'."
        )
    return cleaned


def validate_gz_partition(partition: Optional[str]) -> str:
    """Validate a Gazebo transport partition name (empty string selects default partition)."""
    if partition is None:
        return ""
    if not isinstance(partition, str):
        raise ValidationError(f"partition must be a string, got {type(partition).__name__}.")
    cleaned = partition.strip()
    if ".." in cleaned or not _GZ_PARTITION_RE.match(cleaned):
        raise ValidationError(
            f"Invalid Gazebo partition {partition!r}. Allowed characters: letters, digits, '_', '.', '-'."
        )
    return cleaned


def validate_gz_topic(topic_name: str) -> str:
    """Validate a Gazebo transport topic name."""
    if not isinstance(topic_name, str) or not topic_name.strip():
        raise ValidationError("Gazebo topic_name must be a non-empty string.")
    cleaned = topic_name.strip()
    if "//" in cleaned or ".." in cleaned or not _GZ_TOPIC_RE.match(cleaned):
        raise ValidationError(
            f"Invalid Gazebo topic name {topic_name!r}. Expected an absolute topic path starting with '/'."
        )
    return cleaned


def validate_instance_id(instance_id: Optional[int]) -> Optional[int]:
    """Validate an optional PX4 SITL instance index (0..63)."""
    if instance_id is None:
        return None
    return validate_bounded_int(instance_id, "instance_id", minimum=0, maximum=MAX_INSTANCE_ID)


def validate_log_path(log_path: str, allowed_roots: Iterable[Path]) -> Path:
    """Resolve and validate that a requested log file lies strictly inside allowed read-only roots."""
    if not isinstance(log_path, str) or not log_path.strip():
        raise ValidationError("log_path must be a non-empty file path string.")
    raw = Path(log_path).expanduser()
    try:
        resolved = raw.resolve(strict=False)
    except OSError as exc:
        raise ValidationError(f"Could not resolve log_path {log_path!r}: {exc}") from exc

    resolved_roots: List[Path] = []
    for root in allowed_roots:
        try:
            resolved_roots.append(root.expanduser().resolve(strict=False))
        except OSError:
            continue

    for root in resolved_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    roots_display = ", ".join(str(r) for r in resolved_roots)
    raise ValidationError(
        f"Access denied for log_path {log_path!r}: resolved path {resolved} is outside "
        f"allowed diagnostic log directories ({roots_display})."
    )
