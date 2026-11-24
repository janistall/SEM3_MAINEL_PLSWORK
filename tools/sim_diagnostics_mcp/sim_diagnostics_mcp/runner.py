"""Allowlisted, read-only, timeout-bounded subprocess runner.

Never uses shell=True and never permits arbitrary executables or mutating subcommands.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .config import MAX_OUTPUT_BYTES
from .validation import ValidationError, validate_timeout

ALLOWED_EXECUTABLES = frozenset({"ros2", "gz", "ps", "ulog_info"})

# Explicit prefix allowlists for read-only subcommands of ros2 and gz.
_ROS2_ALLOWED_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("node", "list"),
    ("node", "info"),
    ("topic", "list"),
    ("topic", "info"),
    ("topic", "echo"),
    ("param", "list"),
    ("param", "get"),
    ("doctor", "--report"),
)

_GZ_ALLOWED_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("--version",),
    ("topic", "-l"),
    ("topic", "-i"),
    ("topic", "-e"),
    ("model", "--list"),
)

_GZ_FORBIDDEN_FLAGS = frozenset({"-p", "--pub", "--publish", "-s", "--service"})
_ROS2_FORBIDDEN_TOKENS = frozenset({"pub", "call", "set", "delete", "load", "unload", "send_goal"})


@dataclass(frozen=True)
class CommandResult:
    """Structured result from an allowlisted read-only command."""

    ok: bool
    executable: str
    args: Tuple[str, ...]
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False
    error: Optional[str] = None


def _verify_allowlisted_command(executable: str, args: Sequence[str]) -> None:
    if executable not in ALLOWED_EXECUTABLES:
        raise ValidationError(
            f"Executable {executable!r} is not permitted. Allowed read-only tools: "
            f"{sorted(ALLOWED_EXECUTABLES)}."
        )
    arg_tuple = tuple(args)

    if executable == "ros2":
        if not any(arg_tuple[: len(prefix)] == prefix for prefix in _ROS2_ALLOWED_PREFIXES):
            raise ValidationError(
                f"Disallowed ros2 subcommand {list(arg_tuple[:2])!r}; only read-only graph/info/echo/param inspection is permitted."
            )
        if any(token in _ROS2_FORBIDDEN_TOKENS for token in arg_tuple):
            raise ValidationError("Mutating ROS 2 arguments are prohibited in read-only mode.")

    elif executable == "gz":
        if not any(arg_tuple[: len(prefix)] == prefix for prefix in _GZ_ALLOWED_PREFIXES):
            raise ValidationError(
                f"Disallowed gz subcommand {list(arg_tuple[:2])!r}; only read-only topic/model/version inspection is permitted."
            )
        if any(token in _GZ_FORBIDDEN_FLAGS for token in arg_tuple):
            raise ValidationError("Mutating Gazebo flags are prohibited in read-only mode.")

    elif executable == "ps":
        if arg_tuple != ("-eo", "pid,ppid,stat,pcpu,pmem,etime,args", "--no-headers"):
            raise ValidationError("Only the fixed read-only process table query is permitted for ps.")

    elif executable == "ulog_info":
        if len(arg_tuple) != 1 or arg_tuple[0].startswith("-"):
            raise ValidationError("ulog_info accepts exactly one validated .ulg file path.")


def _truncate_text(data: bytes, max_bytes: int) -> Tuple[str, bool]:
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace"), False
    return data[:max_bytes].decode("utf-8", errors="replace"), True


def run_readonly_command(
    executable: str,
    args: Sequence[str],
    *,
    timeout_sec: float = 4.0,
    env_overrides: Optional[Mapping[str, str]] = None,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> CommandResult:
    """Execute an allowlisted read-only host command with strict timeout and output caps."""
    validated_timeout = validate_timeout(timeout_sec)
    _verify_allowlisted_command(executable, args)

    resolved_bin = shutil.which(executable)
    if resolved_bin is None:
        return CommandResult(
            ok=False,
            executable=executable,
            args=tuple(args),
            returncode=None,
            stdout="",
            stderr="",
            error=(
                f"Required executable {executable!r} was not found on PATH. "
                f"Ensure the host environment (e.g. /opt/ros/humble/setup.bash or Gazebo Harmonic) is sourced."
            ),
        )

    env: Dict[str, str] = dict(os.environ)
    if env_overrides:
        for key, value in env_overrides.items():
            env[str(key)] = str(value)

    try:
        proc = subprocess.run(
            [resolved_bin, *args],
            capture_output=True,
            timeout=validated_timeout,
            env=env,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raw_out = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode("utf-8")
        raw_err = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode("utf-8")
        stdout_text, out_trunc = _truncate_text(raw_out, max_output_bytes)
        stderr_text, err_trunc = _truncate_text(raw_err, max_output_bytes)
        return CommandResult(
            ok=False,
            executable=executable,
            args=tuple(args),
            returncode=None,
            stdout=stdout_text,
            stderr=stderr_text,
            timed_out=True,
            truncated=out_trunc or err_trunc,
            error=f"Command '{executable} {' '.join(args)}' timed out after {validated_timeout:.2f}s.",
        )
    except OSError as exc:
        return CommandResult(
            ok=False,
            executable=executable,
            args=tuple(args),
            returncode=None,
            stdout="",
            stderr="",
            error=f"Failed to launch {executable!r}: {exc}",
        )

    stdout_text, out_trunc = _truncate_text(proc.stdout or b"", max_output_bytes)
    stderr_text, err_trunc = _truncate_text(proc.stderr or b"", max_output_bytes)
    ok = proc.returncode == 0
    error_msg: Optional[str] = None
    if not ok:
        detail = stderr_text.strip() or stdout_text.strip() or f"exit code {proc.returncode}"
        error_msg = f"Command '{executable} {' '.join(args)}' failed (rc={proc.returncode}): {detail}"

    return CommandResult(
        ok=ok,
        executable=executable,
        args=tuple(args),
        returncode=proc.returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        timed_out=False,
        truncated=out_trunc or err_trunc,
        error=error_msg,
    )
