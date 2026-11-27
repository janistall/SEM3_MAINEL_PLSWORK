"""Read-only Gazebo Harmonic world, topic, and simulation statistics diagnostics.

Pins GZ_IP=127.0.0.1 and never invokes mutating gz commands (no gz sim, gz service, or publish).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from .config import BridgeConfig, DEFAULT_GZ_IP
from .discovery import discover_gazebo_worlds, natural_sort_key
from .runner import CommandResult, run_readonly_command
from .validation import (
    validate_gz_partition,
    validate_gz_topic,
    validate_gz_world,
    validate_timeout,
)

CommandRunner = Callable[..., CommandResult]

_BLOCK_SEC_RE = re.compile(
    r"(?P<block>sim_time|real_time|pause_time)\s*\{(?P<body>[^}]*)\}",
    re.DOTALL,
)
_SEC_RE = re.compile(r"\bsec:\s*(-?\d+)")
_NSEC_RE = re.compile(r"\bnsec:\s*(-?\d+)")
_PAUSED_RE = re.compile(r"\bpaused:\s*(true|false)", re.IGNORECASE)
_ITERATIONS_RE = re.compile(r"\biterations:\s*(\d+)")
_RTF_RE = re.compile(r"\breal_time_factor:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_STEP_SIZE_RE = re.compile(r"step_size\s*\{([^}]*)\}", re.DOTALL)


def _gz_env(partition: str) -> Dict[str, str]:
    env = {"GZ_IP": DEFAULT_GZ_IP}
    if partition:
        env["GZ_PARTITION"] = partition
    return env


def _parse_time_block(body: str) -> float:
    sec_m = _SEC_RE.search(body)
    nsec_m = _NSEC_RE.search(body)
    sec = int(sec_m.group(1)) if sec_m else 0
    nsec = int(nsec_m.group(1)) if nsec_m else 0
    return round(sec + nsec * 1e-9, 6)


def parse_world_stats(stdout: str) -> Dict[str, Any]:
    """Parse protobuf text output from `gz topic -e -n 1 -t /world/<world>/stats`."""
    parsed: Dict[str, Any] = {
        "sim_time_sec": None,
        "real_time_sec": None,
        "pause_time_sec": None,
        "paused": None,
        "iterations": None,
        "real_time_factor": None,
        "step_size_sec": None,
    }
    for match in _BLOCK_SEC_RE.finditer(stdout):
        block_name = match.group("block")
        parsed[f"{block_name}_sec"] = _parse_time_block(match.group("body"))

    paused_m = _PAUSED_RE.search(stdout)
    if paused_m:
        parsed["paused"] = paused_m.group(1).lower() == "true"

    iter_m = _ITERATIONS_RE.search(stdout)
    if iter_m:
        parsed["iterations"] = int(iter_m.group(1))

    rtf_m = _RTF_RE.search(stdout)
    if rtf_m:
        parsed["real_time_factor"] = float(rtf_m.group(1))

    step_m = _STEP_SIZE_RE.search(stdout)
    if step_m:
        parsed["step_size_sec"] = _parse_time_block(step_m.group(1))

    return parsed


def parse_model_list(stdout: str) -> List[str]:
    """Parse `gz model --list` output into a naturally sorted list of model names."""
    models: List[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            name = line.lstrip("-").strip()
            if name:
                models.append(name)
        elif not line.lower().startswith("available models") and not line.endswith(":"):
            models.append(line)
    return sorted(set(models), key=natural_sort_key)


def get_gazebo_world_stats(
    config: BridgeConfig,
    *,
    world: Optional[str] = None,
    partition: Optional[str] = None,
    include_models: bool = True,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
) -> Dict[str, Any]:
    """Inspect Gazebo Harmonic topics, active worlds, /world/<world>/stats, and optional models."""
    clean_world = validate_gz_world(world)
    clean_partition = validate_gz_partition(
        config.default_gz_partition if partition is None else partition
    )
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )
    env = _gz_env(clean_partition)

    topics_res = runner(
        "gz",
        ["topic", "-l"],
        timeout_sec=timeout,
        env_overrides=env,
        max_output_bytes=config.max_output_bytes,
    )
    if not topics_res.ok:
        return {
            "ok": False,
            "partition": clean_partition,
            "gz_ip": DEFAULT_GZ_IP,
            "discovered_worlds": [],
            "selected_world": clean_world,
            "stats": None,
            "models": [],
            "topic_count": 0,
            "topics": [],
            "errors": [topics_res.error] if topics_res.error else ["Failed to list Gazebo topics."],
        }

    topics = sorted(
        {line.strip() for line in topics_res.stdout.splitlines() if line.strip().startswith("/")},
        key=natural_sort_key,
    )
    discovered_worlds = discover_gazebo_worlds(topics)

    selected_world = clean_world
    if selected_world is None:
        if discovered_worlds:
            selected_world = discovered_worlds[0]
        else:
            return {
                "ok": True,
                "partition": clean_partition,
                "gz_ip": DEFAULT_GZ_IP,
                "discovered_worlds": [],
                "selected_world": None,
                "stats": None,
                "models": [],
                "topic_count": len(topics),
                "topics": topics,
                "warnings": [
                    f"No Gazebo worlds were discovered in partition {clean_partition!r}. "
                    f"Verify Gazebo Harmonic is running with GZ_PARTITION={clean_partition or '<default>'}."
                ],
                "errors": [],
            }

    stats_topic = f"/world/{selected_world}/stats"
    stats_res = runner(
        "gz",
        ["topic", "-e", "-n", "1", "-t", stats_topic],
        timeout_sec=timeout,
        env_overrides=env,
        max_output_bytes=config.max_output_bytes,
    )

    errors: List[str] = []
    warnings: List[str] = []
    stats_parsed: Optional[Dict[str, Any]] = None
    if stats_res.ok and stats_res.stdout.strip():
        stats_parsed = parse_world_stats(stats_res.stdout)
    else:
        if stats_res.timed_out:
            warnings.append(
                f"Timed out waiting {timeout:.2f}s for {stats_topic} in partition {clean_partition!r}."
            )
        elif stats_res.error:
            errors.append(stats_res.error)

    models: List[str] = []
    if include_models:
        model_res = runner(
            "gz",
            ["model", "--list"],
            timeout_sec=timeout,
            env_overrides=env,
            max_output_bytes=config.max_output_bytes,
        )
        if model_res.ok:
            models = parse_model_list(model_res.stdout)
        elif model_res.error:
            warnings.append(f"Could not list Gazebo models: {model_res.error}")

    return {
        "ok": len(errors) == 0,
        "partition": clean_partition,
        "gz_ip": DEFAULT_GZ_IP,
        "discovered_worlds": discovered_worlds,
        "selected_world": selected_world,
        "stats_topic": stats_topic,
        "stats": stats_parsed,
        "models": models,
        "model_count": len(models),
        "topic_count": len(topics),
        "topics": topics,
        "warnings": warnings,
        "errors": errors,
    }


def inspect_gazebo_topic(
    config: BridgeConfig,
    topic_name: str,
    *,
    include_sample: bool = False,
    partition: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
) -> Dict[str, Any]:
    """Inspect a Gazebo Harmonic topic's publishers/subscribers and optional single message."""
    clean_topic = validate_gz_topic(topic_name)
    clean_partition = validate_gz_partition(
        config.default_gz_partition if partition is None else partition
    )
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )
    env = _gz_env(clean_partition)

    info_res = runner(
        "gz",
        ["topic", "-i", "-t", clean_topic],
        timeout_sec=timeout,
        env_overrides=env,
        max_output_bytes=config.max_output_bytes,
    )
    errors: List[str] = []
    if not info_res.ok and info_res.error:
        errors.append(info_res.error)

    sample: Optional[str] = None
    sample_truncated = False
    if include_sample and info_res.ok:
        echo_res = runner(
            "gz",
            ["topic", "-e", "-n", "1", "-t", clean_topic],
            timeout_sec=timeout,
            env_overrides=env,
            max_output_bytes=config.max_output_bytes,
        )
        if echo_res.ok:
            sample = echo_res.stdout.strip()
            sample_truncated = echo_res.truncated
        else:
            if echo_res.timed_out:
                errors.append(
                    f"No Gazebo message received on {clean_topic} within {timeout:.2f}s "
                    f"(partition={clean_partition!r})."
                )
            elif echo_res.error:
                errors.append(echo_res.error)

    return {
        "ok": info_res.ok and len(errors) == 0,
        "partition": clean_partition,
        "gz_ip": DEFAULT_GZ_IP,
        "topic_name": clean_topic,
        "info": info_res.stdout.strip(),
        "sample": sample,
        "sample_truncated": sample_truncated,
        "errors": errors,
    }
