"""Read-only PX4 SITL build, local process, and diagnostic log inspection.

Strictly limited to local SITL processes and allowlisted local log directories.
Never accesses physical drones, serial ports, or network flight controllers.
"""

from __future__ import annotations

import os
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import BridgeConfig
from .discovery import discover_px4_rootfs_instances
from .runner import CommandResult, run_readonly_command
from .validation import (
    MAX_LIST_LIMIT,
    MAX_TAIL_LINES,
    validate_bounded_int,
    validate_instance_id,
    validate_log_path,
    validate_timeout,
)

CommandRunner = Callable[..., CommandResult]

_PS_ARGS = ("-eo", "pid,ppid,stat,pcpu,pmem,etime,args", "--no-headers")
_PS_LINE_RE = re.compile(
    r"^\s*(?P<pid>\d+)\s+(?P<ppid>\d+)\s+(?P<stat>\S+)\s+"
    r"(?P<pcpu>\S+)\s+(?P<pmem>\S+)\s+(?P<etime>\S+)\s+(?P<args>.+)$"
)
_PX4_INSTANCE_FLAG_RE = re.compile(r"(?:^|\s)-i\s+(\d+)(?:\s|$)")

_ALLOWED_TEXT_EXTENSIONS = frozenset({".log", ".txt", ".out", ".err", ".json", ".yaml", ".yml"})
_ULOG_EXTENSION = ".ulg"


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _classify_sim_process(args: str) -> Optional[str]:
    """Classify local simulation processes; ignores unrelated host processes."""
    tokens = args.split()
    if not tokens:
        return None
    exe = os.path.basename(tokens[0])
    lower = args.lower()

    # Local PX4 SITL binary only (never physical hardware tools).
    if exe == "px4" or "/build/px4_sitl_default/bin/px4" in args:
        return "px4_sitl"
    if exe == "MicroXRCEAgent" or "microxrceagent" in lower:
        return "micro_xrce_agent"
    if (exe == "gz" and "sim" in tokens[1:4]) or "gz sim" in lower or "ruby" in exe and "/gz" in lower:
        return "gazebo_sim"
    if "adapter_node" in lower or "px4_adapter" in lower:
        return "px4_adapter"
    if "swarm_control" in lower or "/controller" in lower:
        return "swarm_control"
    return None


def parse_sim_processes(ps_stdout: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse fixed-format `ps` output into categorized local simulation processes."""
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "px4_sitl": [],
        "micro_xrce_agent": [],
        "gazebo_sim": [],
        "px4_adapter": [],
        "swarm_control": [],
    }
    for raw_line in ps_stdout.splitlines():
        match = _PS_LINE_RE.match(raw_line)
        if not match:
            continue
        args = match.group("args").strip()
        category = _classify_sim_process(args)
        if category is None:
            continue

        entry: Dict[str, Any] = {
            "pid": int(match.group("pid")),
            "ppid": int(match.group("ppid")),
            "state": match.group("stat"),
            "cpu_percent": match.group("pcpu"),
            "mem_percent": match.group("pmem"),
            "elapsed": match.group("etime"),
            "command": args[:400],
        }
        if category == "px4_sitl":
            inst_m = _PX4_INSTANCE_FLAG_RE.search(args)
            entry["instance_id"] = int(inst_m.group(1)) if inst_m else 0
        grouped[category].append(entry)

    grouped["px4_sitl"].sort(key=lambda item: (item.get("instance_id", 0), item["pid"]))
    return grouped


def get_px4_sitl_status(
    config: BridgeConfig,
    *,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
) -> Dict[str, Any]:
    """Inspect local PX4 SITL build artifacts, discovered rootfs instances, and running processes."""
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )

    ps_res = runner(
        "ps",
        _PS_ARGS,
        timeout_sec=timeout,
        max_output_bytes=config.max_output_bytes,
    )
    processes = parse_sim_processes(ps_res.stdout) if ps_res.ok else {
        "px4_sitl": [],
        "micro_xrce_agent": [],
        "gazebo_sim": [],
        "px4_adapter": [],
        "swarm_control": [],
    }

    rootfs_instances = discover_px4_rootfs_instances(config.px4_sitl_rootfs_dir)
    swarm_airframe = config.px4_airframes_dir / "4002_gz_x500_swarm"
    last_target_path = config.px4_dir / "build" / ".last_target"
    last_target: Optional[str] = None
    if last_target_path.is_file():
        try:
            last_target = last_target_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            last_target = None

    px4_binary = config.px4_sitl_binary
    binary_exists = px4_binary.is_file()

    errors: List[str] = []
    if not ps_res.ok and ps_res.error:
        errors.append(ps_res.error)

    return {
        "ok": ps_res.ok,
        "build": {
            "px4_dir": str(config.px4_dir),
            "px4_dir_exists": config.px4_dir.is_dir(),
            "last_target": last_target,
            "sitl_binary": str(px4_binary),
            "sitl_binary_exists": binary_exists,
            "sitl_binary_executable": binary_exists and os.access(px4_binary, os.X_OK),
            "swarm_airframe_path": str(swarm_airframe),
            "swarm_airframe_installed": swarm_airframe.is_file(),
            "rootfs_dir": str(config.px4_sitl_rootfs_dir),
            "rootfs_instances": rootfs_instances,
            "rootfs_instance_count": len(rootfs_instances),
        },
        "processes": processes,
        "summary": {
            "running_px4_sitl_count": len(processes["px4_sitl"]),
            "running_px4_instance_ids": [p["instance_id"] for p in processes["px4_sitl"]],
            "micro_xrce_agent_running": len(processes["micro_xrce_agent"]) > 0,
            "gazebo_sim_running": len(processes["gazebo_sim"]) > 0,
            "px4_adapter_count": len(processes["px4_adapter"]),
            "swarm_control_count": len(processes["swarm_control"]),
        },
        "errors": errors,
    }


def _collect_log_files(
    config: BridgeConfig,
    *,
    instance_id: Optional[int],
    limit: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    # 1. PX4 SITL rootfs logs (dynamically supports 0..N instances).
    rootfs_dir = config.px4_sitl_rootfs_dir
    if rootfs_dir.is_dir():
        instances = (
            [instance_id]
            if instance_id is not None
            else discover_px4_rootfs_instances(rootfs_dir)
        )
        for inst in instances:
            inst_dir = rootfs_dir / str(inst)
            if not inst_dir.is_dir():
                continue
            for root_path, _, filenames in os.walk(inst_dir):
                for fname in filenames:
                    fpath = Path(root_path) / fname
                    ext = fpath.suffix.lower()
                    if ext != _ULOG_EXTENSION and ext not in _ALLOWED_TEXT_EXTENSIONS:
                        continue
                    try:
                        st = fpath.stat()
                    except OSError:
                        continue
                    candidates.append(
                        {
                            "source": "px4_sitl",
                            "instance_id": inst,
                            "path": str(fpath.resolve()),
                            "name": fname,
                            "kind": "ulog" if ext == _ULOG_EXTENSION else "text",
                            "size_bytes": st.st_size,
                            "modified_utc": _utc_iso(st.st_mtime),
                            "_mtime": st.st_mtime,
                        }
                    )

    # 2. ROS 2 diagnostic logs (~/.ros/log and workspace/log), skipped if filtering to a PX4 instance.
    if instance_id is None:
        for source_name, log_root in (
            ("ros2_log", config.ros_log_dir),
            ("workspace_log", config.workspace_dir / "log"),
        ):
            if not log_root.is_dir():
                continue
            for root_path, _, filenames in os.walk(log_root):
                for fname in filenames:
                    fpath = Path(root_path) / fname
                    ext = fpath.suffix.lower()
                    if ext not in _ALLOWED_TEXT_EXTENSIONS:
                        continue
                    try:
                        st = fpath.stat()
                    except OSError:
                        continue
                    candidates.append(
                        {
                            "source": source_name,
                            "instance_id": None,
                            "path": str(fpath.resolve()),
                            "name": fname,
                            "kind": "text",
                            "size_bytes": st.st_size,
                            "modified_utc": _utc_iso(st.st_mtime),
                            "_mtime": st.st_mtime,
                        }
                    )

    candidates.sort(key=lambda item: item["_mtime"], reverse=True)
    trimmed = candidates[:limit]
    for item in trimmed:
        item.pop("_mtime", None)
    return trimmed


def _read_text_tail(path: Path, tail_lines: int, max_bytes: int) -> Dict[str, Any]:
    lines: deque[str] = deque(maxlen=tail_lines)
    total_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            total_lines += 1
            lines.append(line.rstrip("\r\n"))

    joined = "\n".join(lines)
    encoded = joined.encode("utf-8", errors="replace")
    truncated = False
    if len(encoded) > max_bytes:
        joined = encoded[-max_bytes:].decode("utf-8", errors="replace")
        truncated = True

    return {
        "total_lines": total_lines,
        "returned_lines": len(lines),
        "truncated_by_byte_cap": truncated,
        "content": joined,
    }


def inspect_px4_sitl_logs(
    config: BridgeConfig,
    *,
    instance_id: Optional[int] = None,
    log_path: Optional[str] = None,
    tail_lines: int = 100,
    limit: int = 30,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
) -> Dict[str, Any]:
    """List recent PX4 SITL / ROS 2 logs or inspect one allowlisted diagnostic log file."""
    clean_instance = validate_instance_id(instance_id)
    clean_tail = validate_bounded_int(tail_lines, "tail_lines", minimum=1, maximum=MAX_TAIL_LINES)
    clean_limit = validate_bounded_int(limit, "limit", minimum=1, maximum=MAX_LIST_LIMIT)
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )

    if log_path is None:
        files = _collect_log_files(config, instance_id=clean_instance, limit=clean_limit)
        return {
            "ok": True,
            "mode": "list",
            "instance_id_filter": clean_instance,
            "allowed_roots": [str(r) for r in config.allowed_log_roots],
            "count": len(files),
            "logs": files,
            "errors": [],
        }

    resolved = validate_log_path(log_path, config.allowed_log_roots)
    if not resolved.is_file():
        return {
            "ok": False,
            "mode": "inspect",
            "path": str(resolved),
            "errors": [f"Log file does not exist: {resolved}"],
        }

    st = resolved.stat()
    ext = resolved.suffix.lower()
    base_meta = {
        "path": str(resolved),
        "name": resolved.name,
        "size_bytes": st.st_size,
        "modified_utc": _utc_iso(st.st_mtime),
    }

    if ext == _ULOG_EXTENSION:
        ulog_res = runner(
            "ulog_info",
            [str(resolved)],
            timeout_sec=timeout,
            max_output_bytes=config.max_output_bytes,
        )
        return {
            "ok": True,
            "mode": "inspect",
            "kind": "ulog",
            "file": base_meta,
            "ulog_info_available": ulog_res.ok,
            "ulog_summary": ulog_res.stdout.strip() if ulog_res.ok else None,
            "truncated": ulog_res.truncated,
            "warnings": (
                []
                if ulog_res.ok
                else [
                    f"ulog_info summary unavailable ({ulog_res.error}). "
                    f"Install pyulog on the host if you want .ulg header summaries."
                ]
            ),
            "errors": [],
        }

    if ext not in _ALLOWED_TEXT_EXTENSIONS:
        return {
            "ok": False,
            "mode": "inspect",
            "file": base_meta,
            "errors": [
                f"Unsupported log file extension {ext!r}. Allowed: "
                f"{sorted(_ALLOWED_TEXT_EXTENSIONS | {_ULOG_EXTENSION})}."
            ],
        }

    tail_data = _read_text_tail(resolved, clean_tail, config.max_output_bytes)
    return {
        "ok": True,
        "mode": "inspect",
        "kind": "text",
        "file": base_meta,
        **tail_data,
        "errors": [],
    }
