"""Read-only dependency, runtime, and environment readiness checks for the host bridge."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional

from .config import BridgeConfig, DEFAULT_GZ_IP
from .discovery import discover_px4_rootfs_instances
from .runner import CommandResult, run_readonly_command
from .validation import validate_timeout

CommandRunner = Callable[..., CommandResult]


def _check_python_module(module_name: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(module_name)
        return {
            "available": True,
            "version": getattr(mod, "__version__", None),
            "location": getattr(mod, "__file__", "builtin"),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_dependencies(
    config: BridgeConfig,
    *,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
    which_fn: Callable[[str], Optional[str]] = shutil.which,
    module_checker: Callable[[str], Dict[str, Any]] = _check_python_module,
) -> Dict[str, Any]:
    """Audit host dependencies, ROS 2 overlay, Gazebo Harmonic, PX4 SITL, and Python packages."""
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )

    executables = {
        name: {
            "available": which_fn(name) is not None,
            "path": which_fn(name),
        }
        for name in ("ros2", "gz", "MicroXRCEAgent", "ps", "ulog_info")
    }

    gz_version: Optional[str] = None
    if executables["gz"]["available"]:
        for gz_args in (["sim", "--version"], ["sim", "--versions"], ["--version"]):
            gz_ver_res = runner(
                "gz",
                gz_args,
                timeout_sec=timeout,
                max_output_bytes=config.max_output_bytes,
            )
            if gz_ver_res.ok:
                first_line = next(
                    (line.strip() for line in gz_ver_res.stdout.splitlines() if line.strip()),
                    None,
                )
                if first_line:
                    gz_version = first_line
                    break

    python_modules = {
        name: module_checker(name)
        for name in ("mcp", "rclpy", "swarm_interfaces", "px4_msgs")
    }

    px4_binary = config.px4_sitl_binary
    px4_rootfs_instances = discover_px4_rootfs_instances(config.px4_sitl_rootfs_dir)
    workspace_install = config.workspace_dir / "install"

    ros_localhost_only = os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    ros_domain_id_raw = os.environ.get("ROS_DOMAIN_ID")

    env_snapshot = {
        "ROS_DISTRO": os.environ.get("ROS_DISTRO"),
        "ROS_DOMAIN_ID": ros_domain_id_raw if ros_domain_id_raw is not None else "0 (default)",
        "ROS_LOCALHOST_ONLY": os.environ.get("ROS_LOCALHOST_ONLY"),
        "AMENT_PREFIX_PATH": os.environ.get("AMENT_PREFIX_PATH"),
        "GZ_PARTITION": os.environ.get("GZ_PARTITION"),
        "GZ_IP": os.environ.get("GZ_IP"),
        "PX4_DIR": os.environ.get("PX4_DIR"),
        "SNAP": os.environ.get("SNAP"),
    }

    hints: List[str] = []
    if env_snapshot["SNAP"]:
        hints.append(
            "SNAP environment detected: run this bridge on the host outside Snap confinement "
            "so ROS 2 Humble, Gazebo Harmonic, and Python 3.10 rclpy are accessible."
        )
    if not ros_localhost_only:
        hints.append(
            "ROS_LOCALHOST_ONLY is not '1': export ROS_LOCALHOST_ONLY=1 and a dedicated "
            "ROS_DOMAIN_ID (e.g. 42) in both the MCP bridge and simulation terminals for isolated loopback testing."
        )
    if not executables["ros2"]["available"] or not python_modules["rclpy"]["available"]:
        hints.append("Source ROS 2 Humble on the host: `source /opt/ros/humble/setup.bash`.")
    if not python_modules["swarm_interfaces"]["available"]:
        hints.append(
            f"Build and source the workspace overlay: `source {workspace_install}/setup.bash`."
        )
    if not python_modules["px4_msgs"]["available"]:
        hints.append(
            "Source your px4_msgs ROS 2 underlay so PX4 /fmu/out/* messages can be decoded via rclpy."
        )
    if not executables["gz"]["available"]:
        hints.append("Ensure Gazebo Harmonic (`/usr/bin/gz`) is installed and on PATH.")
    if not executables["MicroXRCEAgent"]["available"]:
        hints.append("Ensure `MicroXRCEAgent` is installed and on PATH for PX4 uXRCE-DDS telemetry.")
    if not px4_binary.is_file():
        hints.append(
            f"PX4 SITL binary not found at {px4_binary}. Set PX4_DIR or build `px4_sitl_default`."
        )
    if not python_modules["mcp"]["available"]:
        hints.append(
            "Install the official Python MCP SDK in your host environment: "
            "`pip install -r tools/sim_diagnostics_mcp/requirements.txt`."
        )

    py_ver = sys.version_info
    ready_core = (
        executables["ros2"]["available"]
        and executables["gz"]["available"]
        and python_modules["mcp"]["available"]
        and px4_binary.is_file()
    )

    return {
        "ok": True,
        "ready_for_live_diagnostics": ready_core,
        "bridge": {
            "bind_host": config.host,
            "bind_port": config.port,
            "endpoint_url": config.endpoint_url,
            "read_only": True,
            "physical_drone_access": False,
        },
        "isolation": {
            "bridge_loopback_only": config.host == "127.0.0.1",
            "ros_localhost_only": ros_localhost_only,
            "ros_domain_id": env_snapshot["ROS_DOMAIN_ID"],
            "gazebo_ip": env_snapshot["GZ_IP"] or DEFAULT_GZ_IP,
            "gazebo_partition": env_snapshot["GZ_PARTITION"] or config.default_gz_partition,
        },
        "python": {
            "executable": sys.executable,
            "version": f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}",
            "compatible_with_ros2_humble": (py_ver.major, py_ver.minor) == (3, 10),
        },
        "executables": executables,
        "gazebo_version": gz_version,
        "python_modules": python_modules,
        "paths": {
            "workspace_dir": str(config.workspace_dir),
            "workspace_install_exists": workspace_install.is_dir(),
            "px4_dir": str(config.px4_dir),
            "px4_sitl_binary": str(px4_binary),
            "px4_sitl_binary_exists": px4_binary.is_file(),
            "px4_rootfs_instances": px4_rootfs_instances,
            "ros_log_dir": str(config.ros_log_dir),
            "ros_log_dir_exists": config.ros_log_dir.is_dir(),
        },
        "environment": env_snapshot,
        "remediation_hints": hints,
    }
