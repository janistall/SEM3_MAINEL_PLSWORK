"""CLI entrypoint for the read-only simulation diagnostic MCP bridge.

Usage (on host terminal outside Snap):
    python3 -m sim_diagnostics_mcp
    python3 -m sim_diagnostics_mcp --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import ALLOWED_HOST, ALLOWED_PORT, BridgeConfig, DEFAULT_GZ_PARTITION
from .health import check_dependencies
from .server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim_diagnostics_mcp",
        description=(
            f"Read-only Streamable HTTP MCP diagnostic bridge for ROS 2, Gazebo Harmonic, "
            f"and PX4 SITL (strictly bound to {ALLOWED_HOST}:{ALLOWED_PORT})."
        ),
    )
    parser.add_argument(
        "--px4-dir",
        type=Path,
        default=None,
        help="Path to local PX4-Autopilot checkout (default: $PX4_DIR or ~/PX4-Autopilot).",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Path to ROS 2 swarm workspace root (default: auto-detected repo root).",
    )
    parser.add_argument(
        "--ros-log-dir",
        type=Path,
        default=None,
        help="Path to ROS 2 log directory (default: $ROS_LOG_DIR or ~/.ros/log).",
    )
    parser.add_argument(
        "--gz-partition",
        type=str,
        default=DEFAULT_GZ_PARTITION,
        help=f"Default Gazebo transport partition (default: {DEFAULT_GZ_PARTITION!r}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run read-only dependency and environment check, print JSON, and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    kwargs = {
        "host": ALLOWED_HOST,
        "port": ALLOWED_PORT,
        "default_gz_partition": args.gz_partition,
    }
    if args.px4_dir is not None:
        kwargs["px4_dir"] = args.px4_dir.expanduser()
    if args.workspace_dir is not None:
        kwargs["workspace_dir"] = args.workspace_dir.expanduser()
    if args.ros_log_dir is not None:
        kwargs["ros_log_dir"] = args.ros_log_dir.expanduser()

    config = BridgeConfig(**kwargs)

    if args.check:
        report = check_dependencies(config)
        print(json.dumps(report, indent=2))
        return 0

    print(
        f"[sim_diagnostics_mcp] Starting read-only Streamable HTTP MCP bridge on "
        f"{config.endpoint_url}",
        file=sys.stderr,
        flush=True,
    )
    run_server(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
