"""Configuration and strict loopback binding policy for the diagnostic MCP bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

ALLOWED_HOST: str = "127.0.0.1"
ALLOWED_PORT: int = 8765
DEFAULT_MCP_PATH: str = "/mcp"
DEFAULT_GZ_PARTITION: str = "swarm"
DEFAULT_GZ_IP: str = "127.0.0.1"
DEFAULT_TIMEOUT_SEC: float = 4.0
MIN_TIMEOUT_SEC: float = 0.5
MAX_TIMEOUT_SEC: float = 15.0
MAX_OUTPUT_BYTES: int = 65536


def _default_workspace_dir() -> Path:
    env_ws = os.environ.get("SWARM_WORKSPACE_DIR", "").strip()
    if env_ws:
        return Path(env_ws).expanduser()
    # tools/sim_diagnostics_mcp/sim_diagnostics_mcp/config.py -> repo root
    return Path(__file__).resolve().parents[3]


def _real_home() -> Path:
    snap_real = os.environ.get("SNAP_REAL_HOME", "").strip()
    if snap_real:
        return Path(snap_real).expanduser()
    return Path.home()


def _default_px4_dir() -> Path:
    env_px4 = os.environ.get("PX4_DIR", "").strip()
    if env_px4:
        return Path(env_px4).expanduser()
    return _real_home() / "PX4-Autopilot"


def _default_ros_log_dir() -> Path:
    env_log = os.environ.get("ROS_LOG_DIR", "").strip()
    if env_log:
        return Path(env_log).expanduser()
    return _real_home() / ".ros" / "log"


@dataclass(frozen=True)
class BridgeConfig:
    """Immutable configuration for the read-only simulation diagnostic bridge.

    Binding is strictly enforced to 127.0.0.1:8765 so the server cannot be
    accidentally exposed on non-loopback interfaces.
    """

    host: str = ALLOWED_HOST
    port: int = ALLOWED_PORT
    mcp_path: str = DEFAULT_MCP_PATH
    px4_dir: Path = field(default_factory=_default_px4_dir)
    workspace_dir: Path = field(default_factory=_default_workspace_dir)
    ros_log_dir: Path = field(default_factory=_default_ros_log_dir)
    default_gz_partition: str = DEFAULT_GZ_PARTITION
    default_timeout_sec: float = DEFAULT_TIMEOUT_SEC
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if self.host != ALLOWED_HOST:
            raise ValueError(
                f"Refusing to bind to {self.host!r}: this diagnostic bridge is restricted "
                f"strictly to {ALLOWED_HOST!r}."
            )
        if int(self.port) != ALLOWED_PORT:
            raise ValueError(
                f"Refusing to bind to port {self.port!r}: this diagnostic bridge is restricted "
                f"strictly to port {ALLOWED_PORT!r}."
            )
        if not self.mcp_path.startswith("/") or " " in self.mcp_path:
            raise ValueError(
                f"mcp_path must start with '/' and contain no whitespace, got {self.mcp_path!r}."
            )
        if not (MIN_TIMEOUT_SEC <= float(self.default_timeout_sec) <= MAX_TIMEOUT_SEC):
            raise ValueError(
                f"default_timeout_sec must be within [{MIN_TIMEOUT_SEC}, {MAX_TIMEOUT_SEC}], "
                f"got {self.default_timeout_sec!r}."
            )
        if int(self.max_output_bytes) < 1024:
            raise ValueError(
                f"max_output_bytes must be >= 1024, got {self.max_output_bytes!r}."
            )

    @property
    def px4_sitl_build_dir(self) -> Path:
        return self.px4_dir / "build" / "px4_sitl_default"

    @property
    def px4_sitl_binary(self) -> Path:
        return self.px4_sitl_build_dir / "bin" / "px4"

    @property
    def px4_sitl_rootfs_dir(self) -> Path:
        return self.px4_sitl_build_dir / "rootfs"

    @property
    def px4_airframes_dir(self) -> Path:
        return self.px4_dir / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes"

    @property
    def allowed_log_roots(self) -> Tuple[Path, ...]:
        return (
            self.px4_sitl_rootfs_dir,
            self.ros_log_dir,
            self.workspace_dir / "log",
        )

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.mcp_path}"
