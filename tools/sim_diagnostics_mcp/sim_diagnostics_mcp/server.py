"""Official MCP Python SDK (FastMCP) Streamable HTTP server for read-only simulation diagnostics.

Strictly binds to 127.0.0.1:8765 and exposes read-only tools only.
No shell execution, no ROS 2 publishers/services/actions, no Gazebo control,
no PX4 commands, and no physical drone access.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional

from .config import ALLOWED_HOST, ALLOWED_PORT, BridgeConfig
from .discovery import discover_px4_rootfs_instances
from .gazebo_diagnostics import get_gazebo_world_stats, inspect_gazebo_topic
from .health import check_dependencies
from .px4_diagnostics import get_px4_sitl_status, inspect_px4_sitl_logs
from .ros2_diagnostics import (
    get_ros2_graph_overview,
    get_ros2_node_details,
    get_swarm_diagnostics_snapshot,
    inspect_ros2_topic,
)
from .runner import run_readonly_command
from .validation import ValidationError

SERVER_NAME = "swarm-sim-diagnostics"

SERVER_INSTRUCTIONS = (
    "Read-only local diagnostic bridge for ROS 2 Humble, Gazebo Harmonic, and PX4 SITL. "
    "All tools are strictly read-only, loopback-bound (127.0.0.1:8765), and dynamically "
    "discover drone namespaces (supporting 1..N drones, including 3-drone and 10-drone swarms). "
    "No vehicle commands, ROS 2 publishers/services, simulation control, or physical drone access "
    "are exposed."
)


def _error_response(exc: Exception) -> Dict[str, Any]:
    kind = "validation_error" if isinstance(exc, ValidationError) else "runtime_error"
    return {
        "ok": False,
        "error_type": kind,
        "error": str(exc),
    }


def _build_readonly_annotations() -> Optional[Any]:
    try:
        from mcp.types import ToolAnnotations

        return ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    except Exception:
        return None


def _register_tool(mcp: Any, fn: Callable[..., Dict[str, Any]], annotations: Optional[Any]) -> None:
    tool_kwargs: Dict[str, Any] = {}
    if annotations is not None:
        try:
            sig = inspect.signature(mcp.tool)
            if "annotations" in sig.parameters:
                tool_kwargs["annotations"] = annotations
        except (TypeError, ValueError):
            pass
    mcp.tool(**tool_kwargs)(fn)


def create_server(
    config: Optional[BridgeConfig] = None,
    *,
    fastmcp_cls: Optional[Any] = None,
    runner: Any = run_readonly_command,
) -> Any:
    """Create and configure the read-only FastMCP Streamable HTTP server."""
    cfg = config if config is not None else BridgeConfig()
    if cfg.host != ALLOWED_HOST or cfg.port != ALLOWED_PORT:
        raise ValueError(
            f"Diagnostic bridge must bind strictly to {ALLOWED_HOST}:{ALLOWED_PORT}."
        )

    if fastmcp_cls is None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise RuntimeError(
                "Official Python MCP SDK ('mcp') is not installed in this Python environment. "
                "Install it on your host with: "
                "python3 -m pip install -r tools/sim_diagnostics_mcp/requirements.txt"
            ) from exc
        fastmcp_cls = FastMCP

    mcp = fastmcp_cls(
        SERVER_NAME,
        instructions=SERVER_INSTRUCTIONS,
        host=cfg.host,
        port=cfg.port,
        streamable_http_path=cfg.mcp_path,
    )

    # Double-check SDK settings so non-loopback overrides via env vars cannot leak in.
    settings = getattr(mcp, "settings", None)
    if settings is not None:
        settings.host = ALLOWED_HOST
        settings.port = ALLOWED_PORT
        if hasattr(settings, "streamable_http_path"):
            settings.streamable_http_path = cfg.mcp_path

    annotations = _build_readonly_annotations()

    def sim_check_dependencies(timeout_sec: float = 4.0) -> Dict[str, Any]:
        """Check host ROS 2 Humble, Gazebo Harmonic, PX4 SITL, Python 3.10, and MCP readiness."""
        try:
            return check_dependencies(cfg, timeout_sec=timeout_sec, runner=runner)
        except Exception as exc:
            return _error_response(exc)

    def sim_discover_drones(timeout_sec: float = 4.0) -> Dict[str, Any]:
        """Dynamically discover active swarm drone namespaces, PX4 namespaces, nodes, and SITL instances."""
        try:
            graph = get_ros2_graph_overview(cfg, timeout_sec=timeout_sec, runner=runner)
            rootfs_instances = discover_px4_rootfs_instances(cfg.px4_sitl_rootfs_dir)
            return {
                "ok": graph["ok"],
                "ros2_discovery": graph["discovery"],
                "px4_rootfs_instances": rootfs_instances,
                "px4_rootfs_instance_count": len(rootfs_instances),
                "errors": graph["errors"],
            }
        except Exception as exc:
            return _error_response(exc)

    def ros2_graph_overview(timeout_sec: float = 4.0) -> Dict[str, Any]:
        """Return read-only ROS 2 nodes, topics with types, topic categories, and dynamic drone discovery."""
        try:
            return get_ros2_graph_overview(cfg, timeout_sec=timeout_sec, runner=runner)
        except Exception as exc:
            return _error_response(exc)

    def ros2_node_details(
        node_name: str,
        include_parameters: bool = True,
        timeout_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """Inspect a ROS 2 node's publishers, subscribers, services, and parameter names (read-only)."""
        try:
            return get_ros2_node_details(
                cfg,
                node_name,
                include_parameters=include_parameters,
                timeout_sec=timeout_sec,
                runner=runner,
            )
        except Exception as exc:
            return _error_response(exc)

    def ros2_topic_inspect(
        topic_name: str,
        include_sample: bool = False,
        qos_reliability: str = "auto",
        timeout_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """Inspect a ROS 2 topic's verbose QoS profile and optionally read one bounded message sample."""
        try:
            return inspect_ros2_topic(
                cfg,
                topic_name,
                include_sample=include_sample,
                qos_reliability=qos_reliability,
                timeout_sec=timeout_sec,
                runner=runner,
            )
        except Exception as exc:
            return _error_response(exc)

    def swarm_diagnostics_snapshot(
        drone_ids: Optional[List[str]] = None,
        telemetry_stale_sec: float = 0.5,
        timeout_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """Sample telemetry freshness, controller status (mode/shield_feasible), and PX4 status across all discovered drones."""
        try:
            return get_swarm_diagnostics_snapshot(
                cfg,
                drone_ids=drone_ids,
                telemetry_stale_sec=telemetry_stale_sec,
                timeout_sec=timeout_sec,
                runner=runner,
            )
        except Exception as exc:
            return _error_response(exc)

    def gazebo_world_stats(
        world: Optional[str] = None,
        partition: Optional[str] = None,
        include_models: bool = True,
        timeout_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """Inspect Gazebo Harmonic world discovery, sim/real time, RTF, pause state, and models (read-only)."""
        try:
            return get_gazebo_world_stats(
                cfg,
                world=world,
                partition=partition,
                include_models=include_models,
                timeout_sec=timeout_sec,
                runner=runner,
            )
        except Exception as exc:
            return _error_response(exc)

    def gazebo_topic_inspect(
        topic_name: str,
        include_sample: bool = False,
        partition: Optional[str] = None,
        timeout_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """Inspect a Gazebo Harmonic topic's endpoints and optionally read one bounded message (read-only)."""
        try:
            return inspect_gazebo_topic(
                cfg,
                topic_name,
                include_sample=include_sample,
                partition=partition,
                timeout_sec=timeout_sec,
                runner=runner,
            )
        except Exception as exc:
            return _error_response(exc)

    def px4_sitl_status(timeout_sec: float = 4.0) -> Dict[str, Any]:
        """Inspect local PX4 SITL build readiness, discovered rootfs instances, and running sim processes."""
        try:
            return get_px4_sitl_status(cfg, timeout_sec=timeout_sec, runner=runner)
        except Exception as exc:
            return _error_response(exc)

    def px4_sitl_logs(
        instance_id: Optional[int] = None,
        log_path: Optional[str] = None,
        tail_lines: int = 100,
        limit: int = 30,
        timeout_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """List recent PX4 SITL / ROS 2 diagnostic logs or read tail/ULog summary of one allowlisted log file."""
        try:
            return inspect_px4_sitl_logs(
                cfg,
                instance_id=instance_id,
                log_path=log_path,
                tail_lines=tail_lines,
                limit=limit,
                timeout_sec=timeout_sec,
                runner=runner,
            )
        except Exception as exc:
            return _error_response(exc)

    for tool_fn in (
        sim_check_dependencies,
        sim_discover_drones,
        ros2_graph_overview,
        ros2_node_details,
        ros2_topic_inspect,
        swarm_diagnostics_snapshot,
        gazebo_world_stats,
        gazebo_topic_inspect,
        px4_sitl_status,
        px4_sitl_logs,
    ):
        _register_tool(mcp, tool_fn, annotations)

    return mcp


def run_server(config: Optional[BridgeConfig] = None) -> None:
    """Launch the read-only Streamable HTTP MCP server strictly on 127.0.0.1:8765."""
    cfg = config if config is not None else BridgeConfig()
    mcp = create_server(cfg)
    mcp.run(transport="streamable-http")
