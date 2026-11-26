"""Read-only ROS 2 graph, node, topic, and dynamic multi-drone swarm diagnostics.

Never creates publishers, service clients, or action clients.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import BridgeConfig
from .discovery import discover_from_ros_graph, natural_sort_key
from .runner import CommandResult, run_readonly_command
from .validation import (
    ValidationError,
    validate_drone_ids,
    validate_positive_float,
    validate_ros_name,
    validate_timeout,
)

CommandRunner = Callable[..., CommandResult]

_RCLPY_LOCK = threading.Lock()
_TOPIC_LINE_RE = re.compile(r"^(?P<topic>/\S+)\s+\[(?P<types>[^\]]+)\]\s*$")
_YAML_DOC_SEP = "\n---"

# PX4 VehicleStatus arming_state / nav_state display maps (PX4 v1.14 / v1.15 standard enum values).
_ARMING_STATE_NAMES = {
    1: "DISARMED",
    2: "ARMED",
}
_NAV_STATE_NAMES = {
    0: "MANUAL",
    1: "ALTCTL",
    2: "POSCTL",
    3: "AUTO_MISSION",
    4: "AUTO_LOITER",
    5: "AUTO_RTL",
    14: "OFFBOARD",
    17: "AUTO_TAKEOFF",
    18: "AUTO_LAND",
    19: "AUTO_FOLLOW_TARGET",
    20: "AUTO_PRECLAND",
}


def parse_topic_list_with_types(stdout: str) -> List[Dict[str, Any]]:
    """Parse `ros2 topic list -t` output into sorted structured entries."""
    items: List[Dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TOPIC_LINE_RE.match(line)
        if match:
            topic = match.group("topic")
            types = [t.strip() for t in match.group("types").split(",") if t.strip()]
            items.append({"topic": topic, "types": types})
        elif line.startswith("/"):
            items.append({"topic": line, "types": []})
    items.sort(key=lambda item: natural_sort_key(item["topic"]))
    return items


def parse_node_list(stdout: str) -> List[str]:
    """Parse `ros2 node list` output into a naturally sorted list."""
    nodes = [line.strip() for line in stdout.splitlines() if line.strip().startswith("/")]
    return sorted(set(nodes), key=natural_sort_key)


def parse_topic_info_verbose(stdout: str) -> Dict[str, Any]:
    """Parse summary counts and QoS profile lines from `ros2 topic info --verbose`."""
    topic_type: Optional[str] = None
    pub_count: Optional[int] = None
    sub_count: Optional[int] = None
    qos_profiles: List[Dict[str, str]] = []

    current_endpoint: Optional[Dict[str, str]] = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Type:"):
            topic_type = line.split(":", 1)[1].strip()
        elif line.startswith("Publisher count:"):
            try:
                pub_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Subscription count:"):
            try:
                sub_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Node name:"):
            if current_endpoint:
                qos_profiles.append(current_endpoint)
            current_endpoint = {"node_name": line.split(":", 1)[1].strip()}
        elif current_endpoint is not None and ":" in line:
            key, val = line.split(":", 1)
            norm_key = key.strip().lower().replace(" ", "_")
            if norm_key in ("node_namespace", "endpoint_type", "reliability", "durability", "history_(depth)", "liveliness"):
                current_endpoint[norm_key] = val.strip()
    if current_endpoint:
        qos_profiles.append(current_endpoint)

    return {
        "type": topic_type,
        "publisher_count": pub_count,
        "subscription_count": sub_count,
        "endpoints": qos_profiles,
    }


def _categorize_topics(topics_with_types: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "swarm_telemetry": [],
        "swarm_status": [],
        "swarm_cmd_vel": [],
        "px4_fmu_out": [],
        "px4_fmu_in": [],
        "diagnostics": [],
        "other": [],
    }
    for entry in topics_with_types:
        topic = entry["topic"]
        if topic.endswith("/telemetry"):
            categories["swarm_telemetry"].append(topic)
        elif topic.endswith("/status"):
            categories["swarm_status"].append(topic)
        elif topic.endswith("/cmd_vel"):
            categories["swarm_cmd_vel"].append(topic)
        elif "/fmu/out/" in topic:
            categories["px4_fmu_out"].append(topic)
        elif "/fmu/in/" in topic:
            categories["px4_fmu_in"].append(topic)
        elif topic == "/diagnostics" or topic.startswith("/diagnostics/"):
            categories["diagnostics"].append(topic)
        else:
            categories["other"].append(topic)
    return categories


def get_ros2_graph_overview(
    config: BridgeConfig,
    *,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
) -> Dict[str, Any]:
    """Return read-only ROS 2 nodes, topics with types, categories, and dynamic drone discovery."""
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )
    topics_res = runner(
        "ros2",
        ["topic", "list", "-t"],
        timeout_sec=timeout,
        max_output_bytes=config.max_output_bytes,
    )
    nodes_res = runner(
        "ros2",
        ["node", "list"],
        timeout_sec=timeout,
        max_output_bytes=config.max_output_bytes,
    )

    errors: List[str] = []
    if not topics_res.ok and topics_res.error:
        errors.append(topics_res.error)
    if not nodes_res.ok and nodes_res.error:
        errors.append(nodes_res.error)

    topics_with_types = parse_topic_list_with_types(topics_res.stdout) if topics_res.ok else []
    nodes = parse_node_list(nodes_res.stdout) if nodes_res.ok else []
    topic_names = [entry["topic"] for entry in topics_with_types]
    discovery = discover_from_ros_graph(topic_names, nodes)

    return {
        "ok": topics_res.ok and nodes_res.ok,
        "node_count": len(nodes),
        "topic_count": len(topics_with_types),
        "nodes": nodes,
        "topics": topics_with_types,
        "categories": _categorize_topics(topics_with_types),
        "discovery": discovery,
        "errors": errors,
    }


def get_ros2_node_details(
    config: BridgeConfig,
    node_name: str,
    *,
    include_parameters: bool = True,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
) -> Dict[str, Any]:
    """Inspect a single ROS 2 node's graph connections and optional parameter names (read-only)."""
    clean_node = validate_ros_name(node_name, kind="node_name")
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )

    info_res = runner(
        "ros2",
        ["node", "info", clean_node],
        timeout_sec=timeout,
        max_output_bytes=config.max_output_bytes,
    )
    result: Dict[str, Any] = {
        "ok": info_res.ok,
        "node_name": clean_node,
        "node_info": info_res.stdout.strip(),
        "truncated": info_res.truncated,
        "errors": [info_res.error] if (not info_res.ok and info_res.error) else [],
    }

    if include_parameters and info_res.ok:
        param_res = runner(
            "ros2",
            ["param", "list", clean_node],
            timeout_sec=timeout,
            max_output_bytes=config.max_output_bytes,
        )
        if param_res.ok:
            params = [
                line.strip()
                for line in param_res.stdout.splitlines()
                if line.strip() and not line.strip().endswith(":")
            ]
            result["parameters"] = params
        else:
            result["parameters"] = []
            if param_res.error:
                result["errors"].append(param_res.error)

    return result


def inspect_ros2_topic(
    config: BridgeConfig,
    topic_name: str,
    *,
    include_sample: bool = False,
    qos_reliability: str = "auto",
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
) -> Dict[str, Any]:
    """Inspect a ROS 2 topic's verbose QoS info and optionally read a single bounded message."""
    clean_topic = validate_ros_name(topic_name, kind="topic_name")
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )
    if qos_reliability not in ("auto", "reliable", "best_effort"):
        raise ValidationError(
            f"qos_reliability must be 'auto', 'reliable', or 'best_effort', got {qos_reliability!r}."
        )

    info_res = runner(
        "ros2",
        ["topic", "info", clean_topic, "--verbose"],
        timeout_sec=timeout,
        max_output_bytes=config.max_output_bytes,
    )
    parsed = parse_topic_info_verbose(info_res.stdout) if info_res.ok else {}
    errors: List[str] = []
    if not info_res.ok and info_res.error:
        errors.append(info_res.error)

    result: Dict[str, Any] = {
        "ok": info_res.ok,
        "topic_name": clean_topic,
        "info": parsed,
        "raw_info": info_res.stdout.strip(),
        "sample": None,
        "errors": errors,
    }

    if include_sample and info_res.ok:
        effective_rel = qos_reliability
        if effective_rel == "auto":
            if "/fmu/" in clean_topic:
                effective_rel = "best_effort"
            else:
                # Inspect publisher QoS if reported
                pub_rels = [
                    ep.get("reliability", "").lower()
                    for ep in parsed.get("endpoints", [])
                    if ep.get("endpoint_type", "").upper() == "PUBLISHER"
                ]
                if pub_rels and all("best_effort" in r for r in pub_rels):
                    effective_rel = "best_effort"
                else:
                    effective_rel = "reliable"

        echo_args = [
            "topic",
            "echo",
            clean_topic,
            "--once",
            "--qos-reliability",
            effective_rel,
        ]
        sample_res = runner(
            "ros2",
            echo_args,
            timeout_sec=timeout,
            max_output_bytes=config.max_output_bytes,
        )
        result["sample_qos_reliability"] = effective_rel
        if sample_res.ok:
            text = sample_res.stdout.strip()
            if _YAML_DOC_SEP in text:
                text = text.split(_YAML_DOC_SEP, 1)[0].strip()
            result["sample"] = text
            result["sample_truncated"] = sample_res.truncated
        else:
            if sample_res.timed_out:
                errors.append(
                    f"No message received on {clean_topic} within {timeout:.2f}s "
                    f"(qos_reliability={effective_rel})."
                )
            elif sample_res.error:
                errors.append(sample_res.error)

    return result


def _decode_agent_telemetry(msg: Any, received_monotonic: float) -> Dict[str, Any]:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    stamp_sec = (
        float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
        if stamp is not None
        else None
    )
    pos = getattr(msg, "position", None)
    vel = getattr(msg, "velocity", None)
    return {
        "drone_id": str(getattr(msg, "drone_id", "")),
        "position_enu": [
            round(float(getattr(pos, "x", 0.0)), 4),
            round(float(getattr(pos, "y", 0.0)), 4),
            round(float(getattr(pos, "z", 0.0)), 4),
        ],
        "velocity_enu": [
            round(float(getattr(vel, "x", 0.0)), 4),
            round(float(getattr(vel, "y", 0.0)), 4),
            round(float(getattr(vel, "z", 0.0)), 4),
        ],
        "yaw_rad": round(float(getattr(msg, "yaw", 0.0)), 4),
        "active_status": bool(getattr(msg, "active_status", False)),
        "header_stamp_sec": stamp_sec,
        "_rx_monotonic": received_monotonic,
    }


def _decode_status_string(msg: Any, received_monotonic: float) -> Dict[str, Any]:
    raw_data = str(getattr(msg, "data", ""))
    try:
        parsed = json.loads(raw_data)
        if isinstance(parsed, dict):
            parsed["_rx_monotonic"] = received_monotonic
            return parsed
    except json.JSONDecodeError:
        pass
    return {"raw": raw_data, "_rx_monotonic": received_monotonic}


def _decode_px4_status(msg: Any, received_monotonic: float) -> Dict[str, Any]:
    arming_state = int(getattr(msg, "arming_state", 0))
    nav_state = int(getattr(msg, "nav_state", -1))
    return {
        "arming_state": arming_state,
        "arming_state_label": _ARMING_STATE_NAMES.get(arming_state, f"UNKNOWN({arming_state})"),
        "nav_state": nav_state,
        "nav_state_label": _NAV_STATE_NAMES.get(nav_state, f"UNKNOWN({nav_state})"),
        "pre_flight_checks_pass": bool(getattr(msg, "pre_flight_checks_pass", False)),
        "failsafe": bool(getattr(msg, "failsafe", False)),
        "gcs_connection_lost": bool(getattr(msg, "gcs_connection_lost", False)),
        "_rx_monotonic": received_monotonic,
    }


def _decode_px4_odometry(msg: Any, received_monotonic: float) -> Dict[str, Any]:
    position = getattr(msg, "position", None)
    velocity = getattr(msg, "velocity", None)
    return {
        "position_ned": [round(float(v), 4) for v in list(position)[:3]] if position is not None else None,
        "velocity_ned": [round(float(v), 4) for v in list(velocity)[:3]] if velocity is not None else None,
        "pose_frame": int(getattr(msg, "pose_frame", 0)),
        "velocity_frame": int(getattr(msg, "velocity_frame", 0)),
        "_rx_monotonic": received_monotonic,
    }


def _decode_px4_ack(msg: Any, received_monotonic: float) -> Dict[str, Any]:
    return {
        "command": int(getattr(msg, "command", 0)),
        "result": int(getattr(msg, "result", 0)),
        "from_external": bool(getattr(msg, "from_external", False)),
        "_rx_monotonic": received_monotonic,
    }


def _sample_with_rclpy(
    drones: Sequence[Dict[str, Any]],
    px4_topics_by_ns: Dict[str, List[str]],
    timeout_sec: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Use an isolated read-only rclpy Context to sample swarm and PX4 topics concurrently."""
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
    except Exception as exc:
        return None, f"rclpy unavailable ({type(exc).__name__}: {exc})"

    try:
        from swarm_interfaces.msg import AgentTelemetry
    except Exception:
        AgentTelemetry = None

    try:
        from px4_msgs.msg import VehicleCommandAck, VehicleOdometry, VehicleStatus
    except Exception:
        VehicleCommandAck = None
        VehicleOdometry = None
        VehicleStatus = None

    telemetry_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    px4_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    samples: Dict[str, Dict[str, Any]] = {
        "telemetry": {},
        "status": {},
        "px4_status": {},
        "px4_odometry": {},
        "px4_ack": {},
    }
    missing_types: List[str] = []
    if AgentTelemetry is None:
        missing_types.append("swarm_interfaces.msg.AgentTelemetry")
    if VehicleStatus is None:
        missing_types.append("px4_msgs.msg.{VehicleStatus,VehicleOdometry,VehicleCommandAck}")

    expected_keys = set()

    with _RCLPY_LOCK:
        ctx = Context()
        rclpy.init(args=None, context=ctx)
        node = Node("sim_diagnostics_mcp_observer", context=ctx)
        executor = SingleThreadedExecutor(context=ctx)
        executor.add_node(node)
        subs = []
        try:
            for drone in drones:
                drone_id = drone["drone_id"]
                tel_topic = drone.get("telemetry_topic") or f"/{drone_id}/telemetry"
                status_topic = drone.get("status_topic") or f"/{drone_id}/status"

                if AgentTelemetry is not None:
                    expected_keys.add(("telemetry", drone_id))
                    subs.append(
                        node.create_subscription(
                            AgentTelemetry,
                            tel_topic,
                            lambda msg, did=drone_id: samples["telemetry"].__setitem__(
                                did, _decode_agent_telemetry(msg, time.monotonic())
                            ),
                            telemetry_qos,
                        )
                    )

                expected_keys.add(("status", drone_id))
                subs.append(
                    node.create_subscription(
                        String,
                        status_topic,
                        lambda msg, did=drone_id: samples["status"].__setitem__(
                            did, _decode_status_string(msg, time.monotonic())
                        ),
                        telemetry_qos,
                    )
                )

                px4_ns = drone.get("paired_px4_namespace")
                if px4_ns and VehicleStatus is not None:
                    ns_prefix = "" if px4_ns == "/" else px4_ns.rstrip("/")
                    available_out = set(px4_topics_by_ns.get(px4_ns, []))

                    status_candidates = [
                        f"{ns_prefix}/fmu/out/vehicle_status",
                        f"{ns_prefix}/fmu/out/vehicle_status_v1",
                        f"{ns_prefix}/fmu/out/vehicle_status_v4",
                    ]
                    for topic in status_candidates:
                        if not available_out or topic in available_out:
                            expected_keys.add(("px4_status", drone_id))
                            subs.append(
                                node.create_subscription(
                                    VehicleStatus,
                                    topic,
                                    lambda msg, did=drone_id: samples["px4_status"].__setitem__(
                                        did, _decode_px4_status(msg, time.monotonic())
                                    ),
                                    px4_qos,
                                )
                            )

                    odom_topic = f"{ns_prefix}/fmu/out/vehicle_odometry"
                    if VehicleOdometry is not None and (not available_out or odom_topic in available_out):
                        expected_keys.add(("px4_odometry", drone_id))
                        subs.append(
                            node.create_subscription(
                                VehicleOdometry,
                                odom_topic,
                                lambda msg, did=drone_id: samples["px4_odometry"].__setitem__(
                                    did, _decode_px4_odometry(msg, time.monotonic())
                                ),
                                px4_qos,
                            )
                        )

                    if VehicleCommandAck is not None:
                        for ack_leaf in ("vehicle_command_ack", "vehicle_command_ack_v1"):
                            ack_topic = f"{ns_prefix}/fmu/out/{ack_leaf}"
                            if not available_out or ack_topic in available_out:
                                subs.append(
                                    node.create_subscription(
                                        VehicleCommandAck,
                                        ack_topic,
                                        lambda msg, did=drone_id: samples["px4_ack"].__setitem__(
                                            did, _decode_px4_ack(msg, time.monotonic())
                                        ),
                                        px4_qos,
                                    )
                                )

            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
                if expected_keys and all(
                    did in samples[bucket] for bucket, did in expected_keys
                ):
                    break
        finally:
            try:
                executor.remove_node(node)
                executor.shutdown()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown(context=ctx)
            except Exception:
                pass

    return {
        "samples": samples,
        "missing_message_types": missing_types,
        "snapshot_monotonic": time.monotonic(),
    }, None


def _strip_internal_fields(payload: Optional[Dict[str, Any]], now_mono: float) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    if not payload:
        return None, None
    cleaned = dict(payload)
    rx = cleaned.pop("_rx_monotonic", None)
    age = round(max(0.0, now_mono - float(rx)), 4) if rx is not None else None
    return cleaned, age


def get_swarm_diagnostics_snapshot(
    config: BridgeConfig,
    *,
    drone_ids: Optional[Sequence[str]] = None,
    telemetry_stale_sec: float = 0.5,
    timeout_sec: Optional[float] = None,
    runner: CommandRunner = run_readonly_command,
    rclpy_sampler: Optional[Callable[..., Tuple[Optional[Dict[str, Any]], Optional[str]]]] = None,
) -> Dict[str, Any]:
    """Collect a read-only diagnostic snapshot across all dynamically discovered drones."""
    validated_filter = validate_drone_ids(drone_ids)
    stale_threshold = validate_positive_float(
        telemetry_stale_sec, "telemetry_stale_sec", minimum=0.05, maximum=30.0
    )
    timeout = validate_timeout(
        config.default_timeout_sec if timeout_sec is None else timeout_sec
    )

    graph = get_ros2_graph_overview(config, timeout_sec=timeout, runner=runner)
    if not graph["ok"]:
        return {
            "ok": False,
            "backend": "none",
            "drones": [],
            "fleet_summary": {
                "discovered_drone_count": 0,
                "selected_drone_count": 0,
            },
            "errors": graph["errors"],
        }

    discovery = graph["discovery"]
    discovered_drones: List[Dict[str, Any]] = list(discovery["drones"])

    if validated_filter is not None:
        by_id = {d["drone_id"]: d for d in discovered_drones}
        selected: List[Dict[str, Any]] = []
        for did in validated_filter:
            if did in by_id:
                selected.append(by_id[did])
            else:
                selected.append(
                    {
                        "drone_id": did,
                        "index": None,
                        "telemetry_topic": f"/{did}/telemetry",
                        "status_topic": f"/{did}/status",
                        "cmd_vel_topic": f"/{did}/cmd_vel",
                        "paired_px4_namespace": None,
                        "adapter_node": None,
                        "control_node": None,
                    }
                )
    else:
        selected = discovered_drones

    if not selected:
        return {
            "ok": True,
            "backend": "graph_only",
            "drones": [],
            "fleet_summary": {
                "discovered_drone_count": 0,
                "selected_drone_count": 0,
                "px4_namespace_count": discovery["px4_namespace_count"],
                "note": "No active swarm drone namespaces (/<drone>/telemetry or /<drone>/status) were discovered on the ROS 2 graph.",
            },
            "errors": [],
        }

    # Build PX4 out-topic lookup per namespace from graph topics.
    px4_topics_by_ns: Dict[str, List[str]] = {}
    for entry in graph["topics"]:
        topic = entry["topic"]
        if "/fmu/out/" in topic:
            ns = topic.split("/fmu/out/", 1)[0] or "/"
            px4_topics_by_ns.setdefault(ns, []).append(topic)

    sampler = rclpy_sampler or _sample_with_rclpy
    sampled_data, sampler_err = sampler(selected, px4_topics_by_ns, timeout)

    warnings: List[str] = []
    backend = "rclpy"
    if sampled_data is None:
        backend = "graph_only_fallback"
        warnings.append(
            f"Native rclpy live sampling unavailable ({sampler_err}). "
            f"Source /opt/ros/humble/setup.bash and workspace install/setup.bash before launching "
            f"the bridge for concurrent telemetry/PX4 QoS sampling. Returning graph metadata only."
        )
        samples = {"telemetry": {}, "status": {}, "px4_status": {}, "px4_odometry": {}, "px4_ack": {}}
        now_mono = time.monotonic()
    else:
        samples = sampled_data["samples"]
        now_mono = float(sampled_data.get("snapshot_monotonic", time.monotonic()))
        missing = sampled_data.get("missing_message_types", [])
        if missing:
            warnings.append(
                "Some ROS 2 Python message packages were not importable: "
                + ", ".join(missing)
                + ". Source your workspace and px4_msgs overlay for full decoding."
            )

    drone_reports: List[Dict[str, Any]] = []
    fresh_count = 0
    shield_infeasible: List[str] = []
    stale_drones: List[str] = []
    stall_recovery_drones: List[str] = []
    failsafe_drones: List[str] = []

    for drone in selected:
        did = drone["drone_id"]
        tel_payload, tel_age = _strip_internal_fields(samples["telemetry"].get(did), now_mono)
        status_payload, status_age = _strip_internal_fields(samples["status"].get(did), now_mono)
        px4_status, px4_status_age = _strip_internal_fields(samples["px4_status"].get(did), now_mono)
        px4_odom, px4_odom_age = _strip_internal_fields(samples["px4_odometry"].get(did), now_mono)
        px4_ack, px4_ack_age = _strip_internal_fields(samples["px4_ack"].get(did), now_mono)

        telemetry_fresh = bool(
            tel_payload is not None
            and tel_payload.get("active_status", False)
            and tel_age is not None
            and tel_age <= stale_threshold
        )
        if telemetry_fresh:
            fresh_count += 1
        elif backend == "rclpy":
            stale_drones.append(did)

        if isinstance(status_payload, dict):
            if status_payload.get("shield_feasible") is False:
                shield_infeasible.append(did)
            if status_payload.get("stall_recovery") is True:
                stall_recovery_drones.append(did)

        if isinstance(px4_status, dict) and px4_status.get("failsafe") is True:
            failsafe_drones.append(did)

        drone_reports.append(
            {
                "drone_id": did,
                "index": drone.get("index"),
                "paired_px4_namespace": drone.get("paired_px4_namespace"),
                "adapter_node": drone.get("adapter_node"),
                "control_node": drone.get("control_node"),
                "telemetry": tel_payload,
                "telemetry_age_sec": tel_age,
                "telemetry_fresh": telemetry_fresh,
                "controller_status": status_payload,
                "controller_status_age_sec": status_age,
                "px4_status": px4_status,
                "px4_status_age_sec": px4_status_age,
                "px4_odometry": px4_odom,
                "px4_odometry_age_sec": px4_odom_age,
                "px4_last_ack": px4_ack,
                "px4_last_ack_age_sec": px4_ack_age,
            }
        )

    fleet_summary: Dict[str, Any] = {
        "discovered_drone_count": len(discovered_drones),
        "selected_drone_count": len(selected),
        "px4_namespace_count": discovery["px4_namespace_count"],
        "telemetry_stale_threshold_sec": stale_threshold,
        "telemetry_fresh_count": fresh_count,
        "all_telemetry_fresh": backend == "rclpy" and fresh_count == len(selected),
        "stale_or_inactive_drones": stale_drones,
        "shield_infeasible_drones": shield_infeasible,
        "stall_recovery_drones": stall_recovery_drones,
        "px4_failsafe_drones": failsafe_drones,
    }

    return {
        "ok": True,
        "backend": backend,
        "drones": drone_reports,
        "fleet_summary": fleet_summary,
        "warnings": warnings,
        "errors": [],
    }
