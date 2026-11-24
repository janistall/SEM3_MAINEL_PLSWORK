"""Dynamic multi-drone namespace and PX4 SITL instance discovery.

Avoids hardcoding a fixed drone count (e.g. 3) so the same diagnostics work seamlessly
across 1, 3, 10, or N-drone swarm deployments.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_NATURAL_SPLIT_RE = re.compile(r"(\d+)")
_SWARM_TOPIC_RE = re.compile(
    r"^/(?P<ns>[A-Za-z][A-Za-z0-9_-]*)/(?P<leaf>telemetry|status|cmd_vel|neighbors)$"
)
_PX4_TOPIC_RE = re.compile(
    r"^/(?P<ns>[A-Za-z][A-Za-z0-9_-]*)/fmu/(?P<direction>out|in)/(?P<leaf>[A-Za-z0-9_]+)$"
)
_TRAILING_INDEX_RE = re.compile(r"^(?P<prefix>[A-Za-z][A-Za-z0-9_-]*?)[_-]?(?P<idx>\d+)$")
_ADAPTER_NODE_RE = re.compile(r"^/?adapter[_-](?P<idx>\d+)$")
_CONTROL_NODE_RE = re.compile(r"^/?control[_-](?P<idx>\d+)$")


def natural_sort_key(value: str) -> Tuple[Any, ...]:
    """Sort identifiers naturally so drone_2 precedes drone_10."""
    parts = _NATURAL_SPLIT_RE.split(value)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def extract_index(identifier: str) -> Optional[int]:
    """Extract trailing numeric index from names like 'drone_9' or 'uav_0'."""
    match = _TRAILING_INDEX_RE.match(identifier.lstrip("/"))
    if not match:
        return None
    return int(match.group("idx"))


def discover_from_ros_graph(
    topics: Sequence[str],
    nodes: Sequence[str] = (),
) -> Dict[str, Any]:
    """Dynamically discover swarm drone namespaces, PX4 namespaces, and node roles."""
    swarm_map: Dict[str, Dict[str, Any]] = {}
    px4_map: Dict[str, Dict[str, Any]] = {}

    for topic in topics:
        swarm_match = _SWARM_TOPIC_RE.match(topic)
        if swarm_match:
            ns = swarm_match.group("ns")
            leaf = swarm_match.group("leaf")
            entry = swarm_map.setdefault(
                ns,
                {
                    "drone_id": ns,
                    "index": extract_index(ns),
                    "topics": {},
                },
            )
            entry["topics"][leaf] = topic
            continue

        px4_match = _PX4_TOPIC_RE.match(topic)
        if px4_match:
            ns = px4_match.group("ns")
            direction = px4_match.group("direction")
            leaf = px4_match.group("leaf")
            entry = px4_map.setdefault(
                ns,
                {
                    "px4_namespace": f"/{ns}",
                    "name": ns,
                    "index": extract_index(ns),
                    "out_topics": [],
                    "in_topics": [],
                },
            )
            bucket = "out_topics" if direction == "out" else "in_topics"
            if topic not in entry[bucket]:
                entry[bucket].append(topic)

    # Also detect standalone PX4 topics at root (/fmu/out/*) in single-drone setups.
    root_fmu_out = sorted(t for t in topics if t.startswith("/fmu/out/"))
    root_fmu_in = sorted(t for t in topics if t.startswith("/fmu/in/"))
    if root_fmu_out or root_fmu_in:
        px4_map[""] = {
            "px4_namespace": "/",
            "name": "",
            "index": 0,
            "out_topics": root_fmu_out,
            "in_topics": root_fmu_in,
        }

    adapter_nodes: Dict[int, str] = {}
    control_nodes: Dict[int, str] = {}
    for node in nodes:
        cleaned = node.strip()
        m_adapter = _ADAPTER_NODE_RE.match(cleaned)
        if m_adapter:
            adapter_nodes[int(m_adapter.group("idx"))] = cleaned
        m_control = _CONTROL_NODE_RE.match(cleaned)
        if m_control:
            control_nodes[int(m_control.group("idx"))] = cleaned

    swarm_ids = sorted(swarm_map.keys(), key=natural_sort_key)
    px4_keys = sorted(px4_map.keys(), key=natural_sort_key)

    # Pair swarm drones to PX4 namespaces dynamically by trailing numeric index when unambiguous.
    px4_by_index: Dict[int, str] = {}
    for key in px4_keys:
        idx = px4_map[key]["index"]
        if idx is not None and idx not in px4_by_index:
            px4_by_index[idx] = px4_map[key]["px4_namespace"]

    drones: List[Dict[str, Any]] = []
    for drone_id in swarm_ids:
        entry = swarm_map[drone_id]
        idx = entry["index"]
        drones.append(
            {
                "drone_id": drone_id,
                "index": idx,
                "telemetry_topic": entry["topics"].get("telemetry"),
                "status_topic": entry["topics"].get("status"),
                "cmd_vel_topic": entry["topics"].get("cmd_vel"),
                "neighbors_topic": entry["topics"].get("neighbors"),
                "paired_px4_namespace": px4_by_index.get(idx) if idx is not None else None,
                "adapter_node": adapter_nodes.get(idx) if idx is not None else None,
                "control_node": control_nodes.get(idx) if idx is not None else None,
            }
        )

    px4_namespaces: List[Dict[str, Any]] = []
    for key in px4_keys:
        item = px4_map[key]
        item["out_topics"] = sorted(item["out_topics"])
        item["in_topics"] = sorted(item["in_topics"])
        px4_namespaces.append(
            {
                "px4_namespace": item["px4_namespace"],
                "index": item["index"],
                "out_topic_count": len(item["out_topics"]),
                "in_topic_count": len(item["in_topics"]),
                "has_vehicle_odometry": any(
                    t.endswith("/fmu/out/vehicle_odometry") for t in item["out_topics"]
                ),
                "has_vehicle_status": any(
                    t.endswith("/fmu/out/vehicle_status")
                    or t.endswith("/fmu/out/vehicle_status_v4")
                    for t in item["out_topics"]
                ),
                "has_vehicle_command_ack": any(
                    t.endswith("/fmu/out/vehicle_command_ack")
                    or t.endswith("/fmu/out/vehicle_command_ack_v1")
                    for t in item["out_topics"]
                ),
            }
        )

    return {
        "swarm_drone_count": len(drones),
        "px4_namespace_count": len(px4_namespaces),
        "drones": drones,
        "px4_namespaces": px4_namespaces,
        "adapter_nodes": [adapter_nodes[k] for k in sorted(adapter_nodes)],
        "control_nodes": [control_nodes[k] for k in sorted(control_nodes)],
    }


def discover_px4_rootfs_instances(rootfs_dir: Path) -> List[int]:
    """Discover numeric PX4 SITL instance directories under build/px4_sitl_default/rootfs."""
    if not rootfs_dir.is_dir():
        return []
    instances: List[int] = []
    try:
        for child in rootfs_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                instances.append(int(child.name))
    except OSError:
        return []
    return sorted(instances)


def discover_gazebo_worlds(gz_topics: Iterable[str]) -> List[str]:
    """Extract unique Gazebo world names from `/world/<name>/...` transport topics."""
    worlds = set()
    for topic in gz_topics:
        cleaned = topic.strip()
        if cleaned.startswith("/world/"):
            parts = cleaned.split("/")
            if len(parts) >= 3 and parts[2]:
                worlds.add(parts[2])
    return sorted(worlds, key=natural_sort_key)
