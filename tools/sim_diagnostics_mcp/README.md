# Simulation Diagnostics MCP Bridge (`sim_diagnostics_mcp`)

A standalone, **read-only** Streamable HTTP Model Context Protocol (MCP) server for inspecting a local **ROS 2 Humble**, **Gazebo Harmonic**, and **PX4 SITL** multi-drone simulation stack from Antigravity CLI running inside strict Snap confinement.

* **Transport:** Official Python MCP SDK (`mcp.server.fastmcp.FastMCP`), Streamable HTTP (`2025-03-26`)
* **Endpoint:** Strictly bound to `http://127.0.0.1:8765/mcp`
* **Python compatibility:** Host Python `3.10` (matches ROS 2 Humble `rclpy` ABI)
* **Scale:** Dynamically discovers `1..N` drones (supports 3-drone bringup and 10-drone target architectures without hardcoded drone counts)
* **Isolation:** Completely separate from `src/` and preserves the frozen `swarm-coop-enu-v2` contract

## Safety Guarantees

* **Strictly loopback-only:** `BridgeConfig` refuses any host other than `127.0.0.1` and any port other than `8765`.
* **Strictly read-only:**
  * No arbitrary shell execution (`shell=False` everywhere; fixed allowlist of read-only binaries: `ros2`, `gz`, `ps`, `ulog_info`).
  * No ROS 2 publishers, service clients, action clients, `topic pub`, `service call`, or `param set`.
  * No Gazebo simulation control (`gz sim`, `gz service`, `gz topic -p`).
  * No PX4 vehicle commands (`VehicleCommand`, arm/offboard/land, or simulation start/stop).
  * No physical drone access (never touches serial ports, hardware devices, or non-local MAVLink endpoints; pins `GZ_IP=127.0.0.1`).
* **Bounded execution & path containment:** All subprocesses and ROS 2 waits enforce bounded timeouts (`0.5..15.0 s`), output byte caps (`64 KiB`), and canonical path checks confining log reads to PX4 SITL `rootfs/`, `~/.ros/log`, and `<workspace>/log`.

## Exposed Read-Only MCP Tools

| Tool | Purpose |
| :--- | :--- |
| `sim_check_dependencies` | Checks Python 3.10, `mcp`, `rclpy`, `swarm_interfaces`, `px4_msgs`, `ros2`, `gz`, `MicroXRCEAgent`, PX4 SITL binary, and environment variables. |
| `sim_discover_drones` | Dynamically discovers active swarm drone namespaces (`/<drone>/telemetry`, `/<drone>/status`), PX4 uXRCE-DDS namespaces (`/<uav>/fmu/out/*`), `adapter_*`/`control_*` nodes, and PX4 SITL `rootfs/<id>` instances. |
| `ros2_graph_overview` | Lists ROS 2 nodes, topics with types, categorized topics (swarm/PX4/diagnostics), and dynamic drone discovery. |
| `ros2_node_details` | Read-only `ros2 node info` and optional `ros2 param list` for one validated node. |
| `ros2_topic_inspect` | Read-only `ros2 topic info --verbose` (QoS inspection) plus optional bounded single-message sample (`--once`) with automatic `best_effort` QoS for `/fmu/*` topics. |
| `swarm_diagnostics_snapshot` | Concurrent `rclpy` read-only observer (using `RELIABLE` QoS for swarm topics and `BEST_EFFORT / VOLATILE / depth=1` for PX4 `/fmu/out/*`) reporting per-drone telemetry freshness, controller supervisor mode / `shield_feasible` / `stall_recovery`, PX4 arming/offboard/failsafe status, and fleet-level summary. |
| `gazebo_world_stats` | Lists Gazebo Harmonic topics in the selected partition (`GZ_IP=127.0.0.1`), dynamically detects worlds, reads `/world/<world>/stats` (sim time, real time, RTF, pause state, iterations), and lists models. |
| `gazebo_topic_inspect` | Read-only `gz topic -i` and optional single-message sample (`gz topic -e -n 1`). |
| `px4_sitl_status` | Reports PX4 SITL build/airframe readiness, discovered `rootfs/<id>` instances, and running local simulation processes (`px4`, `MicroXRCEAgent`, `gz sim`, `px4_adapter`, `swarm_control`). |
| `px4_sitl_logs` | Lists recent PX4 SITL `.ulg`/text logs and ROS 2 logs, or reads bounded tail lines / `ulog_info` metadata for one allowlisted log file. |

## Host Installation (Outside Snap)

Run these commands in a **normal host terminal** (Ubuntu host Python 3.10.12, NOT inside Antigravity's Snap shell).

Create an isolated virtual environment that **inherits system site-packages** so it can import ROS 2 Humble's system Python 3.10 `rclpy` alongside the MCP SDK:

```bash
cd /home/siddharthg/SEM3_MAINEL_PLSWORK
python3 -m venv --system-site-packages tools/sim_diagnostics_mcp/.venv
tools/sim_diagnostics_mcp/.venv/bin/python3 -m pip install --upgrade pip
tools/sim_diagnostics_mcp/.venv/bin/python3 -m pip install -r tools/sim_diagnostics_mcp/requirements.txt
```

> **Note:** `--system-site-packages` is required so the venv's Python 3.10 can see `/opt/ros/humble/lib/python3.10/site-packages` (`rclpy`) and your sourced colcon overlays (`swarm_interfaces`, `px4_msgs`) without touching system pip packages.
> Optional: install `pyulog` inside the same venv (`tools/sim_diagnostics_mcp/.venv/bin/python3 -m pip install pyulog`) if you want `ulog_info` header summaries for `.ulg` flight logs.

## Host Startup Commands

1. **Run the read-only dependency check first:**
   ```bash
   source /opt/ros/humble/setup.bash
   # Optional: source your px4_msgs underlay if built separately
   # Optional: source /home/siddharthg/SEM3_MAINEL_PLSWORK/install/setup.bash
   bash /home/siddharthg/SEM3_MAINEL_PLSWORK/tools/sim_diagnostics_mcp/run_bridge.sh --check
   ```

2. **Start the Streamable HTTP MCP bridge (binds strictly to `127.0.0.1:8765`):**
   ```bash
   source /opt/ros/humble/setup.bash
   # Optional: source your px4_msgs underlay and workspace install/setup.bash
   bash /home/siddharthg/SEM3_MAINEL_PLSWORK/tools/sim_diagnostics_mcp/run_bridge.sh
   ```
   Or directly with Python:
   ```bash
   source /opt/ros/humble/setup.bash
   PYTHONPATH=/home/siddharthg/SEM3_MAINEL_PLSWORK/tools/sim_diagnostics_mcp \
     /home/siddharthg/SEM3_MAINEL_PLSWORK/tools/sim_diagnostics_mcp/.venv/bin/python3 \
     -m sim_diagnostics_mcp
   ```

   Environment variables supported by `run_bridge.sh`:
   * `PX4_DIR` — path to PX4-Autopilot (default: `/home/siddharthg/PX4-Autopilot`)
   * `PX4_MSGS_SETUP` — optional path to a separate `px4_msgs` overlay `setup.bash`
   * `SWARM_WORKSPACE_DIR` — workspace root (default: auto-detected repo root)
   * `ROS_LOG_DIR` — ROS 2 log directory (default: `~/.ros/log`)

## Registering the Server with Antigravity CLI (Snap)

Because Antigravity CLI runs under Snap confinement with `HOME=/home/siddharthg/snap/antigravity-cli/common`, its global MCP configuration file is:

```text
/home/siddharthg/snap/antigravity-cli/common/.gemini/config/mcp_config.json
```

Add the `swarm-sim-diagnostics` entry inside `"mcpServers"` (preserving any existing entries such as `github`):

```json
{
  "mcpServers": {
    "swarm-sim-diagnostics": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

Then start the bridge in your host terminal and restart or reload Antigravity CLI. You can verify the registered tools in Antigravity via **Additional Options (...) > MCP Servers** or by asking the agent to run `sim_check_dependencies`.

## Running Tests

The unit test suite uses only the Python standard library and requires no live ROS 2, Gazebo, PX4, or `mcp` installation:

```bash
PYTHONPATH=/home/siddharthg/SEM3_MAINEL_PLSWORK/tools/sim_diagnostics_mcp \
  python3 -m unittest discover -s /home/siddharthg/SEM3_MAINEL_PLSWORK/tools/sim_diagnostics_mcp/tests -v
```
