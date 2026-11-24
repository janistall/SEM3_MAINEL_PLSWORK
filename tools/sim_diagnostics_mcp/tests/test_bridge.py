"""Comprehensive read-only unit tests for sim_diagnostics_mcp.

Requires only the Python standard library (no live ROS 2, Gazebo, PX4, or mcp needed).
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sim_diagnostics_mcp.config import ALLOWED_HOST, ALLOWED_PORT, BridgeConfig
from sim_diagnostics_mcp.discovery import (
    discover_from_ros_graph,
    discover_gazebo_worlds,
    discover_px4_rootfs_instances,
)
from sim_diagnostics_mcp.gazebo_diagnostics import (
    get_gazebo_world_stats,
    inspect_gazebo_topic,
    parse_model_list,
    parse_world_stats,
)
from sim_diagnostics_mcp.health import check_dependencies
from sim_diagnostics_mcp.px4_diagnostics import (
    get_px4_sitl_status,
    inspect_px4_sitl_logs,
    parse_sim_processes,
)
from sim_diagnostics_mcp.ros2_diagnostics import (
    get_ros2_graph_overview,
    get_ros2_node_details,
    get_swarm_diagnostics_snapshot,
    inspect_ros2_topic,
    parse_topic_info_verbose,
)
from sim_diagnostics_mcp.runner import CommandResult, _verify_allowlisted_command
from sim_diagnostics_mcp.server import create_server
from sim_diagnostics_mcp.validation import (
    ValidationError,
    validate_drone_ids,
    validate_gz_partition,
    validate_gz_topic,
    validate_gz_world,
    validate_instance_id,
    validate_log_path,
    validate_ros_name,
    validate_timeout,
)


def _ok_result(executable: str, args: List[str], stdout: str = "") -> CommandResult:
    return CommandResult(
        ok=True,
        executable=executable,
        args=tuple(args),
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _err_result(
    executable: str,
    args: List[str],
    *,
    error: str = "failed",
    timed_out: bool = False,
) -> CommandResult:
    return CommandResult(
        ok=False,
        executable=executable,
        args=tuple(args),
        returncode=None if timed_out else 1,
        stdout="",
        stderr=error,
        timed_out=timed_out,
        error=error,
    )


class TestConfigAndValidation(unittest.TestCase):
    def test_strict_loopback_binding(self) -> None:
        cfg = BridgeConfig()
        self.assertEqual(cfg.host, ALLOWED_HOST)
        self.assertEqual(cfg.port, ALLOWED_PORT)
        self.assertEqual(cfg.endpoint_url, "http://127.0.0.1:8765/mcp")

        for bad_host in ("0.0.0.0", "localhost", "::1", "192.168.1.10"):
            with self.assertRaises(ValueError):
                BridgeConfig(host=bad_host)

        for bad_port in (8000, 8766, 0, -1):
            with self.assertRaises(ValueError):
                BridgeConfig(port=bad_port)

    def test_timeout_validation(self) -> None:
        self.assertEqual(validate_timeout(4.0), 4.0)
        for bad in (0.1, 16.0, float("nan"), float("inf"), True, "abc"):
            with self.assertRaises(ValidationError):
                validate_timeout(bad)  # type: ignore[arg-type]

    def test_ros_and_gazebo_name_validation(self) -> None:
        self.assertEqual(validate_ros_name("/drone_9/telemetry"), "/drone_9/telemetry")
        for bad in ("drone_0/telemetry", "/drone_0;rm -rf /", "/a//b", "/a/../b", ""):
            with self.assertRaises(ValidationError):
                validate_ros_name(bad)

        self.assertEqual(validate_drone_ids(["drone_0", "/drone_9", "drone_0"]), ["drone_0", "drone_9"])
        with self.assertRaises(ValidationError):
            validate_drone_ids("drone_0")  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            validate_drone_ids(["../bad"])

        self.assertEqual(validate_gz_world("swarm"), "swarm")
        with self.assertRaises(ValidationError):
            validate_gz_world("../etc")

        self.assertEqual(validate_gz_partition("swarm"), "swarm")
        self.assertEqual(validate_gz_partition(""), "")
        with self.assertRaises(ValidationError):
            validate_gz_partition("swarm;id")

        self.assertEqual(validate_gz_topic("/world/swarm/stats"), "/world/swarm/stats")
        with self.assertRaises(ValidationError):
            validate_gz_topic("world/swarm/stats")

        self.assertEqual(validate_instance_id(9), 9)
        with self.assertRaises(ValidationError):
            validate_instance_id(-1)
        with self.assertRaises(ValidationError):
            validate_instance_id(True)  # type: ignore[arg-type]

    def test_runner_allowlist_blocks_mutations_and_shell(self) -> None:
        # Allowed read-only commands pass verification
        _verify_allowlisted_command("ros2", ["topic", "list", "-t"])
        _verify_allowlisted_command("ros2", ["topic", "echo", "/drone_0/status", "--once"])
        _verify_allowlisted_command("gz", ["topic", "-e", "-n", "1", "-t", "/world/swarm/stats"])
        _verify_allowlisted_command("ps", ["-eo", "pid,ppid,stat,pcpu,pmem,etime,args", "--no-headers"])

        # Disallowed executables and mutating subcommands raise ValidationError
        for exe, args in (
            ("bash", ["-c", "echo hi"]),
            ("sh", ["-c", "id"]),
            ("ros2", ["topic", "pub", "/drone_0/cmd_vel", "geometry_msgs/msg/Twist"]),
            ("ros2", ["service", "call", "/drone_0/arm", "std_srvs/srv/SetBool"]),
            ("ros2", ["param", "set", "/control_0", "mode", "zero"]),
            ("ros2", ["launch", "swarm_bringup", "swarm.launch.py"]),
            ("gz", ["sim", "-r", "-s", "swarm.sdf"]),
            ("gz", ["service", "-s", "/world/swarm/control"]),
            ("gz", ["topic", "-p", "-t", "/cmd"]),
            ("ps", ["aux"]),
            ("ulog_info", ["--help"]),
        ):
            with self.assertRaises(ValidationError, msg=f"Should reject {exe} {args}"):
                _verify_allowlisted_command(exe, args)

    def test_log_path_containment_blocks_traversal_and_escaping_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()

            good_file = root / "sim.log"
            good_file.write_text("ok\n", encoding="utf-8")
            secret_file = outside / "secret.log"
            secret_file.write_text("secret\n", encoding="utf-8")

            escape_link = root / "escape.log"
            escape_link.symlink_to(secret_file)

            self.assertEqual(validate_log_path(str(good_file), [root]), good_file.resolve())

            with self.assertRaises(ValidationError):
                validate_log_path(str(secret_file), [root])
            with self.assertRaises(ValidationError):
                validate_log_path(str(root / ".." / "outside" / "secret.log"), [root])
            with self.assertRaises(ValidationError):
                validate_log_path(str(escape_link), [root])


class TestDynamicTenDroneDiscovery(unittest.TestCase):
    def test_discovers_ten_plus_drones_with_natural_sorting(self) -> None:
        topics: List[str] = []
        nodes: List[str] = []
        # Deliberately create 12 drones in shuffled order to verify natural sort (drone_2 < drone_10)
        for idx in (10, 2, 0, 9, 1, 11, 5, 3, 8, 4, 7, 6):
            topics.extend(
                [
                    f"/drone_{idx}/telemetry",
                    f"/drone_{idx}/status",
                    f"/drone_{idx}/cmd_vel",
                    f"/uav_{idx}/fmu/out/vehicle_odometry",
                    f"/uav_{idx}/fmu/out/vehicle_status_v4",
                    f"/uav_{idx}/fmu/in/trajectory_setpoint",
                ]
            )
            nodes.extend([f"/adapter_{idx}", f"/control_{idx}"])

        discovered = discover_from_ros_graph(topics, nodes)
        self.assertEqual(discovered["swarm_drone_count"], 12)
        self.assertEqual(discovered["px4_namespace_count"], 12)
        drone_ids = [d["drone_id"] for d in discovered["drones"]]
        self.assertEqual(drone_ids, [f"drone_{i}" for i in range(12)])
        self.assertEqual(discovered["drones"][9]["paired_px4_namespace"], "/uav_9")
        self.assertEqual(discovered["drones"][9]["adapter_node"], "/adapter_9")
        self.assertEqual(discovered["drones"][9]["control_node"], "/control_9")
        self.assertTrue(discovered["px4_namespaces"][9]["has_vehicle_odometry"])
        self.assertTrue(discovered["px4_namespaces"][9]["has_vehicle_status"])

    def test_discovers_px4_rootfs_instances_dynamically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rootfs = Path(tmp) / "rootfs"
            rootfs.mkdir()
            for name in ("0", "1", "2", "9", "10", "etc", "fs"):
                (rootfs / name).mkdir()
            self.assertEqual(discover_px4_rootfs_instances(rootfs), [0, 1, 2, 9, 10])


class TestRos2Diagnostics(unittest.TestCase):
    def _ten_drone_runner(self) -> Callable[..., CommandResult]:
        topic_lines = []
        node_lines = []
        for idx in range(10):
            topic_lines.append(f"/drone_{idx}/telemetry [swarm_interfaces/msg/AgentTelemetry]")
            topic_lines.append(f"/drone_{idx}/status [std_msgs/msg/String]")
            topic_lines.append(f"/drone_{idx}/cmd_vel [geometry_msgs/msg/Twist]")
            topic_lines.append(f"/uav_{idx}/fmu/out/vehicle_odometry [px4_msgs/msg/VehicleOdometry]")
            topic_lines.append(f"/uav_{idx}/fmu/out/vehicle_status_v4 [px4_msgs/msg/VehicleStatus]")
            node_lines.append(f"/adapter_{idx}")
            node_lines.append(f"/control_{idx}")
        topic_lines.append("/diagnostics [diagnostic_msgs/msg/DiagnosticArray]")

        def fake_runner(executable: str, args: List[str], **kwargs: Any) -> CommandResult:
            if executable == "ros2" and args == ["topic", "list", "-t"]:
                return _ok_result(executable, args, "\n".join(topic_lines))
            if executable == "ros2" and args == ["node", "list"]:
                return _ok_result(executable, args, "\n".join(node_lines))
            if executable == "ros2" and args[:2] == ["node", "info"]:
                return _ok_result(executable, args, f"{args[2]}\n  Subscribers:\n    /drone_0/telemetry")
            if executable == "ros2" and args[:2] == ["param", "list"]:
                return _ok_result(executable, args, f"{args[2]}:\n  mode\n  enable_shield\n")
            if executable == "ros2" and args[:2] == ["topic", "info"]:
                rel = "BEST_EFFORT" if "/fmu/" in args[2] else "RELIABLE"
                return _ok_result(
                    executable,
                    args,
                    (
                        "Type: swarm_interfaces/msg/AgentTelemetry\n"
                        "Publisher count: 1\n"
                        "Subscription count: 10\n"
                        "Node name: adapter_0\n"
                        "Endpoint type: PUBLISHER\n"
                        f"Reliability: {rel}\n"
                    ),
                )
            if executable == "ros2" and args[:2] == ["topic", "echo"]:
                return _ok_result(executable, args, "drone_id: drone_0\nactive_status: true\n---\n")
            return _err_result(executable, args, error=f"unexpected {executable} {args}")

        return fake_runner

    def test_graph_overview_and_node_details(self) -> None:
        cfg = BridgeConfig()
        runner = self._ten_drone_runner()

        graph = get_ros2_graph_overview(cfg, runner=runner)
        self.assertTrue(graph["ok"])
        self.assertEqual(graph["node_count"], 20)
        self.assertEqual(graph["discovery"]["swarm_drone_count"], 10)
        self.assertEqual(len(graph["categories"]["swarm_telemetry"]), 10)
        self.assertEqual(len(graph["categories"]["diagnostics"]), 1)

        node_info = get_ros2_node_details(cfg, "/control_0", runner=runner)
        self.assertTrue(node_info["ok"])
        self.assertEqual(node_info["parameters"], ["mode", "enable_shield"])

    def test_topic_inspect_auto_qos_and_parsing(self) -> None:
        cfg = BridgeConfig()
        recorded_calls: List[List[str]] = []
        base_runner = self._ten_drone_runner()

        def tracking_runner(executable: str, args: List[str], **kwargs: Any) -> CommandResult:
            recorded_calls.append(list(args))
            return base_runner(executable, args, **kwargs)

        px4_res = inspect_ros2_topic(
            cfg,
            "/uav_7/fmu/out/vehicle_status_v4",
            include_sample=True,
            runner=tracking_runner,
        )
        self.assertTrue(px4_res["ok"])
        self.assertEqual(px4_res["sample_qos_reliability"], "best_effort")
        self.assertIn("--qos-reliability", recorded_calls[-1])
        self.assertIn("best_effort", recorded_calls[-1])

        swarm_res = inspect_ros2_topic(
            cfg,
            "/drone_7/telemetry",
            include_sample=True,
            runner=tracking_runner,
        )
        self.assertEqual(swarm_res["sample_qos_reliability"], "reliable")
        self.assertEqual(swarm_res["info"]["publisher_count"], 1)
        self.assertEqual(swarm_res["info"]["subscription_count"], 10)

        parsed = parse_topic_info_verbose("Type: std_msgs/msg/String\nPublisher count: 1\nSubscription count: 2\n")
        self.assertEqual(parsed["type"], "std_msgs/msg/String")

    def test_swarm_snapshot_across_ten_drones_detects_stale_and_shield_infeasible(self) -> None:
        cfg = BridgeConfig()
        runner = self._ten_drone_runner()
        now = time.monotonic()

        def fake_sampler(drones: Any, px4_topics: Any, timeout: float) -> Any:
            samples: Dict[str, Dict[str, Any]] = {
                "telemetry": {},
                "status": {},
                "px4_status": {},
                "px4_odometry": {},
                "px4_ack": {},
            }
            for idx, drone in enumerate(drones):
                did = drone["drone_id"]
                # Make drone_3 stale (>0.5s) and drone_8 report shield_feasible=False + failsafe
                rx_time = now - 1.2 if idx == 3 else now - 0.05
                samples["telemetry"][did] = {
                    "drone_id": did,
                    "position_enu": [float(idx), 0.0, 2.5],
                    "velocity_enu": [0.1, 0.0, 0.0],
                    "yaw_rad": 0.0,
                    "active_status": True,
                    "_rx_monotonic": rx_time,
                }
                samples["status"][did] = {
                    "drone_id": did,
                    "mode": "APPROACH",
                    "shield_feasible": idx != 8,
                    "stall_recovery": idx == 5,
                    "_rx_monotonic": now - 0.1,
                }
                samples["px4_status"][did] = {
                    "arming_state": 2,
                    "arming_state_label": "ARMED",
                    "nav_state": 14,
                    "nav_state_label": "OFFBOARD",
                    "pre_flight_checks_pass": True,
                    "failsafe": idx == 8,
                    "_rx_monotonic": now - 0.05,
                }
            return {
                "samples": samples,
                "missing_message_types": [],
                "snapshot_monotonic": now,
            }, None

        snap = get_swarm_diagnostics_snapshot(
            cfg,
            telemetry_stale_sec=0.5,
            runner=runner,
            rclpy_sampler=fake_sampler,
        )
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["backend"], "rclpy")
        summary = snap["fleet_summary"]
        self.assertEqual(summary["discovered_drone_count"], 10)
        self.assertEqual(summary["telemetry_fresh_count"], 9)
        self.assertFalse(summary["all_telemetry_fresh"])
        self.assertEqual(summary["stale_or_inactive_drones"], ["drone_3"])
        self.assertEqual(summary["shield_infeasible_drones"], ["drone_8"])
        self.assertEqual(summary["stall_recovery_drones"], ["drone_5"])
        self.assertEqual(summary["px4_failsafe_drones"], ["drone_8"])

        # Fallback path when rclpy is unavailable
        fallback = get_swarm_diagnostics_snapshot(
            cfg,
            drone_ids=["drone_0", "drone_9"],
            runner=runner,
            rclpy_sampler=lambda d, p, t: (None, "ModuleNotFoundError: No module named 'rclpy'"),
        )
        self.assertTrue(fallback["ok"])
        self.assertEqual(fallback["backend"], "graph_only_fallback")
        self.assertEqual(fallback["fleet_summary"]["selected_drone_count"], 2)
        self.assertTrue(fallback["warnings"])


class TestGazeboDiagnostics(unittest.TestCase):
    def test_world_discovery_stats_parsing_and_loopback_env(self) -> None:
        cfg = BridgeConfig()
        seen_envs: List[Dict[str, str]] = []

        stats_output = (
            "sim_time {\n  sec: 42\n  nsec: 500000000\n}\n"
            "real_time {\n  sec: 44\n  nsec: 250000000\n}\n"
            "pause_time {\n  sec: 0\n  nsec: 0\n}\n"
            "paused: false\n"
            "iterations: 10625\n"
            "real_time_factor: 0.96\n"
            "step_size {\n  sec: 0\n  nsec: 4000000\n}\n"
        )

        def fake_runner(executable: str, args: List[str], **kwargs: Any) -> CommandResult:
            seen_envs.append(dict(kwargs.get("env_overrides") or {}))
            if args == ["topic", "-l"]:
                return _ok_result(
                    executable,
                    args,
                    "/world/swarm/stats\n/world/swarm/clock\n/world/swarm/pose/info\n",
                )
            if args == ["topic", "-e", "-n", "1", "-t", "/world/swarm/stats"]:
                return _ok_result(executable, args, stats_output)
            if args == ["model", "--list"]:
                return _ok_result(
                    executable,
                    args,
                    "Available models:\n    - ground_plane\n    - x500_0\n    - x500_9\n    - x500_2\n",
                )
            if args == ["topic", "-i", "-t", "/world/swarm/clock"]:
                return _ok_result(executable, args, "Publishers [Address, Message Type]:\n  tcp://127.0.0.1:41234, gz.msgs.Clock\n")
            if args == ["topic", "-e", "-n", "1", "-t", "/world/swarm/clock"]:
                return _ok_result(executable, args, "sim {\n  sec: 42\n}\n")
            return _err_result(executable, args)

        res = get_gazebo_world_stats(cfg, runner=fake_runner)
        self.assertTrue(res["ok"])
        self.assertEqual(res["discovered_worlds"], ["swarm"])
        self.assertEqual(res["selected_world"], "swarm")
        self.assertAlmostEqual(res["stats"]["sim_time_sec"], 42.5)
        self.assertAlmostEqual(res["stats"]["real_time_factor"], 0.96)
        self.assertFalse(res["stats"]["paused"])
        self.assertEqual(res["stats"]["iterations"], 10625)
        self.assertEqual(res["models"], ["ground_plane", "x500_0", "x500_2", "x500_9"])
        self.assertTrue(all(env.get("GZ_IP") == "127.0.0.1" for env in seen_envs))
        self.assertTrue(all(env.get("GZ_PARTITION") == "swarm" for env in seen_envs))

        topic_res = inspect_gazebo_topic(cfg, "/world/swarm/clock", include_sample=True, runner=fake_runner)
        self.assertTrue(topic_res["ok"])
        self.assertIn("sec: 42", topic_res["sample"])

        # Direct parser checks
        self.assertEqual(discover_gazebo_worlds(["/world/alpha/stats", "/world/beta/clock"]), ["alpha", "beta"])
        self.assertEqual(parse_world_stats(stats_output)["step_size_sec"], 0.004)
        self.assertEqual(parse_model_list("- x500_10\n- x500_2\n"), ["x500_2", "x500_10"])


class TestPx4SitlDiagnosticsAndLogs(unittest.TestCase):
    def test_process_parsing_and_build_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            px4_dir = Path(tmp) / "PX4-Autopilot"
            sitl_bin = px4_dir / "build" / "px4_sitl_default" / "bin" / "px4"
            sitl_bin.parent.mkdir(parents=True)
            sitl_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(sitl_bin, 0o755)
            (px4_dir / "build" / ".last_target").write_text("px4_sitl_default\n", encoding="utf-8")

            airframe = px4_dir / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes" / "4002_gz_x500_swarm"
            airframe.parent.mkdir(parents=True)
            airframe.write_text("# airframe\n", encoding="utf-8")

            for i in range(10):
                (px4_dir / "build" / "px4_sitl_default" / "rootfs" / str(i) / "log").mkdir(parents=True)

            cfg = BridgeConfig(px4_dir=px4_dir, workspace_dir=Path(tmp) / "ws", ros_log_dir=Path(tmp) / "roslog")

            ps_lines = [
                "  1001 1000 S  4.1 0.5 01:12 MicroXRCEAgent udp4 -p 8888",
                "  1002 1000 S 25.0 3.2 01:12 gz sim -r -s /path/swarm.sdf --verbose=1",
                "  1099 1000 S  0.1 0.1 05:00 /usr/bin/unrelated_daemon --foo",
            ]
            for i in range(10):
                ps_lines.append(
                    f"  {2000 + i} 1000 S  3.0 0.8 01:05 {sitl_bin} -i {i}"
                )
                ps_lines.append(
                    f"  {3000 + i} 1000 S  1.0 0.4 01:05 python3 /ws/install/px4_adapter/lib/px4_adapter/adapter_node"
                )
                ps_lines.append(
                    f"  {4000 + i} 1000 S  1.2 0.5 01:05 python3 /ws/install/swarm_control/lib/swarm_control/controller"
                )

            status = get_px4_sitl_status(
                cfg,
                runner=lambda exe, args, **kw: _ok_result(exe, args, "\n".join(ps_lines)),
            )
            self.assertTrue(status["ok"])
            self.assertTrue(status["build"]["sitl_binary_executable"])
            self.assertTrue(status["build"]["swarm_airframe_installed"])
            self.assertEqual(status["build"]["rootfs_instance_count"], 10)
            self.assertEqual(status["summary"]["running_px4_sitl_count"], 10)
            self.assertEqual(status["summary"]["running_px4_instance_ids"], list(range(10)))
            self.assertTrue(status["summary"]["micro_xrce_agent_running"])
            self.assertTrue(status["summary"]["gazebo_sim_running"])
            self.assertEqual(status["summary"]["px4_adapter_count"], 10)
            self.assertEqual(status["summary"]["swarm_control_count"], 10)

            parsed_ps = parse_sim_processes("\n".join(ps_lines))
            self.assertEqual(len(parsed_ps["px4_sitl"]), 10)

    def test_log_listing_tail_and_ulog_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            px4_dir = Path(tmp) / "PX4-Autopilot"
            ros_log = Path(tmp) / "roslog"
            ws_dir = Path(tmp) / "ws"

            log9_dir = px4_dir / "build" / "px4_sitl_default" / "rootfs" / "9" / "log"
            log9_dir.mkdir(parents=True)
            ros_log.mkdir(parents=True)

            ulog_file = log9_dir / "2026-11-24_00_01.ulg"
            ulog_file.write_bytes(b"ULog\x01\x12\x35\x01")
            txt_log = ros_log / "launch.log"
            txt_log.write_text("\n".join(f"line {i}" for i in range(1, 21)) + "\n", encoding="utf-8")

            cfg = BridgeConfig(px4_dir=px4_dir, workspace_dir=ws_dir, ros_log_dir=ros_log)

            listed_all = inspect_px4_sitl_logs(cfg)
            self.assertTrue(listed_all["ok"])
            self.assertEqual(listed_all["count"], 2)

            listed_inst9 = inspect_px4_sitl_logs(cfg, instance_id=9)
            self.assertEqual(listed_inst9["count"], 1)
            self.assertEqual(listed_inst9["logs"][0]["instance_id"], 9)

            tail_res = inspect_px4_sitl_logs(cfg, log_path=str(txt_log), tail_lines=5)
            self.assertTrue(tail_res["ok"])
            self.assertEqual(tail_res["returned_lines"], 5)
            self.assertEqual(tail_res["content"], "line 16\nline 17\nline 18\nline 19\nline 20")

            ulog_res = inspect_px4_sitl_logs(
                cfg,
                log_path=str(ulog_file),
                runner=lambda exe, args, **kw: _ok_result(exe, args, "Logging start time: 0:00:01\nDuration: 0:00:30\n"),
            )
            self.assertTrue(ulog_res["ok"])
            self.assertTrue(ulog_res["ulog_info_available"])
            self.assertIn("Duration: 0:00:30", ulog_res["ulog_summary"])

            # Path outside allowed roots is rejected
            outside = Path(tmp) / "outside.log"
            outside.write_text("nope", encoding="utf-8")
            with self.assertRaises(ValidationError):
                inspect_px4_sitl_logs(cfg, log_path=str(outside))


class TestHealthAndServerWiring(unittest.TestCase):
    def test_dependency_health_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            px4_dir = Path(tmp) / "PX4-Autopilot"
            sitl_bin = px4_dir / "build" / "px4_sitl_default" / "bin" / "px4"
            sitl_bin.parent.mkdir(parents=True)
            sitl_bin.write_text("", encoding="utf-8")
            cfg = BridgeConfig(px4_dir=px4_dir, workspace_dir=Path(tmp) / "ws", ros_log_dir=Path(tmp) / "roslog")

            report = check_dependencies(
                cfg,
                which_fn=lambda name: f"/usr/bin/{name}",
                module_checker=lambda name: {"available": True, "version": "1.0"},
                runner=lambda exe, args, **kw: _ok_result(exe, args, "Gazebo Sim, version 8.15.0\n"),
            )
            self.assertTrue(report["ok"])
            self.assertTrue(report["ready_for_live_diagnostics"])
            self.assertEqual(report["gazebo_version"], "Gazebo Sim, version 8.15.0")
            self.assertEqual(report["bridge"]["endpoint_url"], "http://127.0.0.1:8765/mcp")
            self.assertTrue(report["bridge"]["read_only"])
            self.assertFalse(report["bridge"]["physical_drone_access"])

    def test_create_server_registers_only_readonly_tools_on_strict_loopback(self) -> None:
        class FakeSettings:
            def __init__(self, host: str, port: int, streamable_http_path: str) -> None:
                self.host = host
                self.port = port
                self.streamable_http_path = streamable_http_path

        class FakeFastMCP:
            def __init__(
                self,
                name: str,
                *,
                instructions: str = "",
                host: str = "",
                port: int = 0,
                streamable_http_path: str = "",
            ) -> None:
                self.name = name
                self.instructions = instructions
                self.settings = FakeSettings(host, port, streamable_http_path)
                self.tools: Dict[str, Callable[..., Any]] = {}

            def tool(self, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                    self.tools[fn.__name__] = fn
                    return fn

                return decorator

        cfg = BridgeConfig()
        server = create_server(
            cfg,
            fastmcp_cls=FakeFastMCP,
            runner=lambda exe, args, **kw: _ok_result(exe, args, ""),
        )

        self.assertEqual(server.settings.host, "127.0.0.1")
        self.assertEqual(server.settings.port, 8765)
        self.assertEqual(server.settings.streamable_http_path, "/mcp")

        expected_tools = {
            "sim_check_dependencies",
            "sim_discover_drones",
            "ros2_graph_overview",
            "ros2_node_details",
            "ros2_topic_inspect",
            "swarm_diagnostics_snapshot",
            "gazebo_world_stats",
            "gazebo_topic_inspect",
            "px4_sitl_status",
            "px4_sitl_logs",
        }
        self.assertEqual(set(server.tools.keys()), expected_tools)

        # Tool wrappers convert ValidationError into structured error responses instead of raising
        bad_resp = server.tools["ros2_topic_inspect"]("invalid_topic;rm -rf /")
        self.assertFalse(bad_resp["ok"])
        self.assertEqual(bad_resp["error_type"], "validation_error")

        bad_log = server.tools["px4_sitl_logs"](log_path="/etc/passwd")
        self.assertFalse(bad_log["ok"])
        self.assertEqual(bad_log["error_type"], "validation_error")


if __name__ == "__main__":
    unittest.main()
