"""Offline verification tests for the upgraded 54-second 10-drone visual research demo."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
SCENARIO_PATH = REPO_ROOT / "src" / "swarm_bringup" / "config" / "scenario.json"
WORLD_PATH = DEMO_DIR / "worlds" / "swarm_visual_demo.sdf"
TRAJ_PATH = DEMO_DIR / "trajectories.json"
DASHBOARD_HTML = DEMO_DIR / "dashboard.html"
LAUNCH_SCRIPT = DEMO_DIR / "launch_visual_demo.sh"
PLAYER_SCRIPT = DEMO_DIR / "play_demo.py"
README_FILE = DEMO_DIR / "README.md"

REQUIRED_LABEL = "Scripted illustration of proposed LiDAR and AI-assisted navigation."

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import play_demo  # noqa: E402


class TestUpgradedVisualDemo(unittest.TestCase):
    def test_world_sdf_obstacles_lidar_trails_camera_and_zero_px4_ros_coupling(self) -> None:
        self.assertTrue(WORLD_PATH.is_file(), f"Missing generated world SDF: {WORLD_PATH}")
        sdf_text = WORLD_PATH.read_text(encoding="utf-8")
        self.assertIn(REQUIRED_LABEL, sdf_text)

        root = ET.fromstring(sdf_text)
        self.assertEqual(root.tag, "sdf")
        world = root.find("world")
        self.assertIsNotNone(world)
        assert world is not None
        self.assertEqual(world.attrib.get("name"), "swarm_visual_demo")

        # Strict isolation in serialized XML tree (zero PX4/ROS/MAVLink/DDS/motor/sensor elements).
        xml_no_comments = ET.tostring(root, encoding="unicode").lower()
        for forbidden in (
            "multicoptermotormodel",
            "motorfailure",
            "rclpy",
            "px4_msgs",
            "mavlink",
            "microxrce",
            "fastdds",
            "<sensor",
        ):
            self.assertNotIn(forbidden, xml_no_comments)

        # Improved camera angle configured in <gui><plugin filename="MinimalScene"><camera_pose>.
        gui = world.find("gui")
        self.assertIsNotNone(gui)
        assert gui is not None
        self.assertIn("<camera_pose>", ET.tostring(gui, encoding="unicode"))

        # Verify reused arena models, showcase obstacles of varying shapes/heights, LiDAR rings,
        # obstacle proximity glows, and movement trail beads.
        model_names = {m.attrib.get("name") for m in world.findall("model")}
        for required in (
            "ground_plane",
            "arena_grid",
            "START_ZONE",
            "GOAL_ZONE",
            "showcase_low_bunker",
            "showcase_monolith",
            "showcase_north_block",
            "showcase_south_wall",
            "showcase_radar_tower",
            "showcase_gantry_arch",
        ):
            self.assertIn(required, model_names)

        for i in range(7):
            self.assertIn(f"obstacle_{i}", model_names)
        for i in range(10):
            self.assertIn(f"goal_beacon_{i}", model_names)
            self.assertIn(f"demo_drone_{i}", model_names)
            self.assertIn(f"lidar_clear_{i}", model_names)
            self.assertIn(f"lidar_alert_{i}", model_names)
            self.assertIn(f"trail_{i}_0", model_names)
        self.assertIn("obs_glow_showcase_monolith", model_names)

        # Verify x500_base meshes are reused.
        for mesh_uri in (
            "model://x500_base/meshes/NXP-HGD-CF.dae",
            "model://x500_base/meshes/5010Base.dae",
            "model://x500_base/meshes/5010Bell.dae",
            "model://x500_base/meshes/1345_prop_ccw.stl",
            "model://x500_base/meshes/1345_prop_cw.stl",
        ):
            self.assertIn(mesh_uri, sdf_text)

    def test_trajectories_duration_separation_regrouping_clearance_and_arrival(self) -> None:
        self.assertTrue(TRAJ_PATH.is_file(), f"Missing trajectories: {TRAJ_PATH}")
        scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
        data = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))

        self.assertEqual(data["label"], REQUIRED_LABEL)
        self.assertEqual(data["mode"], "SCRIPTED_VISUALIZATION_ONLY")
        self.assertGreaterEqual(float(data["duration"]), 45.0)
        self.assertLessEqual(float(data["duration"]), 60.0)
        self.assertEqual(len(data["drones"]), 10)
        self.assertGreaterEqual(len(data["trail_schedule"]), 12)

        # Starts match START_ZONE at ground rest z = 0.24 m.
        first = data["frames"][0]["poses"]
        for i, (pose, start) in enumerate(zip(first, scenario["starts"])):
            self.assertAlmostEqual(pose[0], start[0], places=3, msg=f"drone {i} start x")
            self.assertAlmostEqual(pose[1], start[1], places=3, msg=f"drone {i} start y")
            self.assertAlmostEqual(pose[2], 0.24, places=3, msg=f"drone {i} ground z")

        # Final poses arrive cleanly at exact destination markers (goal_beacon_0..9).
        last = data["frames"][-1]["poses"]
        for i, (pose, goal) in enumerate(zip(last, scenario["goals"])):
            err = math.sqrt(
                (pose[0] - goal[0]) ** 2 + (pose[1] - goal[1]) ** 2 + (pose[2] - goal[2]) ** 2
            )
            self.assertLess(err, 1e-3, msg=f"drone {i} final arrival error {err}")

        # Verify formation separation (North Wing vs South Wing centroid spread > 8 m mid-weave)
        # followed by regrouping back to the compact GOAL_ZONE ring (< 4 m wing spread).
        mid_frame = play_demo.sample_frame_state(data, 25.0)
        end_frame = play_demo.sample_frame_state(data, float(data["duration"]))

        def _wing_centroid_dist(state: dict) -> float:
            n_x = sum(d["x"] for d in state["drones"][:5]) / 5.0
            n_y = sum(d["y"] for d in state["drones"][:5]) / 5.0
            s_x = sum(d["x"] for d in state["drones"][5:]) / 5.0
            s_y = sum(d["y"] for d in state["drones"][5:]) / 5.0
            return math.hypot(n_x - s_x, n_y - s_y)

        self.assertGreater(_wing_centroid_dist(mid_frame), 8.0)
        self.assertLess(_wing_centroid_dist(end_frame), 4.0)

        # Verify collision-free visual paths and non-zero scripted obstacle-proximity highlights.
        checks = data["geometry_checks"]
        self.assertGreaterEqual(checks["min_pair_separation_m"], 1.15)
        self.assertGreaterEqual(checks["min_obstacle_clearance_m"], 0.45)
        self.assertGreater(checks["scripted_alert_drone_frames"], 200)

        # Verify smooth velocities (no abrupt teleport jumps).
        dt = float(data["dt"])
        for prev, curr in zip(data["frames"], data["frames"][1:]):
            for p0, p1 in zip(prev["poses"], curr["poses"]):
                step_dist = math.sqrt(
                    (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
                )
                self.assertLess(step_dist / dt, 4.5, "Unexpected jump in prerecorded trajectory")

    def test_player_scene_updates_dashboard_and_strict_honesty_labels(self) -> None:
        data = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))
        mid = play_demo.sample_frame_state(data, 22.0)
        updates = play_demo.build_scene_pose_updates(
            data, mid, newly_dropped_beads=[(0, data["trail_schedule"][0]["positions"])]
        )
        names = {n for n, _ in updates}
        self.assertIn("demo_drone_0", names)
        self.assertIn("lidar_clear_0", names)
        self.assertIn("lidar_alert_0", names)
        self.assertIn("obs_glow_showcase_monolith", names)
        self.assertIn("trail_0_0", names)

        req_str = play_demo.format_pose_vector_req(updates[:12])
        self.assertIn('name: "demo_drone_0"', req_str)

        term = play_demo.render_terminal_dashboard(
            mid, dashboard_url="http://127.0.0.1:8788", backend="python:gz.transport13"
        )
        self.assertIn(REQUIRED_LABEL, term)
        self.assertIn("demo_drone_0", term)
        self.assertIn("demo_drone_9", term)

        html = DASHBOARD_HTML.read_text(encoding="utf-8")
        self.assertIn(REQUIRED_LABEL, html)
        self.assertIn("No live sensor data or AI performance metrics", html)
        self.assertIn(REQUIRED_LABEL, README_FILE.read_text(encoding="utf-8"))

        player_src = PLAYER_SCRIPT.read_text(encoding="utf-8").lower()
        for forbidden in ("import rclpy", "px4_msgs", "swarm_interfaces", "pymavlink", "microxrce"):
            self.assertNotIn(forbidden, player_src)

    def test_launcher_script_syntax_and_isolation(self) -> None:
        proc = subprocess.run(["bash", "-n", str(LAUNCH_SCRIPT)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        launch_src = LAUNCH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(REQUIRED_LABEL, launch_src)
        self.assertIn('GZ_PARTITION="swarm_visual_demo"', launch_src)
        exec_lines = "\n".join(
            line for line in launch_src.splitlines() if not line.lstrip().startswith("#")
        )
        for forbidden in ("MicroXRCEAgent", "ros2 launch", "px4_sitl", "mavlink"):
            self.assertNotIn(forbidden, exec_lines)


if __name__ == "__main__":
    unittest.main(verbosity=2)
