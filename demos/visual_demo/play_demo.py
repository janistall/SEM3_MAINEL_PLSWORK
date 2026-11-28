#!/usr/bin/env python3
"""Prerecorded animation & live dashboard player for the 10-drone Gazebo visual demo.

Scripted illustration of proposed LiDAR and AI-assisted navigation.
Replays fixed prerecorded poses, spinning LiDAR scan rings/rays, obstacle proximity
highlights, and growing movement trails over Gazebo Harmonic's
`/world/swarm_visual_demo/set_pose_vector` service. Completely disconnected from
PX4, ROS 2 flight-control topics, MAVLink, DDS, live sensing, and policy inference.
"""
from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJECTORIES = DEMO_DIR / "trajectories.json"
DASHBOARD_HTML = DEMO_DIR / "dashboard.html"
HIDDEN_Z = -12.0
REQUIRED_LABEL = "Scripted illustration of proposed LiDAR and AI-assisted navigation."


def _euler_yaw_quat(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]


def _slerp(
    q0: list[float], q1: list[float], alpha: float
) -> tuple[float, float, float, float]:
    """Shortest-arc spherical linear quaternion interpolation."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
        dot = -dot
    if dot > 0.9995:
        x = x0 + alpha * (x1 - x0)
        y = y0 + alpha * (y1 - y0)
        z = z0 + alpha * (z1 - z0)
        w = w0 + alpha * (w1 - w0)
    else:
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * alpha
        s0 = math.sin(theta_0 - theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        x = s0 * x0 + s1 * x1
        y = s0 * y0 + s1 * y1
        z = s0 * z0 + s1 * z1
        w = s0 * w0 + s1 * w1
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


def current_act(data: dict[str, Any], t_sec: float) -> tuple[str, str]:
    for act in data.get("acts", []):
        if float(act["start"]) <= t_sec <= float(act["end"]):
            aid = str(act["id"])
            short_map = {
                "ground_staging": "STAGED",
                "takeoff": "TAKEOFF",
                "separation": "SPLITTING",
                "avoidance_weave": "WEAVING",
                "regrouping": "REGROUPING",
                "arrival_hold": "ARRIVED",
            }
            return str(act["name"]), short_map.get(aid, "IN FLIGHT")
    return "Destination Arrival & Final Hold", "ARRIVED"


def sample_frame_state(data: dict[str, Any], t_sec: float) -> dict[str, Any]:
    """Interpolate fixed prerecorded poses and lookup scripted visual perception flags."""
    drones: list[str] = data["drones"]
    wings: list[str] = data.get("wings", ["North Wing"] * 5 + ["South Wing"] * 5)
    frames: list[dict[str, Any]] = data["frames"]
    dt = float(data["dt"])
    duration = float(data["duration"])

    t_clamped = max(0.0, min(duration, float(t_sec)))
    idx_f = t_clamped / dt
    i0 = min(int(math.floor(idx_f)), len(frames) - 1)
    i1 = min(i0 + 1, len(frames) - 1)
    alpha = max(0.0, min(1.0, idx_f - i0))

    f0 = frames[i0]
    f1 = frames[i1]
    poses_out: list[tuple[str, list[float]]] = []
    drone_rows: list[dict[str, Any]] = []

    for idx, (name, p0, p1) in enumerate(zip(drones, f0["poses"], f1["poses"])):
        if i0 == i1 or alpha <= 1e-9:
            pose = list(p0)
        else:
            x = p0[0] + alpha * (p1[0] - p0[0])
            y = p0[1] + alpha * (p1[1] - p0[1])
            z = p0[2] + alpha * (p1[2] - p0[2])
            qx, qy, qz, qw = _slerp(p0[3:7], p1[3:7], alpha)
            pose = [x, y, z, qx, qy, qz, qw]
        poses_out.append((name, pose))
        alert = bool(f0["alerts"][idx])
        nearest = str(f0["nearest"][idx])
        arrived = bool(f0["arrived"][idx])
        drone_rows.append(
            {
                "id": name,
                "wing": wings[idx],
                "x": round(pose[0], 3),
                "y": round(pose[1], 3),
                "z": round(pose[2], 3),
                "alert": alert,
                "nearest": nearest,
                "arrived": arrived,
            }
        )

    act_name, short_stage = current_act(data, t_clamped)
    return {
        "t": t_clamped,
        "duration": duration,
        "progress_pct": (100.0 * t_clamped / duration) if duration > 0 else 100.0,
        "label": str(data.get("label", REQUIRED_LABEL)),
        "act_name": act_name,
        "short_stage": short_stage,
        "named_poses": poses_out,
        "drones": drone_rows,
        "glows": list(f0.get("glows", [])),
    }


def build_scene_pose_updates(
    data: dict[str, Any],
    state: dict[str, Any],
    *,
    newly_dropped_beads: list[tuple[int, list[list[float]]]] = (),
    reset_all_trails: bool = False,
) -> list[tuple[str, list[float]]]:
    """Assemble Gazebo Pose_V updates for drones, spinning LiDAR rings, obstacle glows, and trails."""
    t = float(state["t"])
    updates: list[tuple[str, list[float]]] = list(state["named_poses"])

    # Animated rotating LiDAR rings/rays: show green (clear) or orange/red (nearby obstacle).
    for idx, d in enumerate(state["drones"]):
        x, y, z = float(d["x"]), float(d["y"]), float(d["z"])
        airborne = z > 0.34
        spin_yaw = 2.8 * t + idx * 0.62
        q_spin = _euler_yaw_quat(spin_yaw)
        hidden_pose = [x, y, HIDDEN_Z, 0.0, 0.0, 0.0, 1.0]
        active_pose = [x, y, z, q_spin[0], q_spin[1], q_spin[2], q_spin[3]]

        if not airborne:
            updates.append((f"lidar_clear_{idx}", hidden_pose))
            updates.append((f"lidar_alert_{idx}", hidden_pose))
        elif d["alert"]:
            updates.append((f"lidar_clear_{idx}", hidden_pose))
            updates.append((f"lidar_alert_{idx}", active_pose))
        else:
            updates.append((f"lidar_clear_{idx}", active_pose))
            updates.append((f"lidar_alert_{idx}", hidden_pose))

    # Scripted obstacle proximity glow collars.
    active_glows = set(state.get("glows", []))
    for obs in data.get("glow_obstacles", []):
        oid = str(obs["id"])
        ox, oy = float(obs["x"]), float(obs["y"])
        gz = 0.0 if oid in active_glows else HIDDEN_Z
        updates.append((f"obs_glow_{oid}", [ox, oy, gz, 0.0, 0.0, 0.0, 1.0]))

    # Movement trails (incremental drop or reset on loop).
    num_drones = len(data["drones"])
    if reset_all_trails:
        for i in range(num_drones):
            for k in range(int(data.get("trail_beads_per_drone", 16))):
                updates.append((f"trail_{i}_{k}", [0.0, 0.0, HIDDEN_Z, 0.0, 0.0, 0.0, 1.0]))
    for bead_idx, positions in newly_dropped_beads:
        for i, pos in enumerate(positions):
            updates.append((f"trail_{i}_{bead_idx}", [pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 1.0]))

    return updates


def format_pose_vector_req(named_poses: list[tuple[str, list[float]]]) -> str:
    """Format `gz.msgs.Pose_V` protobuf text payload for `gz service`."""
    entries: list[str] = []
    for name, p in named_poses:
        x, y, z, qx, qy, qz, qw = p
        entries.append(
            f'pose {{ name: "{name}" '
            f"position {{ x: {x:.4f} y: {y:.4f} z: {z:.4f} }} "
            f"orientation {{ x: {qx:.5f} y: {qy:.5f} z: {qz:.5f} w: {qw:.5f} }} }}"
        )
    return " ".join(entries)


class _GzPoseClient:
    """Sends prerecorded `Pose_V` frames via gz.transport Python bindings or `gz service` CLI."""

    def __init__(self, world_name: str) -> None:
        self.service = f"/world/{world_name}/set_pose_vector"
        self._node = None
        self._pose_v_cls = None
        self._bool_cls = None
        self.backend = "gz-service-cli"

        for transport_mod, msgs_mod in (
            ("gz.transport13", "gz.msgs10.pose_v_pb2"),
            ("gz.transport13", "gz.msgs.pose_v_pb2"),
            ("gz.transport", "gz.msgs.pose_v_pb2"),
        ):
            try:
                t_mod = __import__(transport_mod, fromlist=["Node"])
                p_mod = __import__(msgs_mod, fromlist=["Pose_V"])
                b_mod_name = msgs_mod.replace("pose_v_pb2", "boolean_pb2")
                b_mod = __import__(b_mod_name, fromlist=["Boolean"])
                self._node = t_mod.Node()
                self._pose_v_cls = p_mod.Pose_V
                self._bool_cls = b_mod.Boolean
                self.backend = f"python:{transport_mod}"
                break
            except Exception:
                continue

    def send(self, named_poses: list[tuple[str, list[float]]], timeout_ms: int = 500) -> bool:
        if self._node is not None and self._pose_v_cls is not None and self._bool_cls is not None:
            req = self._pose_v_cls()
            for name, p in named_poses:
                item = req.pose.add()
                item.name = name
                item.position.x, item.position.y, item.position.z = p[0], p[1], p[2]
                item.orientation.x, item.orientation.y, item.orientation.z, item.orientation.w = (
                    p[3],
                    p[4],
                    p[5],
                    p[6],
                )
            ok, rep = self._node.request(
                self.service, req, self._pose_v_cls, self._bool_cls, int(timeout_ms)
            )
            return bool(ok and getattr(rep, "data", True))

        req_str = format_pose_vector_req(named_poses)
        try:
            proc = subprocess.run(
                [
                    "gz",
                    "service",
                    "-s",
                    self.service,
                    "--reqtype",
                    "gz.msgs.Pose_V",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    str(int(timeout_ms)),
                    "--req",
                    req_str,
                ],
                capture_output=True,
                text=True,
                timeout=max(1.8, timeout_ms / 1000.0 + 1.0),
                check=False,
            )
            return proc.returncode == 0 and "data: true" in proc.stdout.lower()
        except Exception:
            return False


class _DashboardServer:
    """Loopback-only HTTP server (127.0.0.1) serving the live visual dashboard."""

    def __init__(self, meta_payload: dict[str, Any], port: int = 8788) -> None:
        self._lock = threading.Lock()
        self._meta_bytes = json.dumps(meta_payload).encode("utf-8")
        self._state_bytes = b"{}"
        self._html_bytes = DASHBOARD_HTML.read_bytes() if DASHBOARD_HTML.is_file() else b"<h1>Dashboard missing</h1>"
        self.server: http.server.ThreadingHTTPServer | None = None
        self.url: str = ""

        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(outer._html_bytes)
                elif self.path == "/api/meta":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(outer._meta_bytes)
                elif self.path == "/api/state":
                    with outer._lock:
                        payload = outer._state_bytes
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

        for candidate_port in (int(port), 0):
            try:
                self.server = http.server.ThreadingHTTPServer(("127.0.0.1", candidate_port), _Handler)
                bound_port = self.server.server_address[1]
                self.url = f"http://127.0.0.1:{bound_port}"
                thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                thread.start()
                break
            except OSError:
                continue

    def update(self, public_state: dict[str, Any]) -> None:
        raw = json.dumps(public_state).encode("utf-8")
        with self._lock:
            self._state_bytes = raw

    def close(self) -> None:
        if self.server is not None:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass


def render_terminal_dashboard(
    state: dict[str, Any], *, dashboard_url: str, backend: str
) -> str:
    """Build a compact ANSI terminal status readout for the host terminal."""
    pct = float(state["progress_pct"])
    filled = int(round(pct / 100.0 * 32))
    bar = "#" * filled + "-" * (32 - filled)
    clear_n = sum(1 for d in state["drones"] if not d["alert"])
    alert_n = sum(1 for d in state["drones"] if d["alert"])
    arrived_n = sum(1 for d in state["drones"] if d["arrived"])

    lines = [
        "=" * 82,
        f"  {state['label']}",
        "  VISUAL ANIMATION ONLY — NO PX4 / ROS 2 / MAVLINK / DDS / LIVE SENSING / AI METRICS",
        "=" * 82,
        f"  Act      : {state['act_name']}",
        f"  Progress : [{bar}] {pct:5.1f}%  ({state['t']:4.1f}s / {state['duration']:.1f}s)",
        f"  Summary  : Clear={clear_n}/10 | Nearby-Obstacle Highlight={alert_n}/10 | Arrived={arrived_n}/10",
        f"  Web UI   : {dashboard_url or 'disabled'}   (Gazebo backend: {backend})",
        "-" * 82,
        f"  {'DRONE ID':<14} {'WING':<11} {'ALT(m)':>6}   {'ILLUSTRATED PERCEPTION':<28} {'DESTINATION'}",
    ]
    for d in state["drones"]:
        perc = (
            f"\033[91mNEAR: {d['nearest']}\033[0m"
            if d["alert"]
            else "\033[92mCLEAR SPACE\033[0m"
        )
        if d["arrived"]:
            dest = "\033[92mARRIVED\033[0m"
        elif d["z"] > 0.35:
            dest = state["short_stage"]
        else:
            dest = "STAGED"
        # Pad visual width ignoring ANSI codes
        raw_perc = f"NEAR: {d['nearest']}" if d["alert"] else "CLEAR SPACE"
        pad = " " * max(1, 28 - len(raw_perc))
        lines.append(
            f"  {d['id']:<14} {d['wing']:<11} {d['z']:>6.2f}   {perc}{pad} {dest}"
        )
    lines.append("-" * 82)
    lines.append("  Press Ctrl+C or close Gazebo to exit.")
    return "\n".join(lines)


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay the scripted 10-drone Gazebo visual research demo with live dashboard."
    )
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--fps", type=float, default=18.0, help="Target animation update Hz.")
    parser.add_argument("--loop", action="store_true", help="Continuously loop the 54 s animation.")
    parser.add_argument("--dashboard-port", type=int, default=8788, help="Loopback HTTP port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser dashboard.")
    parser.add_argument("--gz-pid", type=int, default=None, help="Gazebo PID to monitor.")
    parser.add_argument("--ready-timeout", type=float, default=35.0)
    args = parser.parse_args(argv)

    if args.speed <= 0.0 or args.fps <= 0.0:
        raise SystemExit("--speed and --fps must be > 0")

    data = json.loads(args.trajectories.read_text(encoding="utf-8"))
    world_name = str(data["world_name"])
    duration = float(data["duration"])
    trail_schedule: list[dict[str, Any]] = data.get("trail_schedule", [])

    running = True

    def _handle_stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    meta_payload = {
        "label": data.get("label", REQUIRED_LABEL),
        "acts": data.get("acts", []),
        "drones": data["drones"],
        "colors_rgb": data["colors_rgb"],
        "starts": data["starts"],
        "goals": data["goals"],
        "obstacles": data["obstacles"],
    }
    dashboard = _DashboardServer(meta_payload, port=args.dashboard_port)
    client = _GzPoseClient(world_name)

    initial_state = sample_frame_state(data, 0.0)
    dashboard.update({k: v for k, v in initial_state.items() if k != "named_poses"} | {"dropped_trails": []})

    print("=" * 82)
    print(f"  {data.get('label', REQUIRED_LABEL)}")
    print("  VISUAL ANIMATION ONLY — NO PX4 / ROS 2 / MAVLINK / DDS")
    print("=" * 82)
    print(f"Dashboard  : {dashboard.url}")
    print(f"Backend    : {client.backend} (partition={os.environ.get('GZ_PARTITION', 'default')})")
    print("Waiting for Gazebo scene to become ready...")

    if dashboard.url and not args.no_browser:
        try:
            webbrowser.open(dashboard.url, new=2)
        except Exception:
            pass

    init_updates = build_scene_pose_updates(data, initial_state, reset_all_trails=True)
    deadline = time.monotonic() + float(args.ready_timeout)
    ready = False
    while running and _pid_alive(args.gz_pid) and time.monotonic() < deadline:
        if client.send(init_updates, timeout_ms=600):
            ready = True
            break
        time.sleep(0.35)

    if not running or not _pid_alive(args.gz_pid):
        dashboard.close()
        return 0
    if not ready:
        print(f"ERROR: Gazebo service {client.service} did not respond in time.", file=sys.stderr)
        dashboard.close()
        return 1

    frame_period = 1.0 / float(args.fps)
    is_tty = sys.stdout.isatty()

    try:
        while running and _pid_alive(args.gz_pid):
            t0 = time.monotonic()
            next_trail_idx = 0
            dropped_trails_2d: list[dict[str, Any]] = []
            need_trail_reset = False
            last_term_draw = 0.0

            while running and _pid_alive(args.gz_pid):
                tick_start = time.monotonic()
                sim_t = (tick_start - t0) * float(args.speed)
                state = sample_frame_state(data, sim_t)

                newly_dropped: list[tuple[int, list[list[float]]]] = []
                while next_trail_idx < len(trail_schedule) and sim_t >= float(
                    trail_schedule[next_trail_idx]["t"]
                ):
                    entry = trail_schedule[next_trail_idx]
                    newly_dropped.append((int(entry["index"]), entry["positions"]))
                    for d_idx, pos in enumerate(entry["positions"]):
                        dropped_trails_2d.append({"drone_idx": d_idx, "x": pos[0], "y": pos[1]})
                    next_trail_idx += 1

                updates = build_scene_pose_updates(
                    data,
                    state,
                    newly_dropped_beads=newly_dropped,
                    reset_all_trails=need_trail_reset,
                )
                need_trail_reset = False
                client.send(updates, timeout_ms=450)

                public_state = {
                    k: v for k, v in state.items() if k != "named_poses"
                } | {"dropped_trails": dropped_trails_2d}
                dashboard.update(public_state)

                if tick_start - last_term_draw >= 0.4:
                    last_term_draw = tick_start
                    frame_text = render_terminal_dashboard(
                        state, dashboard_url=dashboard.url, backend=client.backend
                    )
                    if is_tty:
                        sys.stdout.write("\033[2J\033[H" + frame_text + "\n")
                        sys.stdout.flush()
                    else:
                        print(
                            f"[t={state['t']:4.1f}s/{duration:.1f}s] {state['act_name']} | "
                            f"alerts={sum(d['alert'] for d in state['drones'])}/10 | "
                            f"arrived={sum(d['arrived'] for d in state['drones'])}/10",
                            flush=True,
                        )

                if sim_t >= duration:
                    break

                sleep_for = frame_period - (time.monotonic() - tick_start)
                if sleep_for > 0.0:
                    time.sleep(sleep_for)

            if not args.loop:
                final_state = sample_frame_state(data, duration)
                dashboard.update(
                    {k: v for k, v in final_state.items() if k != "named_poses"}
                    | {"dropped_trails": dropped_trails_2d}
                )
                if is_tty:
                    sys.stdout.write(
                        "\033[2J\033[H"
                        + render_terminal_dashboard(
                            final_state, dashboard_url=dashboard.url, backend=client.backend
                        )
                        + "\n"
                    )
                    sys.stdout.flush()
                while running and _pid_alive(args.gz_pid):
                    # Keep LiDAR rings smoothly spinning in final formation hold.
                    hold_t = time.monotonic() - t0
                    hold_state = sample_frame_state(data, duration)
                    hold_state["t"] = duration + (hold_t - duration)
                    client.send(build_scene_pose_updates(data, hold_state), timeout_ms=450)
                    time.sleep(max(0.10, frame_period))
                break

            # Reset trails before next loop iteration.
            client.send(
                build_scene_pose_updates(data, sample_frame_state(data, 0.0), reset_all_trails=True),
                timeout_ms=600,
            )
    finally:
        dashboard.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
