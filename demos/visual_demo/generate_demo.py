#!/usr/bin/env python3
"""Generate the upgraded 54-second scripted 10-drone Gazebo visual research demo.

Scripted illustration of proposed LiDAR and AI-assisted navigation.
Generates:
  - demos/visual_demo/worlds/swarm_visual_demo.sdf
  - demos/visual_demo/trajectories.json

All trajectories, obstacle highlights, LiDAR scan rings/rays, and movement trails
are fixed prerecorded visualization effects computed strictly from manually authored
keyframes. Completely disconnected from PX4, ROS 2, MAVLink, DDS, live sensing,
autonomous path planning, flight control, and policy inference.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent
PACKAGED_SCENARIO = REPO_ROOT / "src" / "swarm_bringup" / "config" / "scenario.json"
PACKAGED_WORLD = REPO_ROOT / "src" / "swarm_bringup" / "worlds" / "swarm.sdf"
OUT_WORLD = DEMO_DIR / "worlds" / "swarm_visual_demo.sdf"
OUT_TRAJECTORIES = DEMO_DIR / "trajectories.json"

WORLD_NAME = "swarm_visual_demo"
GROUND_REST_Z = 0.24
HIDDEN_Z = -12.0  # Parked cleanly beneath opaque ground_plane when inactive.

REQUIRED_LABEL = "Scripted illustration of proposed LiDAR and AI-assisted navigation."

# Matches the exact 10-color palette of goal_beacon_0..9 in worlds/swarm.sdf.
DRONE_COLORS: list[tuple[float, float, float]] = [
    (0.10, 0.75, 1.00),
    (1.00, 0.45, 0.15),
    (0.70, 0.30, 1.00),
    (0.20, 0.90, 0.35),
    (1.00, 0.85, 0.15),
    (1.00, 0.25, 0.55),
    (0.25, 0.90, 0.80),
    (0.95, 0.55, 0.75),
    (0.55, 0.75, 1.00),
    (0.85, 0.95, 0.30),
]

# Total animation length: 54.0 s @ 20 Hz (1081 prerecorded frames).
TOTAL_DURATION = 54.0
SAMPLE_DT = 0.05
TRAIL_BEADS_PER_DRONE = 16
PROXIMITY_HIGHLIGHT_DIST = 2.45  # Illustrated proximity threshold (surface-to-surface, m).

# Prominent showcase obstacles of different shapes and heights added to the arena.
# Each entry defines visual SDF geometry plus conservative 2-D footprint for clearance checks.
SHOWCASE_OBSTACLES: list[dict[str, Any]] = [
    {
        "id": "showcase_low_bunker",
        "label": "Low Bunker Block",
        "shape": "box",
        "x": -0.75,
        "y": -2.05,
        "sx": 1.45,
        "sy": 1.45,
        "height": 2.60,
        "yaw": 0.7854,
        "radius_eq": 1.03,
        "color": (0.34, 0.38, 0.44),
        "accent": (0.92, 0.55, 0.12),
        "glow": True,
    },
    {
        "id": "showcase_monolith",
        "label": "Central Stepped Tower",
        "shape": "stepped_tower",
        "x": 0.55,
        "y": 0.65,
        "sx": 2.00,
        "sy": 2.00,
        "height": 7.20,
        "yaw": 0.0,
        "radius_eq": 1.42,
        "color": (0.24, 0.28, 0.34),
        "accent": (0.95, 0.42, 0.12),
        "glow": True,
    },
    {
        "id": "showcase_north_block",
        "label": "North Stepped Structure",
        "shape": "box",
        "x": -3.10,
        "y": 9.40,
        "sx": 2.40,
        "sy": 1.25,
        "height": 4.60,
        "yaw": -0.30,
        "radius_eq": 1.35,
        "color": (0.22, 0.35, 0.48),
        "accent": (0.95, 0.72, 0.16),
        "glow": True,
    },
    {
        "id": "showcase_south_wall",
        "label": "South Barrier Wall",
        "shape": "box",
        "x": 2.20,
        "y": -2.85,
        "sx": 2.70,
        "sy": 1.05,
        "height": 3.80,
        "yaw": 0.52,
        "radius_eq": 1.45,
        "color": (0.42, 0.30, 0.26),
        "accent": (0.95, 0.48, 0.14),
        "glow": True,
    },
    {
        "id": "showcase_radar_tower",
        "label": "Tall Cylindrical Tower",
        "shape": "radar_tower",
        "x": 2.40,
        "y": 3.10,
        "radius": 0.76,
        "height": 6.40,
        "yaw": 0.0,
        "radius_eq": 0.76,
        "color": (0.28, 0.32, 0.38),
        "accent": (0.95, 0.35, 0.15),
        "glow": True,
    },
    {
        "id": "showcase_gantry_arch",
        "label": "East Portal Frame",
        "shape": "archway",
        "x": 12.60,
        "y": -2.80,
        "span": 3.60,
        "pillar_w": 0.55,
        "height": 5.80,
        "yaw": 0.7854,
        "radius_eq": 2.10,
        "color": (0.26, 0.34, 0.40),
        "accent": (0.92, 0.62, 0.14),
        "glow": False,
    },
]

# Act timeline for dashboard & choreography clarity (54.0 s total).
ACTS: list[dict[str, Any]] = [
    {"id": "ground_staging", "name": "Ground Staging (START_ZONE)", "start": 0.0, "end": 3.0},
    {"id": "takeoff", "name": "Staggered Vertical Takeoff", "start": 3.0, "end": 9.0},
    {"id": "separation", "name": "Formation Separation (Dual Wings)", "start": 9.0, "end": 19.0},
    {"id": "avoidance_weave", "name": "Scripted Obstacle-Avoidance Weave", "start": 19.0, "end": 34.0},
    {"id": "regrouping", "name": "Formation Regrouping & Approach", "start": 34.0, "end": 45.0},
    {"id": "arrival_hold", "name": "Destination Arrival & Final Hold", "start": 45.0, "end": TOTAL_DURATION},
]


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def _quintic_ease(u: float) -> float:
    u = _clamp(u, 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _eval_catmull_rom_track(
    keyframes: list[tuple[float, float, float, float]], t: float
) -> tuple[float, float, float]:
    """Evaluate C^1 smooth Catmull-Rom spline over manually authored `(time, x, y, z)` keyframes."""
    if t <= keyframes[0][0]:
        return keyframes[0][1], keyframes[0][2], keyframes[0][3]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1], keyframes[-1][2], keyframes[-1][3]

    # Locate active segment [i, i+1]
    seg = 0
    for idx in range(len(keyframes) - 1):
        if keyframes[idx][0] <= t <= keyframes[idx + 1][0]:
            seg = idx
            break

    t0, x0, y0, z0 = keyframes[seg]
    t1, x1, y1, z1 = keyframes[seg + 1]
    dt = max(1e-6, t1 - t0)
    u = (t - t0) / dt

    # Compute zero-clamped / finite-difference tangents for overshoot-free smooth motion.
    def _tangent(idx: int, axis: int) -> float:
        if idx <= 0 or idx >= len(keyframes) - 1:
            return 0.0
        tp, vp = keyframes[idx - 1][0], keyframes[idx - 1][axis]
        tc, vc = keyframes[idx][0], keyframes[idx][axis]
        tn, vn = keyframes[idx + 1][0], keyframes[idx + 1][axis]
        # Zero tangent if segment before or after is stationary on this axis (clean holds).
        if abs(vc - vp) < 1e-5 or abs(vn - vc) < 1e-5:
            return 0.0
        # Centripetal-weighted finite difference clamped to avoid spline overshoot.
        m_prev = (vc - vp) / max(1e-6, tc - tp)
        m_next = (vn - vc) / max(1e-6, tn - tc)
        if m_prev * m_next <= 0.0:
            return 0.0
        return 0.5 * (m_prev + m_next)

    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2

    vals: list[float] = []
    for axis in (1, 2, 3):
        p0 = keyframes[seg][axis]
        p1 = keyframes[seg + 1][axis]
        m0 = _tangent(seg, axis) * dt
        m1 = _tangent(seg + 1, axis) * dt
        vals.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    return vals[0], vals[1], vals[2]


def _build_manual_drone_keyframes(
    starts: list[list[float]], goals: list[list[float]]
) -> list[list[tuple[float, float, float, float]]]:
    """Manually authored 6-act visual keyframes for all 10 drones.

    Uses the natural non-crossing half-ring geometry of the 10-drone formation so
    North Wing (drones 0..4) and South Wing (drones 5..9) separate dramatically
    around the central obstacle cluster, weave smoothly around the multi-shape
    structures with zero drone-drone crossings, and regroup into GOAL_ZONE.
    """
    north_start_cx = sum(starts[i][0] for i in range(5)) / 5.0
    north_start_cy = sum(starts[i][1] for i in range(5)) / 5.0
    south_start_cx = sum(starts[i][0] for i in range(5, 10)) / 5.0
    south_start_cy = sum(starts[i][1] for i in range(5, 10)) / 5.0

    # Manually authored wing formation keyframes: (t, wing_cx, wing_cy, wing_cz, spread_scale)
    north_wing_centers = [
        (14.5, -4.80, 0.50, 2.85, 0.86),
        (20.5, -2.50, 5.10, 3.35, 0.84),
        (27.0, 0.10, 7.05, 3.55, 0.84),
        (33.5, 3.10, 7.45, 3.15, 0.88),
        (39.5, 5.05, 7.75, 2.75, 0.96),
    ]
    south_wing_centers = [
        (14.5, 0.90, -7.80, 2.55, 0.86),
        (20.5, 6.10, -5.80, 2.95, 0.84),
        (27.0, 7.60, -1.10, 3.15, 0.84),
        (33.5, 7.20, 2.55, 2.90, 0.88),
        (39.5, 7.35, 4.40, 2.65, 0.96),
    ]

    # Subtle vertical tiering within each 5-drone wing for 3-D cinematic depth.
    wing_z_wave = (-0.18, 0.06, 0.22, 0.08, -0.14)

    all_tracks: list[list[tuple[float, float, float, float]]] = []
    for i in range(10):
        sx, sy, _ = starts[i]
        gx, gy, gz = goals[i]

        t_lift_0 = 3.0 + i * 0.22
        t_lift_1 = t_lift_0 + 3.6
        hover_z = 2.50 + (0.20 if i < 5 else 0.0)

        kf: list[tuple[float, float, float, float]] = [
            (0.0, sx, sy, GROUND_REST_Z),
            (t_lift_0, sx, sy, GROUND_REST_Z),
            (t_lift_1, sx, sy, hover_z),
            (9.0, sx, sy, hover_z),
        ]

        if i < 5:
            rel_x = sx - north_start_cx
            rel_y = sy - north_start_cy
            dz = wing_z_wave[i]
            for wt, wx, wy, wz, sc in north_wing_centers:
                kf.append((wt, wx + sc * rel_x, wy + sc * rel_y, wz + dz))
        else:
            rel_x = sx - south_start_cx
            rel_y = sy - south_start_cy
            dz = wing_z_wave[i - 5]
            for wt, wx, wy, wz, sc in south_wing_centers:
                kf.append((wt, wx + sc * rel_x, wy + sc * rel_y, wz + dz))

        # Final synchronized arrival at exact goal_beacon_0..9 coordinates.
        kf.append((45.0, gx, gy, gz))
        kf.append((TOTAL_DURATION, gx, gy, gz))
        all_tracks.append(kf)

    return all_tracks


def _all_arena_obstacles(scenario: dict) -> list[dict[str, Any]]:
    """Combine existing seed-7 cylinders + new showcase obstacles for clearance checks."""
    obs_list: list[dict[str, Any]] = []
    for idx, (ox, oy, orad, oheight) in enumerate(scenario["obstacles"]):
        obs_list.append(
            {
                "id": f"obstacle_{idx}",
                "label": f"Arena Pillar {idx}",
                "shape": "cylinder",
                "x": float(ox),
                "y": float(oy),
                "radius_eq": float(orad),
                "height": float(oheight),
                "glow": idx == 1,  # Highlight obstacle_1 when South Wing weaves around it.
            }
        )
    for item in SHOWCASE_OBSTACLES:
        obs_list.append(dict(item))
    return obs_list


def _distance_to_obstacle_surface(x: float, y: float, obs: dict[str, Any]) -> float:
    """2-D surface-to-surface clearance (accounting for 0.45 m visual drone radius)."""
    return math.hypot(x - float(obs["x"]), y - float(obs["y"])) - float(obs["radius_eq"]) - 0.45


def build_prerecorded_trajectories(scenario: dict) -> dict[str, Any]:
    """Sample all manually authored visual keyframes into fixed 20 Hz prerecorded frames."""
    starts: list[list[float]] = scenario["starts"]
    goals: list[list[float]] = scenario["goals"]
    num_drones = len(starts)
    if num_drones != 10 or len(goals) != 10:
        raise ValueError(f"Expected 10 starts and 10 goals, got {len(starts)} and {len(goals)}")

    start_cx = sum(p[0] for p in starts) / num_drones
    start_cy = sum(p[1] for p in starts) / num_drones
    goal_cx = sum(p[0] for p in goals) / num_drones
    goal_cy = sum(p[1] for p in goals) / num_drones
    corridor_yaw = math.atan2(goal_cy - start_cy, goal_cx - start_cx)

    tracks = _build_manual_drone_keyframes(starts, goals)
    all_obs = _all_arena_obstacles(scenario)
    glow_obs_ids = [o["id"] for o in all_obs if o.get("glow")]

    num_frames = int(round(TOTAL_DURATION / SAMPLE_DT)) + 1
    times = [round(idx * SAMPLE_DT, 4) for idx in range(num_frames)]

    # First pass: sample smooth positions for all frames.
    raw_pos: list[list[tuple[float, float, float]]] = []
    for t in times:
        raw_pos.append([_eval_catmull_rom_track(tracks[i], t) for i in range(num_drones)])

    # Trail drop times: evenly spaced during active flight (t = 6.0 s .. 44.0 s).
    trail_drop_times = [
        round(6.0 + k * (38.0 / max(1, TRAIL_BEADS_PER_DRONE - 1)), 2)
        for k in range(TRAIL_BEADS_PER_DRONE)
    ]

    frames: list[dict[str, Any]] = []
    min_pair_sep = float("inf")
    min_obs_clear = float("inf")
    alert_frame_count = 0

    prev_yaws = [corridor_yaw for _ in range(num_drones)]

    for f_idx, t in enumerate(times):
        i_prev = max(0, f_idx - 1)
        i_next = min(num_frames - 1, f_idx + 1)
        dt_span = max(SAMPLE_DT, (i_next - i_prev) * SAMPLE_DT)

        poses: list[list[float]] = []
        alerts: list[int] = []
        nearest_obs_names: list[str] = []
        arrived_flags: list[int] = []
        active_glows: set[str] = set()

        for i in range(num_drones):
            x, y, z = raw_pos[f_idx][i]
            xp, yp, zp = raw_pos[i_prev][i]
            xn, yn, zn = raw_pos[i_next][i]

            vx = (xn - xp) / dt_span
            vy = (yn - yp) / dt_span
            ax = (xn - 2.0 * x + xp) / (SAMPLE_DT * SAMPLE_DT) if 0 < f_idx < num_frames - 1 else 0.0
            ay = (yn - 2.0 * y + yp) / (SAMPLE_DT * SAMPLE_DT) if 0 < f_idx < num_frames - 1 else 0.0

            speed_xy = math.hypot(vx, vy)
            if t < 8.8 or t > 44.2:
                target_yaw = corridor_yaw
            elif speed_xy > 0.15:
                target_yaw = math.atan2(vy, vx)
            else:
                target_yaw = prev_yaws[i]

            # Smooth yaw tracking without abrupt flips.
            yaw_diff = _wrap_angle(target_yaw - prev_yaws[i])
            yaw = _wrap_angle(prev_yaws[i] + 0.18 * yaw_diff)
            prev_yaws[i] = yaw

            if 9.0 <= t <= 44.8:
                cy_a, sy_a = math.cos(yaw), math.sin(yaw)
                a_fwd = cy_a * ax + sy_a * ay
                a_left = -sy_a * ax + cy_a * ay
                pitch = _clamp(0.11 * a_fwd + 0.03 * speed_xy, -0.15, 0.15)
                roll = _clamp(-0.11 * a_left, -0.15, 0.15)
            else:
                roll, pitch = 0.0, 0.0

            qx, qy, qz, qw = _euler_to_quat(roll, pitch, yaw)
            poses.append(
                [
                    round(x, 4),
                    round(y, 4),
                    round(z, 4),
                    round(qx, 5),
                    round(qy, 5),
                    round(qz, 5),
                    round(qw, 5),
                ]
            )

            # Illustrated proximity evaluation (precomputed at generation time — NOT live sensing).
            best_clear = float("inf")
            best_obs_id = ""
            best_obs_label = "Clear Space"
            for obs in all_obs:
                c = _distance_to_obstacle_surface(x, y, obs)
                if c < min_obs_clear:
                    min_obs_clear = c
                if c < best_clear:
                    best_clear = c
                    best_obs_id = str(obs["id"])
                    best_obs_label = str(obs["label"])

            is_airborne = z > GROUND_REST_Z + 0.35
            is_alert = bool(is_airborne and t < 44.5 and best_clear < PROXIMITY_HIGHLIGHT_DIST)
            alerts.append(1 if is_alert else 0)
            nearest_obs_names.append(best_obs_label if is_alert else "Clear Space")
            if is_alert:
                alert_frame_count += 1
                if best_obs_id in glow_obs_ids:
                    active_glows.add(best_obs_id)

            gx, gy, gz = goals[i]
            goal_dist = math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)
            arrived_flags.append(1 if (t >= 44.8 and goal_dist < 0.05) else 0)

        for i in range(num_drones):
            xi, yi, zi = raw_pos[f_idx][i]
            for j in range(i + 1, num_drones):
                xj, yj, zj = raw_pos[f_idx][j]
                d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
                if d < min_pair_sep:
                    min_pair_sep = d

        frames.append(
            {
                "t": t,
                "poses": poses,
                "alerts": alerts,
                "nearest": nearest_obs_names,
                "arrived": arrived_flags,
                "glows": sorted(active_glows),
            }
        )

    # Precompute exact fixed trail bead positions for each drop index.
    trail_schedule: list[dict[str, Any]] = []
    for bead_idx, drop_t in enumerate(trail_drop_times):
        f_idx = min(num_frames - 1, int(round(drop_t / SAMPLE_DT)))
        bead_positions = [
            [round(raw_pos[f_idx][i][0], 4), round(raw_pos[f_idx][i][1], 4), round(raw_pos[f_idx][i][2], 4)]
            for i in range(num_drones)
        ]
        trail_schedule.append({"index": bead_idx, "t": drop_t, "positions": bead_positions})

    return {
        "label": REQUIRED_LABEL,
        "disclaimer": (
            f"{REQUIRED_LABEL} "
            "Fixed prerecorded animation keyframes and scripted visual highlights only — "
            "NOT validated autonomous flight, live LiDAR sensing, path planning, or policy inference. "
            "Completely disconnected from PX4, ROS 2 flight-control topics, MAVLink, and DDS."
        ),
        "mode": "SCRIPTED_VISUALIZATION_ONLY",
        "world_name": WORLD_NAME,
        "source_scenario_seed": scenario.get("seed", 7),
        "dt": SAMPLE_DT,
        "duration": TOTAL_DURATION,
        "acts": ACTS,
        "drones": [f"demo_drone_{i}" for i in range(num_drones)],
        "wings": ["North Wing" if i < 5 else "South Wing" for i in range(num_drones)],
        "colors_rgb": [list(c) for c in DRONE_COLORS],
        "starts": starts,
        "goals": goals,
        "obstacles": all_obs,
        "glow_obstacles": [o for o in all_obs if o.get("glow")],
        "trail_beads_per_drone": TRAIL_BEADS_PER_DRONE,
        "trail_schedule": trail_schedule,
        "geometry_checks": {
            "min_pair_separation_m": round(min_pair_sep, 4),
            "min_obstacle_clearance_m": round(min_obs_clear, 4),
            "scripted_alert_drone_frames": alert_frame_count,
        },
        "frames": frames,
    }


def _visual_drone_sdf(
    index: int,
    start_xy: tuple[float, float],
    yaw: float,
    color_rgb: tuple[float, float, float],
) -> str:
    """Visual-only x500 SDF model reusing PX4's exact x500_base meshes (zero sensors/motors)."""
    r, g, b = color_rgb
    sx, sy = start_xy
    prop_scale = "0.8461538461538461 0.8461538461538461 0.8461538461538461"

    rotors = [
        ("rotor_0", (0.174, -0.174, 0.028), (0.152, -0.320384615, 0.044), "1345_prop_ccw.stl"),
        ("rotor_1", (-0.174, 0.174, 0.028), (-0.196, 0.027615385, 0.044), "1345_prop_ccw.stl"),
        ("rotor_2", (0.174, 0.174, 0.028), (0.152, 0.027615385, 0.044), "1345_prop_cw.stl"),
        ("rotor_3", (-0.174, -0.174, 0.028), (-0.196, -0.320384615, 0.044), "1345_prop_cw.stl"),
    ]
    rotor_blocks = [
        f"""      <visual name="{rname}_bell">
        <pose>{bx:.6f} {by:.6f} {bz:.6f} 0 0 0</pose><cast_shadows>false</cast_shadows>
        <geometry><mesh><scale>1 1 1</scale><uri>model://x500_base/meshes/5010Bell.dae</uri></mesh></geometry>
      </visual>
      <visual name="{rname}_prop">
        <pose>{px:.6f} {py:.6f} {pz:.6f} 0 0 0</pose><cast_shadows>false</cast_shadows>
        <geometry><mesh><scale>{prop_scale}</scale><uri>model://x500_base/meshes/{pmesh}</uri></mesh></geometry>
        <material><ambient>0.16 0.16 0.18 1</ambient><diffuse>0.20 0.20 0.22 1</diffuse><specular>0.35 0.35 0.35 1</specular></material>
      </visual>"""
        for rname, (bx, by, bz), (px, py, pz), pmesh in rotors
    ]
    motor_bases = [
        f"""      <visual name="5010_motor_base_{k}">
        <pose>{mpose}</pose><cast_shadows>false</cast_shadows>
        <geometry><mesh><scale>1 1 1</scale><uri>model://x500_base/meshes/5010Base.dae</uri></mesh></geometry>
      </visual>"""
        for k, mpose in enumerate(
            (
                "0.174 0.174 0.032 0 0 -0.45",
                "-0.174 0.174 0.032 0 0 -0.45",
                "0.174 -0.174 0.032 0 0 -0.45",
                "-0.174 -0.174 0.032 0 0 -0.45",
            )
        )
    ]

    return f"""  <model name="demo_drone_{index}">
    <pose>{sx:.5f} {sy:.5f} {GROUND_REST_Z:.4f} 0 0 {yaw:.5f}</pose>
    <static>false</static><self_collide>false</self_collide>
    <link name="base_link">
      <gravity>false</gravity><kinematic>true</kinematic>
      <inertial><mass>1.0</mass><inertia><ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.01</iyy><iyz>0</iyz><izz>0.02</izz></inertia></inertial>
      <visual name="base_link_visual">
        <pose>0 0 0.025 0 0 3.141592654</pose><cast_shadows>false</cast_shadows>
        <geometry><mesh><scale>1 1 1</scale><uri>model://x500_base/meshes/NXP-HGD-CF.dae</uri></mesh></geometry>
      </visual>
{chr(10).join(motor_bases)}
{chr(10).join(rotor_blocks)}
      <visual name="NXP_FMUK66_FRONT">
        <pose>0.047 0.001 0.043 1 0 1.57</pose><cast_shadows>false</cast_shadows>
        <geometry><plane><normal>0 0 1</normal><size>0.013 0.007</size></plane></geometry>
        <material><diffuse>1 1 1 1</diffuse><specular>1 1 1 1</specular><pbr><metal><albedo_map>model://x500_base/materials/textures/nxp.png</albedo_map></metal></pbr></material>
      </visual>
      <visual name="NXP_FMUK66_TOP">
        <pose>-0.023 0 0.0515 0 0 -1.57</pose><cast_shadows>false</cast_shadows>
        <geometry><plane><normal>0 0 1</normal><size>0.013 0.007</size></plane></geometry>
        <material><diffuse>1 1 1 1</diffuse><specular>1 1 1 1</specular><pbr><metal><albedo_map>model://x500_base/materials/textures/nxp.png</albedo_map></metal></pbr></material>
      </visual>
      <visual name="RDDRONE_FMUK66_TOP">
        <pose>-0.03 0 0.0515 0 0 -1.57</pose><cast_shadows>false</cast_shadows>
        <geometry><plane><normal>0 0 1</normal><size>0.032 0.0034</size></plane></geometry>
        <material><diffuse>1 1 1 1</diffuse><specular>1 1 1 1</specular><pbr><metal><albedo_map>model://x500_base/materials/textures/rd.png</albedo_map></metal></pbr></material>
      </visual>
      <visual name="slot_id_beacon">
        <pose>0 0 0.075 0 0 0</pose><cast_shadows>false</cast_shadows>
        <geometry><cylinder><radius>0.068</radius><length>0.018</length></cylinder></geometry>
        <material><ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient><diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse><emissive>{r*0.65:.2f} {g*0.65:.2f} {b*0.65:.2f} 1</emissive></material>
      </visual>
    </link>
  </model>"""


def _lidar_ring_sdf(name: str, *, alert: bool) -> str:
    """Build a visual-only animated LiDAR scanning ring + 8 radial rays (green or orange-red)."""
    if alert:
        r, g, b = 1.00, 0.28, 0.06
        ray_len = 1.25
        ring_r = 0.72
        alpha_disc = 0.24
    else:
        r, g, b = 0.08, 0.92, 0.45
        ray_len = 0.95
        ring_r = 0.62
        alpha_disc = 0.16

    # Build octagonal ring perimeter + 8 radiating LiDAR beams + 1 bold sweeping scanner beam.
    visuals: list[str] = []
    segs = 12
    seg_len = 2.0 * ring_r * math.sin(math.pi / segs) + 0.02
    for s in range(segs):
        ang = 2.0 * math.pi * s / segs
        px = ring_r * math.cos(ang)
        py = ring_r * math.sin(ang)
        visuals.append(
            f'      <visual name="ring_{s}"><pose>{px:.4f} {py:.4f} 0.01 0 0 {ang + math.pi/2:.4f}</pose>'
            f"<cast_shadows>false</cast_shadows>"
            f"<geometry><box><size>{seg_len:.4f} 0.024 0.014</size></box></geometry>"
            f"<material><ambient>{r:.2f} {g:.2f} {b:.2f} 0.85</ambient><diffuse>{r:.2f} {g:.2f} {b:.2f} 0.85</diffuse>"
            f"<emissive>{r*0.85:.2f} {g*0.85:.2f} {b*0.85:.2f} 0.85</emissive></material></visual>"
        )

    # 8 radial LiDAR rays (alternating full/medium beams so rotation is vividly visible).
    for k in range(8):
        ang = 2.0 * math.pi * k / 8.0
        cur_len = ray_len * (1.28 if k == 0 else (0.95 if k % 2 == 0 else 0.70))
        thick = 0.032 if k == 0 else 0.016
        dist = 0.22 + 0.5 * cur_len
        px = dist * math.cos(ang)
        py = dist * math.sin(ang)
        visuals.append(
            f'      <visual name="ray_{k}"><pose>{px:.4f} {py:.4f} 0.012 0 0 {ang:.4f}</pose>'
            f"<cast_shadows>false</cast_shadows>"
            f"<geometry><box><size>{cur_len:.4f} {thick:.4f} 0.010</size></box></geometry>"
            f"<material><ambient>{r:.2f} {g:.2f} {b:.2f} 0.80</ambient><diffuse>{r:.2f} {g:.2f} {b:.2f} 0.80</diffuse>"
            f"<emissive>{r*0.90:.2f} {g*0.90:.2f} {b*0.90:.2f} 0.80</emissive></material></visual>"
        )

    # Translucent inner scan disc.
    visuals.append(
        f'      <visual name="scan_disc"><pose>0 0 0.005 0 0 0</pose><cast_shadows>false</cast_shadows>'
        f"<geometry><cylinder><radius>{ring_r:.3f}</radius><length>0.006</length></cylinder></geometry>"
        f"<material><ambient>{r:.2f} {g:.2f} {b:.2f} {alpha_disc:.2f}</ambient>"
        f"<diffuse>{r:.2f} {g:.2f} {b:.2f} {alpha_disc:.2f}</diffuse>"
        f"<emissive>{r*0.55:.2f} {g*0.55:.2f} {b*0.55:.2f} {alpha_disc:.2f}</emissive></material></visual>"
    )

    return f"""  <model name="{name}">
    <pose>0 0 {HIDDEN_Z:.1f} 0 0 0</pose>
    <static>false</static><self_collide>false</self_collide>
    <link name="scan_link">
      <gravity>false</gravity><kinematic>true</kinematic>
      <inertial><mass>0.1</mass><inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia></inertial>
{chr(10).join(visuals)}
    </link>
  </model>"""


def _showcase_obstacle_sdf(obs: dict[str, Any]) -> str:
    """Build prominent multi-shape, multi-height architectural showcase obstacle."""
    oid = obs["id"]
    x, y, h = float(obs["x"]), float(obs["y"]), float(obs["height"])
    yaw = float(obs.get("yaw", 0.0))
    cr, cg, cb = obs["color"]
    ar, ag, ab = obs["accent"]
    shape = obs["shape"]

    visuals: list[str] = []
    if shape == "stepped_tower":
        sx, sy = float(obs["sx"]), float(obs["sy"])
        # Base plinth + main tall core + upper stepped crown + hazard bands + red top beacon.
        visuals.append(
            f'      <visual name="base"><pose>0 0 {h*0.18:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{sx:.3f} {sy:.3f} {h*0.36:.3f}</size></box></geometry>"
            f"<material><ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1</ambient><diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1</diffuse></material></visual>"
        )
        visuals.append(
            f'      <visual name="mid"><pose>0 0 {h*0.56:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{sx*0.80:.3f} {sy*0.80:.3f} {h*0.42:.3f}</size></box></geometry>"
            f"<material><ambient>{cr+0.05:.2f} {cg+0.05:.2f} {cb+0.05:.2f} 1</ambient><diffuse>{cr+0.05:.2f} {cg+0.05:.2f} {cb+0.05:.2f} 1</diffuse></material></visual>"
        )
        visuals.append(
            f'      <visual name="top"><pose>0 0 {h*0.88:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{sx*0.58:.3f} {sy*0.58:.3f} {h*0.24:.3f}</size></box></geometry>"
            f"<material><ambient>{cr+0.10:.2f} {cg+0.10:.2f} {cb+0.10:.2f} 1</ambient><diffuse>{cr+0.10:.2f} {cg+0.10:.2f} {cb+0.10:.2f} 1</diffuse></material></visual>"
        )
        for idx_b, (bz, bscale) in enumerate(((h * 0.36, 1.02), (h * 0.77, 0.82))):
            visuals.append(
                f'      <visual name="band_{idx_b}"><pose>0 0 {bz:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
                f"<geometry><box><size>{sx*bscale:.3f} {sy*bscale:.3f} 0.22</size></box></geometry>"
                f"<material><ambient>{ar:.2f} {ag:.2f} {ab:.2f} 1</ambient><diffuse>{ar:.2f} {ag:.2f} {ab:.2f} 1</diffuse>"
                f"<emissive>{ar*0.4:.2f} {ag*0.4:.2f} {ab*0.4:.2f} 1</emissive></material></visual>"
            )
        visuals.append(
            f'      <visual name="spire"><pose>0 0 {h + 0.35:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><sphere><radius>0.18</radius></sphere></geometry>"
            f"<material><ambient>1.0 0.22 0.10 1</ambient><diffuse>1.0 0.22 0.10 1</diffuse><emissive>0.9 0.18 0.08 1</emissive></material></visual>"
        )
    elif shape == "radar_tower":
        rad = float(obs["radius"])
        visuals.append(
            f'      <visual name="shaft"><pose>0 0 {h*0.5:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><cylinder><radius>{rad:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>"
            f"<material><ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1</ambient><diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1</diffuse></material></visual>"
        )
        visuals.append(
            f'      <visual name="collar"><pose>0 0 {h*0.72:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><cylinder><radius>{rad*1.16:.3f}</radius><length>0.32</length></cylinder></geometry>"
            f"<material><ambient>{ar:.2f} {ag:.2f} {ab:.2f} 1</ambient><diffuse>{ar:.2f} {ag:.2f} {ab:.2f} 1</diffuse>"
            f"<emissive>{ar*0.4:.2f} {ag*0.4:.2f} {ab*0.4:.2f} 1</emissive></material></visual>"
        )
        visuals.append(
            f'      <visual name="dome"><pose>0 0 {h + 0.35:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><sphere><radius>{rad*0.85:.3f}</radius></sphere></geometry>"
            f"<material><ambient>0.82 0.86 0.90 1</ambient><diffuse>0.82 0.86 0.90 1</diffuse></material></visual>"
        )
    elif shape == "archway":
        span = float(obs["span"])
        pw = float(obs["pillar_w"])
        visuals.append(
            f'      <visual name="left_col"><pose>0 {-span*0.5:.3f} {h*0.5:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{pw:.3f} {pw:.3f} {h:.3f}</size></box></geometry>"
            f"<material><ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1</ambient><diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1</diffuse></material></visual>"
        )
        visuals.append(
            f'      <visual name="right_col"><pose>0 {span*0.5:.3f} {h*0.5:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{pw:.3f} {pw:.3f} {h:.3f}</size></box></geometry>"
            f"<material><ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1</ambient><diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1</diffuse></material></visual>"
        )
        visuals.append(
            f'      <visual name="beam"><pose>0 0 {h - 0.30:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{pw*1.15:.3f} {span + pw:.3f} 0.60</size></box></geometry>"
            f"<material><ambient>{ar:.2f} {ag:.2f} {ab:.2f} 1</ambient><diffuse>{ar:.2f} {ag:.2f} {ab:.2f} 1</diffuse></material></visual>"
        )
    else:
        sx, sy = float(obs["sx"]), float(obs["sy"])
        visuals.append(
            f'      <visual name="body"><pose>0 0 {h*0.5:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{sx:.3f} {sy:.3f} {h:.3f}</size></box></geometry>"
            f"<material><ambient>{cr:.2f} {cg:.2f} {cb:.2f} 1</ambient><diffuse>{cr:.2f} {cg:.2f} {cb:.2f} 1</diffuse></material></visual>"
        )
        visuals.append(
            f'      <visual name="trim"><pose>0 0 {h - 0.25:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>'
            f"<geometry><box><size>{sx+0.08:.3f} {sy+0.08:.3f} 0.26</size></box></geometry>"
            f"<material><ambient>{ar:.2f} {ag:.2f} {ab:.2f} 1</ambient><diffuse>{ar:.2f} {ag:.2f} {ab:.2f} 1</diffuse>"
            f"<emissive>{ar*0.35:.2f} {ag*0.35:.2f} {ab*0.35:.2f} 1</emissive></material></visual>"
        )

    return f"""  <model name="{oid}">
    <static>true</static>
    <pose>{x:.4f} {y:.4f} 0 0 0 {yaw:.4f}</pose>
    <link name="structure">
{chr(10).join(visuals)}
    </link>
  </model>"""


def _obstacle_glow_sdf(obs: dict[str, Any]) -> str:
    """Orange-red translucent proximity highlight collar for an obstacle (hidden until active)."""
    oid = obs["id"]
    rad = float(obs["radius_eq"]) + 0.32
    h = min(float(obs["height"]), 4.2)
    return f"""  <model name="obs_glow_{oid}">
    <pose>{float(obs['x']):.4f} {float(obs['y']):.4f} {HIDDEN_Z:.1f} 0 0 0</pose>
    <static>false</static><self_collide>false</self_collide>
    <link name="glow_link">
      <gravity>false</gravity><kinematic>true</kinematic>
      <inertial><mass>0.1</mass><inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia></inertial>
      <visual name="glow_shell">
        <pose>0 0 {h*0.5:.3f} 0 0 0</pose><cast_shadows>false</cast_shadows>
        <geometry><cylinder><radius>{rad:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>
        <material>
          <ambient>1.0 0.30 0.05 0.26</ambient>
          <diffuse>1.0 0.30 0.05 0.26</diffuse>
          <emissive>1.0 0.24 0.04 0.26</emissive>
        </material>
      </visual>
      <visual name="glow_base_ring">
        <pose>0 0 0.06 0 0 0</pose><cast_shadows>false</cast_shadows>
        <geometry><cylinder><radius>{rad + 0.18:.3f}</radius><length>0.05</length></cylinder></geometry>
        <material>
          <ambient>1.0 0.22 0.04 0.65</ambient>
          <diffuse>1.0 0.22 0.04 0.65</diffuse>
          <emissive>1.0 0.20 0.04 0.65</emissive>
        </material>
      </visual>
    </link>
  </model>"""


def _trail_bead_sdf(drone_idx: int, bead_idx: int, color_rgb: tuple[float, float, float]) -> str:
    """Visual-only glowing movement trail marker parked beneath the floor until dropped."""
    r, g, b = color_rgb
    return (
        f'  <model name="trail_{drone_idx}_{bead_idx}"><pose>0 0 {HIDDEN_Z:.1f} 0 0 0</pose>'
        f"<static>false</static><self_collide>false</self_collide>"
        f'<link name="bead"><gravity>false</gravity><kinematic>true</kinematic>'
        f"<inertial><mass>0.05</mass><inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia></inertial>"
        f'<visual name="v"><cast_shadows>false</cast_shadows>'
        f"<geometry><sphere><radius>0.095</radius></sphere></geometry>"
        f"<material><ambient>{r:.2f} {g:.2f} {b:.2f} 0.85</ambient>"
        f"<diffuse>{r:.2f} {g:.2f} {b:.2f} 0.85</diffuse>"
        f"<emissive>{r*0.75:.2f} {g*0.75:.2f} {b*0.75:.2f} 0.85</emissive></material>"
        f"</visual></link></model>"
    )


def build_visual_world_sdf(
    packaged_sdf_path: Path, scenario: dict, trajectories: dict[str, Any]
) -> str:
    """Build the upgraded visual-only Gazebo Harmonic SDF world with cinematic camera pose."""
    root = ET.parse(packaged_sdf_path).getroot()
    src_world = root.find("world")
    if src_world is None:
        raise ValueError(f"No <world> element found in {packaged_sdf_path}")

    starts: list[list[float]] = scenario["starts"]
    goals: list[list[float]] = scenario["goals"]
    num_drones = len(starts)
    start_cx = sum(p[0] for p in starts) / num_drones
    start_cy = sum(p[1] for p in starts) / num_drones
    goal_cx = sum(p[0] for p in goals) / num_drones
    goal_cy = sum(p[1] for p in goals) / num_drones
    corridor_yaw = math.atan2(goal_cy - start_cy, goal_cx - start_cx)

    reused_elements: list[str] = []
    for child in src_world:
        if child.tag in ("scene", "light", "model"):
            reused_elements.append("  " + ET.tostring(child, encoding="unicode"))

    showcase_elements = [_showcase_obstacle_sdf(obs) for obs in SHOWCASE_OBSTACLES]
    glow_elements = [_obstacle_glow_sdf(obs) for obs in trajectories["glow_obstacles"]]
    drone_elements = [
        _visual_drone_sdf(
            idx,
            (float(starts[idx][0]), float(starts[idx][1])),
            corridor_yaw,
            DRONE_COLORS[idx % len(DRONE_COLORS)],
        )
        for idx in range(num_drones)
    ]
    lidar_elements: list[str] = []
    for idx in range(num_drones):
        lidar_elements.append(_lidar_ring_sdf(f"lidar_clear_{idx}", alert=False))
        lidar_elements.append(_lidar_ring_sdf(f"lidar_alert_{idx}", alert=True))

    trail_elements: list[str] = []
    for idx in range(num_drones):
        for b in range(TRAIL_BEADS_PER_DRONE):
            trail_elements.append(_trail_bead_sdf(idx, b, DRONE_COLORS[idx % len(DRONE_COLORS)]))

    # Elevated SE -> NW 3/4 camera framing START_ZONE, all showcase obstacles, both
    # North/South weave wings, and GOAL_ZONE destination markers cleanly from launch.
    gui_block = """  <gui fullscreen="0">
    <plugin filename="MinimalScene" name="3D View">
      <gz-gui>
        <title>3D View</title>
        <property type="bool" key="showTitleBar">false</property>
        <property type="string" key="state">docked</property>
      </gz-gui>
      <engine>ogre2</engine>
      <scene>scene</scene>
      <ambient_light>0.48 0.48 0.50</ambient_light>
      <background_color>0.74 0.78 0.84</background_color>
      <camera_pose>17.0 -16.0 15.5 0.0 0.52 2.34</camera_pose>
    </plugin>
    <plugin filename="GzSceneManager" name="Scene Manager">
      <gz-gui>
        <property key="resizable" type="bool">false</property>
        <property key="width" type="double">5</property>
        <property key="height" type="double">5</property>
        <property key="state" type="string">floating</property>
        <property key="showTitleBar" type="bool">false</property>
      </gz-gui>
    </plugin>
    <plugin filename="InteractiveViewControl" name="Interactive view control">
      <gz-gui>
        <property key="resizable" type="bool">false</property>
        <property key="width" type="double">5</property>
        <property key="height" type="double">5</property>
        <property key="state" type="string">floating</property>
        <property key="showTitleBar" type="bool">false</property>
      </gz-gui>
    </plugin>
    <plugin filename="WorldControl" name="World control">
      <gz-gui>
        <title>World control</title>
        <property type="bool" key="showTitleBar">false</property>
        <property type="bool" key="resizable">false</property>
        <property type="double" key="height">72</property>
        <property type="double" key="width">121</property>
        <property type="double" key="z">1</property>
        <property type="string" key="state">floating</property>
        <anchors target="3D View">
          <line own="left" target="left"/>
          <line own="bottom" target="bottom"/>
        </anchors>
      </gz-gui>
      <play_pause>true</play_pause>
      <step>false</step>
      <start_paused>false</start_paused>
    </plugin>
    <plugin filename="WorldStats" name="World stats">
      <gz-gui>
        <title>World stats</title>
        <property type="bool" key="showTitleBar">false</property>
        <property type="bool" key="resizable">false</property>
        <property type="double" key="height">90</property>
        <property type="double" key="width">260</property>
        <property type="double" key="z">1</property>
        <property type="string" key="state">floating</property>
        <anchors target="3D View">
          <line own="right" target="right"/>
          <line own="bottom" target="bottom"/>
        </anchors>
      </gz-gui>
      <sim_time>true</sim_time>
      <real_time>true</real_time>
      <real_time_factor>true</real_time_factor>
      <iterations>false</iterations>
    </plugin>
  </gui>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  {REQUIRED_LABEL}
  STANDALONE SCRIPTED VISUALIZATION ONLY — NOT VALIDATED AUTONOMOUS FLIGHT.
  Reuses the packaged 10-drone arena appearance, obstacles, destination markers,
  and x500_base drone meshes alongside scripted showcase obstacles, animated LiDAR
  rings/rays, proximity glow halos, and movement trails. Completely disconnected
  from PX4, ROS 2, MAVLink, and DDS.
-->
<sdf version="1.9">
<world name="{WORLD_NAME}">
  <physics name="visual_demo_physics" type="ode">
    <max_step_size>0.01</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>100</real_time_update_rate>
  </physics>
  <gravity>0 0 0</gravity>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
{gui_block}
{chr(10).join(reused_elements)}
{chr(10).join(showcase_elements)}
{chr(10).join(glow_elements)}
{chr(10).join(drone_elements)}
{chr(10).join(lidar_elements)}
{chr(10).join(trail_elements)}
</world>
</sdf>
"""


def main() -> int:
    scenario = json.loads(PACKAGED_SCENARIO.read_text(encoding="utf-8"))
    trajectories = build_prerecorded_trajectories(scenario)
    world_sdf = build_visual_world_sdf(PACKAGED_WORLD, scenario, trajectories)

    OUT_WORLD.parent.mkdir(parents=True, exist_ok=True)
    OUT_WORLD.write_text(world_sdf, encoding="utf-8")
    OUT_TRAJECTORIES.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    checks = trajectories["geometry_checks"]
    print(f"Wrote {OUT_WORLD}")
    print(
        f"Wrote {OUT_TRAJECTORIES} ({len(trajectories['frames'])} frames, "
        f"{trajectories['duration']:.1f} s)"
    )
    print(
        f"Checks: min_pair_sep={checks['min_pair_separation_m']} m, "
        f"min_obstacle_clearance={checks['min_obstacle_clearance_m']} m, "
        f"scripted_alert_drone_frames={checks['scripted_alert_drone_frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
