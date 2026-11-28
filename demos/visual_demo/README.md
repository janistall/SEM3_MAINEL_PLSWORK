# 10-Drone Gazebo Visual Research Demonstration

> **Scripted illustration of proposed LiDAR and AI-assisted navigation.**
>
> All drone motion, obstacle highlights, LiDAR scanning rings/rays, movement trails, and dashboard states are **fixed, prerecorded visualization effects** authored manually for visual demonstration only. This demo is **completely disconnected** from PX4 SITL, ROS 2 flight-control topics, MAVLink, MicroXRCEAgent, DDS, live sensing, autonomous path planning, and policy inference. No AI performance metrics are produced or implied.

## Visual Highlights (54-Second 6-Act Sequence)

1. **Multi-shape, multi-height obstacles:** Reuses all 7 packaged cylindrical pillars (`obstacle_0..6`) plus 6 prominent architectural structures (`showcase_low_bunker` `2.6 m`, `showcase_monolith` `7.2 m` stepped tower, `showcase_north_block` `4.6 m`, `showcase_south_wall` `3.8 m`, `showcase_radar_tower` `6.4 m`, and `showcase_gantry_arch` `5.8 m`).
2. **Formation separation, obstacle weave & regrouping:** Manually authored 6-act choreography:
   - `0–3 s`: Ground staging on `START_ZONE`
   - `3–9 s`: Staggered smooth takeoff
   - `9–19 s`: Formation split into **North Wing (`demo_drone_0..4`)** and **South Wing (`demo_drone_5..9`)** around the central obstacle cluster
   - `19–34 s`: Curved twin-wing S-curve weave with subtle cosmetic banking and altitude variation
   - `34–45 s`: Mid-arena formation regrouping and approach to `GOAL_ZONE`
   - `45–54 s`: Exact arrival and level hold at color-matched `goal_beacon_0..9` markers
3. **Animated LiDAR-style rings & proximity highlights:** Spinning green scanning rings and radial rays around all 10 drones in clear space, switching to vivid orange-red rings/rays alongside glowing orange-red obstacle collars when passing nearby obstacles.
4. **Visible 3-D movement trails:** Color-matched glowing trail beads dropped behind all 10 drones showing the dual-wing separation and regrouping paths across the arena.
5. **Improved overview camera:** Pre-configured SE-to-NW 3/4 elevated camera framing `START_ZONE`, all showcase obstacles, both weave corridors, and `GOAL_ZONE` beacons immediately on launch.
6. **Live Research Dashboard:** Both an in-terminal live dashboard and a loopback web dashboard (`http://127.0.0.1:8788`, auto-opened) displaying animation progress, act timeline, 2-D synchronized arena view, drone IDs (`demo_drone_0..9`), illustrated perception status, and destination arrival.

## Launch Command (Single Host Terminal)

```bash
bash /home/siddharthg/SEM3_MAINEL_PLSWORK/demos/visual_demo/launch_visual_demo.sh
```

Optional flags:
- `--loop` — continuously replay the 54-second sequence.
- `--no-browser` — keep the web dashboard available at `http://127.0.0.1:8788` without auto-opening a browser tab.
- `--speed 1.25` — adjust playback speed multiplier (default `1.0`).

Press `Ctrl+C` in the terminal or close the Gazebo window to exit cleanly.
