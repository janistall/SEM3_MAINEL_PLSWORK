# AGENTS.md

ROS 2 (ament) workspace for a 3-drone PX4/Gazebo swarm. Four packages under `src/`; there is no README or CI.

## Packages

- `swarm_interfaces` — `ament_cmake`, generates `AgentTelemetry` / `NeighborList`.
- `swarm_control` — 23-D observation / 4-D action controller, supervisor, CBF shield, A* macro.
- `px4_adapter` — only package that talks to `px4_msgs`; PX4 NED ↔ world ENU bridge.
- `swarm_bringup` — launch file, seed-7 scenario/world, checkpoints, PX4 airframe.

## Build and test

- `px4_msgs` is an external ROS package and is not vendored; put it in `src/` or an underlay or `colcon build` will fail.
- `source /opt/ros/<distro>/setup.bash && colcon build --symlink-install && source install/setup.bash`
  Use `colcon build --packages-select swarm_control` for a focused rebuild.
- `swarm_bringup` imports `swarm_control.planning`, so its console script requires `swarm_control` to be built (colcon derives the rest of the order from `package.xml`).
- Tests are pure NumPy and need no ROS, PyTorch, or simulator:
  `PYTHONPATH=src/swarm_control python3 -m unittest discover -s src/swarm_control/test -v`
  Filter to one test with `-k test_shield_avoids_obstacle_cylinder`. If pytest is installed, `python3 -m pytest src/swarm_control/test/test_control.py -q` runs the same tests. `colcon test --packages-select swarm_control` also works.
- There is no lint, formatter, or typecheck configuration; do not assume commands for them.

## Running

- `ros2 launch swarm_bringup swarm.launch.py mode:=classical headless:=true`
- `mode` is `learned`, `classical`, or `zero`. `learned` defaults to the packaged `checkpoints/swarm_macro_policy.pt`, which carries `swarm-coop-enu-v1` contract metadata.
- The launch file requires exactly 3 starts and 3 goals and a PX4 SITL binary. It defaults to a hard-coded path from another machine (`/home/siddharthg/PX4-Autopilot`); set `PX4_DIR` or pass `px4_dir:=`.
- It also expects `MicroXRCEAgent` and `gz` on `PATH`; the airframe `4002_gz_x500_swarm` is installed into the PX4 checkout by `scripts/install_px4_airframe.sh`.
- `maddpg_iter00460.pt` has no `contract` metadata and is rejected unless `allow_legacy_checkpoint:=true`; only do that with independently verified semantics.

## Gotchas

- The controller contract is centralized in `swarm_control/contract.py` (`SCHEMA = "swarm-coop-enu-v1"`). Changing observation layout or action decoding means updating `observation.py`, `contract.py`, `test/test_control.py`, and checkpoint metadata together.
- Control flow per drone: policy or `ClassicalMacro` → `Supervisor` (`APPROACH`/`DOCKING`/`HOLD`/`RECOVERY`) → `safe_velocity` shield → adapter. The shield can zero the command when constraints are infeasible, so a "stuck" drone may be reporting `shield_feasible: false`.
- The controller emits zero velocity unless all 3 `/{drone}/telemetry` streams are active and fresh within 0.5 s. Telemetry uses RELIABLE QoS; PX4 topics use BEST_EFFORT depth 1.
- Coordinates are world ENU everywhere in `swarm_control`/scenario JSON. `px4_adapter._enu_state()` and `_publish_setpoint()` do the NED swap and add the `spawn_x/y` offset; change both together.
- Exactly 3 agents are supported: `agent_ids` must have length 3 and `drone_id` must equal `agent_ids[agent_index]`.
- `ros2 run swarm_bringup generate_world --seed 7 --obstacles 7 --out DIR` writes both `scenario.json` and `swarm.sdf`. The packaged `config/scenario.json` and `worlds/swarm.sdf` are pre-generated; the launch file does not regenerate either, so keep them consistent by hand or regenerate both.
- `contract.py` references a `README.md` that does not exist; its "checkpoint missing" error text is also stale (the packaged checkpoints are present).
