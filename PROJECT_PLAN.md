# Project Plan — Decentralised UAV Swarm with MARL

**Status:** Latest verified implementation commit **`144f0c73165a467339bd6451d336e0b56bfa78b9`** on `origin/feature/m1-m2-integration` implements Phases 1–6 (frozen Contract v2 `swarm-coop-enu-v2`, 10-drone Scenario/World, 3-D Slack QP Shield, Pure-PyTorch MAPPO Stack, Colab Package, read-only simulation diagnostics MCP bridge with `gazebo_version` & loopback fixes, Fast DDS loopback isolation `fastdds_loopback.xml`, 10-drone Contract v2 `node.py`/`controller_core.py` & `swarm.launch.py`/`launch_plan.py` migration, M1-owned Phase 8 adapter noise & wind/rain hooks, and 10 Hz adapter telemetry decimation). **All 51/51 regression tests pass (`36` swarm/M1/MARL + `15` MCP)** and all 4 ROS 2 Humble packages build cleanly. Grounded 10-drone headless SITL validation and clean shutdown have **PASSED** live MCP verification: **10/10 grounded drones connected**, **sustained RTF `0.999`** (`413.4%` CPU, `0/10` freshness violations under host `performance` CPU governor at `4813 MHz`), **10/10 valid PX4 ULogs with zero dropouts**, **clean shutdown (`0` orphaned simulation processes, `0` stale ROS nodes)**, and **27 GB host disk space available** (`71%` used, `~9 GB` existing PX4 logs retained untouched). **Active-flight M1 validation (Steps 2–5: armed hover `Z`, classical transit `C+S`, runtime hooks/quorum, graphical rendering) remains PENDING**, M3/M4 nodes remain pending, and **M2 training remains DEFERRED** (Stage 1 acceptance gate NOT met; Stage 2 NOT started).
**Scope:** make the neural policy the actual controller of the swarm, **start directly with 10 drones on Contract v2 (`swarm-coop-enu-v2`)** (skipping an intermediate 3-drone v1 training cycle), allow runtime target selection, and demonstrate robustness to rain / birds / obstacles.
**Audience:** all four team members.

---

## THIS IS WHAT WAS DONE SO FAR

1. **Contract v2 (`swarm-coop-enu-v2`) implemented as the single source of truth in `src/swarm_control/swarm_control/contract.py`:**
   - Default 10-drone configuration: `agents = 10`, `k_peer = 4`, `k_static = 3`, `k_dyn = 2`, computed `obs_dim = 62`, `act_dim = 4`, `hidden = (256, 256)`.
   - Unified scales & physical limits: `goal_scale = 25.0`, `vel_scale = 1.5`, `peer_vel_scale = 3.0`, `obstacle_scale = 25.0`, `speed = 1.5 m/s`, `accel = 2.0 m/s²`, `yaw_rate = 0.6 rad/s`, `radius = 0.45 m`, `alt = 2.5 m`.
   - Tiered safety thresholds: AI soft spacing `pair_safe = 1.1 m` vs. shield hard backstop `hard_pair = 0.9 m`; AI soft obstacle clearance `obs_margin = 0.25 m` vs. shield hard backstop `hard_margin = 0.0 m`.
   - Added pure-Python metadata & checkpoint helpers: `validate_action_spec()`, `apply_contract_metadata()` (non-mutating copy of `DEFAULT_CONTRACT`), `normalize_actor_state_dict()` (supports both bare `nn.Sequential` keys and `net.`-prefixed keys), and `make_deploy_payload()` (`weights_only=True`-safe deployment export).
   - Fixed float64 precision in `action_to_velocity()` under NumPy 2.x.

2. **62-D Observation v2 Builder implemented in `src/swarm_control/swarm_control/observation.py`:**
   - Added optional `agent_id: str = ""` to `Telemetry` and introduced `DynamicObstacle(position, velocity, radius=0.35, active=True)`.
   - Implemented `build_observation(agent_index, telemetry, goal, obstacles=(), contract=DEFAULT_CONTRACT, dynamic_obstacles=())` with nearest-4 active peers sorted by `(dist_3d, agent_id, index)`, nearest-3 static obstacles sorted by `(dist_2d - o.radius, index)`, nearest-2 active dynamic obstacles sorted by `(dist_3d - d.radius, index)`, explicit `valid` flags, and strict zero-padding (including `clearance = 0.0`).

3. **Step 3 — Upgraded Shield (`src/swarm_control/swarm_control/safety.py`):**
   - Updated `ShieldConfig` defaults to v2 (`speed=1.5`, `accel=2.0`, `pair_safe=0.9`, `radius=0.45`, `obstacle_margin=0.0`, `workspace=±15 m`) and added `ShieldConfig.from_contract(contract)`.
   - Locked `safe_velocity(..., dynamic_obstacles: Sequence[DynamicObstacle] = ())` signature with moving-obstacle CBF (`n · (v_i - v_o) >= -alpha * (||p_i - p_o|| - r_o - r_i - margin)`) and optional `lookahead_tau`.
   - Implemented the **fast-path zero-violation check** (skips solver when `v_box` satisfies all constraints), **pure-NumPy prioritized-slack 3-D QP** (`weight_peer=400 > weight_dyn=300 > weight_static=250 > weight_workspace=150`), and **active brake/retreat fallback** when inside a hard collision envelope (`feasible=False` with a non-zero separating velocity instead of freezing at `[0, 0, 0]`).

4. **Single-Source-of-Truth Training Re-export in `marl_training/contracts.py` (+ `marl_training/__init__.py`):**
   - Replaced the divergent Gen-1 `marl_training/contracts.py` with a direct re-export of `swarm_control.contract`, `swarm_control.observation`, and `swarm_control.safety`.

5. **Frozen ROS 2 Message & Service Schemas in `src/swarm_interfaces/` (for Phases 7 & 8):**
   - Added `msg/DynamicObstacle.msg` (`id`, `position`, `velocity`, `radius`, `active`) and `msg/DynamicObstacleArray.msg` (`header`, `obstacles[]`) for `/obstacles/dynamic`.
   - Added `srv/SetFormation.srv` (`center`, `shape`, `spacing` → `accepted`, `message`, `slots[]`) for `/swarm/set_formation`.
   - Registered all new interfaces in `src/swarm_interfaces/CMakeLists.txt`.
   - Locked `node.py` single-owner integration order (M1 integrates Phase 5 → 6 → 7 → 8 hooks).

6. **Step 2 — 10-Drone Scenario & Gazebo World Generator (`src/swarm_bringup/swarm_bringup/scenario.py` + `src/swarm_control/swarm_control/planning.py`):**
   - Expanded `astar()` default bounds in `planning.py` to `(-14.8, 14.8)` for the `±15 m` 10-drone arena.
   - Upgraded `scenario.py` with `formation_slots(count=10, shape="ring"|"grid"|"v", spacing=1.6)`, 10-drone A*-reachable map sampling, `-15..15` ground grid, `>= 4.0 m` start/goal zone discs, and a 10-color beacon palette.
   - Regenerated both packaged `src/swarm_bringup/config/scenario.json` and `src/swarm_bringup/worlds/swarm.sdf` together for 10 drones (`seed=7, obstacles=7`).

7. **Phase 2 — 10-Drone Contract v2 Training Environment (`marl_training/envs.py` + `src/swarm_control/test/test_env_v2.py`):**
   - Rewrote `DroneSwarmEnv` (`drone_swarm_enu_v2`, 10 agents `drone_0..drone_9`) calling `swarm_control` deployment functions (`build_observation`, `action_to_velocity`, `safe_velocity(..., dynamic_obstacles=...)`) directly.
   - Implemented `generate_formation_slots`, 4-stage curriculum presets (`STAGE_CONFIGS`), static cylinders (`0..15`), crossing dynamic birds (`0..4` with arena bounce), mid-episode formation goal shifts, domain randomisation (`lag_tau`, wind gusts, odometry noise, `0..30%` peer dropout, `0..200 ms` latency), `scenario_path` JSON loading, per-slot reward shaping (including terminal `swarm_success` `+80` bonus, closing-speed/TTC avoidance shaping, per-culprit collision penalty, and `--shield-off-prob` episode sampling in commits `f331d6b`, `6e959c0`, `5a971c1`), and §8.4 synchronized 10-drone termination.

8. **Phase 3 & Phase 4 (+ Step 7 Colab Package) — Pure-PyTorch Parameter-Shared MAPPO Training Stack:**
   - Rewrote `marl_training/models.py` (`SwarmActorNet` `62→256→256→4` with bare `nn.Sequential` `self.net` + clamped `log_std`, and `CentralizedValueNet` `682→256→256→1` with zero Ray RLlib dependency).
   - Created `marl_training/mappo.py` (`RolloutBuffer` with unclipped `raw_actions`, GAE with truncation `V(s_next)` bootstrapping vs. termination zeroing, and `MAPPOTrainer` with clipped PPO + resumable `state_dict`).
   - Rewrote `marl_training/train.py` (4-stage curriculum CLI, `--shield-off-prob`, `<out>/metrics.jsonl`, deterministic stage-matched evaluation + shield-off pass, dual checkpoint export `swarm_policy_best.pt` / `swarm_policy_final.pt` vs. `train_state_latest.pt` / rolling last-3 `train_state_iterNNNNN.pt`, `--resume`, `--init-weights`).
   - Created `marl_training/eval.py` (headless kinematic evaluator for `open`, `clean`, `rain`, `birds`, `clutter`, `target_change`, and `--stage <curriculum_stage>` with both `shield=True` and `shield=False`).
   - Added `requirements-colab.txt`, updated `marl_training/requirements.txt` (removed `ray[rllib]`), and created the 9-cell `marl_training/colab_train.ipynb` pointing to `https://github.com/janistall/SEM3_MAINEL_PLSWORK.git`.

9. **Stage 1 MAPPO Training Run, Evaluation-Mismatch Fix, and Verification (`b311940`):**
   - **Fixed Stage 1 training vs. standalone evaluation mismatch (commit `b311940`):** Added `open` evaluation mode and `--stage` CLI flag to `marl_training/eval.py` (mirroring `train.py` stage-matched evaluation via `DroneSwarmEnv.for_stage`), added a stage-mismatch `UserWarning` when a `stage1_open` checkpoint is evaluated on obstacle modes/stages, updated `marl_training/colab_train.ipynb` Cell 8, and added regression coverage (`test_eval_stage1_open_mode_cli_and_mismatch_warning`).
   - **Regression test suite:** **All 25 regression tests passed** across `test_contract_v2_parity.py` (10), `test_control.py` (5), `test_env_v2.py` (5), and `test_mappo_and_steps_2_3.py` (5).
   - **Branch & checkpoint storage:** Commit `b311940` was successfully pushed to `origin/feature/m1-m2-integration`. Training checkpoints (`swarm_policy_*.pt`, `train_state_*.pt`, `metrics.jsonl`) are stored in **Google Drive**, not GitHub.
   - **Verified Stage 1 (`stage1_open`) MAPPO training & evaluation results (Google Colab T4 GPU):**

     | Metric / Check | Verified Result | Status vs. Gate |
     |---|---|---|
     | Training budget | **250 iterations / 600,000 environment steps** (Colab T4 GPU) | Completed |
     | Best training evaluation (5 episodes) | **100% success** at **iteration 210** | Training peak |
     | Final training evaluation | **80% success** at **iteration 250** | Late-stage drift |
     | Independent 20-ep Stage 1 eval (`L+S`, shield enabled) | **70% success**, **92% docking**, **0 collisions** | **Below ≥ 85% Stage 1 gate** |
     | Independent 20-ep Stage 1 eval (`L`, shield disabled) | **10% success**, **10% docking**, **75% collision rate** | **Shield carrying avoidance** |
     | Stage 1 acceptance criteria | **NOT met** | Blocked on Stage 1 |
     | Stage 2 (`stage2_static`) training | **NOT started** | Waiting on Stage 1 gate |

10. **GitHub, ROS 2, Gazebo, and PX4 MCP Integration & Workspace Build Verification (`116e87bbaa06f740f105a3c7242774bc66ffd7dd`):**
    - **Built all 4 ROS 2 Humble workspace packages** (`swarm_interfaces`, `swarm_control`, `px4_adapter`, `swarm_bringup`) cleanly with `colcon`, chaining the `px4_msgs` underlay (`/home/siddharthg/uav_ws/install`) and the project overlay (`/home/siddharthg/SEM3_MAINEL_PLSWORK/install`).
    - **Implemented & pushed the standalone read-only simulation diagnostics MCP bridge** in `tools/sim_diagnostics_mcp/` (commit `116e87bbaa06f740f105a3c7242774bc66ffd7dd` on `origin/feature/m1-m2-integration`; added `.venv/` to `.gitignore` so local virtual environments are never committed).
    - **Strictly loopback & read-only:** Bound to `http://127.0.0.1:8765/mcp` (Python `3.10.12`, FastMCP Streamable HTTP), `read_only: true`, `physical_drone_access: false`, allowlisted read-only subprocesses (`shell=False`), canonical log-path containment, and dynamic `1..N` drone discovery across 10 read-only tools (`sim_check_dependencies`, `sim_discover_drones`, `ros2_graph_overview`, `ros2_node_details`, `ros2_topic_inspect`, `swarm_diagnostics_snapshot`, `gazebo_world_stats`, `gazebo_topic_inspect`, `px4_sitl_status`, `px4_sitl_logs`).
    - **Verified MCP unit tests & live dependency checks:** **All 15 MCP unit tests passed** (`tools/sim_diagnostics_mcp/tests/test_bridge.py`). Live `sim_check_dependencies` confirmed `rclpy`, `swarm_interfaces`, `px4_msgs`, `mcp`, `ros2`, `gz`, `MicroXRCEAgent`, `ps`, `ulog_info`, `workspace_install_exists: true`, `px4_sitl_binary_exists: true`, `4002_gz_x500_swarm` airframe installed, and `remediation_hints: []`. GitHub MCP connectivity confirmed via `get_me` (`s1nz991`) and `get_commit(116e87b)`.
    - **Preserved frozen contract:** Zero modifications to `src/swarm_control/`, `src/swarm_interfaces/`, `src/px4_adapter/`, `src/swarm_bringup/`, or `swarm-coop-enu-v2`.

11. **M1 10-Drone Simulation Migration, Fast DDS Loopback Isolation, MCP Fixes, Grounded 10-Drone Performance Gate & Clean Shutdown (`144f0c73165a467339bd6451d336e0b56bfa78b9`, 2026-11-26/27):**
    - **Verified commit on GitHub:** Pushed to `origin/feature/m1-m2-integration` at **`144f0c73165a467339bd6451d336e0b56bfa78b9`** (20 files, `+2320 / -467`, frozen Contract v2 untouched).
    - **Mandatory MCP fixes & live verification:** Fixed `gazebo_version` (`gz sim --version` / `--versions` allowlisted in `runner.py` & `health.py`), defaulted `ROS_LOCALHOST_ONLY=1` + `ROS_DOMAIN_ID=42` + `GZ_IP=127.0.0.1` + `GZ_PARTITION=swarm` in `tools/sim_diagnostics_mcp/run_bridge.sh`, and added PX4 v1.15+ `vehicle_status_v1` support in `discovery.py` & `ros2_diagnostics.py`. All **15/15 MCP tests pass**.
    - **Loopback Fast DDS discovery fix:** Added `src/swarm_bringup/config/fastdds_loopback.xml` (UDPv4 `127.0.0.1` whitelist, `px4_participant` default profile, hyphen-free filename for `MicroXRCEAgent -r`), wired `FASTDDS_DEFAULT_PROFILES_FILE` + `PX4_PARAM_UXRCE_DDS_PTCFG=2` + `PX4_PARAM_UXRCE_DDS_AG_IP=2130706433` in `swarm.launch.py`, and validated live DDS discovery on the grounded 3-drone baseline (`224` topics, `RTF = 0.999985`, live odometry/status/ENU telemetry).
    - **10-drone Contract v2 controller & launch migration:** Added pure-Python `src/swarm_control/swarm_control/controller_core.py` and upgraded `node.py` to dynamic `N` (default 10), 62-D obs / 4-D action, `ShieldConfig.from_contract` (`±15 m`, `hard_pair=0.9`, `hard_margin=0.0`), unconditional `Supervisor` & stall-recovery bypass in `mode:=learned`, geometric obstacle-clearance stall recovery in `mode:=classical`, §12.3 quorum telemetry gating (`quorum_min_peers=3`, stale peers zero-padded `valid=0`), `Telemetry.agent_id`, `/{drone}/goal` (`PoseStamped`, `TRANSIENT_LOCAL`), `/obstacles/dynamic` (`DynamicObstacleArray`, timeout expiry), `/{drone}/policy_debug` (75 floats), and expanded 1 Hz `/{drone}/status` JSON. Added `src/swarm_bringup/swarm_bringup/launch_plan.py` and upgraded `swarm.launch.py` for default 10-drone & configurable 3/6/10-drone launches, staggered PX4 startup (`6 + 2*i`), synced `takeoff_altitude`, `auto_arm:=false` default, checkpoint-free `zero`/`classical`, and graphical/headless modes.
    - **M1 Phase 8 disturbance hooks & lightweight 10-drone optimizations:** Added `src/px4_adapter/px4_adapter/noise.py` (optional Gaussian ENU telemetry noise, default `0.0`), `src/px4_adapter/px4_adapter/rate.py` (`TelemetryRateLimiter`, default `telemetry_rate:=10.0` Hz while keeping PX4 offboard setpoints at `20 Hz` and preserving all timeout thresholds), BLAS/OpenMP/PyTorch thread pinning (`*=1`) in `swarm.launch.py`, optional `--wind` / cosmetic `--rain` SDF hooks in `scenario.py`, and shadow-disabled / single-`arena_grid` regeneration of `worlds/swarm.sdf`.
    - **Regression tests & build:** Added `test_dds_isolation_config.py` and `test_m1_migration.py`; **all 51/51 regression tests pass** (`36` swarm/M1/MARL + `15` MCP), and all 4 ROS 2 Humble packages build cleanly with `colcon`. Frozen `swarm-coop-enu-v2` files (`contract.py`, `observation.py`, `safety.py`, `swarm_interfaces/`, `config/scenario.json`, `marl_training/`) remain untouched.
    - **Grounded 10-drone live MCP benchmark, ULog & clean shutdown verification (PASS):**

      | Configuration (10 drones, `mode:=zero`, `auto_arm:=false`) | Sustained RTF | Aggregate CPU% | RAM% | `px4_status_age` mean / max | `>1.0 s` Violations |
      |---|---:|---:|---:|---:|---:|
      | Original Graphical (20 Hz tel., `powersave` 1171 MHz) | `0.666` *(inst)* | `1,160.6%` | `21.0%` | `0.937 s / 1.248 s` | `5 / 10` |
      | Optimized Graphical (10 Hz tel., `powersave` 1171 MHz) | `0.591` | `1,060.4%` | `19.1%` | `0.684 s / 1.096 s` | `6 / 30` |
      | Optimized Headless (10 Hz tel., `powersave` 1171 MHz) | `0.693` | `939.2%` *(steady)* | `13.9%` | `0.606 s / 0.955 s` | `0 / 30` |
      | **Optimized Headless (10 Hz tel., `performance` 4813 MHz)** | **`0.999`** *(95.3 s)* | **`413.4%`** *(~26% of 16T)* | **`13.9%`** | **`0.264 s / 0.468 s`** | **`0 / 10`** |

      - **10/10 grounded drones connected:** `x500_0..9`, `px4 -i 0..9`, 20 ROS nodes, 773 topics, `gz model --list` 0 timeouts, all disarmed, `telemetry_age` `0.042–0.130 s`, `px4_odometry_age` `0.040–0.052 s`, full loopback isolation verified.
      - **10/10 valid PX4 ULogs:** Verified via MCP `px4_sitl_logs` + `ulog_info` (`18_03_09.ulg` .. `18_03_27.ulg`, `36–38 min`, **`No Dropouts`** on all 10 instances).
      - **Clean shutdown verified (Step 1):** `Ctrl+C` (`SIGINT`) exited all 32 simulation processes cleanly (`0` orphaned `px4`/`gz`/`MicroXRCEAgent`/`adapter`/`controller` processes, `0` stale ROS nodes, `0` Gazebo worlds/topics).
      - **Host disk space verified:** **27 GB available** (`71%` utilization; `~9 GB` of existing PX4 logs retained untouched).
      - **Pre-flight audit caveat noted (no code modified):** `PX4AdapterNode._on_land()` currently sends `VEHICLE_CMD_NAV_LAND` without a `LANDING` state, so the 20 Hz `_tick()` loop re-requests `OFFBOARD` after `1.0 s`; decide whether to add a minimal `LANDING` phase before armed flight (Option A) or use direct `SIGINT` shutdown (Option B).

---

## M1 & M2 Implementation Status Audit (Code & Test Evidence)

All statuses below are derived directly from verified commit **`144f0c73165a467339bd6451d336e0b56bfa78b9`** on `origin/feature/m1-m2-integration`, the **51-test regression suite (36 swarm/M1/MARL + 15 MCP)**, the 4-package colcon build, and live read-only MCP diagnostics. The frozen integration contract (`swarm-coop-enu-v2`, 62-D observation, 4-D action, tiered shield thresholds, and `swarm_interfaces` messages/services) remains intact and unchanged.

### Completed

| Owner | Component | Code & Test Evidence |
|---|---|---|
| **M1** | Contract v2 (`swarm-coop-enu-v2`) & `ActorPolicy` strict `weights_only=True` loader | `src/swarm_control/swarm_control/contract.py` (`Contract`, `apply_contract_metadata`, `validate_action_spec`, `normalize_actor_state_dict`, `make_deploy_payload`, `ActorPolicy`); verified by `test_contract_v2_parity.py` (10 tests). |
| **M1** | 62-D Observation v2 builder (`k_peer=4`, `k_static=3`, `k_dyn=2`) | `src/swarm_control/swarm_control/observation.py` (`Telemetry.agent_id`, `DynamicObstacle`, `build_observation`); verified by `test_control.py::test_observation_contract_layout` & `test_contract_v2_parity.py`. |
| **M1** | Upgraded 3-D prioritized-slack QP CBF shield (Step 3) | `src/swarm_control/swarm_control/safety.py` (`ShieldConfig.from_contract`, moving-obstacle `n · v_o` term + `lookahead_tau`, zero-violation fast-path, `_solve_prioritized_slack_qp`, active retreat-on-infeasible fallback); verified by `test_control.py` (3 shield tests), `test_contract_v2_parity.py`, and `test_mappo_and_steps_2_3.py::test_step3_shield_fast_path_qp_and_retreat_fallback`. |
| **M1** | 10-drone scenario & Gazebo world generator (Step 2) | `src/swarm_control/swarm_control/planning.py` (`±14.8 m` A* bounds), `src/swarm_bringup/swarm_bringup/scenario.py` (`formation_slots`, 10-drone sampling, `±15 m` merged `arena_grid`, `>= 4.0 m` zone discs, 10 beacons, `--wind`/`--rain`), regenerated `config/scenario.json` & `worlds/swarm.sdf`; verified by `test_mappo_and_steps_2_3.py` & `test_m1_migration.py`. |
| **M1** | Frozen ROS 2 interface definitions & 4-package colcon build | `src/swarm_interfaces/msg/DynamicObstacle.msg`, `msg/DynamicObstacleArray.msg`, `srv/SetFormation.srv`, registered in `CMakeLists.txt`; all 4 workspace packages built and verified importable (`rclpy`, `swarm_interfaces`, `px4_msgs`) via `sim_check_dependencies`. |
| **M1** | Phase 5 & Phase 6 ROS 2 10-drone deployment migration & grounded SITL + clean shutdown (`144f0c7`) | Dynamic `N`-drone controller (default 10), `Supervisor` & stall bypass in `learned`, geometric stall recovery in `classical`, §12.3 quorum telemetry gating, `/{drone}/goal`, `/obstacles/dynamic`, `/{drone}/policy_debug`, 1 Hz `/{drone}/status`, 3/6/10-drone staggered launch, `fastdds_loopback.xml` loopback DDS isolation, 10 Hz adapter telemetry rate limiter (`rate.py`), adapter ENU noise (`noise.py`); verified by `test_dds_isolation_config.py`, `test_m1_migration.py` (`36/36`), live 10-drone grounded SITL (`RTF = 0.999`, `0/10` freshness violations, `10/10` valid ULogs with `No Dropouts`), and clean `SIGINT` shutdown (`0` orphans, `0` stale ROS nodes). |
| **M1/M2** | Single-source-of-truth contract re-export | `marl_training/contracts.py` directly re-exports `swarm_control` primitives; verified by `test_contract_v2_parity.py::test_single_source_of_truth_reexports`. |
| **M2** | 10-drone Contract v2 training env, MAPPO stack, CLI, evaluator, & Colab notebook (Phases 2–4, Step 7) | `marl_training/envs.py`, `models.py`, `mappo.py`, `train.py`, `eval.py` (incl. `open` mode & `--stage` fix in `b311940`), `colab_train.ipynb`, `requirements-colab.txt`; verified by `test_env_v2.py` (5 tests) and `test_mappo_and_steps_2_3.py` (3 MAPPO/eval tests). |
| **Infra** | Read-only Simulation Diagnostics MCP Bridge & mandatory hardening | `tools/sim_diagnostics_mcp/` (10 read-only tools, strict `127.0.0.1:8765/mcp` binding, `gz sim --version(s)` detection, `ROS_LOCALHOST_ONLY=1` + `ROS_DOMAIN_ID=42` defaults, PX4 `vehicle_status_v1` support); verified by **15/15 MCP unit tests** and live grounded 3-drone & 10-drone SITL diagnostics. |

### Partially Completed

| Owner | Phase / Task | What Is Done | What Remains Pending (Code & Live Evidence) |
|---|---|---|---|
| **M1/M3** | **Phase 7 — Selectable targets** | `SetFormation.srv` schema frozen; `/{drone}/goal` (`PoseStamped`, `TRANSIENT_LOCAL`) runtime validation & classical A* replanning implemented in `node.py`/`controller_core.py` and unit-tested. | Live active-flight verification of `/{drone}/goal` (Step 4a, M1) and M3 `formation_allocator` node (`/swarm/formation_pose`, `/swarm/set_formation`). |
| **M1/M3** | **Phase 8 — Rain, birds, other obstacles** | `DynamicObstacle` obs/contract slots, moving-obstacle CBF QP + retreat fallback (`safety.py`), `DynamicObstacle*.msg`, training/eval sim (`envs.py`, `eval.py`), ROS `/obstacles/dynamic` subscriber hook (`node.py`), adapter ENU noise (`noise.py`/`adapter_node.py`), and Gazebo `--wind`/`--rain` SDF hooks (`scenario.py`/`swarm.launch.py`) are complete. | Live active-flight verification of `/obstacles/dynamic` and M1 disturbance hooks (Steps 4b & 4d, M1); M3 comms packet-drop/latency injector and scripted `/obstacles/dynamic` bird publisher node. |
| **M1** | **Step 12 — Simulation-only SITL validation (Active Flight)** | Grounded 10-drone headless SITL (`RTF = 0.999`, `0/10` freshness violations), `10/10` valid PX4 ULogs (`No Dropouts`), clean shutdown (`0` orphans, `0` stale ROS nodes), and host disk check (`27 GB` free) all **PASSED**. | **PENDING active-flight validation:** Step 2 (recorded 10-drone headless hover `Z`), Step 3 (recorded 10-drone headless classical transit `C+S`), Step 4 (live `/{drone}/goal`, `/obstacles/dynamic`, quorum resilience, M1 wind/noise/rain hooks), and Step 5 (graphical `headless:=false` rendering & diagnostics). |
| **M2** | **Step 11 — Stage 1 MAPPO training (`stage1_open`)** | Completed 250 iterations / 600,000 env steps on Colab T4 GPU; fixed Stage 1 evaluation mismatch (`b311940`); checkpoints persisted in Google Drive. | **DEFERRED / BLOCKED — Stage 1 acceptance criteria NOT met:** independent 20-ep eval achieved 70% `L+S` success (92% docking, 0 collisions) vs. ≥ 85% gate, and 10% `L` shield-off success (10% docking, 75% collision rate). Stage 2 has NOT started; M2 training is deferred until M1 simulation validation completes. |

### Pending

- **M1 — Active-Flight Simulation Validation (Immediate Next Priority, Steps 2–5):** Execute recorded 10-drone headless hover (`Z`), classical transit (`C+S`), runtime hooks/quorum pass, and graphical (`headless:=false`) validation under the `performance` CPU governor with full loopback isolation.
- **M2 — Stage 1 Convergence & Stage 2–4 Curriculum Training (DEFERRED / BLOCKED on Stage 1 Gate):** Resume after M1 simulation validation; pass Stage 1 (`L+S ≥ 85%` + strong shield-OFF `L`), then complete Stages 2–4 on Colab and export a gate-passing Contract v2 `swarm_policy_best.pt`.
- **M1/M2 — Learned-Policy SITL Validation (DEFERRED / BLOCKED on M2):** Formal `L+S` (`≥ 90%`) and `L` (`≥ 80%`) SITL runs remain blocked until M2 passes Stages 1–4.
- **M3 — Comms Fault Injector, Formation Allocator & Bird Publisher:** Implement `formation_allocator` (`/swarm/formation_pose`, `/swarm/set_formation`), packet-drop/latency comms filter, and scripted `/obstacles/dynamic` bird publisher.
- **M4 — Analytics & Final Ablation Matrix:** Complete rosbag2/CSV statistical pipeline, multi-seed plots, and final `L+S` / `L` / `C+S` / `C` / `Z` report.

---

## Session Handoff & Resume Checklist (2026-11-27 — Commit `144f0c73165a467339bd6451d336e0b56bfa78b9`)

### Verified Host Environment & Isolation Configuration

| Parameter | Verified Setting | Notes |
|---|---|---|
| **Git branch & verified code commit** | `origin/feature/m1-m2-integration` @ `144f0c73165a467339bd6451d336e0b56bfa78b9` | All 51 regression tests pass (`36` swarm/M1/MARL + `15` MCP); all 4 ROS 2 packages built cleanly. |
| **ROS 2 & DDS Isolation** | `ROS_DISTRO=humble`, `ROS_LOCALHOST_ONLY=1`, `ROS_DOMAIN_ID=42` | Must be exported in both MCP bridge and simulation terminals. |
| **Underlay & Workspace Overlay** | `/opt/ros/humble` → `/home/siddharthg/uav_ws/install` (`px4_msgs`) → `/home/siddharthg/SEM3_MAINEL_PLSWORK/install` | Source in this exact order before running ROS 2 or MCP. |
| **Gazebo & PX4 SITL** | `GZ_IP=127.0.0.1`, `GZ_PARTITION=swarm`, `PX4_DIR=/home/siddharthg/PX4-Autopilot` | Gazebo Harmonic `8.15.0`; `4002_gz_x500_swarm` airframe installed; 10 rootfs instances (`0..9`). |
| **Fast DDS Loopback Profile** | `src/swarm_bringup/config/fastdds_loopback.xml` | Loaded by `MicroXRCEAgent -r ...` + `FASTDDS_DEFAULT_PROFILES_FILE`; PX4 uses `PTCFG=2`, `AG_IP=2130706433`. |
| **Temporary Host Firewall Rule** | Blocks non-loopback ingress on UDP `8888` (`MicroXRCEAgent` allows `127.0.0.1` only) | **Re-verify after host reboot**; temporary rule may not persist across reboots. |
| **Host Hardware & CPU Governor** | AMD Ryzen 9 270 (8 physical cores / 16 logical threads); **`performance`** governor (`4813 MHz`) | Switched from `powersave` (`1171 MHz`, RTF `0.693`) to `performance` (`4813 MHz`, RTF `0.999`). **Re-verify after reboot/power-state change.** |
| **Read-Only MCP Bridge** | `http://127.0.0.1:8765/mcp` (`read_only: true`, `physical_drone_access: false`) | Launch via `bash tools/sim_diagnostics_mcp/run_bridge.sh` outside Snap. |
| **Host Disk Space** | **27 GB available** (`71%` used; `~9 GB` existing PX4 `.ulg` logs preserved) | Keep armed runs to `2–4 min` (`~0.5 GB` `.ulg` + rosbag2 per run); do not delete existing logs unless approved. |

### Exact Steps to Resume Tomorrow (M1 Active-Flight Validation; M2 Deferred)

1. **Verify branch, clean working tree, CPU governor, temporary UDP `8888` firewall, and `ros2 bag`:**
   ```bash
   cd /home/siddharthg/SEM3_MAINEL_PLSWORK
   git status -sb
   git log -n 2 --oneline
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
   sudo iptables -S INPUT 2>/dev/null | grep 8888 || sudo ufw status 2>/dev/null | grep 8888 || true
   source /opt/ros/humble/setup.bash && ros2 bag --help | head -n 5
   ```
   *(If `scaling_governor` reverted to `powersave` after reboot, switch back to `performance`; if the temporary UDP `8888` loopback firewall rule cleared after reboot, reinstall it before starting `MicroXRCEAgent`.)*
2. **Start the isolated read-only MCP bridge (Terminal 1, outside Snap):**
   ```bash
   cd /home/siddharthg/SEM3_MAINEL_PLSWORK
   source /opt/ros/humble/setup.bash
   source /home/siddharthg/uav_ws/install/setup.bash
   source /home/siddharthg/SEM3_MAINEL_PLSWORK/install/setup.bash
   export ROS_LOCALHOST_ONLY=1
   export ROS_DOMAIN_ID=42
   export GZ_IP=127.0.0.1
   export GZ_PARTITION=swarm
   bash tools/sim_diagnostics_mcp/run_bridge.sh --check
   bash tools/sim_diagnostics_mcp/run_bridge.sh
   ```
3. **Decide soft-landing handling before armed flight:**
   - **Option A (Recommended):** Implement a minimal `LANDING` phase in `src/px4_adapter/px4_adapter/adapter_node.py` (with regression test) so `/drone_{0..9}/land` keeps PX4 in `AUTO_LAND` instead of re-requesting `OFFBOARD` after `1.0 s`, then rebuild `px4_adapter`.
   - **Option B (Zero code change):** Keep `144f0c7` unchanged and use `Ctrl+C` (`SIGINT`) directly for teardown after Steps 2–3.
4. **Execute remaining M1 active-flight validation sequence (keep M2 deferred, preserve frozen Contract v2):**
   - **Step 2:** Recorded 10-drone headless hover (`Z`: `mode:=zero auto_arm:=true enable_shield:=true headless:=true control_rate:=5.0 telemetry_rate:=10.0`) with `ros2 bag record` + `launch.log` + MCP verification + clean shutdown.
   - **Step 3:** Recorded 10-drone headless classical transit (`C+S`: `mode:=classical enable_shield:=true auto_arm:=true headless:=true control_rate:=5.0 telemetry_rate:=10.0`) across packaged seed-7 arena + MCP verification + clean shutdown.
   - **Step 4:** Single-session live verification of M1 runtime hooks (`/{drone}/goal`, `/obstacles/dynamic`, §12.3 quorum resilience, and M1 wind/noise/rain disturbance hooks).
   - **Step 5:** Graphical 10-drone rendering & diagnostics check (`headless:=false` under `performance` governor; fallback `drones:=6` only if GUI drops `RTF < 0.90`) + clean shutdown.

---

## Next Steps for Improving Stage 1 Performance (Frozen Contract Preserved)

> **Contract constraint:** All Stage 1 improvements below are confined to `marl_training/` training, evaluation, and hyperparameter/reward tuning. **Do not modify** the frozen `swarm-coop-enu-v2` contract (`obs_dim=62`, `k_peer=4`, `k_static=3`, `k_dyn=2`, normalization scales, `±1.5 m/s` & `±0.6 rad/s` action decoding), `ShieldConfig` hard backstops (`0.9 m` / `0.0 m`), or `src/swarm_interfaces/` schemas.

### Root-Cause Diagnosis from Verified Stage 1 Metrics
1. **Shield dependency (75% collision rate with shield OFF vs. 0% with shield ON):** The CBF shield is absorbing peer-to-peer crossing conflicts during transit, so the actor has not yet learned proactive self-separation on its own.
2. **Straggler / docking-hold gap (92% per-drone docking vs. 70% full-swarm success with shield ON):** Most drones reach their slots, but 1–2 drones frequently miss the synchronized 10-drone hold window (`success_radius = 0.35 m`, `hold_speed = 0.12 m/s`, `hold_dwell = 10` steps) or get displaced late by nearby peers.
3. **Training-eval variance and late-stage drift (100% over 5 eps at iter 210 → 80% at iter 250 → 70% over 20 held-out eps):** 5-episode in-training evaluation is too noisy for reliable best-checkpoint selection, and policy updates past iter ~210 drifted slightly on held-out seeds.

### Action Plan Before Advancing to Stage 2
1. **Inspect Stage 1 diagnostics from Google Drive (`metrics.jsonl`) and per-seed 20-episode rollouts (`seeds 11..30`):**
   - Check `approx_kl`, `clip_frac`, `entropy`, `value_loss`, `eval_mean_shield_intervention`, `eval_min_pair_distance`, and `eval_final_slot_error` across iterations `1..250` to pinpoint whether instability came from aggressive PPO steps (`approx_kl > 0.03`), entropy collapse, or rising shield intervention.
   - Separate shield-OFF failures into early/mid-flight crossing collisions vs. late-stage slot-docking collisions, and identify why the final ~8% of drones fail to hold in shield-ON runs.
2. **Increase unshielded exposure during Stage 1 open-arena training:**
   - Because Stage 1 has zero static obstacles, increase `--shield-off-prob` progressively (e.g., `0.50 → 0.75` or schedule up to `1.0` after initial goal-reaching converges) so the actor consistently experiences the true consequences of peer crossings instead of leaning on the CBF shield.
   - Track `eval_shield_off_success_rate` and `eval_shield_off_collision_rate` alongside `eval_mean_shield_intervention` at every evaluation interval.
3. **Reduce evaluation noise and tighten checkpoint selection:**
   - Increase `--eval-episodes` from `5` to `10–20` on fixed held-out seeds during Stage 1 runs so `swarm_policy_best.pt` is not selected on a lucky 5-episode sample.
   - Select or gate Stage 1 promotion using **both** shield-ON (`L+S`) and shield-OFF (`L`) metrics plus low `mean_shield_intervention`.
4. **Stabilize MAPPO optimization & extend training budget if learning is still improving:**
   - Keep `approx_kl` controlled (e.g., lower `--lr` to `1.5e-4`–`2e-4`, reduce `--ppo-epochs` from `6` to `4` if `clip_frac > 0.25`, add optional KL early-stopping or value clipping in `mappo.py`, and test larger `--rollout-steps` such as `4800` for lower-variance 10-drone updates).
   - Verify whether `log_std` is hitting its `-2.0` clamp too early or remaining too noisy during fine slot docking.
5. **Refine Stage 1 training-only reward shaping (without changing the contract):**
   - Audit the trade-off between transit progress reward, TTC/soft-spacing penalties (`soft_pair_zone`, closing-speed penalty, per-culprit collision penalty), and final docking/hold incentives so drones smoothly decelerate into `0.35 m` / `0.12 m/s` and proactively yield during peer crossings rather than rushing straight through peers.
6. **Verify strict Stage 1 gate before starting Stage 2 (`stage2_static`):**
   - Run independent 20-episode evaluation (`python3 -m marl_training.eval --stage stage1_open --episodes 20 --seeds 11 30`) and confirm **`L+S` ≥ 85% success with 0 collisions and low intervention (< 0.05 m/s)** **and** strong shield-OFF performance (high success, near-zero collisions) before initializing Stage 2 via `--init-weights`.
   - Re-run all 25 unit/parity/env/MAPPO regression tests before launching any Colab training run.

---

## TL;DR for teammates

1. **The AI is not currently the controller.** In `mode:=learned`, the policy output passes through a hand-written `Supervisor` whose PD docking law is what actually lands each drone on its slot. Until that is removed from the learned path, the project is "a supervisor with an AI advisor", not "MARL".
2. **Training and deployment use two different 23-D contracts.** Same tensor shape, different meaning in every slot. Training on one and deploying the other cannot work.
3. **The fix is one shared contract, one shared observation builder, and a training env that calls the deployment code directly.**
4. **Go straight to 10 drones on `swarm-coop-enu-v2` from Day 1.** Do not build and train a 3-drone v1 policy only to throw it away and re-engineer everything for 10 drones, selectable targets, and birds. Build v2 once with `agents=10` and use a curriculum on obstacles/disturbances instead.
5. **Robustness claims must come from domain-randomised training plus a shield-off ablation** — not from the shield alone.
6. **All training runs on Google Colab** with Google Drive checkpointing, separate clean inference exports vs. full `--resume` training snapshots, and a mandatory parity smoke test before training starts.

---

## Table of contents

1. [What the project is](#1-what-the-project-is)
2. [Current state, verified](#2-current-state-verified)
3. [The three contract generations](#3-the-three-contract-generations)
4. [MAINEL.md vs the codebase](#4-mainelmd-vs-the-codebase)
5. [Guiding principle: AI is the project](#5-guiding-principle-ai-is-the-project)
6. [Target responsibility split](#6-target-responsibility-split)
7. [Phase 1 — One contract (v2 for 10 drones), one implementation](#7-phase-1--one-contract-v2-for-10-drones-one-implementation)
8. [Phase 2 — Rewrite the training environment (directly for 10 drones)](#8-phase-2--rewrite-the-training-environment-directly-for-10-drones)
9. [Phase 3 — Models and algorithm: Parameter-Shared MAPPO (at N = 10)](#9-phase-3--models-and-algorithm-parameter-shared-mappo-at-n--10)
10. [Phase 4 — The training script](#10-phase-4--the-training-script)
11. [Phase 5 — Deployment changes](#11-phase-5--deployment-changes)
12. [Phase 6 — Ten drones (ROS 2, Gazebo & Scenario)](#12-phase-6--ten-drones-ros-2-gazebo--scenario)
13. [Phase 7 — Selectable targets](#13-phase-7--selectable-targets)
14. [Phase 8 — Rain, birds, other obstacles](#14-phase-8--rain-birds-other-obstacles)
15. [Contract v2 specification](#15-contract-v2-specification)
16. [Ablation and demo matrix](#16-ablation-and-demo-matrix)
17. [Training on Google Colab (Detailed Step-by-Step Procedure — Pure-PyTorch MAPPO)](#17-training-on-google-colab-detailed-step-by-step-procedure--pure-pytorch-mappo)
18. [Work order and effort (Direct-to-10-Drones with MAPPO)](#18-work-order-and-effort-direct-to-10-drones-with-mappo)
19. [Finalised Technical Decisions](#19-finalised-technical-decisions)
20. [What not to do](#20-what-not-to-do)
21. [Glossary](#21-glossary)

---

## 1. What the project is

A multi-drone UAV testbed built on **ROS 2 + PX4 SITL + Gazebo Sim + PyTorch**, plus a separate offline **pure-PyTorch MAPPO (CTDE)** training stack. The existing codebase is hardcoded to 3 drones (with a legacy Ray/MADDPG prototype); this plan upgrades it **directly to 10 drones** on `swarm-coop-enu-v2` trained with parameter-shared **MAPPO**.

Each drone runs its own controller which:

1. reads a fixed observation of the world,
2. produces a 4-D action (velocity + yaw rate),
3. passes it through safety layers,
4. publishes a `Twist` to a PX4 adapter,
5. the adapter converts world **ENU** to PX4 **NED**, arms, enters offboard, takes off, and streams velocity setpoints.

### Packages

| Package | Build | Responsibility |
|---|---|---|
| `swarm_interfaces` | `ament_cmake` | `AgentTelemetry`, `NeighborList` messages |
| `swarm_control` | `ament_python` | Observation, contract, supervisor, shield, A*, ROS node |
| `px4_adapter` | `ament_python` | Only package importing `px4_msgs`; ENU↔NED bridge |
| `swarm_bringup` | `ament_python` | Launch, scenario/world generation, checkpoints, airframe |

### Runtime chain

```text
Gazebo ──► PX4 SITL ──► MicroXRCEAgent ──► px4_adapter
                                              │  /{drone}/telemetry
                                              ▼
                                        swarm_control
                                              │  /{drone}/cmd_vel
                                              ▼
                                        px4_adapter ──► PX4
```

### Commands

```bash
# build
source /opt/ros/<distro>/setup.bash
colcon build --symlink-install
source install/setup.bash

# tests (pure NumPy, no ROS required)
PYTHONPATH=src/swarm_control python3 -m unittest discover -s src/swarm_control/test -v

# run
ros2 launch swarm_bringup swarm.launch.py mode:=classical headless:=true

# regenerate a scenario + world
ros2 run swarm_bringup generate_world --seed 7 --obstacles 7 --out DIR
```

---

## 2. Current state, verified

### 2.1 What works

- The five unit tests in `src/swarm_control/test/test_control.py` pass (observation layout, supervisor transitions, three shield cases).
- A* routes match the paper's table when recomputed from the packaged scenario:

  | Drone | Waypoints | Path length | Min obstacle clearance |
  |---|---:|---:|---:|
  | 0 | 4 | 15.491 m | 1.326 m |
  | 1 | 4 | 15.431 m | 1.386 m |
  | 2 | 6 | 20.252 m | 1.320 m |

- The packaged scenario and world are consistent (obstacles, starts, goals, beacons).
- The ENU↔NED conversion in `px4_adapter` is explicit and correct in structure.
- The default checkpoint `src/swarm_bringup/checkpoints/swarm_macro_policy.pt` carries `swarm-coop-enu-v1` metadata and loads through `ActorPolicy`.

### 2.2 What is broken or missing

| Problem | Evidence |
|---|---|
| `marl_training` cannot import | all modules do `from common.contracts import ...`; no `common/` package exists. File is at `marl_training/contracts.py` |
| Training and deployment contracts disagree | see §3 |
| The supervisor, not the AI, does docking | `node.py::tick` always calls `self.supervisor.step(...)` |
| `maddpg_iter00460.pt` has no contract metadata | rejected unless `allow_legacy_checkpoint:=true` |
| No DAgger/BC training code | `swarm_macro_policy.pt` note says "DAgger path-projection expert, BC init", but no such code exists |
| No closed-loop PX4/Gazebo logs | paper explicitly does not claim any |
| Fault injection absent | no packet loss, no node kill, no jammer |
| Analytics absent | no rosbag2, no CSV/Parquet, no plots, no statistics |
| `NeighborList.msg` unused | generated but never published |
| Stale text | `node.py` claims `maddpg_iter00460.pt` is absent (it is present); `contract.py` references a missing `README.md` |
| Contract constant drift | `Contract.radius=0.30` vs shield `radius=0.45`; `Contract.pair_safe=1.0` vs shield `1.1` |

### 2.3 Current per-tick pipeline (`node.py::tick`)

```text
freshness gate (all 3 telemetry active + fresh within 0.5 s)
  └─ else publish zero and return
build_observation()                        # 23-D
command source:
  learned   → ActorPolicy → action_to_velocity
  classical → ClassicalMacro (A*)
  zero      → zeros
Supervisor.step()                          # APPROACH / DOCKING / HOLD / RECOVERY
stall recovery heuristic                   # overrides supervisor in APPROACH
safe_velocity()                            # CBF-style shield
publish Twist
```

**The problem is the third and fourth lines.** `Supervisor.step` computes a PD docking command, so in learned mode the AI only supplies the approach velocity; the final docking is hand-written.

---

## 3. The three contract generations

This is the root cause of most confusion. "23" appears three times with three different meanings.

| | **Gen 1 — `MAINEL.md`** | **Gen 2 — `marl_training/`** | **Gen 3 — `swarm_control/` (deployed)** |
|---|---|---|---|
| Dims | 23 / 4 | 23 / 4 | 23 / 4 |
| Target | one shared point `[40,40,5]` | one shared random point | **per-drone formation slot** |
| Obstacles | none | none | **slots 18–20** |
| Peer order | distance-sorted | distance-sorted | **fixed `agent_ids` order** |
| Valid flag | yes | yes | **no** |
| Comms radius | 15 m filter | 15 m filter | **no filter** |
| Action scale | ±3, ±3, ±1.5, ±π/2 | ±3, ±3, ±1.5, ±π/2 | **±0.8, ±0.8, ±0.8, ±0.6** |
| Goal scale | /50 | /50 | **/15** |
| Vel scale | /[3,3,1.5] | /[3,3,1.5] | **/0.8** |
| Yaw | indices 6–7 | indices 6–7 | **indices 21–22** |
| Altitude | index 8 | index 8 | **not present** |
| Agent names | `drone_1..3` | `drone_1..3` | **`drone_0..2`** |

### Current deployed observation (Gen 3, `observation.py::build_observation`)

| Index | Meaning | Scale |
|---|---|---|
| 0–2 | goal delta `goal − own.position` | `/15` |
| 3–5 | own velocity | `/0.8` |
| 6–8 | peer 1 relative position | `/15` |
| 9–11 | peer 1 relative velocity | `/1.6` |
| 12–14 | peer 2 relative position | `/15` |
| 15–17 | peer 2 relative velocity | `/1.6` |
| 18–19 | nearest obstacle delta in x/y | `/15` |
| 20 | nearest obstacle clearance | `/15`; `1.0` if none |
| 21–22 | `sin(yaw)`, `cos(yaw)` | — |

### Current deployed action decoding (`contract.py::action_to_velocity`)

```text
vx, vy, vz = action[:3] * 0.8     # m/s
yaw_rate   = action[3] * 0.6      # rad/s
```

### Current deployed safety parameters (`safety.py::ShieldConfig`)

| Parameter | Value |
|---|---:|
| `alpha` | 1.4 |
| `speed` | 0.8 m/s |
| `accel` | 1.2 m/s² |
| `pair_safe` | 1.1 m |
| `radius` (drone) | 0.45 m |
| `obstacle_margin` | 0.25 m |
| workspace x/y | ±9.3 m |
| workspace z | 0.35–4.0 m |
| iterations | 60 |

Note the drift: observation clearance uses `0.30`, shield enforces `0.45 + 0.25 = 0.70`. The shield is **0.40 m** more conservative than the observation suggests.

---

## 4. MAINEL.md vs the codebase

`MAINEL.md` is a saved Gemini conversation that produced a 4-person project blueprint. The codebase is roughly "Member 1 + Member 2" of that blueprint, and it has already evolved past it.

### 4.1 Matches

| MAINEL.md | Codebase |
|---|---|
| Decentralised UAV swarm on ROS 2 + PX4 + Gazebo + PyTorch | Same |
| Local ENU; PX4 NED; manual translation under XRCE-DDS | `adapter_node.py::_enu_state`, `_publish_setpoint` |
| `AgentTelemetry.msg` fields | identical, `src/swarm_interfaces/msg/AgentTelemetry.msg` |
| `common/contracts.py` constants | verbatim at `marl_training/contracts.py` |
| 23-D obs (9 self + 2×7 peer, sorted, 15 m, zero-padded) | exactly implemented in `marl_training/envs.py` |
| 4-D action → ±3 / ±1.5 / ±π/2 | `marl_training/contracts.py::scale_action` |
| PettingZoo `ParallelEnv`, 3 drones, `dt=0.1`, 500 steps | `marl_training/envs.py` |
| PyTorch actor → `.pt` → ROS loads it | `contract.py::ActorPolicy` |
| `/drone_x/cmd_vel` as `Twist` | `node.py:161` |
| Training without Gazebo | true; `marl_training` imports no ROS |
| Obstacle density as a variable | `generate_world --obstacles N` |

### 4.2 Partial matches

| MAINEL.md | Codebase | Gap |
|---|---|---|
| `/drone_x/telemetry_raw` | direct `/uav_i/fmu/out/vehicle_odometry` | no separate topic |
| `/drone_x/swarm_telemetry` | `/{drone}/telemetry` | renamed |
| `/drone_x/neighbor_states` | `NeighborList.msg` exists but unused | vestigial |
| `/drone_x/fault_command` | nothing | not implemented |
| `drone_1..3` | `drone_0..2` | off-by-one |
| `common/ + ros_ws/ + marl_training/ + analytics/ + docs/` | `src/ + marl_training/` | three dirs missing |
| 50 Hz rosbag2 logging | 20 Hz telemetry, 10 Hz control, 1 Hz status | no logging |

### 4.3 Missing entirely

**Member 3 — communications and fault injection:**

- 15 m comms-radius filter node
- packet loss / jamming / latency
- kill-drone-mid-flight trigger
- degraded neighbour list

**Member 4 — analytics:**

- rosbag2 scripts
- CSV/Parquet schema
- pandas / NumPy analysis
- matplotlib comparisons
- ANOVA / t-tests

**The headline experiment** ("fly with 40 % loss, kill drone_2, show the swarm adapts") is not implemented. The deployed gate does the **opposite**: `_fresh()` requires all three telemetry streams, so one dead drone freezes everyone. That is fail-closed, not fault-tolerant.

---

## 5. Guiding principle: AI is the project

> **In `mode:=learned`, the only things between the policy and PX4 are `action_to_velocity()` and `safe_velocity()`. No supervisor, no A\*, no stall heuristic.**

The shield remains as a **hard safety backstop**. Everything behavioural must be learned:

- navigation to the slot,
- formation keeping,
- peer spacing,
- obstacle avoidance,
- docking and holding,
- altitude hold,
- yaw control.

The shield is allowed to own **hard collision and hard obstacle penetration** only. If the shield owns the soft spacing, the AI is not doing coordination, and the shield-off ablation will fail.

### Tiered safety thresholds

| Threshold | Owner | Value |
|---|---|---|
| Comfortable peer spacing | **AI** | 1.1 m (`Contract.pair_safe`) |
| Hard peer collision | Shield | 0.9 m (`ShieldConfig.pair_safe`) |
| Comfortable obstacle margin | **AI** | 0.25 m (`Contract.obs_margin`) |
| Hard obstacle penetration | Shield | 0.0 m (`ShieldConfig.obstacle_margin`) |

With this split, `shield_intervention` becomes a direct measure of "is the AI doing the work".

---

## 6. Target responsibility split

| Behaviour | Owner after this work |
|---|---|
| Navigate to own slot | **AI** |
| Formation / peer spacing ≥ 1.1 m | **AI** |
| Obstacle avoidance ≥ 0.25 m | **AI** (shield = backstop) |
| Docking and hold at slot | **AI** |
| Altitude hold | **AI** |
| Yaw control | **AI** |
| Hard collision prevention < 0.9 m | Shield |
| Hard obstacle penetration | Shield |
| Workspace bounds, accel limit | Shield |
| Arm / offboard / takeoff / land | Adapter |
| ENU↔NED | Adapter |
| A* route | Classical baseline only |

---

## 7. Phase 1 — One contract (v2 for 10 drones), one implementation

> **Direct-to-10-drones rule:** Implement **`swarm-coop-enu-v2`** (`agents=10`, `k_peer=4`, `k_static=3`, `k_dyn=2`, `obs_dim=62` — full spec in §15) right here in Phase 1 instead of training an intermediate 3-drone v1 policy first. Dynamic-obstacle slots simply sit at `valid=0` during early curriculum stages until birds are turned on.

### 7.1 Make `swarm_control` the single source of truth

`swarm_control/observation.py`, `contract.py`, and `safety.py` are pure NumPy (`contract.py` imports torch lazily; `safety.py` adds only a lightweight QP solver in Phase 8). Training imports them directly.

Change in `marl_training/envs.py`, `models.py`, `mappo.py`, `train.py` (leaving `maddpg.py` untouched as legacy):

```python
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "swarm_control"))

from swarm_control.contract import Contract, DEFAULT_CONTRACT, action_to_velocity
from swarm_control.observation import DynamicObstacle, Obstacle, Telemetry, build_observation
from swarm_control.safety import ShieldConfig, safe_velocity
```

Then **delete `marl_training/contracts.py`** (or turn it into a re-export) so there is exactly one definition.

**Long-term option:** create `src/swarm_contract/` as a plain, non-ament package holding `Contract`, `build_observation`, `action_to_velocity`, `Obstacle`, `DynamicObstacle`, `Telemetry`, `safe_velocity`. Both `swarm_control` and `marl_training` import it. Do this after the fast path works.

### 7.2 Unify drifting constants (Contract v2 defaults for 10 drones)

| Constant | Today (v1) | v2 Target (10 drones) |
|---|---|---|
| `Contract.schema` | `swarm-coop-enu-v1` | **`swarm-coop-enu-v2`** |
| `Contract.agents` | 3 | **10** |
| `Contract.k_peer / k_static / k_dyn` | fixed 2 / 1 / 0 | **4 / 3 / 2** (`obs_dim = 62`) |
| `Contract.goal_scale / obstacle_scale` | 15.0 / 15.0 | **25.0 / 25.0** |
| `Contract.speed / accel` | 0.8 / 1.2 | **1.5 / 2.0** |
| `Contract.radius` | 0.30 | **0.45** |
| `Contract.pair_safe` | 1.0 | **1.1** (AI soft threshold) |
| `ShieldConfig.pair_safe` | 1.1 | **0.9** (hard backstop) |
| `Contract.obs_margin` | 0.25 | **0.25** (AI soft threshold) |
| `ShieldConfig.obstacle_margin` | 0.25 | **0.0** (hard backstop) |

### 7.3 Parity tests (do this before training anything)

Add to `src/swarm_control/test/`:

1. **Observation parity (62-D v2)** — random 10-drone state (with static + dynamic obstacles and missing peers) → training builder vs deployment builder → `np.allclose`, verifying distance-sorted nearest-4 peers and `valid` flags.
2. **Action parity** — training scaler vs `action_to_velocity` → identical physical velocities (`±1.5 m/s`, `±0.6 rad/s`).
3. **Golden vectors** — a stored 62-D observation / 4-D action pair that fails if the v2 layout changes.
4. **Checkpoint metadata & `state_dict` key parity** — save a deployment checkpoint via `save_deploy_pt()` (which exports bare `nn.Sequential` keys like `0.weight`, `2.weight`, `4.weight` from `trainer.actor.net.state_dict()`), load it through `ActorPolicy(..., weights_only=True)`, and assert schema (`swarm-coop-enu-v2`), `obs_dim=62`, `agents=10`, `hidden`, `peer_order`, strict bare-key loading (as well as stripping `net.` prefixes if present), and finite output matching `trainer.actor.deterministic(obs)`.

Training for days on a mismatched contract is exactly the failure this project is about.

---

## 8. Phase 2 — Rewrite the training environment (directly for 10 drones)

`marl_training/envs.py` must mirror the 10-drone v2 deployment from the first run, not approximate a 3-drone toy world.

### 8.1 What changes

| Aspect | Current | Must become |
|---|---|---|
| Agent count & names | 3 (`drone_1..3`) | **10 (`drone_0..drone_9`)** |
| Observation | 23-D Gen-1 layout | call `swarm_control.observation.build_observation` (**62-D v2**) |
| Action scale | `[3, 3, 1.5, π/2]` | call `swarm_control.contract.action_to_velocity` (**`[1.5, 1.5, 1.5, 0.6]`**) |
| Goal | one shared target | **10 per-drone formation slots** (ring / grid / V, spacing ≥ 1.5 m) + mid-episode target shift option |
| Arena & obstacles | ±50 m, no obstacles | **±15 m workspace**, 0–15 static cylinders + 0–4 moving dynamic obstacles ("birds") |
| Peer order | distance-sorted (2) | **nearest `k_peer=4` with valid flags** (missing/dropped peers → `valid=0`) |
| Dynamics | point mass + 0.5 lag | PX4-like lag (`lag_tau ∈ [0.1, 0.3]`) + accel limit + deployment `safe_velocity` shield |
| Disturbances | none | optional wind, odometry noise, 0–30% peer dropout, 0–200 ms latency (domain randomisation) |
| Termination | never | success (all 10 holding), collision, out-of-bounds |
| Reward | shared-target progress | per-slot progress + hold + spacing + static/dynamic clearance + smoothness + shield intervention |

### 8.2 Skeleton

```python
class DroneSwarmEnv(ParallelEnv):
    metadata = {"name": "drone_swarm_enu_v2"}

    def __init__(self, num_drones=10, max_steps=600, dt=0.1,
                 scenario_path=None, shield=True, randomize=True,
                 num_static_range=(0, 10), num_dyn_range=(0, 0),
                 goal_change_prob=0.0, domain_rand=False,
                 lag_tau=0.2, success_radius=0.35, hold_speed=0.12,
                 hold_dwell=10, contract=DEFAULT_CONTRACT, seed=None):
        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        ...

    def reset(self, seed=None, options=None):
        # 10-drone starts & goal formation (ring/grid/V) inside ±15 m arena; z = goal z (post-takeoff)
        # sample static cylinders (curriculum range) + dynamic crossing birds (if enabled)
        # sample domain-rand parameters (wind vector, odom noise, dropout rate, lag_tau)
        ...

    def step(self, actions):
        # advance dynamic obstacles (birds)
        for i, a in enumerate(self.agents):
            desired, yaw_rate = action_to_velocity(actions[a], self.contract)
            peers = [tel[j] for j in range(N) if j != i]
            if self.shield:
                res = safe_velocity(pos, vel, desired, peers,
                                    self.obstacles, self.dt, self.shield_cfg,
                                    dynamic_obstacles=self.dyn_obstacles)
                cmd, inter, feas = res.velocity, res.intervention, res.feasible
            else:
                cmd, inter, feas = desired, 0.0, True

            # PX4-like velocity tracking + optional wind disturbance
            alpha = min(1.0, self.dt / self.lag_tau)
            vel = vel + alpha * (cmd - vel) + wind_accel * self.dt
            vel = np.clip(vel, -self.contract.speed, self.contract.speed)
            pos = pos + vel * self.dt
            yaw = wrap(yaw + yaw_rate * self.dt)
        ...
```

### 8.3 Reward

```python
r  = 1.0 * (prev_dist - dist) / (speed * dt)   # normalised progress
if dist < success_radius:
    r += 0.3
    if ||vel|| < hold_speed:
        r += 0.3                                # holding still
r -= 2.0 * max(0.0, pair_safe - min_pair)       # soft spacing (< 1.1 m)
r -= 2.0 * max(0.0, obs_margin - min_clearance) # soft static & dynamic obstacle margin (< 0.25 m)
r -= 2.0 if out_of_bounds else 0.0
r -= 0.05 * ||action - prev_action||            # smoothness
r -= 0.5 * shield_intervention                  # don't fight the shield
```

### 8.4 Termination (Synchronized Across All 10 Drones)

- **collision** — any pair distance < `2 * radius` or obstacle/bird penetration < 0 → **terminates all 10 drones simultaneously**
- **out of bounds** — any drone outside `±15 m` workspace or altitude bounds → **terminates all 10 drones simultaneously**
- **success** — all 10 agents within `success_radius` and speed < `hold_speed` for `hold_dwell` steps → **terminates all 10 drones simultaneously**
- **truncation** — `max_steps` (600 steps = 60 s at 10 Hz for the larger ±15 m arena)

Synchronizing termination across all 10 drones keeps `RolloutBuffer` tensors a clean `(T, 10, 62)` shape with no ragged per-agent masks. `RolloutBuffer` bootstraps truncated episodes with `V(s_next)` and zeroes the bootstrap value only on true synchronized `terminated` events. Also log `fraction_docked` alongside full-swarm `success_rate` so Stage 1 progress is visible even when 8 or 9 drones are docking cleanly and 1 is still catching up.

### 8.5 Scenario handling (10-drone arena)

- **Train:** randomised 10-drone formations (ring/grid/V) over the ±15 m arena with curriculum-controlled obstacles/birds/disturbances.
- **Evaluate:** exact packaged 10-drone `config/scenario.json` (seed 7).
- **Generalisation:** held-out 10-drone seeds 11, 23, 42 + mid-flight target change test.

---

## 9. Phase 3 — Models and algorithm: Parameter-Shared MAPPO (at N = 10)

### 9.1 Why MAPPO directly (Option A)

Starting directly at **N = 10** drones with 62-D observations means MADDPG's centralized Q-critic would have to estimate `Q(s_all, a_all)` over a **660-D continuous joint state-action space**, which frequently diverges or plateaus beyond ~5–6 agents — burning days on free Colab just to confirm a stall.

**Parameter-shared MAPPO (Multi-Agent PPO)** solves this cleanly while remaining 100% **MARL / CTDE**:
- **Decentralised actor (`SwarmActorNet`)**: identical `62 → 256 → 256 → 4 (Tanh)` MLP shared by all 10 drones; exports the exact same `swarm_policy_best.pt` loaded by ROS 2 `ActorPolicy`. During training, a learnable 4-D `log_std` provides Gaussian exploration; at evaluation and deployment, the deterministic Tanh mean is executed directly.
- **Centralised value critic (`CentralizedValueNet`)**: estimates per-agent state value `V(s_all, o_i)` from the concatenated observations of all 10 drones plus own observation (`10 × 62 + 62 = 682` dims, **no** 40-D joint action input). Discarded at deployment.
- **On-policy clipped PPO + GAE**: uses a tiny on-policy rollout buffer (~10–15 MB RAM vs. ~400 MB–1 GB for a MADDPG replay buffer), needs **no Ray RLlib dependency** (pure PyTorch — faster on Colab's 2 vCPUs and immune to RLlib old-API-stack breakages), and resumes instantly after disconnects with zero replay-buffer warmup.

```python
class SwarmActorNet(nn.Module):
    def __init__(self, obs_dim=62, act_dim=4, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, hidden, nn.Tanh())  # exported to ROS 2 (bare Sequential keys)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        mean = self.net(obs)
        std = self.log_std.clamp(-2.0, 0.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

class CentralizedValueNet(nn.Module):
    def __init__(self, num_drones=10, obs_dim=62, hidden=(256, 256)):
        super().__init__()
        # Global all-drone obs (10*62=620) + own obs (62) -> 682 -> scalar V_i
        self.net = _mlp(obs_dim * num_drones + obs_dim, 1, hidden)
```

> **Gaussian sampling vs. clipping rule:** During rollout collection, sample `raw_action ~ Normal(mean, std)` (with `log_std` clamped to `[-2.0, 0.0]` so exploration neither explodes nor collapses prematurely), store the **unclipped `raw_action`** in `RolloutBuffer` so PPO `log_prob` ratios remain mathematically exact, and apply `np.clip(raw_action, -1.0, 1.0)` **only** when sending the command into `env.step()`. At evaluation and ROS 2 deployment, execute the deterministic `self.net(obs)` mean directly.

### 9.2 `marl_training/mappo.py` (replaces Ray/MADDPG for training)

- Pure PyTorch CTDE trainer (`MAPPOTrainer` + `RolloutBuffer`) — keeps `marl_training/maddpg.py` untouched only as a legacy reference (not used by `train.py`).
- Per iteration: collect `rollout_steps = 2400` env steps (`4` full 600-step episodes = `24,000` drone transitions), bootstrap truncated episodes with `V(s_next)` (zeroing only on true `terminated` collision/OOB/success), compute GAE (`gamma=0.99`, `gae_lambda=0.95`), normalize advantages, and run `ppo_epochs = 6` over minibatches (`minibatch_size = 1024`) with clip `eps = 0.2`, entropy coefficient `0.005`, value loss coefficient `0.5`, and grad-norm clip `0.5`.
- Exposes `state_dict()` / `load_state_dict()` for clean Colab `--resume` and `--init-weights`.

---

## 10. Phase 4 — The training script

`marl_training/train.py` drives pure-PyTorch MAPPO directly:

### 10.1 CLI

```text
--iters 300             --seed 7
--num-drones 10         --max-steps 600
--rollout-steps 2400    --ppo-epochs 6
--minibatch-size 1024   --lr 3e-4
--scenario PATH         --randomize/--no-randomize
--shield/--no-shield    --eval-every 10
--checkpoint-every 10   --out /content/drive/MyDrive/uav_swarm_marl/runs/exp01
--device cpu|cuda       --resume PATH
--init-weights PATH     --stage stage1_open|stage2_static|stage3_dynamic|stage4_domain_rand
```

### 10.2 Four-stage curriculum (always at N = 10)

Instead of scaling agent count (3 → 10), keep **N = 10 from step 1** and ramp environment difficulty:

| Stage | Static obstacles | Dynamic birds | Goal shifts | Domain rand (wind/noise/dropout) | Promotion gate | Current status |
|---|---:|---:|---:|---|---|---|
| **Stage 1 (`stage1_open`)** | 0 | 0 | 0% | Off | ≥ 85% eval success, 0 collisions | **In progress — gate NOT met** (250 iters / 600k steps on Colab T4; best train eval 100% @ iter 210 / 5 eps, final 80% @ iter 250; 20-ep eval: `L+S` 70% success / 92% docking / 0 collisions, `L` 10% success / 10% docking / 75% collision) |
| **Stage 2 (`stage2_static`)** | 3 → 10 | 0 | 0% | Off | ≥ 85% eval success, 0 collisions | **Not started** (blocked on Stage 1 gate) |
| **Stage 3 (`stage3_dynamic`)** | 5 → 12 | 1 → 4 | 10% mid-episode | Off | ≥ 80% eval success, 0 collisions | **Not started** |
| **Stage 4 (`stage4_domain_rand`)** | 3 → 15 | 0 → 4 | 10% mid-episode | Full (wind, noise, 0–30% dropout, latency, lag) | ≥ 85% L+S, ≥ 80% shield-off L | **Not started** |

### 10.3 Contents

1. Instantiates `DroneSwarmEnv` (`drone_swarm_enu_v2`, 10 drones) directly — no Ray cluster startup overhead.
2. Runs `MAPPOTrainer` (`hidden=(256, 256)`, `rollout_steps=2400`, `ppo_epochs=6`, `minibatch_size=1024`, `lr=3e-4`, linear LR annealing option).
3. Writes one JSON object per iteration to `<out>/metrics.jsonl` (`iter`, `env_steps_total`, `stage`, `episode_reward_mean`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_frac`).
4. Runs deterministic evaluation every `--eval-every` on the **unmodified 10-drone scenario** (plus a shield-off pass every `2 × eval_every`), computing:

   ```text
   success_rate, fraction_docked, time_to_hold, final_slot_error,
   min_pair_distance, min_obstacle_clearance, min_bird_clearance,
   mean_shield_intervention, infeasible_fraction,
   path_length, mean_action_jerk
   ```

5. Dual checkpoint export (below): **clean deployment weights** (`swarm_policy_best.pt`, selected by **eval success rate**, not reward) and **full resumable snapshots** (`train_state_latest.pt`).

### 10.4 Checkpoint format (keep deployment and resume files separate!)

> **Important `weights_only=True` rule:** `ActorPolicy` loads checkpoints with `torch.load(..., weights_only=True)`. Saving optimizer states or NumPy/Python RNG tuples inside the deployment file can break `weights_only=True` on PyTorch ≥ 2.6. Always save **two separate files**:

1. **Clean deployment file** (`swarm_policy_best.pt` / `swarm_policy_final.pt`) — exports `actor.net` (the deterministic `62 → 256 → 256 → 4 Tanh` MLP only) with plain dicts, lists, scalars, and tensors:

```python
def save_deploy_pt(trainer, contract, path, *, steps, seed, stage, note):
    meta = contract.as_dict()
    meta.update({
        "peer_order": "nearest_k_distance_then_agent_id",
        "action": f"world_ENU_velocity_{contract.speed}mps_yawrate_{contract.yaw_rate}rps",
        "algo": "mappo_ctde_shared",
        "training_steps": int(steps),
        "seed": int(seed),
        "stage": str(stage),
        "note": str(note),
    })
    payload = {
        "contract": meta,
        "actor": {k: v.detach().cpu() for k, v in trainer.actor.net.state_dict().items()},
        "obs_dim": int(contract.obs_dim),
        "action_dim": int(contract.act_dim),
        "num_drones": int(contract.agents),
    }
    torch.save(payload, path)
```

2. **Full resumable training snapshot** (`train_state_iterNNNNN.pt` & `train_state_latest.pt`, saved only on Drive for `--resume`):

```python
train_state = {
    "contract": meta,
    "actor_full": trainer.actor.state_dict(),     # includes net + log_std
    "critic": trainer.critic.state_dict(),
    "optimizer": trainer.optimizer.state_dict(),
    "iteration": i,
    "env_steps": steps,
    "stage": stage,
    "best_success": best_success,
    "rng_torch": torch.get_rng_state(),
    "rng_numpy": np.random.get_state(),
}
```

---

## 11. Phase 5 — Deployment changes

### 11.1 `src/swarm_control/swarm_control/node.py`

```python
# __init__
self.declare_parameter("use_supervisor", False)
self.use_supervisor = bool(self.get_parameter("use_supervisor").value)
if self.mode_name == "learned":
    self.use_supervisor = False          # AI owns behaviour

# tick
if self.policy is not None:
    action = self.policy(observation)
    desired, macro_yaw_rate = action_to_velocity(action, self.contract)
    stop_distance = 0.0
    policy_mode = "LEARNED"
elif self.classical is not None:
    macro_velocity = self.classical(own.position, own.velocity, peers, self.dt)
    desired, stop_distance = self.supervisor.step(
        own.position, own.velocity, self.slot, macro_velocity, self.dt)
    policy_mode = "CLASSICAL"
else:
    desired, macro_yaw_rate, stop_distance = np.zeros(3), 0.0, 0.0
    policy_mode = "ZERO"
```

- Gate stall recovery on `mode == "classical"`.
- Add a debug publisher so every policy tick is auditable:

  ```python
  from std_msgs.msg import Float32MultiArray
  self.debug_pub = self.create_publisher(
      Float32MultiArray, f"/{self.drone_id}/policy_debug", 10)
  # per tick: obs + action + desired + shielded + [intervention, feasible, goal_distance]
  ```

- Status JSON additions: `"policy"`, `"supervisor"`, `"shield_intervention"`, `"action"`.

### 11.2 `contract.py`

- Honour `speed`, `accel`, `yaw_rate`, `radius`, `pair_safe` from checkpoint metadata (currently only scales and `hidden`).
- Validate `peer_order` and `action` strings; reject unknown values.

### 11.3 `safety.py`

- Add `ShieldConfig.from_contract(contract)`.
- Hard backstop: `pair_safe=0.9`, `obstacle_margin=0.0`, `radius=0.45`.

### 11.4 `swarm.launch.py`

- Add `use_supervisor` (default `false`).
- Pass `speed/accel/yaw_rate` from checkpoint metadata, not hard-coded.
- Keep `takeoff_altitude` in sync with scenario goal z.

### 11.5 Do not change

- `px4_adapter/` — only consumes `Twist` and telemetry.
- `swarm_interfaces/` — messages stay the same.
- `planning.py` — keep A* as the classical baseline.
- `supervisor.py` — keep it for classical mode.

---

## 12. Phase 6 — Ten drones (ROS 2, Gazebo & Scenario)

*(Contract v2 is already implemented for `agents=10` in Phase 1; this section covers the ROS 2, Gazebo world, scenario generator, and launch changes needed to fly all 10 drones.)*

### 12.1 What breaks today

| File | Constraint |
|---|---|
| `contract.py` | `N_AGENTS = 3`, `Contract.agents = 3` |
| `observation.py` | `if len(peers) < 2: raise ValueError` |
| `node.py:78` | `if len(self.agent_ids) != 3: raise ValueError` |
| `node.py::_fresh` | requires all agents active and fresh |
| `swarm.launch.py:76` | hard-coded `["drone_0", "drone_1", "drone_2"]` |
| `scenario.py` | 3-slot triangle, 1.6 m start/goal discs |
| `worlds/swarm.sdf:53–54` | 1.6 m zones |
| `safety.py` | workspace ±9.3 m |
| `Contract.goal_scale` | 15.0 m |

### 12.2 Contract v2

See §15. Key points:

- `agents`, `k_peer`, `k_static`, `k_dyn` become parameters.
- `obs_dim` is computed, not a constant.
- Peer slots use **nearest-K with valid flags**, or all peers in fixed order.
- Recommendation: `k_peer = 4` with valid flags. The shield still sees all peers, so safety stays global.

### 12.3 Quorum telemetry gating

Replace "all agents" with:

```text
self is active and fresh
AND at least Q neighbours are active and fresh   (e.g. Q = 3)
```

Missing peers become `valid=0`. This is also required for the comms-loss demo.

### 12.4 Scenario and world must grow

Hard geometric constraints:

- 10 drones at 1.1 m spacing need a formation of ≥ ~3.5 m radius.
- The packaged **start and goal discs are 1.6 m radius** — 10 drones cannot fit.
- The obstacle field is ±6.7 m, workspace ±9.3 m — too tight.

`scenario.py::generate` and `write_world()` need:

- a formation generator for N slots (ring / grid / V) with spacing ≥ 1.5 m,
- start/goal zones of ≥ 4 m radius,
- an arena of ~±15 m,
- A* reachability for all 10,
- obstacle placement respecting the larger formation.

Because the arena grows, `Contract.goal_scale` must grow (25–30 m). Keeping 15 clips goals to ±1 and destroys resolution.

Also update `ShieldConfig.workspace`, and consider raising `speed` from 0.8 to ~1.5 m/s.

### 12.5 Launch

```python
agent_ids = [f"drone_{i}" for i in range(10)]
```

- PX4 start delay `6 + 2*i` seconds (last at ~24 s).
- Loop adapters and controllers over all 10.
- `MicroXRCEAgent` handles multiple namespaces unchanged.

### 12.6 Training at N=10

- `num_drones=10`, `k_peer=4` (`obs_dim=62`).
- Pure-PyTorch parameter-shared **MAPPO** with centralised value critic `V(s_all, o_i)` (`682-D` input).
- On-policy rollout buffer (`2400` env steps = `< 15 MB` RAM) — fits effortlessly on free Google Colab without Ray RLlib.

### 12.7 Compute reality

Rough load: 10 PX4 processes + Gazebo + 10 adapters + 10 controllers ≈ 14–16 cores. On a typical laptop expect **real-time factor < 1** and slow startup.

Mitigations:

- `headless:=true` always,
- lower `control_rate` to 5 Hz during demos,
- raise the physics step / reduce solver iterations,
- stagger PX4 startup,
- **fallback:** demonstrate 10 drones in the kinematic replay + training env, and 6 in full SITL. State clearly which is which.

### 12.8 Feasibility warning

With 10 drones, 1.1 m soft spacing, 0.9 m hard spacing, and 7 pillars, the shield's projection will hit infeasibility more often. Current fallback is "return zero" → frozen drones. Fix per §14.5.

---

## 13. Phase 7 — Selectable targets

### 13.1 Mechanisms

| Mechanism | How | Demo feel |
|---|---|---|
| Topic | `/{drone}/goal` (`PoseStamped`, transient-local) | good, scriptable |
| Service | `/swarm/set_formation` (centre pose, shape, spacing) | best for 10 drones |
| Parameter | `ros2 param set /drone_0 slot "[x,y,z]"` | clunky |

### 13.2 Recommended demo UX

Add a `formation_allocator` node in `swarm_bringup`:

```text
/swarm/formation_pose (PoseStamped) ──► allocator
                                          │ computes N slots (ring/grid/V)
                                          ▼
                            /drone_0/goal ... /drone_9/goal
```

In Foxglove or RViz, use the **2D Goal Pose** tool on `/swarm/formation_pose`. One click moves the whole formation.

Also allow `/{drone}/goal` to override a single slot.

### 13.3 The part that actually makes it work: training

Mechanically trivial; behaviourally it only works if the policy was trained on that goal distribution.

- Randomise the formation pose every episode over the full arena.
- Randomise the formation shape and spacing.
- **Mid-episode goal changes** with probability ~10 %.
- Keep `goal_scale` ≥ the largest possible delta.

Evaluation: exact scenario, held-out poses, mid-flight change.

Without this, "select any target" shows the shield dragging a confused policy — which undermines the claim.

### 13.4 Validation

The allocator must reject:

- goals outside the workspace,
- goals inside an obstacle (+ radius + margin),
- formations with slots closer than `pair_safe`,
- deltas beyond the trained `goal_scale`.

### 13.5 Classical caveat

`ClassicalMacro` plans A* once at construction. Runtime goal changes require replanning and a "no route" fallback. Learned mode has no such problem.

### 13.6 What "any target" means for 10 drones

Ten drones cannot occupy one point. A target is a **formation placement**: centre + heading + shape. A "converge on a target" demo means a ring around it at radius ≥ `pair_safe`.

---

## 14. Phase 8 — Rain, birds, other obstacles

### 14.1 Separate "visual" from "modelled"

| Phenomenon | Can you show it? | Can the controller model it? |
|---|---|---|
| Rain particles | Yes (cosmetic emitter) | No — the shield is kinematic |
| Wind gusts | Yes (PX4 wind params / Gazebo plugin) | Yes — disturbance the AI rejects |
| Sensor noise | Yes (inject in adapter telemetry) | Yes — AI sees noisy state |
| Comms loss | Yes (drop telemetry) | Yes — quorum + valid flags |
| Birds | Yes (scripted actors / ROS publisher) | Yes, **after a contract + shield extension** |
| Static clutter | Yes (more cylinders/boxes) | Yes, already supported |

**Rain is a disturbance/degradation test, not a physics model.** The shield does not "know" it is raining.

### 14.2 Rain: what to inject

- **Visual:** particle emitter in the SDF (check your Gazebo version).
- **Wind:** PX4 SITL wind/turbulence parameters (check your PX4 version's `SIM_WIND_*` set) or a Gazebo wind plugin.
- **Sensor noise:** Gaussian noise on position/velocity/yaw in `adapter_node._publish_telemetry`.
- **Comms:** drop X % of `/{drone}/telemetry`, or add latency.
- **Actuator degradation:** reduce effective command authority.

Why the AI can handle it: the observation contains **own velocity** (slots 3–5), so a policy trained with wind and noise can learn disturbance rejection.

### 14.3 Birds: dynamic obstacles

The current model is static:

```python
@dataclass
class Obstacle:
    x: float; y: float; radius: float; height: float = 4.0
```

Required changes:

1. **Contract:** add `k_dyn` blocks `(rel_pos 3, rel_vel 3, valid 1)`.
2. **Type:** add `vx, vy` or a `DynamicObstacle`.
3. **Shield:** static constraint

   ```text
   n · v_i ≥ −α (||p_i − p_o|| − r_o − r_i − margin)
   ```

   becomes, for a moving obstacle,

   ```text
   n · (v_i − v_o) ≥ −α (||p_i − p_o|| − r_o − r_i − margin)
   ```

   i.e. **add `n · v_o` to the right-hand side**, exactly like the peer term.
4. **Source:** scripted Gazebo actors, or a ROS node publishing `/obstacles/dynamic`.
5. **Prediction:** optional look-ahead `d_pred = ||p_i − (p_o + v_o τ)||`, τ ≈ 0.3–0.5 s.

### 14.4 Other obstacles

- **Boxes / walls:** bounding cylinder (conservative) or AABB half-planes.
- **No-fly zones:** extra half-planes.
- **Dense clutter:** more cylinders; test feasibility.

### 14.5 Shield upgrades needed anyway

1. **Moving obstacles** (§14.3).
2. **Feasibility at scale:** replace the 60-iteration projection with a **pure-NumPy prioritized-slack 3-D QP** (smooth quadratic slack `min ||v − v_des||² + Σ w_k max(0, b_k − n_k · v)²` over the acceleration box `[low, high]`, with a fast-path check skipping the solver when `v_des` already satisfies all constraints):

   ```text
   min  0.5 ||v − v_des||² + 0.5 Σ w_k s_k²
   s.t. n_k · v ≥ b_k − s_k,  s_k ≥ 0,  low ≤ v ≤ high
   ```

   Because `v ∈ R³`, a vectorized pure-NumPy projected Newton / damped Gauss-Newton solver converges in microseconds with **zero extra C dependencies** (`osqp`/`cvxpy` not required on ROS 2 or Colab).
3. **Infeasible fallback:** currently returns zero, which freezes the drone. Under a bird swarm that is dangerous. Better: **brake + retreat along the least-violated direction**, then resume. Wire it to `feasible=False`.
4. **Prioritise constraints:** hard collision > hard obstacle > workspace > soft spacing > accel.

### 14.6 The real key: domain randomisation

A policy trained on a clean, static, seed-7 world will not handle rain, birds, or 20 % dropout. Train with:

| Randomised during training | Range |
|---|---|
| Wind / disturbance | magnitude and direction |
| Odometry noise | σ on position and velocity |
| Telemetry dropout | 0–30 % of peer messages |
| Peer latency | 0–200 ms |
| Static obstacles | 3–15 |
| Dynamic obstacles | 0–4, random crossing paths |
| Formation pose / shape | full arena |
| Actuator lag | τ ∈ [0.1, 0.3] s |
| Goal changes | mid-episode |

This is what turns "the shield saved it" into "the AI learned to cope".

### 14.7 How to demonstrate

1. Baseline: clean scenario, 10 drones, formation to goal.
2. Rain: particles + wind gusts + sensor noise + 20 % dropout.
3. Birds: 3 scripted crossing trajectories.
4. Dense clutter: 15 cylinders.
5. Mid-flight target change.
6. Ablation: each with `enable_shield:=false`.

Metrics per run: success, time-to-hold, min pair distance, min obstacle/bird clearance, mean intervention, infeasible fraction, position error.

**Honest claims:**

- ✅ "The shield guarantees hard clearance against modelled static and dynamic obstacles."
- ✅ "The AI tolerates wind, noise, and dropout because it was trained with domain randomisation."
- ✅ "The AI alone (shield off) still completes the mission under these disturbances."
- ❌ "It flies in rain." — no aerodynamic rain model.
- ❌ "The shield handles birds." — only after the moving-obstacle term is added.

---

## 15. Contract v2 specification

| Field | Value |
|---|---|
| `schema` | `swarm-coop-enu-v2` |
| `agents` | **10** (starting directly at 10 drones from Phase 1) |
| `k_peer` | 4 (or 9 for all peers) |
| `k_static` | 3 |
| `k_dyn` | 2 |
| `goal_scale` | 25.0 |
| `vel_scale` | 1.5 |
| `peer_vel_scale` | 3.0 |
| `obstacle_scale` | 25.0 |
| `speed` | 1.5 |
| `accel` | 2.0 |
| `yaw_rate` | 0.6 |
| `radius` | 0.45 |
| `pair_safe` (soft, AI) | 1.1 |
| `hard_pair` (shield) | 0.9 |
| `obs_margin` (soft, AI) | 0.25 |
| `hard_margin` (shield) | 0.0 |
| `peer_order` | `nearest_k_distance_then_agent_id` |
| `action` | `world_ENU_velocity_1.5mps_yawrate_0.6rps` |

### Observation layout (exact field definitions & padding)

```text
obs = goal_delta(3) + own_vel(3)
    + k_peer × (rel_pos(3) + rel_vel(3) + valid(1))
    + k_static × (rel_pos_xy(2) + clearance(1) + valid(1))
    + k_dyn × (rel_pos(3) + rel_vel(3) + valid(1))
    + sin(yaw) + cos(yaw)

obs_dim = 6 + 7·k_peer + 4·k_static + 7·k_dyn + 2
```

Exact formulas & divisors (all clipped to `[-1, 1]`):

| Block | Field | Exact Formula | Padded (`valid=0`) |
|---|---|---|---|
| Self (`0:6`) | `goal_delta(3)`, `own_vel(3)` | `(goal - own_pos) / goal_scale`, `own_vel / vel_scale` | — |
| Peers (`k_peer × 7`) | `rel_pos(3)`, `rel_vel(3)`, `valid(1)` | `(peer_pos - own_pos) / goal_scale`, `(peer_vel - own_vel) / peer_vel_scale`, `1.0` | all `0.0` |
| Static (`k_static × 4`) | `rel_pos_xy(2)`, `clearance(1)`, `valid(1)` | **center-to-center** `([o.x, o.y] - own_xy) / obstacle_scale`, `(‖delta‖ - o.radius - contract.radius) / obstacle_scale`, `1.0` | all `0.0` (including `clearance = 0.0`, unlike v1's `1.0` sentinel) |
| Dynamic (`k_dyn × 7`) | `rel_pos(3)`, `rel_vel(3)`, `valid(1)` | `(dyn_pos - own_pos) / obstacle_scale` (`/ 25.0`), `(dyn_vel - own_vel) / peer_vel_scale` (`/ 3.0`), `1.0` | all `0.0` |
| Heading (`-2:`) | `sin(yaw)`, `cos(yaw)` | `sin(own.yaw)`, `cos(own.yaw)` | — |

Examples:

| `k_peer` | `k_static` | `k_dyn` | `obs_dim` |
|---:|---:|---:|---:|
| 4 | 3 | 2 | **62** |
| 9 | 3 | 2 | **97** |
| 4 | 3 | 0 | **48** |

### Migration notes

- `swarm_macro_policy.pt` (v1) will be rejected by a v2 loader. That is correct; archive it.
- `maddpg_iter00460.pt` stays legacy.
- Update `test_control.py` observation assertions.
- Update the paper's contract section.

---

## 16. Ablation and demo matrix

### 16.1 Core comparison (same seed, same scenario)

| Run | AI | Shield | Supervisor | A* | Purpose |
|---|---|---|---|---|---|
| **L+S** | ✓ | ✓ | ✗ | ✗ | deployed configuration |
| **L** | ✓ | ✗ | ✗ | ✗ | **the AI alone can do the mission** |
| C+S | ✗ | ✓ | ✓ | ✓ | classical baseline |
| C | ✗ | ✗ | ✓ | ✓ | raw classical |
| Z | ✗ | ✓ | ✗ | ✗ | hover baseline |

### 16.2 Success criteria

- **L+S:** ≥ 90 % success, 0 collisions, mean intervention < 0.05 m/s.
- **L:** ≥ 80 % success, 0 collisions.
- If L collides but L+S does not, the shield is doing the work and the AI is not.

### 16.3 Robustness matrix

| Scenario | Injection | Metric |
|---|---|---|
| Baseline | none | success, min clearance |
| Rain (visual) | particles + wind | success, position error |
| Rain (sensor) | noise + dropout | success, intervention |
| Birds | 3 crossing trajectories | min bird clearance, collisions |
| Dense clutter | 15 cylinders | success, infeasible fraction |
| Comms loss | 20–30 % dropout | success, quorum behaviour |
| Target change | mid-flight formation move | success, time to recover |

---

## 17. Training on Google Colab (Detailed Step-by-Step Procedure — Pure-PyTorch MAPPO)

All 10-drone MAPPO training runs happen on **Google Colab**. Local machines are used only for code editing, unit tests, inference verification, and PX4/Gazebo SITL.

### 17.1 Compute Reality & Resource Budget (N = 10, `obs_dim = 62`)

| Good for | Not good for |
|---|---|
| Free/cheap long training runs | Speeding up tiny single-env stepping on GPU |
| Fast startup (no Ray RLlib install needed!) | Unattended 24 h runs on free tier |
| Reproducible versioned notebooks | Real-time Gazebo/PX4 simulation |
| Automatic checkpoint persistence to Google Drive | — |

**Important:** Because we use **pure-PyTorch on-policy MAPPO**, there is **no Ray RLlib overhead and no massive replay buffer**:
- Actor (`62→256→256→4`, ~83k params) + Critic (`682→256→256→1`, ~240k params) take **< 5 MB** of memory.
- One on-policy rollout buffer of `2400` env steps (`10` drones × `62-D` obs) takes **< 15 MB of RAM** — completely eliminating OOM risk on free Colab's 12.7 GB RAM.
- Both **CPU-only** and **T4 GPU** run cleanly; T4 accelerates the `6` PPO epochs over `24,000`-transition minibatches.

### 17.2 Google Drive Folder Layout (Persistent Storage)

Never save checkpoints only to `/content` — `/content` is wiped on every disconnect. Use this fixed layout in Google Drive:

```text
/content/drive/MyDrive/uav_swarm_marl/
├── runs/
│   ├── exp01_stage1_open/
│   │   ├── metrics.jsonl             # 1 JSON line per iteration (train + eval)
│   │   ├── swarm_policy_best.pt      # Clean weights_only=True deployment file
│   │   ├── swarm_policy_final.pt     # Clean weights_only=True deployment file
│   │   ├── train_state_latest.pt     # Full resumable snapshot (actor+critic+optim+RNG)
│   │   └── train_state_iter00100.pt  # Rolling last-3 snapshots
│   ├── exp02_stage2_static/
│   ├── exp03_stage3_dynamic/
│   └── exp04_stage4_domain_rand/
└── exports/
    └── swarm_policy_best.pt          # Final verified policy ready to download
```

### 17.3 End-to-End Colab Workflow

```text
GitHub repo ──clone──► Colab runtime (/content/uav_EL)
                          │
                          ├─ Cell 1: Mount Google Drive + verify CPU/RAM/GPU
                          ├─ Cell 2: Clone/pull repo + checkout target branch
                          ├─ Cell 3: Install lightweight requirements-colab.txt (no Ray!)
                          ├─ Cell 4: MANDATORY 10-drone 62-D parity & smoke test
                          ├─ Cell 5: Run MAPPO curriculum stage (writes live to Drive)
                          ├─ Cell 6: Instant resume after disconnect (--resume train_state_latest.pt)
                          ├─ Cell 7: Plot learning curves & PPO diagnostics from metrics.jsonl
                          ├─ Cell 8: Run shield-ON vs shield-OFF ablation inside Colab
                          └─ Cell 9: Copy swarm_policy_best.pt to Drive/exports & download
                                       │
                                       ▼
                        src/swarm_bringup/checkpoints/swarm_policy_best.pt
                                       │
                                       ▼
                     Local ActorPolicy verification ──► ros2 launch ... mode:=learned
```

The GitHub repo is the **only** source of truth for code. The committed notebook (`marl_training/colab_train.ipynb`) is a thin launcher — never edit `.py` files only inside Colab without pushing them back to Git.

### 17.4 Step-by-Step Cell Guide (`marl_training/colab_train.ipynb`)

#### Cell 1 — Mount Google Drive & Inspect Runtime
```python
from google.colab import drive
drive.mount("/content/drive")

import os, psutil, torch
RUN_ROOT = "/content/drive/MyDrive/uav_swarm_marl/runs"
os.makedirs(RUN_ROOT, exist_ok=True)
print("CPUs:", os.cpu_count(),
      "| RAM GB:", round(psutil.virtual_memory().total / 1e9, 1),
      "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")
```

#### Cell 2 — Clone or Pull the Repository
```bash
REPO_URL="https://github.com/<ORG>/<REPO>.git"
BRANCH="main"   # or your feature branch

if [ ! -d /content/uav_EL ]; then
  git clone --branch "$BRANCH" "$REPO_URL" /content/uav_EL
else
  cd /content/uav_EL && git fetch origin && git checkout "$BRANCH" && git pull
fi
cd /content/uav_EL && git log -n 3 --oneline
```

#### Cell 3 — Install Lightweight Dependencies & Verify Versions
```bash
cd /content/uav_EL
pip install -q -r requirements-colab.txt
python3 -c "import torch, gymnasium, pettingzoo, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'pettingzoo', pettingzoo.__version__)"
```
*(No Ray RLlib installation required — installs in seconds using Colab's native PyTorch.)*

#### Cell 4 — Mandatory Parity & 10-Drone Smoke Test (DO NOT SKIP)
```bash
cd /content/uav_EL
# 1. Run full unit + contract-v2 parity test suite
PYTHONPATH=src/swarm_control python3 -m unittest discover -s src/swarm_control/test -v

# 2. Smoke-test 10-drone env + MAPPO actor/critic for 20 steps
python3 - <<'PY'
import numpy as np, torch
from marl_training.envs import DroneSwarmEnv
from marl_training.models import SwarmActorNet, CentralizedValueNet

env = DroneSwarmEnv(num_drones=10, max_steps=600, shield=True)
obs, _ = env.reset(seed=7)
assert len(obs) == 10, f"expected 10 drones, got {len(obs)}"
sample = next(iter(obs.values()))
assert sample.shape == (62,), f"expected 62-D v2 obs, got {sample.shape}"

actor = SwarmActorNet(obs_dim=62, act_dim=4)
critic = CentralizedValueNet(num_drones=10, obs_dim=62)
obs_t = torch.as_tensor(np.stack([obs[a] for a in env.agents]), dtype=torch.float32)
act_mean = actor.deterministic(obs_t)
assert act_mean.shape == (10, 4)

for _ in range(20):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
    if not env.agents:
        break
print("SMOKE TEST PASSED: 10 drones, obs_dim=62, MAPPO actor/critic OK, finite rewards =",
      all(np.isfinite(list(rew.values()))))
PY
```
**Never start training if Cell 4 fails.** This single gate prevents wasting hours on a broken contract.

#### Cell 5 — Launch 4-Stage MAPPO Curriculum Training (Always at N = 10)

Each iteration collects `2400` env steps (`24,000` agent transitions across 10 drones). Run stages sequentially, seeding each stage from the previous stage's weights via `--init-weights`:

```bash
cd /content/uav_EL

# --- STAGE 1: 10 drones, open arena (0 obstacles, 0 birds) ---
# ~150 iters = 360k env steps (3.6M agent steps). Gate: >= 85% eval success, 0 collisions.
python3 -m marl_training.train \
  --num-drones 10 \
  --max-steps 600 \
  --stage stage1_open \
  --iters 150 \
  --rollout-steps 2400 \
  --ppo-epochs 6 \
  --minibatch-size 1024 \
  --lr 3e-4 \
  --eval-every 10 \
  --checkpoint-every 10 \
  --seed 7 \
  --out /content/drive/MyDrive/uav_swarm_marl/runs/exp01_stage1_open
```

Once Stage 1 clears its gate (check Cell 7 plots), advance through Stages 2, 3, and 4 using `--init-weights` (loads actor/critic weights with a fresh optimizer & schedule for the harder stage):

```bash
# --- STAGE 2: 10 drones + 3..10 static cylindrical obstacles (~600k env steps) ---
python3 -m marl_training.train \
  --num-drones 10 --max-steps 600 --stage stage2_static --iters 250 \
  --rollout-steps 2400 --eval-every 10 --checkpoint-every 10 --seed 7 \
  --init-weights /content/drive/MyDrive/uav_swarm_marl/runs/exp01_stage1_open/train_state_latest.pt \
  --out /content/drive/MyDrive/uav_swarm_marl/runs/exp02_stage2_static

# --- STAGE 3: 10 drones + 5..12 static obstacles + 1..4 moving birds + 10% mid-episode goal shifts (~720k env steps) ---
python3 -m marl_training.train \
  --num-drones 10 --max-steps 600 --stage stage3_dynamic --iters 300 \
  --rollout-steps 2400 --eval-every 10 --checkpoint-every 10 --seed 7 \
  --init-weights /content/drive/MyDrive/uav_swarm_marl/runs/exp02_stage2_static/train_state_latest.pt \
  --out /content/drive/MyDrive/uav_swarm_marl/runs/exp03_stage3_dynamic

# --- STAGE 4: Full domain randomisation (wind, odom noise, 0..30% dropout, latency, lag, 3..15 clutter, 0..4 birds) ---
python3 -m marl_training.train \
  --num-drones 10 --max-steps 600 --stage stage4_domain_rand --iters 350 \
  --rollout-steps 2400 --eval-every 10 --checkpoint-every 10 --seed 7 \
  --init-weights /content/drive/MyDrive/uav_swarm_marl/runs/exp03_stage3_dynamic/train_state_latest.pt \
  --out /content/drive/MyDrive/uav_swarm_marl/runs/exp04_stage4_domain_rand
```

#### Cell 6 — Recovering & Resuming After a Colab Disconnect

When Colab disconnects mid-stage:
1. Re-run **Cell 1** (mount Drive), **Cell 2** (clone repo), **Cell 3** (install deps), and **Cell 4** (parity check).
2. Run the **same stage** with `--resume` pointing to `train_state_latest.pt` in Drive:

```bash
cd /content/uav_EL
python3 -m marl_training.train \
  --num-drones 10 --max-steps 600 --stage stage2_static --iters 250 \
  --rollout-steps 2400 --eval-every 10 --checkpoint-every 10 --seed 7 \
  --resume /content/drive/MyDrive/uav_swarm_marl/runs/exp02_stage2_static/train_state_latest.pt \
  --out /content/drive/MyDrive/uav_swarm_marl/runs/exp02_stage2_static
```

`--resume` restores actor (`net + log_std`), critic, Adam optimizer state, iteration/step counters, best success rate, and RNG states. Because MAPPO is on-policy, training resumes **immediately** with zero replay-buffer refill delay. At `--checkpoint-every 10`, a disconnect loses at most ~5–10 minutes of work.

#### Cell 7 — Plot Learning Curves & Diagnose MAPPO Training Health
```python
import json, pathlib, pandas as pd, matplotlib.pyplot as plt

RUN_DIR = pathlib.Path("/content/drive/MyDrive/uav_swarm_marl/runs/exp01_stage1_open")
rows = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text().splitlines() if line.strip()]
df = pd.DataFrame(rows)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df.plot(x="env_steps_total", y="episode_reward_mean", ax=axes[0, 0], title="Train Episode Reward")
if "eval_success_rate" in df:
    df.dropna(subset=["eval_success_rate"]).plot(x="env_steps_total", y="eval_success_rate", ax=axes[0, 1], title="Eval Success Rate (L+S)")
if "eval_mean_shield_intervention" in df:
    df.dropna(subset=["eval_mean_shield_intervention"]).plot(x="env_steps_total", y="eval_mean_shield_intervention", ax=axes[1, 0], title="Mean Shield Intervention (m/s)")
df.plot(x="env_steps_total", y=["value_loss", "approx_kl"], ax=axes[1, 1], title="Critic Value Loss & Approx KL")
plt.tight_layout(); plt.show()
```

**Healthy vs. Unhealthy MAPPO Signals:**
* ✅ **Healthy:** `eval_success_rate` climbs steadily; `approx_kl` stays in `0.005–0.03`; `clip_frac` stays `< 0.25`; `mean_shield_intervention` drops toward `< 0.05 m/s`; `entropy` decreases gradually without collapsing early.
* ⚠️ **Shield carrying the policy:** `L+S` success is high, but `mean_shield_intervention > 0.15 m/s` or shield-off `L` collides → increase shield-intervention penalty (`-0.5 → -1.0`) or train with `shield=False` on 25% of episodes.
* ⚠️ **Policy updates too aggressive / noisy:** `approx_kl > 0.05` or `clip_frac > 0.35` → lower `--lr` (`3e-4 → 1.5e-4`) or reduce `--ppo-epochs` (`6 → 4`).

#### Cell 8 — Run Pre-Download Shield-ON vs. Shield-OFF Ablation in Colab
Before downloading to the ROS 2 laptop, evaluate the best checkpoint across 20 held-out 10-drone seeds (`11..30`) both **with** (`shield=True`) and **without** (`shield=False`) the CBF shield. For Stage 1 checkpoints (trained with 0 static obstacles), use `--stage stage1_open` (or `--modes open`) to match the Stage 1 environment (fixed in commit `b311940`; evaluating a `stage1_open` checkpoint on obstacle modes emits a mismatch warning). Training checkpoints remain stored in Google Drive, not GitHub:
```bash
cd /content/uav_EL
# Stage 1 evaluation (0 static obstacles, matches stage1_open training):
python3 -m marl_training.eval \
  --checkpoint /content/drive/MyDrive/uav_swarm_marl/runs/exp01_stage1_open/swarm_policy_best.pt \
  --num-drones 10 --episodes 20 --seeds 11 30 --stage stage1_open

# Full obstacle/robustness ablation (after Stage 2–4 training):
python3 -m marl_training.eval \
  --checkpoint /content/drive/MyDrive/uav_swarm_marl/runs/exp04_stage4_domain_rand/swarm_policy_best.pt \
  --num-drones 10 --episodes 20 --seeds 11 30 --modes clean rain birds clutter target_change
```
Only promote a checkpoint to SITL once **L+S ≥ 90% success (0 collisions, intervention < 0.05 m/s)** and **L (shield OFF) ≥ 80% success (0 collisions)**.

#### Cell 9 — Export, Download & Verify Locally Before Flying
1. Copy the passing clean deployment file (`swarm_policy_best.pt`, ~350 KB) to `/content/drive/MyDrive/uav_swarm_marl/exports/` and download it to `src/swarm_bringup/checkpoints/swarm_policy_best.pt`.
2. Add `checkpoints/swarm_policy_best.pt` to `src/swarm_bringup/setup.py` `data_files`. Commit only `best` and `final` (never commit `train_state_*.pt` snapshots).
3. Verify locally on the ROS 2 host machine with `weights_only=True` before launching PX4/Gazebo:

```bash
PYTHONPATH=src/swarm_control python3 - <<'PY'
import numpy as np
from swarm_control.contract import ActorPolicy

p = ActorPolicy("src/swarm_bringup/checkpoints/swarm_policy_best.pt", actor_index=0)
print("Contract metadata:", p.metadata)
assert p.contract.schema == "swarm-coop-enu-v2"
assert p.contract.obs_dim == 62
assert p.contract.agents == 10
out = p(np.zeros(62, dtype="float32"))
assert out.shape == (4,) and np.all(np.isfinite(out))
print("VERIFIED LOCAL LOAD OK; zero-obs action =", out)
PY
```

### 17.5 Settings That Matter on Colab (MAPPO)

| Setting | Value on Free Colab | Why |
|---|---|---|
| `num_drones` | `10` | Directly training the 10-drone swarm |
| `max_steps` | `600` | 60 s episodes at 10 Hz across the ±15 m arena |
| `rollout_steps` | `2400` | 4 full episodes (`24,000` drone transitions) per PPO update; < 15 MB RAM |
| `ppo_epochs` | `6` | Standard continuous-control PPO epochs |
| `minibatch_size` | `1024` | Fast vectorized updates on CPU or T4 GPU |
| `gamma / gae_lambda` | `0.99 / 0.95` | Long-horizon credit assignment across 60 s formation transit & hold |
| `lr` | `3e-4` | Standard Adam LR for MAPPO (lower to `1.5e-4` in Stage 4 fine-tuning) |
| `checkpoint_every` | `10` iters | Limits lost work on sudden disconnect to ~5–10 min |
| `keep_last_snapshots` | `3` (+ `best` + `final`) | Keeps Google Drive tidy |
| `device` | `cuda` if T4 else `cpu` | T4 speeds up PPO minibatch epochs; CPU also works cleanly |

### 17.6 Repository Files Required for Colab

| File | Purpose |
|---|---|
| `requirements-colab.txt` | Lightweight pinned `torch`, `gymnasium`, `pettingzoo`, `numpy`, `pandas`, `matplotlib` (no Ray!) |
| `marl_training/colab_train.ipynb` | The thin 9-cell notebook above |
| `marl_training/mappo.py` | Pure-PyTorch parameter-shared MAPPO trainer + GAE rollout buffer |
| `marl_training/train.py` | `--stage`, `--out`, `--resume`, `--init-weights`, JSONL metric log, dual checkpoint export |
| `marl_training/eval.py` | Headless kinematic evaluator for shield-ON / shield-OFF ablations across all test modes |
| `marl_training/envs.py` | 10-drone v2 env with no rendering and no ROS imports |
| `src/swarm_control` | Imported directly via `sys.path` (or `pip install -e src/swarm_control` — pure setuptools, no ROS needed) |

### 17.7 Time Budget & Colab Troubleshooting

| Milestone (at N = 10) | Rough Wall-Clock on Free Colab | Notes |
|---|---|---|
| Stage 1 (open, 150 iters / 360k env steps) | 1 – 2 h | Fast convergence check without Ray overhead |
| Stage 2 (static clutter, 250 iters / 600k steps) | 2 – 4 h | Spread across 1–2 sessions with `--resume` |
| Stage 3 (birds + goal shifts, 300 iters / 720k steps) | 3 – 5 h | Check shield-off clearance before Stage 4 |
| Stage 4 (full domain rand, 350 iters / 840k steps) | 3 – 6 h | Final robustness tuning |
| Full campaign + hyperparameter tuning | 3 – 7 days across sessions | Much faster & more reliable than MADDPG at N=10 |

**Common Colab Gotchas:**
- **Disconnects:** `/content` is wiped on disconnect; Drive is not. Re-run Cells 1–4 and resume via Cell 6 (`--resume train_state_latest.pt`).
- **`weights_only=True` error when loading locally:** You tried to load `train_state_latest.pt` instead of `swarm_policy_best.pt`. Only `swarm_policy_*.pt` goes into `src/swarm_bringup/checkpoints/`.
- **Slow step rate with QP shield inside env:** The pure-NumPy fast-path check (`A @ v_des >= b`) skips the 3-D slack QP solver on > 80% of steps, keeping stepping fast on CPU.

---

## 18. Work order and effort (Direct-to-10-Drones with MAPPO)

| Step | Work | Effort | Status |
|---|---|---|---|
| 0 | Contract v2 parity test harness (62-D obs, action scale, golden vectors, metadata) | 0.5 day | **Completed** (`test_contract_v2_parity.py`, 10 tests) |
| 1 | Implement Contract v2 (`agents=10`, `k_peer=4`, `k_static=3`, `k_dyn=2`, `obs_dim=62`) in `swarm_control`, unify imports, re-export `marl_training/contracts.py` | 1.5 days | **Completed** (`contract.py`, `observation.py`, `marl_training/contracts.py`, `swarm_interfaces`) |
| 2 | 10-drone scenario & Gazebo world generator (`scenario.py`: ±15 m arena, ring/grid/V formations, ≥ 4 m discs, regenerated `scenario.json` + `swarm.sdf`) | 2 days | **Completed** (`planning.py`, `scenario.py`, `scenario.json`, `swarm.sdf`) |
| 3 | Upgrade `safety.py`: moving obstacle term (`n · v_o`), pure-NumPy prioritized-slack 3-D QP (with fast-path), prioritized constraints, retreat-on-infeasible fallback, `ShieldConfig.from_contract` | 2 days | **Completed** (`safety.py`, verified by shield & QP tests) |
| 4 | Rewrite `marl_training/envs.py` directly for 10 drones (v2 obs, deployment shield, static + dynamic obstacles, goal randomization + mid-episode shifts, domain randomization flags) | 2.5 days | **Completed** (`envs.py`, `test_env_v2.py`) |
| 5 | Implement pure-PyTorch parameter-shared MAPPO (`models.py` + `mappo.py`: `SwarmActorNet` + `CentralizedValueNet` + GAE rollout buffer + resumable state) | 1.5 days | **Completed** (`models.py`, `mappo.py`) |
| 6 | Rewrite `train.py` + add `eval.py` (4-stage curriculum, separate clean deploy vs. resumable snapshot export, `--resume`, `--init-weights`, JSONL logging, shield-off eval) | 1.5 days | **Completed** (`train.py`, `eval.py`, incl. Stage 1 eval mismatch fix `b311940`) |
| 7 | Colab package: lightweight `requirements-colab.txt` + `marl_training/colab_train.ipynb` (verified through Cell 4 parity smoke test + short Stage 1 run) | 0.5 day | **Completed** (`requirements-colab.txt`, `colab_train.ipynb`, all 25 regression tests pass) |
| 8 | ROS 2 deployment updates: `node.py` (supervisor bypass in `learned`, quorum telemetry gating, `/{drone}/policy_debug`), `contract.py` (v2 metadata validation), 10-drone `swarm.launch.py`, Fast DDS loopback isolation | 2 days | **Completed** (`node.py`, `controller_core.py`, `swarm.launch.py`, `launch_plan.py`, `fastdds_loopback.xml`, `rate.py`; 51/51 tests pass, grounded 10-drone `RTF = 0.999`) |
| 9 | Selectable targets: `formation_allocator` node + `/swarm/formation_pose` + `/swarm/set_formation` + `/{drone}/goal` | 1 day | **Partially completed** (`SetFormation.srv` frozen + `/{drone}/goal` hook & A* replanning done in M1; live flight test A4a & M3 `formation_allocator` pending) |
| 10 | Environmental & fault injectors: packet-drop/latency comms filter, adapter telemetry noise, PX4 wind params, visual rain SDF, `/obstacles/dynamic` bird publisher | 2 days | **Partially completed** (`DynamicObstacle*.msg`, training sim, `/obstacles/dynamic` subscriber, adapter noise `noise.py`, and `--wind`/`--rain` SDF hooks done in M1; live flight tests A4b/A4d & M3 comms/bird nodes pending) |
| 11 | Train & tune 10-drone MAPPO policy on Google Colab (Stages 1 → 4) | 1 – 2 weeks | **Blocked — Stage 1 gate NOT met** (Stage 1 ran 250 iters / 600k steps on T4; Stage 2 NOT started; checkpoints on Google Drive) |
| 12 | Full SITL integration, ablation matrix (`L+S`, `L` shield-off, `C+S`, `C`, `Z`), analytics plots & demo | 3–5 days | **In progress — grounded 10-drone gate PASSED (`RTF = 0.999`); armed Track A (A1–A5) pending; Track B (`L+S`/`L`) blocked on M2** |

### Suggested team split

| Member | Focus |
|---|---|
| M1 (sim) | Contract v2 in `swarm_control`, 10-drone world/scenario/launch, QP shield, `node.py` deployment bypass, SITL |
| M2 (AI) | `envs.py`, `models.py`, `mappo.py`, `train.py`, `eval.py`, **Colab notebook**, 4-stage MAPPO training & checkpoint export |
| M3 (comms) | Quorum gating, packet-loss/latency fault injector, formation allocator, `/obstacles/dynamic` bird publisher |
| M4 (analytics) | `/{drone}/policy_debug` + status/rosbag2 logging, metrics, ablation runs, statistical plots & report |

M2 owns the Colab notebook and is the only member who needs to run training. M1, M3, and M4 work against Contract v2 and the debug topics and never need Colab access.

---

## 19. Finalised Technical Decisions

All open design choices are locked to the most feasible, lowest-risk options so implementation can proceed without rework:

| Decision | Chosen Option | Rationale |
|---|---|---|
| Starting swarm size & contract | **10 drones directly on `swarm-coop-enu-v2` (62-D)** | Avoids building and throwing away an intermediate 3-drone v1 pipeline; curriculum ramps obstacles/disturbances at `N = 10`. |
| Shared contract location | **`src/swarm_control`** (imported directly by `marl_training`) | Already pure NumPy (lazy torch import) with `setup.py`; zero extra ROS package boilerplate on host or Colab. |
| Neighbor slots (`k_peer`) | **`k_peer = 4`** (`obs_dim = 62`) | Compact observation, scales to any `N`, while the shield still enforces global pair safety across all peers. |
| Speed / acceleration limits | **`speed = 1.5 m/s`, `accel = 2.0 m/s²`, `yaw_rate = 0.6 rad/s`** | Keeps 10-drone transit across the ±15 m arena well within 60 s episodes while remaining conservative for PX4 x500 tracking. |
| MARL algorithm at `N = 10` | **Parameter-shared MAPPO directly (pure PyTorch CTDE, Option A)** | Avoids MADDPG's unstable 660-D joint Q-critic and days of wasted Colab runs; tiny RAM footprint (< 15 MB rollout buffer), zero Ray RLlib breakage risk, exports the exact same deterministic `62→256→256→4` actor to ROS 2. |
| Shield solver | **Pure-NumPy prioritized-slack 3-D QP + fast-path check** | Solves the 3-variable velocity QP in microseconds with zero C-extension/install headaches (`osqp`/`cvxpy` not needed in ROS 2 or Colab). |
| Checkpoint separation | **`swarm_policy_best.pt` (deploy) vs. `train_state_latest.pt` (resume)** | Guarantees `torch.load(..., weights_only=True)` in ROS 2 `ActorPolicy` while preserving Adam optimizer + critic + `log_std` + RNG state for instant Colab resume. |
| Rain demonstration | **Visual rain particles + wind gusts + telemetry noise + 20% dropout** | Honest, quantifiable disturbance rejection trained via domain randomisation (not purely cosmetic). |
| SITL compute strategy | **10 drones headless (`control_rate=5 Hz`); fallback 6 SITL + 10 kinematic replay** | Smooth demo guaranteed even on laptops with < 16 CPU cores; clearly reports which runs are SITL vs. kinematic. |

---

## 20. What not to do

- ❌ Train an intermediate 3-drone v1 policy first only to throw it away when scaling to 10 drones.
- ❌ Keep the supervisor in the learned path "just for docking" — that is the AI not being the project.
- ❌ Train on the Gen-1 contract and remap at inference — permanent semantic debt.
- ❌ Use `allow_legacy_checkpoint:=true` as a fix — it bypasses the check, not the mismatch.
- ❌ Save optimizer/RNG state inside the deployment `.pt` file — it breaks `weights_only=True` in `ActorPolicy`.
- ❌ Let the shield enforce the soft thresholds (1.1 m, 0.25 m) — the no-shield ablation will fail.
- ❌ Claim "the AI avoids obstacles" before the no-shield run shows zero collisions.
- ❌ Bump the schema twice (once for 10 drones, once for birds) — design v2 once.
- ❌ Regenerate only `scenario.json` or only `swarm.sdf` — they must stay consistent.
- ❌ Run the demo with targets outside the trained `goal_scale`.
- ❌ Start a multi-hour Colab run without running Cell 4 (parity + smoke test) or without writing directly to Google Drive.

---

## 21. Glossary

| Term | Meaning |
|---|---|
| **ENU** | East-North-Up world frame used everywhere except PX4 |
| **NED** | North-East-Down frame used by PX4 |
| **Contract** | The frozen observation/action schema (`swarm-coop-enu-v*`) |
| **Shield** | CBF-style velocity projection filter (`safety.py`) |
| **Supervisor** | Hand-written approach/docking/hold state machine (`supervisor.py`) |
| **Classical** | A* macro controller (`planning.py`) |
| **CTDE** | Centralised training, decentralised execution |
| **MAPPO** | Multi-Agent Proximal Policy Optimization — on-policy CTDE with a centralised value critic `V(s_all, o_i)` and clipped policy updates (active algorithm for `N = 10`) |
| **GAE** | Generalized Advantage Estimation — variance-reduced advantage estimator (`gamma=0.99, gae_lambda=0.95`) used by MAPPO |
| **MADDPG** | Multi-agent DDPG with a centralised Q-critic (legacy 3-drone prototype; superseded by MAPPO for `N = 10`) |
| **Intervention** | Magnitude of the shield's correction; low = the AI is doing the work |
| **Infeasible** | The shield cannot satisfy all constraints; currently returns zero |
| **Domain randomisation** | Training with randomised disturbances to improve robustness |

---

*End of document. Updated on branch `feature/m1-m2-integration` after commit `116e87bbaa06f740f105a3c7242774bc66ffd7dd` (25 swarm/MARL regression tests + 15 MCP unit tests passing; 4-package ROS 2 workspace built). Next priorities: close out MCP fixes (`gazebo_version`, `ROS_LOCALHOST_ONLY=1`) + isolated live 3-drone SITL validation, then satisfy Stage 1 (`stage1_open`) acceptance criteria on Google Colab without modifying the frozen `swarm-coop-enu-v2` contract before starting Stage 2, and complete M1 ROS 2 deployment updates (`node.py`, `swarm.launch.py`).*
