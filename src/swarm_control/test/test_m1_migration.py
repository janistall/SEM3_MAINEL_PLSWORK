"""Comprehensive regression tests for M1 10-drone controller, launch, and disturbance interfaces."""
from __future__ import annotations

import ast
import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRINGUP_SRC = _REPO_ROOT / "src" / "swarm_bringup"
_ADAPTER_SRC = _REPO_ROOT / "src" / "px4_adapter"
for _src in (_BRINGUP_SRC, _ADAPTER_SRC):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from px4_adapter.noise import inject_telemetry_noise
from px4_adapter.rate import TelemetryRateLimiter
from swarm_bringup.launch_plan import (
    build_launch_plan,
    resolve_checkpoint_and_limits,
    resolve_scenario_and_world,
)
from swarm_bringup.scenario import Scenario, generate, write_world
from swarm_control.contract import (
    Contract,
    DEFAULT_CONTRACT,
    build_actor,
    make_deploy_payload,
)
from swarm_control.controller_core import ControllerConfig, SwarmControllerCore
from swarm_control.observation import DynamicObstacle, Obstacle


def _populate_fresh_fleet(
    core: SwarmControllerCore,
    *,
    now: float = 10.0,
    active_indices: set[int] | None = None,
    stale_indices: set[int] | None = None,
    own_pos: np.ndarray | None = None,
    own_vel: np.ndarray | None = None,
) -> None:
    n = len(core.agent_ids)
    active_set = set(range(n)) if active_indices is None else set(active_indices)
    stale_set = set() if stale_indices is None else set(stale_indices)
    for idx, aid in enumerate(core.agent_ids):
        if idx == core.agent_index and own_pos is not None:
            pos = own_pos
        else:
            # Spread drones safely > 2 m apart on a ring at z=2.5 m
            angle = 2.0 * math.pi * idx / max(n, 1)
            pos = np.array([4.0 * math.cos(angle), 4.0 * math.sin(angle), 2.5])
        vel = (
            own_vel
            if (idx == core.agent_index and own_vel is not None)
            else np.zeros(3, dtype=np.float64)
        )
        rx_time = now - (core.telemetry_timeout + 0.5) if idx in stale_set else now
        core.update_telemetry(
            agent_id=aid,
            position=pos,
            velocity=vel,
            yaw=0.0,
            stamp=rx_time,
            active=(idx in active_set),
            now=rx_time,
        )


class TestM1ControllerMigration(unittest.TestCase):
    def test_10_drone_learned_bypasses_supervisor_and_syncs_v2_checkpoint(self) -> None:
        import torch

        custom_contract = Contract(
            agents=10,
            speed=1.5,
            accel=2.0,
            yaw_rate=0.6,
            alt=2.5,
        )
        actor = build_actor(custom_contract)
        with torch.no_grad():
            for param in actor.parameters():
                param.zero_()
            # Set final layer bias so tanh([0.5, 0.0, 0.0, 0.5]) -> positive vx & yaw_rate
            actor[-2].bias.copy_(torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=torch.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "swarm_policy_v2.pt"
            payload = make_deploy_payload(
                custom_contract, actor.state_dict(), stage="stage1_open"
            )
            torch.save(payload, str(ckpt))

            # Intentionally request legacy v1 speed/accel and use_supervisor=True;
            # learned mode must force use_supervisor=False and adopt checkpoint v2 limits.
            cfg = ControllerConfig(
                drone_id="drone_0",
                agent_index=0,
                agent_ids=[f"drone_{i}" for i in range(10)],
                slot=(0.2, 0.0, 2.5),  # Very close to drone_0 so legacy Supervisor would DOCK/zero yaw!
                start=(0.0, 0.0, 0.45),
                mode="learned",
                checkpoint=str(ckpt),
                use_supervisor=True,
                speed=0.8,
                accel=1.2,
                enable_shield=True,
                quorum_min_peers=3,
            )
            core = SwarmControllerCore(cfg)
            self.assertFalse(core.use_supervisor)
            self.assertIsNotNone(core.supervisor_override_warning)
            self.assertEqual(core.contract.schema, "swarm-coop-enu-v2")
            self.assertEqual(core.contract.obs_dim, 62)
            self.assertAlmostEqual(core.speed, 1.5)
            self.assertAlmostEqual(core.accel, 2.0)
            self.assertAlmostEqual(core.yaw_rate_max, 0.6)
            self.assertEqual(core.shield.pair_safe, DEFAULT_CONTRACT.hard_pair)
            self.assertEqual(core.shield.workspace[0], (-15.0, 15.0))

            _populate_fresh_fleet(
                core,
                now=20.0,
                own_pos=np.array([0.0, 0.0, 2.5]),
                own_vel=np.zeros(3),
            )
            out = core.step(now=20.05)
            self.assertTrue(out.quorum_ok)
            self.assertEqual(out.fresh_peer_count, 9)
            self.assertEqual(out.policy_mode, "LEARNED")
            self.assertFalse(out.supervisor_enabled)
            self.assertEqual(out.supervisor_mode, "LEARNED")
            self.assertIsNotNone(out.observation)
            assert out.observation is not None
            self.assertEqual(out.observation.shape, (62,))
            # Legacy Supervisor would have zeroed yaw_rate inside docking distance; learned mode preserves it.
            expected_yaw = float(math.tanh(0.5) * 0.6)
            self.assertAlmostEqual(out.yaw_rate, expected_yaw, places=4)
            self.assertGreater(out.desired_velocity[0], 0.5)

            # Debug vector: 62 obs + 4 action + 3 desired + 3 shielded + 3 scalars = 75 floats
            self.assertIsNotNone(out.debug_vector)
            assert out.debug_vector is not None
            self.assertEqual(out.debug_vector.shape, (75,))
            self.assertEqual(out.debug_vector.dtype, np.float32)

            # Status dict keeps all MCP-compatible keys + §11.1 additions
            assert out.status_dict is not None
            for required_key in (
                "drone_id",
                "mode",
                "policy",
                "supervisor",
                "position",
                "velocity",
                "goal_distance",
                "d_stop",
                "action",
                "shield_feasible",
                "shield_intervention",
                "stall_recovery",
                "quorum_peers",
            ):
                self.assertIn(required_key, out.status_dict)
            self.assertEqual(out.status_dict["policy"], "LEARNED")
            self.assertFalse(out.status_dict["supervisor"])
            self.assertEqual(out.status_dict["quorum_peers"], 9)

    def test_quorum_gating_failsafe_and_stale_peer_zero_padding(self) -> None:
        cfg = ControllerConfig(
            drone_id="drone_0",
            agent_index=0,
            agent_ids=[f"drone_{i}" for i in range(10)],
            slot=(5.0, 0.0, 2.5),
            start=(0.0, 0.0, 0.45),
            mode="zero",
            quorum_min_peers=3,
            telemetry_timeout=0.5,
        )
        core = SwarmControllerCore(cfg)

        # 1. Self inactive -> fail-safe zero
        _populate_fresh_fleet(core, now=10.0, active_indices=set(range(1, 10)))
        out_disarmed = core.step(now=10.05)
        self.assertFalse(out_disarmed.quorum_ok)
        self.assertTrue(np.allclose(out_disarmed.velocity, 0.0))

        # 2. Only 2 active peers (< quorum 3) -> fail-safe zero
        _populate_fresh_fleet(core, now=10.0, active_indices={0, 1, 2})
        out_low_quorum = core.step(now=10.05)
        self.assertFalse(out_low_quorum.quorum_ok)
        self.assertEqual(out_low_quorum.fresh_peer_count, 2)
        self.assertTrue(np.allclose(out_low_quorum.velocity, 0.0))

        # 3. Exactly 3 fresh active peers (6 peers stale) -> passes quorum, 4th peer slot padded valid=0
        _populate_fresh_fleet(
            core,
            now=10.0,
            active_indices=set(range(10)),
            stale_indices={4, 5, 6, 7, 8, 9},
            own_pos=np.array([0.0, 0.0, 2.5]),
        )
        out_ok = core.step(now=10.05)
        self.assertTrue(out_ok.quorum_ok)
        self.assertEqual(out_ok.fresh_peer_count, 3)
        assert out_ok.observation is not None
        # 3 nearest peer slots valid=1.0 (indices 12, 19, 26), 4th slot strictly zero (indices 27..33)
        self.assertAlmostEqual(float(out_ok.observation[12]), 1.0)
        self.assertAlmostEqual(float(out_ok.observation[19]), 1.0)
        self.assertAlmostEqual(float(out_ok.observation[26]), 1.0)
        self.assertTrue(np.allclose(out_ok.observation[27:34], 0.0))
        # Agent IDs populated in telemetry dictionary
        self.assertEqual(core.telemetry["drone_3"].agent_id, "drone_3")

    def test_goal_and_dynamic_obstacle_interfaces_and_expiry(self) -> None:
        cfg = ControllerConfig(
            drone_id="drone_0",
            agent_index=0,
            agent_ids=[f"drone_{i}" for i in range(6)],
            slot=(5.0, 0.0, 2.5),
            start=(0.0, 0.0, 0.45),
            mode="learned",
            quorum_min_peers=2,
            enable_shield=True,
            dynamic_obstacle_timeout=1.0,
            shield_lookahead_tau=0.3,
        )
        # Constant forward +x policy
        core = SwarmControllerCore(
            cfg,
            policy_override=lambda _obs: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        _populate_fresh_fleet(
            core,
            now=30.0,
            own_pos=np.array([0.0, 0.0, 2.5]),
            own_vel=np.zeros(3),
        )

        # Runtime goal update with z=0 preserves altitude; out-of-workspace rejected
        accepted, _ = core.update_goal((6.0, -2.0, 0.0))
        self.assertTrue(accepted)
        self.assertTrue(np.allclose(core.slot, [6.0, -2.0, 2.5]))

        rejected, _ = core.update_goal((25.0, 0.0, 2.5))
        self.assertFalse(rejected)
        self.assertTrue(np.allclose(core.slot, [6.0, -2.0, 2.5]))

        # Approaching dynamic obstacle ("bird") enters observation [46:53] AND causes shield avoidance
        bird = DynamicObstacle(
            position=np.array([0.95, 0.0, 2.5]),
            velocity=np.array([-1.2, 0.0, 0.0]),
            radius=0.35,
            active=True,
        )
        core.update_dynamic_obstacles([bird], now=30.0)
        out_bird = core.step(now=30.1)
        self.assertTrue(out_bird.quorum_ok)
        self.assertEqual(out_bird.dynamic_obstacle_count, 1)
        assert out_bird.observation is not None
        self.assertAlmostEqual(float(out_bird.observation[52]), 1.0)
        self.assertLess(out_bird.velocity[0], 0.0)

        # After dynamic_obstacle_timeout (1.0 s), bird expires cleanly
        _populate_fresh_fleet(
            core,
            now=31.5,
            own_pos=np.array([0.0, 0.0, 2.5]),
            own_vel=np.zeros(3),
        )
        out_expired = core.step(now=31.55)
        self.assertEqual(out_expired.dynamic_obstacle_count, 0)
        assert out_expired.observation is not None
        self.assertTrue(np.allclose(out_expired.observation[46:60], 0.0))
        self.assertGreater(out_expired.velocity[0], 0.0)

    def test_classical_mode_uses_supervisor_replanning_and_geometric_stall_recovery(self) -> None:
        obstacles = [Obstacle(2.0, 0.0, 0.5, 5.0)]
        cfg = ControllerConfig(
            drone_id="drone_0",
            agent_index=0,
            agent_ids=["drone_0", "drone_1", "drone_2"],
            slot=(6.0, 0.0, 2.5),
            start=(0.0, 0.0, 0.45),
            mode="classical",
            quorum_min_peers=2,
            stall_recovery=True,
            stall_time=0.2,
            stall_recover_time=0.5,
            stall_clearance=0.6,
        )
        core = SwarmControllerCore(cfg, obstacles=obstacles)
        self.assertTrue(core.use_supervisor)

        # Position right next to obstacle (clearance = 1.2 - 0.5 - 0.45 = 0.25 < 0.6) with 0 vel
        for step_idx in range(4):
            t = 40.0 + step_idx * 0.1
            _populate_fresh_fleet(
                core,
                now=t,
                own_pos=np.array([0.8, 0.0, 2.5]),
                own_vel=np.zeros(3),
            )
            out = core.step(now=t)
        self.assertTrue(out.quorum_ok)
        self.assertTrue(out.stall_active)
        self.assertLess(out.desired_velocity[0], 0.0)

        # While still inside A*'s (o.radius + 1.3 = 1.8 m) exclusion zone at (0.8, 0.0),
        # classical replanning must safely decline instead of weakening A* clearance.
        blocked_from_tight_spot, _ = core.update_goal((0.0, 5.0, 2.5))
        self.assertFalse(blocked_from_tight_spot)
        self.assertTrue(np.allclose(core.slot, [6.0, 0.0, 2.5]))

        # After retreating along -x to (0.0, 0.0) (distance 2.0 m > 1.8 m A* margin),
        # runtime goal replanning succeeds and still rejects an obstacle-interior target.
        _populate_fresh_fleet(
            core,
            now=41.0,
            own_pos=np.array([0.0, 0.0, 2.5]),
            own_vel=np.zeros(3),
        )
        ok_replan, _ = core.update_goal((0.0, 5.0, 2.5))
        self.assertTrue(ok_replan)
        self.assertTrue(np.allclose(core.slot, [0.0, 5.0, 2.5]))

        bad_replan, _ = core.update_goal((2.0, 0.0, 2.5))
        self.assertFalse(bad_replan)
        self.assertTrue(np.allclose(core.slot, [0.0, 5.0, 2.5]))


class TestM1LaunchAndDisturbanceInterfaces(unittest.TestCase):
    def test_launch_plan_supports_default_10_and_configurable_3_6_10_drones(self) -> None:
        pkg_root = _REPO_ROOT / "src" / "swarm_bringup"

        # Default uses packaged 10-drone scenario & world; zero/classical require NO checkpoint
        plan_10 = build_launch_plan(pkg_root, mode="zero", headless=False, auto_arm=False)
        self.assertEqual(plan_10.num_drones, 10)
        self.assertEqual(plan_10.scenario_path, pkg_root / "config" / "scenario.json")
        self.assertEqual(plan_10.world_path, pkg_root / "worlds" / "swarm.sdf")
        self.assertEqual(plan_10.checkpoint, "")
        self.assertFalse(plan_10.headless)
        self.assertFalse(plan_10.auto_arm)
        self.assertEqual(plan_10.agent_ids, [f"drone_{i}" for i in range(10)])
        self.assertEqual([d.px4_namespace for d in plan_10.drones], [f"/uav_{i}" for i in range(10)])
        self.assertEqual([d.system_id for d in plan_10.drones], list(range(1, 11)))
        self.assertEqual([d.px4_start_delay for d in plan_10.drones], [6.0 + 2.0 * i for i in range(10)])
        self.assertTrue(all(math.isclose(d.takeoff_altitude, 2.5) for d in plan_10.drones))

        # Configurable 3-drone & 6-drone plans generate consistent runtime scenario + world
        with tempfile.TemporaryDirectory() as tmpdir:
            for count in (3, 6):
                plan_n = build_launch_plan(
                    pkg_root,
                    drones_arg=count,
                    mode="classical",
                    headless=True,
                    auto_arm=False,
                    runtime_dir=Path(tmpdir) / f"run_{count}d",
                )
                self.assertEqual(plan_n.num_drones, count)
                self.assertTrue(plan_n.scenario_path.is_file())
                self.assertTrue(plan_n.world_path.is_file())
                self.assertEqual(plan_n.checkpoint, "")
                self.assertTrue(plan_n.use_supervisor)
                self.assertIn(f"goal_beacon_{count - 1}", plan_n.world_path.read_text(encoding="utf-8"))
                self.assertEqual(plan_n.quorum_min_peers, min(3, count - 1))

    def test_launch_plan_learned_mode_reads_v2_metadata_and_rejects_missing_ckpt(self) -> None:
        import torch

        pkg_root = _REPO_ROOT / "src" / "swarm_bringup"
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint_and_limits(
                pkg_root, mode="learned", checkpoint_arg="", num_drones=10
            )

        custom = Contract(agents=10, speed=1.4, accel=1.8, yaw_rate=0.5)
        actor = build_actor(custom)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "v2_custom.pt"
            torch.save(make_deploy_payload(custom, actor.state_dict()), str(ckpt))
            resolved_path, speed, accel, yaw_rate = resolve_checkpoint_and_limits(
                pkg_root, mode="learned", checkpoint_arg=str(ckpt), num_drones=10
            )
            self.assertEqual(resolved_path, str(ckpt))
            self.assertAlmostEqual(speed, 1.4)
            self.assertAlmostEqual(accel, 1.8)
            self.assertAlmostEqual(yaw_rate, 0.5)

    def test_disturbance_wind_rain_sdf_and_adapter_telemetry_noise(self) -> None:
        scenario = generate(seed=7, count=4, num_drones=6, shape="ring", spacing=1.6)
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = Path(tmpdir) / "rainy_windy.sdf"
            write_world(scenario, sdf_path, wind=(3.5, -1.2, 0.0), rain=True)
            sdf_text = sdf_path.read_text(encoding="utf-8")
            root = ET.fromstring(sdf_text)
            self.assertEqual(root.tag, "sdf")
            self.assertIn("<wind><linear_velocity>3.5 -1.2 0.0</linear_velocity></wind>", sdf_text)
            self.assertIn('particle_emitter name="rain_emitter"', sdf_text)

        # Adapter telemetry noise: exact identity at std=0.0, reproducible Gaussian noise at std>0.0
        pos = np.array([1.0, -2.0, 2.5])
        vel = np.array([0.4, -0.3, 0.1])
        yaw = 3.1
        p0, v0, y0 = inject_telemetry_noise(pos, vel, yaw, pos_std=0.0, vel_std=0.0, yaw_std=0.0)
        self.assertTrue(np.allclose(p0, pos))
        self.assertTrue(np.allclose(v0, vel))
        self.assertAlmostEqual(y0, yaw)

        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pa, va, ya = inject_telemetry_noise(
            pos, vel, yaw, pos_std=0.05, vel_std=0.02, yaw_std=0.1, rng=rng_a
        )
        pb, vb, yb = inject_telemetry_noise(
            pos, vel, yaw, pos_std=0.05, vel_std=0.02, yaw_std=0.1, rng=rng_b
        )
        self.assertTrue(np.allclose(pa, pb))
        self.assertTrue(np.allclose(va, vb))
        self.assertAlmostEqual(ya, yb)
        self.assertFalse(np.allclose(pa, pos))
        self.assertGreaterEqual(ya, -math.pi)
        self.assertLess(ya, math.pi)

    def test_telemetry_rate_limiter_thread_pinning_and_world_optimization(self) -> None:
        # 1. TelemetryRateLimiter maintains 10 Hz under 20 Hz ticks with +/-3 ms timer jitter
        limiter = TelemetryRateLimiter(10.0)
        pub_times: list[float] = []
        jitter = [0.0, 0.002, -0.003, 0.001, -0.002, 0.003, -0.001, 0.002, -0.002, 0.001]
        for tick_idx in range(20):
            t = tick_idx * 0.05 + jitter[tick_idx % len(jitter)]
            if limiter.should_publish(t):
                pub_times.append(t)
        self.assertEqual(len(pub_times), 10)

        # 20 Hz limiter publishes on every 20 Hz tick
        limiter_20 = TelemetryRateLimiter(20.0)
        self.assertEqual(sum(1 for i in range(10) if limiter_20.should_publish(i * 0.05)), 10)

        # Invalid rates rejected
        with self.assertRaises(ValueError):
            TelemetryRateLimiter(0.0)
        with self.assertRaises(ValueError):
            TelemetryRateLimiter(-5.0)

        # 2. Launch plan exposes telemetry_rate (default 10.0 Hz) and swarm.launch.py pins BLAS/OMP threads
        pkg_root = _REPO_ROOT / "src" / "swarm_bringup"
        plan = build_launch_plan(pkg_root, mode="zero")
        self.assertAlmostEqual(plan.telemetry_rate, 10.0)
        with self.assertRaises(ValueError):
            build_launch_plan(pkg_root, mode="zero", telemetry_rate=0.0)

        launch_text = (pkg_root / "launch" / "swarm.launch.py").read_text(encoding="utf-8")
        for env_var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TORCH_NUM_THREADS",
            "telemetry_rate",
        ):
            self.assertIn(env_var, launch_text)

        # 3. Packaged worlds/swarm.sdf matches write_world(Scenario.load(config/scenario.json))
        #    with shadows disabled and consolidated single arena_grid model (61 fewer ECS models).
        scenario = Scenario.load(pkg_root / "config" / "scenario.json")
        packaged_sdf = (pkg_root / "worlds" / "swarm.sdf").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            regen_path = Path(tmpdir) / "swarm.sdf"
            write_world(scenario, regen_path)
            self.assertEqual(packaged_sdf, regen_path.read_text(encoding="utf-8"))
        self.assertIn("<shadows>false</shadows>", packaged_sdf)
        self.assertIn("<cast_shadows>false</cast_shadows>", packaged_sdf)
        self.assertIn('<model name="arena_grid">', packaged_sdf)
        self.assertNotIn('<model name="grid_0_0">', packaged_sdf)

    def test_ros_entry_modules_parse_cleanly_as_python_ast(self) -> None:
        for rel_path in (
            "src/swarm_control/swarm_control/node.py",
            "src/swarm_control/swarm_control/controller_core.py",
            "src/swarm_bringup/launch/swarm.launch.py",
            "src/swarm_bringup/swarm_bringup/launch_plan.py",
            "src/swarm_bringup/swarm_bringup/scenario.py",
            "src/px4_adapter/px4_adapter/adapter_node.py",
            "src/px4_adapter/px4_adapter/noise.py",
            "src/px4_adapter/px4_adapter/rate.py",
        ):
            source = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
            self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
