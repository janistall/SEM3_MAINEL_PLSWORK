"""Phase 2 unit & smoke tests for the 10-drone Contract v2 training environment (`marl_training/envs.py`)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marl_training.envs import (
    DroneSwarmEnv,
    FORMATION_SHAPES,
    STAGE_CONFIGS,
    generate_formation_slots,
)
from swarm_control.observation import Telemetry


class TestDroneSwarmEnvV2(unittest.TestCase):
    def test_10_drone_reset_and_20_step_smoke(self):
        env = DroneSwarmEnv(num_drones=10, max_steps=600, shield=True)
        obs, infos = env.reset(seed=7)
        self.assertEqual(len(obs), 10)
        self.assertEqual(len(infos), 10)
        self.assertEqual(env.agents, [f"drone_{i}" for i in range(10)])

        for agent_id, vec in obs.items():
            self.assertEqual(vec.shape, (62,))
            self.assertEqual(vec.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(vec)))
            self.assertTrue(np.all(vec >= -1.0) and np.all(vec <= 1.0))
            self.assertTrue(env.observation_space(agent_id).contains(vec))

        for _ in range(20):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            self.assertEqual(len(obs), 10)
            self.assertEqual(len(rew), 10)
            self.assertTrue(all(np.isfinite(list(rew.values()))))
            # Synchronized termination / truncation across all 10 drones
            self.assertEqual(len(set(term.values())), 1)
            self.assertEqual(len(set(trunc.values())), 1)
            sample_info = info["drone_0"]
            for key in (
                "goal_distance",
                "docked",
                "fraction_docked",
                "success",
                "collision",
                "out_of_bounds",
                "min_pair_distance",
                "min_obstacle_clearance",
                "min_bird_clearance",
                "shield_intervention",
                "shield_feasible",
                "action_jerk",
                "step_path_length",
                "goal_shifted",
            ):
                self.assertIn(key, sample_info)
            if not env.agents:
                break

    def test_formation_generator_shapes_and_spacing(self):
        for shape in FORMATION_SHAPES:
            slots = generate_formation_slots(
                num_drones=10, shape=shape, spacing=1.6, heading=0.45
            )
            self.assertEqual(slots.shape, (10, 3))
            self.assertTrue(np.allclose(slots[:, 2], 0.0))
            min_pair = float("inf")
            for i in range(10):
                for j in range(i + 1, 10):
                    d = float(np.linalg.norm(slots[i] - slots[j]))
                    min_pair = min(min_pair, d)
            self.assertGreaterEqual(
                min_pair,
                1.6 - 1e-6,
                msg=f"shape={shape} violated minimum spacing 1.6 m (got {min_pair})",
            )

    def test_curriculum_stages_and_domain_randomization(self):
        for stage_name, cfg in STAGE_CONFIGS.items():
            env = DroneSwarmEnv.for_stage(stage_name, max_steps=40, goal_change_prob=1.0)
            obs, _ = env.reset(seed=11)
            self.assertEqual(len(obs), 10)
            lo_s, hi_s = cfg["num_static_range"]
            lo_d, hi_d = cfg["num_dyn_range"]
            self.assertGreaterEqual(len(env.obstacles), lo_s)
            self.assertLessEqual(len(env.obstacles), hi_s)
            self.assertGreaterEqual(len(env.dyn_obstacles), lo_d)
            self.assertLessEqual(len(env.dyn_obstacles), hi_d)

            shifted_seen = False
            for _ in range(30):
                actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
                obs, rew, term, trunc, info = env.step(actions)
                if info["drone_0"]["goal_shifted"]:
                    shifted_seen = True
                if not env.agents:
                    break
            self.assertTrue(shifted_seen, msg=f"expected mid-episode goal shift in {stage_name}")

    def test_synchronized_termination_collision_oob_and_success(self):
        # 1. Collision termination (two drones overlapping with shield=False)
        env = DroneSwarmEnv(num_drones=10, max_steps=100, shield=False, stage="stage1_open")
        env.reset(seed=7)
        env.telemetry[1] = Telemetry(
            position=env.telemetry[0].position + np.array([0.4, 0.0, 0.0]),
            velocity=np.zeros(3),
            yaw=0.0,
            agent_id="drone_1",
        )
        zero_actions = {a: np.zeros(4, dtype=np.float32) for a in env.possible_agents}
        _, _, term, trunc, info = env.step(zero_actions)
        self.assertTrue(all(term.values()))
        self.assertFalse(any(trunc.values()))
        self.assertTrue(info["drone_0"]["collision"])
        self.assertEqual(env.agents, [])

        # 2. Out-of-bounds termination
        env.reset(seed=7)
        env.telemetry[0] = Telemetry(
            position=np.array([16.5, 0.0, 2.5]),
            velocity=np.zeros(3),
            yaw=0.0,
            agent_id="drone_0",
        )
        _, _, term, trunc, info = env.step(zero_actions)
        self.assertTrue(all(term.values()))
        self.assertTrue(info["drone_0"]["out_of_bounds"])
        self.assertEqual(env.agents, [])

        # 3. Synchronized success after hold_dwell steps at goal slots
        env_hold = DroneSwarmEnv(
            num_drones=10, max_steps=100, shield=True, stage="stage1_open", hold_dwell=3
        )
        env_hold.reset(seed=7)
        for i, agent_id in enumerate(env_hold.possible_agents):
            env_hold.telemetry[i] = Telemetry(
                position=env_hold.goals[i].copy(),
                velocity=np.zeros(3),
                yaw=0.0,
                agent_id=agent_id,
            )
            env_hold._prev_dist[i] = 0.0

        for step_idx in range(3):
            _, _, term, trunc, info = env_hold.step(zero_actions)
            self.assertAlmostEqual(info["drone_0"]["fraction_docked"], 1.0)
            if step_idx < 2:
                self.assertFalse(any(term.values()))
            else:
                self.assertTrue(all(term.values()))
                self.assertTrue(info["drone_0"]["success"])
                self.assertEqual(env_hold.agents, [])

    def test_scenario_path_loading(self):
        scenario_file = _REPO_ROOT / "src" / "swarm_bringup" / "config" / "scenario.json"
        if not scenario_file.is_file():
            self.skipTest("packaged scenario.json not present")
        env = DroneSwarmEnv(
            num_drones=10,
            scenario_path=scenario_file,
            randomize=False,
            shield=True,
        )
        obs, _ = env.reset(seed=7)
        self.assertEqual(len(obs), 10)
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        obs, rew, term, trunc, _ = env.step(actions)
        self.assertEqual(len(obs), 10)
        self.assertTrue(all(np.isfinite(list(rew.values()))))


if __name__ == "__main__":
    unittest.main()
