"""Tests for Step 2 (10-drone scenario/world), Step 3 (3-D slack QP shield), and Phases 3 & 4 (MAPPO stack)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BRINGUP_SRC = _REPO_ROOT / "src" / "swarm_bringup"
if str(_BRINGUP_SRC) not in sys.path:
    sys.path.insert(0, str(_BRINGUP_SRC))

from marl_training.contracts import ActorPolicy, DEFAULT_CONTRACT
from marl_training.envs import DroneSwarmEnv
from marl_training.eval import (
    EVAL_MODES,
    _make_env_for_mode,
    evaluate_policy,
    main as eval_main,
)
from marl_training.mappo import MAPPOTrainer, RolloutBuffer
from marl_training.models import CentralizedValueNet, SwarmActorNet
from marl_training.train import build_arg_parser, run_training, save_deploy_pt
from swarm_bringup.scenario import Scenario, formation_slots, generate, write_world
from swarm_control.observation import DynamicObstacle, Obstacle, Telemetry
from swarm_control.planning import astar
from swarm_control.safety import ShieldConfig, safe_velocity


class TestSteps2And3AndPhases3And4(unittest.TestCase):
    def test_step2_10_drone_scenario_and_world_consistency(self):
        cfg_path = _REPO_ROOT / "src" / "swarm_bringup" / "config" / "scenario.json"
        sdf_path = _REPO_ROOT / "src" / "swarm_bringup" / "worlds" / "swarm.sdf"
        self.assertTrue(cfg_path.is_file())
        self.assertTrue(sdf_path.is_file())

        scenario = Scenario.load(cfg_path)
        self.assertEqual(len(scenario.starts), 10)
        self.assertEqual(len(scenario.goals), 10)
        self.assertEqual(len(scenario.obstacles), 7)

        typed_obs = [Obstacle(x, y, r, h) for x, y, r, h in scenario.obstacles]
        for start, goal in zip(scenario.starts, scenario.goals):
            route = astar(np.asarray(start), np.asarray(goal), typed_obs)
            self.assertIsNotNone(route)

        sdf_text = sdf_path.read_text()
        self.assertIn("goal_beacon_9", sdf_text)
        self.assertIn("START_ZONE", sdf_text)
        self.assertIn("GOAL_ZONE", sdf_text)

        for shape in ("ring", "grid", "v"):
            slots = formation_slots(10, shape=shape, spacing=1.6)
            self.assertEqual(slots.shape, (10, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = generate(seed=11, count=5, num_drones=10, shape="grid", spacing=1.6)
            self.assertEqual(len(gen.starts), 10)
            out_sdf = Path(tmpdir) / "swarm.sdf"
            write_world(gen, out_sdf)
            self.assertIn("goal_beacon_9", out_sdf.read_text())

    def test_step3_shield_fast_path_qp_and_retreat_fallback(self):
        cfg = ShieldConfig.from_contract(DEFAULT_CONTRACT)
        own_pos = np.array([0.0, 0.0, 2.5])
        own_vel = np.array([0.2, -0.1, 0.0])

        # 1. Fast-path: no nearby peers/obstacles and desired velocity within accel box -> 0 intervention
        desired = np.array([0.25, -0.05, 0.0])
        res_fast = safe_velocity(own_pos, own_vel, desired, [], [], 0.1, cfg)
        self.assertTrue(res_fast.feasible)
        self.assertAlmostEqual(res_fast.intervention, 0.0, places=9)
        self.assertTrue(np.allclose(res_fast.velocity, desired))

        # 2. Multi-constraint squeeze: peer inside hard envelope on +x AND static obstacle near -y
        # Priority weights (peer=400 > static=250) must prioritize escaping the peer in -x and +y
        peer = Telemetry(np.array([0.55, 0.0, 2.5]), np.zeros(3), 0.0, agent_id="drone_1")
        obstacle = Obstacle(0.0, -0.75, 0.45, 5.0)
        res_qp = safe_velocity(
            own_pos,
            np.zeros(3),
            np.array([1.0, -1.0, 0.0]),
            [peer],
            [obstacle],
            0.1,
            cfg,
        )
        self.assertFalse(res_qp.feasible)
        # Must actively retreat away from peer (+x) and away from obstacle (-y), not freeze at [0,0,0]
        self.assertLess(res_qp.velocity[0], -0.05)
        self.assertGreater(res_qp.velocity[1], 0.05)
        self.assertGreater(float(np.linalg.norm(res_qp.velocity)), 0.1)

        # 3. Lookahead tau on approaching dynamic bird
        cfg_look = ShieldConfig.from_contract(DEFAULT_CONTRACT, lookahead_tau=0.3)
        bird = DynamicObstacle(
            position=np.array([0.95, 0.0, 2.5]),
            velocity=np.array([-1.2, 0.0, 0.0]),
            radius=0.35,
            active=True,
        )
        res_bird = safe_velocity(
            own_pos,
            np.zeros(3),
            np.array([1.5, 0.0, 0.0]),
            [],
            [],
            0.1,
            cfg_look,
            dynamic_obstacles=[bird],
        )
        self.assertLess(res_bird.velocity[0], 0.0)

    def test_phase3_mappo_models_and_gae_truncation_vs_termination(self):
        actor = SwarmActorNet(obs_dim=62, act_dim=4, hidden=(256, 256))
        critic = CentralizedValueNet(num_drones=10, obs_dim=62, hidden=(256, 256))

        obs_t = torch.zeros((10, 62), dtype=torch.float32)
        det = actor.deterministic(obs_t)
        self.assertEqual(tuple(det.shape), (10, 4))
        raw_act, logp = actor.act(obs_t)
        self.assertEqual(tuple(raw_act.shape), (10, 4))
        self.assertEqual(tuple(logp.shape), (10,))
        eval_logp, entropy = actor.evaluate_actions(obs_t, raw_act)
        self.assertTrue(torch.allclose(logp, eval_logp, atol=1e-5))
        self.assertEqual(tuple(entropy.shape), (10,))

        vals = critic(obs_t)
        self.assertEqual(tuple(vals.shape), (10,))

        # Verify RolloutBuffer truncation bootstrapping vs termination zeroing
        buf = RolloutBuffer(rollout_steps=2, num_drones=2, obs_dim=4, act_dim=2)
        # Step 0: truncated step with non-zero bootstrap next_value=10.0
        buf.add(
            obs=np.zeros((2, 4), dtype=np.float32),
            raw_actions=np.zeros((2, 2), dtype=np.float32),
            log_probs=np.zeros(2, dtype=np.float32),
            rewards=np.ones(2, dtype=np.float32),
            values=np.full(2, 2.0, dtype=np.float32),
            next_values=np.full(2, 10.0, dtype=np.float32),
            dones=np.ones(2, dtype=np.float32),
        )
        # Step 1: true terminated step with next_value=0.0
        buf.add(
            obs=np.zeros((2, 4), dtype=np.float32),
            raw_actions=np.zeros((2, 2), dtype=np.float32),
            log_probs=np.zeros(2, dtype=np.float32),
            rewards=np.ones(2, dtype=np.float32),
            values=np.full(2, 2.0, dtype=np.float32),
            next_values=np.zeros(2, dtype=np.float32),
            dones=np.ones(2, dtype=np.float32),
        )
        buf.compute_gae(gamma=0.99, gae_lambda=0.95)
        # Step 0 truncated advantage = 1.0 + 0.99 * 10.0 - 2.0 = 8.9
        self.assertTrue(np.allclose(buf.advantages[0], 8.9, atol=1e-5))
        # Step 1 terminated advantage = 1.0 + 0.99 * 0.0 - 2.0 = -1.0
        self.assertTrue(np.allclose(buf.advantages[1], -1.0, atol=1e-5))

    def test_phase4_train_resume_deploy_export_and_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run01"
            parser = build_arg_parser()
            args = parser.parse_args(
                [
                    "--iters", "2",
                    "--num-drones", "10",
                    "--max-steps", "12",
                    "--rollout-steps", "24",
                    "--ppo-epochs", "2",
                    "--minibatch-size", "64",
                    "--eval-every", "1",
                    "--eval-episodes", "1",
                    "--checkpoint-every", "1",
                    "--stage", "stage1_open",
                    "--out", str(out_dir),
                ]
            )
            last_row = run_training(args)
            self.assertEqual(last_row["iter"], 2)
            self.assertTrue((out_dir / "swarm_policy_best.pt").is_file())
            self.assertTrue((out_dir / "swarm_policy_final.pt").is_file())
            self.assertTrue((out_dir / "train_state_latest.pt").is_file())
            self.assertTrue((out_dir / "metrics.jsonl").is_file())

            lines = (out_dir / "metrics.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            parsed = json.loads(lines[-1])
            self.assertIn("eval_success_rate", parsed)
            self.assertIn("policy_loss", parsed)

            # Verify ROS 2 ActorPolicy loads swarm_policy_best.pt with weights_only=True
            policy = ActorPolicy(out_dir / "swarm_policy_best.pt")
            self.assertEqual(policy.contract.schema, "swarm-coop-enu-v2")
            self.assertEqual(policy.contract.obs_dim, 62)
            self.assertEqual(policy.contract.agents, 10)
            act = policy(np.zeros(62, dtype=np.float32))
            self.assertEqual(act.shape, (4,))
            self.assertTrue(np.all(np.isfinite(act)))

            # Verify --resume continues from iteration 2 to 3
            args_resume = parser.parse_args(
                [
                    "--iters", "3",
                    "--num-drones", "10",
                    "--max-steps", "12",
                    "--rollout-steps", "24",
                    "--ppo-epochs", "2",
                    "--minibatch-size", "64",
                    "--eval-every", "1",
                    "--eval-episodes", "1",
                    "--checkpoint-every", "1",
                    "--stage", "stage1_open",
                    "--resume", str(out_dir / "train_state_latest.pt"),
                    "--out", str(out_dir),
                ]
            )
            resumed_row = run_training(args_resume)
            self.assertEqual(resumed_row["iter"], 3)
            self.assertEqual(resumed_row["env_steps_total"], 72)

            # Verify evaluate_policy runs cleanly on the loaded policy
            eval_res = evaluate_policy(
                policy,
                num_drones=10,
                max_steps=10,
                episodes=1,
                seeds=[11],
                mode="clean",
                shield=True,
            )
            self.assertIn("success_rate", eval_res)
            self.assertIn("mean_shield_intervention", eval_res)

    def test_eval_stage1_open_mode_cli_and_mismatch_warning(self):
        self.assertIn("open", EVAL_MODES)
        open_env = _make_env_for_mode(
            "open",
            num_drones=10,
            max_steps=20,
            shield=True,
            scenario_path=None,
            contract=DEFAULT_CONTRACT,
        )
        self.assertEqual(open_env.num_static_range, (0, 0))
        self.assertEqual(open_env.num_dyn_range, (0, 0))
        open_env.reset(seed=11)
        self.assertEqual(len(open_env.obstacles), 0)
        self.assertEqual(len(open_env.dyn_obstacles), 0)

        trainer = MAPPOTrainer(contract=DEFAULT_CONTRACT)
        with tempfile.TemporaryDirectory() as tmpdir:
            stage1_ckpt = Path(tmpdir) / "stage1_best.pt"
            stage2_ckpt = Path(tmpdir) / "stage2_best.pt"
            save_deploy_pt(
                trainer,
                DEFAULT_CONTRACT,
                stage1_ckpt,
                steps=2400,
                seed=7,
                stage="stage1_open",
                note="stage1 test",
            )
            save_deploy_pt(
                trainer,
                DEFAULT_CONTRACT,
                stage2_ckpt,
                steps=4800,
                seed=7,
                stage="stage2_static",
                note="stage2 test",
            )

            # 1. --stage stage1_open evaluates with zero obstacles and no mismatch warning
            stage_out = Path(tmpdir) / "stage1_report.json"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rc = eval_main(
                    [
                        "--checkpoint", str(stage1_ckpt),
                        "--stage", "stage1_open",
                        "--max-steps", "8",
                        "--episodes", "1",
                        "--seeds", "11",
                        "--out", str(stage_out),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual([w for w in caught if "stage1_open" in str(w.message)], [])
            stage_report = json.loads(stage_out.read_text())
            self.assertEqual(stage_report["checkpoint_stage"], "stage1_open")
            self.assertIn("stage1_open", stage_report["results"])
            self.assertIn("L+S", stage_report["results"]["stage1_open"])
            self.assertIn("L", stage_report["results"]["stage1_open"])

            # 2. --modes open evaluates with zero obstacles and no mismatch warning
            open_out = Path(tmpdir) / "open_report.json"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rc = eval_main(
                    [
                        "--checkpoint", str(stage1_ckpt),
                        "--modes", "open",
                        "--max-steps", "8",
                        "--episodes", "1",
                        "--seeds", "11",
                        "--out", str(open_out),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual([w for w in caught if "stage1_open" in str(w.message)], [])
            open_report = json.loads(open_out.read_text())
            self.assertIn("open", open_report["results"])

            # 3. stage1_open checkpoint evaluated in obstacle mode ('clean') emits warning
            with self.assertWarnsRegex(UserWarning, "stage1_open"):
                rc = eval_main(
                    [
                        "--checkpoint", str(stage1_ckpt),
                        "--modes", "clean",
                        "--max-steps", "6",
                        "--episodes", "1",
                        "--seeds", "11",
                    ]
                )
            self.assertEqual(rc, 0)

            # 4. stage1_open checkpoint evaluated in obstacle stage ('stage2_static') emits warning
            with self.assertWarnsRegex(UserWarning, "stage1_open"):
                rc = eval_main(
                    [
                        "--checkpoint", str(stage1_ckpt),
                        "--stage", "stage2_static",
                        "--max-steps", "6",
                        "--episodes", "1",
                        "--seeds", "11",
                    ]
                )
            self.assertEqual(rc, 0)

            # 5. stage2_static checkpoint evaluated in 'clean' does NOT emit stage1_open warning
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rc = eval_main(
                    [
                        "--checkpoint", str(stage2_ckpt),
                        "--modes", "clean",
                        "--max-steps", "6",
                        "--episodes", "1",
                        "--seeds", "11",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual([w for w in caught if "stage1_open" in str(w.message)], [])


if __name__ == "__main__":
    unittest.main()
