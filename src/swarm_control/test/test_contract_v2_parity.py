"""Phase 1 Contract v2 (`swarm-coop-enu-v2`) parity test harness.

Verifies:
  1. Single-source-of-truth import identity between `swarm_control` and `marl_training.contracts`.
  2. 62-D observation parity across randomized 10-drone states (peers, static, dynamic, missing/ties).
  3. Action decoding parity (`±1.5 m/s`, `±0.6 rad/s`, clipping).
  4. Hardcoded golden vectors (62-D observation + 4-D action/velocity) guarding layout drift.
  5. Unified Contract v2 / ShieldConfig tiered safety thresholds.
  6. Contract metadata validation, non-mutation of `DEFAULT_CONTRACT`, and `state_dict` key handling.
  7. PyTorch `ActorPolicy(..., weights_only=True)` round-trip and v1 rejection (auto-runs with torch).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure repo root is importable even when invoked via `PYTHONPATH=src/swarm_control`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import marl_training.contracts as train_contracts
from swarm_control.contract import (
    ACT_DIM,
    ActorPolicy,
    Contract,
    DEFAULT_CONTRACT,
    OBS_DIM,
    PEER_ORDER,
    SCHEMA,
    _infer_hidden,
    action_to_velocity,
    apply_contract_metadata,
    build_actor,
    make_deploy_payload,
    normalize_actor_state_dict,
    validate_action_spec,
)
from swarm_control.observation import DynamicObstacle, Obstacle, Telemetry, build_observation
from swarm_control.safety import ShieldConfig, safe_velocity

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _reference_v2_observation(
    agent_index: int,
    telemetry: list[Telemetry],
    goal: np.ndarray,
    obstacles: list[Obstacle],
    dynamic_obstacles: list[DynamicObstacle],
    contract: Contract,
) -> np.ndarray:
    """Independent reference implementation of the §15 `swarm-coop-enu-v2` table."""
    own = telemetry[agent_index]
    own_pos = np.asarray(own.position, dtype=np.float64)
    own_vel = np.asarray(own.velocity, dtype=np.float64)
    goal_pos = np.asarray(goal, dtype=np.float64)

    parts: list[np.ndarray] = [
        (goal_pos - own_pos) / contract.goal_scale,
        own_vel / contract.vel_scale,
    ]

    peers = [
        (idx, t)
        for idx, t in enumerate(telemetry)
        if idx != agent_index and t.active
    ]
    peers.sort(key=lambda pair: (float(np.linalg.norm(pair[1].position - own_pos)), pair[1].agent_id, pair[0]))
    for k in range(contract.k_peer):
        if k < len(peers):
            p = peers[k][1]
            parts.append(
                np.concatenate(
                    [
                        (p.position - own_pos) / contract.goal_scale,
                        (p.velocity - own_vel) / contract.peer_vel_scale,
                        np.array([1.0]),
                    ]
                )
            )
        else:
            parts.append(np.zeros(7, dtype=np.float64))

    static_ranked = sorted(
        enumerate(obstacles),
        key=lambda pair: (
            float(np.linalg.norm(np.array([pair[1].x, pair[1].y]) - own_pos[:2])) - pair[1].radius,
            pair[0],
        ),
    )
    for k in range(contract.k_static):
        if k < len(static_ranked):
            o = static_ranked[k][1]
            delta = np.array([o.x, o.y], dtype=np.float64) - own_pos[:2]
            dist = float(np.linalg.norm(delta))
            clearance = (dist - o.radius - contract.radius) / contract.obstacle_scale
            parts.append(np.array([delta[0] / contract.obstacle_scale, delta[1] / contract.obstacle_scale, clearance, 1.0]))
        else:
            parts.append(np.zeros(4, dtype=np.float64))

    dyn_ranked = sorted(
        [(idx, d) for idx, d in enumerate(dynamic_obstacles) if d.active],
        key=lambda pair: (float(np.linalg.norm(pair[1].position - own_pos)) - pair[1].radius, pair[0]),
    )
    for k in range(contract.k_dyn):
        if k < len(dyn_ranked):
            d = dyn_ranked[k][1]
            parts.append(
                np.concatenate(
                    [
                        (d.position - own_pos) / contract.obstacle_scale,
                        (d.velocity - own_vel) / contract.peer_vel_scale,
                        np.array([1.0]),
                    ]
                )
            )
        else:
            parts.append(np.zeros(7, dtype=np.float64))

    parts.append(np.array([math.sin(own.yaw), math.cos(own.yaw)], dtype=np.float64))
    vec = np.concatenate(parts).astype(np.float32)
    if contract.clip_obs:
        vec = np.clip(vec, -1.0, 1.0)
    return vec


class TestContractV2Parity(unittest.TestCase):
    def test_single_source_of_truth_reexports(self):
        self.assertIs(train_contracts.Contract, Contract)
        self.assertIs(train_contracts.DEFAULT_CONTRACT, DEFAULT_CONTRACT)
        self.assertIs(train_contracts.build_observation, build_observation)
        self.assertIs(train_contracts.action_to_velocity, action_to_velocity)
        self.assertIs(train_contracts.Telemetry, Telemetry)
        self.assertIs(train_contracts.Obstacle, Obstacle)
        self.assertIs(train_contracts.DynamicObstacle, DynamicObstacle)
        self.assertIs(train_contracts.ShieldConfig, ShieldConfig)
        self.assertIs(train_contracts.safe_velocity, safe_velocity)
        self.assertEqual(train_contracts.SCHEMA, "swarm-coop-enu-v2")
        self.assertEqual(train_contracts.OBS_DIM, 62)
        self.assertEqual(train_contracts.ACT_DIM, 4)
        self.assertEqual(train_contracts.N_AGENTS, 10)

    def test_obs_dim_is_computed_from_k_parameters(self):
        self.assertEqual(Contract(k_peer=4, k_static=3, k_dyn=2).obs_dim, 62)
        self.assertEqual(Contract(k_peer=9, k_static=3, k_dyn=2).obs_dim, 97)
        self.assertEqual(Contract(k_peer=4, k_static=3, k_dyn=0).obs_dim, 48)
        self.assertEqual(DEFAULT_CONTRACT.obs_dim, OBS_DIM)

    def test_randomized_10_drone_observation_parity(self):
        rng = np.random.default_rng(2026)
        for _ in range(50):
            telemetry = []
            for i in range(10):
                active = True if i == 0 else bool(rng.random() > 0.25)
                telemetry.append(
                    Telemetry(
                        position=rng.uniform(-15.0, 15.0, size=3),
                        velocity=rng.uniform(-1.5, 1.5, size=3),
                        yaw=float(rng.uniform(-math.pi, math.pi)),
                        active=active,
                        agent_id=f"drone_{i}",
                    )
                )
            n_static = int(rng.integers(0, 8))
            obstacles = [
                Obstacle(
                    x=float(rng.uniform(-14.0, 14.0)),
                    y=float(rng.uniform(-14.0, 14.0)),
                    radius=float(rng.uniform(0.35, 1.0)),
                    height=5.0,
                )
                for _ in range(n_static)
            ]
            n_dyn = int(rng.integers(0, 5))
            dyn_obstacles = [
                DynamicObstacle(
                    position=rng.uniform(-15.0, 15.0, size=3),
                    velocity=rng.uniform(-2.0, 2.0, size=3),
                    radius=float(rng.uniform(0.25, 0.5)),
                    active=bool(rng.random() > 0.2),
                )
                for _ in range(n_dyn)
            ]
            goal = rng.uniform(-15.0, 15.0, size=3)

            for agent_idx in (0, 4, 9):
                deploy_obs = build_observation(
                    agent_idx, telemetry, goal, obstacles, DEFAULT_CONTRACT, dyn_obstacles
                )
                train_obs = train_contracts.build_observation(
                    agent_idx, telemetry, goal, obstacles, DEFAULT_CONTRACT, dyn_obstacles
                )
                ref_obs = _reference_v2_observation(
                    agent_idx, telemetry, goal, obstacles, dyn_obstacles, DEFAULT_CONTRACT
                )
                self.assertEqual(deploy_obs.shape, (62,))
                self.assertEqual(deploy_obs.dtype, np.float32)
                self.assertTrue(np.array_equal(deploy_obs, train_obs))
                self.assertTrue(np.allclose(deploy_obs, ref_obs, atol=1e-6))
                self.assertTrue(np.all(deploy_obs >= -1.0) and np.all(deploy_obs <= 1.0))

    def test_peer_tie_breaking_by_agent_id_and_missing_peers(self):
        # Drones 2 and 5 are equidistant (3.0 m) from drone_0; drone_2 must precede drone_5.
        # Pass them in reverse list order to confirm sorting uses agent_id, not just list order.
        telemetry = [
            Telemetry(np.array([0.0, 0.0, 2.5]), np.zeros(3), 0.0, agent_id="drone_0"),
            Telemetry(np.array([0.0, 3.0, 2.5]), np.zeros(3), 0.0, agent_id="drone_5"),
            Telemetry(np.array([3.0, 0.0, 2.5]), np.zeros(3), 0.0, agent_id="drone_2"),
            Telemetry(np.array([1.0, 0.0, 2.5]), np.zeros(3), 0.0, active=False, agent_id="drone_1"),
        ]
        obs = build_observation(0, telemetry, np.array([0.0, 0.0, 2.5]), [], DEFAULT_CONTRACT)
        # Slot 0: drone_2 at (3, 0, 2.5)
        self.assertTrue(np.allclose(obs[6:9], [3.0 / 25.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(obs[12]), 1.0)
        # Slot 1: drone_5 at (0, 3, 2.5)
        self.assertTrue(np.allclose(obs[13:16], [0.0, 3.0 / 25.0, 0.0]))
        self.assertAlmostEqual(float(obs[19]), 1.0)
        # Slots 2 and 3: padded with zeros and valid=0
        self.assertTrue(np.allclose(obs[20:34], 0.0))

    def test_action_parity_and_clipping(self):
        rng = np.random.default_rng(7)
        samples = [
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, -1.0, 0.5, -1.0], dtype=np.float32),
            np.array([2.5, -3.0, 1.5, 4.0], dtype=np.float32),  # out-of-range -> clipped
            *[rng.uniform(-1.5, 1.5, size=4).astype(np.float32) for _ in range(25)],
        ]
        for action in samples:
            v_dep, y_dep = action_to_velocity(action, DEFAULT_CONTRACT)
            v_trn, y_trn = train_contracts.action_to_velocity(action, DEFAULT_CONTRACT)
            self.assertTrue(np.allclose(v_dep, v_trn))
            self.assertAlmostEqual(y_dep, y_trn)
            clipped = np.clip(action, -1.0, 1.0)
            self.assertTrue(np.allclose(v_dep, clipped[:3] * 1.5, atol=1e-6))
            self.assertAlmostEqual(y_dep, float(clipped[3] * 0.6), places=6)
            self.assertTrue(np.all(np.abs(v_dep) <= 1.5 + 1e-9))
            self.assertLessEqual(abs(y_dep), 0.6 + 1e-9)

    def test_golden_vectors_v2(self):
        """Hardcoded 62-D observation & 4-D action golden vectors guarding layout drift."""
        telemetry = [
            Telemetry(np.array([1.0, -2.0, 2.5]), np.array([0.75, -0.30, 0.15]), 0.0, agent_id="drone_0"),
            Telemetry(np.array([11.0, -2.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_1"),
            Telemetry(np.array([3.5, -2.0, 2.5]), np.array([1.05, 0.0, 0.15]), 0.0, agent_id="drone_2"),
            Telemetry(np.array([1.0, 3.0, 2.5]), np.array([0.75, -0.30, 0.15]), 0.0, agent_id="drone_3"),
            Telemetry(np.array([1.0, -2.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, active=False, agent_id="drone_4"),
            Telemetry(np.array([-5.0, -2.0, 2.5]), np.array([-0.75, 0.30, -0.15]), 0.0, agent_id="drone_5"),
            Telemetry(np.array([1.0, -9.5, 2.5]), np.array([0.75, 0.60, 0.15]), 0.0, agent_id="drone_6"),
            Telemetry(np.array([15.0, 10.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_7"),
            Telemetry(np.array([-14.0, 10.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_8"),
            Telemetry(np.array([0.0, 15.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_9"),
        ]
        goal = np.array([6.0, 3.0, 3.5])
        obstacles = [
            Obstacle(x=1.0, y=2.0, radius=0.55, height=5.0),   # delta=(0, 4), dist=4.0, clear=4-0.55-0.45=3.0
            Obstacle(x=4.0, y=-2.0, radius=0.55, height=5.0),  # delta=(3, 0), dist=3.0, clear=3-0.55-0.45=2.0 (nearest!)
        ]
        dynamic_obstacles = [
            DynamicObstacle(
                position=np.array([6.0, -2.0, 2.5]),
                velocity=np.array([-0.75, -0.30, 0.15]),
                radius=0.35,
                active=True,
            ),
            DynamicObstacle(
                position=np.array([2.0, -2.0, 2.5]),
                velocity=np.zeros(3),
                radius=0.35,
                active=False,  # inactive -> ignored
            ),
        ]

        obs = build_observation(0, telemetry, goal, obstacles, DEFAULT_CONTRACT, dynamic_obstacles)

        expected_obs = np.array(
            [
                # Self: goal_delta / 25, own_vel / 1.5
                0.20, 0.20, 0.04,
                0.50, -0.20, 0.10,
                # Peer 0: drone_2 (dist=2.5), rel_pos=(2.5,0,0)/25, rel_vel=(0.3,0.3,0)/3, valid=1
                0.10, 0.00, 0.00, 0.10, 0.10, 0.00, 1.0,
                # Peer 1: drone_3 (dist=5.0), rel_pos=(0,5,0)/25, rel_vel=(0,0,0)/3, valid=1
                0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 1.0,
                # Peer 2: drone_5 (dist=6.0), rel_pos=(-6,0,0)/25, rel_vel=(-1.5,0.6,-0.3)/3, valid=1
                -0.24, 0.00, 0.00, -0.50, 0.20, -0.10, 1.0,
                # Peer 3: drone_6 (dist=7.5), rel_pos=(0,-7.5,0)/25, rel_vel=(0,0.9,0)/3, valid=1
                0.00, -0.30, 0.00, 0.00, 0.30, 0.00, 1.0,
                # Static 0: (4, -2), delta=(3,0)/25, clear=2.0/25, valid=1
                0.12, 0.00, 0.08, 1.0,
                # Static 1: (1, 2), delta=(0,4)/25, clear=3.0/25, valid=1
                0.00, 0.16, 0.12, 1.0,
                # Static 2: padded zeros (clearance=0.0, valid=0.0)
                0.00, 0.00, 0.00, 0.0,
                # Dynamic 0: pos=(6,-2,2.5) -> delta=(5,0,0)/25, rel_vel=(-1.5,0,0)/3, valid=1
                0.20, 0.00, 0.00, -0.50, 0.00, 0.00, 1.0,
                # Dynamic 1: padded zeros
                0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0,
                # Heading: sin(0)=0, cos(0)=1
                0.00, 1.00,
            ],
            dtype=np.float32,
        )
        self.assertEqual(expected_obs.shape, (62,))
        self.assertTrue(
            np.allclose(obs, expected_obs, atol=1e-6),
            msg=f"golden observation mismatch:\ngot:      {obs}\nexpected: {expected_obs}",
        )

        # Golden action vector
        golden_action = np.array([0.4, -0.8, 0.2, -0.5], dtype=np.float32)
        vel, yaw_rate = action_to_velocity(golden_action, DEFAULT_CONTRACT)
        self.assertTrue(np.allclose(vel, [0.6, -1.2, 0.3], atol=1e-6))
        self.assertAlmostEqual(yaw_rate, -0.3, places=6)

    def test_shield_config_from_contract_tiered_thresholds(self):
        cfg = ShieldConfig.from_contract(DEFAULT_CONTRACT)
        self.assertEqual(cfg.speed, 1.5)
        self.assertEqual(cfg.accel, 2.0)
        self.assertEqual(cfg.radius, 0.45)
        self.assertEqual(cfg.pair_safe, 0.9)
        self.assertEqual(cfg.obstacle_margin, 0.0)
        # AI soft thresholds must remain strictly larger than shield hard backstops
        self.assertGreater(DEFAULT_CONTRACT.pair_safe, cfg.pair_safe)
        self.assertGreater(DEFAULT_CONTRACT.obs_margin, cfg.obstacle_margin)

        # Verify frozen `dynamic_obstacles` parameter in `safe_velocity`
        own_pos = np.array([0.0, 0.0, 2.5])
        bird = DynamicObstacle(
            position=np.array([0.80, 0.0, 2.5]),  # contact boundary = 0.35 + 0.45 + 0.0 = 0.80
            velocity=np.array([-0.15, 0.0, 0.0]),  # bird moving toward own_pos at 0.15 m/s
            radius=0.35,
            active=True,
        )
        res = safe_velocity(
            own_pos,
            np.zeros(3),
            np.array([1.5, 0.0, 0.0]),
            [],
            [],
            0.1,
            cfg,
            dynamic_obstacles=[bird],
        )
        self.assertTrue(res.feasible)
        # Because bird approaches at -0.15 m/s along +x, drone must retreat (vx <= -0.15)
        self.assertLessEqual(res.velocity[0], -0.15 + 1e-6)

    def test_frozen_ros2_interface_definitions_exist(self):
        interfaces_root = _REPO_ROOT / "src" / "swarm_interfaces"
        dyn_msg = interfaces_root / "msg" / "DynamicObstacle.msg"
        dyn_arr_msg = interfaces_root / "msg" / "DynamicObstacleArray.msg"
        set_form_srv = interfaces_root / "srv" / "SetFormation.srv"
        cmake = interfaces_root / "CMakeLists.txt"

        self.assertTrue(dyn_msg.is_file())
        self.assertTrue(dyn_arr_msg.is_file())
        self.assertTrue(set_form_srv.is_file())
        cmake_text = cmake.read_text()
        self.assertIn("msg/DynamicObstacle.msg", cmake_text)
        self.assertIn("msg/DynamicObstacleArray.msg", cmake_text)
        self.assertIn("srv/SetFormation.srv", cmake_text)

    def test_contract_metadata_validation_and_key_normalization(self):
        meta = DEFAULT_CONTRACT.as_dict()
        self.assertEqual(meta["schema"], SCHEMA)
        self.assertEqual(meta["obs_dim"], 62)
        self.assertEqual(meta["act_dim"], ACT_DIM)
        self.assertEqual(meta["agents"], 10)
        self.assertEqual(meta["k_peer"], 4)
        self.assertEqual(meta["k_static"], 3)
        self.assertEqual(meta["k_dyn"], 2)
        self.assertEqual(meta["peer_order"], PEER_ORDER)
        self.assertEqual(meta["action"], "world_ENU_velocity_1.5mps_yawrate_0.6rps")

        # Applying custom metadata must NOT mutate DEFAULT_CONTRACT
        custom_meta = dict(meta)
        custom_meta["speed"] = 1.2
        custom_meta["accel"] = 1.8
        custom_meta["yaw_rate"] = 0.5
        custom_meta["action"] = "world_ENU_velocity_1.2mps_yawrate_0.5rps"
        custom_meta["hidden"] = [128, 128]
        resolved = apply_contract_metadata(DEFAULT_CONTRACT, custom_meta)
        self.assertEqual(resolved.speed, 1.2)
        self.assertEqual(resolved.accel, 1.8)
        self.assertEqual(resolved.yaw_rate, 0.5)
        self.assertEqual(resolved.hidden, (128, 128))
        self.assertEqual(DEFAULT_CONTRACT.speed, 1.5)
        self.assertEqual(DEFAULT_CONTRACT.hidden, (256, 256))

        # Rejection checks
        for bad_update in (
            {"schema": "swarm-coop-enu-v1"},
            {"obs_dim": 23},
            {"agents": 3},
            {"k_peer": 2},
            {"peer_order": "fixed_agent_ids"},
            {"action": "invalid_action_string"},
            {"action": "world_ENU_velocity_0.8mps_yawrate_0.6rps"},
        ):
            bad_meta = dict(meta)
            bad_meta.update(bad_update)
            with self.assertRaises(ValueError, msg=f"expected rejection for {bad_update}"):
                apply_contract_metadata(DEFAULT_CONTRACT, bad_meta)

        validate_action_spec("world_ENU_velocity_1.50mps_yawrate_0.60rps", 1.5, 0.6)

        # Bare vs net.-prefixed state_dict key normalization + hidden width inference
        class _FakeWeight:
            def __init__(self, shape):
                self.shape = shape

        bare = {
            "0.weight": _FakeWeight((256, 62)),
            "0.bias": _FakeWeight((256,)),
            "2.weight": _FakeWeight((256, 256)),
            "2.bias": _FakeWeight((256,)),
            "4.weight": _FakeWeight((4, 256)),
            "4.bias": _FakeWeight((4,)),
        }
        prefixed = {f"net.{k}": v for k, v in bare.items()}
        stripped = normalize_actor_state_dict(prefixed)
        self.assertEqual(set(stripped.keys()), set(bare.keys()))
        self.assertEqual(_infer_hidden(bare, DEFAULT_CONTRACT), (256, 256))
        self.assertEqual(_infer_hidden(stripped, DEFAULT_CONTRACT), (256, 256))

        payload = make_deploy_payload(
            DEFAULT_CONTRACT,
            prefixed,
            algo="mappo_ctde_shared",
            training_steps=12000,
            seed=7,
            stage="stage1_open",
            note="unit_test",
        )
        self.assertEqual(payload["obs_dim"], 62)
        self.assertEqual(payload["action_dim"], 4)
        self.assertEqual(payload["num_drones"], 10)
        self.assertEqual(set(payload["actor"].keys()), set(bare.keys()))
        self.assertEqual(payload["contract"]["schema"], SCHEMA)

    @unittest.skipUnless(_HAS_TORCH, "PyTorch not installed")
    def test_checkpoint_roundtrip_and_v1_rejection_torch(self):
        import torch

        torch.manual_seed(7)
        actor = build_actor(DEFAULT_CONTRACT)
        actor.eval()
        probe_obs = np.linspace(-0.5, 0.5, 62, dtype=np.float32)
        with torch.no_grad():
            expected_action = actor(torch.as_tensor(probe_obs).reshape(1, 62)).numpy().reshape(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Bare nn.Sequential keys via make_deploy_payload
            bare_path = Path(tmpdir) / "swarm_policy_bare.pt"
            bare_payload = make_deploy_payload(
                DEFAULT_CONTRACT,
                actor.state_dict(),
                training_steps=2400,
                seed=7,
                stage="stage1_open",
                note="parity_test_bare",
            )
            torch.save(bare_payload, bare_path)

            loaded_bare = ActorPolicy(bare_path, actor_index=0)
            self.assertEqual(loaded_bare.contract.schema, "swarm-coop-enu-v2")
            self.assertEqual(loaded_bare.contract.obs_dim, 62)
            self.assertEqual(loaded_bare.contract.agents, 10)
            self.assertEqual(loaded_bare.contract.hidden, (256, 256))
            self.assertEqual(loaded_bare.contract.peer_order, PEER_ORDER)
            out_bare = loaded_bare(probe_obs)
            self.assertEqual(out_bare.shape, (4,))
            self.assertTrue(np.all(np.isfinite(out_bare)))
            self.assertTrue(np.allclose(out_bare, expected_action, atol=1e-6))

            # 2. Wrapped `net.`-prefixed keys must also load strictly
            prefixed_path = Path(tmpdir) / "swarm_policy_prefixed.pt"
            prefixed_payload = dict(bare_payload)
            prefixed_payload["actor"] = {
                f"net.{k}": v for k, v in bare_payload["actor"].items()
            }
            torch.save(prefixed_payload, prefixed_path)
            loaded_prefixed = ActorPolicy(prefixed_path, actor_index=0)
            out_prefixed = loaded_prefixed(probe_obs)
            self.assertTrue(np.allclose(out_prefixed, expected_action, atol=1e-6))

            # 3. Invalid peer_order is rejected
            bad_path = Path(tmpdir) / "bad_peer_order.pt"
            bad_payload = dict(bare_payload)
            bad_payload["contract"] = dict(bare_payload["contract"], peer_order="unknown_order")
            torch.save(bad_payload, bad_path)
            with self.assertRaises(ValueError):
                ActorPolicy(bad_path)

            # 4. Metadata-less checkpoint is rejected unless allow_legacy=True
            no_meta_path = Path(tmpdir) / "no_meta.pt"
            torch.save({"actor": bare_payload["actor"]}, no_meta_path)
            with self.assertRaises(ValueError):
                ActorPolicy(no_meta_path, allow_legacy=False)
            loaded_legacy = ActorPolicy(no_meta_path, allow_legacy=True)
            self.assertTrue(np.allclose(loaded_legacy(probe_obs), expected_action, atol=1e-6))

        # 5. Packaged v1 checkpoint must be rejected by the v2 loader
        v1_ckpt = _REPO_ROOT / "src" / "swarm_bringup" / "checkpoints" / "swarm_macro_policy.pt"
        if v1_ckpt.is_file():
            with self.assertRaises(ValueError):
                ActorPolicy(v1_ckpt)


if __name__ == "__main__":
    unittest.main()
