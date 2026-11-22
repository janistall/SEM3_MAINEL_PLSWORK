import unittest

import numpy as np

from swarm_control.contract import DEFAULT_CONTRACT
from swarm_control.observation import DynamicObstacle, Obstacle, Telemetry, build_observation
from swarm_control.safety import ShieldConfig, safe_velocity
from swarm_control.supervisor import Mode, Supervisor


def test_observation_contract_layout():
    telemetry = [
        Telemetry(np.array([0.0, 0.0, 2.5]), np.array([1.5, 0.0, 0.0]), np.pi / 2, agent_id="drone_0"),
        Telemetry(np.array([10.0, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_1"),
        Telemetry(np.array([1.0, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_2"),
        Telemetry(np.array([0.0, 2.0, 2.5]), np.array([0.0, 1.5, 0.0]), 0.0, agent_id="drone_3"),
        Telemetry(np.array([3.0, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_4"),
        Telemetry(np.array([0.0, 4.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_5"),
        Telemetry(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, active=False, agent_id="drone_6"),
        Telemetry(np.array([6.0, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_7"),
        Telemetry(np.array([7.0, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_8"),
        Telemetry(np.array([8.0, 0.0, 2.5]), np.array([0.0, 0.0, 0.0]), 0.0, agent_id="drone_9"),
    ]
    obs = build_observation(
        0,
        telemetry,
        np.array([25.0, 0.0, 2.5]),
        [Obstacle(3.0, 0.0, 0.5, 5.0)],
        DEFAULT_CONTRACT,
        dynamic_obstacles=[DynamicObstacle(np.array([5.0, 0.0, 2.5]), np.array([-1.5, 0.0, 0.0]), radius=0.35)],
    )
    assert obs.shape == (62,)
    assert obs.dtype == np.float32
    # Self block
    assert np.allclose(obs[0:3], [1.0, 0.0, 0.0])
    assert np.allclose(obs[3:6], [1.0, 0.0, 0.0])
    # Nearest active peer is drone_2 at (1, 0, 2.5); inactive drone_6 at (0.5, 0, 2.5) is skipped
    assert np.allclose(obs[6:9], [1.0 / 25.0, 0.0, 0.0])
    assert np.allclose(obs[9:12], [-0.5, 0.0, 0.0])
    assert np.isclose(obs[12], 1.0)
    # 2nd nearest active peer is drone_3 at (0, 2, 2.5)
    assert np.allclose(obs[13:16], [0.0, 2.0 / 25.0, 0.0])
    assert np.allclose(obs[16:19], [-0.5, 0.5, 0.0])
    assert np.isclose(obs[19], 1.0)
    # Static obstacle 0 at (3, 0), radius 0.5, drone radius 0.45 -> clearance = 2.05
    assert np.allclose(obs[34:36], [3.0 / 25.0, 0.0])
    assert np.isclose(obs[36], (3.0 - 0.5 - 0.45) / 25.0)
    assert np.isclose(obs[37], 1.0)
    # Padded static obstacle slots 1 and 2 are strictly zero (including clearance=0.0)
    assert np.allclose(obs[38:46], 0.0)
    # Dynamic obstacle 0 at (5, 0, 2.5), vel (-1.5, 0, 0) -> rel_vel = (-3.0, 0, 0) / 3.0 = (-1, 0, 0)
    assert np.allclose(obs[46:49], [5.0 / 25.0, 0.0, 0.0])
    assert np.allclose(obs[49:52], [-1.0, 0.0, 0.0])
    assert np.isclose(obs[52], 1.0)
    # Padded dynamic obstacle slot 1 is strictly zero
    assert np.allclose(obs[53:60], 0.0)
    # Heading
    assert np.isclose(obs[60], 1.0)
    assert np.isclose(obs[61], 0.0)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_supervisor_approach_docking_hold_and_recovery():
    supervisor = Supervisor()
    goal = np.array([5.0, 0.0, 2.5])
    position = np.array([0.0, 0.0, 2.5])
    velocity = np.array([0.8, 0.0, 0.0])
    command, _ = supervisor.step(position, velocity, goal, velocity, 0.1)
    assert supervisor.mode == Mode.APPROACH
    assert np.allclose(command, velocity)

    position = np.array([4.0, 0.0, 2.5])
    command, stop = supervisor.step(position, velocity, goal, velocity, 0.1)
    assert supervisor.mode == Mode.DOCKING
    assert stop > 0.0
    assert command[0] < velocity[0]

    position = np.array([5.0, 0.0, 2.5])
    for _ in range(16):
        supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.HOLD

    position = np.array([5.3, 0.0, 2.5])
    supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.DOCKING

    position = np.array([2.0, 0.0, 2.5])
    supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.RECOVERY
    command, _ = supervisor.step(position, np.zeros(3), goal, np.array([0.8, 0.0, 0.0]), 0.1)
    assert np.allclose(command, [0.8, 0.0, 0.0])
    for _ in range(11):
        supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.APPROACH


def test_shield_blocks_head_on_pair_closing():
    own = np.array([0.0, 0.0, 2.5])
    cfg = ShieldConfig.from_contract(DEFAULT_CONTRACT)
    peer = Telemetry(np.array([cfg.pair_safe, 0.0, 2.5]), np.zeros(3), 0.0)
    result = safe_velocity(
        own,
        np.zeros(3),
        np.array([1.5, 0.0, 0.0]),
        [peer],
        [],
        0.1,
        cfg,
    )
    assert result.feasible
    assert result.velocity[0] <= 0.01


def test_shield_requires_separation_when_inside_pair_envelope():
    own = np.array([0.0, 0.0, 2.5])
    cfg = ShieldConfig.from_contract(DEFAULT_CONTRACT)
    peer = Telemetry(np.array([0.5, 0.0, 2.5]), np.zeros(3), 0.0)
    result = safe_velocity(
        own,
        np.zeros(3),
        np.zeros(3),
        [peer],
        [],
        0.1,
        cfg,
    )
    assert not result.feasible
    assert np.linalg.norm(result.velocity) <= 1e-9


def test_shield_avoids_obstacle_cylinder():
    own = np.array([0.0, 0.0, 2.5])
    cfg = ShieldConfig.from_contract(DEFAULT_CONTRACT)
    # Hard contact boundary: obstacle.radius (0.5) + drone radius (0.45) + hard_margin (0.0) = 0.95
    obstacle = Obstacle(0.95, 0.0, 0.5, 5.0)
    result = safe_velocity(
        own,
        np.zeros(3),
        np.array([1.5, 0.0, 0.0]),
        [],
        [obstacle],
        0.1,
        cfg,
    )
    assert result.feasible
    assert result.velocity[0] <= 0.01


class TestControlContract(unittest.TestCase):
    def test_observation_contract_layout(self):
        test_observation_contract_layout()

    def test_supervisor_approach_docking_hold_and_recovery(self):
        test_supervisor_approach_docking_hold_and_recovery()

    def test_shield_blocks_head_on_pair_closing(self):
        test_shield_blocks_head_on_pair_closing()

    def test_shield_requires_separation_when_inside_pair_envelope(self):
        test_shield_requires_separation_when_inside_pair_envelope()

    def test_shield_avoids_obstacle_cylinder(self):
        test_shield_avoids_obstacle_cylinder()
