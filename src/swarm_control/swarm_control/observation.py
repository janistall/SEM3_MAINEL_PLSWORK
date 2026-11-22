from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contract import Contract, DEFAULT_CONTRACT, normalize_observation


@dataclass
class Telemetry:
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    stamp: float = 0.0
    active: bool = True
    agent_id: str = ""

    @classmethod
    def zero(cls, agent_id: str = "") -> "Telemetry":
        return cls(np.zeros(3), np.zeros(3), 0.0, 0.0, False, agent_id)


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    height: float = 4.0


@dataclass
class DynamicObstacle:
    position: np.ndarray
    velocity: np.ndarray
    radius: float = 0.35
    active: bool = True


def build_observation(
    agent_index: int,
    telemetry: Sequence[Telemetry],
    goal: np.ndarray,
    obstacles: Sequence[Obstacle] = (),
    contract: Contract = DEFAULT_CONTRACT,
    dynamic_obstacles: Sequence[DynamicObstacle] = (),
) -> np.ndarray:
    """Build the Contract v2 (`swarm-coop-enu-v2`) observation vector.

    Layout (all entries clipped to [-1, 1] when `contract.clip_obs` is True):
      * `[0:3]`                         : `(goal - own_pos) / goal_scale`
      * `[3:6]`                         : `own_vel / vel_scale`
      * `k_peer × 7`                    : nearest active peers sorted by `(dist_3d, agent_id, idx)`:
                                          `rel_pos(3) / goal_scale`, `rel_vel(3) / peer_vel_scale`, `valid(1)`
      * `k_static × 4`                  : nearest static obstacles sorted by `(dist_2d - radius, idx)`:
                                          `rel_pos_xy(2) / obstacle_scale`,
                                          `(dist_2d - o.radius - contract.radius) / obstacle_scale`, `valid(1)`
      * `k_dyn × 7`                     : nearest active dynamic obstacles sorted by `(dist_3d - radius, idx)`:
                                          `rel_pos(3) / obstacle_scale`, `rel_vel(3) / peer_vel_scale`, `valid(1)`
      * `[-2:]`                         : `sin(yaw)`, `cos(yaw)`

    Missing or inactive peer / static / dynamic slots are padded with zeros
    (including `clearance = 0.0` and `valid = 0.0`).
    """
    if not 0 <= agent_index < len(telemetry):
        raise IndexError(
            f"agent_index {agent_index} out of range for telemetry of length {len(telemetry)}"
        )

    own = telemetry[agent_index]
    own_pos = np.asarray(own.position, dtype=np.float64).reshape(3)
    own_vel = np.asarray(own.velocity, dtype=np.float64).reshape(3)
    goal_pos = np.asarray(goal, dtype=np.float64).reshape(3)

    obs = np.zeros(contract.obs_dim, dtype=np.float32)
    obs[0:3] = (goal_pos - own_pos) / contract.goal_scale
    obs[3:6] = own_vel / contract.vel_scale

    # Nearest k_peer active peers, distance-sorted with deterministic agent_id tie-break.
    active_peers = [
        (
            idx,
            peer,
            np.asarray(peer.position, dtype=np.float64).reshape(3),
            np.asarray(peer.velocity, dtype=np.float64).reshape(3),
        )
        for idx, peer in enumerate(telemetry)
        if idx != agent_index and peer.active
    ]
    active_peers.sort(
        key=lambda item: (float(np.linalg.norm(item[2] - own_pos)), item[1].agent_id, item[0])
    )
    for k, (_, _, peer_pos, peer_vel) in enumerate(active_peers[: contract.k_peer]):
        base = 6 + 7 * k
        obs[base : base + 3] = (peer_pos - own_pos) / contract.goal_scale
        obs[base + 3 : base + 6] = (peer_vel - own_vel) / contract.peer_vel_scale
        obs[base + 6] = 1.0

    # Nearest k_static cylindrical obstacles by edge clearance.
    static_offset = 6 + 7 * contract.k_peer
    ranked_static = sorted(
        enumerate(obstacles),
        key=lambda item: (
            float(
                np.linalg.norm(
                    np.array([item[1].x, item[1].y], dtype=np.float64) - own_pos[:2]
                )
            )
            - float(item[1].radius),
            item[0],
        ),
    )
    for k, (_, obstacle) in enumerate(ranked_static[: contract.k_static]):
        base = static_offset + 4 * k
        delta = np.array([obstacle.x, obstacle.y], dtype=np.float64) - own_pos[:2]
        distance = float(np.linalg.norm(delta))
        obs[base : base + 2] = delta / contract.obstacle_scale
        obs[base + 2] = (
            distance - float(obstacle.radius) - contract.radius
        ) / contract.obstacle_scale
        obs[base + 3] = 1.0

    # Nearest k_dyn active dynamic obstacles ("birds") by 3-D surface distance.
    dyn_offset = static_offset + 4 * contract.k_static
    active_dyn = [
        (
            idx,
            dyn,
            np.asarray(dyn.position, dtype=np.float64).reshape(3),
            np.asarray(dyn.velocity, dtype=np.float64).reshape(3),
        )
        for idx, dyn in enumerate(dynamic_obstacles)
        if dyn.active
    ]
    active_dyn.sort(
        key=lambda item: (
            float(np.linalg.norm(item[2] - own_pos)) - float(item[1].radius),
            item[0],
        )
    )
    for k, (_, _, dyn_pos, dyn_vel) in enumerate(active_dyn[: contract.k_dyn]):
        base = dyn_offset + 7 * k
        obs[base : base + 3] = (dyn_pos - own_pos) / contract.obstacle_scale
        obs[base + 3 : base + 6] = (dyn_vel - own_vel) / contract.peer_vel_scale
        obs[base + 6] = 1.0

    obs[-2] = np.sin(own.yaw)
    obs[-1] = np.cos(own.yaw)
    return normalize_observation(obs, contract)
