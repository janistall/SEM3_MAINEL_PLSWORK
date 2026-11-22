"""PettingZoo ParallelEnv for decentralised UAV swarm coordination.

All dimensions / bounds come from ``common.contracts`` (no hardcoded values).
Actions are normalised to [-1, 1] and converted to physical commands with
``scale_action``. Observations follow the 23-dim contract:

    0-2   relative target position (/MAP_EXTENT, clipped)
    3-5   own velocity (/[MAX_XY, MAX_XY, MAX_Z], clipped)
    6-7   sin / cos of yaw
    8     altitude / MAX_ALTITUDE, clipped to [0, 1]
    9-15  nearest neighbour  : rel. pos (/COMMS_RADIUS), rel. vel (/(2*vmax)), valid flag
    16-22 2nd-nearest neighbour (same layout)
Neighbours beyond COMMS_RADIUS (or missing) are padded with explicit zeros.
"""
import functools
import os
import sys

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.contracts import (  # noqa: E402
    ACTION_DIM,
    COMMS_RADIUS,
    MAP_EXTENT,
    MAX_ALTITUDE,
    MAX_XY_VELOCITY,
    MAX_Z_VELOCITY,
    OBS_DIM,
    RawDroneState,
    scale_action,
)

_VEL_SCALE = np.array([MAX_XY_VELOCITY, MAX_XY_VELOCITY, MAX_Z_VELOCITY], dtype=np.float32)
_NUM_NEIGHBOURS = 2
_NEIGH_BLOCK = 7  # rel pos (3) + rel vel (3) + valid flag (1)
assert 9 + _NUM_NEIGHBOURS * _NEIGH_BLOCK == OBS_DIM, "OBS_DIM no longer matches the 23-dim layout"


def build_observation(self_state: RawDroneState, others, target_pos: np.ndarray) -> np.ndarray:
    """Build the OBS_DIM vector for one drone. ``others`` = list of RawDroneState."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    pos, vel = self_state.position, self_state.velocity

    obs[0:3] = np.clip((target_pos - pos) / MAP_EXTENT, -1.0, 1.0)
    obs[3:6] = np.clip(vel / _VEL_SCALE, -1.0, 1.0)
    obs[6] = np.sin(self_state.yaw)
    obs[7] = np.cos(self_state.yaw)
    obs[8] = np.clip(pos[2] / MAX_ALTITUDE, 0.0, 1.0)

    active = [o for o in others if o.is_active]
    active.sort(key=lambda o: float(np.linalg.norm(o.position - pos)))
    for k, n in enumerate(active[:_NUM_NEIGHBOURS]):
        if np.linalg.norm(n.position - pos) > COMMS_RADIUS:
            break  # sorted: every later neighbour is also out of range -> stay zero
        base = 9 + k * _NEIGH_BLOCK
        obs[base:base + 3] = np.clip((n.position - pos) / COMMS_RADIUS, -1.0, 1.0)
        obs[base + 3:base + 6] = np.clip((n.velocity - vel) / (2.0 * _VEL_SCALE), -1.0, 1.0)
        obs[base + 6] = 1.0
    return obs


class DroneSwarmEnv(ParallelEnv):
    """Cooperative swarm: reach a shared target while avoiding collisions.

    Simplified point-mass kinematics (commanded velocity is applied with a
    first-order lag). Agents never terminate early, only truncate at
    ``max_steps``, so every episode has all drones for the full length.
    """

    metadata = {"name": "drone_swarm_v0", "render_modes": []}

    def __init__(self, num_drones=3, max_steps=500, dt=0.1, collision_dist=1.0, vel_lag=0.5,
                 collision_penalty=50.0, boundary_penalty=10.0):
        self.num_drones = num_drones
        self.max_steps = max_steps
        self.dt = dt
        self.collision_dist = collision_dist
        self.vel_lag = vel_lag
        self.collision_penalty = collision_penalty
        self.boundary_penalty = boundary_penalty
        self.possible_agents = [f"drone_{i}" for i in range(1, self.num_drones + 1)]
        self.agents = []
        self._rng = np.random.default_rng()
        self._states = {}
        self._target = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._prev_dist = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _observe(self):
        obs = {}
        for a in self.agents:
            others = [self._states[b] for b in self.agents if b != a]
            obs[a] = build_observation(self._states[a], others, self._target)
        return obs

    def _dist_to_target(self, a):
        return float(np.linalg.norm(self._target - self._states[a].position))

    # ---------------------------------------------------------------- PettingZoo
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0

        spawn = 0.2 * MAP_EXTENT
        self._states = {}
        for a in self.agents:
            p = self._rng.uniform(-spawn, spawn, size=3).astype(np.float32)
            p[2] = self._rng.uniform(0.2 * MAX_ALTITUDE, 0.8 * MAX_ALTITUDE)
            self._states[a] = RawDroneState(
                drone_id=a,
                position=p,
                velocity=np.zeros(3, dtype=np.float32),
                yaw=float(self._rng.uniform(-np.pi, np.pi)),
            )
        t_ext = 0.4 * MAP_EXTENT
        self._target = self._rng.uniform(-t_ext, t_ext, size=3).astype(np.float32)
        self._target[2] = self._rng.uniform(0.2 * MAX_ALTITUDE, 0.8 * MAX_ALTITUDE)
        self._prev_dist = {a: self._dist_to_target(a) for a in self.agents}

        return self._observe(), {a: {} for a in self.agents}

    def step(self, actions):
        self._step_count += 1
        for a in self.agents:
            cmd = scale_action(np.asarray(actions[a], dtype=np.float32))  # [vx, vy, vz, yaw_rate]
            s = self._states[a]
            s.velocity = (s.velocity + self.vel_lag * (cmd[:3] - s.velocity)).astype(np.float32)
            s.position = (s.position + s.velocity * self.dt).astype(np.float32)
            s.position[2] = np.clip(s.position[2], 0.0, MAX_ALTITUDE)
            s.yaw = float((s.yaw + cmd[3] * self.dt + np.pi) % (2 * np.pi) - np.pi)

        rewards = {}
        for a in self.agents:
            s = self._states[a]
            d = self._dist_to_target(a)
            r = (self._prev_dist[a] - d)  # progress towards target (m/step)
            self._prev_dist[a] = d
            if d < 1.0:
                r += 0.1  # on-target bonus
            for b in self.agents:
                if b != a and np.linalg.norm(self._states[b].position - s.position) < self.collision_dist:
                    r -= self.collision_penalty  # collision penalty
            if abs(s.position[0]) > MAP_EXTENT or abs(s.position[1]) > MAP_EXTENT:
                r -= self.boundary_penalty  # left the map
            rewards[a] = float(r)

        truncated = self._step_count >= self.max_steps
        obs = self._observe()
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {} for a in self.agents}
        if truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self):
        pass

    def close(self):
        pass
