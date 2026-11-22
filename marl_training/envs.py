"""PettingZoo ParallelEnv for 10-drone UAV swarm coordination on Contract v2 (`swarm-coop-enu-v2`).

Calls deployment primitives from `src/swarm_control` directly via `marl_training.contracts`:
  * 62-D observation via `build_observation` (`k_peer=4`, `k_static=3`, `k_dyn=2`)
  * 4-D action decoding via `action_to_velocity` (`±1.5 m/s`, `±0.6 rad/s`)
  * Safety filtering via `safe_velocity(..., dynamic_obstacles=...)` and `ShieldConfig.from_contract`
"""
from __future__ import annotations

import functools
import json
import math
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from marl_training.contracts import (
    Contract,
    DEFAULT_CONTRACT,
    DynamicObstacle,
    Obstacle,
    ShieldConfig,
    Telemetry,
    action_to_velocity,
    build_observation,
    safe_velocity,
)

try:
    from gymnasium import spaces as _gym_spaces
except Exception:  # pragma: no cover - pure-NumPy fallback if gymnasium is absent
    _gym_spaces = None

try:
    from pettingzoo import ParallelEnv as _ParallelEnvBase
except Exception:  # pragma: no cover - pure-NumPy fallback if pettingzoo is absent
    class _ParallelEnvBase:  # type: ignore[no-redef]
        metadata: dict[str, Any] = {}


class _FallbackBox:
    """Lightweight gymnasium.spaces.Box replacement when gymnasium is not installed."""

    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.low = np.full(self.shape, low, dtype=self.dtype)
        self.high = np.full(self.shape, high, dtype=self.dtype)
        self._rng = np.random.default_rng()

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return (
            arr.shape == self.shape
            and bool(np.all(arr >= self.low - 1e-6))
            and bool(np.all(arr <= self.high + 1e-6))
        )


def _make_box(low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
    if _gym_spaces is not None:
        return _gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _FallbackBox(low=low, high=high, shape=shape, dtype=dtype)


FORMATION_SHAPES: tuple[str, ...] = ("ring", "grid", "v")

STAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "stage1_open": {
        "num_static_range": (0, 0),
        "num_dyn_range": (0, 0),
        "goal_change_prob": 0.0,
        "domain_rand": False,
    },
    "stage2_static": {
        "num_static_range": (3, 10),
        "num_dyn_range": (0, 0),
        "goal_change_prob": 0.0,
        "domain_rand": False,
    },
    "stage3_dynamic": {
        "num_static_range": (5, 12),
        "num_dyn_range": (1, 4),
        "goal_change_prob": 0.10,
        "domain_rand": False,
    },
    "stage4_domain_rand": {
        "num_static_range": (3, 15),
        "num_dyn_range": (0, 4),
        "goal_change_prob": 0.10,
        "domain_rand": True,
    },
}


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def generate_formation_slots(
    num_drones: int = 10,
    shape: str = "ring",
    spacing: float = 1.6,
    heading: float = 0.0,
) -> np.ndarray:
    """Return `(num_drones, 3)` zero-centered XY formation offsets with pairwise distance >= spacing."""
    n = int(num_drones)
    if n < 1:
        raise ValueError(f"num_drones must be >= 1, got {n}")
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    shape_key = shape.strip().lower()
    if shape_key not in FORMATION_SHAPES:
        raise ValueError(f"unsupported formation shape {shape!r}; expected one of {FORMATION_SHAPES}")

    if n == 1:
        return np.zeros((1, 3), dtype=np.float64)

    pts = np.zeros((n, 3), dtype=np.float64)
    if shape_key == "ring":
        radius = spacing / (2.0 * math.sin(math.pi / n))
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            pts[i, 0] = radius * math.cos(theta)
            pts[i, 1] = radius * math.sin(theta)
    elif shape_key == "grid":
        cols = int(math.ceil(math.sqrt(n)))
        for i in range(n):
            r = i // cols
            c = i % cols
            pts[i, 0] = float(r) * spacing
            pts[i, 1] = float(c) * spacing
        pts[:, :2] -= np.mean(pts[:, :2], axis=0, keepdims=True)
    elif shape_key == "v":
        # Half-angle 35 deg guarantees cross-wing distance 2*sin(35 deg)*spacing = 1.147*spacing >= spacing.
        half_angle = math.radians(35.0)
        for i in range(1, n):
            rank = (i + 1) // 2
            side = 1.0 if (i % 2 == 1) else -1.0
            pts[i, 0] = -float(rank) * spacing * math.cos(half_angle)
            pts[i, 1] = side * float(rank) * spacing * math.sin(half_angle)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0, keepdims=True)

    cos_h = math.cos(float(heading))
    sin_h = math.sin(float(heading))
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    pts[:, :2] = pts[:, :2] @ rot.T
    return pts


class DroneSwarmEnv(_ParallelEnvBase):
    """10-drone Contract v2 (`swarm-coop-enu-v2`) cooperative training environment."""

    metadata = {"name": "drone_swarm_enu_v2", "render_modes": []}

    def __init__(
        self,
        num_drones: int = 10,
        max_steps: int = 600,
        dt: float = 0.1,
        scenario_path: str | Path | None = None,
        shield: bool = True,
        shield_off_prob: float = 0.0,
        randomize: bool = True,
        num_static_range: tuple[int, int] | None = None,
        num_dyn_range: tuple[int, int] | None = None,
        goal_change_prob: float | None = None,
        domain_rand: bool | None = None,
        lag_tau: float = 0.2,
        success_radius: float = 0.35,
        hold_speed: float = 0.12,
        hold_dwell: int = 10,
        contract: Contract = DEFAULT_CONTRACT,
        seed: int | None = None,
        stage: str | None = None,
        wind_max: float = 0.4,
        odom_pos_noise_std: float = 0.03,
        odom_vel_noise_std: float = 0.04,
        dropout_range: tuple[float, float] = (0.0, 0.30),
        max_latency_steps: int = 2,
    ):
        if stage is not None:
            if stage not in STAGE_CONFIGS:
                raise ValueError(
                    f"unknown curriculum stage {stage!r}; expected one of {tuple(STAGE_CONFIGS)}"
                )
            preset = STAGE_CONFIGS[stage]
        else:
            preset = {}

        self.stage = stage
        self.num_drones = int(num_drones)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.scenario_path = Path(scenario_path).expanduser() if scenario_path else None
        self.shield = bool(shield)
        self.shield_off_prob = float(np.clip(shield_off_prob, 0.0, 1.0))
        self._ep_shield = self.shield
        self.randomize = bool(randomize)

        resolved_static = (
            num_static_range
            if num_static_range is not None
            else preset.get("num_static_range", (0, 10))
        )
        resolved_dyn = (
            num_dyn_range
            if num_dyn_range is not None
            else preset.get("num_dyn_range", (0, 0))
        )
        self.num_static_range = (int(resolved_static[0]), int(resolved_static[1]))
        self.num_dyn_range = (int(resolved_dyn[0]), int(resolved_dyn[1]))
        self.goal_change_prob = float(
            goal_change_prob
            if goal_change_prob is not None
            else preset.get("goal_change_prob", 0.0)
        )
        self.domain_rand = bool(
            domain_rand if domain_rand is not None else preset.get("domain_rand", False)
        )

        self.lag_tau = float(lag_tau)
        self.success_radius = float(success_radius)
        self.hold_speed = float(hold_speed)
        self.hold_dwell = int(hold_dwell)
        self.contract = replace(contract, agents=self.num_drones)
        self.shield_cfg = ShieldConfig.from_contract(self.contract)

        self.wind_max = float(wind_max)
        self.odom_pos_noise_std = float(odom_pos_noise_std)
        self.odom_vel_noise_std = float(odom_vel_noise_std)
        self.dropout_range = (float(dropout_range[0]), float(dropout_range[1]))
        self.max_latency_steps = int(max(0, max_latency_steps))

        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.agents: list[str] = []
        self._rng = np.random.default_rng(seed)

        # Episode state
        self.telemetry: list[Telemetry] = []
        self.starts = np.zeros((self.num_drones, 3), dtype=np.float64)
        self.goals = np.zeros((self.num_drones, 3), dtype=np.float64)
        self.obstacles: list[Obstacle] = []
        self.dyn_obstacles: list[DynamicObstacle] = []
        self._step_count = 0
        self._hold_counter = 0
        self._prev_dist = np.zeros(self.num_drones, dtype=np.float64)
        self._prev_actions = np.zeros((self.num_drones, self.contract.act_dim), dtype=np.float64)
        self._goal_shift_step: int | None = None
        self._goal_shifted = False

        # Per-episode domain randomisation state
        self._ep_lag_tau = self.lag_tau
        self._ep_wind_bias = np.zeros(3, dtype=np.float64)
        self._ep_dropout_prob = 0.0
        self._telemetry_history: deque[list[Telemetry]] = deque(
            maxlen=self.max_latency_steps + 1
        )

    @classmethod
    def for_stage(cls, stage: str, **overrides: Any) -> "DroneSwarmEnv":
        return cls(stage=stage, **overrides)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return _make_box(-1.0, 1.0, shape=(self.contract.obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return _make_box(-1.0, 1.0, shape=(self.contract.act_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ layout helpers
    def _arena_xy_limit(self) -> float:
        wx = self.shield_cfg.workspace[0]
        wy = self.shield_cfg.workspace[1]
        return float(min(abs(wx[0]), abs(wx[1]), abs(wy[0]), abs(wy[1]))) - 1.5

    def _sample_formation_centers(self, slot_radius: float) -> tuple[np.ndarray, np.ndarray, float]:
        limit = max(4.0, self._arena_xy_limit() - slot_radius)
        goal_z = (
            float(self._rng.uniform(1.6, 3.2))
            if self.randomize
            else float(self.contract.alt)
        )
        for _ in range(200):
            start_xy = self._rng.uniform(-limit, limit, size=2)
            goal_xy = self._rng.uniform(-limit, limit, size=2)
            if float(np.linalg.norm(goal_xy - start_xy)) >= 8.0:
                return (
                    np.array([start_xy[0], start_xy[1], goal_z], dtype=np.float64),
                    np.array([goal_xy[0], goal_xy[1], goal_z], dtype=np.float64),
                    goal_z,
                )
        return (
            np.array([-6.0, -5.0, goal_z], dtype=np.float64),
            np.array([6.0, 5.0, goal_z], dtype=np.float64),
            goal_z,
        )

    def _sample_static_obstacles(
        self, starts: np.ndarray, goals: np.ndarray, count: int
    ) -> list[Obstacle]:
        obstacles: list[Obstacle] = []
        if count <= 0:
            return obstacles
        xy_bound = self._arena_xy_limit()
        slot_keepout = self.contract.radius + self.contract.obs_margin + 0.65
        for _ in range(800):
            if len(obstacles) >= count:
                break
            xy = self._rng.uniform(-xy_bound, xy_bound, size=2)
            radius = float(self._rng.uniform(0.40, 0.85))
            d_start = float(np.min(np.linalg.norm(starts[:, :2] - xy, axis=1)))
            d_goal = float(np.min(np.linalg.norm(goals[:, :2] - xy, axis=1)))
            if min(d_start, d_goal) < radius + slot_keepout:
                continue
            if any(
                float(np.linalg.norm(xy - np.array([o.x, o.y]))) < radius + o.radius + 0.55
                for o in obstacles
            ):
                continue
            obstacles.append(Obstacle(float(xy[0]), float(xy[1]), radius, 5.0))
        return obstacles

    def _sample_dynamic_obstacles(
        self, starts: np.ndarray, goal_z: float, count: int
    ) -> list[DynamicObstacle]:
        birds: list[DynamicObstacle] = []
        if count <= 0:
            return birds
        xy_bound = self._arena_xy_limit()
        for _ in range(400):
            if len(birds) >= count:
                break
            pos_xy = self._rng.uniform(-xy_bound, xy_bound, size=2)
            if float(np.min(np.linalg.norm(starts[:, :2] - pos_xy, axis=1))) < 2.5:
                continue
            z = float(np.clip(goal_z + self._rng.uniform(-0.35, 0.35), 1.0, 3.5))
            speed = float(self._rng.uniform(0.6, 1.35))
            heading = float(self._rng.uniform(-math.pi, math.pi))
            vz = float(self._rng.uniform(-0.1, 0.1))
            vel = np.array(
                [speed * math.cos(heading), speed * math.sin(heading), vz],
                dtype=np.float64,
            )
            radius = float(self._rng.uniform(0.30, 0.42))
            birds.append(
                DynamicObstacle(
                    position=np.array([pos_xy[0], pos_xy[1], z], dtype=np.float64),
                    velocity=vel,
                    radius=radius,
                    active=True,
                )
            )
        return birds

    def _load_from_scenario(self, path: Path) -> tuple[np.ndarray, np.ndarray, list[Obstacle]]:
        payload = json.loads(path.read_text())
        raw_starts = np.asarray(payload["starts"], dtype=np.float64)
        raw_goals = np.asarray(payload["goals"], dtype=np.float64)
        obstacles = [
            Obstacle(float(x), float(y), float(r), float(h))
            for x, y, r, h in payload.get("obstacles", [])
        ]
        goal_z = float(np.mean(raw_goals[:, 2])) if len(raw_goals) else float(self.contract.alt)
        if len(raw_starts) == self.num_drones and len(raw_goals) == self.num_drones:
            starts = raw_starts.copy()
            goals = raw_goals.copy()
            starts[:, 2] = goal_z
            return starts, goals, obstacles

        # Compatibility fallback if scenario.json still has 3 slots before Step 2 regeneration
        start_center = np.array(
            [float(np.mean(raw_starts[:, 0])), float(np.mean(raw_starts[:, 1])), goal_z],
            dtype=np.float64,
        )
        goal_center = np.array(
            [float(np.mean(raw_goals[:, 0])), float(np.mean(raw_goals[:, 1])), goal_z],
            dtype=np.float64,
        )
        offsets = generate_formation_slots(self.num_drones, shape="ring", spacing=1.6, heading=0.0)
        starts = start_center + offsets
        goals = goal_center + offsets
        # Filter any obstacle that overlaps the expanded 10-drone start or goal ring
        keepout = self.contract.radius + self.contract.obs_margin + 0.35
        filtered = [
            o
            for o in obstacles
            if min(
                float(np.min(np.linalg.norm(starts[:, :2] - np.array([o.x, o.y]), axis=1))),
                float(np.min(np.linalg.norm(goals[:, :2] - np.array([o.x, o.y]), axis=1))),
            )
            >= o.radius + keepout
        ]
        return starts, goals, filtered

    def _maybe_shift_goals(self) -> bool:
        if (
            self._goal_shifted
            or self._goal_shift_step is None
            or self._step_count != self._goal_shift_step
        ):
            return False
        shape = str(self._rng.choice(FORMATION_SHAPES))
        spacing = float(self._rng.uniform(1.5, 2.0))
        heading = float(self._rng.uniform(-math.pi, math.pi))
        offsets = generate_formation_slots(self.num_drones, shape=shape, spacing=spacing, heading=heading)
        slot_radius = float(np.max(np.linalg.norm(offsets[:, :2], axis=1)))
        limit = max(3.5, self._arena_xy_limit() - slot_radius)
        goal_z = float(np.mean(self.goals[:, 2]))
        keepout = self.contract.radius + self.contract.obs_margin + 0.55

        for _ in range(200):
            center_xy = self._rng.uniform(-limit, limit, size=2)
            candidate = offsets.copy()
            candidate[:, 0] += center_xy[0]
            candidate[:, 1] += center_xy[1]
            candidate[:, 2] = goal_z
            if all(
                float(np.min(np.linalg.norm(candidate[:, :2] - np.array([o.x, o.y]), axis=1)))
                >= o.radius + keepout
                for o in self.obstacles
            ):
                self.goals = candidate
                self._goal_shifted = True
                for i in range(self.num_drones):
                    self._prev_dist[i] = float(
                        np.linalg.norm(self.goals[i] - self.telemetry[i].position)
                    )
                return True
        return False

    # ------------------------------------------------------------------ observation builder
    def _build_agent_telemetry_view(self, agent_index: int) -> list[Telemetry]:
        if not self.domain_rand:
            return self.telemetry

        history_len = len(self._telemetry_history)
        view: list[Telemetry] = []
        for j in range(self.num_drones):
            if j == agent_index:
                src = self.telemetry[j]
                active = True
            else:
                delay = (
                    int(self._rng.integers(0, history_len))
                    if history_len > 1
                    else 0
                )
                snapshot = self._telemetry_history[-1 - delay]
                src = snapshot[j]
                active = bool(self._rng.random() >= self._ep_dropout_prob)

            noisy_pos = src.position + self._rng.normal(0.0, self.odom_pos_noise_std, size=3)
            noisy_vel = src.velocity + self._rng.normal(0.0, self.odom_vel_noise_std, size=3)
            noisy_yaw = _wrap_angle(src.yaw + float(self._rng.normal(0.0, 0.02)))
            view.append(
                Telemetry(
                    position=noisy_pos,
                    velocity=noisy_vel,
                    yaw=noisy_yaw,
                    stamp=src.stamp,
                    active=active,
                    agent_id=src.agent_id,
                )
            )
        return view

    def _observe(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        for i, agent_id in enumerate(self.possible_agents):
            tel_view = self._build_agent_telemetry_view(i)
            obs[agent_id] = build_observation(
                i,
                tel_view,
                self.goals[i],
                self.obstacles,
                self.contract,
                dynamic_obstacles=self.dyn_obstacles,
            )
        return obs

    # ------------------------------------------------------------------ PettingZoo API
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            for agent_id in self.possible_agents:
                space = self.action_space(agent_id)
                if hasattr(space, "seed"):
                    space.seed(int(self._rng.integers(0, 2**31 - 1)))

        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._hold_counter = 0
        self._prev_actions.fill(0.0)
        self._goal_shifted = False
        self._ep_shield = bool(
            self.shield
            and (
                self.shield_off_prob <= 0.0
                or float(self._rng.random()) >= self.shield_off_prob
            )
        )

        if self.scenario_path is not None and not self.randomize:
            self.starts, self.goals, self.obstacles = self._load_from_scenario(self.scenario_path)
            goal_z = float(np.mean(self.goals[:, 2]))
            n_dyn = (
                int(self._rng.integers(self.num_dyn_range[0], self.num_dyn_range[1] + 1))
                if self.num_dyn_range[1] > 0
                else 0
            )
            self.dyn_obstacles = self._sample_dynamic_obstacles(self.starts, goal_z, n_dyn)
        else:
            start_shape = (
                str(self._rng.choice(FORMATION_SHAPES)) if self.randomize else "ring"
            )
            goal_shape = (
                str(self._rng.choice(FORMATION_SHAPES)) if self.randomize else "ring"
            )
            start_spacing = float(self._rng.uniform(1.5, 2.1)) if self.randomize else 1.6
            goal_spacing = float(self._rng.uniform(1.5, 2.1)) if self.randomize else 1.6
            start_heading = (
                float(self._rng.uniform(-math.pi, math.pi)) if self.randomize else 0.0
            )
            goal_heading = (
                float(self._rng.uniform(-math.pi, math.pi)) if self.randomize else 0.0
            )

            start_offsets = generate_formation_slots(
                self.num_drones, shape=start_shape, spacing=start_spacing, heading=start_heading
            )
            goal_offsets = generate_formation_slots(
                self.num_drones, shape=goal_shape, spacing=goal_spacing, heading=goal_heading
            )
            max_radius = max(
                float(np.max(np.linalg.norm(start_offsets[:, :2], axis=1))),
                float(np.max(np.linalg.norm(goal_offsets[:, :2], axis=1))),
            )
            start_center, goal_center, goal_z = self._sample_formation_centers(max_radius)
            self.starts = start_center + start_offsets
            self.goals = goal_center + goal_offsets

            if self.randomize:
                jitter = self._rng.uniform(-0.10, 0.10, size=(self.num_drones, 2))
                self.starts[:, :2] += jitter

            n_static = int(
                self._rng.integers(self.num_static_range[0], self.num_static_range[1] + 1)
            )
            n_dyn = int(
                self._rng.integers(self.num_dyn_range[0], self.num_dyn_range[1] + 1)
            )
            self.obstacles = self._sample_static_obstacles(self.starts, self.goals, n_static)
            self.dyn_obstacles = self._sample_dynamic_obstacles(self.starts, goal_z, n_dyn)

        if self.goal_change_prob > 0.0 and float(self._rng.random()) < self.goal_change_prob:
            lo = max(10, int(0.25 * self.max_steps))
            hi = max(lo + 1, int(0.60 * self.max_steps))
            self._goal_shift_step = int(self._rng.integers(lo, hi))
        else:
            self._goal_shift_step = None

        if self.domain_rand:
            self._ep_lag_tau = float(self._rng.uniform(0.10, 0.30))
            wind_dir = self._rng.normal(0.0, 1.0, size=3)
            wind_dir[2] *= 0.25
            norm = float(np.linalg.norm(wind_dir))
            wind_mag = float(self._rng.uniform(0.0, 0.7 * self.wind_max))
            self._ep_wind_bias = (wind_dir / max(norm, 1e-9)) * wind_mag
            self._ep_dropout_prob = float(
                self._rng.uniform(self.dropout_range[0], self.dropout_range[1])
            )
        else:
            self._ep_lag_tau = self.lag_tau
            self._ep_wind_bias = np.zeros(3, dtype=np.float64)
            self._ep_dropout_prob = 0.0

        self.telemetry = []
        for i, agent_id in enumerate(self.possible_agents):
            yaw = (
                float(self._rng.uniform(-math.pi, math.pi))
                if self.randomize
                else 0.0
            )
            pos = self.starts[i].copy()
            if self.domain_rand:
                pos[2] = float(
                    np.clip(
                        pos[2] + self._rng.uniform(-0.12, 0.12),
                        self.shield_cfg.workspace[2][0] + 0.2,
                        self.shield_cfg.workspace[2][1] - 0.2,
                    )
                )
            self.telemetry.append(
                Telemetry(
                    position=pos,
                    velocity=np.zeros(3, dtype=np.float64),
                    yaw=yaw,
                    stamp=0.0,
                    active=True,
                    agent_id=agent_id,
                )
            )
            self._prev_dist[i] = float(np.linalg.norm(self.goals[i] - pos))

        self._telemetry_history.clear()
        self._telemetry_history.append(
            [replace(t, position=t.position.copy(), velocity=t.velocity.copy()) for t in self.telemetry]
        )

        obs = self._observe()
        infos = {agent_id: {"stage": self.stage or "custom"} for agent_id in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        self._step_count += 1
        stamp = self._step_count * self.dt

        # 1. Advance dynamic obstacles ("birds") with bounce inside arena bounds
        xy_limit = self._arena_xy_limit() + 1.0
        z_low, z_high = self.shield_cfg.workspace[2]
        for dyn in self.dyn_obstacles:
            if not dyn.active:
                continue
            dyn.position = np.asarray(dyn.position, dtype=np.float64) + np.asarray(dyn.velocity, dtype=np.float64) * self.dt
            for axis in (0, 1):
                if dyn.position[axis] < -xy_limit or dyn.position[axis] > xy_limit:
                    dyn.position[axis] = float(np.clip(dyn.position[axis], -xy_limit, xy_limit))
                    dyn.velocity[axis] = -dyn.velocity[axis]
            if dyn.position[2] < z_low + 0.3 or dyn.position[2] > z_high - 0.3:
                dyn.position[2] = float(np.clip(dyn.position[2], z_low + 0.3, z_high - 0.3))
                dyn.velocity[2] = -dyn.velocity[2]

        # 2. Optional mid-episode target formation shift
        goal_shifted = self._maybe_shift_goals()

        # 3. Compute desired velocities, shield commands, and step drone kinematics
        snapshot = [
            replace(t, position=t.position.copy(), velocity=t.velocity.copy())
            for t in self.telemetry
        ]
        interventions = np.zeros(self.num_drones, dtype=np.float64)
        feasibles = np.ones(self.num_drones, dtype=bool)
        action_jerks = np.zeros(self.num_drones, dtype=np.float64)
        step_paths = np.zeros(self.num_drones, dtype=np.float64)

        alpha = min(1.0, self.dt / max(self._ep_lag_tau, 1e-6))
        max_dv = self.contract.accel * self.dt

        for i, agent_id in enumerate(self.possible_agents):
            raw_act = np.asarray(actions[agent_id], dtype=np.float64).reshape(self.contract.act_dim)
            clipped_act = np.clip(raw_act, -1.0, 1.0)
            action_jerks[i] = float(np.linalg.norm(clipped_act - self._prev_actions[i]))
            self._prev_actions[i] = clipped_act

            desired, yaw_rate = action_to_velocity(clipped_act, self.contract)
            own = snapshot[i]
            peers = [snapshot[j] for j in range(self.num_drones) if j != i]

            if self._ep_shield:
                res = safe_velocity(
                    own.position,
                    own.velocity,
                    desired,
                    peers,
                    self.obstacles,
                    self.dt,
                    self.shield_cfg,
                    dynamic_obstacles=self.dyn_obstacles,
                )
                cmd, inter, feas = res.velocity, res.intervention, res.feasible
            else:
                low = np.maximum(-self.contract.speed, own.velocity - max_dv)
                high = np.minimum(self.contract.speed, own.velocity + max_dv)
                low, high = np.minimum(low, high), np.maximum(low, high)
                cmd = np.minimum(np.maximum(desired, low), high)
                inter, feas = 0.0, True

            interventions[i] = float(inter)
            feasibles[i] = bool(feas)

            if self.domain_rand:
                gust = self._rng.normal(0.0, 0.3 * self.wind_max, size=3)
                gust[2] *= 0.25
                wind_accel = self._ep_wind_bias + gust
            else:
                wind_accel = np.zeros(3, dtype=np.float64)

            target_vel = own.velocity + alpha * (cmd - own.velocity) + wind_accel * self.dt
            dv = np.clip(target_vel - own.velocity, -max_dv, max_dv)
            new_vel = np.clip(own.velocity + dv, -self.contract.speed, self.contract.speed)
            new_pos = own.position + new_vel * self.dt
            new_yaw = _wrap_angle(own.yaw + yaw_rate * self.dt)

            self.telemetry[i] = Telemetry(
                position=new_pos,
                velocity=new_vel,
                yaw=new_yaw,
                stamp=stamp,
                active=True,
                agent_id=agent_id,
            )
            step_paths[i] = float(np.linalg.norm(new_vel) * self.dt)

        self._telemetry_history.append(
            [replace(t, position=t.position.copy(), velocity=t.velocity.copy()) for t in self.telemetry]
        )

        # 4. Pairwise distances, static/dynamic clearances, OOB, and reward computation
        positions = np.stack([t.position for t in self.telemetry], axis=0)
        velocities = np.stack([t.velocity for t in self.telemetry], axis=0)

        wx, wy, wz = self.shield_cfg.workspace
        per_agent_oob = (
            (positions[:, 0] < wx[0])
            | (positions[:, 0] > wx[1])
            | (positions[:, 1] < wy[0])
            | (positions[:, 1] > wy[1])
            | (positions[:, 2] < wz[0])
            | (positions[:, 2] > wz[1])
        )

        min_pair_per_agent = np.full(self.num_drones, float("inf"), dtype=np.float64)
        pair_ttc_penalty = np.zeros(self.num_drones, dtype=np.float64)
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                diff_ij = positions[j] - positions[i]
                d_ij = float(np.linalg.norm(diff_ij))
                if d_ij < min_pair_per_agent[i]:
                    min_pair_per_agent[i] = d_ij
                if d_ij < min_pair_per_agent[j]:
                    min_pair_per_agent[j] = d_ij
                if d_ij < 2.6:
                    unit = diff_ij / max(d_ij, 1e-9)
                    rel_vel = velocities[j] - velocities[i]
                    closing = max(0.0, -float(np.dot(unit, rel_vel)))
                    if closing > 0.0:
                        pen = 1.5 * closing * ((2.6 - d_ij) / 2.6)
                        pair_ttc_penalty[i] += pen
                        pair_ttc_penalty[j] += pen

        min_static_per_agent = np.full(self.num_drones, float("inf"), dtype=np.float64)
        obs_ttc_penalty = np.zeros(self.num_drones, dtype=np.float64)
        for o in self.obstacles:
            o_xy = np.array([o.x, o.y], dtype=np.float64)
            deltas_xy = positions[:, :2] - o_xy
            dists = np.linalg.norm(deltas_xy, axis=1)
            clearances = dists - float(o.radius) - self.contract.radius
            min_static_per_agent = np.minimum(min_static_per_agent, clearances)
            close_mask = clearances < 1.2
            if np.any(close_mask):
                for idx in np.flatnonzero(close_mask):
                    unit_xy = deltas_xy[idx] / max(float(dists[idx]), 1e-9)
                    closing = max(0.0, -float(np.dot(unit_xy, velocities[idx, :2])))
                    if closing > 0.0:
                        obs_ttc_penalty[idx] += 1.5 * closing * (
                            (1.2 - max(0.0, float(clearances[idx]))) / 1.2
                        )

        min_dyn_per_agent = np.full(self.num_drones, float("inf"), dtype=np.float64)
        for dyn in self.dyn_obstacles:
            if not dyn.active:
                continue
            d_pos = np.asarray(dyn.position, dtype=np.float64).reshape(1, 3)
            d_vel = np.asarray(dyn.velocity, dtype=np.float64).reshape(3)
            deltas = positions - d_pos
            dists = np.linalg.norm(deltas, axis=1)
            clearances = dists - float(dyn.radius) - self.contract.radius
            min_dyn_per_agent = np.minimum(min_dyn_per_agent, clearances)
            close_mask = clearances < 1.5
            if np.any(close_mask):
                for idx in np.flatnonzero(close_mask):
                    unit = deltas[idx] / max(float(dists[idx]), 1e-9)
                    closing = max(0.0, -float(np.dot(unit, velocities[idx] - d_vel)))
                    if closing > 0.0:
                        obs_ttc_penalty[idx] += 1.5 * closing * (
                            (1.5 - max(0.0, float(clearances[idx]))) / 1.5
                        )

        min_clearance_per_agent = np.minimum(min_static_per_agent, min_dyn_per_agent)

        rewards: dict[str, float] = {}
        goal_dists = np.zeros(self.num_drones, dtype=np.float64)
        docked_mask = np.zeros(self.num_drones, dtype=bool)
        norm_denom = max(self.contract.speed * self.dt, 1e-9)
        soft_pair_zone = max(self.contract.pair_safe, 1.35)
        soft_obs_zone = max(self.contract.obs_margin, 0.45)

        for i, agent_id in enumerate(self.possible_agents):
            dist = float(np.linalg.norm(self.goals[i] - positions[i]))
            goal_dists[i] = dist
            speed_i = float(np.linalg.norm(velocities[i]))
            r = 1.0 * (self._prev_dist[i] - dist) / norm_denom
            self._prev_dist[i] = dist

            if dist < self.success_radius:
                r += 0.3
                if speed_i < self.hold_speed:
                    r += 0.3
                    docked_mask[i] = True

            if math.isfinite(min_pair_per_agent[i]):
                r -= 4.0 * max(0.0, soft_pair_zone - min_pair_per_agent[i])
            if math.isfinite(min_clearance_per_agent[i]):
                r -= 4.0 * max(0.0, soft_obs_zone - min_clearance_per_agent[i])
            r -= float(pair_ttc_penalty[i])
            r -= float(obs_ttc_penalty[i])
            if per_agent_oob[i]:
                r -= 2.0
            r -= 0.05 * action_jerks[i]
            r -= 1.0 * interventions[i]
            rewards[agent_id] = float(r)

        # 5. Synchronized termination across all 10 drones (§8.4)
        if bool(np.all(docked_mask)):
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        swarm_min_pair = float(np.min(min_pair_per_agent))
        swarm_min_static = float(np.min(min_static_per_agent))
        swarm_min_dyn = float(np.min(min_dyn_per_agent))

        pair_collision = swarm_min_pair < 2.0 * self.contract.radius
        obstacle_collision = min(swarm_min_static, swarm_min_dyn) < 0.0
        swarm_collision = bool(pair_collision or obstacle_collision)
        swarm_oob = bool(np.any(per_agent_oob))
        swarm_success = bool(self._hold_counter >= self.hold_dwell)

        # Terminal bonuses/penalties so finishing swarm_success early strictly beats
        # farming the +0.6/step hold bonus (0.6 / (1 - 0.99) = 60.0 discounted max),
        # and culprit drones receive direct credit assignment on collisions.
        if swarm_success:
            for agent_id in self.possible_agents:
                rewards[agent_id] += 80.0
        elif swarm_collision:
            for i, agent_id in enumerate(self.possible_agents):
                culprit = (
                    min_pair_per_agent[i] < 2.0 * self.contract.radius
                    or min_clearance_per_agent[i] < 0.0
                )
                rewards[agent_id] -= 25.0 if culprit else 2.0

        synchronized_terminated = bool(swarm_collision or swarm_oob or swarm_success)
        synchronized_truncated = bool(
            (not synchronized_terminated) and (self._step_count >= self.max_steps)
        )

        fraction_docked = float(np.mean(docked_mask))
        obs = self._observe()

        terminations = {agent_id: synchronized_terminated for agent_id in self.possible_agents}
        truncations = {agent_id: synchronized_truncated for agent_id in self.possible_agents}
        infos: dict[str, dict[str, Any]] = {}
        for i, agent_id in enumerate(self.possible_agents):
            infos[agent_id] = {
                "goal_distance": float(goal_dists[i]),
                "docked": bool(docked_mask[i]),
                "fraction_docked": fraction_docked,
                "success": swarm_success,
                "collision": swarm_collision,
                "out_of_bounds": swarm_oob,
                "min_pair_distance": swarm_min_pair,
                "min_obstacle_clearance": swarm_min_static,
                "min_bird_clearance": swarm_min_dyn,
                "shield_intervention": float(interventions[i]),
                "shield_feasible": bool(feasibles[i]),
                "action_jerk": float(action_jerks[i]),
                "step_path_length": float(step_paths[i]),
                "goal_shifted": goal_shifted,
            }

        if synchronized_terminated or synchronized_truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
