"""Pure-Python Contract v2 controller core for `SwarmControlNode`.

Separates the deterministic control, quorum gating, goal/dynamic-obstacle state,
safety shielding, and debug/status packaging from the ROS 2 `rclpy` transport
layer so the exact production logic can be unit-tested without ROS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from .contract import ActorPolicy, Contract, DEFAULT_CONTRACT, action_to_velocity
from .observation import DynamicObstacle, Obstacle, Telemetry, build_observation
from .planning import ClassicalMacro
from .safety import ShieldConfig, safe_velocity
from .supervisor import Mode, Supervisor, SupervisorConfig


@dataclass
class ControllerConfig:
    drone_id: str = "drone_0"
    agent_index: int = 0
    agent_ids: list[str] = field(
        default_factory=lambda: [f"drone_{i}" for i in range(DEFAULT_CONTRACT.agents)]
    )
    slot: Sequence[float] = (0.0, 0.0, 2.5)
    start: Sequence[float] = (0.0, 0.0, 0.45)
    mode: str = "learned"
    checkpoint: str = ""
    allow_legacy_checkpoint: bool = False
    actor_index: int = 0
    control_rate: float = 10.0
    enable_shield: bool = True
    use_supervisor: bool = False
    quorum_min_peers: int = 3
    altitude: float = DEFAULT_CONTRACT.alt
    speed: float = DEFAULT_CONTRACT.speed
    accel: float = DEFAULT_CONTRACT.accel
    yaw_rate: float = DEFAULT_CONTRACT.yaw_rate
    docking_timeout: float = 15.0
    hold_dwell: float = 1.5
    telemetry_timeout: float = 0.5
    dynamic_obstacle_timeout: float = 1.0
    shield_lookahead_tau: float = 0.0
    stall_recovery: bool = True
    stall_time: float = 1.0
    stall_recover_time: float = 1.5
    stall_clearance: float = 0.6


@dataclass
class TickOutput:
    quorum_ok: bool
    fresh_peer_count: int
    velocity: np.ndarray
    yaw_rate: float
    desired_velocity: np.ndarray
    action: np.ndarray
    observation: np.ndarray | None
    policy_mode: str
    supervisor_enabled: bool
    supervisor_mode: str
    stop_distance: float
    shield_feasible: bool
    shield_intervention: float
    stall_active: bool
    goal_distance: float
    dynamic_obstacle_count: int
    debug_vector: np.ndarray | None
    status_dict: dict[str, Any] | None


def _nearest_static_clearance(
    position: np.ndarray, obstacles: Sequence[Obstacle], drone_radius: float
) -> float:
    if not obstacles:
        return float("inf")
    pos_xy = np.asarray(position, dtype=np.float64)[:2]
    return min(
        float(np.linalg.norm(pos_xy - np.array([o.x, o.y], dtype=np.float64)))
        - float(o.radius)
        - float(drone_radius)
        for o in obstacles
    )


class SwarmControllerCore:
    """Contract v2 (`swarm-coop-enu-v2`) decentralized controller state machine."""

    def __init__(
        self,
        config: ControllerConfig,
        obstacles: Sequence[Obstacle] = (),
        *,
        policy_override: Callable[[np.ndarray], np.ndarray] | None = None,
        contract_override: Contract | None = None,
    ) -> None:
        self.drone_id = str(config.drone_id)
        self.agent_index = int(config.agent_index)
        self.agent_ids = [str(aid) for aid in config.agent_ids]
        self.slot = np.asarray(config.slot, dtype=np.float64).reshape(-1)
        self.start = np.asarray(config.start, dtype=np.float64).reshape(-1)
        self.mode_name = str(config.mode).strip().lower()
        self.checkpoint = str(config.checkpoint).strip()
        self.allow_legacy = bool(config.allow_legacy_checkpoint)
        self.actor_index = int(config.actor_index)
        self.control_rate = float(config.control_rate)
        self.enable_shield = bool(config.enable_shield)
        self.altitude = float(config.altitude)
        self.speed = float(config.speed)
        self.accel = float(config.accel)
        self.yaw_rate_max = float(config.yaw_rate)
        self.telemetry_timeout = float(config.telemetry_timeout)
        self.dynamic_obstacle_timeout = float(config.dynamic_obstacle_timeout)
        self.shield_lookahead_tau = float(config.shield_lookahead_tau)
        self.stall_recovery = bool(config.stall_recovery)
        self.stall_time = float(config.stall_time)
        self.stall_recover_time = float(config.stall_recover_time)
        self.stall_clearance = float(config.stall_clearance)

        if len(self.agent_ids) < 1:
            raise ValueError("agent_ids must contain at least one agent")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be unique")
        if not 0 <= self.agent_index < len(self.agent_ids):
            raise ValueError(
                f"agent_index {self.agent_index} out of range for {len(self.agent_ids)} agents"
            )
        if self.drone_id != self.agent_ids[self.agent_index]:
            raise ValueError("drone_id must equal agent_ids[agent_index]")
        if self.slot.shape != (3,) or self.start.shape != (3,):
            raise ValueError("slot and start must each have three values")
        if not (np.all(np.isfinite(self.slot)) and np.all(np.isfinite(self.start))):
            raise ValueError("slot and start must contain finite numbers")
        if self.control_rate <= 0.0:
            raise ValueError(f"control_rate must be positive, got {self.control_rate}")
        if self.telemetry_timeout <= 0.0:
            raise ValueError(
                f"telemetry_timeout must be positive, got {self.telemetry_timeout}"
            )
        if int(config.quorum_min_peers) < 0:
            raise ValueError(
                f"quorum_min_peers must be >= 0, got {config.quorum_min_peers}"
            )

        self.requested_use_supervisor = bool(config.use_supervisor)
        self.quorum_min_peers = min(int(config.quorum_min_peers), len(self.agent_ids) - 1)

        base_contract = (
            replace(contract_override)
            if contract_override is not None
            else Contract(
                agents=len(self.agent_ids),
                speed=self.speed,
                accel=self.accel,
                yaw_rate=self.yaw_rate_max,
                alt=self.altitude,
            )
        )
        self.contract = base_contract
        self.obstacles: list[Obstacle] = list(obstacles)
        self.dynamic_obstacles: list[DynamicObstacle] = []
        self.last_dyn_rx: float = 0.0

        self.policy: Callable[[np.ndarray], np.ndarray] | None = None
        self.classical: ClassicalMacro | None = None
        self.supervisor_override_warning: str | None = None

        if self.mode_name == "learned":
            # Phase 5 (§11.1): AI owns behavior in learned mode; supervisor is always bypassed.
            if self.requested_use_supervisor:
                self.supervisor_override_warning = (
                    "use_supervisor=True ignored in mode:=learned (AI owns behavior)"
                )
            self.use_supervisor = False
            if policy_override is not None:
                self.policy = policy_override
                if hasattr(policy_override, "contract") and isinstance(
                    policy_override.contract, Contract
                ):
                    self.contract = replace(policy_override.contract)
            else:
                if not self.checkpoint:
                    raise RuntimeError(
                        "mode:=learned requires checkpoint:=/path/to/v2_policy.pt "
                        "(swarm-coop-enu-v2 deployment checkpoint)."
                    )
                loaded = ActorPolicy(
                    self.checkpoint,
                    contract=self.contract,
                    actor_index=self.actor_index,
                    allow_legacy=self.allow_legacy,
                )
                self.policy = loaded
                self.contract = replace(loaded.contract)
            self.speed = float(self.contract.speed)
            self.accel = float(self.contract.accel)
            self.yaw_rate_max = float(self.contract.yaw_rate)
        elif self.mode_name == "classical":
            # Classical baseline uses A* macro + Supervisor docking/hold (§11.1, §16.1).
            self.use_supervisor = True
            self.classical = ClassicalMacro(
                self.start,
                self.slot,
                self.obstacles,
                speed=self.speed,
                altitude=self.altitude,
            )
        elif self.mode_name == "zero":
            # Hover baseline publishes zero velocity; supervisor must stay off.
            self.use_supervisor = False
        else:
            raise ValueError("mode must be learned, classical or zero")

        self.supervisor = Supervisor(
            SupervisorConfig(
                brake_accel=0.8,
                dock_speed=self.speed,
                docking_timeout=float(config.docking_timeout),
                hold_dwell=float(config.hold_dwell),
            )
        )
        self.shield = ShieldConfig.from_contract(
            self.contract,
            lookahead_tau=self.shield_lookahead_tau,
        )

        self.telemetry: dict[str, Telemetry] = {
            agent_id: Telemetry.zero(agent_id=agent_id) for agent_id in self.agent_ids
        }
        self._inactive_telemetry: dict[str, Telemetry] = {
            agent_id: Telemetry.zero(agent_id=agent_id) for agent_id in self.agent_ids
        }
        self._zero_vel = np.zeros(3, dtype=np.float64)
        self._zero_act = np.zeros(self.contract.act_dim, dtype=np.float32)
        self.last_rx: dict[str, float] = {agent_id: 0.0 for agent_id in self.agent_ids}

        self.dt = 1.0 / self.control_rate
        self.stall_timer = 0.0
        self.recover_timer = 0.0

    def update_telemetry(
        self,
        agent_id: str,
        position: Sequence[float],
        velocity: Sequence[float],
        yaw: float,
        stamp: float,
        active: bool,
        now: float,
    ) -> None:
        tel = self.telemetry.get(agent_id)
        if tel is None:
            return
        px, py, pz = float(position[0]), float(position[1]), float(position[2])
        vx, vy, vz = float(velocity[0]), float(velocity[1]), float(velocity[2])
        yaw_f = float(yaw)
        if not (
            math.isfinite(px)
            and math.isfinite(py)
            and math.isfinite(pz)
            and math.isfinite(vx)
            and math.isfinite(vy)
            and math.isfinite(vz)
            and math.isfinite(yaw_f)
        ):
            return
        tel.position[0] = px
        tel.position[1] = py
        tel.position[2] = pz
        tel.velocity[0] = vx
        tel.velocity[1] = vy
        tel.velocity[2] = vz
        tel.yaw = yaw_f
        tel.stamp = float(stamp)
        tel.active = bool(active)
        self.last_rx[agent_id] = float(now)

    def update_goal(self, goal_xyz: Sequence[float]) -> tuple[bool, str]:
        """Accept a runtime goal update (`/{drone}/goal`), validating workspace bounds."""
        raw = np.asarray(goal_xyz, dtype=np.float64).reshape(-1)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            return False, "goal must contain 3 finite ENU coordinates"
        candidate = raw.copy()
        if candidate[2] <= 0.0:
            candidate[2] = self.altitude

        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.shield.workspace
        if not (
            x_min <= candidate[0] <= x_max
            and y_min <= candidate[1] <= y_max
            and z_min <= candidate[2] <= z_max
        ):
            return (
                False,
                f"goal {candidate.round(3).tolist()} outside workspace {self.shield.workspace}",
            )

        if self.classical is not None:
            own = self.telemetry[self.drone_id]
            replan_start = own.position if self.last_rx[self.drone_id] > 0.0 else self.start
            try:
                replanned = ClassicalMacro(
                    replan_start,
                    candidate,
                    self.obstacles,
                    speed=self.speed,
                    altitude=float(candidate[2]),
                )
            except ValueError as exc:
                return False, f"classical replanning failed ({exc}); retaining previous goal"
            self.classical = replanned
            self.supervisor.reset()
            self.stall_timer = 0.0
            self.recover_timer = 0.0

        self.slot = candidate
        self.altitude = float(candidate[2])
        return True, f"updated goal to {self.slot.round(3).tolist()}"

    def update_dynamic_obstacles(
        self, obstacles: Sequence[DynamicObstacle], now: float
    ) -> None:
        cleaned: list[DynamicObstacle] = []
        for obs in obstacles:
            pos = np.asarray(obs.position, dtype=np.float64).reshape(3)
            vel = np.asarray(obs.velocity, dtype=np.float64).reshape(3)
            radius = float(obs.radius)
            if not (
                np.all(np.isfinite(pos))
                and np.all(np.isfinite(vel))
                and np.isfinite(radius)
                and radius >= 0.0
            ):
                continue
            cleaned.append(
                DynamicObstacle(
                    position=pos,
                    velocity=vel,
                    radius=radius,
                    active=bool(obs.active),
                )
            )
        self.dynamic_obstacles = cleaned
        self.last_dyn_rx = float(now)

    def current_dynamic_obstacles(self, now: float) -> list[DynamicObstacle]:
        if not self.dynamic_obstacles or self.last_dyn_rx <= 0.0:
            return []
        if float(now) - self.last_dyn_rx > self.dynamic_obstacle_timeout:
            return []
        return [obs for obs in self.dynamic_obstacles if obs.active]

    def evaluate_quorum(self, now: float) -> tuple[bool, int, list[Telemetry]]:
        """Evaluate §12.3 quorum telemetry gating.

        Requires self active & fresh AND at least `quorum_min_peers` peers active & fresh.
        Stale or inactive peers are passed with `active=False` so `build_observation`
        pads missing slots with `valid=0.0` and `safe_velocity` ignores stale peers.
        """
        now_f = float(now)
        own = self.telemetry[self.drone_id]
        own_fresh = (
            own.active
            and self.last_rx[self.drone_id] > 0.0
            and (now_f - self.last_rx[self.drone_id]) <= self.telemetry_timeout
        )

        effective: list[Telemetry] = []
        fresh_peers = 0
        for agent_id in self.agent_ids:
            tel = self.telemetry[agent_id]
            is_fresh = (
                tel.active
                and self.last_rx[agent_id] > 0.0
                and (now_f - self.last_rx[agent_id]) <= self.telemetry_timeout
            )
            if agent_id != self.drone_id and is_fresh:
                fresh_peers += 1
            if is_fresh:
                effective.append(tel)
            else:
                effective.append(self._inactive_telemetry[agent_id])

        quorum_ok = bool(own_fresh and fresh_peers >= self.quorum_min_peers)
        return quorum_ok, fresh_peers, effective

    def step(self, now: float) -> TickOutput:
        quorum_ok, fresh_peer_count, effective_telemetry = self.evaluate_quorum(now)
        if not quorum_ok:
            return TickOutput(
                quorum_ok=False,
                fresh_peer_count=fresh_peer_count,
                velocity=self._zero_vel,
                yaw_rate=0.0,
                desired_velocity=self._zero_vel,
                action=self._zero_act,
                observation=None,
                policy_mode=self.mode_name.upper(),
                supervisor_enabled=self.use_supervisor,
                supervisor_mode=self.supervisor.mode.value if self.use_supervisor else "DISABLED",
                stop_distance=0.0,
                shield_feasible=True,
                shield_intervention=0.0,
                stall_active=False,
                goal_distance=0.0,
                dynamic_obstacle_count=0,
                debug_vector=None,
                status_dict=None,
            )

        own = effective_telemetry[self.agent_index]
        peers = [
            tel for idx, tel in enumerate(effective_telemetry) if idx != self.agent_index
        ]
        active_dyn = self.current_dynamic_obstacles(now)

        observation = build_observation(
            self.agent_index,
            effective_telemetry,
            self.slot,
            self.obstacles,
            self.contract,
            dynamic_obstacles=active_dyn,
        )

        stall_active = False
        if self.policy is not None:
            action = np.asarray(self.policy(observation), dtype=np.float32).reshape(
                self.contract.act_dim
            )
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            desired, macro_yaw_rate = action_to_velocity(action, self.contract)
            stop_distance = 0.0
            policy_mode = "LEARNED"
            display_mode = "LEARNED"
        elif self.classical is not None:
            action = np.zeros(self.contract.act_dim, dtype=np.float32)
            active_peers = [p for p in peers if p.active]
            macro_velocity = self.classical(own.position, own.velocity, active_peers, self.dt)
            macro_yaw_rate = 0.0
            desired, stop_distance = self.supervisor.step(
                own.position, own.velocity, self.slot, macro_velocity, self.dt
            )
            policy_mode = "CLASSICAL"

            # Stall recovery is strictly gated to classical mode (§11.1) and uses exact
            # geometric obstacle clearance (avoiding the legacy v1 observation[20] index).
            if (
                self.stall_recovery
                and self.supervisor.mode == Mode.APPROACH
                and self.obstacles
                and own.position[2] > self.altitude - 0.25
            ):
                distance = float(np.linalg.norm(self.slot - own.position))
                speed_xy = float(np.linalg.norm(own.velocity[:2]))
                clearance = _nearest_static_clearance(
                    own.position, self.obstacles, self.contract.radius
                )
                if distance > 1.0 and speed_xy < 0.05 and clearance < self.stall_clearance:
                    self.stall_timer += self.dt
                else:
                    self.stall_timer = max(0.0, self.stall_timer - self.dt)
                if self.stall_timer >= self.stall_time:
                    self.recover_timer = self.stall_recover_time
                    self.stall_timer = 0.0
                if self.recover_timer > 0.0:
                    nearest = min(
                        self.obstacles,
                        key=lambda o: float(
                            np.linalg.norm(
                                own.position[:2] - np.array([o.x, o.y], dtype=np.float64)
                            )
                        )
                        - float(o.radius),
                    )
                    delta = own.position[:2] - np.array(
                        [nearest.x, nearest.y], dtype=np.float64
                    )
                    length = float(np.linalg.norm(delta))
                    if length < 1e-9:
                        delta = np.array([1.0, 0.0], dtype=np.float64)
                        length = 1.0
                    desired = np.array(
                        [
                            delta[0] / length * self.speed,
                            delta[1] / length * self.speed,
                            float(
                                np.clip(
                                    1.1 * (self.altitude - own.position[2]),
                                    -self.speed,
                                    self.speed,
                                )
                            ),
                        ],
                        dtype=np.float64,
                    )
                    self.recover_timer = max(0.0, self.recover_timer - self.dt)
                    stall_active = True

            display_mode = self.supervisor.mode.value
        else:
            action = np.zeros(self.contract.act_dim, dtype=np.float32)
            desired = np.zeros(3, dtype=np.float64)
            macro_yaw_rate = 0.0
            stop_distance = 0.0
            policy_mode = "ZERO"
            display_mode = "ZERO"

        if self.use_supervisor and (
            stall_active or self.supervisor.mode in (Mode.DOCKING, Mode.HOLD)
        ):
            yaw_rate = 0.0
        else:
            yaw_rate = float(np.clip(macro_yaw_rate, -self.yaw_rate_max, self.yaw_rate_max))

        if self.enable_shield:
            result = safe_velocity(
                own.position,
                own.velocity,
                desired,
                peers,
                self.obstacles,
                self.dt,
                self.shield,
                dynamic_obstacles=active_dyn,
            )
            velocity = result.velocity
            feasible = bool(result.feasible)
            intervention = float(result.intervention)
        else:
            current = np.clip(own.velocity, -self.speed, self.speed)
            low = np.maximum(-self.speed, current - self.accel * self.dt)
            high = np.minimum(self.speed, current + self.accel * self.dt)
            low, high = np.minimum(low, high), np.maximum(low, high)
            velocity = np.minimum(np.maximum(desired, low), high)
            feasible = True
            intervention = 0.0

        goal_distance = float(np.linalg.norm(self.slot - own.position))
        debug_vector = np.concatenate(
            [
                observation.astype(np.float32, copy=False),
                action.astype(np.float32, copy=False),
                np.asarray(desired, dtype=np.float32).reshape(3),
                np.asarray(velocity, dtype=np.float32).reshape(3),
                np.array(
                    [float(intervention), 1.0 if feasible else 0.0, goal_distance],
                    dtype=np.float32,
                ),
            ]
        )
        status_dict = {
            "drone_id": self.drone_id,
            "mode": display_mode,
            "policy": policy_mode,
            "supervisor": bool(self.use_supervisor),
            "position": own.position.round(3).tolist(),
            "velocity": own.velocity.round(3).tolist(),
            "slot": self.slot.round(3).tolist(),
            "goal_distance": round(goal_distance, 3),
            "d_stop": round(float(stop_distance), 3),
            "action": [round(float(v), 4) for v in action],
            "shield_feasible": bool(feasible),
            "shield_intervention": round(float(intervention), 4),
            "stall_recovery": bool(stall_active),
            "quorum_peers": int(fresh_peer_count),
            "dynamic_obstacles": len(active_dyn),
        }

        return TickOutput(
            quorum_ok=True,
            fresh_peer_count=fresh_peer_count,
            velocity=velocity,
            yaw_rate=yaw_rate,
            desired_velocity=desired,
            action=action,
            observation=observation,
            policy_mode=policy_mode,
            supervisor_enabled=self.use_supervisor,
            supervisor_mode=display_mode,
            stop_distance=float(stop_distance),
            shield_feasible=feasible,
            shield_intervention=intervention,
            stall_active=stall_active,
            goal_distance=goal_distance,
            dynamic_obstacle_count=len(active_dyn),
            debug_vector=debug_vector,
            status_dict=status_dict,
        )
