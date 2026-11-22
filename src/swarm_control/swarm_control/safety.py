from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contract import Contract, DEFAULT_CONTRACT
from .observation import DynamicObstacle, Obstacle, Telemetry


@dataclass
class ShieldConfig:
    speed: float = 1.5
    accel: float = 2.0
    pair_safe: float = 0.9
    radius: float = 0.45
    obstacle_margin: float = 0.0
    workspace: tuple[tuple[float, float], ...] = ((-15.0, 15.0), (-15.0, 15.0), (0.35, 4.0))
    alpha: float = 1.4
    iterations: int = 60
    qp_iterations: int = 18
    lookahead_tau: float = 0.0
    weight_peer: float = 400.0
    weight_dyn: float = 300.0
    weight_static: float = 250.0
    weight_workspace: float = 150.0

    @classmethod
    def from_contract(
        cls,
        contract: Contract = DEFAULT_CONTRACT,
        *,
        workspace: tuple[tuple[float, float], ...] | None = None,
        alpha: float = 1.4,
        iterations: int = 60,
        qp_iterations: int = 18,
        lookahead_tau: float = 0.0,
    ) -> "ShieldConfig":
        """Build hard-backstop shield limits directly from a Contract."""
        resolved_workspace = (
            workspace
            if workspace is not None
            else ((-15.0, 15.0), (-15.0, 15.0), (0.35, max(4.0, float(contract.alt) + 1.5)))
        )
        return cls(
            speed=float(contract.speed),
            accel=float(contract.accel),
            pair_safe=float(contract.hard_pair),
            radius=float(contract.radius),
            obstacle_margin=float(contract.hard_margin),
            workspace=resolved_workspace,
            alpha=float(alpha),
            iterations=int(iterations),
            qp_iterations=int(qp_iterations),
            lookahead_tau=float(lookahead_tau),
        )


@dataclass
class ShieldResult:
    velocity: np.ndarray
    feasible: bool
    intervention: float


def _clip_box(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)


def _solve_prioritized_slack_qp(
    v_box: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    matrix: np.ndarray,
    bounds: np.ndarray,
    weights: np.ndarray,
    qp_iterations: int,
) -> np.ndarray:
    """Pure-NumPy 3-D prioritized quadratic-slack QP solver with active retreat fallback.

    Minimizes over `low <= v <= high`:
        0.5 * ||v - v_box||^2 + 0.5 * sum_k w_k * max(0, b_k - n_k . v)^2
    """
    def _objective(v: np.ndarray) -> float:
        viol = np.maximum(0.0, bounds - matrix @ v)
        return 0.5 * float(np.dot(v - v_box, v - v_box)) + 0.5 * float(
            np.dot(weights, viol * viol)
        )

    # Seed from v_box nudged along the priority-weighted retreat normal
    init_viol = np.maximum(0.0, bounds - matrix @ v_box)
    if np.any(init_viol > 0.0):
        push = matrix.T @ (weights * init_viol)
        push_norm = float(np.linalg.norm(push))
        if push_norm > 1e-9:
            candidate = _clip_box(v_box + (push / push_norm) * float(np.linalg.norm(high - low)), low, high)
            if _objective(candidate) > _objective(v_box):
                candidate = v_box.copy()
        else:
            candidate = v_box.copy()
    else:
        candidate = v_box.copy()

    eye3 = np.eye(3, dtype=np.float64)
    for _ in range(max(1, qp_iterations)):
        violations = bounds - matrix @ candidate
        active = violations > 0.0
        if not np.any(active):
            break
        m_act = matrix[active]
        v_act = violations[active]
        w_act = weights[active]
        wm = m_act * w_act[:, None]
        grad = (candidate - v_box) - (m_act.T @ (w_act * v_act))
        hess = eye3 + (wm.T @ m_act)
        try:
            step_dir = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step_dir = -grad / max(float(np.trace(hess)), 1.0)

        base_obj = _objective(candidate)
        improved = False
        for alpha_step in (1.0, 0.5, 0.25, 0.1):
            trial = _clip_box(candidate + alpha_step * step_dir, low, high)
            if _objective(trial) < base_obj - 1e-12:
                candidate = trial
                improved = True
                break
        if not improved:
            break

    # Retreat fallback: if opposing constraints cancel to near-zero while still violated,
    # bias along the highest-priority violated normal so the drone actively separates.
    final_viol = bounds - matrix @ candidate
    if float(np.linalg.norm(candidate)) <= 1e-6 and float(np.max(final_viol)) > 1e-6:
        priority_score = weights * np.maximum(0.0, final_viol)
        worst_idx = int(np.argmax(priority_score))
        escape_dir = matrix[worst_idx]
        candidate = _clip_box(escape_dir * float(np.max(np.abs(high))), low, high)

    return candidate


def safe_velocity(
    position: np.ndarray,
    velocity: np.ndarray,
    desired: np.ndarray,
    peers: Sequence[Telemetry],
    obstacles: Sequence[Obstacle],
    dt: float,
    config: ShieldConfig | None = None,
    dynamic_obstacles: Sequence[DynamicObstacle] = (),
) -> ShieldResult:
    cfg = config or ShieldConfig()
    position = np.asarray(position, dtype=np.float64).reshape(3)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    target = np.asarray(desired, dtype=np.float64).reshape(3)

    current = np.clip(velocity, -cfg.speed, cfg.speed)
    low = np.maximum(-cfg.speed, current - cfg.accel * dt)
    high = np.minimum(cfg.speed, current + cfg.accel * dt)
    low, high = np.minimum(low, high), np.maximum(low, high)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    weights: list[float] = []

    for peer in peers:
        if not getattr(peer, "active", True):
            continue
        delta = position - np.asarray(peer.position, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(delta))
        normal = delta / max(distance, 1e-9)
        rows.append(normal)
        rhs.append(
            -cfg.alpha * (distance - cfg.pair_safe)
            + float(np.dot(normal, np.asarray(peer.velocity, dtype=np.float64).reshape(3)))
        )
        weights.append(cfg.weight_peer)

    for obstacle in obstacles:
        delta = position[:2] - np.array([obstacle.x, obstacle.y], dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        normal = np.array([delta[0] / max(distance, 1e-9), delta[1] / max(distance, 1e-9), 0.0])
        rows.append(normal)
        rhs.append(
            -cfg.alpha * (distance - obstacle.radius - cfg.radius - cfg.obstacle_margin)
        )
        weights.append(cfg.weight_static)

    for dyn in dynamic_obstacles:
        if not getattr(dyn, "active", True):
            continue
        dyn_pos = np.asarray(dyn.position, dtype=np.float64).reshape(3)
        dyn_vel = np.asarray(dyn.velocity, dtype=np.float64).reshape(3)
        if cfg.lookahead_tau > 0.0:
            pred_delta = (position + current * cfg.lookahead_tau) - (
                dyn_pos + dyn_vel * cfg.lookahead_tau
            )
            delta = (
                pred_delta
                if float(np.linalg.norm(pred_delta)) < float(np.linalg.norm(position - dyn_pos))
                else (position - dyn_pos)
            )
        else:
            delta = position - dyn_pos
        distance = float(np.linalg.norm(delta))
        normal = delta / max(distance, 1e-9)
        rows.append(normal)
        rhs.append(
            -cfg.alpha * (distance - float(dyn.radius) - cfg.radius - cfg.obstacle_margin)
            + float(np.dot(normal, dyn_vel))
        )
        weights.append(cfg.weight_dyn)

    for axis, (minimum, maximum) in enumerate(cfg.workspace):
        row = np.zeros(3, dtype=np.float64)
        row[axis] = 1.0
        rows.append(row)
        rhs.append(-cfg.alpha * (position[axis] - minimum))
        weights.append(cfg.weight_workspace)
        rows.append(-row)
        rhs.append(-cfg.alpha * (maximum - position[axis]))
        weights.append(cfg.weight_workspace)

    matrix = np.asarray(rows, dtype=np.float64)
    bounds = np.asarray(rhs, dtype=np.float64)
    weight_vec = np.asarray(weights, dtype=np.float64)

    v_box = _clip_box(target, low, high)

    # 1. Fast-path check: skip solver entirely when v_box already satisfies all CBF constraints
    if float(np.max(bounds - matrix @ v_box)) <= 1e-9:
        return ShieldResult(v_box, True, 0.0)

    # 2. Hard projection pass for feasible constraint sets
    candidate = v_box.copy()
    for _ in range(cfg.iterations):
        worst = 0.0
        for row, bound in zip(matrix, bounds):
            violation = float(bound - np.dot(row, candidate))
            if violation > 1e-9:
                candidate = candidate + (violation / float(np.dot(row, row))) * row
                candidate = _clip_box(candidate, low, high)
                worst = max(worst, violation)
        if worst <= 1e-9:
            break

    residual = float(np.min(matrix @ candidate - bounds))
    if residual >= -1e-6:
        intervention = float(np.linalg.norm(candidate - v_box))
        return ShieldResult(candidate, True, intervention)

    # 3. Infeasible hard intersection: solve prioritized-slack 3-D QP with active retreat fallback
    retreat_vel = _solve_prioritized_slack_qp(
        v_box, low, high, matrix, bounds, weight_vec, cfg.qp_iterations
    )
    feasible_after_qp = float(np.min(matrix @ retreat_vel - bounds)) >= -1e-6
    intervention = float(np.linalg.norm(retreat_vel - v_box))
    return ShieldResult(retreat_vel, feasible_after_qp, intervention)
