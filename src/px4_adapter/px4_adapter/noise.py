"""Pure-NumPy Gaussian sensor noise injector for ENU adapter telemetry (Phase 8 §14.2)."""
from __future__ import annotations

import math

import numpy as np


def inject_telemetry_noise(
    position: np.ndarray,
    velocity: np.ndarray,
    yaw: float,
    *,
    pos_std: float = 0.0,
    vel_std: float = 0.0,
    yaw_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Optionally inject zero-mean Gaussian sensor noise into ENU telemetry.

    Returns exact copies when all standard deviations are <= 0.0.
    """
    pos_out = np.asarray(position, dtype=np.float64).reshape(3).copy()
    vel_out = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    yaw_out = float(yaw)
    if pos_std <= 0.0 and vel_std <= 0.0 and yaw_std <= 0.0:
        return pos_out, vel_out, yaw_out

    generator = rng if rng is not None else np.random.default_rng()
    if pos_std > 0.0:
        pos_out = pos_out + generator.normal(0.0, float(pos_std), size=3)
    if vel_std > 0.0:
        vel_out = vel_out + generator.normal(0.0, float(vel_std), size=3)
    if yaw_std > 0.0:
        yaw_out = yaw_out + float(generator.normal(0.0, float(yaw_std)))
        yaw_out = (yaw_out + math.pi) % (2.0 * math.pi) - math.pi
    return pos_out, vel_out, yaw_out
