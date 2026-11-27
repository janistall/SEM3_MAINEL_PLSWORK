"""Jitter-tolerant rate limiter for decoupling adapter telemetry from the 20 Hz PX4 tick."""
from __future__ import annotations

import math


class TelemetryRateLimiter:
    """Publish gate maintaining a target rate under fixed-period timer jitter."""

    def __init__(self, rate_hz: float = 10.0, *, tolerance_sec: float = 0.005) -> None:
        rate = float(rate_hz)
        if rate <= 0.0 or not math.isfinite(rate):
            raise ValueError(f"telemetry_rate must be positive and finite, got {rate_hz}")
        self.rate_hz = rate
        self.period_sec = 1.0 / rate
        self.tolerance_sec = min(max(0.0, float(tolerance_sec)), 0.25 * self.period_sec)
        self._next_due: float | None = None

    def should_publish(self, now: float) -> bool:
        now_f = float(now)
        if self._next_due is None:
            self._next_due = now_f + self.period_sec
            return True
        if now_f + self.tolerance_sec >= self._next_due:
            while self._next_due <= now_f + self.tolerance_sec:
                self._next_due += self.period_sec
            return True
        return False
