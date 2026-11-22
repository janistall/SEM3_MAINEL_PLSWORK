"""Single-source-of-truth contract re-export for offline MARL training.

All Contract v2 (`swarm-coop-enu-v2`) definitions, observation builders, action
decoders, and safety shield primitives live in `src/swarm_control`. This module
re-exports them directly so training and deployment cannot drift.
"""
from __future__ import annotations

import os
import sys

_SWARM_CONTROL_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "swarm_control")
)
if _SWARM_CONTROL_SRC not in sys.path:
    sys.path.insert(0, _SWARM_CONTROL_SRC)

from swarm_control.contract import (  # noqa: E402, F401
    ACT_DIM,
    ActorPolicy,
    Contract,
    DEFAULT_CONTRACT,
    GOAL_SCALE,
    K_DYN,
    K_PEER,
    K_STATIC,
    N_AGENTS,
    OBS_DIM,
    OBSTACLE_SCALE,
    PEER_ORDER,
    PEER_VEL_SCALE,
    SCHEMA,
    SUPPORTED_PEER_ORDERS,
    VEL_SCALE,
    action_to_velocity,
    apply_contract_metadata,
    build_actor,
    make_deploy_payload,
    normalize_actor_state_dict,
    normalize_observation,
    validate_action_spec,
)
from swarm_control.observation import (  # noqa: E402, F401
    DynamicObstacle,
    Obstacle,
    Telemetry,
    build_observation,
)
from swarm_control.safety import (  # noqa: E402, F401
    ShieldConfig,
    ShieldResult,
    safe_velocity,
)

__all__ = [
    "ACT_DIM",
    "ActorPolicy",
    "Contract",
    "DEFAULT_CONTRACT",
    "DynamicObstacle",
    "GOAL_SCALE",
    "K_DYN",
    "K_PEER",
    "K_STATIC",
    "N_AGENTS",
    "OBS_DIM",
    "OBSTACLE_SCALE",
    "Obstacle",
    "PEER_ORDER",
    "PEER_VEL_SCALE",
    "SCHEMA",
    "SUPPORTED_PEER_ORDERS",
    "ShieldConfig",
    "ShieldResult",
    "Telemetry",
    "VEL_SCALE",
    "action_to_velocity",
    "apply_contract_metadata",
    "build_actor",
    "build_observation",
    "make_deploy_payload",
    "normalize_actor_state_dict",
    "normalize_observation",
    "safe_velocity",
    "validate_action_spec",
]
