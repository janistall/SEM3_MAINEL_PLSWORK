"""Observation/action contract v2 and checkpoint loader for the swarm controller.

The contract in this file is `swarm-coop-enu-v2`: a configurable ENU observation
and 4-D velocity/yaw-rate action schema shared directly by both ROS 2 deployment
and offline MARL training. Default configuration targets 10 drones with
`k_peer=4`, `k_static=3`, `k_dyn=2` (`obs_dim=62`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = "swarm-coop-enu-v2"
ACT_DIM = 4
N_AGENTS = 10
K_PEER = 4
K_STATIC = 3
K_DYN = 2
OBS_DIM = 6 + 7 * K_PEER + 4 * K_STATIC + 7 * K_DYN + 2  # 62

# Fixed v2 scales from the contract. A checkpoint may override these in its metadata.
GOAL_SCALE = 25.0
VEL_SCALE = 1.5
PEER_VEL_SCALE = 3.0
OBSTACLE_SCALE = 25.0

PEER_ORDER = "nearest_k_distance_then_agent_id"
SUPPORTED_PEER_ORDERS = (PEER_ORDER,)

_ACTION_SPEC_RE = re.compile(
    r"^world_ENU_velocity_([0-9]+(?:\.[0-9]+)?)mps_yawrate_([0-9]+(?:\.[0-9]+)?)rps$"
)


@dataclass
class Contract:
    schema: str = SCHEMA
    act_dim: int = ACT_DIM
    agents: int = N_AGENTS
    k_peer: int = K_PEER
    k_static: int = K_STATIC
    k_dyn: int = K_DYN
    hidden: Sequence[int] = (256, 256)
    goal_scale: float = GOAL_SCALE
    vel_scale: float = VEL_SCALE
    peer_vel_scale: float = PEER_VEL_SCALE
    obstacle_scale: float = OBSTACLE_SCALE
    speed: float = 1.5
    accel: float = 2.0
    yaw_rate: float = 0.6
    radius: float = 0.45
    pair_safe: float = 1.1
    hard_pair: float = 0.9
    obs_margin: float = 0.25
    hard_margin: float = 0.0
    alt: float = 2.5
    peer_order: str = PEER_ORDER
    clip_obs: bool = True

    def __post_init__(self) -> None:
        self.agents = int(self.agents)
        self.act_dim = int(self.act_dim)
        self.k_peer = int(self.k_peer)
        self.k_static = int(self.k_static)
        self.k_dyn = int(self.k_dyn)
        self.hidden = tuple(int(w) for w in self.hidden)
        if self.agents < 1:
            raise ValueError(f"agents must be >= 1, got {self.agents}")
        if self.act_dim != ACT_DIM:
            raise ValueError(f"act_dim must be {ACT_DIM}, got {self.act_dim}")
        if self.k_peer < 0 or self.k_static < 0 or self.k_dyn < 0:
            raise ValueError(
                f"k_peer/k_static/k_dyn must be non-negative, got "
                f"({self.k_peer}, {self.k_static}, {self.k_dyn})"
            )
        for name in (
            "goal_scale",
            "vel_scale",
            "peer_vel_scale",
            "obstacle_scale",
            "speed",
            "accel",
            "yaw_rate",
            "radius",
        ):
            value = float(getattr(self, name))
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be positive and finite, got {value}")
            setattr(self, name, value)
        for name in ("pair_safe", "hard_pair", "obs_margin", "hard_margin"):
            value = float(getattr(self, name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be non-negative and finite, got {value}")
            setattr(self, name, value)
        self.alt = float(self.alt)
        self.clip_obs = bool(self.clip_obs)
        if self.peer_order not in SUPPORTED_PEER_ORDERS:
            raise ValueError(
                f"unsupported peer_order {self.peer_order!r}; "
                f"supported: {SUPPORTED_PEER_ORDERS}"
            )

    @property
    def obs_dim(self) -> int:
        return 6 + 7 * self.k_peer + 4 * self.k_static + 7 * self.k_dyn + 2

    @property
    def action_spec(self) -> str:
        return f"world_ENU_velocity_{self.speed}mps_yawrate_{self.yaw_rate}rps"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "agents": self.agents,
            "k_peer": self.k_peer,
            "k_static": self.k_static,
            "k_dyn": self.k_dyn,
            "hidden": list(self.hidden),
            "goal_scale": self.goal_scale,
            "vel_scale": self.vel_scale,
            "peer_vel_scale": self.peer_vel_scale,
            "obstacle_scale": self.obstacle_scale,
            "speed": self.speed,
            "accel": self.accel,
            "yaw_rate": self.yaw_rate,
            "radius": self.radius,
            "pair_safe": self.pair_safe,
            "hard_pair": self.hard_pair,
            "obs_margin": self.obs_margin,
            "hard_margin": self.hard_margin,
            "alt": self.alt,
            "peer_order": self.peer_order,
            "action": self.action_spec,
            "clip_obs": self.clip_obs,
        }


DEFAULT_CONTRACT = Contract()


class MissingPolicyDependency(RuntimeError):
    pass


def validate_action_spec(action_str: str, speed: float, yaw_rate: float) -> None:
    if not isinstance(action_str, str):
        raise ValueError(f"action metadata must be a string, got {type(action_str).__name__}")
    match = _ACTION_SPEC_RE.match(action_str)
    if match is None:
        raise ValueError(
            f"unsupported action specification {action_str!r}; expected "
            f"'world_ENU_velocity_<speed>mps_yawrate_<yaw_rate>rps'"
        )
    parsed_speed = float(match.group(1))
    parsed_yaw = float(match.group(2))
    if not math.isclose(parsed_speed, float(speed), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"action spec speed {parsed_speed} != contract speed {speed}"
        )
    if not math.isclose(parsed_yaw, float(yaw_rate), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"action spec yaw_rate {parsed_yaw} != contract yaw_rate {yaw_rate}"
        )


def apply_contract_metadata(contract: Contract, meta: dict[str, Any]) -> Contract:
    """Validate checkpoint contract metadata and return a resolved Contract copy."""
    if not isinstance(meta, dict):
        raise ValueError("checkpoint contract metadata must be a dictionary")

    resolved = replace(contract)

    declared_schema = meta.get("schema")
    if declared_schema is not None and declared_schema != resolved.schema:
        raise ValueError(
            f"checkpoint schema {declared_schema!r} != controller schema {resolved.schema!r}"
        )

    for key in ("agents", "k_peer", "k_static", "k_dyn", "act_dim"):
        if key in meta:
            expected = getattr(resolved, key)
            actual = int(meta[key])
            if actual != int(expected):
                raise ValueError(
                    f"checkpoint contract.{key}={actual} != controller {key}={expected}"
                )

    if "obs_dim" in meta and int(meta["obs_dim"]) != resolved.obs_dim:
        raise ValueError(
            f"checkpoint contract.obs_dim={meta['obs_dim']} != controller obs_dim={resolved.obs_dim}"
        )

    if "peer_order" in meta:
        peer_order = meta["peer_order"]
        if peer_order not in SUPPORTED_PEER_ORDERS:
            raise ValueError(
                f"checkpoint peer_order {peer_order!r} is not supported "
                f"(expected one of {SUPPORTED_PEER_ORDERS})"
            )
        resolved.peer_order = str(peer_order)

    float_fields = (
        "goal_scale",
        "vel_scale",
        "peer_vel_scale",
        "obstacle_scale",
        "speed",
        "accel",
        "yaw_rate",
        "radius",
        "pair_safe",
        "hard_pair",
        "obs_margin",
        "hard_margin",
        "alt",
    )
    for name in float_fields:
        if name in meta:
            setattr(resolved, name, float(meta[name]))

    if "hidden" in meta:
        resolved.hidden = tuple(int(x) for x in meta["hidden"])
    if "clip_obs" in meta:
        resolved.clip_obs = bool(meta["clip_obs"])

    if "action" in meta:
        validate_action_spec(meta["action"], resolved.speed, resolved.yaw_rate)

    # Re-run __post_init__ validation on the resolved copy.
    return replace(resolved)


def normalize_actor_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Strip a uniform leading `net.` prefix from an actor state_dict if present."""
    if state and all(str(key).startswith("net.") for key in state):
        return {str(key)[4:]: value for key, value in state.items()}
    return dict(state)


def make_deploy_payload(
    contract: Contract,
    actor_state_dict: dict[str, Any],
    *,
    algo: str = "mappo_ctde_shared",
    training_steps: int = 0,
    seed: int = 0,
    stage: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Build a weights_only=True-safe deployment checkpoint payload."""
    clean_state = normalize_actor_state_dict(actor_state_dict)
    cpu_state: dict[str, Any] = {}
    for key, value in clean_state.items():
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        cpu_state[str(key)] = value

    meta = contract.as_dict()
    meta.update(
        {
            "algo": str(algo),
            "training_steps": int(training_steps),
            "seed": int(seed),
            "stage": str(stage),
            "note": str(note),
        }
    )
    return {
        "contract": meta,
        "actor": cpu_state,
        "obs_dim": int(contract.obs_dim),
        "action_dim": int(contract.act_dim),
        "num_drones": int(contract.agents),
    }


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        raise MissingPolicyDependency(
            "PyTorch is required for mode:=learned. Install a CPU wheel, e.g. "
            "python3 -m pip install 'torch>=2.6,<3' --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def build_actor(contract: Contract):
    torch = _require_torch()
    from torch import nn

    layers: list[nn.Module] = []
    last = contract.obs_dim
    for width in contract.hidden:
        layers += [nn.Linear(last, int(width)), nn.ReLU()]
        last = int(width)
    layers += [nn.Linear(last, contract.act_dim), nn.Tanh()]
    return nn.Sequential(*layers)


def _state_dict_from_payload(payload: Any, actor_index: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not a dictionary or TorchScript module")
    if "actors" in payload:
        actors = payload["actors"]
        if not isinstance(actors, (list, tuple)) or len(actors) <= actor_index:
            raise ValueError(f"checkpoint has no actor at index {actor_index}")
        return actors[actor_index]
    for key in ("actor", "actor_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    # A bare state dict is accepted only with an explicit legacy override.
    if all(isinstance(v, (dict,)) or hasattr(v, "shape") for v in payload.values()):
        return payload
    raise ValueError("could not locate actor weights in checkpoint")


def _infer_hidden(state: dict[str, Any], contract: Contract) -> tuple[int, ...]:
    keys = sorted(
        (
            key
            for key in state
            if key.endswith(".weight") and key.split(".")[0].isdigit()
        ),
        key=lambda key: int(key.split(".")[0]),
    )
    if not keys:
        return tuple(int(width) for width in contract.hidden)
    widths: list[int] = []
    previous_out = None
    for key in keys:
        weight = state[key]
        shape = tuple(int(value) for value in weight.shape)
        if len(shape) != 2:
            raise ValueError(f"unexpected actor weight shape for {key}: {shape}")
        out_features, in_features = shape
        if previous_out is None:
            if in_features != contract.obs_dim:
                raise ValueError(
                    f"actor input width {in_features} != observation dimension {contract.obs_dim}"
                )
        elif in_features != previous_out:
            raise ValueError(f"actor layer {key} input width {in_features} != {previous_out}")
        widths.append(out_features)
        previous_out = out_features
    if widths[-1] != contract.act_dim:
        raise ValueError(
            f"actor output width {widths[-1]} != action dimension {contract.act_dim}"
        )
    return tuple(widths[:-1])


class ActorPolicy:
    """Strict checkpoint loader for `swarm-coop-enu-v2`.

    Accepted formats:
      * TorchScript module (`.pt` produced with torch.jit.script/trace)
      * dict with `actor` / `actor_state_dict` / `state_dict` / `model`
      * dict with `actors` list of state_dicts
      * bare state_dict (requires allow_legacy=True)

    A checkpoint carrying `contract` metadata must match this contract's schema,
    dimensions, peer ordering, and action specification.
    """

    def __init__(
        self,
        path: str | Path,
        contract: Contract = DEFAULT_CONTRACT,
        actor_index: int = 0,
        allow_legacy: bool = False,
        device: str = "cpu",
    ):
        torch = _require_torch()
        self.torch = torch
        self.contract = replace(contract)
        self.actor_index = int(actor_index)
        self.device = torch.device(device)
        self.scripted = None
        self.module = None
        self.source = str(path)
        self.metadata: dict[str, Any] = {}

        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"policy checkpoint not found: {checkpoint}")

        payload = None
        try:
            loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = None

        if payload is None:
            # Fallback to TorchScript module if the file is not a plain weights_only dict.
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    scripted = torch.jit.load(str(checkpoint), map_location=self.device)
                scripted.eval()
                with torch.no_grad():
                    probe = scripted(
                        torch.zeros(1, self.contract.obs_dim, device=self.device)
                    )
                if tuple(probe.shape) != (1, self.contract.act_dim):
                    raise ValueError(
                        f"TorchScript output shape {tuple(probe.shape)} != (1, {self.contract.act_dim})"
                    )
                self.scripted = scripted
                return
            except Exception as exc:
                raise ValueError(
                    "could not load checkpoint without executing pickle code; "
                    "export a clean deployment checkpoint via make_deploy_payload() or TorchScript"
                ) from exc

        if isinstance(payload, dict) and "contract" in payload:
            meta = payload["contract"]
            if not isinstance(meta, dict):
                raise ValueError("checkpoint 'contract' metadata must be a dict")
            self.metadata = dict(meta)
            self.contract = apply_contract_metadata(self.contract, meta)

        if isinstance(payload, dict):
            for key, expected in (
                ("obs_dim", self.contract.obs_dim),
                ("action_dim", self.contract.act_dim),
                ("num_drones", self.contract.agents),
            ):
                if key in payload and int(payload[key]) != int(expected):
                    raise ValueError(
                        f"checkpoint {key}={payload[key]} != controller {key}={expected}"
                    )

        state = normalize_actor_state_dict(_state_dict_from_payload(payload, self.actor_index))
        inferred_hidden = _infer_hidden(state, self.contract)
        if tuple(int(width) for width in self.contract.hidden) != inferred_hidden:
            self.contract.hidden = inferred_hidden

        if not isinstance(payload, dict) or "contract" not in payload:
            if not allow_legacy:
                raise ValueError(
                    "checkpoint has no contract metadata. Legacy checkpoints require "
                    "their exact actor class, observation order, and action decoding. "
                    "Set allow_legacy:=true only if you have independently verified "
                    "those semantics."
                )

        module = build_actor(self.contract)
        try:
            module.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ValueError(
                "actor state_dict is incompatible with the configured architecture "
                f"({self.contract.obs_dim}->{list(self.contract.hidden)}->{self.contract.act_dim}). "
                "Export clean nn.Sequential weights or a TorchScript module."
            ) from exc
        module.to(self.device)
        module.eval()
        self.module = module

    @property
    def network(self):
        return self.scripted if self.scripted is not None else self.module

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        torch = self.torch
        obs = np.asarray(observation, dtype=np.float32).reshape(1, self.contract.obs_dim)
        with torch.no_grad():
            out = self.network(torch.as_tensor(obs, device=self.device))
        action = out.detach().cpu().numpy().reshape(self.contract.act_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def normalize_observation(raw: Sequence[float], contract: Contract = DEFAULT_CONTRACT) -> np.ndarray:
    obs = np.asarray(raw, dtype=np.float32).reshape(contract.obs_dim)
    if contract.clip_obs:
        obs = np.clip(obs, -1.0, 1.0)
    return obs


def action_to_velocity(action: np.ndarray, contract: Contract = DEFAULT_CONTRACT):
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(contract.act_dim), -1.0, 1.0)
    velocity = action[:3] * contract.speed
    yaw_rate = float(action[3] * contract.yaw_rate)
    return velocity, yaw_rate
