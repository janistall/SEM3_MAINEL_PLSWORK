"""Pure-Python launch planning helpers for `swarm_bringup/launch/swarm.launch.py`.

Resolves 3/6/10-drone (or custom N-drone) scenarios and worlds, extracts Contract v2
limits from learned checkpoints without requiring checkpoints in `zero`/`classical`
modes, syncs takeoff altitudes, and prepares staggered PX4 + adapter + controller plans.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_control.contract import DEFAULT_CONTRACT, apply_contract_metadata

from .scenario import Scenario, generate, write_world

SUPPORTED_MODES = ("learned", "classical", "zero")


@dataclass(frozen=True)
class DroneLaunchSpec:
    index: int
    drone_id: str
    px4_namespace: str
    system_id: int
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    pose_str: str
    px4_start_delay: float
    takeoff_altitude: float


@dataclass(frozen=True)
class SwarmLaunchPlan:
    scenario_path: Path
    world_path: Path
    num_drones: int
    agent_ids: list[str]
    mode: str
    checkpoint: str
    use_supervisor: bool
    speed: float
    accel: float
    yaw_rate: float
    quorum_min_peers: int
    control_rate: float
    telemetry_rate: float
    headless: bool
    auto_arm: bool
    drones: list[DroneLaunchSpec]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def resolve_scenario_and_world(
    package_root: Path,
    *,
    scenario_arg: str = "",
    world_arg: str = "",
    drones_arg: str | int = "",
    seed: int = 7,
    obstacles: int = 7,
    shape: str = "",
    spacing: float = 1.6,
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rain: bool = False,
    runtime_dir: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve consistent scenario JSON and Gazebo SDF world paths.

    - Uses the packaged 10-drone `config/scenario.json` & `worlds/swarm.sdf` by default
      when no custom paths, drone-count override, wind, or rain are requested.
    - When `drones` is set to another count (e.g. 3 or 6) without custom `scenario`/`world`,
      generates a consistent runtime pair into `runtime_dir` (default `/tmp/swarm_bringup_runtime_*`).
    - When custom `wind` or `rain` is requested without a custom `world`, writes a runtime
      world SDF matching the resolved scenario so the packaged `worlds/swarm.sdf` stays clean.
    """
    scenario_clean = str(scenario_arg).strip()
    world_clean = str(world_arg).strip()
    drones_clean = str(drones_arg).strip()
    requested_drones = int(drones_clean) if drones_clean else None
    if requested_drones is not None and requested_drones < 1:
        raise ValueError(f"drones must be >= 1, got {requested_drones}")

    wx, wy, wz = (float(wind[0]), float(wind[1]), float(wind[2]))
    has_env_fx = bool(rain) or any(abs(v) > 1e-9 for v in (wx, wy, wz))

    default_scenario_path = package_root / "config" / "scenario.json"
    default_world_path = package_root / "worlds" / "swarm.sdf"

    if scenario_clean:
        scenario_path = Path(scenario_clean).expanduser()
        if not scenario_path.is_file():
            raise FileNotFoundError(f"Scenario JSON not found: {scenario_path}")
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        starts = payload.get("starts", [])
        goals = payload.get("goals", [])
        if len(starts) < 1 or len(starts) != len(goals):
            raise ValueError(
                f"Scenario must have matching non-empty starts and goals, got "
                f"{len(starts)} starts and {len(goals)} goals"
            )
        if requested_drones is not None and len(starts) != requested_drones:
            raise ValueError(
                f"drones:={requested_drones} does not match scenario count {len(starts)} "
                f"in {scenario_path}"
            )
        if world_clean:
            world_path = Path(world_clean).expanduser()
            if not world_path.is_file():
                raise FileNotFoundError(f"Gazebo world SDF not found: {world_path}")
        elif has_env_fx:
            out_dir = runtime_dir or (
                Path(tempfile.gettempdir()) / f"swarm_bringup_runtime_{len(starts)}d_fx"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            world_path = out_dir / "swarm.sdf"
            write_world(Scenario(**payload), world_path, wind=(wx, wy, wz), rain=bool(rain))
        else:
            sibling_world = scenario_path.parent / "swarm.sdf"
            world_path = sibling_world if sibling_world.is_file() else default_world_path
        return scenario_path, world_path, payload

    # No explicit scenario path: inspect packaged scenario first.
    packaged_payload = json.loads(default_scenario_path.read_text(encoding="utf-8"))
    packaged_n = len(packaged_payload.get("starts", []))
    target_n = requested_drones if requested_drones is not None else packaged_n

    if target_n == packaged_n and not has_env_fx and not world_clean:
        return default_scenario_path, default_world_path, packaged_payload

    if target_n == packaged_n and world_clean:
        world_path = Path(world_clean).expanduser()
        if not world_path.is_file():
            raise FileNotFoundError(f"Gazebo world SDF not found: {world_path}")
        return default_scenario_path, world_path, packaged_payload

    if target_n == packaged_n and has_env_fx:
        out_dir = runtime_dir or (
            Path(tempfile.gettempdir()) / f"swarm_bringup_runtime_{target_n}d_fx"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        world_path = out_dir / "swarm.sdf"
        write_world(
            Scenario(**packaged_payload), world_path, wind=(wx, wy, wz), rain=bool(rain)
        )
        return default_scenario_path, world_path, packaged_payload

    # Generate a consistent scenario + world pair for the requested drone count (e.g. 3 or 6).
    resolved_shape = shape.strip().lower() if shape.strip() else (
        "triangle" if target_n == 3 else "ring"
    )
    out_dir = runtime_dir or (
        Path(tempfile.gettempdir())
        / f"swarm_bringup_runtime_{target_n}d_seed{int(seed)}_obs{int(obstacles)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = generate(
        seed=int(seed),
        count=int(obstacles),
        num_drones=target_n,
        shape=resolved_shape,
        spacing=float(spacing),
    )
    scenario_path = out_dir / "scenario.json"
    world_path = Path(world_clean).expanduser() if world_clean else (out_dir / "swarm.sdf")
    generated.save(scenario_path)
    if not world_clean:
        write_world(generated, world_path, wind=(wx, wy, wz), rain=bool(rain))
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    return scenario_path, world_path, payload


def resolve_checkpoint_and_limits(
    package_root: Path,
    *,
    mode: str,
    checkpoint_arg: str,
    num_drones: int,
    allow_legacy: bool = False,
) -> tuple[str, float, float, float]:
    """Resolve checkpoint path and Contract v2 `(speed, accel, yaw_rate)` limits.

    - `zero` and `classical` modes NEVER require a checkpoint and return empty `""` plus
      Contract v2 defaults `(1.5, 2.0, 0.6)`.
    - `learned` mode requires an explicit v2 checkpoint (or packaged `swarm_policy_best.pt`)
      and extracts `speed`, `accel`, `yaw_rate` from its `contract` metadata when present.
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}"
        )

    speed = float(DEFAULT_CONTRACT.speed)
    accel = float(DEFAULT_CONTRACT.accel)
    yaw_rate = float(DEFAULT_CONTRACT.yaw_rate)

    if mode_key in ("zero", "classical"):
        return "", speed, accel, yaw_rate

    ckpt_clean = str(checkpoint_arg).strip()
    if ckpt_clean:
        ckpt_path = Path(ckpt_clean).expanduser()
    else:
        ckpt_path = package_root / "checkpoints" / "swarm_policy_best.pt"

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"mode:=learned requires a Contract v2 checkpoint, but none was found at "
            f"{ckpt_path}. Pass checkpoint:=/path/to/swarm_policy_best.pt or use "
            f"mode:=zero / mode:=classical."
        )

    # Read v2 contract metadata safely with weights_only=True when PyTorch is available.
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        payload = None
        try:
            loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            # Allow TorchScript checkpoints to pass through to ActorPolicy's torch.jit.load path.
            payload = None

        if isinstance(payload, dict):
            if "contract" in payload:
                meta = payload["contract"]
                if not isinstance(meta, dict):
                    raise ValueError("checkpoint 'contract' metadata must be a dict")
                base = DEFAULT_CONTRACT
                if num_drones != base.agents:
                    from dataclasses import replace

                    base = replace(base, agents=int(num_drones))
                resolved = apply_contract_metadata(base, meta)
                speed = float(resolved.speed)
                accel = float(resolved.accel)
                yaw_rate = float(resolved.yaw_rate)
            elif not allow_legacy:
                raise ValueError(
                    f"Checkpoint {ckpt_path} has no Contract v2 metadata. "
                    f"Set allow_legacy_checkpoint:=true only after independently verifying semantics."
                )

    return str(ckpt_path), speed, accel, yaw_rate


def build_launch_plan(
    package_root: Path,
    *,
    scenario_arg: str = "",
    world_arg: str = "",
    drones_arg: str | int = "",
    mode: str = "learned",
    checkpoint_arg: str = "",
    allow_legacy: bool = False,
    use_supervisor: bool = False,
    quorum_min_peers: int = 3,
    control_rate: float = 10.0,
    telemetry_rate: float = 10.0,
    takeoff_altitude_arg: str = "",
    px4_stagger: float = 2.0,
    headless: bool = False,
    auto_arm: bool = False,
    seed: int = 7,
    obstacles: int = 7,
    shape: str = "",
    spacing: float = 1.6,
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rain: bool = False,
    runtime_dir: Path | None = None,
) -> SwarmLaunchPlan:
    scenario_path, world_path, payload = resolve_scenario_and_world(
        package_root,
        scenario_arg=scenario_arg,
        world_arg=world_arg,
        drones_arg=drones_arg,
        seed=seed,
        obstacles=obstacles,
        shape=shape,
        spacing=spacing,
        wind=wind,
        rain=rain,
        runtime_dir=runtime_dir,
    )
    starts = payload["starts"]
    goals = payload["goals"]
    num_drones = len(starts)
    if num_drones < 1 or len(goals) != num_drones:
        raise ValueError(
            f"Scenario must define matching non-empty starts ({len(starts)}) and goals ({len(goals)})"
        )

    mode_key = str(mode).strip().lower()
    ckpt_str, speed, accel, yaw_rate = resolve_checkpoint_and_limits(
        package_root,
        mode=mode_key,
        checkpoint_arg=checkpoint_arg,
        num_drones=num_drones,
        allow_legacy=allow_legacy,
    )

    # Effective supervisor state mirrors node.py (§11.1): off in learned & zero, on in classical.
    effective_supervisor = True if mode_key == "classical" else False
    if mode_key == "learned" and use_supervisor:
        effective_supervisor = False

    takeoff_override = (
        float(str(takeoff_altitude_arg).strip())
        if str(takeoff_altitude_arg).strip()
        else None
    )
    stagger = max(0.0, float(px4_stagger))

    agent_ids = [f"drone_{i}" for i in range(num_drones)]
    drones: list[DroneLaunchSpec] = []
    for idx, (start, goal) in enumerate(zip(starts, goals)):
        s_tuple = (float(start[0]), float(start[1]), float(start[2]))
        g_tuple = (float(goal[0]), float(goal[1]), float(goal[2]))
        takeoff_alt = takeoff_override if takeoff_override is not None else g_tuple[2]
        drones.append(
            DroneLaunchSpec(
                index=idx,
                drone_id=agent_ids[idx],
                px4_namespace=f"/uav_{idx}",
                system_id=idx + 1,
                start=s_tuple,
                goal=g_tuple,
                pose_str=f"{s_tuple[0]},{s_tuple[1]},0.0,0,0,0",
                px4_start_delay=6.0 + stagger * idx,
                takeoff_altitude=float(takeoff_alt),
            )
        )

    effective_quorum = min(max(0, int(quorum_min_peers)), max(0, num_drones - 1))
    resolved_telemetry_rate = float(telemetry_rate)
    if resolved_telemetry_rate <= 0.0:
        raise ValueError(f"telemetry_rate must be > 0, got {telemetry_rate}")

    return SwarmLaunchPlan(
        scenario_path=scenario_path,
        world_path=world_path,
        num_drones=num_drones,
        agent_ids=agent_ids,
        mode=mode_key,
        checkpoint=ckpt_str,
        use_supervisor=effective_supervisor,
        speed=speed,
        accel=accel,
        yaw_rate=yaw_rate,
        quorum_min_peers=effective_quorum,
        control_rate=float(control_rate),
        telemetry_rate=resolved_telemetry_rate,
        headless=bool(headless),
        auto_arm=bool(auto_arm),
        drones=drones,
    )
