"""Headless kinematic evaluator and shield-ON vs. shield-OFF ablation runner for Contract v2 (`N = 10`).

Computes all 11 evaluation metrics specified in PROJECT_PLAN.md §10.3 across test modes:
  * `clean`         : baseline 10-drone scenario (7 static cylinders, no disturbances)
  * `rain`          : wind gusts + odometry noise + 20% peer telemetry dropout
  * `birds`         : 7 static cylinders + 3 crossing dynamic obstacles
  * `clutter`       : 15 static cylinders
  * `target_change` : mid-episode formation goal shift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from marl_training.contracts import ActorPolicy, Contract, DEFAULT_CONTRACT
from marl_training.envs import DroneSwarmEnv

EVAL_MODES: tuple[str, ...] = ("clean", "rain", "birds", "clutter", "target_change")


def _make_env_for_mode(
    mode: str,
    *,
    num_drones: int,
    max_steps: int,
    shield: bool,
    scenario_path: str | Path | None,
    contract: Contract,
) -> DroneSwarmEnv:
    mode_key = mode.strip().lower()
    if mode_key == "clean":
        return DroneSwarmEnv(
            num_drones=num_drones,
            max_steps=max_steps,
            scenario_path=scenario_path,
            randomize=scenario_path is None,
            shield=shield,
            num_static_range=(7, 7),
            num_dyn_range=(0, 0),
            goal_change_prob=0.0,
            domain_rand=False,
            contract=contract,
        )
    if mode_key == "rain":
        return DroneSwarmEnv(
            num_drones=num_drones,
            max_steps=max_steps,
            scenario_path=scenario_path,
            randomize=True,
            shield=shield,
            num_static_range=(7, 7),
            num_dyn_range=(0, 0),
            goal_change_prob=0.0,
            domain_rand=True,
            dropout_range=(0.20, 0.20),
            contract=contract,
        )
    if mode_key == "birds":
        return DroneSwarmEnv(
            num_drones=num_drones,
            max_steps=max_steps,
            scenario_path=scenario_path,
            randomize=True,
            shield=shield,
            num_static_range=(7, 7),
            num_dyn_range=(3, 3),
            goal_change_prob=0.0,
            domain_rand=False,
            contract=contract,
        )
    if mode_key == "clutter":
        return DroneSwarmEnv(
            num_drones=num_drones,
            max_steps=max_steps,
             randomize=True,
            shield=shield,
            num_static_range=(15, 15),
            num_dyn_range=(0, 0),
            goal_change_prob=0.0,
            domain_rand=False,
            contract=contract,
        )
    if mode_key == "target_change":
        return DroneSwarmEnv(
            num_drones=num_drones,
            max_steps=max_steps,
            randomize=True,
            shield=shield,
            num_static_range=(7, 7),
            num_dyn_range=(0, 0),
            goal_change_prob=1.0,
            domain_rand=False,
            contract=contract,
        )
    raise ValueError(f"unknown evaluation mode {mode!r}; expected one of {EVAL_MODES}")


def evaluate_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    *,
    num_drones: int = 10,
    max_steps: int = 600,
    episodes: int = 5,
    seeds: Sequence[int] | None = None,
    mode: str = "clean",
    stage: str | None = None,
    shield: bool = True,
    scenario_path: str | Path | None = None,
    contract: Contract = DEFAULT_CONTRACT,
) -> dict[str, float]:
    """Run deterministic evaluation episodes and return the 11 §10.3 metrics."""
    if stage is not None:
        env = DroneSwarmEnv.for_stage(
            stage,
            num_drones=num_drones,
            max_steps=max_steps,
            scenario_path=scenario_path,
            randomize=scenario_path is None,
            shield=shield,
            contract=contract,
        )
    else:
        env = _make_env_for_mode(
            mode,
            num_drones=num_drones,
            max_steps=max_steps,
            shield=shield,
            scenario_path=scenario_path,
            contract=contract,
        )
    if seeds is None:
        seed_list = [7 + k for k in range(int(episodes))]
    else:
        seed_list = [int(s) for s in seeds][: int(episodes)]
        while len(seed_list) < int(episodes):
            seed_list.append((seed_list[-1] + 1) if seed_list else 7)

    agent_ids = list(env.possible_agents)

    ep_success: list[float] = []
    ep_collisions: list[float] = []
    ep_fraction_docked: list[float] = []
    ep_time_to_hold: list[float] = []
    ep_final_slot_error: list[float] = []
    ep_min_pair: list[float] = []
    ep_min_obs: list[float] = []
    ep_min_bird: list[float] = []
    ep_mean_inter: list[float] = []
    ep_infeasible_frac: list[float] = []
    ep_path_len: list[float] = []
    ep_mean_jerk: list[float] = []

    for ep_seed in seed_list:
        obs_dict, _ = env.reset(seed=ep_seed)
        min_pair = float("inf")
        min_obs = float("inf")
        min_bird = float("inf")
        interventions: list[float] = []
        infeasibles: list[float] = []
        jerks: list[float] = []
        total_path = np.zeros(num_drones, dtype=np.float64)
        hold_time = float(max_steps * env.dt)
        last_info: dict[str, dict[str, Any]] = {}

        for step_idx in range(1, max_steps + 1):
            actions: dict[str, np.ndarray] = {}
            for a in agent_ids:
                act = np.asarray(policy_fn(obs_dict[a]), dtype=np.float32).reshape(
                    contract.act_dim
                )
                actions[a] = np.clip(act, -1.0, 1.0)
            obs_dict, _, term_dict, trunc_dict, last_info = env.step(actions)
            s0 = last_info[agent_ids[0]]
            min_pair = min(min_pair, float(s0["min_pair_distance"]))
            min_obs = min(min_obs, float(s0["min_obstacle_clearance"]))
            min_bird = min(min_bird, float(s0["min_bird_clearance"]))
            for idx, a in enumerate(agent_ids):
                info_a = last_info[a]
                interventions.append(float(info_a["shield_intervention"]))
                infeasibles.append(0.0 if info_a["shield_feasible"] else 1.0)
                jerks.append(float(info_a["action_jerk"]))
                total_path[idx] += float(info_a["step_path_length"])

            if s0.get("success", False):
                hold_time = float(step_idx * env.dt)
            if any(term_dict.values()) or any(trunc_dict.values()) or not env.agents:
                break

        s0 = last_info[agent_ids[0]]
        ep_success.append(1.0 if s0.get("success", False) else 0.0)
        ep_collisions.append(1.0 if s0.get("collision", False) else 0.0)
        ep_fraction_docked.append(float(s0.get("fraction_docked", 0.0)))
        ep_time_to_hold.append(hold_time)
        ep_final_slot_error.append(
            float(np.mean([last_info[a]["goal_distance"] for a in agent_ids]))
        )
        ep_min_pair.append(min_pair if np.isfinite(min_pair) else 0.0)
        ep_min_obs.append(min_obs if np.isfinite(min_obs) else 0.0)
        ep_min_bird.append(min_bird if np.isfinite(min_bird) else 0.0)
        ep_mean_inter.append(float(np.mean(interventions)) if interventions else 0.0)
        ep_infeasible_frac.append(float(np.mean(infeasibles)) if infeasibles else 0.0)
        ep_path_len.append(float(np.mean(total_path)))
        ep_mean_jerk.append(float(np.mean(jerks)) if jerks else 0.0)

    return {
        "success_rate": float(np.mean(ep_success)),
        "collision_rate": float(np.mean(ep_collisions)),
        "fraction_docked": float(np.mean(ep_fraction_docked)),
        "time_to_hold": float(np.mean(ep_time_to_hold)),
        "final_slot_error": float(np.mean(ep_final_slot_error)),
        "min_pair_distance": float(np.mean(ep_min_pair)),
        "min_obstacle_clearance": float(np.mean(ep_min_obs)),
        "min_bird_clearance": float(np.mean(ep_min_bird)),
        "mean_shield_intervention": float(np.mean(ep_mean_inter)),
        "infeasible_fraction": float(np.mean(ep_infeasible_frac)),
        "path_length": float(np.mean(ep_path_len)),
        "mean_action_jerk": float(np.mean(ep_mean_jerk)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to swarm_policy_best.pt")
    parser.add_argument("--num-drones", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 30],
        help="Seed range `[start end]` (inclusive) or explicit list of seeds",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["clean", "rain", "birds", "clutter", "target_change"],
        choices=EVAL_MODES,
    )
    parser.add_argument("--scenario", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="", help="Optional path to write JSON report")
    args = parser.parse_args(argv)

    policy = ActorPolicy(args.checkpoint, device=args.device)
    if len(args.seeds) == 2 and args.seeds[1] >= args.seeds[0]:
        seeds = list(range(int(args.seeds[0]), int(args.seeds[1]) + 1))
    else:
        seeds = [int(s) for s in args.seeds]

    scenario_path = args.scenario.strip() or None
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "contract": policy.contract.as_dict(),
        "results": {},
    }

    for mode in args.modes:
        res_on = evaluate_policy(
            policy,
            num_drones=args.num_drones,
            max_steps=args.max_steps,
            episodes=args.episodes,
            seeds=seeds,
            mode=mode,
            shield=True,
            scenario_path=scenario_path,
            contract=policy.contract,
        )
        res_off = evaluate_policy(
            policy,
            num_drones=args.num_drones,
            max_steps=args.max_steps,
            episodes=args.episodes,
            seeds=seeds,
            mode=mode,
            shield=False,
            scenario_path=scenario_path,
            contract=policy.contract,
        )
        report["results"][mode] = {"L+S": res_on, "L": res_off}
        print(
            f"[{mode:13s}] L+S success={res_on['success_rate']:.2f} "
            f"(docked={res_on['fraction_docked']:.2f}, coll={res_on['collision_rate']:.2f}, "
            f"inter={res_on['mean_shield_intervention']:.4f}) | "
            f"L (shield OFF) success={res_off['success_rate']:.2f} "
            f"(docked={res_off['fraction_docked']:.2f}, coll={res_off['collision_rate']:.2f})",
            flush=True,
        )

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
