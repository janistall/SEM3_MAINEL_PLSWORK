"""Pure-PyTorch 10-drone MAPPO training driver with 4-stage curriculum and dual checkpoint export.

Exports two strictly separated checkpoint formats (PROJECT_PLAN.md §10.4):
  1. `swarm_policy_best.pt` / `swarm_policy_final.pt` : clean `weights_only=True` deployment file
     containing only `contract` metadata, bare `actor.net` state_dict, and integer dimensions.
  2. `train_state_latest.pt` / `train_state_iterNNNNN.pt` : full resumable training snapshot
     (actor + log_std, centralized critic, Adam optimizer, counters, best success, and RNG state).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from marl_training.contracts import Contract, DEFAULT_CONTRACT, make_deploy_payload
from marl_training.envs import DroneSwarmEnv, STAGE_CONFIGS
from marl_training.eval import evaluate_policy
from marl_training.mappo import MAPPOTrainer


def save_deploy_pt(
    trainer: MAPPOTrainer,
    contract: Contract,
    path: str | Path,
    *,
    steps: int,
    seed: int,
    stage: str,
    note: str,
) -> Path:
    """Save a clean `weights_only=True` deployment checkpoint for ROS 2 `ActorPolicy`."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = make_deploy_payload(
        contract,
        trainer.actor.net.state_dict(),
        algo="mappo_ctde_shared",
        training_steps=int(steps),
        seed=int(seed),
        stage=str(stage),
        note=str(note),
    )
    torch.save(payload, target)
    return target


def save_train_state_pt(
    trainer: MAPPOTrainer,
    contract: Contract,
    out_dir: str | Path,
    *,
    iteration: int,
    env_steps: int,
    stage: str,
    best_success: float,
    best_docked: float,
    best_reward: float,
    seed: int,
    keep_last: int = 3,
) -> Path:
    """Save a full resumable training snapshot (`train_state_latest.pt` + rolling `train_state_iterNNNNN.pt`)."""
    folder = Path(out_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    meta = contract.as_dict()
    meta.update(
        {
            "algo": "mappo_ctde_shared",
            "training_steps": int(env_steps),
            "seed": int(seed),
            "stage": str(stage),
        }
    )
    state = {
        "contract": meta,
        **trainer.state_dict(),
        "iteration": int(iteration),
        "env_steps": int(env_steps),
        "stage": str(stage),
        "best_success": float(best_success),
        "best_docked": float(best_docked),
        "best_reward": float(best_reward),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
    }
    latest_path = folder / "train_state_latest.pt"
    iter_path = folder / f"train_state_iter{int(iteration):05d}.pt"
    torch.save(state, latest_path)
    torch.save(state, iter_path)

    # Prune older periodic snapshots, keeping only the most recent `keep_last`
    if keep_last > 0:
        snapshots = sorted(folder.glob("train_state_iter*.pt"))
        for old_path in snapshots[:-int(keep_last)]:
            try:
                old_path.unlink()
            except OSError:
                pass
    return latest_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-drones", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--rollout-steps", type=int, default=2400)
    p.add_argument("--ppo-epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly anneal learning rate across iterations",
    )
    p.add_argument("--scenario", default="", help="Optional fixed scenario.json path")
    p.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize formations and obstacle layouts during training",
    )
    p.add_argument(
        "--shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deployment CBF shield inside training environment",
    )
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--keep-last-snapshots", type=int, default=3)
    p.add_argument(
        "--out",
        default="marl_training/runs/exp01_stage1_open",
        help="Output directory for checkpoints and metrics.jsonl",
    )
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--resume", default="", help="Resume training state from train_state_latest.pt")
    p.add_argument(
        "--init-weights",
        default="",
        help="Initialize actor/critic weights from previous curriculum stage snapshot",
    )
    p.add_argument(
        "--stage",
        default="stage1_open",
        choices=tuple(STAGE_CONFIGS.keys()),
        help="Four-stage curriculum preset (always at N=10)",
    )
    return p


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    contract = replace(DEFAULT_CONTRACT, agents=int(args.num_drones))
    scenario_path = args.scenario.strip() or None

    env = DroneSwarmEnv(
        num_drones=int(args.num_drones),
        max_steps=int(args.max_steps),
        scenario_path=scenario_path,
        shield=bool(args.shield),
        randomize=bool(args.randomize),
        contract=contract,
        seed=int(args.seed),
        stage=str(args.stage),
    )

    trainer = MAPPOTrainer(
        contract=contract,
        hidden=contract.hidden,
        lr=float(args.lr),
        ppo_epochs=int(args.ppo_epochs),
        minibatch_size=int(args.minibatch_size),
        device=device,
    )

    start_iter = 1
    env_steps_total = 0
    best_success = -1.0
    best_docked = -1.0
    best_reward = float("-inf")

    if args.init_weights:
        init_path = Path(args.init_weights).expanduser()
        payload = torch.load(str(init_path), map_location="cpu", weights_only=False)
        trainer.load_state_dict(payload, load_optimizer=False)

    if args.resume:
        resume_path = Path(args.resume).expanduser()
        payload = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        trainer.load_state_dict(payload, load_optimizer=True)
        start_iter = int(payload.get("iteration", 0)) + 1
        env_steps_total = int(payload.get("env_steps", 0))
        best_success = float(payload.get("best_success", -1.0))
        best_docked = float(payload.get("best_docked", -1.0))
        best_reward = float(payload.get("best_reward", float("-inf")))
        if "rng_torch" in payload:
            try:
                torch.set_rng_state(payload["rng_torch"].cpu())
            except Exception:
                pass
        if "rng_numpy" in payload:
            try:
                np.random.set_state(payload["rng_numpy"])
            except Exception:
                pass

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    print(
        f"Starting MAPPO training: stage={args.stage} device={device} "
        f"num_drones={args.num_drones} iters={start_iter}..{args.iters} "
        f"rollout_steps={args.rollout_steps} out={out_dir}",
        flush=True,
    )

    last_row: dict[str, Any] = {}
    for i in range(start_iter, int(args.iters) + 1):
        if args.anneal_lr and int(args.iters) > 1:
            frac = 1.0 - (i - 1) / float(args.iters)
            trainer.set_lr(max(1e-5, float(args.lr) * frac))

        rollout_seed = int(args.seed) + i * 100
        buffer, rollout_stats = trainer.collect_rollout(
            env, rollout_steps=int(args.rollout_steps), seed=rollout_seed
        )
        env_steps_total += int(args.rollout_steps)
        ppo_stats = trainer.update(buffer, rng=rng)

        row: dict[str, Any] = {
            "iter": int(i),
            "env_steps_total": int(env_steps_total),
            "stage": str(args.stage),
            "lr": float(trainer.lr),
            **rollout_stats,
            **ppo_stats,
        }

        is_eval_iter = (int(args.eval_every) > 0 and i % int(args.eval_every) == 0) or (
            i == int(args.iters)
        )
        if is_eval_iter:
            eval_metrics = evaluate_policy(
                trainer.act_deterministic,
                num_drones=int(args.num_drones),
                max_steps=int(args.max_steps),
                episodes=int(args.eval_episodes),
                stage=str(args.stage),
                shield=True,
                scenario_path=scenario_path,
                contract=contract,
            )
            for k, v in eval_metrics.items():
                row[f"eval_{k}"] = v

            if int(args.eval_every) > 0 and i % (2 * int(args.eval_every)) == 0:
                eval_off = evaluate_policy(
                    trainer.act_deterministic,
                    num_drones=int(args.num_drones),
                    max_steps=int(args.max_steps),
                    episodes=max(1, int(args.eval_episodes) // 2),
                    stage=str(args.stage),
                    shield=False,
                    scenario_path=scenario_path,
                    contract=contract,
                )
                row["eval_shield_off_success_rate"] = eval_off["success_rate"]
                row["eval_shield_off_collision_rate"] = eval_off["collision_rate"]

            curr_succ = float(eval_metrics["success_rate"])
            curr_dock = float(eval_metrics["fraction_docked"])
            curr_rew = float(row["episode_reward_mean"])
            if (curr_succ, curr_dock, curr_rew) >= (best_success, best_docked, best_reward):
                best_success = curr_succ
                best_docked = curr_dock
                best_reward = curr_rew
                save_deploy_pt(
                    trainer,
                    contract,
                    out_dir / "swarm_policy_best.pt",
                    steps=env_steps_total,
                    seed=int(args.seed),
                    stage=str(args.stage),
                    note=f"best eval_success_rate={best_success:.3f} at iter {i}",
                )

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if (int(args.checkpoint_every) > 0 and i % int(args.checkpoint_every) == 0) or (
            i == int(args.iters)
        ):
            save_train_state_pt(
                trainer,
                contract,
                out_dir,
                iteration=i,
                env_steps=env_steps_total,
                stage=str(args.stage),
                best_success=best_success,
                best_docked=best_docked,
                best_reward=best_reward,
                seed=int(args.seed),
                keep_last=int(args.keep_last_snapshots),
            )

        print(
            f"[iter {i:4d}/{args.iters}] steps={env_steps_total} stage={args.stage} "
            f"rew={row['episode_reward_mean']:.2f} docked={row['train_fraction_docked']:.2f} "
            f"pi_loss={row['policy_loss']:.4f} v_loss={row['value_loss']:.4f} "
            f"kl={row['approx_kl']:.4f}"
            + (
                f" | eval_succ={row['eval_success_rate']:.2f} "
                f"eval_dock={row['eval_fraction_docked']:.2f} "
                f"eval_err={row['eval_final_slot_error']:.2f}m "
                f"eval_inter={row['eval_mean_shield_intervention']:.4f}"
                if "eval_success_rate" in row
                else ""
            ),
            flush=True,
        )
        last_row = row

    save_deploy_pt(
        trainer,
        contract,
        out_dir / "swarm_policy_final.pt",
        steps=env_steps_total,
        seed=int(args.seed),
        stage=str(args.stage),
        note=f"final checkpoint at iter {args.iters}",
    )
    if not (out_dir / "swarm_policy_best.pt").is_file():
        save_deploy_pt(
            trainer,
            contract,
            out_dir / "swarm_policy_best.pt",
            steps=env_steps_total,
            seed=int(args.seed),
            stage=str(args.stage),
            note=f"fallback best checkpoint at iter {args.iters}",
        )
    return last_row


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
