"""Run MADDPG on DroneSwarmEnv with Ray RLlib.

Usage (from the repo root):
    python -m marl_training.train --iters 200 --num-drones 3
Policy checkpoints (.pt) are written ONLY to marl_training/weights/.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import ray  # noqa: E402
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from common.contracts import ACTION_DIM, OBS_DIM  # noqa: E402
from marl_training.envs import DroneSwarmEnv  # noqa: E402
from marl_training.maddpg import POLICY_ID, MADDPGConfig  # noqa: E402
from marl_training.models import SwarmMADDPGModel  # noqa: E402

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
ENV_NAME = "drone_swarm_v0"
MODEL_NAME = "swarm_maddpg_model"


def find_key(d, key):
    """Recursively look up ``key`` in the (nested) RLlib result dict."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def save_pt(algo, num_drones, tag):
    """Save actor/critic weights as .pt files inside marl_training/weights/."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model = algo.get_policy(POLICY_ID).model
    ckpt = {
        "actor": {k: v.cpu() for k, v in model.actor.state_dict().items()},
        "critic": {k: v.cpu() for k, v in model.critic.state_dict().items()},
        "obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "num_drones": num_drones,
    }
    path = os.path.join(WEIGHTS_DIR, f"maddpg_{tag}.pt")
    torch.save(ckpt, path)
    # actor-only file: the only thing needed for decentralised deployment
    torch.save({"actor": ckpt["actor"], "obs_dim": OBS_DIM, "action_dim": ACTION_DIM},
               os.path.join(WEIGHTS_DIR, "actor_latest.pt"))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--num-drones", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-env-runners", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--learning-starts", type=int, default=2000)
    p.add_argument("--collision-penalty", type=float, default=50.0)
    args = p.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env(ENV_NAME, lambda cfg: ParallelPettingZooEnv(DroneSwarmEnv(**cfg)))
    ModelCatalog.register_custom_model(MODEL_NAME, SwarmMADDPGModel)

    config = (
        MADDPGConfig()
        .environment(ENV_NAME, env_config={
            "num_drones": args.num_drones,
            "max_steps": args.max_steps,
            "collision_penalty": args.collision_penalty,
        })
        .framework("torch")
        .env_runners(num_env_runners=args.num_env_runners, rollout_fragment_length=args.max_steps)
        .multi_agent(policies={POLICY_ID},
                     policy_mapping_fn=lambda agent_id, *a, **kw: POLICY_ID)
        .training(num_drones=args.num_drones, learning_starts=args.learning_starts,
                  model={"custom_model": MODEL_NAME,
                         "custom_model_config": {"num_drones": args.num_drones}})
    )
    # MADDPG here runs on the (old) RLlib API stack with TorchModelV2.
    config.api_stack(enable_rl_module_and_learner=False,
                     enable_env_runner_and_connector_v2=False)

    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
    for i in range(1, args.iters + 1):
        res = algo.train()
        reward = find_key(res, "episode_reward_mean")
        if reward is None:
            reward = find_key(res, "sampled_team_reward_mean")
        print(f"[iter {i:4d}] steps={find_key(res, 'env_steps_total')} reward={reward} "
              f"critic_loss={find_key(res, 'critic_loss')} actor_loss={find_key(res, 'actor_loss')}",
              flush=True)
        if i % args.checkpoint_every == 0:
            print("  saved", save_pt(algo, args.num_drones, f"iter{i:05d}"), flush=True)
    print("final:", save_pt(algo, args.num_drones, "final"))
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
