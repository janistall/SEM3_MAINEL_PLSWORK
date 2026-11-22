"""Pure-PyTorch parameter-shared MAPPO trainer and GAE rollout buffer (`N = 10`, `obs_dim = 62`).

Key properties (PROJECT_PLAN.md §8.4 & §9):
  * Stores unclipped Gaussian `raw_action` samples in `RolloutBuffer` so PPO `log_prob` ratios remain exact,
    while clipping commands to `[-1, 1]` only when passed into `env.step()`.
  * Bootstraps truncated episodes with `V(s_next)` and zeroes the bootstrap value only on true synchronized
    `terminated` events (collision, out-of-bounds, or hold-success).
  * Cuts GAE recursion cleanly across episode boundaries (`terminated or truncated`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl_training.contracts import Contract, DEFAULT_CONTRACT
from marl_training.models import CentralizedValueNet, SwarmActorNet


@dataclass
class Minibatch:
    critic_obs: torch.Tensor
    own_obs: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    """On-policy GAE rollout buffer for `T` environment steps across `N` parameter-shared drones."""

    def __init__(
        self,
        rollout_steps: int,
        num_drones: int = 10,
        obs_dim: int = 62,
        act_dim: int = 4,
    ):
        self.rollout_steps = int(rollout_steps)
        self.num_drones = int(num_drones)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.obs = np.zeros((self.rollout_steps, self.num_drones, self.obs_dim), dtype=np.float32)
        self.raw_actions = np.zeros(
            (self.rollout_steps, self.num_drones, self.act_dim), dtype=np.float32
        )
        self.log_probs = np.zeros((self.rollout_steps, self.num_drones), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps, self.num_drones), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps, self.num_drones), dtype=np.float32)
        self.next_values = np.zeros((self.rollout_steps, self.num_drones), dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.num_drones), dtype=np.float32)

        self.advantages = np.zeros((self.rollout_steps, self.num_drones), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps, self.num_drones), dtype=np.float32)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        raw_actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise IndexError("RolloutBuffer is full")
        t = self.ptr
        self.obs[t] = obs
        self.raw_actions[t] = raw_actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.next_values[t] = next_values
        self.dones[t] = dones
        self.ptr += 1

    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """Compute GAE advantages and value targets.

        Note: `self.next_values[t]` already carries `V(s_{t+1})` for normal and truncated
        steps and `0.0` for true `terminated` steps, while `self.dones[t]` is `1.0` on
        any episode boundary (`terminated or truncated`) so GAE never leaks across resets.
        """
        gae = np.zeros(self.num_drones, dtype=np.float64)
        gamma_f = float(gamma)
        lam_f = float(gae_lambda)
        for t in reversed(range(self.ptr)):
            delta = (
                self.rewards[t].astype(np.float64)
                + gamma_f * self.next_values[t].astype(np.float64)
                - self.values[t].astype(np.float64)
            )
            non_boundary = 1.0 - self.dones[t].astype(np.float64)
            gae = delta + gamma_f * lam_f * non_boundary * gae
            self.advantages[t] = gae.astype(np.float32)
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_minibatches(
        self,
        minibatch_size: int,
        device: torch.device | str = "cpu",
        rng: np.random.Generator | None = None,
    ) -> Iterator[Minibatch]:
        steps = self.ptr
        total = steps * self.num_drones
        if total == 0:
            return

        # Build (T, N, N*obs_dim + obs_dim) critic inputs
        obs_slice = self.obs[:steps]
        global_flat = np.repeat(
            obs_slice.reshape(steps, 1, self.num_drones * self.obs_dim),
            self.num_drones,
            axis=1,
        )
        critic_inputs = np.concatenate([global_flat, obs_slice], axis=-1).reshape(
            total, self.num_drones * self.obs_dim + self.obs_dim
        )

        flat_obs = obs_slice.reshape(total, self.obs_dim)
        flat_act = self.raw_actions[:steps].reshape(total, self.act_dim)
        flat_logp = self.log_probs[:steps].reshape(total)
        flat_adv = self.advantages[:steps].reshape(total).copy()
        flat_ret = self.returns[:steps].reshape(total)
        flat_val = self.values[:steps].reshape(total)

        adv_mean = float(np.mean(flat_adv))
        adv_std = float(np.std(flat_adv))
        flat_adv = (flat_adv - adv_mean) / (adv_std + 1e-8)

        generator = rng or np.random.default_rng()
        indices = generator.permutation(total)
        mb_size = max(1, min(int(minibatch_size), total))

        dev = torch.device(device)
        for start in range(0, total, mb_size):
            batch_idx = indices[start : start + mb_size]
            yield Minibatch(
                critic_obs=torch.as_tensor(critic_inputs[batch_idx], dtype=torch.float32, device=dev),
                own_obs=torch.as_tensor(flat_obs[batch_idx], dtype=torch.float32, device=dev),
                raw_actions=torch.as_tensor(flat_act[batch_idx], dtype=torch.float32, device=dev),
                old_log_probs=torch.as_tensor(flat_logp[batch_idx], dtype=torch.float32, device=dev),
                advantages=torch.as_tensor(flat_adv[batch_idx], dtype=torch.float32, device=dev),
                returns=torch.as_tensor(flat_ret[batch_idx], dtype=torch.float32, device=dev),
                old_values=torch.as_tensor(flat_val[batch_idx], dtype=torch.float32, device=dev),
            )


class MAPPOTrainer:
    """Parameter-shared MAPPO trainer (CTDE) for the 10-drone Contract v2 swarm."""

    def __init__(
        self,
        contract: Contract = DEFAULT_CONTRACT,
        hidden: Sequence[int] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 6,
        minibatch_size: int = 1024,
        entropy_coef: float = 0.005,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        self.contract = contract
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.entropy_coef = float(entropy_coef)
        self.value_loss_coef = float(value_loss_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = torch.device(device)

        self.actor = SwarmActorNet(
            obs_dim=self.contract.obs_dim,
            act_dim=self.contract.act_dim,
            hidden=self.hidden,
        ).to(self.device)
        self.critic = CentralizedValueNet(
            num_drones=self.contract.agents,
            obs_dim=self.contract.obs_dim,
            hidden=self.hidden,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
            eps=1e-5,
        )

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    @torch.no_grad()
    def act_deterministic(self, obs_arr: np.ndarray) -> np.ndarray:
        """Evaluate deterministic Tanh policy mean for `(N, obs_dim)` or `(obs_dim,)` numpy array."""
        self.actor.eval()
        single = obs_arr.ndim == 1
        inp = np.asarray(obs_arr, dtype=np.float32)
        if single:
            inp = inp.reshape(1, self.contract.obs_dim)
        obs_t = torch.as_tensor(inp, dtype=torch.float32, device=self.device)
        out = self.actor.deterministic(obs_t).detach().cpu().numpy()
        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        return out.reshape(self.contract.act_dim) if single else out

    def collect_rollout(
        self,
        env,
        rollout_steps: int = 2400,
        seed: int | None = None,
    ) -> tuple[RolloutBuffer, dict[str, float]]:
        self.actor.eval()
        self.critic.eval()

        n = self.contract.agents
        agent_ids = list(env.possible_agents)
        buffer = RolloutBuffer(
            rollout_steps=rollout_steps,
            num_drones=n,
            obs_dim=self.contract.obs_dim,
            act_dim=self.contract.act_dim,
        )

        obs_dict, _ = env.reset(seed=seed)
        obs_mat = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)

        ep_return_running = 0.0
        completed_returns: list[float] = []
        completed_successes: list[float] = []
        completed_collisions: list[float] = []
        completed_docked: list[float] = []
        step_interventions: list[float] = []

        for _ in range(int(rollout_steps)):
            obs_t = torch.as_tensor(obs_mat, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                raw_act_t, log_prob_t = self.actor.act(obs_t)
                values_t = self.critic(obs_t)

            raw_act_np = raw_act_t.detach().cpu().numpy().astype(np.float32)
            log_prob_np = log_prob_t.detach().cpu().numpy().astype(np.float32)
            values_np = values_t.detach().cpu().numpy().astype(np.float32)

            # Clip ONLY when sending to env.step(); store unclipped raw_act_np in RolloutBuffer
            clipped_act_np = np.clip(raw_act_np, -1.0, 1.0)
            action_dict = {agent_ids[i]: clipped_act_np[i] for i in range(n)}
            next_obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)

            next_obs_mat = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(
                np.float32
            )
            rew_vec = np.array([rew_dict[a] for a in agent_ids], dtype=np.float32)
            terminated = bool(any(term_dict.values()))
            truncated = bool(any(trunc_dict.values()))
            done_flag = 1.0 if (terminated or truncated) else 0.0
            dones_vec = np.full(n, done_flag, dtype=np.float32)

            ep_return_running += float(np.mean(rew_vec))
            sample_info = info_dict[agent_ids[0]]
            step_interventions.append(
                float(np.mean([info_dict[a]["shield_intervention"] for a in agent_ids]))
            )

            if terminated:
                # True terminal event (collision / OOB / hold-success): bootstrap value is 0.0
                next_values_np = np.zeros(n, dtype=np.float32)
            else:
                # Normal step OR truncated episode: bootstrap with V(s_{t+1}) before any reset
                with torch.no_grad():
                    next_obs_t = torch.as_tensor(
                        next_obs_mat, dtype=torch.float32, device=self.device
                    )
                    next_values_np = (
                        self.critic(next_obs_t).detach().cpu().numpy().astype(np.float32)
                    )

            buffer.add(
                obs=obs_mat,
                raw_actions=raw_act_np,
                log_probs=log_prob_np,
                rewards=rew_vec,
                values=values_np,
                next_values=next_values_np,
                dones=dones_vec,
            )

            if terminated or truncated:
                completed_returns.append(ep_return_running)
                completed_successes.append(1.0 if sample_info.get("success", False) else 0.0)
                completed_collisions.append(1.0 if sample_info.get("collision", False) else 0.0)
                completed_docked.append(float(sample_info.get("fraction_docked", 0.0)))
                ep_return_running = 0.0
                obs_dict, _ = env.reset()
                obs_mat = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
            else:
                obs_mat = next_obs_mat

        buffer.compute_gae(gamma=self.gamma, gae_lambda=self.gae_lambda)
        stats = {
            "episode_reward_mean": (
                float(np.mean(completed_returns))
                if completed_returns
                else float(ep_return_running)
            ),
            "train_success_rate": (
                float(np.mean(completed_successes)) if completed_successes else 0.0
            ),
            "train_collision_rate": (
                float(np.mean(completed_collisions)) if completed_collisions else 0.0
            ),
            "train_fraction_docked": (
                float(np.mean(completed_docked))
                if completed_docked
                else float(sample_info.get("fraction_docked", 0.0))
            ),
            "train_mean_shield_intervention": (
                float(np.mean(step_interventions)) if step_interventions else 0.0
            ),
            "episodes_completed": int(len(completed_returns)),
        }
        return buffer, stats

    def update(
        self,
        buffer: RolloutBuffer,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        self.actor.train()
        self.critic.train()

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        for _ in range(self.ppo_epochs):
            for mb in buffer.get_minibatches(
                self.minibatch_size, device=self.device, rng=rng
            ):
                new_log_probs, entropy = self.actor.evaluate_actions(
                    mb.own_obs, mb.raw_actions
                )
                log_ratio = new_log_probs - mb.old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb.advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb.advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_mean = entropy.mean()

                new_values = self.critic(mb.critic_obs)
                value_loss = 0.5 * F.mse_loss(new_values, mb.returns)

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    clip_frac = float(
                        (torch.abs(ratio - 1.0) > self.clip_eps).float().mean().item()
                    )

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor_full": {
                k: v.detach().cpu() for k, v in self.actor.state_dict().items()
            },
            "critic": {
                k: v.detach().cpu() for k, v in self.critic.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(
        self, state: dict[str, Any], load_optimizer: bool = True
    ) -> None:
        if "actor_full" in state:
            self.actor.load_state_dict(state["actor_full"], strict=True)
        elif "actor" in state:
            actor_state = state["actor"]
            if "log_std" in actor_state or any(k.startswith("net.") for k in actor_state):
                self.actor.load_state_dict(actor_state, strict=False)
            else:
                self.actor.net.load_state_dict(actor_state, strict=True)
        if "critic" in state:
            self.critic.load_state_dict(state["critic"], strict=True)
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
