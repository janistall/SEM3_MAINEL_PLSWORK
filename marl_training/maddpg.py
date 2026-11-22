"""MADDPG for Ray RLlib (>= 2.40, old API stack), built on a custom TorchPolicyV2.

Design
------
* ONE parameter-shared policy ("shared_policy") for all drones -> decentralised
  execution uses only the local 23-dim observation, and the policy scales to any
  swarm size. (Ray 2.x ships no MADDPG, so the loss is implemented here.)
* Centralised training: the critic receives the concatenated observations and
  actions of all ``num_drones`` agents (ordered by agent index).
* Cooperative setting -> the critic regresses the team reward (mean of the
  per-drone rewards).
* RLlib is used for (parallel) experience collection and weight sync; a joint
  replay buffer is rebuilt from per-agent rows using ``eps_id``/``t``/``agent_index``.
"""
import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDeterministic
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

from marl_training.models import SwarmMADDPGModel  # noqa: F401  (also registered in train.py)

POLICY_ID = "shared_policy"


# --------------------------------------------------------------------------- policy
class MADDPGTorchPolicy(TorchPolicyV2):
    def __init__(self, observation_space, action_space, config):
        TorchPolicyV2.__init__(
            self, observation_space, action_space, config,
            max_seq_len=config["model"]["max_seq_len"],
        )
        self._actor_opt = torch.optim.Adam(self.model.actor_variables(), lr=config["actor_lr"])
        self._critic_opt = torch.optim.Adam(self.model.critic_variables(), lr=config["critic_lr"])

    # -- model / acting ---------------------------------------------------------
    def make_model_and_action_dist(self):
        from common.contracts import ACTION_DIM
        model = ModelCatalog.get_model_v2(
            self.observation_space, self.action_space, ACTION_DIM,
            self.config["model"], framework="torch",
        )
        return model, TorchDeterministic

    def optimizer(self):
        # Optimisers are managed manually in `maddpg_update`.
        return []

    def action_distribution_fn(self, model, *, obs_batch, state_batches=None, **kwargs):
        obs = obs_batch["obs"] if isinstance(obs_batch, dict) else obs_batch
        action = model.actor(obs.float())
        if kwargs.get("explore", True):
            action = (action + self.config["noise_std"] * torch.randn_like(action)).clamp(-1.0, 1.0)
        return action, TorchDeterministic, []

    def loss(self, model, dist_class, train_batch):
        # Unused (training happens in `maddpg_update`); dummy-batch init only.
        return sum(p.sum() for p in model.actor.parameters()) * 0.0

    # -- the actual MADDPG update -----------------------------------------------
    def maddpg_update(self, joint):
        """joint: dict of numpy arrays: obs [B,N,O], act [B,N,A], rew [B], nobs, done [B]."""
        dev = self.device
        t = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in joint.items()}
        B = t["obs"].shape[0]
        gamma, tau, m = self.config["gamma"], self.config["tau"], self.model

        obs, act, nobs = t["obs"].reshape(B, -1), t["act"].reshape(B, -1), t["nobs"]
        with torch.no_grad():
            n_act = m.target_actor(nobs.reshape(B * nobs.shape[1], -1)).reshape(B, -1)
            y = t["rew"].unsqueeze(-1) + gamma * (1.0 - t["done"].unsqueeze(-1)) * \
                m.target_critic(nobs.reshape(B, -1), n_act)
        critic_loss = F.mse_loss(m.critic(obs, act), y)
        self._critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(m.critic_variables(), 1.0)
        self._critic_opt.step()

        # Actor: all drones act with the current (shared) actor; maximise Q.
        cur_act = m.actor(t["obs"].reshape(B * t["obs"].shape[1], -1)).reshape(B, -1)
        actor_loss = -m.critic(obs, cur_act).mean()
        self._actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(m.actor_variables(), 1.0)
        self._actor_opt.step()

        m.soft_update_targets(tau)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "mean_q_target": y.mean().item()}


# --------------------------------------------------------------------- replay buffer
class JointReplayBuffer:
    def __init__(self, capacity):
        self.capacity, self.size, self.pos, self.data = capacity, 0, 0, None

    def __len__(self):
        return self.size

    def add(self, joint):
        n = len(joint["rew"])
        if self.data is None:
            self.data = {k: np.zeros((self.capacity,) + v.shape[1:], np.float32) for k, v in joint.items()}
        idx = (self.pos + np.arange(n)) % self.capacity
        for k, v in joint.items():
            self.data[k][idx] = v
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self.data.items()}


def rows_to_joint(sb, num_drones):
    """Rebuild joint (all-agent) transitions from per-agent rows of a SampleBatch."""
    eps, t, ai = (np.asarray(sb[SampleBatch.EPS_ID]), np.asarray(sb[SampleBatch.T]),
                  np.asarray(sb[SampleBatch.AGENT_INDEX]))
    order = np.lexsort((ai, t, eps))
    n_groups = len(order) // num_drones
    order = order[: n_groups * num_drones]
    g = lambda a: a[order].reshape((n_groups, num_drones) + a.shape[1:])  # noqa: E731
    eps_g, t_g, ai_g = g(eps), g(t), g(ai)
    valid = (
        (eps_g == eps_g[:, :1]).all(1) & (t_g == t_g[:, :1]).all(1)
        & (ai_g == np.arange(num_drones)).all(1)
    )
    obs, act = g(np.asarray(sb[SampleBatch.OBS])), g(np.asarray(sb[SampleBatch.ACTIONS]))
    rew, nobs = g(np.asarray(sb[SampleBatch.REWARDS])), g(np.asarray(sb[SampleBatch.NEXT_OBS]))
    term = g(np.asarray(sb[SampleBatch.TERMINATEDS]))
    return {
        "obs": obs[valid], "act": act[valid], "nobs": nobs[valid],
        "rew": rew[valid].mean(1), "done": term[valid].any(1).astype(np.float32),
    }


# ------------------------------------------------------------------ config / algorithm
class MADDPGConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or MADDPG)
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.gamma = 0.95
        self.tau = 0.01
        self.noise_std = 0.2
        self.buffer_size = 200_000
        self.batch_size = 256
        self.learning_starts = 2_000
        self.updates_per_env_step = 0.25   # gradient steps per sampled env step
        self.num_drones = 3
        self.batch_mode = "complete_episodes"
        self.clip_actions = True
        # Samples from TorchDeterministic (actor output + noise added in the policy).
        self.exploration_config = {"type": "StochasticSampling"}

    def training(self, *, actor_lr=NotProvided, critic_lr=NotProvided, gamma=NotProvided,
                 tau=NotProvided, noise_std=NotProvided, buffer_size=NotProvided,
                 batch_size=NotProvided, learning_starts=NotProvided,
                 updates_per_env_step=NotProvided, num_drones=NotProvided, **kwargs):
        super().training(**kwargs)
        for k, v in dict(actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
                         noise_std=noise_std, buffer_size=buffer_size, batch_size=batch_size,
                         learning_starts=learning_starts, updates_per_env_step=updates_per_env_step,
                         num_drones=num_drones).items():
            if v is not NotProvided:
                setattr(self, k, v)
        return self


class MADDPG(Algorithm):
    @classmethod
    def get_default_config(cls):
        return MADDPGConfig()

    @classmethod
    def get_default_policy_class(cls, config):
        return MADDPGTorchPolicy

    def setup(self, config):
        super().setup(config)
        self.buffer = JointReplayBuffer(self.config.buffer_size)
        self._env_steps = 0

    def training_step(self):
        runners = getattr(self, "env_runner_group", None) or self.workers
        batch = synchronous_parallel_sample(worker_set=runners, concat=True)
        self._env_steps += batch.env_steps()

        cfg = self.config
        joint = rows_to_joint(batch.policy_batches[POLICY_ID], cfg.num_drones)
        self.buffer.add(joint)

        policy = self.get_policy(POLICY_ID)
        results = {"buffer_size": len(self.buffer), "env_steps_total": self._env_steps,
                   "sampled_team_reward_mean": float(joint["rew"].mean())}
        if len(self.buffer) >= cfg.learning_starts:
            n_updates = max(1, int(batch.env_steps() * cfg.updates_per_env_step))
            for _ in range(n_updates):
                results.update(policy.maddpg_update(self.buffer.sample(cfg.batch_size)))
            results["num_grad_updates"] = n_updates
        runners.sync_weights(global_vars={"timestep": self._env_steps})
        return results
