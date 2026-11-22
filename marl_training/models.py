"""RLlib TorchModelV2 with a decentralised Actor and a centralised Critic (CTDE).

* Actor  : local OBS_DIM observation -> ACTION_DIM action, Tanh-bounded to [-1, 1].
* Critic : concat(obs of ALL drones, actions of ALL drones) -> scalar Q.
Layer sizes are derived from ``common.contracts`` and ``num_drones``.
Target networks (for MADDPG) are held in the same model so they travel with
the policy's weights.
"""
import copy
import os
import sys

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.contracts import ACTION_DIM, OBS_DIM  # noqa: E402


def _mlp(in_dim, out_dim, hidden, out_act=None):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class SwarmActorNet(nn.Module):
    def __init__(self, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(OBS_DIM, ACTION_DIM, hidden, nn.Tanh())  # bounded [-1, 1]

    def forward(self, obs):
        return self.net(obs)


class CentralizedCriticNet(nn.Module):
    def __init__(self, num_drones, hidden=(256, 256)):
        super().__init__()
        self.num_drones = num_drones
        self.net = _mlp(OBS_DIM * num_drones + ACTION_DIM * num_drones, 1, hidden)

    def forward(self, all_obs, all_actions):
        """all_obs: [B, N*OBS_DIM]; all_actions: [B, N*ACTION_DIM]."""
        return self.net(torch.cat([all_obs, all_actions], dim=-1))


class SwarmMADDPGModel(TorchModelV2, nn.Module):
    """Pass ``custom_model_config={"num_drones": N, "hidden": (256, 256)}``."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 num_drones=3, hidden=(256, 256), **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        assert obs_space.shape == (OBS_DIM,), f"obs shape {obs_space.shape} != ({OBS_DIM},)"
        assert action_space.shape == (ACTION_DIM,), f"action shape {action_space.shape} != ({ACTION_DIM},)"
        self.num_drones = num_drones

        self.actor = SwarmActorNet(hidden)
        self.critic = CentralizedCriticNet(num_drones, hidden)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        for p in list(self.target_actor.parameters()) + list(self.target_critic.parameters()):
            p.requires_grad_(False)

    def forward(self, input_dict, state, seq_lens):
        """Decentralised execution: only the local observation is used."""
        return self.actor(input_dict["obs_flat"].float()), state

    def value_function(self):
        raise NotImplementedError("MADDPG uses Q(s, a) via `critic`, not V(s).")

    def actor_variables(self):
        return list(self.actor.parameters())

    def critic_variables(self):
        return list(self.critic.parameters())

    @torch.no_grad()
    def soft_update_targets(self, tau):
        for net, tgt in ((self.actor, self.target_actor), (self.critic, self.target_critic)):
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)
