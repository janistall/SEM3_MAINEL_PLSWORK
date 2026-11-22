"""Pure-PyTorch parameter-shared MAPPO actor and centralised value critic for Contract v2 (`N = 10`).

* `SwarmActorNet`       : decentralised `obs_dim (62) -> 256 -> 256 -> act_dim (4)` Tanh MLP (`self.net`)
                          plus a learnable 4-D `log_std` (clamped to `[-2.0, 0.0]`) for Gaussian exploration.
                          `self.net.state_dict()` exports bare `nn.Sequential` keys (`0.weight`, `2.weight`,
                          `4.weight`) that load directly into ROS 2 `ActorPolicy(..., weights_only=True)`.
* `CentralizedValueNet` : CTDE state-value critic `V(s_all, o_i)` taking the concatenated observations of all
                          `N` drones (`10 * 62 = 620`) plus own observation (`62`) -> `682 -> 256 -> 256 -> 1`.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from marl_training.contracts import ACT_DIM, N_AGENTS, OBS_DIM


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (256, 256),
    out_act: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(in_dim)
    for h in hidden:
        layers += [nn.Linear(last, int(h)), nn.ReLU()]
        last = int(h)
    layers.append(nn.Linear(last, int(out_dim)))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class SwarmActorNet(nn.Module):
    """Parameter-shared decentralised actor for 10-drone MAPPO."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden: Sequence[int] = (256, 256),
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.net = _mlp(self.obs_dim, self.act_dim, self.hidden, nn.Tanh())
        self.log_std = nn.Parameter(torch.full((self.act_dim,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(obs)
        std = self.log_std.clamp(-2.0, 0.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample unclipped `raw_action` and return `(raw_action, summed_log_prob)`."""
        distribution = self.dist(obs)
        raw_action = distribution.sample()
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        return raw_action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(summed_log_prob, summed_entropy)` for unclipped `raw_actions`."""
        distribution = self.dist(obs)
        log_prob = distribution.log_prob(raw_actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralizedValueNet(nn.Module):
    """Centralised CTDE state-value critic `V(s_all, o_i)` (682-D input for N=10, obs_dim=62)."""

    def __init__(
        self,
        num_drones: int = N_AGENTS,
        obs_dim: int = OBS_DIM,
        hidden: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.num_drones = int(num_drones)
        self.obs_dim = int(obs_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.in_dim = self.obs_dim * self.num_drones + self.obs_dim
        self.net = _mlp(self.in_dim, 1, self.hidden)

    def build_critic_inputs(self, obs_all: torch.Tensor) -> torch.Tensor:
        """Convert `(..., N, obs_dim)` joint observations into `(..., N, N*obs_dim + obs_dim)`."""
        if obs_all.shape[-2:] != (self.num_drones, self.obs_dim):
            raise ValueError(
                f"expected obs_all ending with ({self.num_drones}, {self.obs_dim}), "
                f"got {tuple(obs_all.shape)}"
            )
        prefix = obs_all.shape[:-2]
        global_flat = obs_all.reshape(*prefix, 1, self.num_drones * self.obs_dim).expand(
            *prefix, self.num_drones, self.num_drones * self.obs_dim
        )
        return torch.cat([global_flat, obs_all], dim=-1)

    def forward(
        self, global_obs: torch.Tensor, own_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        if own_obs is not None:
            joint = torch.cat([global_obs, own_obs], dim=-1)
        elif global_obs.shape[-1] == self.in_dim:
            joint = global_obs
        else:
            joint = self.build_critic_inputs(global_obs)
        return self.net(joint).squeeze(-1)
