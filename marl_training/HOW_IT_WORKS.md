# How `marl_training` works

`marl_training` is an offline Ray RLlib training stack for a cooperative 3-drone swarm. It trains a shared actor with a centralized critic (CTDE) and exports PyTorch `.pt` weights. It has no ROS/rclpy dependency; it is a separate simulation environment from the one in `src/`.

The code implements a MADDPG-style algorithm on Ray's **old API stack** (`TorchPolicyV2`, `TorchModelV2`), because Ray does not ship MADDPG.

## Files

| File | Role |
| --- | --- |
| `train.py` | CLI, Ray/RLlib registration, training loop, checkpoint saving |
| `envs.py` | `DroneSwarmEnv` PettingZoo `ParallelEnv`, observation builder, dynamics, rewards |
| `models.py` | Actor, centralized critic, target networks, RLlib `TorchModelV2` wrapper |
| `maddpg.py` | Custom policy, MADDPG losses, joint replay buffer, RLlib `Algorithm` |
| `contracts.py` | Shared dimensions, physical limits, `RawDroneState`, `scale_action()` |
| `requirements.txt` | Ray RLlib, PyTorch, Gymnasium, PettingZoo, NumPy, etc. |
| `weights/` | Exported actor/critic checkpoints |
| `README.md` | One-line scratch note; contains no usage documentation |

### `contracts.py`

`marl_training/contracts.py` is the single source of truth for the training-side contract. The other modules import it as `from common.contracts import ...`; that import name does not match the file's current location, so it must be made importable under `common.contracts` when running.

It defines the observation/action dimensions, the velocity/yaw limits, the communication radius, altitude limit, and map extent listed under “Contract constants” below.

`RawDroneState` is a dataclass with:

| Field | Meaning |
| --- | --- |
| `drone_id` | agent identifier |
| `position` | NumPy `[x, y, z]` |
| `velocity` | NumPy `[vx, vy, vz]` |
| `yaw` | heading in radians |
| `is_active` | defaults to `True`; inactive neighbors are ignored |

`scale_action()` clips each normalized action component to `[-1, 1]`, then converts it to physical units:

```text
[vx, vy, vz, yaw_rate] = [a0*3.0, a1*3.0, a2*1.5, a3*(pi/2)]
```

The result is `float32`. `contracts.py` does not build observations; that logic lives in `envs.py::build_observation()`.

## End-to-end flow

```text
Ray Algorithm (MADDPG)
  -> ParallelPettingZooEnv(DroneSwarmEnv)
  -> per-drone 23-D observation, 4-D normalized action
  -> one shared actor policy (decentralized execution)
  -> old-API SampleBatch (per-agent rows)
  -> rows_to_joint() groups rows into joint transitions
  -> JointReplayBuffer (uniform joint replay)
  -> maddpg_update(): centralized critic + target networks
  -> sync weights to env runners
  -> save_pt(): actor/critic .pt files in weights/
```

## 1. Environment

`DroneSwarmEnv` is a lightweight point-mass simulation. It does not use Gazebo, PX4, or the ROS controller.

### Contract constants

From `marl_training/contracts.py`:

| Constant | Value | Meaning |
| --- | --- | --- |
| `OBS_DIM` | 23 | observation size |
| `ACTION_DIM` | 4 | `[vx, vy, vz, yaw_rate]` |
| `MAX_XY_VELOCITY` | 3.0 m/s | horizontal action scale |
| `MAX_Z_VELOCITY` | 1.5 m/s | vertical action scale |
| `MAX_YAW_RATE` | π/2 rad/s | yaw-rate action scale |
| `COMMS_RADIUS` | 15.0 m | neighbor cutoff |
| `MAX_ALTITUDE` | 10.0 m | z clamp / normalization |
| `MAP_EXTENT` | 50.0 m | target-observation and boundary scale |

`scale_action()` clips the policy output to `[-1, 1]` and multiplies by the limits above.

### Observation layout (per drone)

`envs.py::build_observation()`:

| Index | Contents |
| --- | --- |
| 0–2 | `(target - own_position) / MAP_EXTENT`, clipped |
| 3–5 | `own_velocity / [3, 3, 1.5]`, clipped |
| 6 | `sin(yaw)` |
| 7 | `cos(yaw)` |
| 8 | `z / MAX_ALTITUDE`, clipped to `[0, 1]` |
| 9–11 | nearest active neighbor: relative position `/ COMMS_RADIUS` |
| 12–14 | nearest active neighbor: relative velocity `/ (2 * [3, 3, 1.5])` |
| 15 | neighbor-valid flag (`1` if within `COMMS_RADIUS`) |
| 16–18 | second-nearest active neighbor: relative position |
| 19–21 | second-nearest active neighbor: relative velocity |
| 22 | second-neighbor-valid flag |

Only active neighbors are considered. They are sorted by distance; the two nearest are encoded. If the nearest is beyond `COMMS_RADIUS`, the remaining neighbor blocks stay zero.

Important: all drones share **one** random target. This is not a per-drone formation/slot assignment.

### Reset

`reset(seed=...)`:

- Spawns every drone uniformly in `[-10, 10]` in x/y (`0.2 * MAP_EXTENT`) and in `[2, 8]` in z (`0.2–0.8 * MAX_ALTITUDE`).
- Sets initial velocity to zero and a random yaw.
- Samples a shared target uniformly in `[-20, 20]` in x/y (`0.4 * MAP_EXTENT`) and `[2, 8]` in z.
- Agent IDs are `drone_1 ... drone_N` (not `drone_0 ... drone_N-1`).
- If `seed=None`, the RNG is not reseeded.

### Step dynamics

For each drone:

```text
cmd_velocity = scale_action(action)[:3]
velocity     = velocity + 0.5 * (cmd_velocity - velocity)
position     = position + velocity * dt            # dt = 0.1 s
z            = clip(z, 0, MAX_ALTITUDE)
yaw          = wrap(yaw + cmd_yaw_rate * dt)
```

The 0.5 factor is a first-order velocity lag, not a physics model. Yaw changes but does **not** affect translation; the point mass moves directly in world x/y/z.

### Reward

Per drone, per step:

```text
r_i = (previous_distance_to_target - current_distance_to_target)
      + 0.1 if distance_to_target < 1.0
      - 50 for each other drone within 1.0 m
      - 10 if |x_i| > MAP_EXTENT or |y_i| > MAP_EXTENT
```

- The progress term is measured in metres per step.
- Collision penalties stack: being within 1 m of both other drones costs `-100` in that step.
- There is no obstacle model and no ground/ceiling crash penalty; z is simply clamped.
- Boundary penalty applies only to x/y; there are no walls.

### Episode semantics

- `terminations` is always `False`.
- At `step_count >= max_steps` (default 500), the episode is truncated for every drone and `agents` is cleared.
- There is no success termination and no early collision termination; drones stay in the episode after reaching the target.

## 2. Networks

`models.py`:

### Actor (`SwarmActorNet`)

```text
23 -> Linear/ReLU 256 -> Linear/ReLU 256 -> Linear 4 -> Tanh
```

- Output is bounded to `[-1, 1]`.
- One actor is shared by every drone, so execution only needs the local 23-D observation.
- `hidden=(256, 256)` is the default.

### Centralized critic (`CentralizedCriticNet`)

```text
concat(all N observations, all N actions) -> 256 -> 256 -> 1
```

For `N = 3`, the input is `3*23 + 3*4 = 81`. It outputs a scalar joint Q-value. The critic is used only during training; it is not needed for deployment.

### `SwarmMADDPGModel`

- `TorchModelV2` + `nn.Module` wrapper used by RLlib.
- Holds `actor`, `critic`, and frozen `target_actor` / `target_critic` copies.
- Asserts `obs_space.shape == (23,)` and `action_space.shape == (4,)`.
- `forward()` runs only the actor on the local observation.
- `value_function()` raises `NotImplementedError`; MADDPG uses Q(s, a), not V(s).
- `soft_update_targets(tau)` Polyak-averages both target networks.
- Target networks are inside the model, so they move with RLlib weight synchronization.

## 3. MADDPG update

`maddpg.py::MADDPGTorchPolicy.maddpg_update()` receives a joint batch:

```text
obs  [B, N, 23]   act [B, N, 4]
nobs [B, N, 23]   rew [B]        done [B]
```

With shared actor `mu` and centralized critic `Q`:

```text
next_actions = target_actor(nobs)                  # all drones
y            = rew + gamma * (1 - done) * target_critic(nobs, next_actions)

critic_loss  = MSE(critic(obs, act), y)

current_actions = actor(obs)                       # all drones
actor_loss      = -critic(obs, current_actions).mean()
```

Implementation details:

- The actor is updated through the centralized critic using the current actions of **all** drones. Because the actor is shared, this is parameter-shared CTDE rather than one independent actor per drone.
- The team reward is the mean of per-drone rewards.
- Exploration adds Gaussian noise `N(0, noise_std)` to actor output during acting and clamps to `[-1, 1]`.
- Optimizers are created manually: Adam with `actor_lr=1e-4`, `critic_lr=1e-3`.
- Gradients are clipped to norm 1.0 for both actor and critic.
- Targets are soft-updated after each update with `tau=0.01`.
- There is no target-policy smoothing and no discrete action handling.

The policy overrides `action_distribution_fn`, returns `TorchDeterministic`, and adds noise itself. `loss()` is a dummy because RLlib's normal trainer is bypassed. `optimizer()` returns `[]` because optimizers are managed manually.

## 4. Joint replay and row grouping

RLlib collects per-agent rows in a `SampleBatch`. `rows_to_joint()` reconstructs joint transitions:

1. Read `EPS_ID`, `T`, and `AGENT_INDEX`.
2. Sort with `lexsort((agent_index, t, eps_id))`.
3. Group every `num_drones` rows.
4. Keep a group only if all rows share the same `eps_id` and `t` and the agent indices are exactly `0 .. N-1`.
5. Drop incomplete groups at the end.
6. Set `rew = mean(rewards across drones)` and `done = any(TERMINATEDS)`.

`JointReplayBuffer`:

- Capacity 200,000 joint transitions.
- Circular overwrite; uniform random sampling.
- Each entry stores `obs`, `act`, `nobs`, `rew`, `done`.
- The environment only truncates at 500 steps and never sets terminations; `rows_to_joint` reads `TERMINATEDS`, not truncations. Whether the final transition is bootstrapped depends on how the RLlib adapter records truncation. There is no early/success termination either way.

## 5. Algorithm loop and defaults

`MADDPG.training_step()`:

1. Sample a batch with `synchronous_parallel_sample` from the env runner group (or old `self.workers`).
2. Add one training step's env-step count to `_env_steps`.
3. Convert the batch to joint transitions and append to the replay buffer.
4. If `buffer_size >= learning_starts`, run
   `max(1, int(env_steps_in_batch * updates_per_env_step))` gradient steps.
5. Sync weights to runners and return metrics (`buffer_size`, `env_steps_total`, reward/loss values).

`MADDPGConfig` defaults:

| Setting | Value |
| --- | --- |
| `actor_lr` | `1e-4` |
| `critic_lr` | `1e-3` |
| `gamma` | `0.95` |
| `tau` | `0.01` |
| `noise_std` | `0.2` |
| `buffer_size` | `200_000` |
| `batch_size` | `256` |
| `learning_starts` | `2_000` env steps |
| `updates_per_env_step` | `0.25` |
| `num_drones` | `3` |
| `batch_mode` | `complete_episodes` |
| `clip_actions` | `True` |
| exploration | `StochasticSampling` |

`train.py` sets `rollout_fragment_length = max_steps` (500 by default), so with the default one local runner each iteration collects roughly one full episode (500 env steps) and performs about `500 * 0.25 = 125` gradient steps.

The config explicitly uses the old API stack:

```python
config.api_stack(enable_rl_module_and_learner=False,
                 enable_env_runner_and_connector_v2=False)
```

A single shared policy (`POLICY_ID = "shared_policy"`) is mapped to every agent.

## 6. Running

From the repository root:

```bash
python -m pip install -r marl_training/requirements.txt

# `contracts.py` must be importable as `common.contracts`
# (currently it is at `marl_training/contracts.py`)
python -m marl_training.train \
  --iters 240 \
  --num-drones 3 \
  --max-steps 500 \
  --checkpoint-every 20
```

CLI defaults:

| Flag | Default | Effect |
| --- | --- | --- |
| `--iters` | 200 | training iterations |
| `--num-drones` | 3 | agents in the environment and critic |
| `--max-steps` | 500 | episode length / rollout fragment length |
| `--num-env-runners` | 0 | 0 uses the local worker; >0 enables parallel sampling |
| `--checkpoint-every` | 20 | save interval in iterations |
| `--learning-starts` | 2000 | env steps before gradient updates begin |
| `--collision-penalty` | 50.0 | per-neighbor collision penalty |

There is no evaluation loop, no test suite, and no CI for this folder. Training logs one line per iteration and writes checkpoints only to `marl_training/weights/`.

## 7. Checkpoints

`save_pt()` writes:

- `maddpg_iterNNNNN.pt` every `checkpoint_every` iterations:
  `{actor, critic, obs_dim, action_dim, num_drones}` state dicts.
- `maddpg_final.pt` after the loop.
- `actor_latest.pt` on every save:
  `{actor, obs_dim, action_dim}` only; this is the deployment-oriented file.

The uploaded weights contain checkpoints from iteration 20 through 240 plus `actor_latest.pt`. All full checkpoints have 256-wide actor and critic layers; `actor_latest.pt` has the same actor.

Checkpoints contain **no** optimizer state, target-network state, replay buffer, config, RNG state, or contract metadata. They are inference exports, not resumable training snapshots.

Observed architecture in the uploaded files:

- Actor: `23 -> 256 -> 256 -> 4`.
- Critic: `81 -> 256 -> 256 -> 1` (for 3 drones).

## 8. Relationship to the ROS 2 controller

The training environment and `src/swarm_control` both use 23-D observations and 4-D actions, but the layouts and scales are **not the same**. A `.pt` from this folder is not a drop-in replacement for the deployed policy.

| Aspect | `marl_training` | `src/swarm_control` |
| --- | --- | --- |
| Target | one shared random point | per-drone assigned slot/goal |
| Observation layout | target delta `/ 50`; velocity `/ [3, 3, 1.5]`; `sin/cos(yaw)`; altitude `/10`; two neighbor blocks (`rel pos /15`, `rel vel /[6, 6, 3]`, valid flag) | goal delta `/ 15`; velocity `/ 0.8`; two peer blocks (`rel pos /15`, `rel vel /1.6`); nearest-obstacle delta and clearance `/15`; `sin/cos(yaw)` |
| Action scaling | `[3, 3, 1.5]`, yaw `π/2` | `[0.8, 0.8, 0.8]`, yaw `0.6` |
| Agent IDs | `drone_1 ... drone_N` | `drone_0 ... drone_2` |
| Obstacles | none | cylinder obstacles in observation and shield |
| Safety layer | none | supervisor + CBF velocity shield |
| Default network | 256×256 | 128×128 in packaged `swarm_macro_policy.pt` |

The observation layouts use the same first six quantities but with different scales; from index 6 onward the fields mean different things.

Consequences:

- The ROS `ActorPolicy` loader may load the actor tensor shapes (it infers 256×256 from the state dict), but the policy would receive observations with different meanings and produce actions with different scales.
- The exported files have no `contract` metadata, so `src/swarm_control` rejects them unless `allow_legacy_checkpoint:=true`. That flag only bypasses the metadata check; it does not make the observation/action semantics compatible.
- The packaged `src/swarm_bringup/checkpoints/swarm_macro_policy.pt` is a different policy: 128×128, with `swarm-coop-enu-v1` contract metadata and a note indicating DAgger/BC training. The packaged `maddpg_iter00460.pt` is also a different file from these weights (same 256-wide architecture, different parameters).

## 9. Limitations and gotchas

- `contracts.py` lives at `marl_training/contracts.py`, while imports expect the top-level module `common.contracts`; make those agree before running.
- Training may not be reproducible by default: `train.py` does not configure a seed, and `reset(seed=None)` leaves the env RNG unseeded.
- All drones share a target, and the collision penalty applies while they cluster around it; the reward does not define a formation or separate arrival slots.
- Yaw is part of the observation but yaw rate does not affect translation, so the action has limited physical meaning.
- There is no obstacle avoidance, no ground crash, and no wall collision; only a soft x/y boundary penalty.
- Spawn positions are random and not collision-checked at reset.
- `rows_to_joint` silently drops incomplete or misaligned groups; if agent rows do not align, usable data is lost.
- The update count is integer-truncated and forced to at least 1: `max(1, int(env_steps * 0.25))`.
- The code depends on Ray RLlib's old API stack and imports such as `ray.rllib.execution.rollout_ops`; future Ray releases may remove these.
- Checkpoints are not resumable and do not carry the training configuration or contract.
