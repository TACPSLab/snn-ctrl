from functools import partial

import torch
import torch.optim as optim
import wandb

from deeprl.actor_critic_methods import SAC
from deeprl.actor_critic_methods.neural_network import mlp
from deeprl.actor_critic_methods.experience_replay import UER

from env import Env


cuda = torch.device("cuda:1")
env = Env(device=cuda)
obs_dim = 36 + 3 + 10
action_dim = 6
# TODO: Use Optuna for hyperparameter optimisation
agent = SAC.init(
    mlp.TanhGaussianPolicy.init(obs_dim, action_dim, [256, 256]),
    mlp.Quality.init(obs_dim, action_dim, [256, 256]),
    partial(optim.Adam, lr=3e-4),
    partial(optim.Adam, lr=3e-4),
    partial(optim.Adam, lr=3e-4),
    UER(1_000_000),
    256,
    0.99,
    -action_dim,
    5e-3,
    device=cuda,
)
run = wandb.init(project="MaplessNavi")

obs, _ = env.reset(seed=None); episodic_return = 0
for step in range(1_000_000):
    action = agent.compute_action(obs)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    agent.step(obs, action, reward, next_obs, terminated)
    episodic_return += reward

    if not terminated and not truncated:
        obs = next_obs
    else:
        run.log({
            "episodic_return": episodic_return,
        })
        obs, _ = env.reset(); episodic_return = 0

    # TODO: Save checkpoints into wandb local dir for syncing
    if step % 300 == 0:
        torch.onnx.export(
            agent._policy,
            obs,
            "policy.onnx",
            input_names = ["state"],
            output_names = ["action"],
        )
