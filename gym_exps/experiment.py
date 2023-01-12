"""
FIXME
Assume Numpy-based env.
Notice that there is a plan to add JAX-based env to Gymnasium in 2023. See
- https://github.com/Farama-Foundation/Gymnasium/pull/168
- https://github.com/Farama-Foundation/Gymnasium/issues/12

FIXME
Torch can infer the dtype for tensors constructed using Python floats. See
https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html
However, most envs return reward of numpy.float32/64 instead of float, and
Torch adheres to the precision of Numpy objects. Hence, reward has to be
converted to Tensor with explict dtype such as
    reward = torch.tensor([reward], device=cuda, dtype=torch.float32)
which is verbose.

TODO
Use NumpyToTorch wrapper. As of Gymnasium 0.27.0, if info dict's values
are numpy.float type, the wrapper fails in the line
https://github.com/Farama-Foundation/Gymnasium/blob/8f0baf62aa13c69c77dc00bed0060e09d700890a/gymnasium/experimental/wrappers/numpy_to_torch.py#L140
Notice that the wrapper returns float(reward) instead of regular numpy
object, which is confusing.

TODO
Why NormalizeObservation reduces traning performace?

TODO
Automate hyperparameter search by Optuna.
- https://docs.cleanrl.dev/advanced/hyperparameter-tuning
- https://docs.ray.io/en/latest/tune/index.html

TODO
Periodical checkpointing the policy to Weights & Biases.
- https://docs.cleanrl.dev/advanced/resume-training

TODO
https://github.com/wandb/wandb/commit/93e4974a05ce9b1d2fdcaf270290afde1f44e102
Why using RUN?

TODO
Interpolate Hydra's output directory management into WanB or vice versa.
https://github.com/facebookresearch/hydra/issues/1773
https://github.com/facebookresearch/hydra/issues/910
Wandb hardcodes its run_dir, which causes the use of loggers to store multiple directories
https://github.com/wandb/wandb/issues/1620
It's typically a bad idea to put files within the wandb folder as it is used to upload logs and files internally.
wandb folder is mainly used for offline use (and later sync), for uploading files/media/objects and for debugging.
"""

import logging
import math
import os

import ray
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from wrappers import SinglePrecisionObservation


log = logging.getLogger(__name__)

def experiment_process(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()

    wandb.init(
        project="MuJoCo",
        group=cfg.wandb.group,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        dir=hydra_cfg.runtime.output_dir,
    )

    if len(ray.get_gpu_ids()) != 1:
        log.critical(f"One and only one GPU should be allocated to this job. Got {ray.get_gpu_ids()}.")
        return
    cuda = torch.device("cuda")

    env = instantiate(cfg.env)
    env = SinglePrecisionObservation(env)

    agent = instantiate(
        cfg.algo,
        device=cuda,
        state_dim=math.prod(env.observation_space.shape),
        action_dim=math.prod(env.action_space.shape),
    )

    log.info(f"PID {os.getpid()} on GPU {ray.get_gpu_ids()[0]} experimenting combination {hydra_cfg.job.override_dirname}")

    for episode in range(cfg.num_episodes):
        state, _ = env.reset(seed=cfg.seed)
        state = torch.tensor(state, device=cuda)
        episodic_return = torch.zeros(1, device=cuda)

        while True:
            action = agent.compute_action(state)

            # Perform an action
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

            next_state = torch.tensor(next_state, device=cuda)
            # Convert to size(1,) tensor
            reward = torch.tensor([reward], device=cuda, dtype=torch.float32)
            terminated = torch.tensor([terminated], device=cuda, dtype=torch.bool)

            agent.step(state, action, reward, next_state, terminated)
            episodic_return += reward

            if terminated or truncated:
                break
            state = next_state

        # Logging
        wandb.log({
            "episodic_return": episodic_return,
        })
