"""
FIXME
Assume Numpy-based env.
Notice that there is a plan to add JAX-based env to Gymnasium in 2023. See
- https://github.com/Farama-Foundation/Gymnasium/pull/168
- https://github.com/Farama-Foundation/Gymnasium/issues/12

FIXME
Ray IDLE workers holding GPU memory.

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
"""

import logging
import os
from pathlib import Path

import ray
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from torch.utils.tensorboard import SummaryWriter

from wrappers import SinglePrecisionObservation


log = logging.getLogger(__name__)

def experiment(cfg) -> None:
    if len(ray.get_gpu_ids()) != 1:
        log.critical(f"One and only one GPU should be allocated to this job. Got {ray.get_gpu_ids()}.")
        return

    cuda = torch.device("cuda")  # Default CUDA device
    hydra_cfg = HydraConfig.get()
    log.info(f"PID {os.getpid()} on GPU {ray.get_gpu_ids()[0]} experimenting combination {hydra_cfg.job.override_dirname}")

    env = instantiate(cfg.env)
    env = SinglePrecisionObservation(env)

    agent = instantiate(
        cfg.algo,
        device=cuda,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    with SummaryWriter(log_dir=Path(hydra_cfg.runtime.output_dir)) as writer:
        for episode in range(cfg.num_episodes):
            state, _ = env.reset()
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
            writer.add_scalar(f'{env.spec.name}/episodic_return', episodic_return.item(), episode)
