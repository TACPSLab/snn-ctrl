import logging
import os
from pathlib import Path

import ray
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

def experiment(cfg) -> None:
    if len(ray.get_gpu_ids()) != 1:
        log.critical(f"One and only one GPU should be allocated to this job. Got {ray.get_gpu_ids()}.")
        return

    cuda = torch.device("cuda")  # Default CUDA device
    hydra_cfg = HydraConfig.get()
    log.info(f"PID {os.getpid()} on GPU {ray.get_gpu_ids()[0]} experimenting combination {hydra_cfg.job.override_dirname}")

    env = instantiate(cfg.env)
    agent = instantiate(
        cfg.algo,
        device=cuda,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    with SummaryWriter(log_dir=Path(hydra_cfg.runtime.output_dir)) as writer:
        for episode in range(cfg.num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, device=cuda, dtype=torch.float32)
            episodic_return = torch.zeros(1, device=cuda)

            while True:
                action = agent.compute_action(state)

                # Perform an action
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

                episodic_return += reward
                # Convert to size(1,) tensor
                next_state = torch.tensor(next_state  , device=cuda, dtype=torch.float32)
                reward     = torch.tensor([reward]    , device=cuda, dtype=torch.float32)
                terminated = torch.tensor([terminated], device=cuda, dtype=torch.bool)

                agent.step(state, action, reward, next_state, terminated)

                if terminated or truncated:
                    break
                state = next_state

            # Logging
            writer.add_scalar(f'{env.spec.name}/episodic_return', episodic_return.item(), episode)
