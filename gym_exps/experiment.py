import logging
import os
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

def experiment(cfg) -> None:
    hydra_cfg = HydraConfig.get()
    log.info(f"PID {os.getpid()} experimenting combination {hydra_cfg.job.override_dirname}")

    env = instantiate(cfg.env)
    device = instantiate(cfg.device)
    agent = instantiate(
        cfg.algo,
        device=device,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    with SummaryWriter(log_dir=Path(hydra_cfg.runtime.output_dir)) as writer:
        for episode in range(cfg.episodes):
            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32)
            cumulative_reward = torch.zeros(1, device=device)

            while True:
                action = agent.compute_action(state)

                # Perform an action
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

                cumulative_reward += reward
                # Convert to size(1,) tensor
                next_state = torch.tensor(next_state  , device=device, dtype=torch.float32)
                reward     = torch.tensor([reward]    , device=device, dtype=torch.float32)
                terminated = torch.tensor([terminated], device=device, dtype=torch.bool)

                agent.step(state, action, reward, next_state, terminated)

                if terminated or truncated:
                    break
                state = next_state

            # Logging
            writer.add_scalar(f'{env.spec.name}/cumulative_reward', cumulative_reward.item(), episode)
