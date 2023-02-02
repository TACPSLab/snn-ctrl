import gymnasium
import torch.optim as optim
import wandb
from callbacks import LogJobReturnCallback
from hydra.conf import HydraConf, SweepDir
from hydra_zen import MISSING, builds, make_config, make_custom_builds_fn, store

from deeprl.actor_critic_methods import SAC, TD3
from deeprl.actor_critic_methods.experience_replay import UER
from deeprl.actor_critic_methods.neural_network import mlp
from deeprl.actor_critic_methods.noise_injection.action_space import Gaussian

pbuilds = make_custom_builds_fn(zen_partial=True)

env_store = store(group="env", name=lambda cfg: cfg.id)  # Automatically infer the entry-name from the config's `.id` attribute.
env_store(builds(gymnasium.make, id="Ant-v4"))
env_store(builds(gymnasium.make, id="HalfCheetah-v4"))
env_store(builds(gymnasium.make, id="Hopper-v4"))
env_store(builds(gymnasium.make, id="Humanoid-v4"))
env_store(builds(gymnasium.make, id="InvertedDoublePendulum-v4"))
env_store(builds(gymnasium.make, id="Walker2d-v4"))

algo_store = store(group="algo")
algo_store(builds(
    SAC,
    policy=builds(mlp.GaussianPolicy, state_dim=MISSING, action_dim=MISSING, hidden_dims=[256, 256]),
    quality=builds(mlp.Quality, state_dim="${..policy.state_dim}", action_dim="${..policy.action_dim}", hidden_dims=[256, 256]),
    policy_optimiser_init=pbuilds(optim.Adam, lr=3e-4),
    quality_optimiser_init=pbuilds(optim.Adam, lr=3e-4),
    temperature_optimiser_init=pbuilds(optim.Adam, lr=3e-4),
    experience_replay=builds(UER, capacity=1_000_000),
    batch_size=256,
    discount_factor=0.99,
    target_entropy="-${.policy.action_dim}",
    polyak_factor=5e-3,
), name="SAC-ANN")
algo_store(builds(
    TD3,
    policy=builds(mlp.Policy, state_dim=MISSING, action_dim=MISSING, hidden_dims=[256, 256]),
    quality=builds(mlp.Quality, state_dim="${..policy.state_dim}", action_dim="${..policy.action_dim}", hidden_dims=[256, 256]),
    policy_optimiser_init=pbuilds(optim.Adam, lr=3e-4),
    quality_optimiser_init=pbuilds(optim.Adam, lr=3e-4),
    experience_replay=builds(UER, capacity=1_000_000),
    batch_size=256,
    discount_factor=0.99,
    polyak_factor=5e-3,
    exploration_noise=builds(Gaussian, stdev=0.1),
    smoothing_noise_stdev=0.2,
    smoothing_noise_clip=0.5,
), name="TD3-ANN")

# TODO: https://github.com/mit-ll-responsible-ai/hydra-zen/issues/395
store(
    HydraConf(
        callbacks={"log_job_return": builds(LogJobReturnCallback)},
        sweep=SweepDir(
            dir="${hydra.job.name}s/${wandb.group}",
            subdir="${hydra.job.override_dirname}"),
    ),
    group="hydra",
    name="config",
)

# TODO: Configure multiple experiments. Refer to https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html
# experiment_store = store(group="experiment", package="_global_")
ExperimentConf = make_config(
    seed=MISSING,
    env=MISSING,
    algo=MISSING,
    wandb={"group": wandb.sdk.lib.runid.generate_id()},
    hydra_defaults=[
        "_self_",
        {"override /hydra/launcher": "ray"}
    ],
)

store.add_to_hydra_store(overwrite_ok=True)

# TODO: ZenWrapper customises the behavior of hydra_zen.zen.
