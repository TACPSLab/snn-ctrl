import gymnasium
import torch.optim as optim
import wandb
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, SweepDir
from hydra_zen import MISSING, builds, make_config, make_custom_builds_fn

from deeprl.actor_critic_methods import SAC, TD3
from deeprl.actor_critic_methods.experience_replay import UER
from deeprl.actor_critic_methods.neural_network import mlp
from deeprl.actor_critic_methods.noise_injection.action_space import Gaussian

pbuilds = make_custom_builds_fn(zen_partial=True)
cs = ConfigStore.instance()


Ant = builds(gymnasium.make, id="Ant-v4")
cs.store(group="env", name="Ant", node=Ant)

HalfCheetah = builds(gymnasium.make, id="HalfCheetah-v4")
cs.store(group="env", name="HalfCheetah", node=HalfCheetah)

Hopper = builds(gymnasium.make, id="Hopper-v4")
cs.store(group="env", name="Hopper", node=Hopper)

Humanoid = builds(gymnasium.make, id="Humanoid-v4")
cs.store(group="env", name="Humanoid", node=Humanoid)

InvertedDoublePendulum = builds(gymnasium.make, id="InvertedDoublePendulum-v4")
cs.store(group="env", name="InvertedDoublePendulum", node=InvertedDoublePendulum)

Walker2d = builds(gymnasium.make, id="Walker2d-v4")
cs.store(group="env", name="Walker2d", node=Walker2d)


SAC_ANN = builds(
    SAC,
    policy=pbuilds(mlp.GaussianPolicy, hidden_dims=[256, 256]),
    critic=pbuilds(mlp.ActionValue, hidden_dims=[256, 256]),
    policy_optimiser=pbuilds(optim.Adam, lr=3e-4),
    critic_optimiser=pbuilds(optim.Adam, lr=3e-4),
    temperature_optimiser=pbuilds(optim.Adam, lr=3e-4),
    experience_replay=builds(UER, capacity=1_000_000),
    batch_size=256,
    discount_factor=0.99,
    target_smoothing_factor=5e-3,
)
cs.store(group="algo", name="SAC-ANN", node=SAC_ANN)

TD3_ANN = builds(
    TD3,
    policy=pbuilds(mlp.Policy, hidden_dims=[256, 256]),
    critic=pbuilds(mlp.ActionValue, hidden_dims=[256, 256]),
    policy_optimiser=pbuilds(optim.Adam, lr=3e-4),
    critic_optimiser=pbuilds(optim.Adam, lr=3e-4),
    experience_replay=builds(UER, capacity=1_000_000),
    batch_size=256,
    discount_factor=0.99,
    target_smoothing_factor=5e-3,
    policy_noise=builds(Gaussian, stddev=0.1),
    smoothing_noise_stddev=0.2,
    smoothing_noise_clip=0.5,
)
cs.store(group="algo", name="TD3-ANN", node=TD3_ANN)


ExperimentConf = make_config(
    defaults=[
        "_self_",
        {"override /hydra/launcher": "ray"},
        {"override /hydra/hydra_logging": "colorlog"},
        {"override /hydra/job_logging": "colorlog"},
    ],
    hydra=HydraConf(
        sweep=SweepDir(
            dir="${hydra.job.name}s/${wandb.group}",
            subdir="${hydra.job.override_dirname}"),
        job_logging={
            "handlers": {"file": {"filename": "${hydra.runtime.output_dir}/hydra.log"}}
        }
        # output_subdir=None,  # controls the creation of the .hydra dir
    ),
    seed=MISSING,
    env=MISSING,
    algo=MISSING,
    num_episodes=1_000_000,
    wandb={"group": wandb.sdk.lib.runid.generate_id()},
)
