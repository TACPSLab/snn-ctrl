import gymnasium
import torch
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, builds, make_config, make_custom_builds_fn

from deeprl.actor_critic_methods import TD3
from deeprl.actor_critic_methods.experience_replay import UER
from deeprl.actor_critic_methods.neural_network.mlp import Actor, Critic
from deeprl.actor_critic_methods.noise_injection.action_space import Gaussian

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)
cs = ConfigStore.instance()


Ant = builds(gymnasium.make, id="Ant-v4")
cs.store(group="env", name="Ant", node=Ant)

Hopper = builds(gymnasium.make, id="Hopper-v4")
cs.store(group="env", name="Hopper", node=Hopper)

HalfCheetah = builds(gymnasium.make, id="HalfCheetah-v4")
cs.store(group="env", name="HalfCheetah", node=HalfCheetah)

InvertedDoublePendulum = builds(gymnasium.make, id="InvertedDoublePendulum-v4")
cs.store(group="env", name="InvertedDoublePendulum", node=InvertedDoublePendulum)


TD3Conf = builds(
    TD3,
    device=MISSING,
    state_dim=MISSING,
    action_dim=MISSING,
    policy="${policy}",
    critic=pbuilds(Critic, hidden_dims=[400, 300], activation_func='relu'),
    policy_optimiser=pbuilds(optim.Adam, lr=1e-4),
    critic_optimiser=pbuilds(optim.Adam, lr=1e-3),
    experience_replay=builds(UER, capacity=10_000),
    batch_size=2**10,
    discount_factor=0.99,
    polyak=0.99,
    policy_noise=builds(Gaussian, stddev=0.2),
    smoothing_noise_stddev=0.2,
    smoothing_noise_clip=0.5,
)
cs.store(group="algo", name="TD3", node=TD3Conf)


ANN = pbuilds(Actor, hidden_dims=[400, 300], activation_func='relu', output_func='tanh')
cs.store(group="policy", name="ANN", node=ANN)


ExperimentConf = make_config(
    defaults=[
        "_self_",
        {"override /hydra/job_logging": "colorlog"},
        {"override /hydra/hydra_logging": "colorlog"},
        {"override /hydra/launcher": "joblib"},
    ],
    env=MISSING,
    algo=MISSING,
    policy=MISSING,
    episodes=10_000,
    device=builds(torch.device, device="cuda:1")
)
