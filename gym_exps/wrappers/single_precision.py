from typing import Union

import numpy as np
from gymnasium import ObservationWrapper, Env
from gymnasium.spaces import Space, Box, Dict


def _convert_observation_space(obs: Space) -> Space:
    # TODO: Adopt PEP 634
    if isinstance(obs, Box):
        obs = Box(obs.low, obs.high, obs.shape, dtype=np.float32)
    elif isinstance(obs, Dict):
        for k, v in obs.spaces.items():
            obs.spaces[k] = _convert_observation_space(v)
        obs = Dict(obs.spaces)
    else:
        raise NotImplementedError(f"No known conversion for space {type(obs)}.")
    return obs


def _convert_observation(obs: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    # TODO: Adopt PEP 634
    if isinstance(obs, np.ndarray) and obs.dtype == np.float64:
        return obs.astype(np.float32)
    elif isinstance(obs, dict):
        for k, v in obs.items():
            obs[k] = _convert_observation(v)
        return obs
    else:
        raise TypeError(f"Expected observation type is numpy.ndarray or dict. Got {type(obs)}.")


class SinglePrecisionObservation(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = _convert_observation_space(env.observation_space)

    def observation(self, observation):  # type: ignore[no-untyped-def]
        return _convert_observation(observation)
