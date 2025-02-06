import torch
import torch.nn as nn
import numpy as np

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces
from pufferlib.ocean.torch import Policy


class DronePolicy(Policy):
    def __init__(self, env, hidden_size=128):
        super().__init__(env)
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(
            env.single_action_space, pufferlib.spaces.MultiDiscrete
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(
                env.env.observation_space, pufferlib.spaces.Dict
            )
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict)

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(
                sum(np.prod(v.shape) for v in env.env.observation_space.values())
            )
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space.shape), hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        if self.is_multidiscrete:
            action_nvec = env.single_action_space.nvec
            self.decoder = nn.ModuleList(
                [
                    pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
                    for n in action_nvec
                ]
            )
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01
            )
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
            )
            self.decoder_logstd = nn.Parameter(
                torch.zeros(1, env.single_action_space.shape[0])
            )

        self.value_head = nn.Linear(hidden_size, 1)
