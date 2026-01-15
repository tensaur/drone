from types import SimpleNamespace
from typing import Any, Tuple

from gymnasium import spaces

from torch import nn
import torch
from torch.distributions.normal import Normal
from torch import nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy
from pufferlib.models import Convolutional as Conv

Recurrent = pufferlib.models.LSTMWrapper
from pufferlib.pytorch import layer_init, _nativize_dtype, nativize_tensor
import numpy as np


class Drone(nn.Module):
    def __init__(self, env, linear_size=128, lstm_size=16):
        super().__init__()
        self.hidden_size = lstm_size  # keep name for wrapper compatibility

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
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(input_size, linear_size)),
                nn.GELU(),
                layer_init(nn.Linear(linear_size, lstm_size)),
                nn.GELU(),
            )
        else:
            num_obs = int(np.prod(env.single_observation_space.shape))
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(num_obs, linear_size)),
                nn.GELU(),
                layer_init(nn.Linear(linear_size, lstm_size)),
                nn.GELU(),
            )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = layer_init(nn.Linear(lstm_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = layer_init(nn.Linear(lstm_size, num_atns), std=0.01)
        else:
            self.decoder_mean = layer_init(nn.Linear(lstm_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))

        self.value = layer_init(nn.Linear(lstm_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        # zero out rpms
        observations = torch.cat([observations[:, :-4], torch.zeros(observations.shape[0], 4, device=observations.device)], dim=1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            logits = Normal(mean, torch.exp(logstd))
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values
