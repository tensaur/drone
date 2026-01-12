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
    def __init__(self, env, hidden_size=128):
        super().__init__()
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
            num_obs = np.prod(env.single_observation_space.shape)
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
                nn.GELU(),
            )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_atns), std=0.01
            )
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_atns), std=0.01
            )
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
            )
            self.decoder_logstd = nn.Parameter(
                torch.zeros(1, env.single_action_space.shape[0])
            )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size+4, 1), std=1)

    def forward_eval(self, observations, state=None):
        rpms = observations[:, -4:]
        actor_obs = torch.cat([observations[:, :-4], torch.zeros_like(rpms)], dim=1) # not edited inplace bc not sure if use elsewhere
        hidden = self.encode_observations(actor_obs, state=state)
        logits, values = self.decode_actions(hidden, rpms)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        """Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers)."""
        return self.encoder(observations.float())

    def decode_actions(self, hidden, rpms=None):
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)
        
        # only critic gets rpms
        if rpms is not None:
            critic_input = torch.cat([hidden, rpms], dim=1)
        else:
            critic_input = torch.cat([hidden, torch.zeros(hidden.shape[0], 4, device=hidden.device)], dim=1)
        
        values = self.value(critic_input)
        return logits, values
