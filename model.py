import sys
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions.categorical import Categorical

#import argparse
from matplotlib import pyplot as plt

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
#from time import perf_counter
import random
import os

torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)

# Inicialización de pesos ortogonal para las capas lineales
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#next(micro_model.parameters()).device)


# Aplica mascara a distribucion categorica, para escoger acciones
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None, show_mask=False):
        self.masks = masks
        #print("Mask recibida en numpy:", masks)
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor)
            if sw:
                print(self.masks)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        #print("Si lees esto, la mascara es disntito de vacia!")
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.))
        return -p_log_p.sum(-1)


# Bloques Residuales para evitar Gradiente Descendiente
class Residual_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels= channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def my_device(self):
        print(" - Res block 1:", next(self.conv0.parameters()).device)
        print(" - Res block 2:", next(self.conv1.parameters()).device)

    def forward(self, x):
        nn_input = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + nn_input


# Secuencia Convolucional
class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        first_dim = input_shape[0]
        self.conv = nn.Conv2d(in_channels=first_dim, out_channels=self._out_channels, kernel_size=3, padding=1)

        self.res_block0 = Residual_Block(self._out_channels)
        self.res_block1 = Residual_Block(self._out_channels)

    def my_device(self):
        self.conv = self.conv.cuda()
        print("Conv_seq:", next(self.conv.parameters()).device)
        self.res_block0.my_device()
        self.res_block1.my_device()


    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)

        assert x.shape[1:] == self.get_output_shape()
        return x


    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h+1)//2, (w+1)//2)



# Agente
class Agent(nn.Module):
    def __init__(self, obs_space_shape: tuple, nvec: list, mapsize=8*8, device="cpu"):
        super(Agent, self).__init__()
        h, w, c = obs_space_shape
        _shape = (c, h, w)
        self.mapsize = mapsize
        self.nvec = nvec
        convseqs = nn.ModuleList()

        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(_shape, out_channels)
            _shape = conv_seq.get_output_shape()
            convseqs.append(conv_seq)

        convseqs += nn.ModuleList([
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=_shape[0]*_shape[1]*_shape[2], out_features=256),
                nn.ReLU(),
                ])

        self.network = nn.Sequential(*convseqs)


        self.actor = layer_init(nn.Linear(256, sum(self.nvec)), std=0.0)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def initial_state(self, batch_size):
        # Si no se usa lstm, retorna una tupla vacia  (No usaremos lstm)
        return tuple()

    def my_device(self):
        print("Self device:", next(self.parameters()).device)
        print(next(self.actor.parameters()).device)
        print(next(self.critic.parameters()).device)
        for i in range(3):
            self.network[i].my_device()
        print(next(self.network.parameters()).device)

    # Lo de agent state es para satisfacer MicroBeast (MonoBeast)
    def forward(self, input_dict: dict, agent_state, learning=False, inds=[]):
        #print("---  OBS SIZE ---\n", input_dict["obs"][0, 0, ...].size())

            # Si no se dan los indices, entonces se está haciendo un step en el ambiente
        if not learning:
            return self.network(input_dict["obs"][0, 0, ...].permute((0, 3, 1, 2)))

        # Si se dan los indices, estamos actualizando
        return self.network(input_dict["obs"].permute((0, 3, 1, 2)))



    # Get action ahora recibe un diccionario en lugar de solo las observaciones!
    def get_action(self, input_dict: dict, learning=False, inds=[], agent_state=()):
        # Siempre entra x (obs) de shape [n_envs, h, w, 27]
        logits = self.actor(self.forward(input_dict, agent_state, learning, inds))    # [n_envs, 78hw]
        split_logits = torch.split(logits, self.nvec, dim=1)   #  (24, 7 * hw)   (7 actions * grid_size)  7 * 64 = 448

        # Si no se dan los indices del batch, se ejecutan los steps en el ambiente
        if not learning:
            #print("Action mask dim:", input_dict["action_mask"][0, ...].shape)
            action_mask = input_dict["action_mask"][0, ...]  # shape (n_envs, 78hw)

            split_action_mask = torch.split(action_mask, self.nvec, dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=ams) for (logits, ams) in zip(split_logits, split_action_mask)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])

        # Si se dan los indices del batch, se esta haciendo el update
        else:
            # Obtener acciones de los indices obtenidos
            a_dims = input_dict["action"].size()
            action = input_dict["action"]                

            # T para calzar con action mask [n_envs, 7hw] -> [7hw, n_envs]
            action = action.transpose(1, 0)

            action_mask = input_dict["action_mask"]    # [num_envs, 4992]
            split_action_mask = torch.split(action_mask, self.nvec, dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_action_mask)]


        # Retornar logpobs, entropia, accion y action_mask
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])

        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])

        num_predicted_parameters = len(self.nvec)
        logprob = logprob.T.view(-1, num_predicted_parameters)
        entropy = entropy.T.view(-1, num_predicted_parameters)
        action = action.T.view(-1, num_predicted_parameters)
        action_mask = action_mask.view(-1, sum(self.nvec))

        # ahora obtenemos el valor dentro de get_action
        baseline = self.get_value(input_dict, (), learning, inds).view(1, -1)


        if not learning:
            output = (dict(action=action, policy_logits=logits, 
            logprobs=logprob.sum(1), baseline=baseline), ()) 

        else:
            output = (dict(action=action, policy_logits=logits, 
            logprobs=logprob.sum(1), baseline=baseline, entropy=entropy.sum(1)), ()) 

        return output
    

    def get_value(self, input_dict: dict, agent_state=(), learning=False, inds=[]) -> torch.Tensor:
        return self.critic(self.forward(input_dict, agent_state, learning, inds=inds))



