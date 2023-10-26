import torch
import torch.nn as nn
from gymnasium.spaces.utils import flatdim
from gymnasium.spaces import Space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

class FullyConnected(TorchModelV2, nn.Module):
    def __init__(self, 
                    obs_space: Space,
                    action_space: Space,
                    num_outputs: int,
                    model_config: dict,
                    name: str,):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # custom parameters are passed via the model_config dict from ray
        self.default_config = self.model_config
        self.custom_config = self.model_config["custom_model_config"]

        assert self.custom_config["nn_action_hiddens"] and self.custom_config["nn_action_activation"] and \
            self.custom_config["nn_value_hiddens"] and self.custom_config["nn_value_activation"]

        self.input_size = flatdim(obs_space)
        self.action_hiddens = self.custom_config["nn_action_hiddens"]
        self.action_activation = self.custom_config["nn_action_activation"]
        self.value_hiddens = self.custom_config["nn_value_hiddens"]
        self.value_activation = self.custom_config["nn_value_activation"]

        # build action network
        layers = list()
        prev_layer_size = self.input_size
        for curr_layer_size in self.action_hiddens:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn=self.action_activation))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=num_outputs))
        self._action_net = nn.Sequential(*layers)

        # build value network
        layers = list()
        prev_layer_size = self.input_size
        for curr_layer_size in self.value_hiddens:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn=self.value_activation))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=1))
        self._value_net = nn.Sequential(*layers)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        logits = self._action_net(input_dict["obs_flat"])
        self._value_out = self._value_net(input_dict["obs_flat"])
        return logits, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])