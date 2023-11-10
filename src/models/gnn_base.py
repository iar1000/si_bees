from typing import Dict, List
from ray.rllib.utils.framework import TensorType
import torch
from torch.nn import Module, Sequential
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim





class GNN_base(TorchModelV2, Module):
    """
    base class for one-round gnn models.
    implements following process:
    - encode agents obs -> h_0 = encode(obs)
    - aggregate neighbours hs -> c_0 = aggregator(hs)
    - compute hidden state -> h_1 = f(h_0, c_0)
    - decode to get actions (distribution) -> q ~ decode(h_1)
    """
    def __init__(self, 
                    obs_space: Space,
                    action_space: Space,
                    num_outputs: int,
                    model_config: dict,
                    name: str,):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        Module.__init__(self)

        # custom parameters are passed via the model_config dict from ray
        self.config = self.model_config
        self.custom_config = self.model_config["custom_model_config"]

        self.num_inputs = flatdim(obs_space)
        self.num_outputs = num_outputs

        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()
        self._aggretator = self._build_aggregator()
        self._f = self._build_f()

        self.test_net_action = Sequential(
            SlimFC(in_size=30, out_size=128),
            SlimFC(in_size=128, out_size=self.num_outputs)
        )
        self.test_net_value = Sequential(
            SlimFC(in_size=30, out_size=128),
            SlimFC(in_size=128, out_size=1)
        )
        self.last_v = None

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs_flat = input_dict["obs_flat"]

        # @note: this could lead to a bug
        num_states = len(input_dict["obs"])
        state_size = int(len(obs_flat[0]) / num_states)
        assert num_states * state_size == self.num_inputs

        # get batched per agent states
        agent_states = [obs_flat[:, i * state_size: (i+1) * state_size] for i in range(num_states)]
        assert len(agent_states) == num_states, "need as many agent states as agents in the neighborhood"
        assert len(agent_states[0]) == len(obs_flat), "batch size should be that same after slicing"
        assert len(agent_states[0][0]) == state_size, "per state size should still be the same"
        #print(f"num_states: {num_states}, state size: {state_size}, total obs size: {self.num_inputs}")
        
        self.last_v = self.test_net_value(agent_states[0])
        return self.test_net_action(agent_states[0]), state
    
    def value_function(self):
        return torch.reshape(self.last_v, [-1])
    
    def _build_encoder(self) -> Module:
        """builds encoder network that takes an agents raw obs and encodes it to h"""
        raise NotImplementedError
    
    def _build_decoder(self) -> Module:
        """builds decoder network that takes h_1 and outputs q"""
        raise NotImplementedError
    
    def _build_aggregator(self) -> Module:
        """builds aggregator function that takes in hs and returns an aggregated vector c_0"""
        raise NotImplementedError
        
    def _build_f(self) -> Module:
        """builds NN that takes h_0 and c_0 to output the hidden state h_1"""
        raise NotImplementedError


class GNN_ComNet(GNN_base):
    def _build_encoder(self) -> Module:
        pass
    
    def _build_decoder(self) -> Module:
        pass
    
    def _build_aggregator(self) -> Module:
        pass
    
    def _build_f(self) -> Module:
        pass