from typing import Dict, List
import numpy as np
from torch import TensorType
import torch
from torch.nn import Module, Sequential
from torch_geometric.data import Data
from torch_geometric.nn.conv.gin_conv import GINConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

class GNN_PyG_base(TorchModelV2, Module):
    """
    base class for one-round gnn models.
    implements following process:
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
        self.n_agents = self.custom_config["n_agents"]
        self.agent_state_size = int((self.num_inputs - self.n_agents**2) / self.n_agents)
        
        print("\n=== backend model ===")
        print(f"num_inputes      = {self.num_inputs}")
        print(f"num_outputs      = {self.num_outputs}")
        print(f"n_agents         = {self.n_agents}")
        print(f"agent_state_size = {self.agent_state_size}")
        print(f"size adj. mat    = {self.n_agents ** 2}")
        print(f"total obs_space  = {self.num_inputs}")

        self._encoder = None
        self._gnn = GCNConv(self.agent_state_size, int(num_outputs/ self.n_agents))
        self._critic = SlimFC(self.num_inputs, 1)
        self.last_values = None


    def value_function(self):
        return torch.reshape(self.last_values, [-1])

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        note: the construction of the graph is tightly coupled to the format of the obs_space defined in the model class
        """    
        outs = []
        values = []

        # iterate through the batch
        for sample in input_dict["obs"]:
            # node values
            x = []
            for i in range(self.n_agents):
                x.append(sample[i * self.agent_state_size: (i+1) * self.agent_state_size])
            x = torch.stack(x)

            # edge indexes
            froms = []
            tos = []
            adj_matrix_offset = self.n_agents * self.agent_state_size # skip the part of the obs which dedicated to states
            for i in range(self.n_agents**2):
                if sample[adj_matrix_offset + i] == 1:
                    froms.append(i // self.n_agents)
                    tos.append(i % self.n_agents)
            edge_index = torch.tensor([froms, tos], dtype=torch.int64)

            outs.append(torch.flatten(self._gnn(x, edge_index)))
            values.append(self._critic(sample))
       
        # re-batch outputs
        outs = torch.stack(outs)
        self.last_values = torch.stack(values)

        return outs, state