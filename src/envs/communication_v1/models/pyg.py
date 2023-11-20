from typing import Dict, List
import torch
from torch import TensorType
from torch.nn import Module, Sequential
from torch_geometric.nn.conv.gin_conv import GINConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.gat_conv import GATConv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

class GNN_PyG(TorchModelV2, Module):
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
        self.custom_config = self.model_config["custom_model_config"]
        self.gnn_config = self.custom_config["gnn_config"]
        self.critic_config = self.custom_config["critic_config"]
        self.n_agents = self.custom_config["n_agents"]

        self.num_inputs = flatdim(obs_space)
        self.num_outputs = num_outputs
        self.agent_state_size = int((self.num_inputs - self.n_agents**2) / self.n_agents)
        
        self._gnn = self._build_gnn()
        self._critic = self._build_critic()
        self.last_values = None
        
        print("\n=== backend model ===")
        print(f"num_inputes      = {self.num_inputs}")
        print(f"num_outputs      = {self.num_outputs}")
        print(f"n_agents         = {self.n_agents}")
        print(f"agent_state_size = {self.agent_state_size}")
        print(f"size adj. mat    = {self.n_agents ** 2}")
        print(f"total obs_space  = {self.num_inputs}")
        print("gnn: ", self._gnn)
        print("critic: ", self._critic)
        print()

    def value_function(self):
        return torch.reshape(self.last_values, [-1])

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        extract the node info from the flat observation to create an X tensor
        extract the adjacency relations to create the edge_indexes
        feed it to the _gnn and _critic methods to get the outputs, those methods are implemented by a subclass

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
    
    def _build_gnn(self):
        """
        builds the model that is used to compute the node embeddings
        """
        ins = self.agent_state_size
        outs = int(self.num_outputs/ self.n_agents)

        if self.gnn_config["model"] == "PyG_GIN":
            layers = list()
            prev_layer_size = ins
            for curr_layer_size in [self.gnn_config["size_hidden"] for _ in range(self.gnn_config["num_hidden_layers"])]:
                layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn="relu"))           
                prev_layer_size = curr_layer_size
            layers.append(SlimFC(in_size=prev_layer_size, out_size=outs))
            return GINConv(Sequential(*layers))
        elif self.gnn_config["model"] == "PyG_GCN":
            return GCNConv(ins, outs)
        elif self.gnn_config["model"] == "PyG_GAT":
            return GATConv(ins, outs, heads=self.gnn_config["heads"], concat=False, dropout=self.gnn_config["dropout"])
    
    def _build_critic(self):
        """
        builds the model that is used to compute the value of a step
        """
        # fully connected critic that takes the whole observation as input and outputs value
        if self.critic_config["model"] == "fc":
            layers = list()
            prev_layer_size = self.num_inputs
            for curr_layer_size in [self.critic_config["size_hidden"] for _ in range(self.critic_config["num_hidden_layers"])]:
                layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn="relu"))           
                prev_layer_size = curr_layer_size
            layers.append(SlimFC(in_size=prev_layer_size, out_size=1))
            return Sequential(*layers)