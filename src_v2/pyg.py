from typing import Dict, List
import torch
from torch import TensorType
from torch.nn import Module, Sequential
from torch_geometric.nn.conv.gin_conv import GINEConv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space, Tuple
from gymnasium.spaces.utils import flatdim
from utils import build_graph_v2

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
    
        # model dimensions
        og_obs_space = obs_space.original_space
        self.num_inputs = flatdim(og_obs_space)
        self.num_agents = len(og_obs_space[0])
        self.adj_matrix_size = len(og_obs_space[1])
        self.node_state_size = flatdim(og_obs_space[0][0])
        self.edge_state_size = flatdim(og_obs_space[1][0])
        self.out_state_size = num_outputs // (self.num_agents - 1)

        self.actor = GINEConv(self.__build_fc(self.node_state_size, self.out_state_size, [16]))
        self.critic = self.__build_fc(self.num_inputs, 1, [16])

        self.encoding_size = 8
        self.node_encoder = self.__build_fc(self.node_state_size, self.encoding_size, [16])
        self.edge_encoder = self.__build_fc(self.edge_state_size, self.encoding_size, [])

        print("actor: ", self.actor)
        print("critic: ", self.critic)
        print("node encoder: ", self.node_encoder)
        print("edge encoder: ", self.edge_encoder)
        
    def __build_fc(self, ins: int, outs: int, hiddens: list):
        """builds a fully connected network with relu activation"""
        layers = list()
        prev_layer_size = ins
        for curr_layer_size in hiddens:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn="relu"))           
            prev_layer_size = curr_layer_size
        layers.append(SlimFC(in_size=prev_layer_size, out_size=outs))
        return Sequential(*layers)        
    
    def value_function(self):
        return torch.reshape(self.last_values, [-1])
    
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        extract the node info from the flat observation to create an X tensor
        extract the adjacency relations to create the edge_indexes
        feed it to the _actor and _critic methods to get the outputs, those methods are implemented by a subclass

        note: the construction of the graph is tightly coupled to the format of the obs_space defined in the model class
        """    
        outs = []
        values = []

        obss = input_dict["obs"]
        obss_flat = input_dict["obs_flat"]
        agent_obss = obss[0]
        edge_obss = obss[1]
        batch_size = len(obss_flat)

        # iterate through the batch
        for i in range(batch_size):
            x, actor_edge_index, actor_edge_attr, fc_edge_index, fc_edge_attr = build_graph_v2(self.num_agents, agent_obss, edge_obss, i) 

            # format graph to torch
            x = torch.stack([self.node_encoder(v) for v in x])
            actor_edge_index = torch.tensor(actor_edge_index, dtype=torch.int64)
            fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.int64)
            actor_edge_attr = torch.stack([self.edge_encoder(e) for e in actor_edge_attr]) if actor_edge_attr else torch.zeros((0, self.encoding_size), dtype=torch.float32)
            fc_edge_attr = torch.stack([self.edge_encoder(e) for e in fc_edge_attr]) if fc_edge_attr else torch.zeros((0, self.encoding_size), dtype=torch.float32)

            # compute results of all individual actors and concatenate the results
            all_actions = self.actor(x=x, edge_index=actor_edge_index, edge_attr=actor_edge_attr)
            outs.append(torch.flatten(all_actions)[self.out_state_size:])
            # compute values
            values.append(torch.flatten(self.critic(obss_flat[i])))
            
        # re-batch outputs
        outs = torch.stack(outs)
        self.last_values = torch.stack(values)
        return outs, state