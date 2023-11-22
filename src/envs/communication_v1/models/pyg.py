from typing import Dict, List
import torch
from torch import TensorType
from torch.nn import Module, Sequential
from torch_geometric.nn import Sequential as PyG_Sequential, global_mean_pool
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
        self.actor_config = self.custom_config["actor_config"]
        self.critic_config = self.custom_config["critic_config"]
        self.n_agents = self.custom_config["n_agents"]

        self.num_inputs = flatdim(obs_space)
        self.num_outputs = num_outputs
        self.agent_state_size = int((self.num_inputs - self.n_agents**2) / self.n_agents)
        self.agent_action_size = int(self.num_outputs / self.n_agents)
        
        self.last_values = None
        self._actor = self._build_model(config=self.actor_config)
        self._critic = self._build_model(config=self.critic_config, is_critic=True)
        
        print("\n=== backend model ===")
        print(f"num_inputes      = {self.num_inputs}")
        print(f"num_outputs      = {self.num_outputs}")
        print(f"n_agents         = {self.n_agents}")
        print(f"agent_state_size = {self.agent_state_size}")
        print(f"agent_action_size = {self.agent_action_size}")
        print(f"size adj. mat    = {self.n_agents ** 2}")
        print(f"total obs_space  = {self.num_inputs}")
        print("actor: ", self._actor)
        print("critic: ", self._critic)
        print()

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

            # compute actions
            if self.actor_config["model"] in ["PyG_GIN", "PyG_GCN", "PyG_GAT"]:
                outs.append(torch.flatten(self._actor(x, edge_index, batch=torch.zeros(x.shape[0],dtype=int))))
            elif self.actor_config["model"] == "fc":
                outs.append(torch.flatten(self._actor(sample)))
           
            # compute values
            if self.critic_config["model"] in ["PyG_GIN", "PyG_GCN", "PyG_GAT"]:
                values.append(torch.flatten(self._critic(x=x, edge_index=edge_index, batch=torch.zeros(x.shape[0],dtype=int))))
            elif self.critic_config["model"] == "fc":
                values.append(torch.flatten(self._critic(sample)))
       
        # re-batch outputs
        outs = torch.stack(outs)
        self.last_values = torch.stack(values)

        return outs, state
    
    def _build_gnn(self, config: dict, ins: int, outs: int):
        """creates one instance of the in the config specified gnn"""
        if config["model"] == "PyG_GIN":
            layers = list()
            prev_layer_size = ins
            for curr_layer_size in [config["mlp_hiddens_size"] for _ in range(config["mlp_hiddens"])]:
                layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn="relu"))           
                prev_layer_size = curr_layer_size
            layers.append(SlimFC(in_size=prev_layer_size, out_size=outs))
            return GINConv(Sequential(*layers))
        elif config["model"] == "PyG_GCN":
            return GCNConv(ins, outs)
        elif config["model"] == "PyG_GAT":
            return GATConv(ins, outs, heads=config["heads"], concat=False, dropout=config["dropout"])

    def _build_model(self, config: dict, is_critic = False):
        """creates an NN model based on the config dict"""
        assert config["model"] in ["PyG_GIN", "PyG_GCN", "PyG_GAT", "fc"], f"unknown model {config['model']}"

        if "PyG" in config["model"]:
            # if used as critic, can use multiple message passing rounds, before flatten all agents outputs and connect them to a single output
            gnn_rounds = list()
            if is_critic:
                prev_gnn_layer_size = self.agent_state_size
                for curr_gnn_layer_size in [config["gnn_hiddens_size"] for _ in range(config["gnn_num_rounds"] - 1)]:
                    gnn_rounds.append((self._build_gnn(config, prev_gnn_layer_size, curr_gnn_layer_size), 'x, edge_index -> x'))
                    gnn_rounds.append(torch.nn.ReLU(inplace=True))
                    prev_gnn_layer_size = curr_gnn_layer_size
                gnn_rounds.append((self._build_gnn(config, prev_gnn_layer_size, config["gnn_hiddens_size"]), 'x, edge_index -> x'))
                gnn_rounds.append((global_mean_pool, 'x, batch -> x'))
                gnn_rounds.append(SlimFC(in_size=config["gnn_hiddens_size"], out_size=1))
            else:
                gnn_rounds.append((self._build_gnn(config, self.agent_state_size, self.agent_action_size), 'x, edge_index -> x'))

            return PyG_Sequential('x, edge_index, batch', gnn_rounds)


        elif config["model"] == "fc":
            # fc specific inputs and outputs
            ins = self.num_inputs
            outs = self.num_outputs if not is_critic else 32
            
            layers = list()
            prev_layer_size = ins
            for curr_layer_size in [config["mlp_hiddens_size"] for _ in range(config["mlp_hiddens"])]:
                layers.append(SlimFC(in_size=prev_layer_size, out_size=curr_layer_size, activation_fn="relu"))           
                prev_layer_size = curr_layer_size
            layers.append(SlimFC(in_size=prev_layer_size, out_size=outs))

            # if used as critic, flatten all agents outputs and connect them to a single output
            if is_critic:
                layers.append(SlimFC(in_size=outs, out_size=1))

            return Sequential(*layers)
        
        else:
            raise NotImplementedError(f"unknown model {config['model']}")        