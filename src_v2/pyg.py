from typing import Dict, List
import torch
from torch import TensorType
from torch.nn import Module, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv.gin_conv import GINEConv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn import Sequential as PyG_Sequential, global_mean_pool
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Space
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

        # configs
        config = model_config["custom_model_config"]
        actor_config = config["actor_config"]
        critic_config = config["critic_config"]
        self.critic_is_fc = config["critic_config"]["model"] == "fc"

        # model dimensions
        og_obs_space = obs_space.original_space
        self.num_inputs = flatdim(og_obs_space)
        self.num_agents = len(og_obs_space[0])
        self.adj_matrix_size = len(og_obs_space[1])
        self.node_state_size = flatdim(og_obs_space[0][0])
        self.edge_state_size = flatdim(og_obs_space[1][0])
        self.out_state_size = num_outputs // (self.num_agents - 1)

        self.encoding_size = 8
        self.node_encoder = self.__build_fc(ins=self.node_state_size, outs=self.encoding_size, hiddens=[])
        self.edge_encoder = self.__build_fc(ins=self.edge_state_size, outs=self.encoding_size, hiddens=[])
        self.actor = self._build_model(config=actor_config, 
                                       ins=self.encoding_size, 
                                       outs=self.out_state_size, 
                                       edge_dim=self.encoding_size, 
                                       add_pooling=False)
        self.critic = self._build_model(config=critic_config, 
                                        ins=self.num_inputs if self.critic_is_fc else self.encoding_size, 
                                        outs=1, 
                                        edge_dim=self.encoding_size, 
                                        add_pooling=True)

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

    def __build_gnn(self, config: dict, ins: int, outs: int, edge_dim: int):
        """creates one instance of the in the config specified gnn"""
        if config["model"] == "GATConv":
            return GATConv(ins, outs, dropout=config["dropout"])
        elif config["model"] == "GATv2Conv":
            return GATv2Conv(ins, outs, edge_dim=edge_dim, dropout=config["dropout"])
        elif config["model"] == "GINEConv":
            return GINEConv(self.__build_fc(ins, outs, [config["hidden_mlp_size"] for _ in range(config["n_hidden_mlp"])]))
        else:
            raise NotImplementedError(f"unknown model {config['model']}")  
        
    def _build_model(self, config: dict, ins: int, outs: int, edge_dim: int, add_pooling: bool) -> Module:
        """creates an NN model based on the config dict"""
        if config["model"] != "fc":
            gnn_rounds = list()
            for _ in range(config["rounds"] - 1):
                gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
                gnn_rounds.append(torch.nn.ReLU(inplace=True))
            
            # add last layer, pooling if necessecary
            if add_pooling:
                gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=ins, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   
                gnn_rounds.append((global_mean_pool, 'x, batch -> x'))
                gnn_rounds.append(SlimFC(in_size=ins, out_size=outs))
            else:
                gnn_rounds.append((self.__build_gnn(config, ins=ins, outs=outs, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))                   

            return PyG_Sequential('x, edge_index, edge_attr, batch', gnn_rounds)
        elif config["model"] == "fc":
            return self.__build_fc(ins=ins, outs=outs, hiddens=[config["hidden_mlp_size"] for _ in range(config["n_hidden_mlp"])])
        else:
            raise NotImplementedError(f"unknown model {config['model']}")   
    
    def value_function(self):
        return torch.reshape(self.last_values, [-1])
    
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        extract the node info from the flat observation to create an X tensor
        extract the adjacency relations to create the edge_indexes
        feed it to the _actor and _critic methods to get the outputs, those methods are implemented by a subclass

        note: the construction of the graph is tightly coupled to the format of the obs_space defined in the model class
        """ 
        obss = input_dict["obs"]
        obss_flat = input_dict["obs_flat"]
        agent_obss = obss[0]
        edge_obss = obss[1]
        batch_size = len(obss_flat)

        # iterate through the batch
        actor_graphs = list()
        fc_graphs = list()
        for i in range(batch_size):
            x, actor_edge_index, actor_edge_attr, fc_edge_index, fc_edge_attr = build_graph_v2(self.num_agents, agent_obss, edge_obss, i) 

            # format graph to torch
            x = torch.stack([self.node_encoder(v) for v in x])
            actor_edge_index = torch.tensor(actor_edge_index, dtype=torch.int64)
            actor_edge_attr = torch.stack([self.edge_encoder(e) for e in actor_edge_attr]) if actor_edge_attr else torch.zeros((0, self.encoding_size), dtype=torch.float32)
            fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.int64)
            fc_edge_attr = torch.stack([self.edge_encoder(e) for e in fc_edge_attr]) if fc_edge_attr else torch.zeros((0, self.encoding_size), dtype=torch.float32)

            actor_graphs.append(Data(x=x, edge_index=actor_edge_index, edge_attr=actor_edge_attr))
            fc_graphs.append(Data(x=x, edge_index=fc_edge_index, edge_attr=fc_edge_attr))

        actor_dataloader = DataLoader(dataset=actor_graphs, batch_size=batch_size)
        critic_dataloader = DataLoader(dataset=fc_graphs, batch_size=batch_size)
        actor_batch = next(iter(actor_dataloader))
        critic_batch = next(iter(critic_dataloader))
        assert torch.all(actor_batch.batch.eq(critic_batch.batch))

        actions = self.actor(x=actor_batch.x, edge_index=actor_batch.edge_index, edge_attr=actor_batch.edge_attr, batch=actor_batch.batch)
        if self.critic_is_fc:
            values = self.critic(obss_flat)
        else:
            values = self.critic(x=critic_batch.x, edge_index=critic_batch.edge_index, edge_attr=critic_batch.edge_attr, batch=critic_batch.batch)
        
        # node to original graph mapping
        curr_batch = 0
        node_to_sample_mapping = [0]
        for i, b in enumerate(actor_batch.batch):
            if curr_batch != b:
                curr_batch += 1
                node_to_sample_mapping.append(i)
        node_to_sample_mapping.append(len(actions))

        # create actions output for batch
        actions_batch = list()
        for s_i in range(len(node_to_sample_mapping) - 1):
            curr_actions = actions[node_to_sample_mapping[s_i]:node_to_sample_mapping[s_i+1]]
            actions_batch.append(torch.flatten(curr_actions[1:]))

        self.last_values = values
        return torch.stack(actions_batch), state